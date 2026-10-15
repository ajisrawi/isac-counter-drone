"""OFDM-radar processing: resource grid -> range-Doppler map.

Works for any reference (sparse pilots or full grid) because everything is
done per resource element:

1. channel estimate     H[k,l] = Y[k,l] / X[k,l] on known REs
2. common-phase-error   per-symbol phase referenced to the static channel
   (removes residual CFO / phase noise, makes the direct path zero-Doppler)
3. clutter removal      subtract the per-subcarrier mean over the CPI
   (cancels direct path and all static clutter, the equivalent of ECA)
4. fast time            zero-filled IFFT over subcarriers -> delay profile
5. slow time            non-uniform DFT over the true symbol times -> Doppler
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .numerology import C, Numerology


@dataclass
class SensingConfig:
    max_range_m: float = 4000.0         # max bistatic excess range processed
    neg_range_m: float = 150.0          # margin before the direct path
    range_oversample: int = 1
    max_range_rate_mps: float = 80.0    # +- bistatic range-rate span
    doppler_oversample: float = 2.0
    cpe_correction: bool = True
    clutter_removal: bool = True
    window: bool = True
    cfar_pfa: float = 1e-6
    cfar_guard: tuple = (3, 5)          # (range, doppler) cells; Hann main lobe + 2x Doppler oversampling
    cfar_train: tuple = (8, 10)
    min_snr_db: float = 13.0
    notch_mps: float = 1.0              # zero-Doppler exclusion (residual clutter)
    min_range_m: float = 20.0           # ignore detections hugging the direct path
    max_detections: int = 32
    range_reference: str = "direct_path"  # or "absolute" (needs baseline_m)
    baseline_m: float = 0.0


@dataclass
class RDMap:
    power: np.ndarray            # (n_range, n_doppler) linear power
    excess_range_m: np.ndarray   # bistatic excess range axis
    doppler_hz: np.ndarray
    range_rate_mps: np.ndarray   # bistatic range rate = -lambda * f_D
    tau_direct_s: float
    direct_power: float
    n_symbols: int
    wavelength: float = 0.0


def _hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(max(n, 1))
    return 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(n) + 0.5) / n)


def _static_profile(Hbar, have, q, M, n_lo, n_hi):
    buf = np.zeros(M, dtype=np.complex128)
    buf[np.mod(q[have], M)] = Hbar[have]
    prof = np.fft.ifft(buf)
    taps = np.mod(np.arange(n_lo, n_hi + 1), M)
    return prof, taps


def _parabolic(y_m, y_0, y_p) -> float:
    den = y_m - 2 * y_0 + y_p
    return 0.0 if den == 0 else 0.5 * (y_m - y_p) / den


def range_doppler(Y: np.ndarray, X: np.ndarray, valid: np.ndarray, times: np.ndarray,
                  num: Numerology, cfg: SensingConfig, timing_offset_s: float = 0.0) -> RDMap:
    """Y, X, valid: (n_sc, n_sym); times: (n_sym,) seconds.
    timing_offset_s: how much later the Rx FFT window was placed than the Tx
    one (only used for cfg.range_reference == "absolute")."""
    use = valid.any(axis=0)
    Y, X, valid, times = Y[:, use], X[:, use], valid[:, use], times[use]
    n_sc, n_s = Y.shape
    if n_s < 2:
        raise ValueError("need at least two symbols with known REs")

    H = np.zeros(Y.shape, dtype=np.complex128)
    H[valid] = Y[valid] / X[valid]
    cnt = valid.sum(axis=1)
    have = cnt > 0
    q = num.sc_freq_index
    scs = num.scs_hz
    cp = int(num.cp_len(1))

    # --- static channel model + common phase error ------------------------
    M_s = num.nfft
    n_lo, n_hi = -cp // 2, cp
    if cfg.cpe_correction:
        for _ in range(3):
            Hbar = np.where(have, H.sum(axis=1) / np.maximum(cnt, 1), 0)
            prof, taps = _static_profile(Hbar, have, q, M_s, n_lo, n_hi)
            # model = truncated delay profile evaluated on all subcarriers
            n = np.arange(n_lo, n_hi + 1)
            Hm = np.exp(-2j * np.pi * np.outer(q, n) / M_s) @ prof[taps]
            z = np.sum(H * np.conj(Hm)[:, None] * valid, axis=0)
            H *= np.exp(-1j * np.angle(z))[None, :]
            H[~valid] = 0

    Hbar = np.where(have, H.sum(axis=1) / np.maximum(cnt, 1), 0)

    # --- direct path delay (sub-sample) ------------------------------------
    zp = 8
    Mz = num.nfft * zp
    prof, taps = _static_profile(Hbar, have, q, Mz, n_lo * zp, n_hi * zp)
    mag = np.abs(prof[taps])
    i = int(np.argmax(mag))
    frac = _parabolic(*mag[[i - 1, i, (i + 1) % len(mag)]]) if 0 < i < len(mag) - 1 else 0.0
    tau_d = (n_lo * zp + i + frac) / (Mz * scs)
    direct_power = float(mag[i] ** 2)

    # --- clutter removal ----------------------------------------------------
    if cfg.clutter_removal:
        H = H - Hbar[:, None]
        H[~valid] = 0

    # --- fast time ------------------------------------------------------------
    if cfg.window:
        H = H * _hann(n_sc)[:, None]
    M = num.nfft * cfg.range_oversample
    dr = C / (M * scs)
    ref_tau = tau_d if cfg.range_reference == "direct_path" else cfg.baseline_m / C - timing_offset_s
    n0 = int(np.floor(ref_tau * M * scs)) - int(np.ceil(cfg.neg_range_m / dr))
    n_r = int(np.ceil((cfg.max_range_m + cfg.neg_range_m) / dr)) + 1
    n_r = min(n_r, M)
    buf = np.zeros((M, n_s), dtype=np.complex64)
    buf[np.mod(q, M), :] = H
    prof_all = np.fft.ifft(buf, axis=0)
    rows = np.mod(n0 + np.arange(n_r), M)
    r = prof_all[rows, :]
    excess = (n0 + np.arange(n_r)) * dr - C * ref_tau

    # --- slow time (non-uniform DFT) ---------------------------------------
    t = times - times[0]
    T = max(t[-1], 1e-9)
    lam = num.wavelength
    f_max = cfg.max_range_rate_mps / lam
    n_d = int(np.ceil(2 * f_max * T * cfg.doppler_oversample)) | 1
    f = np.linspace(-f_max, f_max, n_d)
    wt = _hann_cont(t / T) if cfg.window else np.ones(n_s)
    E = wt[None, :] * np.exp(-2j * np.pi * f[:, None] * t[None, :])
    rd = r @ E.T.astype(np.complex64)
    return RDMap(power=np.abs(rd) ** 2, excess_range_m=excess, doppler_hz=f,
                 range_rate_mps=-lam * f, tau_direct_s=tau_d,
                 direct_power=direct_power, n_symbols=n_s, wavelength=lam)


def _hann_cont(x: np.ndarray) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.clip(x, 0, 1))


def combine(maps: list[RDMap]) -> RDMap:
    """Non-coherent sum of maps with identical axes (e.g. antenna ports)."""
    out = maps[0]
    for m in maps[1:]:
        out = RDMap(out.power + m.power, out.excess_range_m, out.doppler_hz,
                    out.range_rate_mps, out.tau_direct_s,
                    out.direct_power + m.direct_power, out.n_symbols + m.n_symbols,
                    out.wavelength)
    return out
