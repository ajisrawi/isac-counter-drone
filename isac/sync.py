"""Passive LTE acquisition: PSS timing, CFO, half-frame and PCI search.

The receiver does not decode anything; it only needs symbol/frame timing and
the cell identity so the CRS can be regenerated locally.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

from . import lte
from .numerology import Numerology
from .ofdm import demodulate


@dataclass
class SyncResult:
    n_id: int
    start: int          # sample index of a frame-aligned symbol boundary
    l0: int             # frame symbol index of the symbol starting at `start`
    cfo_hz: float
    pss_snr_db: float
    pci_confidence_db: float


def pss_template(num: Numerology, n_id2: int) -> np.ndarray:
    from .sequences import lte_pss
    buf = np.zeros(num.nfft, dtype=np.complex64)
    buf[num.sc_bins[lte.pss_k(num.n_rb)]] = lte_pss(n_id2)
    return (np.fft.ifft(buf) * np.sqrt(num.nfft)).astype(np.complex64)


def find_pss(x: np.ndarray, num: Numerology):
    """Fold PSS correlation power over 5 ms half-frames.
    Returns (n_id2, position in half-frame of the PSS FFT-window start, snr_db)."""
    L = num.samples_per_frame // 2
    best = None
    for n_id2 in range(3):
        tpl = pss_template(num, n_id2)
        c = fftconvolve(x, np.conj(tpl[::-1]), mode="valid")
        p = np.abs(c) ** 2
        n_half = len(p) // L
        if n_half == 0:
            raise ValueError("need at least 5 ms of samples")
        folded = p[: n_half * L].reshape(n_half, L).sum(axis=0)
        i = int(np.argmax(folded))
        snr = 10 * np.log10(folded[i] / np.median(folded))
        if best is None or snr > best[2]:
            best = (n_id2, i, snr)
    return best


def estimate_cfo(x: np.ndarray, num: Numerology, start: int, max_sym: int = 280) -> float:
    """Fractional CFO from cyclic-prefix correlation (|cfo| < scs/2)."""
    lens = num.cp_len(np.arange(max_sym)) + num.nfft
    z = 0j
    pos = start
    for l in range(max_sym):
        cp = int(num.cp_len(l))
        if pos + cp + num.nfft > len(x):
            break
        a = x[pos + cp // 4: pos + cp]       # skip the ISI-affected CP head
        b = x[pos + cp // 4 + num.nfft: pos + cp + num.nfft]
        z += np.vdot(a, b)
        pos += int(lens[l])
    return float(np.angle(z) * num.scs_hz / (2 * np.pi))


def _pci_metric(Y: np.ndarray, num: Numerology, n_id: int, l0: int, chunk: int = 8) -> float:
    m = 0.0
    for j in range(Y.shape[1]):
        res = lte.crs_pilots(n_id, l0 + j, num.n_rb, 0)
        if res is None:
            continue
        k, xr = res
        z = Y[k, j] * np.conj(xr)
        n = (len(z) // chunk) * chunk
        m += float(np.sum(np.abs(z[:n].reshape(-1, chunk).sum(axis=1)) ** 2))
    return m


def acquire(x: np.ndarray, num: Numerology, n_id: int | None = None) -> SyncResult:
    """Full acquisition on a buffer of >= ~11 ms. If ``n_id`` is known only the
    half-frame ambiguity is resolved."""
    n_id2, p, snr = find_pss(x, num)
    L = num.samples_per_frame // 2
    pss_sym = lte.pss_symbols()[0]
    # CP start of the half-frame that contains this PSS
    b = p - int(num.cp_len(pss_sym)) - num.frame_symbol_start(pss_sym)
    b %= L
    cfo = estimate_cfo(x, num, b)
    n = np.arange(len(x))
    xc = x * np.exp(-2j * np.pi * cfo * n / num.fs).astype(np.complex64)
    Y = demodulate(xc, num, 14, l0=0, start=b)

    if n_id is not None:
        cand = [n_id]
        if n_id % 3 != n_id2:
            raise ValueError(f"PSS says N_ID2={n_id2}, configured PCI {n_id} disagrees")
    else:
        cand = [3 * n1 + n_id2 for n1 in range(168)]
    scores = []
    for cid in cand:
        for l0 in (0, 70):
            scores.append((_pci_metric(Y, num, cid, l0), cid, l0))
    scores.sort(reverse=True)
    best = scores[0]
    rest = np.median([s[0] for s in scores[1:]]) if len(scores) > 2 else scores[1][0]
    conf = 10 * np.log10(best[0] / max(rest, 1e-30))
    return SyncResult(n_id=best[1], start=int(b), l0=best[2], cfo_hz=cfo,
                      pss_snr_db=float(snr), pci_confidence_db=float(conf))
