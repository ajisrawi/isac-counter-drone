"""2-D cell-averaging CFAR and peak extraction on a range-Doppler map."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

from .processing import RDMap, SensingConfig, _parabolic


@dataclass
class Detection:
    excess_range_m: float     # bistatic excess path length (R_tx + R_rx - baseline)
    range_rate_mps: float     # d/dt of bistatic path length
    doppler_hz: float
    snr_db: float
    pair_id: str = ""
    t: float = 0.0

    def to_dict(self):
        return asdict(self)


def ca_cfar(power: np.ndarray, guard=(2, 3), train=(6, 8), pfa=1e-6):
    """Returns (detection mask, noise estimate). Doppler axis wraps."""
    g = np.array(guard)
    tr = np.array(train)
    outer = 2 * (g + tr) + 1
    inner = 2 * g + 1
    mode = ("reflect", "wrap")
    s_out = uniform_filter(power, size=tuple(outer), mode=mode) * np.prod(outer)
    s_in = uniform_filter(power, size=tuple(inner), mode=mode) * np.prod(inner)
    n_train = int(np.prod(outer) - np.prod(inner))
    noise = np.maximum((s_out - s_in) / n_train, np.finfo(float).tiny)
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    return power > alpha * noise, noise


def detect(rd: RDMap, cfg: SensingConfig, pair_id: str = "", t: float = 0.0) -> list[Detection]:
    P = rd.power
    mask, noise = ca_cfar(P, cfg.cfar_guard, cfg.cfar_train, cfg.cfar_pfa)
    snr = P / noise
    peaks = mask & (P == maximum_filter(P, size=3, mode=("reflect", "wrap")))
    peaks &= 10 * np.log10(snr) >= cfg.min_snr_db
    peaks &= (rd.excess_range_m >= cfg.min_range_m)[:, None]
    peaks &= (np.abs(rd.range_rate_mps) >= cfg.notch_mps)[None, :]
    ii, jj = np.nonzero(peaks)
    order = np.argsort(-snr[ii, jj])[: cfg.max_detections]
    dr = rd.excess_range_m[1] - rd.excess_range_m[0]
    df = rd.doppler_hz[1] - rd.doppler_hz[0]
    lam = rd.wavelength
    L = np.log(P + np.finfo(float).tiny)
    out = []
    nr_, nd = P.shape
    for i, j in zip(ii[order], jj[order]):
        di = _parabolic(L[i - 1, j], L[i, j], L[i + 1, j]) if 0 < i < nr_ - 1 else 0.0
        dj = _parabolic(L[i, (j - 1) % nd], L[i, j], L[i, (j + 1) % nd])
        f = rd.doppler_hz[j] + dj * df
        out.append(Detection(excess_range_m=float(rd.excess_range_m[i] + di * dr),
                             range_rate_mps=float(-lam * f), doppler_hz=float(f),
                             snr_db=float(10 * np.log10(snr[i, j])), pair_id=pair_id, t=t))
    return out
