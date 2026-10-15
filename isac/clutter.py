"""Time-domain clutter cancellation (Extensive Cancellation Algorithm, ECA).

Projects the surveillance signal onto delayed copies of the reference and
subtracts the result, removing the direct path and every static echo up to
``max_delay``. Unlike the per-subcarrier mean in ``processing``, it also
removes the inter-symbol interference of clutter whose delay exceeds the
cyclic prefix, the main limiter of OFDM radar near mountains and high-rises.
Needs the full transmitted signal, so it applies to the integrated/TX-tap
and two-channel modes, not to CRS-only passive reception.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import fftconvolve


def eca(ref: np.ndarray, surv: np.ndarray, n_taps: int, n_neg: int = 4,
        diag_load: float = 1e-6) -> np.ndarray:
    """ref, surv: time-aligned (direct path near lag 0) equal-length arrays.
    Cancels delays -n_neg .. n_taps-n_neg-1 samples over the whole buffer
    (one batch = notch width ~ 1/duration around zero Doppler)."""
    n = min(len(ref), len(surv))
    x = ref[:n].astype(np.complex128)
    y = surv[:n].astype(np.complex128)
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.fft(x, nf)
    r = np.fft.ifft(np.abs(X) ** 2)[:n_taps]                       # r(k) = sum x[m+k] x*[m]
    c_yx = np.fft.ifft(np.fft.fft(y, nf) * np.conj(X))             # sum y[m] x*[m-d]
    d = np.arange(n_taps) - n_neg
    b = c_yx[np.mod(d, nf)]
    col = r.copy()
    col[0] *= 1 + diag_load
    w = solve_toeplitz((col, np.conj(col)), b)
    z = fftconvolve(x, w)[n_neg:n_neg + n]
    out = surv.copy()
    out[:n] = (y - z).astype(surv.dtype)
    return out
