"""CP-OFDM modulation / demodulation on a resource grid (n_sc x n_sym)."""
from __future__ import annotations

import numpy as np

from .numerology import Numerology


def modulate(grid: np.ndarray, num: Numerology, l0: int = 0) -> np.ndarray:
    """Resource grid -> complex baseband samples. Unitary FFT scaling, so a
    resource element of power 1 carries energy 1."""
    n_sc, n_sym = grid.shape
    assert n_sc == num.n_sc
    freq = np.zeros((num.nfft, n_sym), dtype=np.complex64)
    freq[num.sc_bins, :] = grid
    td = np.fft.ifft(freq, axis=0) * np.sqrt(num.nfft)
    cps = num.cp_len(np.arange(l0, l0 + n_sym))
    out = np.empty(int(np.sum(cps)) + n_sym * num.nfft, dtype=np.complex64)
    pos = 0
    for l in range(n_sym):
        cp = int(cps[l])
        out[pos:pos + cp] = td[-cp:, l]
        out[pos + cp:pos + cp + num.nfft] = td[:, l]
        pos += cp + num.nfft
    return out


def demodulate(x: np.ndarray, num: Numerology, n_sym: int, l0: int = 0,
               start: int = 0, backoff: int | None = None) -> np.ndarray:
    """Samples -> resource grid.

    start   : sample index of the CP start of symbol l0 in ``x``.
    backoff : FFT window advance into the CP (samples). Makes the receiver
              tolerant to paths arriving slightly *before* the timing
              reference. The resulting linear phase is removed, so the grid is
              as if the window started exactly at the end of the CP.
    """
    if backoff is None:
        backoff = int(num.cp_len(1)) // 4
    starts = start + num.symbol_starts(n_sym, l0)
    cps = num.cp_len(np.arange(l0, l0 + n_sym))
    win0 = starts + cps - backoff
    if win0[-1] + num.nfft > len(x) or win0[0] < 0:
        raise ValueError("buffer too short for requested symbols")
    idx = win0[None, :] + np.arange(num.nfft)[:, None]
    freq = np.fft.fft(x[idx], axis=0) / np.sqrt(num.nfft)
    grid = freq[num.sc_bins, :]
    if backoff:
        q = num.sc_freq_index[:, None]
        grid = grid * np.exp(2j * np.pi * q * backoff / num.nfft)
    return grid.astype(np.complex64)


def symbols_in_buffer(num: Numerology, n_samples: int, start: int, l0: int = 0) -> int:
    """How many complete OFDM symbols fit into a buffer from ``start``."""
    n = 0
    pos = start
    while True:
        need = int(num.cp_len(l0 + n)) + num.nfft
        if pos + need > n_samples:
            return n
        pos += need
        n += 1
