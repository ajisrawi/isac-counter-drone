"""OFDM numerology for LTE (36.211) and NR (38.211) downlink carriers.

All sample-domain quantities assume the sampling rate fs = nfft * scs, i.e. the
"native" rate of the carrier (30.72 Msps for 20 MHz LTE, 61.44 Msps for a
2048-point 30 kHz NR carrier, ...).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

C = 299_792_458.0

# LTE channel bandwidth (MHz) -> (N_RB, NFFT)
LTE_BW = {
    1.4: (6, 128),
    3: (15, 256),
    5: (25, 512),
    10: (50, 1024),
    15: (75, 1536),
    20: (100, 2048),
}


@dataclass(frozen=True)
class Numerology:
    rat: str  # "LTE" or "NR"
    n_rb: int
    nfft: int
    fc_hz: float
    mu: int = 0  # NR numerology index (LTE is always 0)

    # ------------------------------------------------------------------ basics
    @property
    def scs_hz(self) -> float:
        return 15e3 * (2 ** self.mu)

    @property
    def fs(self) -> float:
        return self.nfft * self.scs_hz

    @property
    def n_sc(self) -> int:
        return 12 * self.n_rb

    @property
    def wavelength(self) -> float:
        return C / self.fc_hz

    @property
    def symbols_per_half_subframe(self) -> int:
        """Symbols per 0.5 ms (LTE slot / NR half-subframe), normal CP."""
        return 7 * (2 ** self.mu)

    @property
    def symbols_per_frame(self) -> int:
        return 20 * self.symbols_per_half_subframe

    @property
    def samples_per_frame(self) -> int:
        return int(round(10e-3 * self.fs))

    # --------------------------------------------------------------- CP / time
    def cp_len(self, l_abs) -> np.ndarray:
        """Cyclic-prefix length (samples) of symbol(s) l_abs, counted from a
        0.5 ms boundary (e.g. from the start of a radio frame)."""
        l_abs = np.asarray(l_abs)
        base = 144 * self.nfft // 2048
        extra = 16 * self.nfft * (2 ** self.mu) // 2048
        first = (l_abs % self.symbols_per_half_subframe) == 0
        return np.where(first, base + extra, base)

    def symbol_starts(self, n_sym: int, l0: int = 0) -> np.ndarray:
        """Sample index of the CP start of symbols l0..l0+n_sym-1, relative to
        the CP start of symbol l0."""
        lens = self.cp_len(np.arange(l0, l0 + n_sym)) + self.nfft
        return np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)

    def frame_symbol_start(self, l_abs: int) -> int:
        """CP-start sample of symbol l_abs relative to the start of its frame."""
        l = int(l_abs) % self.symbols_per_frame
        return int(np.sum(self.cp_len(np.arange(l)) + self.nfft))

    def symbol_times(self, n_sym: int, l0: int = 0) -> np.ndarray:
        """Time (s) of the centre of the FFT window of each symbol."""
        starts = self.symbol_starts(n_sym, l0)
        cps = self.cp_len(np.arange(l0, l0 + n_sym))
        return (starts + cps + self.nfft / 2) / self.fs

    # -------------------------------------------------------- subcarrier maps
    @cached_property
    def sc_freq_index(self) -> np.ndarray:
        """Signed frequency index (multiples of scs) of each occupied subcarrier
        k = 0..n_sc-1 (lowest frequency first). LTE skips the DC subcarrier."""
        k = np.arange(self.n_sc)
        half = self.n_sc // 2
        if self.rat == "LTE":
            return np.where(k < half, k - half, k - half + 1)
        return k - half

    @cached_property
    def sc_bins(self) -> np.ndarray:
        """FFT bin (0..nfft-1) of each occupied subcarrier."""
        return np.mod(self.sc_freq_index, self.nfft)

    # ------------------------------------------------------------ resolution
    @property
    def occupied_bw(self) -> float:
        return self.n_sc * self.scs_hz

    @property
    def range_resolution_m(self) -> float:
        """Bistatic (path-length) resolution, rectangular window."""
        return C / self.occupied_bw


def lte_carrier(bw_mhz: float, fc_hz: float) -> Numerology:
    n_rb, nfft = LTE_BW[bw_mhz]
    return Numerology("LTE", n_rb, nfft, fc_hz, 0)


def nr_carrier(n_rb: int, mu: int, fc_hz: float, nfft: int | None = None) -> Numerology:
    if nfft is None:
        nfft = 1 << int(np.ceil(np.log2(12 * n_rb / 0.85)))  # keep ~85 % occupancy max
    if 12 * n_rb >= nfft:
        raise ValueError("n_rb does not fit into nfft")
    return Numerology("NR", n_rb, nfft, fc_hz, mu)
