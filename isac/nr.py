"""NR downlink pieces: SS/PBCH block (PSS, SSS, PBCH-DMRS) and a test-grid
builder (38.211 sec. 7.4.2, 7.4.3, 7.4.1.4).

Symbol indices ``l_abs`` count from the start of an SSB period (the first
half-frame of an SSB-carrying frame).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .numerology import Numerology
from .sequences import gold, nr_pss, nr_sss, qpsk_from_bits, random_qam

SSB_SC = 240
SSB_FIRST_SYMBOLS = {
    # case: (symbol pattern within a period of `step` symbols, step)
    "A": ((2, 8), 14),
    "B": ((4, 8, 16, 20), 28),
    "C": ((2, 8), 14),
}


@dataclass
class SsbConfig:
    n_id: int                 # physical cell id 0..1007
    k0: int                   # carrier subcarrier index of SSB subcarrier 0
    case: str = "C"
    l_max: int = 8
    period_ms: int = 20
    n_hf: int = 0             # half-frame (0 = first)
    active: tuple = field(default=())   # transmitted SSB indices (empty = all)

    def first_symbols(self) -> list[int]:
        pattern, step = SSB_FIRST_SYMBOLS[self.case]
        n_rep = self.l_max // len(pattern)
        syms = [p + step * n for n in range(n_rep) for p in pattern]
        if self.active:
            syms = [syms[i] for i in self.active]
        return syms

    def ssb_index_of(self, start_sym: int) -> int:
        pattern, step = SSB_FIRST_SYMBOLS[self.case]
        n_rep = self.l_max // len(pattern)
        allsyms = [p + step * n for n in range(n_rep) for p in pattern]
        return allsyms.index(start_sym)


@lru_cache(maxsize=4096)
def pbch_dmrs(n_id: int, i_ssb_bar: int) -> np.ndarray:
    c_init = (1 << 11) * (i_ssb_bar + 1) * (n_id // 4 + 1) + (1 << 6) * (i_ssb_bar + 1) + (n_id % 4)
    return qpsk_from_bits(gold(c_init, 288)).astype(np.complex64)


def _dmrs_k(v: int):
    k13 = np.arange(0, 240, 4) + v
    k2 = np.concatenate([np.arange(0, 48, 4), np.arange(192, 240, 4)]) + v
    return k13, k2


def ssb_known_res(cfg: SsbConfig, i_ssb: int):
    """Known REs of one SSB: list over the 4 SSB symbols of (k_rel, values).
    PBCH data REs are *not* included (unknown payload)."""
    n_id1, n_id2 = divmod(cfg.n_id, 3)
    v = cfg.n_id % 4
    i_bar = (i_ssb % 4) + 4 * cfg.n_hf if cfg.l_max == 4 else i_ssb % 8
    dmrs = pbch_dmrs(cfg.n_id, i_bar)
    k13, k2 = _dmrs_k(v)
    k_sync = np.arange(56, 183)
    out = [
        (k_sync, nr_pss(n_id2)),
        (k13, dmrs[0:60]),
        (np.concatenate([k2, k_sync]), np.concatenate([dmrs[60:84], nr_sss(n_id1, n_id2)])),
        (k13, dmrs[84:144]),
    ]
    return out


def symbols_per_ms(num: Numerology) -> int:
    return 14 * (2 ** num.mu)


def ssb_pilots(cfg: SsbConfig, l_abs: int, num: Numerology):
    """Known SSB REs on symbol ``l_abs`` (carrier subcarrier indices) or None."""
    lp = l_abs % (cfg.period_ms * symbols_per_ms(num))
    for s in cfg.first_symbols():
        if s <= lp < s + 4:
            k, x = ssb_known_res(cfg, cfg.ssb_index_of(s))[lp - s]
            return k + cfg.k0, x
    return None


def build_grid(num: Numerology, n_sym: int, l0: int = 0, ssb: SsbConfig | None = None,
               load: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthetic NR downlink grid: random 16-QAM traffic plus SSBs."""
    rng = rng or np.random.default_rng()
    grid = random_qam(rng, (num.n_sc, n_sym), 16)
    if load < 1.0:
        for l in range(n_sym):
            off = np.repeat(rng.random(num.n_rb) > load, 12)
            grid[off, l] = 0
    if ssb is not None:
        sym_per_period = ssb.period_ms * symbols_per_ms(num)
        for j in range(n_sym):
            lp = (l0 + j) % sym_per_period
            for s in ssb.first_symbols():
                if s <= lp < s + 4:
                    r = lp - s
                    sl = slice(ssb.k0, ssb.k0 + SSB_SC)
                    # PBCH payload (random) everywhere in the block, zeros in guard REs
                    blk = (rng.integers(0, 2, SSB_SC) * 2 - 1 + 1j * (rng.integers(0, 2, SSB_SC) * 2 - 1)) / np.sqrt(2)
                    if r in (0,):
                        blk[:] = 0
                    if r == 2:
                        blk[48:56] = 0
                        blk[183:192] = 0
                    grid[sl, j] = blk
                    k, x = ssb_known_res(ssb, ssb.ssb_index_of(s))[r]
                    grid[ssb.k0 + k, j] = x
    return grid
