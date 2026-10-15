"""LTE FDD downlink pieces needed for sensing: CRS, PSS and a test-frame builder.

Symbol indices ``l_abs`` count from the start of a radio frame (0..139 for
normal CP); slot ns = l_abs // 7, symbol-in-slot l = l_abs % 7.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .numerology import Numerology
from .sequences import gold, lte_pss, qpsk_from_bits, random_qam

N_RB_MAX = 110
N_SYMB = 7  # normal CP


@lru_cache(maxsize=8192)
def _crs_seq(n_id: int, ns: int, l: int) -> np.ndarray:
    c_init = (1 << 10) * (7 * (ns + 1) + l + 1) * (2 * n_id + 1) + 2 * n_id + 1
    return qpsk_from_bits(gold(c_init, 4 * N_RB_MAX)).astype(np.complex64)


def crs_pilots(n_id: int, l_abs: int, n_rb: int, port: int = 0):
    """Cell-specific reference signal REs of antenna port 0/1 on symbol
    ``l_abs`` (36.211 sec. 6.10.1). Returns (k, values) or None."""
    ns = (l_abs // N_SYMB) % 20
    l = l_abs % N_SYMB
    if l not in (0, N_SYMB - 3):
        return None
    if port == 0:
        v = 0 if l == 0 else 3
    elif port == 1:
        v = 3 if l == 0 else 0
    else:
        raise ValueError("only ports 0 and 1 supported")
    v_shift = n_id % 6
    m = np.arange(2 * n_rb)
    k = 6 * m + (v + v_shift) % 6
    r = _crs_seq(n_id, ns, l)
    return k, r[m + N_RB_MAX - n_rb]


def crs_reserved(n_id: int, l_abs: int, n_rb: int, n_ports: int):
    """REs occupied by CRS of any configured port (data is never mapped there)."""
    ks = []
    for p in range(n_ports):
        res = crs_pilots(n_id, l_abs, n_rb, p)
        if res is not None:
            ks.append(res[0])
    return np.concatenate(ks) if ks else np.zeros(0, dtype=int)


def pss_symbols() -> tuple[int, int]:
    """Frame symbol indices that carry the PSS (FDD: last symbol of slots 0, 10)."""
    return (N_SYMB - 1, 10 * N_SYMB + N_SYMB - 1)


def pss_k(n_rb: int) -> np.ndarray:
    return np.arange(62) - 31 + 6 * n_rb


def pss_reserved_k(n_rb: int) -> np.ndarray:
    n = np.concatenate([np.arange(-5, 0), np.arange(62, 67)])
    return n - 31 + 6 * n_rb


def build_frame_grid(num: Numerology, n_id: int, n_sym: int, l0: int = 0,
                     n_ports: int = 1, load: float = 1.0,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Synthetic port-0 LTE downlink grid: random QAM "traffic" on a fraction
    ``load`` of the PRBs, CRS, PSS. Other channels (PDCCH, PBCH, SSS) are
    modelled as random data: to a CRS-only receiver they are unknown anyway."""
    rng = rng or np.random.default_rng()
    grid = random_qam(rng, (num.n_sc, n_sym), 16)
    if load < 1.0:
        n_rb = num.n_rb
        for l in range(n_sym):
            off = rng.random(n_rb) > load
            mask = np.repeat(off, 12)
            grid[mask, l] = 0
    pss_syms = pss_symbols()
    for j in range(n_sym):
        la = l0 + j
        lf = la % 140
        k_res = crs_reserved(n_id, la, num.n_rb, n_ports)
        if k_res.size:
            grid[k_res, j] = 0
        res = crs_pilots(n_id, la, num.n_rb, 0)
        if res is not None:
            grid[res[0], j] = res[1]
        if lf in pss_syms:
            grid[pss_reserved_k(num.n_rb), j] = 0
            grid[pss_k(num.n_rb), j] = lte_pss(n_id % 3)
    return grid
