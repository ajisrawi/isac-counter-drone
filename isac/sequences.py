"""Pseudo-random and synchronisation sequences shared by LTE and NR."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

_NC = 1600


@lru_cache(maxsize=1)
def _x1(length: int = 1 << 16) -> np.ndarray:
    x = np.zeros(length + 31, dtype=np.uint8)
    x[0] = 1
    for n in range(length):
        x[n + 31] = x[n + 3] ^ x[n]
    return x


def gold(c_init: int, length: int) -> np.ndarray:
    """Length-31 Gold sequence c(n) (36.211 sec. 7.2 / 38.211 sec. 5.2.1)."""
    total = length + _NC
    x1 = _x1()
    if total + 31 > len(x1):
        raise ValueError("sequence too long")
    x2 = np.zeros(total + 31, dtype=np.uint8)
    for i in range(31):
        x2[i] = (c_init >> i) & 1
    for n in range(total):
        x2[n + 31] = x2[n + 3] ^ x2[n + 2] ^ x2[n + 1] ^ x2[n]
    return (x1[_NC:_NC + length] ^ x2[_NC:_NC + length]).astype(np.uint8)


def qpsk_from_bits(c: np.ndarray) -> np.ndarray:
    """r(m) = (1-2c(2m))/sqrt2 + j(1-2c(2m+1))/sqrt2."""
    c = c.astype(np.float32)
    return ((1 - 2 * c[0::2]) + 1j * (1 - 2 * c[1::2])) / np.sqrt(2)


# ----------------------------------------------------------------- LTE PSS
LTE_PSS_ROOT = {0: 25, 1: 29, 2: 34}


def lte_pss(n_id2: int) -> np.ndarray:
    """62-element Zadoff-Chu PSS (36.211 sec. 6.11.1.1)."""
    u = LTE_PSS_ROOT[n_id2]
    n = np.arange(62)
    d = np.where(n <= 30,
                 np.exp(-1j * np.pi * u * n * (n + 1) / 63),
                 np.exp(-1j * np.pi * u * (n + 1) * (n + 2) / 63))
    return d.astype(np.complex64)


# ------------------------------------------------------------------ NR PSS/SSS
def _mseq(taps, init) -> np.ndarray:
    x = np.zeros(127 + 7, dtype=np.int8)
    x[:7] = init
    for i in range(127):
        x[i + 7] = sum(x[i + t] for t in taps) % 2
    return x[:127]


def nr_pss(n_id2: int) -> np.ndarray:
    """127-element BPSK m-sequence PSS (38.211 sec. 7.4.2.2)."""
    x = _mseq((4, 0), [0, 1, 1, 0, 1, 1, 1])
    m = (np.arange(127) + 43 * n_id2) % 127
    return (1 - 2 * x[m]).astype(np.complex64)


def nr_sss(n_id1: int, n_id2: int) -> np.ndarray:
    """127-element SSS (38.211 sec. 7.4.2.3)."""
    x0 = _mseq((4, 0), [1, 0, 0, 0, 0, 0, 0])
    x1 = _mseq((1, 0), [1, 0, 0, 0, 0, 0, 0])
    m0 = 15 * (n_id1 // 112) + 5 * n_id2
    m1 = n_id1 % 112
    n = np.arange(127)
    return ((1 - 2 * x0[(n + m0) % 127]) * (1 - 2 * x1[(n + m1) % 127])).astype(np.complex64)


def random_qam(rng: np.random.Generator, shape, order: int = 16) -> np.ndarray:
    """Unit-average-power square QAM symbols (stand-in for PDSCH data)."""
    m = int(np.sqrt(order))
    levels = 2 * np.arange(m) - (m - 1)
    norm = np.sqrt(2 * np.mean(levels ** 2))
    i = rng.choice(levels, size=shape)
    q = rng.choice(levels, size=shape)
    return ((i + 1j * q) / norm).astype(np.complex64)
