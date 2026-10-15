"""Reference providers: they tell the sensing processor which resource
elements are known and what was transmitted on them.

* ``LteCrsReference``   - passive LTE, single receive channel. Only the CRS
                          (always-on, cell-specific pilots) are regenerated.
* ``NrSsbReference``    - passive NR from SS/PBCH blocks (PSS, SSS, PBCH-DMRS).
                          Sparse in time (one burst per 20 ms): long-range,
                          slow-target sensing only.
* ``KnownGridReference``- network-integrated ISAC / two-channel passive radar:
                          the full transmitted grid is known, either from the
                          base station (DU / fronthaul tap / TX IQ tap) or by
                          demodulating a clean reference-antenna channel.
"""
from __future__ import annotations

import numpy as np

from . import lte, nr
from .numerology import Numerology


class LteCrsReference:
    def __init__(self, num: Numerology, n_id: int, port: int = 0):
        self.num, self.n_id, self.port = num, n_id, port

    def grid(self, n_sym: int, l0: int):
        X = np.zeros((self.num.n_sc, n_sym), dtype=np.complex64)
        valid = np.zeros((self.num.n_sc, n_sym), dtype=bool)
        for j in range(n_sym):
            res = lte.crs_pilots(self.n_id, l0 + j, self.num.n_rb, self.port)
            if res is not None:
                X[res[0], j] = res[1]
                valid[res[0], j] = True
        return X, valid


class NrSsbReference:
    def __init__(self, num: Numerology, cfg: nr.SsbConfig):
        self.num, self.cfg = num, cfg

    def grid(self, n_sym: int, l0: int):
        X = np.zeros((self.num.n_sc, n_sym), dtype=np.complex64)
        valid = np.zeros((self.num.n_sc, n_sym), dtype=bool)
        for j in range(n_sym):
            res = nr.ssb_pilots(self.cfg, l0 + j, self.num)
            if res is not None:
                X[res[0], j] = res[1]
                valid[res[0], j] = True
        return X, valid


class KnownGridReference:
    """Wraps an already-known transmitted grid X (n_sc x n_sym)."""

    def __init__(self, X: np.ndarray, rel_threshold: float = 0.05):
        self.X = X
        p = np.abs(X) ** 2
        rms = np.sqrt(np.mean(p[p > 0])) if np.any(p > 0) else 1.0
        self.valid = np.abs(X) > rel_threshold * rms

    def grid(self, n_sym: int, l0: int = 0):
        return self.X[:, :n_sym], self.valid[:, :n_sym]
