"""Constant-velocity Kalman tracker with global-nearest-neighbour association
and M-of-N track confirmation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .localize import Fix


@dataclass
class Track:
    track_id: int
    x: np.ndarray
    P: np.ndarray
    t: float
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    history: list = field(default_factory=list)

    def to_dict(self, origin=None):
        d = {"track_id": self.track_id, "t": round(self.t, 3), "confirmed": self.confirmed,
             "pos_enu": self.x[:3].round(1).tolist(), "vel_enu": self.x[3:].round(2).tolist(),
             "speed_mps": round(float(np.linalg.norm(self.x[3:])), 2),
             "pos_sigma_m": np.sqrt(np.diag(self.P)[:3]).round(1).tolist()}
        if origin is not None:
            from .geometry import enu_to_geodetic
            d["lat"], d["lon"], d["alt_m"] = enu_to_geodetic(self.x[:3], *origin)
        return d


class Tracker:
    def __init__(self, q_accel=4.0, gate_chi2=22.5, confirm_hits=3, max_misses=3,
                 r_floor=(10.0, 1.0)):
        self.q = q_accel
        self.gate = gate_chi2          # 6 dof, ~99.9 %
        self.confirm_hits = confirm_hits
        self.max_misses = max_misses
        self.r_floor = r_floor
        self.tracks: list[Track] = []
        self._next = 1

    def _R(self, fx: Fix):
        R = fx.cov.copy()
        R[:3, :3] += np.eye(3) * self.r_floor[0] ** 2
        R[3:, 3:] += np.eye(3) * self.r_floor[1] ** 2
        return R

    def _predict(self, tr: Track, t: float):
        dt = t - tr.t
        if dt <= 0:
            return tr.x, tr.P
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = self.q ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = np.eye(3) * q * dt ** 4 / 4
        Q[:3, 3:] = Q[3:, :3] = np.eye(3) * q * dt ** 3 / 2
        Q[3:, 3:] = np.eye(3) * q * dt ** 2
        return F @ tr.x, F @ tr.P @ F.T + Q

    def update(self, fixes: list[Fix], t: float) -> list[Track]:
        preds = [self._predict(tr, t) for tr in self.tracks]
        n_t, n_f = len(self.tracks), len(fixes)
        cost = np.full((n_t, n_f), 1e9)
        for i, (x, P) in enumerate(preds):
            for j, fx in enumerate(fixes):
                z = np.concatenate([fx.pos, fx.vel])
                S = P + self._R(fx)
                y = z - x
                d2 = float(y @ np.linalg.solve(S, y))
                if d2 <= self.gate:
                    cost[i, j] = d2
        assigned_t, assigned_f = set(), set()
        if n_t and n_f:
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e9:
                    continue
                x, P = preds[i]
                fx = fixes[j]
                z = np.concatenate([fx.pos, fx.vel])
                S = P + self._R(fx)
                K = P @ np.linalg.inv(S)
                tr = self.tracks[i]
                tr.x = x + K @ (z - x)
                tr.P = (np.eye(6) - K) @ P
                tr.t, tr.hits, tr.misses = t, tr.hits + 1, 0
                tr.confirmed |= tr.hits >= self.confirm_hits
                tr.history.append((t, tr.x[:3].copy()))
                assigned_t.add(i)
                assigned_f.add(j)
        survivors = []
        for i, tr in enumerate(self.tracks):
            if i not in assigned_t:
                tr.x, tr.P = preds[i]
                tr.t = t
                tr.misses += 1
            limit = self.max_misses if tr.confirmed else 1
            if tr.misses <= limit:
                survivors.append(tr)
        self.tracks = survivors
        for j, fx in enumerate(fixes):
            if j in assigned_f:
                continue
            z = np.concatenate([fx.pos, fx.vel])
            tr = Track(self._next, z, self._R(fx), t, history=[(t, fx.pos.copy())])
            self._next += 1
            self.tracks.append(tr)
        return self.tracks
