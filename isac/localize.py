"""Multistatic localisation: bistatic (excess range, range-rate) detections
from several Tx/Rx pairs -> 3-D position and velocity.

Each detection constrains the target to an ellipsoid with foci Tx and Rx.
Three or more pairs give a position fix; a soft altitude prior keeps the
(usually poorly observed) vertical axis well conditioned because masts are
all at similar heights. Detections are associated across pairs by testing
combinations and keeping the most consistent ones (ghost suppression).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .cfar import Detection
from .geometry import bistatic_excess


@dataclass
class Pair:
    pair_id: str
    tx: np.ndarray
    rx: np.ndarray

    def __post_init__(self):
        self.tx = np.asarray(self.tx, float)
        self.rx = np.asarray(self.rx, float)


@dataclass
class Fix:
    pos: np.ndarray
    vel: np.ndarray
    cov: np.ndarray            # 6x6 (pos, vel)
    chi2_dof: float
    n_pairs: int
    t: float
    detections: list = field(default_factory=list)

    def to_dict(self):
        return {"t": self.t, "pos": self.pos.round(1).tolist(), "vel": self.vel.round(2).tolist(),
                "n_pairs": self.n_pairs, "chi2_dof": round(self.chi2_dof, 2)}


class MultistaticLocator:
    def __init__(self, pairs: list[Pair], area: tuple[float, float, float, float],
                 sigma_range=12.0, sigma_rate=1.0, z_prior=100.0, sigma_z=80.0,
                 sigma_vh_prior=40.0, sigma_vz_prior=8.0, grid_step=40.0, min_pairs=3,
                 max_per_pair=5, gate_chi2_dof=6.0, n_seeds=4):
        self.pairs = {p.pair_id: p for p in pairs}
        self.sr, self.sv = sigma_range, sigma_rate
        self.z0, self.sz = z_prior, sigma_z
        self.v_prior_w = np.diag([1 / sigma_vh_prior ** 2] * 2 + [1 / sigma_vz_prior ** 2])
        self.min_pairs, self.max_per_pair, self.gate = min_pairs, max_per_pair, gate_chi2_dof
        self.n_seeds, self.step = n_seeds, grid_step
        xs = np.arange(area[0], area[1] + grid_step, grid_step)
        ys = np.arange(area[2], area[3] + grid_step, grid_step)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self.grid = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, z_prior)], axis=1)
        self.grid_model = {pid: bistatic_excess(self.grid, p.tx, p.rx) for pid, p in self.pairs.items()}

    # ------------------------------------------------------------------ core
    def _units(self, p, pr: Pair):
        ut = p - pr.tx
        ur = p - pr.rx
        return ut / np.linalg.norm(ut), ur / np.linalg.norm(ur)

    def _seeds(self, cost):
        order = np.argsort(cost)[:400]
        seeds = []
        for i in order:
            p = self.grid[i]
            if all(np.linalg.norm(p[:2] - q[:2]) > 4 * self.step for q in seeds):
                seeds.append(p.copy())
            if len(seeds) >= self.n_seeds:
                break
        return seeds

    def _refine(self, p, pids, rho):
        lam = 1e-3
        for _ in range(25):
            res, J = [], []
            for i, pid in enumerate(pids):
                pr = self.pairs[pid]
                ut, ur = self._units(p, pr)
                res.append((rho[i] - bistatic_excess(p, pr.tx, pr.rx)) / self.sr)
                J.append((ut + ur) / self.sr)
            res.append((self.z0 - p[2]) / self.sz)
            J.append(np.array([0, 0, 1.0]) / self.sz)
            res, J = np.array(res), np.array(J)
            A = J.T @ J
            step = np.linalg.solve(A + lam * np.diag(np.diag(A)), J.T @ res)
            p = p + step
            if p[2] < 0:
                p[2] = -p[2]                 # mirror solution below ground
            if np.linalg.norm(step) < 0.05:
                break
        return p, J

    def solve(self, combo: list[tuple[str, Detection]]):
        pids = [c[0] for c in combo]
        rho = np.array([c[1].excess_range_m for c in combo])
        rr = np.array([c[1].range_rate_mps for c in combo])
        cost = sum(((rho[i] - self.grid_model[pid]) / self.sr) ** 2 for i, pid in enumerate(pids))
        best = None
        for seed in self._seeds(cost):
            p, J = self._refine(seed, pids, rho)
            # velocity: linear least squares with a weak zero-mean prior (slow climb rates)
            Av = np.array([np.add(*self._units(p, self.pairs[pid])) for pid in pids])
            P_vel = np.linalg.inv(Av.T @ Av / self.sv ** 2 + self.v_prior_w)
            v = P_vel @ (Av.T @ rr / self.sv ** 2)
            r_rng = np.array([(rho[i] - bistatic_excess(p, self.pairs[pid].tx, self.pairs[pid].rx))
                              for i, pid in enumerate(pids)]) / self.sr
            r_rate = (rr - Av @ v) / self.sv
            chi2 = float(np.sum(r_rng ** 2) + np.sum(r_rate ** 2) + ((p[2] - self.z0) / self.sz) ** 2
                         + v @ self.v_prior_w @ v)
            if best is None or chi2 < best[0]:
                best = (chi2, p, v, np.linalg.inv(J.T @ J), P_vel)
        chi2, p, v, P_pos, P_vel = best
        n = len(pids)
        dof = max(2 * n - 2, 1)          # 2n measurements + 4 prior terms - 6 unknowns
        cov = np.zeros((6, 6))
        cov[:3, :3], cov[3:, 3:] = P_pos, P_vel
        return Fix(pos=p, vel=v, cov=cov, chi2_dof=chi2 / dof, n_pairs=n,
                   t=combo[0][1].t, detections=[c[1] for c in combo])

    def locate(self, detections: list[Detection]) -> list[Fix]:
        by_pair: dict[str, list[Detection]] = {}
        for d in detections:
            if d.pair_id in self.pairs:
                by_pair.setdefault(d.pair_id, []).append(d)
        pids = sorted(by_pair)
        if len(pids) < self.min_pairs:
            return []
        options = [[None] + sorted(by_pair[pid], key=lambda d: -d.snr_db)[: self.max_per_pair]
                   for pid in pids]
        cands = []
        for choice in itertools.product(*options):
            combo = [(pid, d) for pid, d in zip(pids, choice) if d is not None]
            if len(combo) < self.min_pairs:
                continue
            fx = self.solve(combo)
            if fx.chi2_dof <= self.gate:
                cands.append(fx)
        cands.sort(key=lambda f: (f.chi2_dof - 1.0 * f.n_pairs))
        used, out = set(), []
        for fx in cands:
            ids = {id(d) for d in fx.detections}
            if ids & used:
                continue
            used |= ids
            out.append(fx)
        return out
