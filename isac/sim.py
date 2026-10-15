"""Physically-scaled time-domain simulator for bistatic/multistatic ISAC.

Link budget: Friis for the direct path, bistatic radar equation for targets
and static clutter, thermal noise kTB*NF over the full sample rate. Signals
are returned normalised to unit noise power. Each path gets its own
fractional delay (exact, FFT-based), carrier phase and Doppler shift.
Simplifications: no range migration within a CPI, free-space propagation,
isotropic Tx pattern, no rotor micro-Doppler.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import bistatic_rate
from .numerology import C


@dataclass
class Transmitter:
    tx_id: str
    pos: np.ndarray
    eirp_dbm: float = 60.0
    n_id: int = 0

    def __post_init__(self):
        self.pos = np.asarray(self.pos, float)


@dataclass
class Receiver:
    rx_id: str
    pos: np.ndarray
    gain_dbi: float = 10.0
    nf_db: float = 6.0
    dpi_rejection_db: float = 20.0   # surveillance-antenna null toward the Tx
    cfo_hz: float = 0.0

    def __post_init__(self):
        self.pos = np.asarray(self.pos, float)


@dataclass
class Target:
    pos0: np.ndarray
    vel: np.ndarray
    rcs_m2: float = 0.02             # small multirotor at 2-4 GHz: 0.01-0.1 m^2
    name: str = "uav"

    def __post_init__(self):
        self.pos0 = np.asarray(self.pos0, float)
        self.vel = np.asarray(self.vel, float)

    def pos(self, t: float) -> np.ndarray:
        return self.pos0 + self.vel * t


@dataclass
class Scatterer:
    pos: np.ndarray
    rcs_m2: float = 500.0            # buildings, masts, terrain patches

    def __post_init__(self):
        self.pos = np.asarray(self.pos, float)


@dataclass
class Path:
    delay_s: float
    power_mw: float
    doppler_hz: float
    kind: str


def paths(tx: Transmitter, rx: Receiver, fc: float, t: float,
          targets=(), scatterers=()) -> list[Path]:
    lam = C / fc
    L = np.linalg.norm(tx.pos - rx.pos)
    eirp = 10 ** (tx.eirp_dbm / 10)
    gr = 10 ** (rx.gain_dbi / 10)
    out = [Path(L / C, eirp * gr * (lam / (4 * np.pi * L)) ** 2 * 10 ** (-rx.dpi_rejection_db / 10),
                0.0, "direct")]
    k = eirp * gr * lam ** 2 / (4 * np.pi) ** 3
    for s in scatterers:
        rt, rr = np.linalg.norm(s.pos - tx.pos), np.linalg.norm(s.pos - rx.pos)
        out.append(Path((rt + rr) / C, k * s.rcs_m2 / (rt * rr) ** 2, 0.0, "clutter"))
    for tg in targets:
        p = tg.pos(t)
        rt, rr = np.linalg.norm(p - tx.pos), np.linalg.norm(p - rx.pos)
        rate = float(bistatic_rate(p, tg.vel, tx.pos, rx.pos))
        out.append(Path((rt + rr) / C, k * tg.rcs_m2 / (rt * rr) ** 2, -rate / lam, tg.name))
    return out


def noise_mw(fs: float, nf_db: float) -> float:
    return 10 ** ((-174 + 10 * np.log10(fs) + nf_db) / 10)


def propagate(x: np.ndarray, fs: float, fc: float, plist: list[Path], noise_power_mw: float,
              cfo_hz: float = 0.0, timing_ref_s: float = 0.0, add_noise: bool = True,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """Apply multipath to unit-power baseband ``x``. Sample n of the output is
    received at absolute time n/fs + timing_ref_s (same clock as the Tx)."""
    rng = rng or np.random.default_rng()
    n = len(x)
    p_x = float(np.mean(np.abs(x) ** 2))
    Xf = np.fft.fft(x / np.sqrt(p_x))
    f = np.fft.fftfreq(n, 1 / fs)
    tn = np.arange(n) / fs
    y = np.zeros(n, dtype=np.complex128)
    for p in plist:
        tau = p.delay_s - timing_ref_s
        d = np.fft.ifft(Xf * np.exp(-2j * np.pi * f * tau))
        amp = np.sqrt(p.power_mw / noise_power_mw)
        ph = np.exp(-2j * np.pi * fc * p.delay_s)
        if p.doppler_hz:
            d *= np.exp(2j * np.pi * p.doppler_hz * tn)
        y += amp * ph * d
    if cfo_hz:
        y *= np.exp(2j * np.pi * cfo_hz * tn)
    if add_noise:
        y += (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    return y.astype(np.complex64)


@dataclass
class Scenario:
    transmitters: list[Transmitter]
    receivers: list[Receiver]
    targets: list[Target] = field(default_factory=list)
    scatterers: list[Scatterer] = field(default_factory=list)

    def pair_list(self):
        return [(t, r) for t in self.transmitters for r in self.receivers]
