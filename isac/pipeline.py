"""Sensing nodes (one per Tx/Rx pair) and the fusion centre.

  SDR / RU samples ──► SensingNode ──► detections ──► FusionCenter ──► tracks
                      (sync, OFDM demod,          (multistatic fix,
                       reference, RD map, CFAR)    Kalman tracking)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import fftconvolve

from .cfar import Detection, detect
from .clutter import eca
from .localize import MultistaticLocator, Pair
from .numerology import C, Numerology
from .ofdm import demodulate, symbols_in_buffer
from .processing import RDMap, SensingConfig, combine, range_doppler
from .reference import KnownGridReference, LteCrsReference
from .sync import SyncResult, acquire
from .tracker import Tracker


@dataclass
class CpiResult:
    pair_id: str
    t: float
    rd: RDMap
    detections: list[Detection]
    info: dict


def symbols_per_cpi(num: Numerology, cpi_s: float) -> int:
    return int(round(cpi_s * 1e3 * 14 * (2 ** num.mu)))


class LtePassiveNode:
    """Single-channel passive LTE receiver using the CRS of one eNB/cell."""

    def __init__(self, pair_id: str, num: Numerology, cfg: SensingConfig,
                 n_id: int | None = None, ports=(0,), cpi_s: float = 0.05):
        self.pair_id, self.num, self.cfg = pair_id, num, cfg
        self.n_id, self.ports, self.cpi_s = n_id, ports, cpi_s
        self.last_sync: SyncResult | None = None

    def process(self, x: np.ndarray, t0: float = 0.0, max_cpis: int | None = None) -> list[CpiResult]:
        num = self.num
        s = acquire(x, num, self.n_id)
        if self.n_id is None and s.pci_confidence_db > 6:
            self.n_id = s.n_id                     # lock the cell after first acquisition
        self.last_sync = s
        n = np.arange(len(x))
        xc = x * np.exp(-2j * np.pi * s.cfo_hz * n / num.fs).astype(np.complex64)
        n_tot = symbols_in_buffer(num, len(x), s.start + 1, s.l0)  # +1: room for backoff
        n_cpi = symbols_per_cpi(num, self.cpi_s)
        starts = num.symbol_starts(max(n_tot, 1), s.l0)
        refs = [LteCrsReference(num, s.n_id, p) for p in self.ports]
        out = []
        for c, off in enumerate(range(0, n_tot - n_cpi // 2, n_cpi)):
            if max_cpis is not None and c >= max_cpis:
                break
            n_sym = min(n_cpi, n_tot - off)
            st = s.start + int(starts[off])
            l0 = s.l0 + off
            Y = demodulate(xc, num, n_sym, l0=l0, start=st)
            times = num.symbol_times(n_sym, l0) + st / num.fs + t0
            maps = []
            for ref in refs:
                X, valid = ref.grid(n_sym, l0)
                maps.append(range_doppler(Y, X, valid, times, num, self.cfg))
            rd = combine(maps)
            tc = float(times[0])
            out.append(CpiResult(self.pair_id, tc, rd, detect(rd, self.cfg, self.pair_id, tc),
                                 {"n_id": s.n_id, "cfo_hz": s.cfo_hz, "pss_snr_db": s.pss_snr_db,
                                  "pci_confidence_db": s.pci_confidence_db}))
        return out


def estimate_lag(tx: np.ndarray, rx: np.ndarray, max_lag: int, n: int = 1 << 16) -> int:
    """Integer lag of the strongest Tx->Rx path (direct path) by cross-correlation."""
    n = min(n, len(tx) - max_lag, len(rx) - max_lag)
    c = fftconvolve(rx[: n + max_lag], np.conj(tx[:n][::-1]), mode="valid")
    return int(np.argmax(np.abs(c)))


class TxTapNode:
    """Network-integrated ISAC (or two-channel passive radar).

    ``tx`` is the transmitted baseband (tapped at the DU / RU / fronthaul or
    a high-SNR reference antenna), ``rx`` the sensing receiver, both on a
    common time base (GNSS/PTP). Every RE is known -> maximum processing gain.
    """

    def __init__(self, pair_id: str, num: Numerology, cfg: SensingConfig, cpi_s: float = 0.05,
                 max_lag_s: float = 50e-6, use_eca: bool = True):
        self.pair_id, self.num, self.cfg, self.cpi_s = pair_id, num, cfg, cpi_s
        self.max_lag = int(max_lag_s * num.fs)
        self.use_eca = use_eca

    def process(self, tx: np.ndarray, rx: np.ndarray, start: int = 0, l0: int = 0,
                t0: float = 0.0, max_cpis: int | None = None) -> list[CpiResult]:
        num = self.num
        lag = estimate_lag(tx[start:], rx[start:], self.max_lag)
        n_tot = symbols_in_buffer(num, len(rx) - lag - 1, start, l0)
        n_cpi = symbols_per_cpi(num, self.cpi_s)
        starts = num.symbol_starts(max(n_tot, 1), l0)
        cfg = self.cfg
        if self.use_eca:
            # ECA removes the direct path, so ranges must be referenced to the
            # Tx/Rx clock; fall back to the measured lag if the baseline is unknown.
            cfg = replace(cfg, cpe_correction=False)
            if cfg.range_reference != "absolute":
                cfg = replace(cfg, range_reference="absolute", baseline_m=lag / num.fs * C)
        n_neg = 4
        n_taps = int(np.ceil(cfg.max_range_m / C * num.fs)) + n_neg + 8
        out = []
        for c, off in enumerate(range(0, n_tot - n_cpi // 2, n_cpi)):
            if max_cpis is not None and c >= max_cpis:
                break
            n_sym = min(n_cpi, n_tot - off)
            st = start + int(starts[off])
            seg = int(num.symbol_starts(n_sym, l0 + off)[-1]) + int(num.cp_len(l0 + off + n_sym - 1)) + num.nfft
            xs = tx[st:st + seg]
            ys = rx[st + lag:st + lag + seg]
            if self.use_eca:
                ys = eca(xs, ys, n_taps, n_neg)
            X = demodulate(xs, num, n_sym, l0=l0 + off, start=0)
            Y = demodulate(ys, num, n_sym, l0=l0 + off, start=0)
            times = num.symbol_times(n_sym, l0 + off) + (st + lag) / num.fs + t0
            ref = KnownGridReference(X)
            Xr, valid = ref.grid(n_sym)
            rd = range_doppler(Y, Xr, valid, times, num, cfg, timing_offset_s=lag / num.fs)
            tc = float(times[0])
            out.append(CpiResult(self.pair_id, tc, rd, detect(rd, cfg, self.pair_id, tc),
                                 {"lag_samples": lag, "lag_m": lag / num.fs * C}))
        return out


def process_grids(pair_id: str, Y: np.ndarray, X: np.ndarray, times: np.ndarray,
                  num: Numerology, cfg: SensingConfig) -> CpiResult:
    """Entry point when frequency-domain grids are available directly, e.g. an
    O-RAN 7.2x fronthaul tap (U-plane IQ per PRB) or a gNB/dApp hook."""
    Xr, valid = KnownGridReference(X).grid(X.shape[1])
    rd = range_doppler(Y, Xr, valid, times, num, cfg)
    return CpiResult(pair_id, float(times[0]), rd, detect(rd, cfg, pair_id, float(times[0])), {})


class FusionCenter:
    def __init__(self, pairs: list[Pair], area, origin=None, locator_kw=None, tracker_kw=None):
        self.locator = MultistaticLocator(pairs, area, **(locator_kw or {}))
        self.tracker = Tracker(**(tracker_kw or {}))
        self.origin = origin

    def step(self, detections: list[Detection], t: float):
        fixes = self.locator.locate(detections)
        tracks = self.tracker.update(fixes, t)
        return fixes, tracks


def config_with(cfg: SensingConfig, **kw) -> SensingConfig:
    return replace(cfg, **kw)
