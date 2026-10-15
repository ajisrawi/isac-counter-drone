import numpy as np
import pytest

from isac import lte as L
from isac import nr as N
from isac import ofdm, sim
from isac.clutter import eca
from isac.geometry import bistatic_excess, bistatic_rate
from isac.localize import MultistaticLocator, Pair
from isac.cfar import Detection
from isac.numerology import C, lte_carrier, nr_carrier
from isac.pipeline import LtePassiveNode, TxTapNode
from isac.processing import SensingConfig, range_doppler
from isac.sequences import gold, lte_pss, nr_pss, nr_sss
from isac.tracker import Tracker


# ---------------------------------------------------------------- waveforms
def test_numerology_cp_and_frame_length():
    for bw in (1.4, 5, 10, 20):
        num = lte_carrier(bw, 2.6e9)
        assert num.symbol_starts(141)[-1] == num.samples_per_frame  # 140 symbols = 10 ms
    nr = nr_carrier(106, 1, 3.5e9)
    assert nr.nfft == 2048 and nr.cp_len(0) == 144 + 32 and nr.cp_len(1) == 144
    assert nr.symbol_starts(281)[-1] == nr.samples_per_frame


def test_ofdm_roundtrip():
    rng = np.random.default_rng(0)
    for num in (lte_carrier(5, 2e9), nr_carrier(24, 1, 3.5e9)):
        g = (rng.standard_normal((num.n_sc, 30)) + 1j * rng.standard_normal((num.n_sc, 30))).astype(np.complex64)
        x = ofdm.modulate(g, num, 3)
        assert np.allclose(ofdm.demodulate(x, num, 30, 3), g, atol=1e-4)


def test_sequences():
    assert set(np.unique(gold(12345, 200))) <= {0, 1}
    d = lte_pss(1)
    assert len(d) == 62 and np.allclose(np.abs(d), 1)
    assert len(nr_pss(2)) == 127 and set(np.unique(nr_pss(2).real)) == {-1.0, 1.0}
    s = [nr_sss(i, 0) for i in range(5)]
    assert all(abs(np.vdot(s[0], s[i])) < 127 for i in range(1, 5))     # distinct cell IDs


def test_crs_density():
    for pci in (0, 137, 503):
        k, v = L.crs_pilots(pci, 0, 50)
        assert len(k) == 100 and np.all(np.diff(k) == 6) and k[0] == pci % 6
        assert L.crs_pilots(pci, 1, 50) is None
        k4, _ = L.crs_pilots(pci, 4, 50)
        assert k4[0] == (3 + pci % 6) % 6


def test_ssb_structure():
    cfg = N.SsbConfig(n_id=301, k0=100)
    res = N.ssb_known_res(cfg, 0)
    assert [len(k) for k, _ in res] == [127, 60, 24 + 127, 60]
    num = nr_carrier(51, 1, 3.5e9)
    assert N.ssb_pilots(cfg, 2, num) is not None and N.ssb_pilots(cfg, 0, num) is None


# ------------------------------------------------------------ core processing
def test_range_doppler_grid_level():
    """Ideal grid-domain target: peak must land at the right range / rate."""
    num = nr_carrier(51, 1, 3.5e9)
    rng = np.random.default_rng(1)
    n_sym = 560
    X = N.build_grid(num, n_sym, rng=rng)
    t = num.symbol_times(n_sym)
    q = num.sc_freq_index[:, None]
    tau, fd = 400 / C, 150.0
    Y = X * (1.0 + 1e-2 * np.exp(-2j * np.pi * q * num.scs_hz * tau) * np.exp(2j * np.pi * fd * t))
    rd = range_doppler(Y, X, np.abs(X) > 0, t, num, SensingConfig(max_range_m=1000))
    i, j = np.unravel_index(np.argmax(rd.power), rd.power.shape)
    assert abs(rd.excess_range_m[i] - 400) < C / num.fs
    assert abs(rd.doppler_hz[j] - fd) < 1 / (t[-1] - t[0])


def _lte_scene(seed=3, cfo=600.0, offset=23456):
    rng = np.random.default_rng(seed)
    num = lte_carrier(10, 2.655e9)
    x = ofdm.modulate(L.build_frame_grid(num, 222, 14 * 70, rng=rng), num)
    tx = sim.Transmitter("enb", [0, 0, 30], 60.0, 222)
    rx = sim.Receiver("rx", [1500, 0, 15], cfo_hz=cfo)
    tg = sim.Target([700, 500, 100], [10, -5, 0], 0.05)
    pl = sim.paths(tx, rx, num.fc_hz, 0.0, [tg], [sim.Scatterer([500, -200, 20], 2000)])
    y = sim.propagate(x, num.fs, num.fc_hz, pl, sim.noise_mw(num.fs, rx.nf_db), cfo_hz=cfo, rng=rng)
    return num, y[offset:], tx, rx, tg


def test_lte_passive_end_to_end():
    num, y, tx, rx, tg = _lte_scene()
    node = LtePassiveNode("p", num, SensingConfig(max_range_m=2000), None, cpi_s=0.05)
    res = node.process(y, max_cpis=1)
    s = node.last_sync
    assert s.n_id == 222 and abs(s.cfo_hz - 600) < 30
    rho = bistatic_excess(tg.pos0, tx.pos, rx.pos)
    rr = bistatic_rate(tg.pos0, tg.vel, tx.pos, rx.pos)
    best = min(res[0].detections, key=lambda d: abs(d.excess_range_m - rho))
    assert abs(best.excess_range_m - rho) < 8 and abs(best.range_rate_mps - rr) < 0.5
    assert len(res[0].detections) <= 2                                   # clutter cancelled


def test_txtap_with_long_delay_clutter_eca():
    """Clutter far beyond the CP: ECA must keep the target visible."""
    rng = np.random.default_rng(5)
    num = nr_carrier(51, 1, 3.5e9)
    x = ofdm.modulate(N.build_grid(num, 28 * 22, rng=rng), num)
    tx = sim.Transmitter("g", [0, 0, 30], 62.0)
    rx = sim.Receiver("r", [1200, 300, 20], dpi_rejection_db=10)
    tg = sim.Target([500, 700, 100], [-9, 5, 0], 0.02)
    far = [sim.Scatterer([2500, -1800, 50], 50000), sim.Scatterer([-1500, 1900, 60], 50000)]
    pl = sim.paths(tx, rx, num.fc_hz, 0.0, [tg], far)
    y = sim.propagate(x, num.fs, num.fc_hz, pl, sim.noise_mw(num.fs, rx.nf_db), rng=rng)
    base = float(np.linalg.norm(tx.pos - rx.pos))
    cfg = SensingConfig(max_range_m=4500, range_reference="absolute", baseline_m=base)
    rho = bistatic_excess(tg.pos0, tx.pos, rx.pos)
    snr = {}
    for use_eca in (False, True):
        res = TxTapNode("p", num, cfg, cpi_s=0.02, use_eca=use_eca).process(x, y, max_cpis=1)
        hits = [d for d in res[0].detections if abs(d.excess_range_m - rho) < 10]
        snr[use_eca] = max((d.snr_db for d in hits), default=0.0)
    assert snr[True] > 15 and snr[True] > snr[False] + 3


def test_eca_cancels_static_paths():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(20000) + 1j * rng.standard_normal(20000)).astype(np.complex64)
    y = 3 * x + 0.5 * np.roll(x, 40) - 0.2j * np.roll(x, 7)
    z = eca(x, y.astype(np.complex64), 64)
    assert np.mean(np.abs(z[100:]) ** 2) < 1e-3 * np.mean(np.abs(y) ** 2)


# ---------------------------------------------------- localisation / tracking
def _pairs():
    tx = np.array([0, 0, 30.0])
    rxs = [np.array(p, float) for p in ([1400, 100, 15], [-200, 1500, 20], [-1200, -800, 15], [900, -1100, 25])]
    return [Pair(f"p{i}", tx, r) for i, r in enumerate(rxs)]


def test_localize_two_targets_with_ghost_rejection():
    pairs = _pairs()
    truth = [(np.array([600, 500, 120.0]), np.array([12, -6, 0.0])),
             (np.array([-500, 300, 80.0]), np.array([-7, -11, 1.0]))]
    rng = np.random.default_rng(0)
    dets = []
    for p in pairs:
        for pos, vel in truth:
            dets.append(Detection(float(bistatic_excess(pos, p.tx, p.rx) + rng.normal(0, 3)),
                                  float(bistatic_rate(pos, vel, p.tx, p.rx) + rng.normal(0, 0.2)),
                                  0.0, 20.0, p.pair_id))
    fixes = MultistaticLocator(pairs, (-2000, 2000, -2000, 2000)).locate(dets)
    assert len(fixes) == 2
    for pos, vel in truth:
        f = min(fixes, key=lambda f: np.linalg.norm(f.pos - pos))
        assert np.linalg.norm(f.pos[:2] - pos[:2]) < 15
        assert np.linalg.norm(f.vel[:2] - vel[:2]) < 1.5


def test_tracker_confirms_and_follows():
    from isac.localize import Fix
    tr = Tracker()
    for k in range(5):
        t = 0.1 * k
        p = np.array([100 + 10 * t, 50 - 5 * t, 100.0])
        fx = Fix(p + np.random.default_rng(k).normal(0, 2, 3), np.array([10, -5, 0.0]),
                 np.diag([25, 25, 400, 1, 1, 1.0]), 1.0, 4, t)
        tracks = tr.update([fx], t)
    assert len(tracks) == 1 and tracks[0].confirmed
    assert np.linalg.norm(tracks[0].x[3:5] - [10, -5]) < 1.0
