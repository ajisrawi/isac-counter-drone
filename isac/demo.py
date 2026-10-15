"""End-to-end multistatic demos on simulated signals.

lte : passive 4G - one eNB (Band 7, 10 MHz), four single-channel SDR sites
      regenerating the CRS. No operator cooperation needed.
nr  : network-integrated 5G - one gNB (n78, 20 MHz, 30 kHz SCS) whose
      transmitted baseband is tapped at the DU, three sensing receivers at
      neighbouring sites on GNSS/PTP time.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import lte as L
from . import nr as N
from . import ofdm, sim
from .geometry import bistatic_excess, bistatic_rate
from .localize import Pair
from .numerology import C, lte_carrier, nr_carrier
from .pipeline import FusionCenter, LtePassiveNode, TxTapNode, symbols_per_cpi
from .processing import SensingConfig

SITES = [np.array(p, float) for p in ([1400, 100, 15], [-200, 1500, 20], [-1200, -800, 15], [900, -1100, 25])]


def default_scenario(tx_pos=(0, 0, 30), receivers=SITES, eirp=60.0, n_id=137):
    tx = sim.Transmitter("bs0", tx_pos, eirp, n_id)
    rxs = [sim.Receiver(f"rx{i+1}", p, cfo_hz=300.0 * (i + 1) - 700) for i, p in enumerate(receivers)]
    targets = [sim.Target([600, 500, 120], [12, -6, 0], 0.05, "uav1"),
               sim.Target([-500, 300, 80], [-7, -11, 1], 0.05, "uav2")]
    clutter = [sim.Scatterer([600, -300, 20], 2000), sim.Scatterer([1500, 400, 10], 800),
               sim.Scatterer([-900, 700, 25], 1500), sim.Scatterer([200, 1500, 15], 1000)]
    return sim.Scenario([tx], rxs, targets, clutter)


def truth(scn: sim.Scenario, t: float):
    out = []
    for tg in scn.targets:
        out.append({"name": tg.name, "pos": tg.pos(t).round(1).tolist(), "vel": tg.vel.tolist()})
    return out


def _report(fusion, t, fixes, tracks, scn, json_out, verbose):
    tr_conf = [tr for tr in tracks if tr.confirmed]
    rec = {"t": round(t, 3), "fixes": [f.to_dict() for f in fixes],
           "tracks": [tr.to_dict(fusion.origin) for tr in tr_conf], "truth": truth(scn, t)}
    if json_out:
        json_out.write(json.dumps(rec) + "\n")
    if verbose:
        print(f"t={t:6.3f}s  fixes={len(fixes)}  confirmed tracks={len(tr_conf)}")
        for tr in tr_conf:
            best = min(scn.targets, key=lambda g: np.linalg.norm(g.pos(t) - tr.x[:3]))
            err = np.linalg.norm(best.pos(t)[:2] - tr.x[:2])
            print(f"   track {tr.track_id}: pos {tr.x[:3].round(0)}  vel {tr.x[3:].round(1)}"
                  f"  -> {best.name} horiz. error {err:5.1f} m")
    return rec


def run_lte(n_cpi=6, cpi_s=0.1, seed=0, plot=None, json_path=None, verbose=True):
    rng = np.random.default_rng(seed)
    num = lte_carrier(10, 2.655e9)
    scn = default_scenario()
    tx = scn.transmitters[0]
    cfg = SensingConfig(max_range_m=3000.0, max_range_rate_mps=60.0)
    nodes = {r.rx_id: LtePassiveNode(f"{tx.tx_id}-{r.rx_id}", num, cfg, None, cpi_s=cpi_s)
             for r in scn.receivers}
    pairs = [Pair(f"{tx.tx_id}-{r.rx_id}", tx.pos, r.pos) for r in scn.receivers]
    fusion = FusionCenter(pairs, area=(-2000, 2000, -2000, 2000), origin=(52.0, 13.0, 0.0))
    n_sym = symbols_per_cpi(num, cpi_s) + 14 * 12            # CPI + 12 ms for acquisition
    jf = open(json_path, "w") if json_path else None
    first_maps = {}
    t_start = time.time()
    for c in range(n_cpi):
        t = c * cpi_s
        l0 = int(round(t * 14e3)) % 140 // 14 * 14           # subframe-aligned
        grid = L.build_frame_grid(num, tx.n_id, n_sym, l0, load=0.8, rng=rng)
        x = ofdm.modulate(grid, num, l0)
        dets = []
        for r in scn.receivers:
            pl = sim.paths(tx, r, num.fc_hz, t, scn.targets, scn.scatterers)
            y = sim.propagate(x, num.fs, num.fc_hz, pl, sim.noise_mw(num.fs, r.nf_db),
                              cfo_hz=r.cfo_hz, rng=rng)
            y = y[int(rng.integers(0, num.samples_per_frame // 2)):]   # unknown Rx timing
            res = nodes[r.rx_id].process(y, t0=t, max_cpis=1)
            for cr in res:
                dets += cr.detections
                first_maps.setdefault(cr.pair_id, cr)
            if verbose and c == 0:
                s = nodes[r.rx_id].last_sync
                print(f"[{r.rx_id}] PCI={s.n_id} (conf {s.pci_confidence_db:.1f} dB) "
                      f"CFO={s.cfo_hz:+.0f} Hz, {len(res[0].detections)} detections")
                _print_truth(scn, tx, r, t, res[0].detections)
        fixes, tracks = fusion.step(dets, t)
        _report(fusion, t, fixes, tracks, scn, jf, verbose)
    if jf:
        jf.close()
    if verbose:
        print(f"processing time {time.time() - t_start:.1f} s for {n_cpi} CPIs x {len(nodes)} receivers")
    if plot:
        save_plot(first_maps, plot, "Passive LTE (CRS) - bistatic range-Doppler maps, CPI 1")
    return fusion


def run_nr(n_cpi=8, cpi_s=0.04, seed=0, plot=None, json_path=None, verbose=True):
    rng = np.random.default_rng(seed)
    num = nr_carrier(51, 1, 3.5e9)
    scn = default_scenario(receivers=SITES[:3], eirp=62.0)
    for r in scn.receivers:
        r.cfo_hz = 0.0                                       # GNSS-disciplined, shared clock
        r.dpi_rejection_db = 15.0
    for tg in scn.targets:
        tg.rcs_m2 = 0.02
    tx = scn.transmitters[0]
    ssb = N.SsbConfig(n_id=301, k0=100)
    nodes, pairs = {}, []
    for r in scn.receivers:
        base = float(np.linalg.norm(tx.pos - r.pos))
        cfg = SensingConfig(max_range_m=3000.0, max_range_rate_mps=60.0,
                            range_reference="absolute", baseline_m=base)
        nodes[r.rx_id] = TxTapNode(f"{tx.tx_id}-{r.rx_id}", num, cfg, cpi_s=cpi_s)
        pairs.append(Pair(f"{tx.tx_id}-{r.rx_id}", tx.pos, r.pos))
    fusion = FusionCenter(pairs, area=(-2000, 2000, -2000, 2000), origin=(52.0, 13.0, 0.0))
    n_sym = symbols_per_cpi(num, cpi_s) + 28
    jf = open(json_path, "w") if json_path else None
    first_maps = {}
    t_start = time.time()
    for c in range(n_cpi):
        t = c * cpi_s
        grid = N.build_grid(num, n_sym, 0, ssb=ssb, load=0.7, rng=rng)
        x = ofdm.modulate(grid, num)
        dets = []
        for r in scn.receivers:
            pl = sim.paths(tx, r, num.fc_hz, t, scn.targets, scn.scatterers)
            y = sim.propagate(x, num.fs, num.fc_hz, pl, sim.noise_mw(num.fs, r.nf_db), rng=rng)
            res = nodes[r.rx_id].process(x, y, t0=t, max_cpis=1)
            for cr in res:
                dets += cr.detections
                first_maps.setdefault(cr.pair_id, cr)
            if verbose and c == 0:
                print(f"[{r.rx_id}] direct-path lag {res[0].info['lag_m']:.0f} m, "
                      f"{len(res[0].detections)} detections")
                _print_truth(scn, tx, r, t, res[0].detections)
        fixes, tracks = fusion.step(dets, t)
        _report(fusion, t, fixes, tracks, scn, jf, verbose)
    if jf:
        jf.close()
    if verbose:
        print(f"processing time {time.time() - t_start:.1f} s for {n_cpi} CPIs x {len(nodes)} receivers")
    if plot:
        save_plot(first_maps, plot, "Integrated 5G NR (full grid) - bistatic range-Doppler maps, CPI 1")
    return fusion


def _print_truth(scn, tx, r, t, dets):
    for tg in scn.targets:
        p = tg.pos(t)
        rho = float(bistatic_excess(p, tx.pos, r.pos))
        rr = float(bistatic_rate(p, tg.vel, tx.pos, r.pos))
        m = min(dets, key=lambda d: abs(d.excess_range_m - rho) + 5 * abs(d.range_rate_mps - rr), default=None)
        got = (f"det {m.excess_range_m:7.1f} m {m.range_rate_mps:+6.2f} m/s SNR {m.snr_db:4.1f} dB"
               if m is not None else "no detection")
        print(f"     {tg.name}: truth {rho:7.1f} m {rr:+6.2f} m/s | {got}")


def save_plot(maps: dict, path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    for ax, (pid, cr) in zip(axes[0], sorted(maps.items())):
        rd = cr.rd
        p = 10 * np.log10(rd.power / np.median(rd.power))
        ax.imshow(p[:, ::-1].T, origin="lower", aspect="auto", vmin=0, vmax=max(25, p.max()), cmap="viridis",
                  extent=[rd.excess_range_m[0], rd.excess_range_m[-1],
                          rd.range_rate_mps[-1], rd.range_rate_mps[0]])
        for d in cr.detections:
            ax.plot(d.excess_range_m, d.range_rate_mps, "o", mfc="none", mec="r", ms=10)
        ax.set_title(pid, fontsize=9)
        ax.set_xlabel("bistatic excess range (m)")
        ax.set_ylabel("bistatic range rate (m/s)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
