"""Command line: ``python -m isac {demo,run,record} ...``"""
from __future__ import annotations

import argparse
import json
import socket
import sys

import numpy as np


def _cmd_demo(a):
    from . import demo
    fn = demo.run_lte if a.mode == "lte" else demo.run_nr
    fn(n_cpi=a.cpis, seed=a.seed, plot=a.plot, json_path=a.json)


def _sensing_cfg(d: dict):
    from .processing import SensingConfig
    d = dict(d or {})
    for k in ("cfar_guard", "cfar_train"):
        if k in d:
            d[k] = tuple(d[k])
    return SensingConfig(**d)


def _carrier(c: dict):
    from .numerology import lte_carrier, nr_carrier
    if c["rat"].upper() == "LTE":
        return lte_carrier(c["bw_mhz"], float(c["fc_hz"]))
    return nr_carrier(int(c["n_rb"]), int(c["mu"]), float(c["fc_hz"]), c.get("nfft"))


def _cmd_run(a):
    """Process time-aligned IQ recordings described by a YAML config."""
    import yaml
    from dataclasses import replace

    from .localize import Pair
    from .pipeline import FusionCenter, LtePassiveNode, TxTapNode
    from .sources import n_samples, read_iq

    conf = yaml.safe_load(open(a.config))
    num = _carrier(conf["carrier"])
    base_cfg = _sensing_cfg(conf.get("sensing"))
    cpi_s = float(conf.get("cpi_s", 0.05))
    pairs, nodes = [], []
    for p in conf["pairs"]:
        tx, rx = np.array(p["tx"], float), np.array(p["rx"], float)
        pairs.append(Pair(p["id"], tx, rx))
        if p.get("mode", "crs") == "crs":
            node = LtePassiveNode(p["id"], num, base_cfg, p.get("pci"), tuple(p.get("ports", [0])), cpi_s)
        else:
            cfg = replace(base_cfg, range_reference="absolute", baseline_m=float(np.linalg.norm(tx - rx)))
            node = TxTapNode(p["id"], num, cfg, cpi_s)
        nodes.append((p, node))
    loc = conf.get("locator", {})
    area = tuple(loc.pop("area", (-5000, 5000, -5000, 5000)))
    org = conf.get("origin")
    origin = (org["lat"], org["lon"], org.get("alt", 0.0)) if org else None
    fusion = FusionCenter(pairs, area, origin, locator_kw=loc, tracker_kw=conf.get("tracker"))

    out = conf.get("output", {})
    jf = open(out["jsonl"], "w") if out.get("jsonl") else None
    udp = None
    if out.get("udp"):
        host, port = out["udp"].rsplit(":", 1)
        udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port)))

    cpi_samples = int(round(cpi_s * num.fs))
    block = cpi_samples + int(0.012 * num.fs)
    fmt = conf.get("format", "cf32")
    total = min(n_samples(p["file"], fmt) for p, _ in nodes)
    off = int(conf.get("skip_s", 0) * num.fs)
    while off + block <= total:
        t = off / num.fs
        dets = []
        for p, node in nodes:
            rx = read_iq(p["file"], fmt, off, block)
            if isinstance(node, LtePassiveNode):
                res = node.process(rx, t0=t, max_cpis=1)
            else:
                txs = read_iq(p["tx_file"], fmt, off, block)
                res = node.process(txs, rx, t0=t, max_cpis=1)
            for r in res:
                dets += r.detections
                if a.verbose:
                    print(f"  {r.pair_id}: {len(r.detections)} det", file=sys.stderr)
        fixes, tracks = fusion.step(dets, t)
        rec = {"t": round(t, 3), "detections": [d.to_dict() for d in dets],
               "fixes": [f.to_dict() for f in fixes],
               "tracks": [tr.to_dict(origin) for tr in tracks if tr.confirmed]}
        line = json.dumps(rec)
        print(line)
        if jf:
            jf.write(line + "\n")
        if udp:
            udp[0].sendto(line.encode(), udp[1])
        off += cpi_samples
    if jf:
        jf.close()


def _cmd_record(a):
    from .sources import record_soapy
    outs = a.out.split(",")
    chans = tuple(range(len(outs)))
    record_soapy(a.args, a.freq, a.rate, a.seconds, a.gain, outs, chans, a.antenna)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="isac", description="4G/5G ISAC drone sensing")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run a simulated multistatic scenario end to end")
    d.add_argument("mode", choices=["lte", "nr"])
    d.add_argument("--cpis", type=int, default=8)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--plot", help="save range-Doppler maps to this PNG")
    d.add_argument("--json", help="write per-CPI fixes/tracks/truth as JSON lines")
    d.set_defaults(fn=_cmd_demo)

    r = sub.add_parser("run", help="process IQ recordings from a YAML config")
    r.add_argument("config")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(fn=_cmd_run)

    c = sub.add_parser("record", help="record IQ with SoapySDR (one file per channel)")
    c.add_argument("--args", default="", help='device args, e.g. "driver=uhd"')
    c.add_argument("--freq", type=float, required=True)
    c.add_argument("--rate", type=float, required=True, help="use the carrier's native rate, e.g. 15.36e6")
    c.add_argument("--seconds", type=float, default=5)
    c.add_argument("--gain", type=float, default=40)
    c.add_argument("--antenna")
    c.add_argument("--out", required=True, help="comma-separated output files, one per channel")
    c.set_defaults(fn=_cmd_record)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
