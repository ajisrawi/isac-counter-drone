"""Regenerate the explanatory figures used in the README (docs/images/)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from isac.demo import default_scenario  # noqa: E402
from isac.geometry import bistatic_excess  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
COLORS = ["#2563eb", "#16a34a", "#d97706", "#9333ea"]


def ellipses_figure(tracks_jsonl: str | None = None):
    scn = default_scenario()
    tx = scn.transmitters[0]
    tg = scn.targets[0]
    p = tg.pos(0.0)
    xs = np.linspace(-1900, 1900, 600)
    X, Y = np.meshgrid(xs, xs)
    P = np.stack([X, Y, np.full_like(X, p[2])], axis=-1)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for i, r in enumerate(scn.receivers):
        rho = float(bistatic_excess(p, tx.pos, r.pos))
        Z = bistatic_excess(P, tx.pos, r.pos)
        ax.contour(X, Y, Z, levels=[rho], colors=[COLORS[i]], linewidths=2)
        ax.plot(*r.pos[:2], "s", color=COLORS[i], ms=11, mec="k")
        ax.annotate(f"sensing Rx {i + 1}", r.pos[:2], xytext=(8, 8), textcoords="offset points",
                    fontsize=9, color=COLORS[i], weight="bold")
    ax.plot(*tx.pos[:2], "^", color="#dc2626", ms=15, mec="k")
    ax.annotate("cell site (Tx)", tx.pos[:2], xytext=(10, -16), textcoords="offset points",
                fontsize=10, color="#dc2626", weight="bold")
    for s in scn.scatterers:
        ax.plot(*s.pos[:2], "o", color="#9ca3af", ms=7)
    ax.plot([], [], "o", color="#9ca3af", label="static clutter (removed)")
    for g in scn.targets:
        ax.plot(*g.pos(0)[:2], "*", color="k", ms=16)
    ax.plot([], [], "*", color="k", ms=12, label="drones (truth)")
    if tracks_jsonl and os.path.exists(tracks_jsonl):
        last = [json.loads(line) for line in open(tracks_jsonl)][-1]
        for t in last["tracks"]:
            ax.plot(*t["pos_enu"][:2], "o", mfc="none", mec="#dc2626", ms=20, mew=2)
        ax.plot([], [], "o", mfc="none", mec="#dc2626", ms=12, mew=2, label="ISAC tracks")
    ax.set_title("Each Tx→Rx pair puts the drone on an ellipse;\nthe ellipses intersect at the drone",
                 fontsize=11)
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "multistatic_geometry.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ellipses_figure(sys.argv[1] if len(sys.argv) > 1 else None)
    print("written to", os.path.abspath(OUT))
