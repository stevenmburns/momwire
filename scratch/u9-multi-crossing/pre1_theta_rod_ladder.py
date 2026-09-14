"""Geometry-only preflight: the below/below grazing angle on a two-node
crossing deck at 5/12/50/200 m (no fill, no Z)."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
import momwire
from momwire import bspline
from momwire.bspline import BSplineSolver
from test_crossing_serve_524 import crossing_deck

print(momwire.__file__)


def build(d):
    one = crossing_deck()
    if d is None:
        return one
    b = dict(one)
    sh = np.array([d, 0.0, 0.0])
    b["wires"] = one["wires"] + [w + sh for w in one["wires"]]
    b["n_per_edge_per_wire"] = one["n_per_edge_per_wire"] + [
        list(e) for e in one["n_per_edge_per_wire"]
    ]
    b["junctions"] = [[(0, "end"), (1, "start")], [(2, "end"), (3, "start")]]
    b["feeds"] = [(1, 4.3333333333, 1 + 0j), (3, 4.3333333333, 1 + 0j)]
    return b


for d in (None, 5.0, 12.0, 50.0, 200.0):
    s = BSplineSolver(**build(d))
    geom = s._build_geometry()
    below = s._below_segments(geom)
    b_idx = np.nonzero(below)[0]
    obs_b, _t, _w = s._buried_nodes(geom, b_idx)
    d_b = 0.0 - obs_b[:, 2]
    r1, th = bspline._pair_extents_below(obs_b[:, 0], obs_b[:, 1], d_b)
    eps_t, eps_m, k_p, k_m, c2, a_m = s._buried_medium()
    lam_m = 2 * math.pi / abs(k_m)
    print(
        f"d={d} nodes_b={obs_b.shape[0]} min_depth={d_b.min():.3e} "
        f"r1_max={r1:.3f} m ({r1 / lam_m:.2f} lam_m) th_min={math.degrees(th):.4f} deg "
        f"lam_m={lam_m:.3f}"
    )
