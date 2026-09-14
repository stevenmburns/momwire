"""Geometry/plan-only: does the serve plan refuse the two-node deck at
eps~ = 1 (the (b) collapse medium)? No fill, no Z."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
import momwire
from momwire.bspline import BSplineSolver
from test_crossing_serve_524 import crossing_deck

print(momwire.__file__)
for eps, d in (((1.0, 0.0), 12.0), ((1.0, 0.0), 200.0), (None, 12.0)):
    one = crossing_deck() if eps is None else crossing_deck(ground_eps=eps)
    b = dict(one)
    sh = np.array([d, 0.0, 0.0])
    b["wires"] = one["wires"] + [w + sh for w in one["wires"]]
    b["n_per_edge_per_wire"] = one["n_per_edge_per_wire"] + [
        list(e) for e in one["n_per_edge_per_wire"]
    ]
    b["junctions"] = [[(0, "end"), (1, "start")], [(2, "end"), (3, "start")]]
    b["feeds"] = [(1, 4.3333333333, 1 + 0j), (3, 4.3333333333, 1 + 0j)]
    s = BSplineSolver(**b)
    geom = s._build_geometry()
    below = s._below_segments(geom)
    a_idx, b_idx = np.nonzero(~below)[0], np.nonzero(below)[0]
    obs_a = s._buried_nodes(geom, a_idx)[0]
    obs_b = s._buried_nodes(geom, b_idx)[0]
    _e, _em, k_p, k_m, _c2, _am = s._buried_medium()
    try:
        plan = s._buried_serve_plan(geom, a_idx, obs_a, obs_b, k_p, k_m, crossing=True)
        print(f"eps={eps} d={d}: SERVED plan keys {sorted(plan)}")
    except (ValueError, NotImplementedError) as exc:
        print(f"eps={eps} d={d}: REFUSED {type(exc).__name__}: {str(exc)[:160]}")
