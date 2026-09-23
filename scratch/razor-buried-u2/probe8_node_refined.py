"""Probe 8 (momwire#1149 U2): probe 2's ladder refines the FAR mesh only --
`crossing_deck`'s node-adjacent edges (-0.1..0 and 0..0.1, two 50 mm
segments each) keep their counts at every rung, so any node-region
discretisation error is a constant of that ladder. Here EVERY edge is
multiplied, node edges included. Both ports stay on knots at every rung.

Usage: probe8_node_refined.py <eps_name> <rungs...>
"""

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import numpy as np
from test_crossing_serve_524 import crossing_deck

import momwire
from momwire import razor as _razor
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

HERE = Path(__file__).resolve().parent
assert str(HERE.parents[1]) in momwire.__file__, momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_CROSSING = True
EPS = {
    "eps1": (1.0, 0.0),
    "soilA": None,
    "diel13": (13.0, 0.0),
    "diel80": (80.0, 0.0),
    "lossy": (13.0, 0.05),
}


def deck(m, eps):
    d = crossing_deck(1)
    d["n_per_edge_per_wire"] = [[n * m for n in e] for e in d["n_per_edge_per_wire"]]
    d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
    if eps is not None:
        d["ground_eps"] = eps
    return d


eps_name = sys.argv[1]
prev = None
for m in [int(x) for x in sys.argv[2:]]:
    d = deck(m, EPS[eps_name])
    d.pop("junctions")
    t0 = time.perf_counter()
    Y = np.asarray(RazorSolver(**d, nec5_quadrature=True).compute_y_matrix())
    dt = time.perf_counter() - t0
    Yb = np.asarray(BSplineSolver(**deck(m, EPS[eps_name])).compute_y_matrix())
    Z, Zb = np.linalg.inv(Y), np.linalg.inv(Yb)
    nr = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
    ratio = "" if prev is None else f" ({prev / nr:4.2f}x)"
    prev = nr
    print(
        f"{eps_name:6s} x{m:<2d} nonrec {nr:.3e}{ratio}  Z11 {Z[0, 0]:.3f} "
        f"(bs {Zb[0, 0]:.3f})  Z22 {Z[1, 1]:.3f} (bs {Zb[1, 1]:.3f})  {dt:.1f}s",
        flush=True,
    )
