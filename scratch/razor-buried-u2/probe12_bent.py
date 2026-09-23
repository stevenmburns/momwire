"""Probe 12 (momwire#1149 U2): reciprocity decay on a BENT crossing deck --
crossing_deck's grading with the above wire leaning `lean` degrees off
vertical in x and the below wire leaning the other way in y, so the node
term's horizontal separations and the cross blocks' horizontal dyads are
all exercised (a plumb line is the easy case for both). Every edge refined;
ports on knots (above 4.5, below 1.0). Soil A and eps~ = 1.

Usage: probe12_bent.py <lean_deg> <rungs...>
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


def deck(m, lean, eps):
    d = crossing_deck(1)
    t = np.radians(lean)
    below, above = d["wires"]
    # arclength-preserving leans: z along the wire becomes (sin, 0, cos)*z
    above = np.array([(np.sin(t) * z, 0.0, np.cos(t) * z) for z in above[:, 2]])
    below = np.array([(0.0, -np.sin(t) * z, np.cos(t) * z) for z in below[:, 2]])
    d["wires"] = [below, above]
    d["n_per_edge_per_wire"] = [[n * m for n in e] for e in d["n_per_edge_per_wire"]]
    d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
    if eps is not None:
        d["ground_eps"] = eps
    return d


lean = float(sys.argv[1])
for eps_name, eps in (("soilA", None), ("eps1", (1.0, 0.0))):
    prev = None
    for m in [int(x) for x in sys.argv[2:]]:
        d = deck(m, lean, eps)
        Yb = np.asarray(BSplineSolver(**d).compute_y_matrix())
        d.pop("junctions")
        t0 = time.perf_counter()
        Y = np.asarray(RazorSolver(**d, nec5_quadrature=True).compute_y_matrix())
        dt = time.perf_counter() - t0
        Z, Zb = np.linalg.inv(Y), np.linalg.inv(Yb)
        nr = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
        ratio = "" if prev is None else f" ({prev / nr:4.2f}x)"
        prev = nr
        print(
            f"lean {lean:g} {eps_name:5s} x{m:<2d} nonrec {nr:.3e}{ratio}  Z11 "
            f"{Z[0, 0]:.3f} (bs {Zb[0, 0]:.3f})  Z22 {Z[1, 1]:.3f} "
            f"(bs {Zb[1, 1]:.3f})  {dt:.1f}s",
            flush=True,
        )
