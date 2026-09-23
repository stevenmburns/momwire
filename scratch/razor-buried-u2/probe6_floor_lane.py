"""Probe 6 (momwire#1149 U2): the ~7e-6 floor at x16 -- does it keep
falling at x32, and is it the same on the Gauss-Legendre path lane?

Usage: probe6_floor_lane.py <nec5|gl> <eps_name> <rungs...>
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
from momwire.razor import RazorSolver

HERE = Path(__file__).resolve().parent
assert str(HERE.parents[1]) in momwire.__file__, momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_CROSSING = True
EPS = {"soilA": None, "diel13": (13.0, 0.0), "eps1": (1.0, 0.0)}


def deck(m, eps):
    d = crossing_deck(1)
    b, a = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [
        [n * m for n in b[:-1]] + [b[-1]],
        [a[0]] + [n * m for n in a[1:]],
    ]
    d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
    if eps is not None:
        d["ground_eps"] = eps
    d.pop("junctions")
    return d


lane, eps_name = sys.argv[1], sys.argv[2]
prev = None
for m in [int(x) for x in sys.argv[3:]]:
    t0 = time.perf_counter()
    s = RazorSolver(**deck(m, EPS[eps_name]), nec5_quadrature=(lane == "nec5"))
    Y = np.asarray(s.compute_y_matrix())
    nr = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
    ratio = "" if prev is None else f" ({prev / nr:4.2f}x)"
    prev = nr
    print(
        f"{lane} {eps_name} x{m} nonrec {nr:.3e}{ratio} "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )
