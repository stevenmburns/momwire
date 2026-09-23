"""Probe 7 (momwire#1149 U2): does the flat ~7e-6 floor scale with the wire
radius -- the only fixed small length at the node?

The family remainder Q is evaluated on the true geometry (observer and source
on the axis, no radius), while every other term, and the U2 node-charge term,
regularises with R = sqrt(d^2 + a^2). If the floor is that mismatch it moves
with a.

Usage: probe7_floor_radius.py <eps_name> <a_factor> <rungs...>
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import numpy as np
from test_crossing_serve_524 import A_WIRE, crossing_deck

import momwire
from momwire import razor as _razor
from momwire.razor import RazorSolver

HERE = Path(__file__).resolve().parent
assert str(HERE.parents[1]) in momwire.__file__, momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_CROSSING = True
EPS = {"soilA": None, "diel13": (13.0, 0.0)}


def deck(m, eps, fa):
    d = crossing_deck(1, wire_radius=A_WIRE * fa)
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


eps_name, fa = sys.argv[1], float(sys.argv[2])
prev = None
for m in [int(x) for x in sys.argv[3:]]:
    Y = np.asarray(
        RazorSolver(
            **deck(m, EPS[eps_name], fa), nec5_quadrature=True
        ).compute_y_matrix()
    )
    nr = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
    ratio = "" if prev is None else f" ({prev / nr:4.2f}x)"
    prev = nr
    print(f"{eps_name} a x{fa:g} x{m} nonrec {nr:.3e}{ratio}", flush=True)
