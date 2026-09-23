"""Probe 5 (momwire#1149 U2): is probe 2's x16 floor the Sommerfeld grids'
interpolation? Refines both families' inner-zone lattices by U2F (dr and
dtheta / U2F). The ABOVE grid's spacings are inline in `_sommerfeld.py`, so
this needed a TEMPORARY, uncommitted edit there (the two inner layout rows
divided by a module `_U2F` read from the environment), reverted after the
run; the below grid's are module constants patched here. Log:
probe5_grid_refine.log. Verdict: Z moves ~1e-3 ohm (the grid did change),
the floor does not move at all -- not the grids. (Probe 8 found the floor:
probe 2's ladder never refines the node-adjacent edges.)
"""

import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import numpy as np
from test_crossing_serve_524 import crossing_deck

from momwire import _sommerfeld, _sommerfeld_below
from momwire import razor as _razor
from momwire.razor import RazorSolver

f = float(os.environ.get("U2F") or "1")
assert getattr(_sommerfeld, "_U2F", 1.0) == f, "the temporary edit is not in place"
_sommerfeld_below._SOMM_BELOW_DR_NEAR_LAMBDA_M /= f
_sommerfeld_below._SOMM_BELOW_DTH_STEEP_DEG /= f
warnings.filterwarnings("ignore")
_razor._SERVE_CROSSING = True


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


for eps_name, eps in (("soilA", None), ("diel13", (13.0, 0.0))):
    row = []
    for m in [int(x) for x in os.environ.get("RUNGS", "4 8 16").split()]:
        Y = np.asarray(
            RazorSolver(**deck(m, eps), nec5_quadrature=True).compute_y_matrix()
        )
        Z = np.linalg.inv(Y)
        nr = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
        row.append(f"x{m} {nr:.3e} Z11 {Z[0, 0]:.4f}")
    print(f"U2F={f} {eps_name}: " + " | ".join(row), flush=True)
