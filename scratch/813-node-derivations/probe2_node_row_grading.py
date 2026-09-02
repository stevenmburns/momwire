"""momwire#813: is the node row's converged 5.3e-5 (Haswell's collapse, and
half 1's BAR_ROW_HALF "measured 5.3e-5 (quadrature ...)") the trunk's GRADED
quadrature at the one a-scale singular pair only the node row has?

Off the node row the cross blocks pair different wires and plain Gauss
converges with `_NEAR_Q`. The node row's chopped half ends AT the node, on
the below wire's last segment: the BT term integrates F'_B V(node, z') over
that segment, ~ 1/sqrt(a^2 + s^2) from s = 0 -- razor does it in closed
form (its static half), the trunk by `_graded_u` panels (a-scale, growth
`_NEAR_GROWTH`, Gauss-`_NEAR_GX` per panel). Sweeping `_NEAR_Q` (Haswell's
sweep) never touches those panels, which is exactly the signature reported:
everything else converges, the node row does not move.

So: sweep the GRADING knobs on half 1's own node-row construction.

    python scratch/813-node-derivations/probe2_node_row_grading.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import test_razor_crossing_axis_813 as T

from momwire import _crossing_fill as CF

warnings.simplefilter("ignore")

f = T.free_space.__wrapped__()
rs, geom, jn = f["rs"], f["geom"], f["jn"]
ref = T._razor_half_row(f, "B")[f["cols_below"]]
scale = np.abs(ref).max()

base = (CF._NEAR_GROWTH, CF._NEAR_GX, CF._NEAR_GW, CF._NEAR_Q)
print("node row, above half, vs razor's chopped kernel (relative to max|ref|)")
print(f"{'growth':>7} {'gauss/panel':>12} {'_NEAR_Q':>8}   rel")
for growth, order, q in [
    (4.0, 4, 4),
    (4.0, 4, 16),
    (4.0, 16, 4),
    (2.0, 8, 4),
    (2.0, 16, 4),
    (1.5, 16, 4),
    (1.25, 24, 8),
]:
    CF._NEAR_GROWTH = growth
    CF._NEAR_GX, CF._NEAR_GW = leggauss(order)
    CF._NEAR_Q = q
    A3 = CF.path_test_axis(
        geom["n_basis_total"], rs._path_test_rows(geom, [jn], halves="B")
    )
    B = CF.axis_data(f["ctx"], f["b_idx"])
    got = -CF.cross_complete_block(f["ctx"], A3, B, corner=False)[jn, f["cols_below"]]
    rel = np.abs(got - ref).max() / scale
    print(f"{growth:>7} {order:>12} {q:>8}   {rel:.3e}")
CF._NEAR_GROWTH, CF._NEAR_GX, CF._NEAR_GW, CF._NEAR_Q = base
