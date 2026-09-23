"""Probe 16 (momwire#1149 U2): hub_deck with N = 1/2/4/8 radials, every edge
refined x1/x2/x4, feed on a knot (monopole 4.0): razor fixed and unfixed
against bspline d2. The scoping signature was a node resistance growing
with the buried members; unfixed reproduces it (+9.9 -> +21.7 ohm R), the
fixed gap shrinks with mesh at every N. Run from the worktree root."""

import sys
import warnings

sys.path.insert(0, "tests")
import numpy as np
from test_crossing_serve_524 import hub_deck
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

warnings.filterwarnings("ignore")
fixed = RazorSolver._crossing_node_charges


def zero(self, geom, tents, *a, **kw):
    return np.zeros((geom["n_basis_total"], len(tents)), dtype=complex)


for n in (1, 2, 4, 8):
    for m in (1, 2, 4):
        d = hub_deck(n_radials=n)
        d["feeds"] = [(len(d["wires"]) - 1, 4.0, 1 + 0j)]
        d["n_per_edge_per_wire"] = [
            [x * m for x in e] for e in d["n_per_edge_per_wire"]
        ]
        zb = complex(BSplineSolver(**d).compute_impedance()[0])
        dd = {k: v for k, v in d.items() if k != "junctions"}
        RazorSolver._crossing_node_charges = fixed
        zf = complex(RazorSolver(**dd, nec5_quadrature=True).compute_impedance()[0])
        RazorSolver._crossing_node_charges = zero
        zu = complex(RazorSolver(**dd, nec5_quadrature=True).compute_impedance()[0])
        print(
            f"N={n} x{m} bs {zb:.3f} fixed gap {zf - zb:.3f} unfixed gap {zu - zb:.3f}",
            flush=True,
        )
