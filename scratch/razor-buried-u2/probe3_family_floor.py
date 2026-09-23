"""Probe 3 (momwire#1149 U2): where does the crossing deck's reciprocity
floor (~7e-6 at x16, probe 2) come from?

The same graded two-wire deck translated rigidly so it touches no plane:
wholly below (the #812 family alone) and wholly above (razor's composing
ground alone), taken to x16. If either family floors at the same level with
no node anywhere, the crossing deck's floor is the families' (grids and
quadrature), not a remaining node defect.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import numpy as np
from test_crossing_serve_524 import crossing_deck

import momwire
from momwire.razor import RazorSolver

HERE = Path(__file__).resolve().parent
assert str(HERE.parents[1]) in momwire.__file__, momwire.__file__
warnings.filterwarnings("ignore")


def deck(m, shift, eps):
    d = crossing_deck(1)
    b, a = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [
        [n * m for n in b[:-1]] + [b[-1]],
        [a[0]] + [n * m for n in a[1:]],
    ]
    d["wires"] = [w + np.array([0.0, 0.0, shift]) for w in d["wires"]]
    d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
    if eps is not None:
        d["ground_eps"] = eps
    d.pop("junctions")
    return d


rungs = [int(x) for x in sys.argv[1:]] or [4, 8, 16]
for eps_name, eps in (("soilA", None), ("diel13", (13.0, 0.0))):
    for label, shift in (("below", -10.3), ("above", 2.3)):
        prev = None
        for m in rungs:
            Y = np.asarray(
                RazorSolver(
                    **deck(m, shift, eps), nec5_quadrature=True
                ).compute_y_matrix()
            )
            nr = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
            ratio = "" if prev is None else f" ({prev / nr:5.2f}x)"
            prev = nr
            print(
                f"{eps_name:6s} {label:5s} x{m:<2d} nonrec {nr:.3e}{ratio}", flush=True
            )
