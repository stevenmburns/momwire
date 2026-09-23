"""Probe 4 (momwire#1149 U2): does the crossing deck's ~7e-6 reciprocity
floor (probe 2, x16) move with a quadrature knob? Each knob alone, at
x4 / x8 / x16, soil A and lossless eps_r 13, on probe 2's deck.

  base        shipped
  somm6/12    n_qp_sommerfeld 3 -> 6 / 12 (the remainder's source Gauss,
              which the BELOW family takes flat -- no grazing keying)
  axis        crossing axis q 12 -> 20, panel order 8 -> 12
"""

import sys
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


KNOBS = {
    "base": ({}, None),
    "somm6": ({"n_qp_sommerfeld": 6}, None),
    "somm12": ({"n_qp_sommerfeld": 12}, None),
    "axis": ({}, (20, 12)),
}
which = sys.argv[1:] or list(KNOBS)
for eps_name, eps in (("soilA", None), ("diel13", (13.0, 0.0))):
    for name in which:
        kw, axis = KNOBS[name]
        q0, p0 = _razor._CROSSING_Q, _razor._CROSSING_PANEL_ORDER
        if axis is not None:
            _razor._CROSSING_Q, _razor._CROSSING_PANEL_ORDER = axis
        prev = None
        row = []
        for m in (4, 8, 16):
            Y = np.asarray(
                RazorSolver(
                    **deck(m, eps), nec5_quadrature=True, **kw
                ).compute_y_matrix()
            )
            nr = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
            row.append(
                f"x{m} {nr:.3e}" + ("" if prev is None else f" ({prev / nr:4.2f}x)")
            )
            prev = nr
        _razor._CROSSING_Q, _razor._CROSSING_PANEL_ORDER = q0, p0
        print(f"{eps_name:6s} {name:7s} " + "  ".join(row), flush=True)
