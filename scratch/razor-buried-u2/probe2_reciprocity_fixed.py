"""Probe 2 (momwire#1149 U2): the fix as implemented in razor.py, through
the REAL constructor (no monkeypatch), on probe 1's two-port crossing deck.

Counts which branch ran: `_crossing_node_charges` is wrapped with a counter
so a green ladder cannot come from a route that skipped it.
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

CALLS = [0]
_orig = RazorSolver._crossing_node_charges


def counted(self, *a, **kw):
    CALLS[0] += 1
    return _orig(self, *a, **kw)


RazorSolver._crossing_node_charges = counted

MEDIA = {
    "eps1": (1.0, 0.0),
    "soilA": None,
    "diel13": (13.0, 0.0),
    "diel80": (80.0, 0.0),
    "lossy": (13.0, 0.05),
}


def deck(m, medium):
    d = crossing_deck(1)
    b, a = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [
        [n * m for n in b[:-1]] + [b[-1]],
        [a[0]] + [n * m for n in a[1:]],
    ]
    d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
    if MEDIA[medium] is not None:
        d["ground_eps"] = MEDIA[medium]
    return d


if __name__ == "__main__":
    rungs = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8]
    for medium in MEDIA:
        prev = None
        for m in rungs:
            d = deck(m, medium)
            d.pop("junctions")
            c0 = CALLS[0]
            t0 = time.perf_counter()
            Y = np.asarray(RazorSolver(**d, nec5_quadrature=True).compute_y_matrix())
            dt = time.perf_counter() - t0
            Yb = np.asarray(BSplineSolver(**deck(m, medium)).compute_y_matrix())
            Z, Zb = np.linalg.inv(Y), np.linalg.inv(Yb)
            nr = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
            ratio = "" if prev is None else f" ({prev / nr:5.2f}x)"
            prev = nr
            print(
                f"{medium:6s} x{m:<2d} calls {CALLS[0] - c0} nonrec {nr:.3e}{ratio}  "
                f"Z11 {Z[0, 0]:.3f} (bs {Zb[0, 0]:.3f})  "
                f"Z22 {Z[1, 1]:.3f} (bs {Zb[1, 1]:.3f})  {dt:.2f}s",
                flush=True,
            )
