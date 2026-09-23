"""Probe 9 (momwire#1149 U2): driving-point convergence onto bspline d2,
EVERY edge refined, a feed on a knot at every rung (crossing_deck at 4.5 on
the above wire, hub_deck at 4.0 on the monopole), soil A. razor with the U2
term ("fixed") and with it zeroed ("unfixed", the pre-U2 fill), bspline d2
alongside. The bar is the GAP to bspline shrinking with mesh; the equal-mesh
number is not a target (razor is first order in the far mesh, #845).

Usage: probe9_convergence.py <crossing|hub|hubN> <rungs...>
"""

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import numpy as np
from test_crossing_serve_524 import crossing_deck, hub_deck

import momwire
from momwire import razor as _razor
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

HERE = Path(__file__).resolve().parent
assert str(HERE.parents[1]) in momwire.__file__, momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_CROSSING = True
_fixed = RazorSolver._crossing_node_charges


def _zero(self, geom, tents, *a, **kw):
    return np.zeros((geom["n_basis_total"], len(tents)), dtype=np.complex128)


def deck(kind, m):
    if kind == "crossing":
        d = crossing_deck(1)
        d["feeds"] = [(1, 4.5, 1 + 0j)]
    else:
        n = int(kind[3:] or 4)
        d = hub_deck(n_radials=n)
        d["feeds"] = [(len(d["wires"]) - 1, 4.0, 1 + 0j)]
    d["n_per_edge_per_wire"] = [[n * m for n in e] for e in d["n_per_edge_per_wire"]]
    return d


kind = sys.argv[1]
prev = {}
for m in [int(x) for x in sys.argv[2:]]:
    out = {}
    for name in ("bspline", "fixed", "unfixed"):
        d = deck(kind, m)
        t0 = time.perf_counter()
        if name == "bspline":
            z = BSplineSolver(**d).compute_impedance()[0]
        else:
            RazorSolver._crossing_node_charges = _fixed if name == "fixed" else _zero
            d.pop("junctions")
            z = RazorSolver(**d, nec5_quadrature=True).compute_impedance()[0]
        out[name] = (complex(z), time.perf_counter() - t0)
    RazorSolver._crossing_node_charges = _fixed
    zb = out["bspline"][0]
    msg = [f"{kind} x{m:<2d} bs {zb.real:8.3f}{zb.imag:+8.3f}j"]
    for name in ("fixed", "unfixed"):
        z = out[name][0]
        gap = z - zb
        tag = ""
        if name in prev:
            tag = f" (/{abs(prev[name]) / abs(gap):4.2f})"
        prev[name] = gap
        msg.append(
            f"{name} {z.real:8.3f}{z.imag:+8.3f}j gap {gap.real:+7.3f}{gap.imag:+7.3f}j{tag}"
        )
    print("  ".join(msg) + f"  [{out['fixed'][1]:.1f}s]", flush=True)
