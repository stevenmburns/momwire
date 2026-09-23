"""Probe 11 (momwire#1149 U2): the node term where one node carries SEVERAL
crossing tents, and on a deck with TWO nodes.

  tripodA   a 10 m monopole over three buried legs leaning 45 deg down and
            out, the MONOPOLE listed first in the junction, so every tent at
            the node pairs it with a leg: three crossing tents.
  tripodB   the same deck with a LEG listed first: one crossing tent, two
            below-family tents. Same current space, different rows.
  twoS      `two_node_deck` at S m, ports at 4.5 and 2.5. bspline's below
            grazing floor refuses 12 m at x1 and 3 m at x4; 2 m serves
            to x4.

Every edge refined, soil A. Two ports on knots at every rung (tripod:
monopole at 4.0 and leg 0 at 1.0; two-node: each rod's above wire at 4.5).
Reported: 2-port non-reciprocity (reference-free) and razor's Z11 against
bspline d2's.

Usage: probe11_multi.py <tripodA|tripodB|two12|two2> <rungs...>
"""

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

import numpy as np
from test_crossing_serve_524 import A_WIRE, SOIL_A, WL7, two_node_deck

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


def counted(self, geom, tents, *a, **kw):
    CALLS[0] = len(tents)
    return _orig(self, geom, tents, *a, **kw)


RazorSolver._crossing_node_charges = counted


def tripod(monopole_first):
    s = 2.0 / np.sqrt(2.0)
    legs = []
    for ang in (0.0, 2 * np.pi / 3, 4 * np.pi / 3):
        legs.append(np.array([(s * np.cos(ang), s * np.sin(ang), -s), (0.0, 0.0, 0.0)]))
    mono = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)])
    # razor pairs every member against the FIRST wire at the node (wire
    # order), so the monopole first makes all three tents crossing ones
    if monopole_first:
        wires, im, il = [mono, *legs], 0, [1, 2, 3]
    else:
        wires, im, il = [*legs, mono], 3, [0, 1, 2]
    npe = [[20] if i == im else [8] for i in range(4)]
    junction = [(i, "end") for i in il] + [(im, "start")]
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[junction],
        feeds=[(im, 4.0, 1 + 0j), (il[0], 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def deck(kind, m):
    if kind.startswith("tripod"):
        d = tripod(kind == "tripodA")
    else:
        d = two_node_deck(float(kind[3:]))
        # different arclengths on the two rods, so Y12 = Y21 is not a symmetry
        d["feeds"] = [(1, 4.5, 1 + 0j), (3, 2.5, 1 + 0j)]
    d["n_per_edge_per_wire"] = [[n * m for n in e] for e in d["n_per_edge_per_wire"]]
    return d


kind = sys.argv[1]
prev = None
for m in [int(x) for x in sys.argv[2:]]:
    d = deck(kind, m)
    Yb = np.asarray(BSplineSolver(**d).compute_y_matrix())
    d.pop("junctions")
    t0 = time.perf_counter()
    Y = np.asarray(RazorSolver(**d, nec5_quadrature=True).compute_y_matrix())
    dt = time.perf_counter() - t0
    Z, Zb = np.linalg.inv(Y), np.linalg.inv(Yb)
    nr = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
    ratio = "" if prev is None else f" ({prev / nr:4.2f}x)"
    prev = nr
    g = Z[0, 0] - Zb[0, 0]
    print(
        f"{kind} x{m:<2d} tents {CALLS[0]} nonrec {nr:.3e}{ratio}  Z11 "
        f"{Z[0, 0]:.3f} (bs {Zb[0, 0]:.3f}, gap {g.real:+.3f}{g.imag:+.3f}j)  "
        f"Z22 {Z[1, 1]:.3f} (bs {Zb[1, 1]:.3f})  {dt:.1f}s",
        flush=True,
    )
