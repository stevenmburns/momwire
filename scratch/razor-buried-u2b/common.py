"""Shared decks and helpers for the momwire#1149 U2b probes."""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tests"))
warnings.simplefilter("ignore")

import momwire  # noqa: E402

assert str(ROOT) in momwire.__file__, momwire.__file__

from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import (  # noqa: E402
    A_WIRE,
    SOIL_A,
    WL7,
    crossing_deck,
    hub_deck,
    two_node_deck,
)
from test_razor_detached_1149 import detached  # noqa: E402

__all__ = [
    "A_WIRE",
    "SOIL_A",
    "WL7",
    "BSplineSolver",
    "RazorSolver",
    "crossing_deck",
    "detached",
    "detached_hub",
    "hub_deck",
    "refined",
    "two_node_deck",
    "razor",
    "y2",
]


def refined(deck, m):
    deck["n_per_edge_per_wire"] = [
        [n * m for n in e] for e in deck["n_per_edge_per_wire"]
    ]
    return deck


def detached_hub(m=1, radii=(A_WIRE,) * 4, gap=0.1, ground=True, eps=SOIL_A):
    """hub_deck(2) with the node pulled apart: two radials (different
    directions) at 0.15 m depth into a buried hub, a rise from the hub up to
    -gap, and the 10 m monopole from +gap up. Radii [rad0, rad1, rise, mono].
    Ports on the monopole (4.0 from its top) and on radial 0 (1.0)."""
    d = hub_deck(n_radials=2)
    w = d["wires"]
    assert 0.0 < gap < 0.15
    w[2] = np.array([(0.0, 0.0, -0.15), (0.0, 0.0, -gap)])
    w[3] = np.array([(0.0, 0.0, 10.0), (0.0, 0.0, gap)])
    d["wires"] = w
    d["junctions"] = [[(0, "end"), (1, "end"), (2, "start")]]
    d["wire_radius"] = list(radii)
    d["feeds"] = [(3, 4.0, 1 + 0j), (0, 1.0, 1 + 0j)]
    if not ground:
        for k in ("ground_z", "ground_eps", "ground_model"):
            d.pop(k)
    else:
        d["ground_eps"] = eps
    return refined(d, m)


def razor(deck, **kw):
    d = {k: v for k, v in deck.items() if k != "junctions"}
    return RazorSolver(**d, nec5_quadrature=True, **kw)


def y2(s):
    Y = np.asarray(s.compute_y_matrix())
    return Y, np.linalg.inv(Y), abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
