"""momwire#1569: razor-2p's node gap at a crossing node.

Dan AC6LA's buried-radial verticals put the source at the base of the
vertical, on the node at z = 0 where the vertical meets buried wires. That
node is grounded by geometry but is a crossing, not a short to the plane
(momwire#813), and bspline serves a gap there (#1282). razor's constructor
read the raw grounded flag and refused it as a grounded end.

* hub (the vertical over one buried stub): a two-wire crossing, served; its
  port is the delta gap on the same through tent, so it is the serve layer's
  answer (`feeds` at the node) to the bit. Under refinement razor converges
  onto bspline and onto NEC-5 (scratch/1569-crossing-gap/README.md).
* slant (radials starting at the base): three or more wires, served too
  (Steve, 10-18), after the check in scratch/1569-crossing-gap/README.md:
  the source one segment up converges onto the node source under
  refinement, and razor and bspline reach the same limit.
* a grounded end with nothing buried keeps its refusal, in words a deck user
  can act on.
"""

import warnings

import numpy as np
import pytest

from momwire import BSplineSolver, RazorSolver

BASE = dict(
    wire_radius=0.003,
    wavelength=299.792458 / 14.0,
    ground_z=0.0,
    ground_eps=(13.0, 0.005),
    ground_model="sommerfeld",
)
HUB = [
    [(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)],
    [(0.0, 0.0, 0.0), (0.0, 0.0, -0.1)],
    [(0.0, 0.0, -0.1), (5.0, 0.0, -0.1)],
    [(0.0, 0.0, -0.1), (-5.0, 0.0, -0.1)],
]
SLANT = [
    [(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)],
    [(0.0, 0.0, 0.0), (5.0, 0.0, -0.1)],
    [(0.0, 0.0, 0.0), (-5.0, 0.0, -0.1)],
]


def _z(cls, wires, counts, **port):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = cls(wires=wires, n_per_edge_per_wire=counts, **BASE, **port)
        return 1.0 / complex(np.asarray(s.compute_y_matrix()).ravel()[0])


def test_the_hub_gap_is_served_and_is_the_delta_gap_on_the_node():
    counts = [[8], [2], [8], [8]]
    z_gap = _z(RazorSolver, HUB, counts, feeds=[], node_gaps=[(0, "start", 1.0)])
    z_feed = _z(RazorSolver, HUB, counts, feeds=[(0, 0.0, 1.0)])
    assert z_gap == pytest.approx(z_feed, rel=1e-12)


def test_the_hub_gap_agrees_with_bspline_at_the_working_mesh():
    counts = [[20], [2], [20], [20]]
    z_r = _z(RazorSolver, HUB, counts, feeds=[], node_gaps=[(0, "start", 1.0)])
    z_b = _z(BSplineSolver, HUB, counts, feeds=[], node_gaps=[(0, "start", 1.0)])
    # 1.6 % at 20 segments a wire, closing first-order under refinement
    assert abs(z_r - z_b) / abs(z_b) < 0.03, (z_r, z_b)


def test_a_three_wire_crossing_gap_is_served_and_agrees_with_bspline():
    counts = [[20], [20], [20]]
    z_r = _z(RazorSolver, SLANT, counts, feeds=[], node_gaps=[(0, "start", 1.0)])
    z_b = _z(BSplineSolver, SLANT, counts, feeds=[], node_gaps=[(0, "start", 1.0)])
    # 4.3 % at 20 segments a wire, 2.2 % at 40, 1.1 % at 80: first order
    assert abs(z_r - z_b) / abs(z_b) < 0.06, (z_r, z_b)


def test_a_grounded_end_keeps_its_refusal_in_deck_words():
    with pytest.raises(ValueError) as exc:
        RazorSolver(
            wires=[
                [(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)],
                [(0.0, 0.0, 0.0), (3.0, 0.0, 2.0)],
            ],
            n_per_edge_per_wire=[[9], [9]],
            feeds=[],
            node_gaps=[(1, "start", 1.0)],
            wire_radius=0.003,
            wavelength=22.0,
            ground_z=0.0,
        )
    msg = str(exc.value)
    assert "meets the ground plane" in msg and "one segment along the wire" in msg
    assert "feeds=" not in msg
