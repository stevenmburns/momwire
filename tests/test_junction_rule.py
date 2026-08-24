"""The shared "which wire ends are one node" rule (momwire#590 step 1).

`RazorSolver` spelled this rule and `HarringtonSolver` walked it a second time
by hand, with a comment saying two solvers disagreeing about the same deck's
CONNECTIVITY is a worse failure than either answer being wrong. Now there is
one spelling. These gates pin the properties that made the duplicate worth
keeping in the first place — above all the non-transitivity, which reads like
a bug until you know it is deliberate and is exactly what a well-meaning
rewrite to union-find would destroy.
"""

import numpy as np

from momwire._junction_rule import JUNCTION_TOL, coincident_groups, grouped
from momwire.harrington import HarringtonSolver
from momwire.razor import RazorSolver


def _pt(x):
    return np.array([float(x), 0.0, 0.0])


# ----------------------------------------------------------------------
# The rule
# ----------------------------------------------------------------------


def test_the_rule_is_not_transitive():
    """A~B and B~C but A!~C gives {A, B} and a separate {C}.

    Placed at 0, 0.6*tol, 1.2*tol: each neighbour pair is inside tolerance and
    the outer pair is not. First-match-against-representatives makes C its own
    group because it is compared against A (the representative), never against
    B. A union-find over the same pairs would merge all three -- that is the
    substantive difference, and it changes the deck's connectivity.
    """
    pts = [_pt(0.0), _pt(0.6 * JUNCTION_TOL), _pt(1.2 * JUNCTION_TOL)]
    assert coincident_groups(pts) == [0, 0, 2]
    assert grouped(["A", "B", "C"], pts) == [["A", "B"], ["C"]]


def test_representatives_are_the_earliest_member():
    """`rep[i] <= i`, and a representative represents itself. Callers rely on
    both: bucketing by `rep` reproduces group-creation order only because a
    representative `r` is first seen at `i == r`."""
    pts = [_pt(0.0), _pt(5.0), _pt(0.0), _pt(5.0), _pt(9.0)]
    rep = coincident_groups(pts)
    assert rep == [0, 1, 0, 1, 4]
    assert all(r <= i for i, r in enumerate(rep))
    assert all(rep[r] == r for r in rep)


def test_grouping_preserves_creation_order():
    labels = ["a", "b", "c", "d"]
    pts = [_pt(2.0), _pt(0.0), _pt(2.0), _pt(0.0)]
    assert grouped(labels, pts) == [["a", "c"], ["b", "d"]]


def test_the_tolerance_boundary_is_inclusive():
    """`<= tol` joins, strictly greater does not -- the boundary a refusal
    window (harrington's `_NEAR_COINCIDENT_TOL`) is measured against."""
    assert coincident_groups([_pt(0.0), _pt(JUNCTION_TOL)]) == [0, 0]
    assert coincident_groups([_pt(0.0), _pt(1.0000001 * JUNCTION_TOL)]) == [0, 1]


def test_a_lone_point_is_its_own_group():
    assert coincident_groups([_pt(0.0)]) == [0]
    assert coincident_groups([]) == []


# ----------------------------------------------------------------------
# The reason it is shared
# ----------------------------------------------------------------------


def _tee_wires():
    """Three arms off one node -- a K=3 junction both solvers must see."""
    return [
        np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [-0.2, 0.35, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [-0.2, -0.35, 0.0]]),
    ]


def test_razor_and_harrington_agree_on_connectivity():
    """The property the duplicated copy existed to protect, now structural.

    Razor reports junction GROUPS and harrington merges knot NODE IDS, so the
    two are compared through what they mean: how many distinct nodes the three
    coincident ends collapse to. Both must say one.
    """
    wires = _tee_wires()
    razor = RazorSolver(wires=wires, n_per_edge_per_wire=[[6]] * 3, wavelength=1.0)
    groups = razor._find_junctions()
    at_origin = [
        g
        for g in groups
        if all(
            np.linalg.norm(wires[w][0 if e == "start" else -1]) <= JUNCTION_TOL
            for w, e in g["ends"]
        )
    ]
    assert len(at_origin) == 1
    assert len(at_origin[0]["ends"]) == 3

    # Harrington reaches the same verdict through its own merge: the three
    # arms' first segments must all report the SAME left-hand cell, which is
    # only true if their start knots collapsed to one node.
    harr = HarringtonSolver(wires=wires, n_per_edge_per_wire=[[6]] * 3, wavelength=1.0)
    geom = harr._build_geometry()
    left_node, _right, _pieces, _cop = harr._node_map(geom)
    starts = geom["seg_offsets"][:3]
    assert len(set(left_node[s] for s in starts)) == 1

    # And the claim has teeth: pulled apart, both solvers must instead see
    # three separate nodes. The gap has to clear harrington's
    # `_NEAR_COINCIDENT_TOL` = 1e-6 as well as the grouping tolerance -- ends
    # inside that window are REFUSED rather than disconnected, which is the
    # same defect class momwire#590 is about, already guarded here.
    apart = [w.copy() for w in wires]
    for i, w in enumerate(apart[1:], start=1):
        w[0, 2] = i * 0.01
    razor_apart = RazorSolver(
        wires=apart, n_per_edge_per_wire=[[6]] * 3, wavelength=1.0
    )
    assert razor_apart._find_junctions() == []
    harr_apart = HarringtonSolver(
        wires=apart, n_per_edge_per_wire=[[6]] * 3, wavelength=1.0
    )
    geom_apart = harr_apart._build_geometry()
    left_apart, _r, _p, _c = harr_apart._node_map(geom_apart)
    starts_apart = geom_apart["seg_offsets"][:3]
    assert len(set(left_apart[s] for s in starts_apart)) == 3


def test_a_closed_loop_groups_its_own_two_ends():
    """A wire whose start and end coincide is a junction with itself -- the
    case razor's docstring calls out, and the one an ends-of-different-wires
    reading of the rule would miss."""
    loop = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rep = coincident_groups([loop[0], loop[-1]])
    assert rep == [0, 0]
