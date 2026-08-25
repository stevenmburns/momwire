"""The one-segment wire, served and refused (momwire#608).

``RazorSolver`` used to refuse EVERY wire with a single segment, on two
grounds.  One was right and one was wrong, and they have to be separated
because the wrong half cost the EZNEC corpus five decks:

* *"a one-segment wire carries no tent"* — true only of a wire junctioned at
  NEITHER end.  A junction contributes a tent per end that meets something,
  and those tents live on the segments either side.
* *"if it were junctioned at both ends its two junction tents would overlap
  on that one segment"* — wrong, and not by a subtlety.  Two tents sharing a
  segment is what every INTERIOR segment of every wire already is: they are
  that segment's two Lagrange bases.  There is nothing degenerate about it.

So the gates here are :mod:`test_razor_junctions`'s split identity again, run
at the one place it was assumed to break.  A wire split at a knot is the same
linear system with one basis re-labelled; split it at its FIRST knot and the
piece that falls out is a one-segment wire junctioned at one end, split it at
two adjacent knots and the middle piece is one junctioned at both.  Neither
is a new object, so both must reproduce the unsplit wire to solver precision
— pinned at 1e-9 relative, ~7 orders tighter than the discretization error
they ride on, and measured at ~1e-14.

What stays refused is the wire junctioned at neither end.  It carries no
basis at all, so it holds no current and scatters nothing, and a solve
including it is bit-identical to one that omits it — which is measured below
rather than asserted.  That is NOT a razor quirk: the licensed NEC-5 binary
counts the same wire as an element and gives it no unknown either, printing
the same impedance with and without it (scratch/608-probes/, run against
LLNL-CODE-746721).  Reproducing it silently is the one thing this class will
not do, because the caller declared a scatterer and would get a
scatterer-free answer with nothing said.
"""

import numpy as np
import pytest

from momwire import BSplineSolver, RazorSolver

# ByDipole1 in free space, the same wire test_razor_junctions.py splits.
BD1_LEN = 10.18946
BD1_RAD = 0.0010262
BD1_WL = 299792458.0 / 14.0e6
BD1_KW = dict(wire_radius=BD1_RAD, wavelength=BD1_WL)
BD1_N = 20
D = BD1_LEN / BD1_N


def _point(arc):
    return np.array([0.0, arc, 0.0])


def _rel(z_a, z_b):
    return abs(z_a - z_b) / abs(z_b)


def _unsplit():
    return RazorSolver(
        wires=[np.array([_point(0.0), _point(BD1_LEN)])], nsegs=BD1_N, **BD1_KW
    ).compute_impedance()


# --------------------------------------------------------------------------
# 1. served: the split identity, at the two knots that make a one-segment piece
# --------------------------------------------------------------------------
def test_one_segment_piece_junctioned_at_one_end_matches_the_unsplit_wire():
    """Split at the FIRST knot: piece 0 is a single segment, free at the tip
    and junctioned at its other end.  It carries exactly one basis — the
    junction tent — which is what the wire's first interior knot carried
    before the split, so the system is the same 19 unknowns."""
    z_ref, c_ref = _unsplit()
    z, coeffs = RazorSolver(
        wires=[
            np.array([_point(0.0), _point(D)]),
            np.array([_point(D), _point(BD1_LEN)]),
        ],
        n_per_edge_per_wire=[[1], [BD1_N - 1]],
        feeds=[(1, BD1_LEN / 2 - D, 1.0 + 0j)],
        **BD1_KW,
    ).compute_impedance()
    assert _rel(z, z_ref) < 1e-9, f"{z} vs {z_ref}"
    assert coeffs.shape == c_ref.shape == (19,)


def test_one_segment_piece_junctioned_at_both_ends_matches_the_unsplit_wire():
    """Split at knots 8 AND 9, so the middle piece is one segment junctioned
    at both ends — the case the old refusal called degenerate.  Its two
    junction tents are the two bases the segment carried as an interior
    segment of the whole wire, and the count is unchanged."""
    z_ref, c_ref = _unsplit()
    z, coeffs = RazorSolver(
        wires=[
            np.array([_point(0.0), _point(8 * D)]),
            np.array([_point(8 * D), _point(9 * D)]),
            np.array([_point(9 * D), _point(BD1_LEN)]),
        ],
        n_per_edge_per_wire=[[8], [1], [BD1_N - 9]],
        feeds=[(2, BD1_LEN / 2 - 9 * D, 1.0 + 0j)],
        **BD1_KW,
    ).compute_impedance()
    assert _rel(z, z_ref) < 1e-9, f"{z} vs {z_ref}"
    assert coeffs.shape == c_ref.shape == (19,)


def test_the_one_segment_piece_may_be_spelled_backwards():
    """THE sign test for this shape.  A one-segment piece has no interior
    tent to carry a direction, so its arc direction shows up ONLY in its two
    junction tents' wings — reverse the spelling and both wings swap which
    side is the reference.  Row and column signs must cancel exactly."""
    z_ref, _ = _unsplit()
    z, _ = RazorSolver(
        wires=[
            np.array([_point(0.0), _point(8 * D)]),
            np.array([_point(9 * D), _point(8 * D)]),  # reversed
            np.array([_point(9 * D), _point(BD1_LEN)]),
        ],
        n_per_edge_per_wire=[[8], [1], [BD1_N - 9]],
        feeds=[(2, BD1_LEN / 2 - 9 * D, 1.0 + 0j)],
        **BD1_KW,
    ).compute_impedance()
    assert _rel(z, z_ref) < 1e-9, f"{z} vs {z_ref}"


def test_a_one_segment_piece_can_be_fed_at_its_junction():
    """The feed lands on a junction knot of the one-segment piece itself.
    A K=2 junction basis is a through-current unknown wherever it sits, so
    the delta gap drives it exactly as it drives an interior knot — this is
    the split-wire feed, on the shape that used to be refused."""
    z_ref, _ = _unsplit()
    z, _ = RazorSolver(
        wires=[
            np.array([_point(0.0), _point(10 * D)]),
            np.array([_point(10 * D), _point(11 * D)]),
            np.array([_point(11 * D), _point(BD1_LEN)]),
        ],
        n_per_edge_per_wire=[[10], [1], [BD1_N - 11]],
        feeds=[(1, 0.0, 1.0 + 0j)],  # the junction at the wire's midpoint
        **BD1_KW,
    ).compute_impedance()
    assert _rel(z, z_ref) < 1e-9, f"{z} vs {z_ref}"


def test_every_piece_may_be_one_segment_at_once():
    """The extreme spelling: a two-segment dipole written as two one-segment
    wires meeting in the middle.  There is no interior knot anywhere in the
    model and exactly one junction tent, which is the whole basis — the case
    the old ``n_interior == 0`` guard in ``_build_geometry`` refused by
    counting interior knots and forgetting the junction tents."""
    two_seg = RazorSolver(
        wires=[np.array([_point(0.0), _point(BD1_LEN)])], nsegs=2, **BD1_KW
    )
    z_ref, c_ref = two_seg.compute_impedance()
    solver = RazorSolver(
        wires=[
            np.array([_point(0.0), _point(BD1_LEN / 2)]),
            np.array([_point(BD1_LEN / 2), _point(BD1_LEN)]),
        ],
        n_per_edge_per_wire=[[1], [1]],
        feeds=[(0, BD1_LEN / 2, 1.0 + 0j)],
        **BD1_KW,
    )
    z, coeffs = solver.compute_impedance()
    assert _rel(z, z_ref) < 1e-9, f"{z} vs {z_ref}"
    assert coeffs.shape == c_ref.shape == (1,)
    assert solver._build_geometry()["n_basis_interior"] == 0


def test_a_closed_triangle_of_one_segment_wires_solves():
    """Three one-segment wires meeting corner to corner: no interior knot in
    the model, three junction tents, a perfectly ordinary small loop.  Gated
    against the same triangle meshed at two segments a side, which is a
    different discretization and so agrees only to a mesh term — the point
    here is that the coarse one is SOLVED and not refused."""
    side = 1.0
    v = [
        np.array([0.0, 0.0, 0.0]),
        np.array([side, 0.0, 0.0]),
        np.array([side / 2, side * 0.8660254, 0.0]),
    ]
    kw = dict(wire_radius=1e-3, wavelength=40.0)
    wires = [np.array([v[0], v[1]]), np.array([v[1], v[2]]), np.array([v[2], v[0]])]
    solver = RazorSolver(
        wires=wires,
        n_per_edge_per_wire=[[1], [1], [1]],
        feeds=[(0, 0.0, 1.0 + 0j)],
        **kw,
    )
    z, coeffs = solver.compute_impedance()
    assert coeffs.shape == (3,)
    assert solver._build_geometry()["n_basis_interior"] == 0
    z_fine, _ = RazorSolver(
        wires=wires,
        n_per_edge_per_wire=[[2], [2], [2]],
        feeds=[(0, 0.0, 1.0 + 0j)],
        **kw,
    ).compute_impedance()
    # A small loop's reactance is what carries the mesh: pinned loosely,
    # because a 3-unknown model and a 6-unknown one are not the same answer.
    assert _rel(z, z_fine) < 0.25, f"{z} vs {z_fine}"


# --------------------------------------------------------------------------
# 2. refused: the wire junctioned at neither end
# --------------------------------------------------------------------------
def _floater():
    """A one-segment wire alongside the dipole, touching nothing."""
    return np.array(
        [
            [0.5, BD1_LEN / 2 - D / 2, 0.0],
            [0.5, BD1_LEN / 2 + D / 2, 0.0],
        ]
    )


def test_a_one_segment_wire_junctioned_at_neither_end_is_refused():
    with pytest.raises(ValueError, match="junctioned at neither end"):
        RazorSolver(
            wires=[np.array([_point(0.0), _point(BD1_LEN)]), _floater()],
            n_per_edge_per_wire=[[BD1_N], [1]],
            feeds=[(0, BD1_LEN / 2, 1.0 + 0j)],
            **BD1_KW,
        )


def test_the_refusal_names_the_wire_and_not_just_the_rule():
    """Wire 1, not "a wire": a 44-piece deck with one bad polyline in it is
    unfixable from a message that does not say which."""
    with pytest.raises(ValueError, match=r"^wire 1: "):
        RazorSolver(
            wires=[np.array([_point(0.0), _point(BD1_LEN)]), _floater()],
            n_per_edge_per_wire=[[BD1_N], [1]],
            feeds=[(0, BD1_LEN / 2, 1.0 + 0j)],
            **BD1_KW,
        )


def test_what_the_refusal_is_protecting_against_is_real():
    """The claim in the message, measured: the inert wire changes NOTHING.

    Neutering the guard and solving both ways gives bit-identical impedance
    and the same number of unknowns, while the same floater at two segments
    carries a basis and moves the answer.  Run through the geometry rather
    than the constructor so the guard itself stays in force.
    """
    lone = RazorSolver(
        wires=[np.array([_point(0.0), _point(BD1_LEN)])],
        n_per_edge_per_wire=[[BD1_N]],
        feeds=[(0, BD1_LEN / 2, 1.0 + 0j)],
        **BD1_KW,
    )
    z_lone, c_lone = lone.compute_impedance()

    with_floater = RazorSolver.__new__(RazorSolver)
    with_floater.__dict__.update(lone.__dict__)
    with_floater.wires_polylines = [*lone.wires_polylines, _floater()]
    with_floater.n_per_edge_per_wire = [*lone.n_per_edge_per_wire, [1]]
    with_floater._cached_geometry = None
    z_with, c_with = with_floater.compute_impedance()
    assert c_with.shape == c_lone.shape
    assert z_with == z_lone

    two_seg = RazorSolver(
        wires=[np.array([_point(0.0), _point(BD1_LEN)]), _floater()],
        n_per_edge_per_wire=[[BD1_N], [2]],
        feeds=[(0, BD1_LEN / 2, 1.0 + 0j)],
        **BD1_KW,
    )
    z_two, c_two = two_seg.compute_impedance()
    assert c_two.shape == (c_lone.shape[0] + 1,)
    assert z_two != z_lone


def test_the_sibling_hosts_what_razor_refuses():
    """``BSplineSolver`` is not being asked to change: a degree-2 spline over
    one segment has ``N + d = 3`` bases, of which the free-end pair is
    dropped and one survives, so the floater scatters there.  Recorded so
    that the divergence is a decision on the record and not a surprise —
    razor refuses this wire because ITS basis cannot host it, not because the
    model is meaningless."""
    kw = dict(feeds=[(0, BD1_LEN / 2, 1.0 + 0j)], **BD1_KW)
    _z_lone, c_lone = BSplineSolver(
        wires=[np.array([_point(0.0), _point(BD1_LEN)])],
        n_per_edge_per_wire=[[BD1_N]],
        **kw,
    ).compute_impedance()
    z_with, c_with = BSplineSolver(
        wires=[np.array([_point(0.0), _point(BD1_LEN)]), _floater()],
        n_per_edge_per_wire=[[BD1_N], [1]],
        **kw,
    ).compute_impedance()
    assert c_with.shape == (c_lone.shape[0] + 1,)
    assert z_with != _z_lone
