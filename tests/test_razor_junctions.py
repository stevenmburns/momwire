"""RazorSolver junctions, loops and the Y matrix (momwire#309, unit 2).

The junction tent is defined so that a wire split at a knot is the SAME
linear system as the unsplit wire — same knots, same segments, same rows,
one basis re-labelled. That makes the split identity exact rather than
approximate, and exactness is what makes it a sharp test: every sign in
the junction's two-sided bookkeeping (which way the current runs on each
side relative to that wire's own arc direction, which way the testing
path is traversed, which sign the charge doublet carries) shows up as a
gross error in Z if it is wrong, not as a small mesh term. The identities
below are therefore pinned at 1e-9 RELATIVE, ~5 orders tighter than the
discretization error they are riding on; the measured agreement is
~1e-14, i.e. the linear solve's own conditioning.

The last two tests are about the Y matrix, and one of them documents
physics rather than guarding a bug: razor-blade testing is not Galerkin,
so [Y] is not reciprocal at a finite mesh.
"""

import numpy as np
import pytest

from momwire import RazorSolver

# ByDipole1 in free space: the antennaknobs validation wire, 14 MHz.
BD1_LEN = 10.18946
BD1_RAD = 0.0010262
BD1_WL = 299792458.0 / 14.0e6
BD1_KW = dict(wire_radius=BD1_RAD, wavelength=BD1_WL)
BD1_N = 20

HALF = 0.962 * 22 / 4


def _rel(z_a, z_b):
    return abs(z_a - z_b) / abs(z_b)


def _point(arc):
    """A point on the ByDipole1 axis at the given arc length."""
    return np.array([0.0, arc, 0.0])


def _bd1_single():
    z, _ = RazorSolver(
        wires=[np.array([_point(0.0), _point(BD1_LEN)])], nsegs=BD1_N, **BD1_KW
    ).compute_impedance()
    return z


# --------------------------------------------------------------------------
# 1. the split identity: a junction tent IS the interior tent it replaces
# --------------------------------------------------------------------------
def test_straight_split_matches_single_wire():
    """Split at the 8th of 20 knots, feed still at the wire's centre (which
    lands on the LONG piece, so the feed row is an ordinary interior row and
    only the junction row is new)."""
    d = BD1_LEN / BD1_N
    cut = 8 * d
    two = RazorSolver(
        wires=[
            np.array([_point(0.0), _point(cut)]),
            np.array([_point(cut), _point(BD1_LEN)]),
        ],
        n_per_edge_per_wire=[[8], [12]],
        feeds=[(1, BD1_LEN / 2 - cut, 1.0 + 0j)],
        **BD1_KW,
    )
    z_two, coeffs = two.compute_impedance()
    z_one = _bd1_single()
    assert _rel(z_two, z_one) < 1e-9, f"{z_two} vs {z_one}"
    # 7 + 11 interior tents + the one junction tent = the single wire's 19.
    assert coeffs.shape == (19,)


def test_junction_fed_split_matches_single_wire():
    """The other half of the identity: the feed itself lands ON the junction.
    A K=2 junction basis is a through-current unknown, so the delta gap drives
    it exactly the way it drives an interior knot — this is the ordinary
    split-wire feed, and it must read the same Z as the unsplit wire."""
    mid = BD1_LEN / 2
    two = RazorSolver(
        wires=[
            np.array([_point(0.0), _point(mid)]),
            np.array([_point(mid), _point(BD1_LEN)]),
        ],
        n_per_edge_per_wire=[[10], [10]],
        feeds=[(0, mid, 1.0 + 0j)],
        **BD1_KW,
    )
    z_two, coeffs = two.compute_impedance()
    z_one = _bd1_single()
    assert _rel(z_two, z_one) < 1e-9, f"{z_two} vs {z_one}"
    # The feed snapped to the junction basis, which is the last one.
    assert two._feed_basis_indices(two._build_geometry()) == [18]
    assert abs(1.0 / coeffs[18] - z_two) < 1e-12


def test_reversed_split_matches_single_wire():
    """THE sign test. The second piece is spelled tip-to-junction instead of
    junction-to-tip, so its arc direction is flipped: its junction wing now
    joins by its END, every one of its interior tents carries the opposite
    physical direction, and its testing paths are traversed backwards. Row and
    column signs must cancel exactly — get one of the two wrong and Z moves by
    order 100%, not by a mesh term."""
    d = BD1_LEN / BD1_N
    cut = 8 * d
    z_rev, _ = RazorSolver(
        wires=[
            np.array([_point(0.0), _point(cut)]),
            np.array([_point(BD1_LEN), _point(cut)]),  # reversed
        ],
        n_per_edge_per_wire=[[8], [12]],
        feeds=[(1, BD1_LEN - BD1_LEN / 2, 1.0 + 0j)],
        **BD1_KW,
    ).compute_impedance()
    z_one = _bd1_single()
    assert _rel(z_rev, z_one) < 1e-9, f"{z_rev} vs {z_one}"


def test_bent_split_matches_single_wire():
    """The unit-1 inverted-V, respelled as two wires meeting at the apex. The
    junction now bends, so the two wings carry different tangents — the same
    thing a kinked single wire does at an interior knot, which is why the
    identity still has to be exact. The feed stays on the arm knot nearest the
    apex (a knot that exists identically in both spellings), not on the apex
    itself."""
    c = 5.0 / np.sqrt(2.0)  # 5 m arms, 90 deg included angle
    left, apex, right = (
        np.array([-c, 0.0, -c]),
        np.array([0.0, 0.0, 0.0]),
        np.array([c, 0.0, -c]),
    )
    kw = dict(wire_radius=1e-3, wavelength=21.4, nsegs=20)
    feed_arc = 5.0 - 5.0 / 20

    z_one, _ = RazorSolver(
        wires=[np.array([left, apex, right])], feeds=[(0, feed_arc, 1.0 + 0j)], **kw
    ).compute_impedance()
    z_two, _ = RazorSolver(
        wires=[np.array([left, apex]), np.array([apex, right])],
        feeds=[(0, feed_arc, 1.0 + 0j)],
        **kw,
    ).compute_impedance()
    assert _rel(z_two, z_one) < 1e-9, f"{z_two} vs {z_one}"


# --------------------------------------------------------------------------
# 2. closed loops — the wire that junctions with itself
# --------------------------------------------------------------------------
SIDE = 6.05  # square loop, perimeter 24.2 m = 1.1 wavelength at 22 m
LOOP_KW = dict(wire_radius=5e-3, wavelength=22.0, nsegs=10)
CORNERS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([SIDE, 0.0, 0.0]),
    np.array([SIDE, SIDE, 0.0]),
    np.array([0.0, SIDE, 0.0]),
]


def test_closed_loop_matches_two_half_loops():
    """A one-wire loop (its start IS its end) against the same loop cut into
    two half-loops joined at two junctions. Exercises loop closure, a wire
    junctioned at BOTH ends, and two separate junctions between the same pair
    of wires. The feed is an interior knot of the first side in both."""
    loop = np.array(CORNERS + [CORNERS[0]])
    z_loop, c_loop = RazorSolver(
        wires=[loop], feeds=[(0, SIDE / 2, 1.0 + 0j)], **LOOP_KW
    ).compute_impedance()
    z_halves, c_halves = RazorSolver(
        wires=[
            np.array(CORNERS[0:3]),
            np.array([CORNERS[2], CORNERS[3], CORNERS[0]]),
        ],
        feeds=[(0, SIDE / 2, 1.0 + 0j)],
        **LOOP_KW,
    ).compute_impedance()
    assert _rel(z_halves, z_loop) < 1e-9, f"{z_halves} vs {z_loop}"
    # A closed loop has as many knots as segments — no free end anywhere, so
    # the seam knot carries a tent like every other knot.
    assert c_loop.shape == (40,)
    assert c_halves.shape == (40,)


def test_closed_loop_current_is_continuous_across_the_seam():
    """The seam is not a special place on the loop: the junction tent's
    coefficient must sit on the smooth current distribution its two
    neighbouring interior knots define, to within the curvature of that
    distribution over one segment pair (measured 1.3% — the seam sits on a
    corner of the square, so there is real curvature to miss). A mis-signed
    seam basis reads 199% off here, a missing one reads 100% off."""
    loop = np.array(CORNERS + [CORNERS[0]])
    solver = RazorSolver(wires=[loop], feeds=[(0, SIDE / 2, 1.0 + 0j)], **LOOP_KW)
    _z, coeffs = solver.compute_impedance()
    # basis order: interior knots 1..39 in arc order, then the seam tent.
    # Side A of the seam is the loop's START, so its through-current runs
    # against the loop's arc direction: it is MINUS the arc-directed current.
    seam = -coeffs[-1]
    neighbours = 0.5 * (coeffs[0] + coeffs[-2])
    assert abs(seam - neighbours) < 0.03 * abs(seam), f"{seam} vs {neighbours}"


# --------------------------------------------------------------------------
# 3. K > 2: the three-way junction
# --------------------------------------------------------------------------
ARM = 5.5
STAR_KW = dict(wire_radius=1e-3, wavelength=22.0, nsegs=10)
STAR = [
    np.array(
        [[0.0, 0.0, 0.0], [ARM * np.cos(a), ARM * np.sin(a), 0.0]],
    )
    for a in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)
]


def _currents_into_junction(geom, coeffs, junction):
    """Current arriving at the junction along each of its wire ends.

    Rebuilt from the wing arrays rather than from the junction bases'
    coefficients, so a mis-set wing sign shows up as a KCL violation. Every
    tent is 1 at its own knot, so the current at the junction end of a
    terminal segment is just the sum of the coefficients of the wings that
    end there, each times that wing's direction sign; converting to "into the
    junction" then flips the wires that join by their start.
    """
    wing_seg, wing_rise, wing_sigma = (
        geom["wing_seg"],
        geom["wing_rise"],
        geom["wing_sigma"],
    )
    out = []
    for w, kind in junction["ends"]:
        at_end = kind == "end"
        seg = geom["seg_offsets"][w + 1] - 1 if at_end else geom["seg_offsets"][w]
        touching = (wing_seg == seg) & (wing_rise == at_end)
        along_arc = float(at_end) * 2.0 - 1.0
        out.append(along_arc * (coeffs[:, None] * wing_sigma * touching).sum())
    return np.array(out)


def test_three_way_junction_solves_and_conserves_current():
    """Three arms at 120 deg. K=3 ends means 2 through-current bases, both
    referred to the same side A, so KCL at the node is structural here: the
    current arriving on A is the sum of the two leaving on B1 and B2 by
    construction, not by a constraint row. What the check below actually
    verifies is that the construction is COHERENT — the currents rebuilt from
    the wing signs at each of the three terminal segments really do cancel —
    and, by symmetry of the geometry, that the two branch currents come out
    equal."""
    solver = RazorSolver(wires=STAR, feeds=[(0, ARM / 10, 1.0 + 0j)], **STAR_KW)
    z, coeffs = solver.compute_impedance()
    geom = solver._build_geometry()
    (junction,) = geom["junctions"]
    assert len(junction["ends"]) == 3
    assert len(junction["bases"]) == 2

    scale = np.abs(coeffs).max()
    into = _currents_into_junction(geom, coeffs, junction)
    assert abs(into.sum()) < 1e-12 * scale, into
    # Side A carries what the two branches take away.
    branches = np.array([coeffs[b] for b in junction["bases"]])
    assert abs(into[0] - branches.sum()) < 1e-12 * scale
    # The three arms are identical and the drive is on arm 0, so the two
    # branch currents are the same to solve precision.
    assert abs(branches[0] - branches[1]) < 1e-9 * scale

    assert np.isfinite(z)
    assert z.real > 0.0, z


def test_feed_at_a_three_way_junction_is_refused():
    """A delta gap at a K>=3 node has no defined polarity — it would have to
    name which pair of branches it drives. Refused rather than silently
    driving the first pair."""
    solver = RazorSolver(wires=STAR, feeds=[(0, 0.0, 1.0 + 0j)], **STAR_KW)
    with pytest.raises(NotImplementedError, match="3 wire ends"):
        solver.compute_impedance()


# --------------------------------------------------------------------------
# 4. the Y matrix
# --------------------------------------------------------------------------
def test_y_matrix_is_the_inverse_of_the_single_port_impedance():
    """One port: Y[0,0] is by definition 1/Z at the same feed, with the same
    solver conventions on both sides."""
    kw = dict(wires=[np.array([_point(0.0), _point(BD1_LEN)])], nsegs=BD1_N, **BD1_KW)
    z, _ = RazorSolver(**kw).compute_impedance()
    y = RazorSolver(**kw).compute_y_matrix()
    assert y.shape == (1, 1)
    assert abs(y[0, 0] - 1.0 / z) < 1e-12 * abs(y[0, 0])


def _pair(n, half_1=HALF):
    wl = 22.0
    return RazorSolver(
        wires=[
            np.array([[0.0, 0.0, -HALF], [0.0, 0.0, HALF]]),
            np.array([[0.1 * wl, 0.0, -half_1], [0.1 * wl, 0.0, half_1]]),
        ],
        nsegs=n,
        wavelength=wl,
        feeds=[(0, None, 1.0 + 0j), (1, None, 1.0 + 0j)],
    )


def _asym(n, half_1):
    y = _pair(n, half_1).compute_y_matrix()
    return abs(y[0, 1] - y[1, 0]) / abs(y[0, 1])


def test_y_matrix_reciprocity_asymmetry_is_the_razor_fingerprint():
    """THE fingerprint distinguishing razor-blade testing from Galerkin.

    Galerkin tests the E-field boundary condition with the basis itself, so
    its matrix is symmetric to machine precision and [Y] is reciprocal
    exactly, at any mesh. Razor-blade testing weights it with a PATH INTEGRAL
    between centroids instead — a different functional from the basis — so the
    matrix is genuinely non-symmetric and Y12 != Y21 at a finite mesh.
    Reciprocity is recovered only in the limit, and this is the same
    first-order-testing story as the O(1/N) impedance walk that momwire#309
    exists to reproduce. The asymmetry is a documented property of the
    formulation, not a defect to fix.

    Measured here (unequal parallel dipoles, 0.1 wavelength apart): 1.6e-3 at
    N=10, falling ~4x per mesh doubling.

    The elements are deliberately UNEQUAL. Two IDENTICAL parallel dipoles on
    identical uniform meshes are a degenerate case — every block of Z is then
    symmetric for a geometric reason (each basis and each testing path is even
    about its own knot, so swapping m and n only reverses the separation
    vector) and the asymmetry vanishes exactly, which the companion test below
    pins so that degeneracy can never be mistaken for reciprocity.
    """
    short = 0.8 * HALF
    asym = {n: _asym(n, short) for n in (10, 20, 40)}
    assert asym[10] > 1e-4, asym
    assert asym[20] < asym[10], asym
    assert asym[40] < asym[20], asym
    assert asym[40] < asym[10] / 2, asym


def test_identical_elements_hide_the_asymmetry():
    """The degeneracy named above, pinned as its own fact: make the two
    dipoles identical and the razor matrix comes out symmetric to machine
    precision, so this geometry says nothing about reciprocity either way."""
    assert _asym(20, HALF) < 1e-12
