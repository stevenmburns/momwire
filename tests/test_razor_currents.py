"""RazorSolver field readout and swept solves (momwire#309, unit 3).

`currents_at_knots` and `element_currents` (the `_ElementCurrents` mixin) are
the read side of a solve: they turn the raw tent/junction coefficients back
into a per-wire current distribution and, from that, the source terms a
field evaluator wants. `compute_impedance_swept` / `compute_y_matrix_swept`
are the write-side counterpart for a frequency sweep — sharing the
wavenumber-independent parts of the fill (geometry, wing/path stencils, the
closed-form static segment moments and the source-to-observer distance R
itself) across every k instead of rebuilding them per point.

Sign gate. `currents_at_knots` reports each wire's current in THAT WIRE'S OWN
arc direction (see the module docstring's "Field readout and sweeps"
section). `test_split_identity_currents` below is the sharp check on that:
splitting ByDipole1 with the second piece spelled BACKWARDS both moves that
piece's own arc direction (needing a plain arc remap to compare against a
common physical grid) *and*, because the feed sits on that same piece,
flips which physical polarity the nominal +1 V delta-gap drives — the whole
solved distribution comes out the negative of the un-reversed spellings'.
Both effects are real, not bugs, and the test accounts for both explicitly
rather than papering over them with a looser tolerance.
"""

import numpy as np
import pytest

from momwire import CancelToken, RazorSolver, SolveAborted

# ByDipole1 in free space: the antennaknobs validation wire, 14 MHz.
BD1_LEN = 10.18946
BD1_RAD = 0.0010262
BD1_WL = 299792458.0 / 14.0e6
BD1_KW = dict(wire_radius=BD1_RAD, wavelength=BD1_WL)
BD1_N = 20


def _point(arc):
    """A point on the ByDipole1 axis at the given arc length."""
    return np.array([0.0, arc, 0.0])


def _bd1_single(nsegs=BD1_N):
    return RazorSolver(
        wires=[np.array([_point(0.0), _point(BD1_LEN)])], nsegs=nsegs, **BD1_KW
    )


# --------------------------------------------------------------------------
# 1. knot-current basics
# --------------------------------------------------------------------------
def test_knot_currents_basics():
    solver = _bd1_single()
    z, coeffs = solver.compute_impedance()
    cur = solver.currents_at_knots(coeffs)

    assert len(cur) == 1
    assert cur[0].shape == (BD1_N + 1,)
    assert cur[0][0] == 0j
    assert cur[0][-1] == 0j

    feed = solver._feed_basis_indices(solver._build_geometry())[0]
    # Midpoint feed on an even mesh is exactly the centre knot.
    assert cur[0][BD1_N // 2] == coeffs[feed]

    arc = solver._build_geometry()["per_wire"][0]["arc_at_knot"]
    (sampled_at_knots,) = solver.currents_at_knots(coeffs, [arc])
    # momwire#809: not a two-solve comparison at all -- both sides are views
    # of ONE solve's coefficients, so there is no second BLAS path to differ.
    # Exact equality is the right assertion here.
    assert np.array_equal(sampled_at_knots, cur[0])

    mid_arcs = 0.5 * (arc[:-1] + arc[1:])
    (sampled_at_mids,) = solver.currents_at_knots(coeffs, [mid_arcs])
    assert np.allclose(sampled_at_mids, 0.5 * (cur[0][:-1] + cur[0][1:]))


# --------------------------------------------------------------------------
# 1b. current slopes — dI/ds, the charge-density readout (momwire#603 U2)
# --------------------------------------------------------------------------
def test_current_slopes_are_the_knot_differences_exactly():
    """A tent expansion IS the piecewise-linear interpolant of its own knot
    currents, so on every segment dI/ds is the constant rise over run — and
    that is not an approximation of the derivative, it is the derivative.

    Gated as an EXACT identity rather than a tolerance: anything else would
    mean the readout and the basis disagree about what the solved current is.
    """
    solver = _bd1_single()
    _z, coeffs = solver.compute_impedance()
    arc = solver._build_geometry()["per_wire"][0]["arc_at_knot"]
    (cur,) = solver.currents_at_knots(coeffs)
    expected = np.diff(cur) / np.diff(arc)

    centres = 0.5 * (arc[:-1] + arc[1:])
    (slopes,) = solver.current_slopes(coeffs, [centres])
    assert slopes.shape == centres.shape
    assert np.array_equal(slopes, expected)

    # Telescoping: the derivative integrated over the wire is the current it
    # gained end to end, which for a free-ended dipole is zero both sides.
    assert (expected * np.diff(arc)).sum() == pytest.approx(cur[-1] - cur[0], abs=1e-18)


def test_current_slopes_pick_the_same_side_of_a_knot_as_the_bspline_twin():
    """The derivative JUMPS at a knot, so a sample taken on one has to choose.

    `BSplineSolver.current_slopes` inherits scipy's choice — the span to the
    RIGHT, and the left span at the final knot. This asks that a caller
    cannot tell the two implementations apart by their tie-break, by running
    the razor knot currents back through scipy at degree 1 and comparing.
    """
    from scipy.interpolate import BSpline

    solver = _bd1_single()
    _z, coeffs = solver.compute_impedance()
    arc = solver._build_geometry()["per_wire"][0]["arc_at_knot"]
    (cur,) = solver.currents_at_knots(coeffs)

    # The same piecewise-linear function, as a clamped degree-1 spline.
    knots = np.concatenate(([arc[0]], arc, [arc[-1]]))
    scipy_slope = BSpline(knots, cur, 1, extrapolate=False).derivative(1)

    (at_knots,) = solver.current_slopes(coeffs)
    assert at_knots.shape == arc.shape
    assert np.allclose(at_knots, scipy_slope(arc), rtol=0, atol=1e-12)

    centres = 0.5 * (arc[:-1] + arc[1:])
    (at_centres,) = solver.current_slopes(coeffs, [centres])
    assert np.allclose(at_centres, scipy_slope(centres), rtol=0, atol=1e-12)


def test_current_slopes_clip_into_the_wire_and_serve_every_piece():
    """The contract the EZNEC seam calls with: one array per wire, sampled at
    that piece's own element centres, positions outside the wire clipped in
    rather than extrapolated — `BSplineSolver.current_slopes`' contract, on a
    multi-wire structure so the per-wire indexing is exercised.
    """
    solver = RazorSolver(
        wires=[
            np.array([_point(0.0), _point(BD1_LEN / 2)]),
            np.array([_point(BD1_LEN / 2), _point(BD1_LEN)]),
        ],
        n_per_edge_per_wire=[[6], [4]],
        junctions=[[(0, "end"), (1, "start")]],
        feeds=[(0, BD1_LEN / 4, 1 + 0j)],
        **BD1_KW,
    )
    _z, coeffs = solver.compute_impedance()
    per_wire = solver._build_geometry()["per_wire"]

    centres = [
        0.5 * (pw["arc_at_knot"][:-1] + pw["arc_at_knot"][1:]) for pw in per_wire
    ]
    slopes = solver.current_slopes(coeffs, centres)
    assert [s.shape[0] for s in slopes] == [6, 4]
    assert all(s.dtype == np.complex128 for s in slopes)

    # Off both ends of wire 0 -> the first and last segment's own constants.
    # `s_array` carries one entry per wire, exactly as the B-spline twin
    # indexes it, so the untested wire still has to be named.
    edge, _ = solver.current_slopes(coeffs, [np.array([-5.0, 1e3]), np.array([])])
    inside, _ = solver.current_slopes(coeffs, [centres[0][[0, -1]], np.array([])])
    assert np.array_equal(edge, inside)

    # An empty request is an empty answer, not an error.
    assert solver.current_slopes(coeffs, [np.array([]), np.array([])])[0].shape == (0,)


# --------------------------------------------------------------------------
# 2. split-identity currents — the sign gate
# --------------------------------------------------------------------------
def test_split_identity_currents():
    d = BD1_LEN / BD1_N
    cut = 8 * d

    single = _bd1_single()
    z1, c1 = single.compute_impedance()

    two = RazorSolver(
        wires=[
            np.array([_point(0.0), _point(cut)]),
            np.array([_point(cut), _point(BD1_LEN)]),
        ],
        n_per_edge_per_wire=[[8], [12]],
        feeds=[(1, BD1_LEN / 2 - cut, 1.0 + 0j)],
        **BD1_KW,
    )
    z2, c2 = two.compute_impedance()

    rev = RazorSolver(
        wires=[
            np.array([_point(0.0), _point(cut)]),
            np.array([_point(BD1_LEN), _point(cut)]),  # reversed
        ],
        n_per_edge_per_wire=[[8], [12]],
        feeds=[(1, BD1_LEN - BD1_LEN / 2, 1.0 + 0j)],
        **BD1_KW,
    )
    z3, c3 = rev.compute_impedance()

    # A common grid, chosen so the split (arc 0.4*BD1_LEN = cut) lands
    # exactly on a grid point — the junction knot itself is then sampled
    # directly, not just interpolated across.
    grid = np.linspace(0.0, BD1_LEN, 41)
    cut_idx = int(np.argmin(np.abs(grid - cut)))
    assert abs(grid[cut_idx] - cut) < 1e-12
    g0 = grid[: cut_idx + 1]
    g1 = grid[cut_idx:]

    (s1,) = single.currents_at_knots(c1, [grid])

    cur2 = two.currents_at_knots(c2, [g0, g1 - cut])
    s2 = np.concatenate([cur2[0][:-1], cur2[1]])
    assert np.allclose(s2, s1, rtol=1e-9, atol=1e-9 * np.abs(s1).max())

    # Reversed piece: map physical position p on wire1 to ITS OWN arc
    # (BD1_LEN - p). Because the feed lives on wire1, reversing that wire
    # also flips the delta-gap's physical polarity, so the whole solved
    # distribution is the negative of the un-reversed spellings' — wire0
    # (arc direction unchanged) picks up that global flip and needs
    # negating; wire1 (arc direction reversed) has the flip cancelled by
    # its own reversal and needs none. Verified numerically, not assumed.
    cur3 = rev.currents_at_knots(c3, [g0, BD1_LEN - g1])
    s3 = np.concatenate([-cur3[0][:-1], cur3[1]])
    assert np.allclose(s3, s1, rtol=1e-9, atol=1e-9 * np.abs(s1).max())


# --------------------------------------------------------------------------
# 3. loop currents — start knot is end knot
# --------------------------------------------------------------------------
SIDE = 6.05  # square loop, perimeter 24.2 m = 1.1 wavelength at 22 m
LOOP_KW = dict(wire_radius=5e-3, wavelength=22.0, nsegs=10)
CORNERS = [
    np.array([0.0, 0.0, 0.0]),
    np.array([SIDE, 0.0, 0.0]),
    np.array([SIDE, SIDE, 0.0]),
    np.array([0.0, SIDE, 0.0]),
]


def test_loop_currents_close_the_seam():
    loop = np.array(CORNERS + [CORNERS[0]])
    solver = RazorSolver(wires=[loop], feeds=[(0, SIDE / 2, 1.0 + 0j)], **LOOP_KW)
    z, coeffs = solver.compute_impedance()
    (cur,) = solver.currents_at_knots(coeffs)

    # Side A of the seam junction is the loop's START (arc 0), so both the
    # start knot and the end knot read the same junction tent coefficient,
    # signed by that wing's sigma (-1 here — see `_junction_wings`).
    assert cur[0] == cur[-1]
    assert cur[0] == -coeffs[-1]


# --------------------------------------------------------------------------
# 4. element_currents contract (the _ElementCurrents mixin)
# --------------------------------------------------------------------------
def test_element_currents_contract():
    d = BD1_LEN / BD1_N
    cut = 8 * d

    single = _bd1_single()
    z1, c1 = single.compute_impedance()
    mid1, mom1, nodes1, delta1 = single.element_currents(c1)
    n_wires1, n_elem1 = 1, BD1_N
    assert mid1.shape == (n_elem1, 3)
    assert mom1.shape == (n_elem1, 3)
    assert mom1.dtype == np.complex128
    assert nodes1.shape == (n_elem1 + n_wires1, 3)
    assert delta1.shape == (n_elem1 + n_wires1,)
    assert delta1.dtype == np.complex128

    two = RazorSolver(
        wires=[
            np.array([_point(0.0), _point(cut)]),
            np.array([_point(cut), _point(BD1_LEN)]),
        ],
        n_per_edge_per_wire=[[8], [12]],
        feeds=[(1, BD1_LEN / 2 - cut, 1.0 + 0j)],
        **BD1_KW,
    )
    z2, c2 = two.compute_impedance()
    mid2, mom2, nodes2, delta2 = two.element_currents(c2)
    n_wires2, n_elem2 = 2, BD1_N
    assert mid2.shape == (n_elem2, 3)
    assert nodes2.shape == (n_elem2 + n_wires2, 3)
    # Same physical elements (the split lands exactly on a mesh knot), so
    # moment (= current * dl, an oriented physical quantity) matches
    # elementwise regardless of how the wire is chopped into pieces.
    assert np.allclose(mom2, mom1, rtol=1e-9, atol=1e-9 * np.abs(mom1).max())

    # Closed loop: no free ends anywhere, so the current steps telescope to
    # zero over the whole structure (charge conservation).
    loop = np.array(CORNERS + [CORNERS[0]])
    loop_solver = RazorSolver(wires=[loop], feeds=[(0, SIDE / 2, 1.0 + 0j)], **LOOP_KW)
    z_loop, c_loop = loop_solver.compute_impedance()
    _mid, _mom, _nodes, delta_loop = loop_solver.element_currents(c_loop)
    scale = np.abs(c_loop).max()
    assert abs(delta_loop.sum()) < 1e-9 * scale


# --------------------------------------------------------------------------
# 5. swept identity
# --------------------------------------------------------------------------
def test_swept_impedance_matches_single_solve():
    solver = _bd1_single()
    z0, _ = solver.compute_impedance()
    (z_swept,) = solver.compute_impedance_swept(np.array([solver.k]))
    assert abs(z_swept - z0) <= 1e-12 * abs(z0)


def test_swept_impedance_is_smooth():
    solver = _bd1_single()
    k_array = solver.k * np.linspace(0.98, 1.02, 5)
    z = solver.compute_impedance_swept(k_array)
    assert np.all(np.isfinite(z))
    dz = np.diff(z)
    assert np.all(np.isfinite(dz))


def test_swept_y_matrix_matches_single_solve():
    wl = 22.0
    half = 0.962 * wl / 4
    w0 = np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])
    w1 = np.array([[0.1 * wl, 0.0, -half], [0.1 * wl, 0.0, half]])
    solver = RazorSolver(
        wires=[w0, w1],
        nsegs=20,
        wavelength=wl,
        feeds=[(0, None, 1.0 + 0j), (1, None, 1.0 + 0j)],
    )
    y0 = solver.compute_y_matrix()
    (y_swept,) = solver.compute_y_matrix_swept(np.array([solver.k]))
    assert y_swept.shape == (2, 2)
    assert np.max(np.abs(y_swept - y0)) <= 1e-12 * np.max(np.abs(y0))


# --------------------------------------------------------------------------
# 6. swept solves share the static fill work
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_swept_shares_static_work(monkeypatch):
    """The sweep prepares the k-independent fill work exactly once.

    Only the smooth kernel remainder exp(−jkR)−1 is k-dependent; the axis
    frames, the closed-form static moments and R itself are shared across
    the sweep via `_assemble_Z_prepare`. Wall-clock is the wrong gate for
    that (the shared sqrt is a modest slice next to the per-k complex
    exponential — measured ~1.15-1.3x on an idle box, far too thin a margin
    for a loaded CI runner), so this test pins the structure instead: one
    prepare per sweep, and the replayed per-k fills agree with independent
    single solves to solver precision.
    """
    wires = [np.array([[0.0, 0.0, 0.0], [0.0, BD1_LEN, 0.0]])]
    N = 48
    solver = RazorSolver(wires=wires, nsegs=N, wire_radius=BD1_RAD, wavelength=BD1_WL)
    k_array = solver.k * np.linspace(0.95, 1.05, 8)

    calls = []
    real_prepare = RazorSolver._assemble_Z_prepare
    monkeypatch.setattr(
        RazorSolver,
        "_assemble_Z_prepare",
        lambda self, geom: calls.append(1) or real_prepare(self, geom),
    )
    z_swept = solver.compute_impedance_swept(k_array)
    assert len(calls) == 1, f"prepare ran {len(calls)} times for one sweep"

    z_single = np.array(
        [
            RazorSolver(
                wires=wires, nsegs=N, wire_radius=BD1_RAD, wavelength=2 * np.pi / k
            ).compute_impedance()[0]
            for k in k_array
        ]
    )
    np.testing.assert_allclose(z_swept, z_single, rtol=1e-9)


# --------------------------------------------------------------------------
# 7. cancellation
# --------------------------------------------------------------------------
def test_precancelled_raises_from_swept():
    token = CancelToken()
    token.cancel()
    with pytest.raises(SolveAborted):
        RazorSolver(
            wires=[np.array([_point(0.0), _point(BD1_LEN)])],
            nsegs=BD1_N,
            wavelength=BD1_WL,
            cancel=token,
        ).compute_impedance_swept([2 * np.pi / BD1_WL])
