"""`SinusoidalSolver.current_slopes` — the closed-form dI/ds (momwire#611).

The NEC-5 dialect prints a CHARGE DENSITY table, `q = −(1/jω)·dI/ds` at each
element's centre, and reads it off the basis through this method rather than
differencing two current samples (#497's argument). `RazorSolver` got one in
#603 U2; this family had none, and every deck that reached the readout died
on a missing attribute rather than on anything about the physics.

Here the derivative is genuinely closed form. The basis is {1, sin kξ, cos kξ}
per segment, so on the well-scaled shape set

    f  = σ(A+C) + B·sin kξ + σC·(cos kξ − 1)
    f' = k·[ B·cos kξ − σC·sin kξ ]

and the interesting part is what is MISSING from f': `AC`. That coefficient is
the one #606 had to rebuild in a per-branch closed form, because A ≈ −C to
O((kΔ)²) makes the float sum `A + C` worthless below kΔ ≈ 1e-4 — 1 % wrong at
2.1e-4 and no correct digits at all by 1e-5. It differentiates away. So the
cancellation that governs the value's accuracy is not in the derivative at
all, and the ladder below is here to say that out loud rather than to guard a
fix: it is the same ladder #606 walks, and this quantity does not walk on it.

What this module does NOT cover is whether the family can answer the seam.
It cannot, and for a reason one layer down — every deck in that dialect drives
a NODE and this family's gap lands on the nearest segment CENTRE — which is
`test_eznec_basis_choice.py`'s to state and `test_capabilities.py`'s to
measure. A method that works and a family that refuses are not in tension:
#611 is exactly the observation that they were being conflated.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

WIRE = [(0.0, -0.25, 0.0), (0.0, 0.25, 0.0)]
RADIUS = 5e-4

BOTH = pytest.mark.parametrize(
    "solver_class", [SinusoidalSolver, SinusoidalGalerkinSolver]
)


def _dipole(solver_class, nsegs=21, wavelength=1.0, **kwargs):
    solver = solver_class(
        wires=[WIRE],
        nsegs=nsegs,
        wavelength=wavelength,
        wire_radius=RADIUS,
        **kwargs,
    )
    _z, alpha = solver.compute_impedance()
    return solver, alpha


def _arc_at_knot(solver, w_idx=0):
    geom = solver._build_geometry()
    first = geom["wire_first"][w_idx]
    last = geom["wire_last"][w_idx]
    h = np.asarray(geom["seg_h"], dtype=float)[first : last + 1]
    return np.concatenate([[0.0], np.cumsum(h)])


def _centres(arc):
    return 0.5 * (arc[:-1] + arc[1:])


def _central_difference(solver, alpha, positions, step):
    ahead = solver.currents_at_knots(alpha, [positions + step])[0]
    behind = solver.currents_at_knots(alpha, [positions - step])[0]
    return (ahead - behind) / (2.0 * step)


# --------------------------------------------------------------------------
# it is the derivative of the current this expansion actually carries
# --------------------------------------------------------------------------


@BOTH
def test_the_closed_form_is_the_derivative_of_currents_at_knots(solver_class):
    """Against a central difference of the VALUE readout, at element centres.

    Centres because that is where the seam reads the charge table, not
    because a knot would be hostile — `dI/ds` is continuous across one in
    this basis, which the continuity test below is about, so a difference
    quotient straddling a knot would be legitimate here too. It is not on
    razor, and using centres keeps the three families' readouts checked the
    same way.
    """
    solver, alpha = _dipole(solver_class, nsegs=21)
    arc = _arc_at_knot(solver)
    centres = _centres(arc)
    step = 1e-4 * float(np.min(np.diff(arc)))

    exact = solver.current_slopes(alpha, [centres])[0]
    approx = _central_difference(solver, alpha, centres, step)

    assert exact.shape == centres.shape
    assert exact.dtype == np.complex128
    assert np.max(np.abs(exact - approx)) / np.max(np.abs(approx)) < 1e-9


@BOTH
def test_the_derivative_does_not_walk_down_the_606_ladder(solver_class):
    """The conditioning claim, on #606's own ladder: N and k·Δ, four decades.

    `AC` is absent from f', so nothing here evaluates a small quantity from
    O(1) terms and there is no relative error growing like 1/(kΔ)² to find.
    Asserted as a FLAT bound rather than a trend, because a bound that holds
    at every rung is the whole claim; a trend fit would pass on a walk that
    started one decade later.
    """
    worst = {}
    for nsegs in (11, 41, 161, 641):
        for wavelength in (1.0, 1e3, 1e5):
            solver, alpha = _dipole(solver_class, nsegs=nsegs, wavelength=wavelength)
            arc = _arc_at_knot(solver)
            centres = _centres(arc)
            step = 1e-4 * float(np.min(np.diff(arc)))
            exact = solver.current_slopes(alpha, [centres])[0]
            approx = _central_difference(solver, alpha, centres, step)
            rel = float(np.max(np.abs(exact - approx)) / np.max(np.abs(approx)))
            k_delta = float(solver.k * np.min(np.diff(arc)))
            worst[(nsegs, wavelength)] = (k_delta, rel)

    assert min(kd for kd, _ in worst.values()) < 1e-7, "ladder did not reach low k*D"
    for (nsegs, wavelength), (k_delta, rel) in worst.items():
        assert rel < 1e-8, (nsegs, wavelength, k_delta, rel)


@BOTH
def test_a_junction_neighbours_sigma_is_a_coefficient_sign_not_a_reversal(
    solver_class,
):
    """The σ = −1 entries, which is where the VALUE readout once had a bug.

    `currents_at_knots` used to multiply the whole bracket by σ, adding a
    spurious 2·B·sin(kξ) on a junction neighbour (#203, fixed 2026-06). The
    derivative carries σ on exactly one of its two terms and must not repeat
    that shape, so this drives a two-wire structure whose junction makes σ
    = −1 entries exist and checks the closed form against the difference
    quotient there — on the wire that meets the junction, not just globally.
    """
    # Both wires point INTO the node, so the shared basis runs against wire
    # 1's arc there and its entries carry σ = −1. Spelling the junction
    # `end -> start` instead — the collinear, same-sense one — produces no
    # σ = −1 entry at all, and this test passed vacuously until the guard
    # below caught that.
    solver = solver_class(
        wires=[
            np.array([(0.0, -0.25, 0.0), (0.0, 0.0, 0.0)]),
            np.array([(0.0, 0.25, 0.0), (0.0, 0.0, 0.0)]),
        ],
        n_per_edge_per_wire=[[7], [5]],
        junctions=[[(0, "end"), (1, "end")]],
        feeds=[(0, 0.125, 1.0 + 0.0j)],
        wavelength=1.0,
        wire_radius=RADIUS,
    )
    _z, alpha = solver.compute_impedance()
    seg_view = solver._basis_coefs(solver._build_geometry(), solver.k)
    assert np.any(np.asarray(seg_view["sigma"]) < 0), "no sigma = -1 entry to test"

    for w_idx in (0, 1):
        arc = _arc_at_knot(solver, w_idx)
        centres = _centres(arc)
        step = 1e-4 * float(np.min(np.diff(arc)))
        positions = [np.array([]), np.array([])]
        positions[w_idx] = centres
        exact = solver.current_slopes(alpha, positions)[w_idx]
        ahead = solver.currents_at_knots(alpha, [c + step for c in positions])[w_idx]
        behind = solver.currents_at_knots(alpha, [c - step for c in positions])[w_idx]
        approx = (ahead - behind) / (2.0 * step)
        assert np.max(np.abs(exact - approx)) / np.max(np.abs(approx)) < 1e-9, w_idx


# --------------------------------------------------------------------------
# the contract, which is the B-spline twin's and razor's
# --------------------------------------------------------------------------


@BOTH
def test_the_two_calling_conventions_agree_at_the_knots(solver_class):
    """`current_slopes(a)` and `current_slopes(a, [knot_arcs])` are one path.

    The same claim `test_currents_at_knots_s_array_matches_default_at_knots`
    makes of the value readout, and here it is bit-for-bit rather than
    approximate: the default path builds the knot arcs and walks the SAME
    `_slope_eval_points` mapping, so there is no second spelling to drift.
    """
    solver, alpha = _dipole(solver_class, nsegs=12)
    arc = _arc_at_knot(solver)
    default = solver.current_slopes(alpha)[0]
    via_s = solver.current_slopes(alpha, [arc])[0]
    assert default.shape == arc.shape
    assert np.array_equal(default, via_s)


@BOTH
def test_the_charge_density_is_CONTINUOUS_across_an_interior_knot(solver_class):
    """Which is a property of NEC-2's basis, and not of the readout.

    The first draft of this test asserted the opposite — that `dI/ds` jumps
    at a knot, so the tie-break has consequences — by analogy with
    `RazorSolver`, where it does. It does not here. NEC-2's basis matches the
    current AND its derivative at every segment junction (Eqs 43-64), so the
    charge density this family reports is continuous where a tent expansion's
    is a staircase. Measured on the same mesh: razor jumps 27-53 % of its
    peak slope at these knots.

    So the tie-break `current_slopes` documents — the span to the RIGHT, the
    left one at the final knot, matching razor and scipy at degree 1 — is
    real as a contract and UNOBSERVABLE as a number on this family. Both
    halves are asserted: a caller who samples on a knot gets the same answer
    from either side, and gets it from the side the contract names.
    """
    solver, alpha = _dipole(solver_class, nsegs=12)
    arc = _arc_at_knot(solver)
    nudge = 1e-9 * float(np.min(np.diff(arc)))

    on_knot = solver.current_slopes(alpha, [arc])[0]
    from_right = solver.current_slopes(alpha, [arc[:-1] + nudge])[0]
    from_left = solver.current_slopes(alpha, [arc[1:] - nudge])[0]
    scale = np.max(np.abs(on_knot))

    # Continuity: the two spans meeting at an interior knot agree.
    assert np.max(np.abs(from_right[1:] - from_left[:-1])) < 1e-8 * scale

    # And the sample ON the knot is the right-hand span, the last one left.
    assert np.allclose(on_knot[:-1], from_right, rtol=0, atol=1e-8 * scale)
    assert on_knot[-1] == pytest.approx(from_left[-1], abs=1e-8 * scale)


@BOTH
def test_positions_clip_into_the_wire_and_every_wire_is_served(solver_class):
    """The contract the EZNEC seam calls with, on a multi-wire structure:
    one array per wire, sampled at that piece's own element centres,
    positions outside the wire clipped in rather than extrapolated.
    """
    solver = solver_class(
        wires=[
            np.array([(0.0, -0.25, 0.0), (0.0, 0.0, 0.0)]),
            np.array([(0.0, 0.0, 0.0), (0.0, 0.25, 0.0)]),
        ],
        n_per_edge_per_wire=[[6], [4]],
        junctions=[[(0, "end"), (1, "start")]],
        feeds=[(0, 0.125, 1.0 + 0.0j)],
        wavelength=1.0,
        wire_radius=RADIUS,
    )
    _z, alpha = solver.compute_impedance()
    centres = [_centres(_arc_at_knot(solver, w)) for w in (0, 1)]

    slopes = solver.current_slopes(alpha, centres)
    assert [s.shape[0] for s in slopes] == [6, 4]
    assert all(s.dtype == np.complex128 for s in slopes)

    # Off both ends of wire 0 -> the first and last segment's own values.
    edge, _ = solver.current_slopes(alpha, [np.array([-5.0, 1e3]), np.array([])])
    arc0 = _arc_at_knot(solver, 0)
    inside, _ = solver.current_slopes(alpha, [arc0[[0, -1]], np.array([])])
    assert np.array_equal(edge, inside)

    # An empty request is an empty answer, not an error.
    empty = solver.current_slopes(alpha, [np.array([]), np.array([])])
    assert [e.shape for e in empty] == [(0,), (0,)]
    assert all(e.dtype == np.complex128 for e in empty)


def test_the_galerkin_port_columns_are_read_as_ordinary_bases():
    """A junction port appends basis columns N…N+P−1 to the same CSR view.

    `_junction_port_view` says every downstream reader — `currents_at_knots`
    named among them — treats those as ordinary bases, and the slope reader
    goes through the identical gather, so it inherits that or contradicts it.
    Checked where it can be seen: the port's own basis carries current into
    the node, so a solve with one has a derivative that still matches the
    difference quotient of the value it differentiates.
    """
    solver = SinusoidalGalerkinSolver(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.25, 0.0)]),
            np.array([(0.0, 0.0, 0.0), (0.25, 0.0, 0.0)]),
            np.array([(0.0, 0.0, 0.0), (-0.25, 0.0, 0.0)]),
        ],
        n_per_edge_per_wire=[[5], [5], [5]],
        junctions=[[(0, "start"), (1, "start"), (2, "start")]],
        junction_ports=[(0, 1.0 + 0.0j)],
        feeds=[],
        wavelength=1.0,
        wire_radius=RADIUS,
    )
    _z, alpha = solver.compute_impedance()
    geom = solver._build_geometry()
    assert alpha.shape[0] == geom["n_segs"] + 1, "no port column to exercise"

    for w_idx in range(3):
        arc = _arc_at_knot(solver, w_idx)
        centres = _centres(arc)
        step = 1e-4 * float(np.min(np.diff(arc)))
        positions = [np.array([])] * 3
        positions[w_idx] = centres
        exact = solver.current_slopes(alpha, positions)[w_idx]
        ahead = solver.currents_at_knots(alpha, [c + step for c in positions])[w_idx]
        behind = solver.currents_at_knots(alpha, [c - step for c in positions])[w_idx]
        approx = (ahead - behind) / (2.0 * step)
        assert np.max(np.abs(exact - approx)) / np.max(np.abs(approx)) < 1e-8, w_idx


def test_the_charge_density_a_printout_would_carry_is_finite_and_signed():
    """The consumer's own expression, `q = −(1/jω)·dI/ds`, end to end.

    Not a second test of the derivative — a test that the quantity the seam
    would print is well-formed on a free-ended dipole: finite everywhere,
    and ANTISYMMETRIC about the centre, because current flows outward on both
    halves and the charge it deposits changes sign across the feed.
    """
    solver, alpha = _dipole(SinusoidalGalerkinSolver, nsegs=21)
    omega = 2.0 * np.pi * solver.k / (2.0 * np.pi) * 3e8  # any positive scale
    arc = _arc_at_knot(solver)
    centres = _centres(arc)
    charge = -solver.current_slopes(alpha, [centres])[0] / (1j * omega)

    assert np.all(np.isfinite(charge.real)) and np.all(np.isfinite(charge.imag))
    assert np.max(np.abs(charge + charge[::-1])) < 1e-3 * np.max(np.abs(charge))
