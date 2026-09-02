"""RazorSolver: the NEC-5 formulation twin (momwire#309, unit 1).

The acceptance gate is instrument parity. `scripts/bench_tri_razor.py` in the
antennaknobs repo is a standalone straight-wire razor-blade solver written to
answer the momwire#890 question; its ByDipole1 ladder is the reference this
class had to reproduce after being generalized to 3D polylines. Those four
values are pinned below at 5e-3 Ω absolute — a drift there means the
formulation changed, not that a tolerance was mis-guessed. (The residual is
~4e-6 Ω: the instrument uses c = 299792458 exactly while the solver family
derives c from its own eps/mu class attributes.)

The rest of the file guards the parts the straight wire cannot see: the
closed-form static moments at an off-axis observation point, the tangent
rotation at a bend, wire-to-wire coupling with no shared basis, and the
O(1/N) convergence signature that is the whole reason this formulation is
worth having in the codebase.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from momwire import CancelToken, RazorSolver, SolveAborted
from momwire.bspline import BSplineSolver
from momwire.razor import _axis_frame, _static_axis_moments

# ByDipole1 in free space: the antennaknobs validation wire, 14 MHz.
BD1_LEN = 10.18946
BD1_RAD = 0.0010262
BD1_WL = 299792458.0 / 14.0e6
BD1_WIRE = [np.array([[0.0, 0.0, 0.0], [0.0, BD1_LEN, 0.0]])]

HALF = 0.962 * 22 / 4


def _bd1(nsegs, **kw):
    z, _ = RazorSolver(
        wires=BD1_WIRE,
        nsegs=nsegs,
        wire_radius=BD1_RAD,
        wavelength=BD1_WL,
        **kw,
    ).compute_impedance()
    return z


# --------------------------------------------------------------------------
# 1. instrument parity — the momwire#309 acceptance gate
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "nsegs,z_pin",
    [
        (12, 65.98753 - 37.33922j),
        (20, 66.84215 - 33.72191j),
        (48, 67.43200 - 30.67024j),
        (100, 67.62576 - 29.44771j),
    ],
)
def test_bydipole1_matches_instrument(nsegs, z_pin):
    z = _bd1(nsegs)
    assert abs(z.real - z_pin.real) <= 5e-3, f"N={nsegs} R: {z} vs {z_pin}"
    assert abs(z.imag - z_pin.imag) <= 5e-3, f"N={nsegs} X: {z} vs {z_pin}"


# --------------------------------------------------------------------------
# 2. the quadratures are converged, so the walk below is the SCHEME's
# --------------------------------------------------------------------------
def test_quadrature_self_consistency():
    z = _bd1(12)
    z_fine = _bd1(12, n_qp_path=64, n_qp_source=24)
    assert abs(z_fine - z) < 5e-3, f"{z} vs {z_fine}"


# --------------------------------------------------------------------------
# 3. the closed-form static moments
# --------------------------------------------------------------------------
def _brute_static(obs, p0, t, h, a):
    """∫₀^h dτ/R and ∫₀^h τ dτ/R by adaptive quadrature.

    `points` puts a breakpoint at the axial projection, which is where the
    integrand peaks (sharply, since a is 1000x smaller than h) — without it
    the adaptive rule can miss the peak entirely on a collinear observer.
    """
    peak = float(np.dot(obs - p0, t))

    def integrand(tau):
        r = obs - (p0 + tau * t)
        return 1.0 / np.sqrt(r @ r + a * a)

    brk = [min(max(peak, 0.0), h)]
    m0 = quad(integrand, 0.0, h, points=brk, limit=400)[0]
    m1 = quad(lambda s: s * integrand(s), 0.0, h, points=brk, limit=400)[0]
    return m0, m1


@pytest.mark.parametrize(
    "obs",
    [
        [0.35, 0.0, 0.0],  # collinear, inside the segment (the self term)
        [1.90, 0.0, 0.0],  # collinear, beyond the far end
        [-0.60, 0.0, 0.0],  # collinear, before the near end
        [0.35, 0.02, 0.0],  # just off axis — a2 no longer dominates rho2
        [0.10, 1.30, -0.80],  # well off axis
        [4.00, 3.00, 2.00],  # far field of the segment
    ],
)
def test_static_moments_match_quadrature(obs):
    p0 = np.array([0.0, 0.0, 0.0])
    t = np.array([1.0, 0.0, 0.0])
    h, a = 0.7, 1e-3
    obs = np.asarray(obs, dtype=float)
    u_r, rho2 = _axis_frame(obs[None, :], p0[None, :], t[None, :], a)
    m0, m1 = _static_axis_moments(u_r, rho2, np.array([h]))
    b0, b1 = _brute_static(obs, p0, t, h, a)
    assert abs(m0[0, 0] - b0) <= 1e-8 * max(1.0, abs(b0)), f"m0 {m0[0, 0]} vs {b0}"
    assert abs(m1[0, 0] - b1) <= 1e-8 * max(1.0, abs(b1)), f"m1 {m1[0, 0]} vs {b1}"


def test_static_moments_rotation_invariant():
    """The closed form projects onto the segment axis, so it must not care
    which way the segment points — the straight-wire pins only ever exercise
    one axis direction."""
    rng = np.random.default_rng(309)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    p0 = rng.normal(size=3)
    obs = p0 + 0.31 * axis + np.cross(axis, [0.0, 0.0, 1.0]) * 0.05
    u_r, rho2 = _axis_frame(obs[None, :], p0[None, :], axis[None, :], 1e-3)
    m0, m1 = _static_axis_moments(u_r, rho2, np.array([0.7]))
    b0, b1 = _brute_static(obs, p0, axis, 0.7, 1e-3)
    assert abs(m0[0, 0] - b0) <= 1e-8 * abs(b0)
    assert abs(m1[0, 0] - b1) <= 1e-8 * abs(b1)


# --------------------------------------------------------------------------
# 4. the O(1/N) signature — why this class exists
# --------------------------------------------------------------------------
def test_reactance_walks_as_one_over_n():
    """X(2N) − X(N) halves down the ladder: the razor-blade testing rule's
    first-order discretization error, the momwire#890 walk. A Galerkin-tested
    tent basis (BSplineSolver degree=1) does not do this."""
    x = {n: _bd1(n).imag for n in (12, 24, 48, 96)}
    diffs = [x[2 * n] - x[n] for n in (12, 24, 48)]
    assert all(d > 0 for d in diffs), diffs
    ratios = [diffs[i] / diffs[i + 1] for i in range(len(diffs) - 1)]
    assert all(1.7 <= r <= 2.4 for r in ratios), f"diffs={diffs} ratios={ratios}"


# --------------------------------------------------------------------------
# 5. bends: the tangent rotation in T1 and the kinked testing path
# --------------------------------------------------------------------------
def test_inverted_v_agrees_with_tent_galerkin():
    """Both solvers use the same tent basis on the same mesh, so they must
    converge to the same answer — only their testing rules differ. Razor's
    O(1/N) walk is removed by a Richardson pair (2·Z(2N) − Z(N)); the
    Galerkin value at the fine mesh is already converged to well inside the
    0.5 Ω gate. N=20/40 per arm keeps the whole test near 0.7 s.
    """
    c = 5.0 / np.sqrt(2.0)  # 5 m arms, 90 deg included angle
    wires = [np.array([[-c, 0.0, -c], [0.0, 0.0, 0.0], [c, 0.0, -c]])]
    kw = dict(wires=wires, wire_radius=1e-3, wavelength=21.4)

    z_n, _ = RazorSolver(nsegs=20, **kw).compute_impedance()
    z_2n, _ = RazorSolver(nsegs=40, **kw).compute_impedance()
    z_rich = 2.0 * z_2n - z_n
    z_bs1, _ = BSplineSolver(nsegs=40, degree=1, **kw).compute_impedance()

    assert abs(z_rich.real - z_bs1.real) < 0.5, f"{z_rich} vs {z_bs1}"
    assert abs(z_rich.imag - z_bs1.imag) < 0.5, f"{z_rich} vs {z_bs1}"


# --------------------------------------------------------------------------
# 6. several wires, no shared basis: coupling is all through the kernel
# --------------------------------------------------------------------------
def test_two_parallel_dipoles_couple():
    wl = 22.0
    w0 = np.array([[0.0, 0.0, -HALF], [0.0, 0.0, HALF]])
    w1 = np.array([[0.1 * wl, 0.0, -HALF], [0.1 * wl, 0.0, HALF]])

    z_iso, _ = RazorSolver(wires=[w0], nsegs=20, wavelength=wl).compute_impedance()
    z_pair, _ = RazorSolver(
        wires=[w0, w1],
        nsegs=20,
        wavelength=wl,
        feeds=[(0, None, 1.0), (1, None, 0.0)],
    ).compute_impedance()
    assert z_pair.shape == (2,)
    assert abs(z_pair[0] - z_iso) > 1.0, f"pair={z_pair[0]} iso={z_iso}"

    # Same structure, driven from the other element. The two dipoles are
    # identical and the geometry is mirror-symmetric about their midplane,
    # so this is a geometry check, not reciprocity.
    z_mirror, _ = RazorSolver(
        wires=[w0, w1],
        nsegs=20,
        wavelength=wl,
        feeds=[(1, None, 1.0), (0, None, 0.0)],
    ).compute_impedance()
    assert abs(z_mirror[0] - z_pair[0]) < 1e-6, f"{z_mirror[0]} vs {z_pair[0]}"


# --------------------------------------------------------------------------
# 7. cancellation
# --------------------------------------------------------------------------
def test_precancelled_raises():
    token = CancelToken()
    token.cancel()
    with pytest.raises(SolveAborted):
        RazorSolver(
            wires=BD1_WIRE, nsegs=40, wavelength=BD1_WL, cancel=token
        ).compute_impedance()


def test_no_token_result_unchanged():
    z_default, c_default = RazorSolver(
        wires=BD1_WIRE, nsegs=12, wavelength=BD1_WL
    ).compute_impedance()
    z_none, c_none = RazorSolver(
        wires=BD1_WIRE, nsegs=12, wavelength=BD1_WL, cancel=None
    ).compute_impedance()
    # momwire#809: the two sides' fills measured BIT-IDENTICAL, so this
    # `==` is structural, not a solve-downstream lottery ticket.
    assert z_default == z_none
    assert np.array_equal(c_default, c_none)


# --------------------------------------------------------------------------
# scope refusals — a wrong answer is worse than no answer
# --------------------------------------------------------------------------
def test_shared_endpoint_is_a_junction():
    """Unit 1 refused this geometry; unit 2 models it. The contact is found
    from the mesh — nothing is declared — and the bent-L below solves. The
    junction physics itself is pinned in `test_razor_junctions.py`."""
    wires = [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        np.array([[0.0, 0.0, 2.0], [1.5, 0.0, 2.0]]),
    ]
    solver = RazorSolver(wires=wires, nsegs=10)
    z, coeffs = solver.compute_impedance()
    assert np.isfinite(z)
    # 9 + 9 interior tents plus the one junction tent.
    assert coeffs.shape == (19,)
    assert len(solver._build_geometry()["junctions"]) == 1


def test_distinct_endpoints_are_fine():
    """Just past the junction tolerance is a gap, not a contact."""
    wires = [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        np.array([[0.0, 0.0, 2.0 + 1e-6], [1.5, 0.0, 2.0 + 1e-6]]),
    ]
    RazorSolver(wires=wires, nsegs=6).compute_impedance()


@pytest.mark.parametrize(
    "kw",
    [
        # `ground_z` is NOT here: the PEC plane is served (momwire#398 unit
        # 2) and so is contact at a wire end (unit 3). It used to raise from
        # this list only because BD1_WIRE lies at z=0 — a geometry refusal
        # wearing an out-of-scope test's clothes. Nor is `ground_eps`, since
        # unit 4: the reflection-coefficient ground is served for a wire
        # standing clear of the plane. Nor is `ground_model="sommerfeld"`,
        # since unit 5 — the whole ground column is served now, and what is
        # refused inside it is a GEOMETRY (contact over a finite ground),
        # not a kwarg.
        {"degree": 2},
        # `junctions` is NOT here since momwire#590 step 3b. The old refusal
        # said a spec was "either redundant with the mesh or disagrees with
        # it" — true, and the disagreement is the point: declaring FEWER
        # junctions than the geometry has is how a caller says two coincident
        # ends are deliberately apart, and no other spelling said it. See
        # tests/test_junction_rule.py.
        {"junction_ports": [(0, 1.0)]},
        # `extended_kernel` is NOT here since momwire#398 D1: the taper study
        # identified the NEC-5 binary as extended-kernel everywhere, so the
        # kernel the old refusal declined is the reference's own. See
        # tests/test_razor_extended_kernel.py.
    ],
)
def test_out_of_scope_kwargs_refused(kw):
    with pytest.raises(NotImplementedError):
        RazorSolver(wires=BD1_WIRE, nsegs=10, **kw)


def test_unknown_kwarg_is_a_typo():
    with pytest.raises(TypeError, match="unexpected keyword"):
        RazorSolver(wires=BD1_WIRE, nsegs=10, wavelenght=22.0)


def test_wire_radius_validation():
    """Per-wire radii are served since momwire#147 (see
    `tests/test_razor_mixed_radius.py`); what is still refused is a bad
    one, in the siblings' words."""
    RazorSolver(wires=BD1_WIRE, nsegs=10, wire_radius=[1e-3])
    with pytest.raises(ValueError, match="length-1"):
        RazorSolver(wires=BD1_WIRE, nsegs=10, wire_radius=[1e-3, 2e-3])
    with pytest.raises(ValueError, match="positive and finite"):
        RazorSolver(wires=BD1_WIRE, nsegs=10, wire_radius=0.0)
    with pytest.raises(ValueError, match="positive and finite"):
        RazorSolver(wires=BD1_WIRE, nsegs=10, wire_radius=[-1e-3])


# --------------------------------------------------------------------------
# feed conventions
# --------------------------------------------------------------------------
def test_feed_snaps_to_nearest_interior_knot():
    """A between-knots arclength has no testing row of its own, so it snaps.
    On an odd mesh the midpoint default therefore lands one half-segment off
    centre, and naming that knot's arclength explicitly gives the same answer.
    """
    z_mid = _bd1(21)
    h = BD1_LEN / 21
    z_knot = _bd1(21, feed_arclength=round(BD1_LEN / 2 / h) * h)
    assert abs(z_mid - z_knot) < 1e-12, f"{z_mid} vs {z_knot}"


def test_coefficients_are_knot_currents():
    """The second return slot is the solved tent coefficients; on a tent
    basis the coefficient at a knot IS the current there, so the feed knot's
    coefficient reproduces Z = V / I and the distribution is smooth."""
    solver = RazorSolver(
        wires=BD1_WIRE, nsegs=20, wire_radius=BD1_RAD, wavelength=BD1_WL
    )
    z, coeffs = solver.compute_impedance()
    feed = solver._feed_basis_indices(solver._build_geometry())[0]
    assert coeffs.shape == (19,)
    assert abs(1.0 / coeffs[feed] - z) < 1e-12
    assert abs(coeffs[feed]) == pytest.approx(np.abs(coeffs).max())
