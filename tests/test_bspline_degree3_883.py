"""Degree 3 on the B-spline basis axis (momwire#883).

The axis used to stop at `bspline-2` because the generated same-edge moment
tables did, not because the formulation did. Raising `MAX_D` in
`scripts/derive_bspline_static_moments.py` moved four things, and each one is
gated here:

* the generated tables now reach p, q = 3, and the NEW entries are checked
  against direct quadrature of the defining integrals — nothing else in the
  repo pins them, and they are machine-generated expressions no one reads;
* the EK family's by-parts route needed a real correction to be valid at
  q = 3 (its `J_{p,0}` term is `J_{p,q-2}`), which the quadrature check is
  what catches;
* `_V_UNIT_INV` gained a degree-3 row, checked against its own Vandermonde
  rather than trusted as a transcription;
* the degree bound is READ from the generated file, so this file asserts the
  refusal tracks `MAX_D` instead of pinning the number 3.

The convergence check is deliberately a CONVERGENCE assertion — the answer
stops moving under refinement — and not a value pin. A value pin at one mesh
passes straight through a non-convergent basis, which is the whole failure it
would exist to catch.
"""

import numpy as np
import pytest
from scipy import integrate

from momwire import ArrayBlockSolver, BSplineSolver, HMatrixSolver
from momwire._bspline_static_moments import MAX_D, J_static_moment
from momwire._bspline_ek_moments import D_ek_moment
from momwire.bspline import _BSPLINE_MAX_DEGREE, _V_UNIT_INV

LAM = 8.0
RADIUS = 0.01


def _dipole(n, degree, cls=BSplineSolver, **kw):
    return cls(
        wires=[np.array([(0.0, 0.0, -1.9), (0.0, 0.0, 1.9)])],
        n_per_edge_per_wire=[[n]],
        wavelength=LAM,
        wire_radius=RADIUS,
        feeds=[(0, 1.9, 1 + 0j)],
        degree=degree,
        **kw,
    )


# ----------------------------------------------------------------------
# The bound is derived, not restated
# ----------------------------------------------------------------------


def test_the_degree_bound_is_the_generated_tables_max_d():
    """One source of truth. A hand-written bound here is how a re-run of the
    deriver silently fails to widen the axis — or, worse, how the bound gets
    widened past what was generated."""
    assert _BSPLINE_MAX_DEGREE == MAX_D
    assert MAX_D >= 3, "this file is about degree 3; the tables no longer reach it"


def test_one_past_the_bound_refuses_by_name():
    with pytest.raises(NotImplementedError, match=r"derive_bspline_static_moments"):
        _dipole(12, MAX_D + 1)


def test_every_degree_up_to_the_bound_constructs_and_solves():
    """Not `degree=3` alone: the point of the axis is that the whole range is
    reachable, and a bound that admits a degree the tables cannot serve would
    show up here rather than as a wrong number."""
    for d in range(1, MAX_D + 1):
        z = complex(_dipole(16, d).compute_impedance()[0])
        assert np.isfinite(z.real) and np.isfinite(z.imag), f"degree {d} → {z}"
        assert 10.0 < z.real < 500.0, f"degree {d} → implausible R: {z}"


# ----------------------------------------------------------------------
# The generated tables, against quadrature
# ----------------------------------------------------------------------

_GEOM = (0.30, 0.95, 0.10, 0.70, 0.023)


def _reduced(xi, a):
    return 1.0 / np.sqrt(xi * xi + a * a)


def _correction(xi, a):
    r = np.sqrt(xi * xi + a * a)
    return -(a**2) / (2.0 * r**3) + 3.0 * a**4 / (4.0 * r**5)


def _quad(kernel, p, q):
    alpha, beta, A, B, a = _GEOM
    val, err = integrate.dblquad(
        lambda t, s: (s - alpha) ** p * (t - A) ** q * kernel(s - t, a),
        alpha,
        beta,
        lambda _s: A,
        lambda _s: B,
        epsabs=1e-15,
        epsrel=1e-13,
    )
    # An unconverged reference is not a reference.
    assert abs(err) < 1e-9 * abs(val) + 1e-15, f"quadrature err {err:.3e} at {p},{q}"
    return val


@pytest.mark.parametrize("q", range(MAX_D + 1))
@pytest.mark.parametrize("p", range(MAX_D + 1))
def test_the_generated_moments_match_direct_quadrature(p, q):
    """Both families, every (p, q). The degree-3 entries are new and unpinned
    anywhere else; the degree ≤ 2 ones are here so a regeneration that moved
    them would fail loudly rather than only shifting a recorded hash.

    Δg is ONE integrand here, never split into its /R³ and /R⁵ halves: those
    are individually O(1) while their sum is O(a²), so differencing them would
    make the reference the residue of a catastrophic cancellation.
    """
    assert J_static_moment(p, q, *_GEOM) == pytest.approx(
        _quad(_reduced, p, q), rel=1e-9
    )
    assert D_ek_moment(p, q, *_GEOM) == pytest.approx(
        _quad(_correction, p, q), rel=1e-9
    )


def test_the_ek_family_is_not_just_the_q_le_2_route_extended():
    """The by-parts route's reused J call carries index q−2, not 0.

    They are the same number for every q the deriver had ever been run at, so
    this is the assertion that the correction is present: at q = 3 the wrong
    index is off by a factor of ~5–21, far outside any tolerance. Checked
    against quadrature rather than against a recorded value, so it cannot be
    satisfied by re-recording.
    """
    assert MAX_D >= 3
    for p in range(3):
        got = D_ek_moment(p, 3, *_GEOM)
        want = _quad(_correction, p, 3)
        assert got == pytest.approx(want, rel=1e-9)
        # And the wrong spelling really would have failed: the two J moments
        # the two spellings would reuse are not close to each other.
        j_right = J_static_moment(p, 1, *_GEOM)
        j_wrong = J_static_moment(p, 0, *_GEOM)
        assert abs(j_right - j_wrong) > 0.1 * abs(j_right), (
            f"J_{{{p},1}} and J_{{{p},0}} are too close for this test to "
            f"distinguish the two spellings"
        )


# ----------------------------------------------------------------------
# The Vandermonde table
# ----------------------------------------------------------------------


@pytest.mark.parametrize("d", sorted(_V_UNIT_INV))
def test_each_vandermonde_inverse_is_the_exact_inverse(d):
    """`_V_UNIT_INV` is transcribed literals, not a computed inverse (so that
    roundoff stays out of the shipping degree-1 and degree-2 answers). That
    makes a typo the failure mode, which is what this checks."""
    u = np.arange(d + 1) / d
    vander = np.vander(u, d + 1, increasing=True)
    prod = _V_UNIT_INV[d] @ vander
    assert prod == pytest.approx(np.eye(d + 1), abs=1e-13), prod


def test_the_table_covers_every_degree_the_solver_admits():
    assert sorted(_V_UNIT_INV) == list(range(1, _BSPLINE_MAX_DEGREE + 1))


# ----------------------------------------------------------------------
# Convergence, not a value pin
# ----------------------------------------------------------------------


MESHES = (10, 20, 40, 80, 160)


def _refinement_steps(degree, cls=BSplineSolver, feed_model="point"):
    zs = [
        complex(_dipole(n, degree, cls, feed_model=feed_model).compute_impedance()[0])
        for n in MESHES
    ]
    return [abs(zs[i + 1] - zs[i]) for i in range(len(zs) - 1)]


def test_this_deck_does_not_converge_in_z_at_any_degree():
    """The precondition for the comparison below, asserted rather than assumed.

    Measured 2026-09-07 on this dipole: the step between successive meshes is
    FLAT at every degree — degree 2 gives 0.53 / 0.57 / 0.56 / 0.53 Ω across
    four doublings, and splitting R from X does not rescue it (dX ≈ 0.45 flat).
    A constant step per doubling is a logarithmic drift, which is the delta-gap
    drive's own behaviour as Δ → 0, not the basis's.

    This test exists so that a future reader does not read
    `test_degree_three_is_not_worse_than_degree_two` as a convergence claim,
    and so that if the drive is ever regularised and this deck DOES start
    converging, this fails and the comparison below can be sharpened into the
    real thing.
    """
    # Degrees 2 and 3 only — the two being compared. Degree 1 is far enough
    # from its own limit at N = 10 that its early steps are dominated by
    # ordinary basis error (3.09 → 0.69 over the ladder) and it is still on
    # its way INTO this regime rather than in it.
    for degree in (2, 3):
        steps = _refinement_steps(degree)
        assert steps[-1] > 0.25 * steps[0], (
            f"degree {degree} steps {steps} now shrink like a converging "
            f"sequence — see this test's docstring, the comparison below can "
            f"now be a convergence assertion instead"
        )


def test_degree_three_is_not_worse_than_degree_two():
    """The issue's own gate for putting `bspline-3` on the axis.

    NOT a claim that degree 3 converges faster: the test above shows the
    observable does not converge here at all, so no order can be read off it.
    What is measurable, and what this pins, is that degree 3's step is never
    larger than degree 2's at matched mesh — across the whole ladder and under
    both drives, so a single lucky rung cannot carry it. Measured margins are
    5-25 %; the tolerance is generous because the point is the sign of the
    difference, not its size.
    """
    for feed_model in ("point", "segment"):
        d2 = _refinement_steps(2, feed_model=feed_model)
        d3 = _refinement_steps(3, feed_model=feed_model)
        for n_from, n_to, a, b in zip(MESHES[:-1], MESHES[1:], d2, d3, strict=True):
            assert b <= a * 1.05, (
                f"{feed_model} feed, N {n_from}→{n_to}: degree 3 step {b:.4g} "
                f"exceeds degree 2's {a:.4g}"
            )


@pytest.mark.parametrize("cls", [HMatrixSolver, ArrayBlockSolver])
def test_the_subclasses_serve_degree_three_too(cls):
    """They inherit `BSplineSolver`'s declared row, so the matrix now says
    they are bspline-3. A declared capability nobody exercised is the thing
    this asserts against — and their solve paths (ACA, element-block) are not
    the dense one degree 3 was measured on."""
    z_dense = complex(_dipole(24, 3).compute_impedance()[0])
    z_other = complex(_dipole(24, 3, cls).compute_impedance()[0])
    assert z_other == pytest.approx(z_dense, rel=2e-3), f"{z_other} vs {z_dense}"


def test_the_declared_row_and_the_axis_vocabulary_agree_on_degree_three():
    from momwire._capabilities import AXIS_VALUES

    assert "bspline-3" in AXIS_VALUES["basis"]
    assert "bspline-3" in BSplineSolver.capabilities.axes["basis"]
    # The row must not claim a degree the constructor refuses.
    declared = {int(v.split("-")[1]) for v in BSplineSolver.capabilities.axes["basis"]}
    assert max(declared) <= _BSPLINE_MAX_DEGREE
