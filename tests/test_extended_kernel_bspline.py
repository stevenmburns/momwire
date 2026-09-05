"""NEC's extended thin-wire kernel on the B-spline family — kernel layer
(momwire#249, unit 1).

`tests/test_extended_kernel.py` gates the sinusoidal solver's EK against
nec2c impedances. This file gates the layer BELOW any solver: the moment
kernels in `momwire._bspline_kernels` and the generated closed-form
correction family in `momwire._bspline_ek_moments`. No solver is
constructed here and no engine oracle is consulted — the engine-level shift
gates against nec2c are a separate file.

ORACLE
------
The exact thin-wire kernel of a tube of radius a, observed on its own
surface at axial offset ζ, is the azimuthal average

    G_exact(ζ) = (1/2π) ∫₀^{2π} e^{-jkR(φ)}/(4πR(φ)) dφ,
    R(φ)² = ζ² + 4a² sin²(φ/2)

and the reduced and extended kernels are its O(a⁰) and O(a²) truncations.
Both this and its MOMENT over a segment pair are computed here in numpy +
scipy alone, to ~1e-12 and ~1e-9 respectively — ten orders below anything
being gated, so the 50-digit mpmath reference #246 suggested is not needed
and mpmath stays out of the test dependencies.

The moment oracle uses the SWAPPED-ORDER construction: the naive triple
quadrature fights two singularities at once and stalls (measured: no output
in 380 s). Instead note that R(φ)² = ζ² + c(φ)² with c(φ) = 2a|sin(φ/2)| has
exactly the REDUCED kernel's form with the regulator c in place of a, so

    M_exact(pair) = (1/π) ∫₀^π M_reduced(pair; c(φ)) dφ

and `M_reduced(pair; c)` is evaluated to machine precision by the same
static-closed-form + smooth-remainder split the solver uses. The outer
integral has only an integrable log endpoint at φ = 0, handled by a
geometric point ladder under `scipy.quad`.
"""

import functools

import numpy as np
import pytest
from scipy.integrate import dblquad, quad

from momwire import _potential_ground
from momwire._bspline_ek_moments import D_ek_moment
from momwire._bspline_kernels import (
    _EK,
    _ek_axis_groups,
    _ek_factor,
    _ek_reg_extra,
    _ek_reg_kernel,
    _seg_seg_full_moments_offedge,
    _seg_seg_full_moments_offedge_swept,
    _seg_seg_reg_geometry,
    _seg_seg_reg_moments,
    _seg_seg_reg_moments_from_geometry_swept,
    _seg_seg_static_moments,
)
from momwire._stable import expm1_neg_jkR

# nec2c's CVEL at 30 MHz, matching the sinusoidal EK file's deck.
LAM = 299.792458 / 30.0
K = 2 * np.pi / LAM
# The #233 ladder deck's segment length: L = 5 m over NS = 41.
H = 5.0 / 41.0

# A same-edge block is eligible in its entirety (one straight run, one
# radius), which is what the all-None spec means.
WHOLE_BLOCK = _EK(a=None, group_i=None, group_j=None)


# ----------------------------------------------------------------------
# The oracle
# ----------------------------------------------------------------------


def _g_exact_tube(zeta, a, k):
    """(1/2π)∫ e^{-jkR(φ)}/(4πR(φ)) dφ — the exact coaxial tube kernel."""

    def integrand(ph, imag):
        R = np.sqrt(zeta * zeta + 4.0 * a * a * np.sin(0.5 * ph) ** 2)
        v = np.exp(-1j * k * R) / R
        return v.imag if imag else v.real

    # Even about φ = 0, so integrate [0, π] and drop the factor of 2 against
    # the 1/2π. The ladder resolves the peak that develops as ζ/a → 0.
    pts = (0.0, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 2.0, np.pi)
    tot = 0.0 + 0.0j
    for lo, hi in zip(pts[:-1], pts[1:]):
        for imag in (False, True):
            val = quad(
                integrand, lo, hi, args=(imag,), limit=200, epsabs=1e-16, epsrel=1e-12
            )[0]
            tot += 1j * val if imag else val
    return tot / np.pi / (4 * np.pi)


def _g_reduced(zeta, a, k):
    R = np.sqrt(zeta * zeta + a * a)
    return np.exp(-1j * k * R) / (4 * np.pi * R)


def _g_ek(zeta, a, k):
    R = np.sqrt(zeta * zeta + a * a)
    return _g_reduced(zeta, a, k) * _ek_factor(R, a, k)


@functools.lru_cache(maxsize=None)
def _gl_nodes(n):
    return np.polynomial.legendre.leggauss(n)


def _pair_moment_reduced(h, m, c, k, n_gl=200):
    """∫₀^h ds ∫_{mh}^{(m+1)h} dt e^{-jkR}/(4πR), R = √((s-t)² + c²).

    Collapsed to the 1-D convolution form ∫(h-|u|)f(u - mh)du and split into
    the closed-form static part plus a smooth remainder by high-order
    Gauss-Legendre — the same split the solver uses, so this is exact to
    machine precision rather than to a quadrature tolerance.
    """

    def F2(x):
        return x * np.arcsinh(x / c) - np.sqrt(x * x + c * c)

    al, be, A, B = 0.0, h, m * h, (m + 1) * h
    stat = F2(be - A) - F2(be - B) - F2(al - A) + F2(al - B)
    x, w = _gl_nodes(n_gl)
    u = h * x
    wu = h * w
    R = np.sqrt((u - m * h) ** 2 + c * c)
    rem = np.sum(wu * (h - np.abs(u)) * (np.exp(-1j * k * R) - 1.0) / R)
    return (stat + rem) / (4 * np.pi)


def _pair_moment_exact(h, m, a, k):
    """The swapped-order exact-tube pair moment (module docstring)."""
    pts = (0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, np.pi / 2)
    tot = 0.0 + 0.0j
    for lo, hi in zip(pts[:-1], pts[1:]):
        for imag in (False, True):

            def f(u, imag=imag):
                v = _pair_moment_reduced(h, m, 2.0 * a * np.sin(u), k)
                return v.imag if imag else v.real

            val = quad(f, lo, hi, limit=200, epsabs=1e-18, epsrel=1e-10)[0]
            tot += 1j * val if imag else val
    # u = φ/2 halves the interval, so (1/π)·2·∫₀^{π/2}.
    return tot * 2.0 / np.pi


def _same_edge_moments(n_seg, h, a, max_d, k, ek):
    """Full same-edge moment tensor, static closed form + GL remainder."""
    ends = np.arange(n_seg + 1) * h
    return _seg_seg_static_moments(ends, a, max_d, ek=ek) + _seg_seg_reg_moments(
        ends, a, k, max_d, 4, ek=ek
    )


# ----------------------------------------------------------------------
# Gate 1 — pointwise: EK vs the exact tube kernel
# ----------------------------------------------------------------------

# Measured on this box (a = 0.05 m, 30 MHz), reproducing the #249 design's
# E1b table to every printed digit:
#
#   ζ/a      1.5      2        3        5        10       30       100
#   reduced  9.2e-2   7.2e-2   4.2e-2   1.8e-2   5.1e-3   7.6e-4   1.6e-4
#   EK       1.3e-3   3.4e-3   2.1e-3   4.5e-4   3.5e-5   5.4e-7   1.5e-8
#   gain     72×      21×      21×      41×      144×     1409×    11305×
#
# The ceiling is EK's own O(a⁴) truncation, not the implementation, so the
# tolerance is set just above the worst measured point rather than at
# machine precision.


@pytest.mark.parametrize("zeta_over_a", [1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0])
def test_g1_pointwise_ek_beats_reduced_against_exact_tube(zeta_over_a):
    a = 0.05
    zeta = zeta_over_a * a
    exact = _g_exact_tube(zeta, a, K)
    err_red = abs(_g_reduced(zeta, a, K) - exact) / abs(exact)
    err_ek = abs(_g_ek(zeta, a, K) - exact) / abs(exact)
    assert err_ek <= 5e-3, f"ζ/a={zeta_over_a}: EK error {err_ek:.3e}"
    assert err_red / err_ek >= 5.0, (
        f"ζ/a={zeta_over_a}: EK gain only {err_red / err_ek:.1f}× "
        f"(reduced {err_red:.3e}, EK {err_ek:.3e})"
    )


# ----------------------------------------------------------------------
# Gate 2 — the O(a⁴) order claim
# ----------------------------------------------------------------------

# #246 claims EK agrees with the exact kernel to O(a⁴) where the reduced
# kernel is O(a²). Confirmed to three digits: fitted exponents 3.981 and
# 1.993 over the three thinnest levels. Note this is a POINTWISE statement;
# it does NOT survive integration over the self pair (gate 3), because the
# moment integrates through the ζ ≲ a region where EK is still O(1) wrong.


def test_g2_order_in_a_is_four_for_ek_and_two_for_reduced():
    zeta = 0.25
    radii = [0.125 * 0.5**i for i in range(6)]
    err_red, err_ek = [], []
    for a in radii:
        exact = _g_exact_tube(zeta, a, K)
        err_red.append(abs(_g_reduced(zeta, a, K) - exact) / abs(exact))
        err_ek.append(abs(_g_ek(zeta, a, K) - exact) / abs(exact))
    p_red = np.polyfit(np.log(radii[-3:]), np.log(err_red[-3:]), 1)[0]
    p_ek = np.polyfit(np.log(radii[-3:]), np.log(err_ek[-3:]), 1)[0]
    assert 1.9 <= p_red <= 2.1, f"reduced order {p_red:.3f}"
    assert 3.8 <= p_ek <= 4.2, f"EK order {p_ek:.3f}"


# ----------------------------------------------------------------------
# Gate 3 — moment level: the quantity the Galerkin fill actually contracts
# ----------------------------------------------------------------------

# Same-edge p = q = 0 moments against the swapped-order exact-tube oracle,
# measured on this box:
#
#   h/a   pair        reduced    EK         gain
#   24    diagonal    3.74e-3    3.28e-4    11.4×
#   24    neighbour   8.05e-3    7.26e-4    11.1×
#   24    two-away    2.78e-4    3.01e-6    92×
#    6    diagonal    2.30e-2    2.31e-3     9.9×
#    6    neighbour   2.96e-2    3.28e-3     9.0×
#    6    two-away    4.33e-3    3.33e-5    130×
#    2    diagonal    8.98e-2    1.39e-2     6.5×
#    2    neighbour   6.22e-2    1.45e-2     4.3×
#    2    two-away    3.15e-2    1.29e-3     24×
#
# Tolerances are 2× the worst measured EK error at each h/a. The ~10× gain,
# against the ~10³× the pointwise order fit would suggest, is the honest
# statement of what EK buys a Galerkin fill.

_G3_TOL = {24.0: 1.5e-3, 6.0: 7e-3, 2.0: 3e-2}


@pytest.mark.parametrize("h_over_a", [24.0, 6.0, 2.0])
@pytest.mark.parametrize("m,pair", [(0, "diagonal"), (1, "neighbour"), (2, "two-away")])
def test_g3_moment_level_ek_beats_reduced_against_exact_tube(h_over_a, m, pair):
    a = H / h_over_a
    red = _same_edge_moments(4, H, a, 0, K, None)[0, 0]
    ek = _same_edge_moments(4, H, a, 0, K, WHOLE_BLOCK)[0, 0]
    exact = _pair_moment_exact(H, m, a, K)
    err_red = abs(red[0, m] - exact) / abs(exact)
    err_ek = abs(ek[0, m] - exact) / abs(exact)
    assert err_ek <= _G3_TOL[h_over_a], f"h/a={h_over_a} {pair}: EK {err_ek:.3e}"
    assert err_red / err_ek >= 3.0, (
        f"h/a={h_over_a} {pair}: gain only {err_red / err_ek:.2f}× "
        f"(reduced {err_red:.3e}, EK {err_ek:.3e})"
    )


# ----------------------------------------------------------------------
# Gate 4 — the a → 0 collapse onto the reduced kernel
# ----------------------------------------------------------------------

# Two different rates, and the difference is physics rather than sloppiness.
#
# OFF-EDGE pairs sit at R ≫ a, where the whole correction is O(a²/R²): at
# a/λ = 1e-9 on a 10 m separation the moments agree to 2.2e-16 relative,
# i.e. to roundoff, which is the design's stated ≤ 1e-15 gate.
#
# SAME-EDGE blocks collapse only as O(a/h). The correction's own diagonal
# corner probes R = a — `D₀₀` on the diagonal pair is
# -(a²/4)·[2/√(h²+a²) - 2/a] → a/2 — while the moment it corrects grows only
# like 2h·ln(2h/a). Measured: 3.0e-5 / 3.0e-8 relative at a/λ = 1e-6 / 1e-9,
# a clean factor of 1000 per decade of a. That is a collapse, just not a
# quadratic one, and the linear-in-a check below is the honest form of the
# gate. (a/λ = 1e-9 is also the practical floor: below a ≈ 1e-10 m the
# EK-OFF `J_static_moment` family itself returns NaN at this h, so there is
# nothing left to compare against.)


def test_g4_off_edge_moments_collapse_to_reduced_at_tiny_a():
    a = 1e-9 * LAM
    lo = np.array([[0.0, 0.0, i * H] for i in range(3)])
    hi = lo + np.array([0.0, 0.0, H])
    # A second collinear run 10 m up the same axis: eligible, and far.
    lo_j = lo[:2] + np.array([0.0, 0.0, 10.0])
    hi_j = hi[:2] + np.array([0.0, 0.0, 10.0])
    spec = _EK(a=None, group_i=np.zeros(3, int), group_j=np.zeros(2, int))
    off = _seg_seg_full_moments_offedge(lo, hi, lo_j, hi_j, a, K, 2, 4)
    on = _seg_seg_full_moments_offedge(lo, hi, lo_j, hi_j, a, K, 2, 4, ek=spec)
    rel = np.abs(on - off).max() / np.abs(off).max()
    assert rel <= 1e-15, f"off-edge collapse {rel:.3e}"


def test_g4_same_edge_moments_collapse_linearly_in_a():
    rels = []
    for a_over_lam in (1e-6, 1e-9):
        a = a_over_lam * LAM
        off = _same_edge_moments(3, H, a, 2, K, None)
        on = _same_edge_moments(3, H, a, 2, K, WHOLE_BLOCK)
        rels.append(float(np.abs(on - off).max() / np.abs(off).max()))
    assert rels[0] <= 1e-4, f"a/λ = 1e-6: {rels[0]:.3e}"
    assert rels[1] <= 1e-7, f"a/λ = 1e-9: {rels[1]:.3e}"
    # Three decades of a must buy at least three decades of collapse (linear)
    # and at most six (quadratic). Measured: 3.25.
    decades = np.log10(rels[0] / rels[1])
    assert 2.8 <= decades <= 6.2, f"collapse rate {decades:.3f} decades per 1e3 in a"


def test_g4_d00_is_exactly_zero_at_zero_radius():
    # D₀₀'s bracket is four bounded reciprocal square roots, so the explicit
    # a² prefactor makes a = 0 exactly 0.0 in IEEE — on any pair whose
    # segments do not touch. (A touching pair has a corner at ξ = 0, where
    # 1/R is 1/0 and the product is 0·inf = NaN. So do the q ≥ 1 moments
    # everywhere: they inherit J's asinh(ξ/a) spelling, which is ±inf at
    # a = 0. a = 0 is no more a supported input to D than it is to J — what
    # is structural is the a² factor, gated on the source below.)
    zero = np.zeros(1)
    val = D_ek_moment(0, 0, zero, zero + H, zero + 5 * H, zero + 6 * H, zero)
    assert np.all(val == 0.0), f"D_00(a=0) = {val}"


def test_g4_generated_expressions_all_carry_an_explicit_a_squared():
    # Structural twin of the gate above: check the emitted source, not just
    # one evaluation. Every branch must be `(1 / 4) * a**2 * (...)`.
    import inspect

    from momwire import _bspline_ek_moments

    src = inspect.getsource(_bspline_ek_moments.D_ek_moment)
    returns = [ln for ln in src.splitlines() if ln.strip().startswith("return")]
    assert len(returns) == 9, f"expected 9 moment branches, found {len(returns)}"
    bodies = src.split("return")[1:]
    for body in bodies:
        head = "".join(body.split())[:20]
        assert head.startswith("((1/4)*a**2*"), f"missing a² prefactor: {head}"


# ----------------------------------------------------------------------
# The generated D family, audited against a brute-force quadrature
# ----------------------------------------------------------------------

# `D_ek_moment` is derived by parts off H = -a²/(4R); this checks it against
# a direct double quadrature of the combined integrand Δg itself, which is
# the one thing the by-parts algebra could get wrong silently. Worst
# relative deviation over all 9 moments × 3 pair geometries × 4 radii down
# to a/h = 1e-3 is 5.2e-11, limited by the reference quadrature's own
# tolerance, not by the closed form.


def _d_by_quadrature(p, q, al, be, A, B, a):
    def f(t, s):
        R = np.sqrt((s - t) ** 2 + a * a)
        return (
            (s - al) ** p
            * (t - A) ** q
            * (-(a**2) / (2 * R**3) + 3 * a**4 / (4 * R**5))
        )

    return dblquad(f, al, be, A, B, epsabs=1e-18, epsrel=1e-12)[0]


_PAIRS = {
    "diagonal": (0.0, H, 0.0, H),
    "neighbour": (0.0, H, H, 2 * H),
    "two-away": (0.0, H, 5 * H, 6 * H),
}


@pytest.mark.parametrize(
    "pair,a_over_h",
    [
        ("diagonal", 0.5),
        ("diagonal", 1 / 6),
        ("neighbour", 1 / 24),
        ("two-away", 1e-3),
    ],
)
def test_d_family_matches_brute_force_quadrature(pair, a_over_h):
    al, be, A, B = _PAIRS[pair]
    a = a_over_h * H
    for p in range(3):
        for q in range(3):
            got = float(D_ek_moment(p, q, al, be, A, B, a))
            ref = _d_by_quadrature(p, q, al, be, A, B, a)
            rel = abs(got - ref) / abs(ref)
            assert rel <= 1e-9, f"D_{p}{q} {pair} a/h={a_over_h}: {rel:.3e}"


@pytest.mark.slow
def test_d_family_matches_brute_force_quadrature_full_sweep():
    worst = 0.0
    for al, be, A, B in _PAIRS.values():
        for a_over_h in (0.5, 1 / 6, 1 / 24, 1e-3):
            a = a_over_h * H
            for p in range(3):
                for q in range(3):
                    got = float(D_ek_moment(p, q, al, be, A, B, a))
                    ref = _d_by_quadrature(p, q, al, be, A, B, a)
                    worst = max(worst, abs(got - ref) / abs(ref))
    assert worst <= 1e-9, f"worst relative deviation {worst:.3e}"


def test_d00_matches_the_hand_four_corner_form():
    # D₀₀ = -(a²/4)·[four-corner difference of 1/R]. The hand derivation and
    # sympy's independently agree, so a disagreement here means the by-parts
    # assembly in the codegen script drifted.
    def hand(al, be, A, B, a):
        def inv_r(x):
            return 1.0 / np.sqrt(x * x + a * a)

        return (
            -(a**2)
            / 4
            * (inv_r(be - A) - inv_r(be - B) - inv_r(al - A) + inv_r(al - B))
        )

    for al, be, A, B in _PAIRS.values():
        for a in (H / 2, H / 24):
            got = D_ek_moment(0, 0, al, be, A, B, a)
            assert got == pytest.approx(hand(al, be, A, B, a), rel=1e-14)


# ----------------------------------------------------------------------
# `_ek_axis_groups` — label semantics
# ----------------------------------------------------------------------

# Label VALUES are an implementation detail; what is gated is the induced
# partition, i.e. which pairs the fill will extend. The cross-check against
# NEC's own IND codes on real decks is an engine-level gate and lives with
# the solver tests.


def _straight(n, radius, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    origin = np.asarray(origin, float)
    axis = np.asarray(axis, float)
    lo = np.array([origin + i * H * axis for i in range(n)])
    hi = lo + H * axis
    return lo, hi, np.tile(axis, (n, 1)), np.full(n, radius)


def _partition(labels):
    return {frozenset(np.flatnonzero(labels == g).tolist()) for g in np.unique(labels)}


def test_axis_groups_straight_wire_is_one_group():
    lo, hi, t, a = _straight(4, 0.01)
    assert _partition(_ek_axis_groups(lo, hi, t, a)) == {frozenset({0, 1, 2, 3})}


def test_axis_groups_antiparallel_tangents_still_collinear():
    # NEC takes ABS() of the tangent dot product (f.2040), so an arc-flipped
    # neighbour on the same line is still collinear.
    lo, hi, t, a = _straight(4, 0.01)
    t[1::2] *= -1.0
    assert _partition(_ek_axis_groups(lo, hi, t, a)) == {frozenset({0, 1, 2, 3})}


def test_axis_groups_radius_step_splits():
    lo, hi, t, a = _straight(4, 0.01)
    a[2:] = 0.02
    assert _partition(_ek_axis_groups(lo, hi, t, a)) == {
        frozenset({0, 1}),
        frozenset({2, 3}),
    }


def test_axis_groups_radius_step_inside_nec_tolerance_does_not_split():
    lo, hi, t, a = _straight(4, 0.01)
    a[2:] = 0.01 * (1.0 + 5e-7)  # inside NEC's 1e-6 radius threshold
    assert _partition(_ek_axis_groups(lo, hi, t, a)) == {frozenset({0, 1, 2, 3})}


def test_axis_groups_bend_splits_into_two_arms():
    lo_z, hi_z, t_z, a_z = _straight(2, 0.01)
    lo_x, hi_x, t_x, a_x = _straight(
        2, 0.01, origin=(0.0, 0.0, 2 * H), axis=(1.0, 0.0, 0.0)
    )
    lo = np.vstack([lo_z, lo_x])
    hi = np.vstack([hi_z, hi_x])
    t = np.vstack([t_z, t_x])
    a = np.concatenate([a_z, a_x])
    assert _partition(_ek_axis_groups(lo, hi, t, a)) == {
        frozenset({0, 1}),
        frozenset({2, 3}),
    }


def test_axis_groups_parallel_but_offset_wires_are_not_coaxial():
    lo_a, hi_a, t_a, a_a = _straight(2, 0.01)
    lo_b, hi_b, t_b, a_b = _straight(2, 0.01, origin=(1.0, 0.0, 0.0))
    labels = _ek_axis_groups(
        np.vstack([lo_a, lo_b]),
        np.vstack([hi_a, hi_b]),
        np.vstack([t_a, t_b]),
        np.concatenate([a_a, a_b]),
    )
    assert _partition(labels) == {frozenset({0, 1}), frozenset({2, 3})}


def test_axis_groups_collinear_wires_with_a_gap_are_one_group():
    # Two separate wires on the SAME line at the same radius — the case the
    # mirrored-source ground contact reduces to, and the one a per-end
    # neighbour rule could not see at all.
    lo_a, hi_a, t_a, a_a = _straight(2, 0.01)
    lo_b, hi_b, t_b, a_b = _straight(2, 0.01, origin=(0.0, 0.0, 5.0))
    labels = _ek_axis_groups(
        np.vstack([lo_a, lo_b]),
        np.vstack([hi_a, hi_b]),
        np.vstack([t_a, t_b]),
        np.concatenate([a_a, a_b]),
    )
    assert _partition(labels) == {frozenset({0, 1, 2, 3})}


# ----------------------------------------------------------------------
# The pair mask is what makes the fill symmetric
# ----------------------------------------------------------------------


def test_offedge_ek_extends_only_the_coaxial_pairs():
    # Observer run on the z axis, source run split between a coaxial
    # continuation and a parallel-but-offset wire. Only the coaxial half may
    # move; the offset half must stay bit-identical to the reduced answer.
    #
    # The reference is an all-ineligible spec rather than a plain EK-off
    # call: EK-off reaches the C++ kernel, whose reduction order differs from
    # numpy's, so "bit-identical" is only a meaningful claim within one
    # backend. `_ek_pair_mask` treats a negative label as never-eligible, so
    # this reference is the reduced kernel evaluated by the EK code path.
    a = 0.05
    lo_i, hi_i, _, _ = _straight(2, a)
    lo_co, hi_co, _, _ = _straight(2, a, origin=(0.0, 0.0, 5.0))
    lo_off, hi_off, _, _ = _straight(2, a, origin=(1.0, 0.0, 0.0))
    lo_j = np.vstack([lo_co, lo_off])
    hi_j = np.vstack([hi_co, hi_off])
    none_spec = _EK(a=None, group_i=np.full(2, -1), group_j=np.full(4, -1))
    spec = _EK(a=None, group_i=np.zeros(2, int), group_j=np.array([0, 0, 1, 1]))
    ref = _seg_seg_full_moments_offedge(
        lo_i, hi_i, lo_j, hi_j, a, K, 1, 4, ek=none_spec
    )
    off = _seg_seg_full_moments_offedge(lo_i, hi_i, lo_j, hi_j, a, K, 1, 4)
    on = _seg_seg_full_moments_offedge(lo_i, hi_i, lo_j, hi_j, a, K, 1, 4, ek=spec)
    # The all-ineligible spec is the reduced kernel, to backend agreement.
    assert np.abs(ref - off).max() <= 1e-18
    assert np.array_equal(on[..., 2:], ref[..., 2:])
    assert np.abs(on[..., :2] - ref[..., :2]).max() > 0.0


def test_offedge_ek_block_is_symmetric_under_pair_exchange():
    # The whole point of the coaxial rule (#249 §4.1): eligibility is a
    # property of the PAIR, so swapping the observer and source runs
    # transposes the moment block instead of changing it.
    a = 0.05
    lo_i, hi_i, _, _ = _straight(3, a)
    lo_j, hi_j, _, _ = _straight(3, a, origin=(0.0, 0.0, 5.0))
    spec_ij = _EK(a=None, group_i=np.zeros(3, int), group_j=np.zeros(3, int))
    m_ij = _seg_seg_full_moments_offedge(lo_i, hi_i, lo_j, hi_j, a, K, 0, 4, ek=spec_ij)
    m_ji = _seg_seg_full_moments_offedge(lo_j, hi_j, lo_i, hi_i, a, K, 0, 4, ek=spec_ij)
    assert np.abs(m_ij[0, 0] - m_ji[0, 0].T).max() <= 1e-18


def test_offedge_ek_mixed_row_radii_use_the_observer_row_radius():
    # Mixed radii bypass the C++ run-splitting under EK, so the numpy branch
    # has to broadcast the per-observer-row `a` into the EK factor as well as
    # into R. Compare against per-run calls with a scalar radius.
    lo_i, hi_i, _, _ = _straight(4, 0.0)
    lo_j, hi_j, _, _ = _straight(2, 0.0, origin=(0.0, 0.0, 5.0))
    a_rows = np.array([0.02, 0.02, 0.05, 0.05])
    spec = _EK(a=None, group_i=np.zeros(4, int), group_j=np.zeros(2, int))
    mixed = _seg_seg_full_moments_offedge(
        lo_i, hi_i, lo_j, hi_j, a_rows, K, 1, 4, ek=spec
    )
    for s, e, a in ((0, 2, 0.02), (2, 4, 0.05)):
        run_spec = _EK(a=None, group_i=np.zeros(2, int), group_j=np.zeros(2, int))
        run = _seg_seg_full_moments_offedge(
            lo_i[s:e], hi_i[s:e], lo_j, hi_j, a, K, 1, 4, ek=run_spec
        )
        assert np.abs(mixed[:, :, s:e, :] - run).max() <= 1e-18


def test_swept_reg_moments_with_ek_match_the_per_k_path():
    a = 0.05
    ends = np.arange(4) * H
    ks = np.array([0.5 * K, K, 2.0 * K])
    geo = _seg_seg_reg_geometry(ends, a, 1, 4, ek=WHOLE_BLOCK)
    swept = _seg_seg_reg_moments_from_geometry_swept(geo, ks)
    for i, k in enumerate(ks):
        single = _seg_seg_reg_moments(ends, a, float(k), 1, 4, ek=WHOLE_BLOCK)
        assert np.abs(swept[i] - single).max() <= 1e-18


def test_swept_offedge_moments_with_ek_match_the_per_k_path():
    a = 0.05
    lo_i, hi_i, _, _ = _straight(2, a)
    lo_j, hi_j, _, _ = _straight(2, a, origin=(0.0, 0.0, 5.0))
    ks = np.array([0.5 * K, K])
    spec = _EK(a=None, group_i=np.zeros(2, int), group_j=np.zeros(2, int))
    swept = _seg_seg_full_moments_offedge_swept(
        lo_i, hi_i, lo_j, hi_j, a, ks, 1, 4, ek=spec
    )
    for i, k in enumerate(ks):
        single = _seg_seg_full_moments_offedge(
            lo_i, hi_i, lo_j, hi_j, a, float(k), 1, 4, ek=spec
        )
        assert np.array_equal(swept[i], single)


# ----------------------------------------------------------------------
# EK-OFF is the pre-#249 code path, unchanged
# ----------------------------------------------------------------------


def test_ek_off_default_matches_explicit_none_bit_for_bit():
    a = 0.02
    ends = np.arange(5) * H
    assert np.array_equal(
        _seg_seg_static_moments(ends, a, 2),
        _seg_seg_static_moments(ends, a, 2, ek=None),
    )
    assert np.array_equal(
        _seg_seg_reg_moments(ends, a, K, 2, 4),
        _seg_seg_reg_moments(ends, a, K, 2, 4, ek=None),
    )
    lo, hi, _, _ = _straight(3, a)
    lo_j, hi_j, _, _ = _straight(3, a, origin=(0.0, 0.0, 5.0))
    assert np.array_equal(
        _seg_seg_full_moments_offedge(lo, hi, lo_j, hi_j, a, K, 2, 4),
        _seg_seg_full_moments_offedge(lo, hi, lo_j, hi_j, a, K, 2, 4, ek=None),
    )


def test_ek_spelling_reduces_to_the_reduced_kernel_exactly_at_zero_radius():
    # The #249 §2.1 claim about the G_ek,reg spelling: at a = 0 it is the
    # pre-existing expression TERM BY TERM, not merely to rounding. `fac` is
    # exactly 1.0 and `extra` exactly 0.0, and multiplying a complex by
    # (1 + 0j) and adding (0 + 0j) is the identity in IEEE — so the whole
    # remainder is bit-identical to what the EK-off branch computes.
    #
    # The reference is the reduced branch's OWN spelling, which since
    # momwire#799 is `expm1_neg_jkR`'s bracket rather than the literal
    # `exp(-jkR) - 1`. Writing the literal here would test that the EK branch
    # reproduces an expression the repo no longer evaluates anywhere — the two
    # differ by the 7e-11 the rewrite removed at kR = 1e-3, which is exactly
    # the point.
    R = np.geomspace(1e-3, 10.0, 41)
    assert np.all(_ek_factor(R, 0.0, K) == 1.0)
    assert np.all(_ek_reg_extra(R, 0.0, K) == 0.0)
    reduced = expm1_neg_jkR(K, R) / (4 * np.pi * R)
    assert np.array_equal(_ek_reg_kernel(R, 0.0, K), reduced)


# ======================================================================
# UNIT 2 — the solver wiring (#249 §6.3-6.5)
# ======================================================================
#
# Everything above gates the kernel layer with no solver in sight. What
# follows gates the THREE SOLVERS that thread `ek=` into it: that the
# eligibility labels reproduce NEC's own gating decisions on real decks,
# that Z stays Galerkin-symmetric, that ArrayBlock's block-Toeplitz
# structure survives, that the solver family still agrees with itself —
# and, at least as important, that `extended_kernel=False` is a true no-op
# down to the bit.
#
# The engine-level shift gates against nec2c (δZ vs the LADDER columns)
# are unit 3's and live with the sinusoidal EK file.

import momwire._bspline_kernels as _bk  # noqa: E402
import momwire.bspline as _bs  # noqa: E402
import momwire.hmatrix as _hm  # noqa: E402
from momwire.array_block import (  # noqa: E402
    ArrayBlockSolver,
    LatticeArrayBlock,
    cache_stats,
    reset_array_caches,
)
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.hmatrix import HMatrixSolver  # noqa: E402
from momwire.sinusoidal import SinusoidalSolver  # noqa: E402

# The #233 ladder deck: a 5 m dipole at 30 MHz.
LEN = 5.0


def _dipole_wire(half, x=0.0, y=0.0, z=0.0):
    return np.array([[x, y, z - half], [x, y, z + half]])


# ----------------------------------------------------------------------
# Gate 5 — Galerkin symmetry survives the extended kernel
# ----------------------------------------------------------------------

# The coaxial rule is symmetric in (i, j) BY CONSTRUCTION (#249 §4.1), and
# that is the whole reason NEC's per-END gating was not transcribed: a
# per-source-segment decision makes G(i, j) != G(j, i), which a collocation
# solver does not notice and a Galerkin fill does. So |Z - Zᵀ| is the
# high-gain detector for a mis-wired eligibility rule — a source-side rule
# would show up here at percent level, not at 1e-12.
#
# Measured on this box (dipole + skew wire, EK off vs on):
#
#   degree   EK off     EK on
#     1      5.7e-15    1.7e-14
#     2      1.2e-12    1.5e-12
#
# Note the degree-2 floor is ~1e-12 with EK OFF too — it is the assembly's
# own reassociation, not the kernel. The design quoted 1.3e-13 (#205's
# sinusoidal floor); that is not this family's degree-2 floor, so the
# tolerances below are pinned just above the measured EK-OFF values and the
# ratio check carries the "EK did not make it worse" claim.

_G5_ABS_TOL = {1: 1e-13, 2: 3e-12}


def _skew_pair_Z(degree, extended_kernel):
    sim = BSplineSolver(
        wires=[
            np.array([[0.0, 0.0, -2.5], [0.0, 0.0, 2.5]]),
            np.array([[3.0, 0.2, -1.0], [3.6, 1.4, 1.2]]),
        ],
        n_per_edge_per_wire=[[12], [9]],
        degree=degree,
        wavelength=LAM,
        wire_radius=0.05,
        extended_kernel=extended_kernel,
    )
    geom = sim._build_geometry()
    supp_seg, polys, _kcl, _wk, _wbg = sim._build_basis_polynomials(geom)
    return sim._compute_Z_operator(geom, supp_seg, polys)


@pytest.mark.parametrize("degree", [1, 2])
def test_g5_galerkin_symmetry_holds_with_ek_on(degree):
    def resid(Z):
        return float(np.abs(Z - Z.T).max() / np.abs(Z).max())

    off = resid(_skew_pair_Z(degree, False))
    on = resid(_skew_pair_Z(degree, True))
    assert on <= _G5_ABS_TOL[degree], f"d={degree}: |Z-Zᵀ|/|Z| = {on:.3e} with EK on"
    assert on <= 10.0 * max(off, 1e-15), (
        f"d={degree}: EK on {on:.3e} vs EK off {off:.3e} — EK degraded symmetry"
    )


# ----------------------------------------------------------------------
# Gate 6 — the coaxial labels against NEC's own IND codes
# ----------------------------------------------------------------------

# The honest form of "shared gating logic" (#249 §4.4): not shared code — a
# pinned cross-check. `SinusoidalSolver._ek_gating` transcribes NEC's
# per-end IND1/IND2 straight out of nec2-1.2.1.2.f:2019-2053; the B-spline
# side computes a per-segment coaxial-and-equal-radius label and extends a
# PAIR iff the labels match. On every deck below the two agree exactly:
#
#   deck                     NEC IND at the feature   coaxial labels
#   straight dipole          1 (free ends), 0         one group
#   vertical monopole/PEC    0 (perpendicular ground) one group WITH its image
#   90° bend                 2                        two groups
#   radius step              2                        two groups
#   K = 3 junction           2                        three groups
#   slanted ground contact   2                        image is a second group
#
# The linkage is asserted structurally rather than by hand-written index
# lists: for every pair of segments that SHARE AN ENDPOINT, the coaxial
# rule extends the pair iff neither of the two facing ends is IND = 2.


def _facing_ends(geom, tol=1e-9):
    """[(i, end_i, j, end_j)] for every pair of segments sharing an endpoint.

    `end` is 0 for NEC's end 1 (the N⁻ / seg_l side, `ind1`) and 1 for its
    end 2 (the N⁺ / seg_r side, `ind2`) — momwire's arc order is NEC's
    ICON1/ICON2 order, which is what makes the comparison legitimate.
    """
    ends = np.stack([geom["seg_l"], geom["seg_r"]], axis=1)  # (N, 2, 3)
    n = ends.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            for ei in (0, 1):
                for ej in (0, 1):
                    if np.linalg.norm(ends[i, ei] - ends[j, ej]) <= tol:
                        out.append((i, ei, j, ej))
    return out


_G6_DECKS = {
    "straight dipole": dict(
        wires=[_dipole_wire(2.5)],
        n_per_edge_per_wire=[[6]],
    ),
    "vertical monopole on PEC ground": dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]])],
        n_per_edge_per_wire=[[6]],
        ground_z=0.0,
    ),
    "90 degree bend": dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [2.0, 0.0, 2.0]])],
        n_per_edge_per_wire=[[3, 3]],
    ),
    "radius step": dict(
        wires=[
            np.array([[0.0, 0.0, -2.5], [0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]]),
        ],
        n_per_edge_per_wire=[[3], [3]],
        wire_radius=[0.01, 0.02],
        junctions=[[(0, "end"), (1, "start")]],
    ),
    "K=3 junction": dict(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        ],
        n_per_edge_per_wire=[[3], [3], [3]],
        junctions=[[(0, "start"), (1, "start"), (2, "start")]],
    ),
}


def _g6_pair(name):
    kw = dict(_G6_DECKS[name])
    kw.setdefault("wire_radius", 0.01)
    sin = SinusoidalSolver(wavelength=LAM, nsegs=6, **kw)
    bsp = BSplineSolver(wavelength=LAM, nsegs=6, degree=2, extended_kernel=True, **kw)
    gs = sin._build_geometry()
    gb = bsp._build_geometry()
    # The comparison is per-SEGMENT, so the two meshes must be the same one.
    assert np.allclose(
        0.5 * (gs["seg_l"] + gs["seg_r"]), 0.5 * (gb["seg_l"] + gb["seg_r"])
    ), f"{name}: the two solvers meshed this deck differently"
    return sin, gs, bsp, gb


@pytest.mark.parametrize("name", list(_G6_DECKS))
def test_g6_coaxial_labels_agree_with_nec_ind_codes(name):
    sin, gs, bsp, gb = _g6_pair(name)
    ind = np.stack(sin._ek_gating(gs))  # (2, N): ind[end, seg]
    labels, _ = bsp._ek_axis_labels(gb, False)
    touching = _facing_ends(gs)
    assert touching, f"{name}: no shared endpoints — the deck proves nothing"
    for i, ei, j, ej in touching:
        nec_extends = ind[ei, i] != 2 and ind[ej, j] != 2
        assert (labels[i] == labels[j]) == nec_extends, (
            f"{name}: segs {i}(end{ei + 1}, IND={ind[ei, i]}) / "
            f"{j}(end{ej + 1}, IND={ind[ej, j]}) — NEC "
            f"{'extends' if nec_extends else 'declines'}, labels "
            f"{labels[i]} vs {labels[j]}"
        )


@pytest.mark.parametrize(
    "name,every_end_extends",
    [("straight dipole", True), ("vertical monopole on PEC ground", True)],
)
def test_g6_decks_nec_extends_everywhere_are_one_coaxial_group(name, every_end_extends):
    sin, gs, bsp, gb = _g6_pair(name)
    ind = np.stack(sin._ek_gating(gs))
    assert (ind != 2).all() == every_end_extends
    labels, _ = bsp._ek_axis_labels(gb, False)
    assert len(np.unique(labels)) == 1, f"{name}: labels {labels}"


def test_g6_ground_contact_branch_matches_through_the_image():
    """NEC's IND = 0 ground case is justified by "the image continues the wire
    straight through", so the B-spline rule has to reproduce it on the
    MIRRORED source geometry rather than on the real one — a vertical
    monopole is coaxial with its own image and extends; a slanted contact
    (NEC: IND = 2) is not and does not."""
    for wire, perpendicular in (
        (np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]]), True),
        (np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 2.0]]), False),
    ):
        kw = dict(
            wires=[wire],
            n_per_edge_per_wire=[[6]],
            wavelength=LAM,
            nsegs=6,
            wire_radius=0.01,
            ground_z=0.0,
        )
        sin = SinusoidalSolver(**kw)
        bsp = BSplineSolver(degree=2, extended_kernel=True, **kw)
        ind1, _ind2 = sin._ek_gating(sin._build_geometry())
        # The ground-contact end is segment 0's end 1 on both decks.
        assert (ind1[0] == 0) == perpendicular, f"NEC IND1[0] = {ind1[0]}"
        real, image = bsp._ek_axis_labels(bsp._build_geometry(), True)
        extends = bool((real[:, None] == image[None, :]).any())
        assert extends == perpendicular, (
            f"perpendicular={perpendicular}: real {real} vs image {image}"
        )


def test_g6_a_horizontal_wire_is_never_coaxial_with_its_own_image():
    """The negative control for the joint real+image label scan. Two
    INDEPENDENT scans would label the real run 0 and the image run 0 and
    declare every real/image pair coaxial — wrong by twice the height."""
    bsp = BSplineSolver(
        wires=[np.array([[-2.5, 0.0, 1.0], [2.5, 0.0, 1.0]])],
        n_per_edge_per_wire=[[6]],
        wavelength=LAM,
        degree=2,
        wire_radius=0.01,
        ground_z=0.0,
        extended_kernel=True,
    )
    real, image = bsp._ek_axis_labels(bsp._build_geometry(), True)
    assert not (real[:, None] == image[None, :]).any(), f"{real} vs {image}"


# ----------------------------------------------------------------------
# Gate 7 — EK OFF is the pre-#249 code path, bit for bit
# ----------------------------------------------------------------------

# Two halves, because numerical identity is necessary and not sufficient
# (#233's own argument): G7 pins that the defaulted solver and an explicit
# `extended_kernel=False` produce the SAME BITS on every solver and every
# path, and G7b pins that no line of EK code was entered to produce them.
#
# Comparisons are within ONE backend only. EK-off reaches the C++ kernels
# and EK-on does not, so a cross-backend "bit identity" claim would be
# meaningless — that is unit 1's note, and it is why nothing here compares
# an EK-on fill against an EK-off one for equality.

_G7_BSPLINE = {
    "free space": dict(
        wires=[_dipole_wire(2.5)], n_per_edge_per_wire=[[21]], wire_radius=0.05
    ),
    "bend": dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.5, 0.0, 2.0]])],
        n_per_edge_per_wire=[[8, 6]],
        wire_radius=0.02,
    ),
    "mixed radii": dict(
        wires=[_dipole_wire(2.0, x=-1.5), _dipole_wire(2.0, x=1.5)],
        n_per_edge_per_wire=[[10], [10]],
        wire_radius=[0.01, 0.04],
    ),
    "PEC ground": dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.4]])],
        n_per_edge_per_wire=[[14]],
        wire_radius=0.02,
        ground_z=0.0,
    ),
    # The two finite grounds momwire#269 opened. They enter the armor here
    # rather than in their own gate so they inherit BOTH halves: the byte
    # identity above and the "no EK code was entered" counter below. Each
    # exercises a route the PEC deck does not — the Fresnel weight tables
    # (`PotentialGround.weight_tables`) and the Sommerfeld remainder
    # respectively.
    # Lifted 0.3 m clear of the plane — the same clearance its Sommerfeld
    # sibling below has always had — since momwire#282 stage 1 (2026-08-18)
    # withdrew ground CONTACT under `ground_model="refl-coef"`. This deck was
    # a contact deck for no reason connected to what it gates: what it is
    # here for is the Fresnel weight-table route, which a clear deck takes
    # identically, and the byte identity and entry-point counters read
    # nothing about the plane at all.
    "refl-coef ground": dict(
        wires=[np.array([[0.0, 0.0, 0.3], [0.0, 0.0, 2.7]])],
        n_per_edge_per_wire=[[14]],
        wire_radius=0.02,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
    ),
    "sommerfeld ground": dict(
        wires=[np.array([[0.0, 0.0, 0.3], [0.0, 0.0, 2.7]])],
        n_per_edge_per_wire=[[14]],
        wire_radius=0.02,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    ),
}


@pytest.mark.parametrize("name", list(_G7_BSPLINE))
def test_g7_bspline_ek_off_is_bit_identical_to_the_default(name):
    kw = dict(_G7_BSPLINE[name], wavelength=LAM, degree=2)
    z_def, c_def = BSplineSolver(**kw).compute_impedance()
    z_off, c_off = BSplineSolver(**kw, extended_kernel=False).compute_impedance()
    # momwire#809: the two sides' fills measured BIT-IDENTICAL, so this
    # `==` is structural, not a solve-downstream lottery ticket.
    assert z_def == z_off, f"{name}: {z_def!r} vs {z_off!r}"
    assert np.array_equal(c_def, c_off)


def test_g7_bspline_swept_ek_off_is_bit_identical_to_the_default():
    kw = dict(_G7_BSPLINE["free space"], wavelength=LAM, degree=2)
    ks = 2 * np.pi / LAM * np.array([0.9, 1.0, 1.1])
    assert np.array_equal(
        BSplineSolver(**kw).compute_impedance_swept(ks),
        BSplineSolver(**kw, extended_kernel=False).compute_impedance_swept(ks),
    )


def _hmatrix_pair(**over):
    kw = dict(
        wires=[_dipole_wire(2.5, x=-4.0), _dipole_wire(2.5, x=4.0)],
        n_per_edge_per_wire=[[24], [24]],
        wavelength=LAM,
        wire_radius=0.05,
        degree=2,
        feeds=[(0, None, 1.0 + 0j), (1, None, 1.0 + 0j)],
        aca_tol=1e-7,
        **over,
    )
    return HMatrixSolver(**kw), HMatrixSolver(extended_kernel=False, **kw)


@pytest.mark.parametrize("eta,want_far", [(0.0, False), (1.0, True)])
def test_g7_hmatrix_ek_off_is_bit_identical_to_the_default(eta, want_far):
    a, b = _hmatrix_pair(aca_eta=eta)
    # Assert the path under test is the one actually taken: eta = 0 admits
    # nothing, so every leaf is a dense near block; eta = 1 gives far blocks
    # and therefore the ACA fill.
    assert (len(a.build_hmatrix().far) > 0) == want_far
    za, _ = a.compute_impedance()
    zb, _ = b.compute_impedance()
    assert np.array_equal(np.atleast_1d(za), np.atleast_1d(zb))


# The finite grounds momwire#269 opened, on the H-matrix path: eta = 1 so
# the far-block ACA fill (and therefore the fused refl / PEC-image block
# assemblers) is the route under test, which is where an EK-off regression
# would land — `_offedge_aca_evaluators` grew a new EK branch there.
_G7_HM_GROUNDS = {
    "refl-coef": dict(ground_z=-3.0, ground_eps=(13.0, 0.005)),
    "sommerfeld": dict(
        ground_z=-3.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld"
    ),
}


@pytest.mark.parametrize("ground", list(_G7_HM_GROUNDS))
def test_g7_hmatrix_finite_ground_ek_off_is_bit_identical_to_the_default(ground):
    a, b = _hmatrix_pair(aca_eta=1.0, **_G7_HM_GROUNDS[ground])
    assert len(a.build_hmatrix().far) > 0
    za, _ = a.compute_impedance()
    zb, _ = b.compute_impedance()
    assert np.array_equal(np.atleast_1d(za), np.atleast_1d(zb))


@pytest.mark.parametrize("lattice_fft", [True, False])
def test_g7_array_block_ek_off_is_bit_identical_to_the_default(lattice_fft):
    kw = dict(
        wires=[_dipole_wire(1.2, z=3.0 * i) for i in range(4)],
        n_per_edge_per_wire=[[8]] * 4,
        wavelength=LAM,
        wire_radius=0.03,
        degree=2,
        feeds=[(i, None, 1.0 + 0j) for i in range(4)],
        lattice_fft=lattice_fft,
    )
    reset_array_caches()
    op_def = ArrayBlockSolver(**kw).build_array_blocks()
    assert isinstance(op_def, LatticeArrayBlock) == lattice_fft
    reset_array_caches()
    op_off = ArrayBlockSolver(extended_kernel=False, **kw).build_array_blocks()
    assert np.array_equal(op_def.to_dense(), op_off.to_dense())


# --- G7b: and no EK code was entered to produce any of it ---------------

_EK_ENTRY_POINTS = [
    (_bk, "_ek_factor"),
    (_bk, "_ek_reg_extra"),
    (_bk, "D_ek_moment"),
    (_bs, "_ek_axis_groups"),
    (BSplineSolver, "_ek_spec"),
]


@pytest.fixture
def ek_call_counts(monkeypatch):
    """Counters on every entry point into the extended kernel."""
    counts = {}
    for owner, attr in _EK_ENTRY_POINTS:
        counts[attr] = 0
        original = getattr(owner, attr)

        def wrapper(*args, _a=attr, _f=original, **kwargs):
            counts[_a] += 1
            return _f(*args, **kwargs)

        monkeypatch.setattr(owner, attr, wrapper)
    return counts


# The two entry points reachable ONLY from the same-edge kernels: #270 unit 1
# moved those to C++, so with the twins present they are legitimately never
# entered. `_ek_factor` is reached from EITHER numpy fallback — the same-edge
# reg kernel's `_ek_reg_kernel` calls it, and so does the off-edge fill's own
# `ek is not None` branch — so #270 unit 2 giving the off-edge fill a C++
# path too means zero calls now requires BOTH backends to be on C++, not just
# one. `_ek_axis_groups` and `_ek_spec` are still always entered: unit 2
# doesn't touch label computation, only what consumes the labels.
def _has_offedge_work(deck):
    """Whether a deck gives the off-edge kernel anything to do.

    True when there is more than one edge (so there are cross-edge pairs), or
    when there is a ground plane (image blocks route through the same kernel).
    A single edge in free space has neither — and since momwire#759 the
    pre-pass that would have been computed and then entirely overwritten is
    not computed at all, so the off-edge entry points are legitimately never
    reached on such a deck.
    """
    n_edges = sum(len(e) for e in deck["n_per_edge_per_wire"])
    return n_edges > 1 or "ground_z" in deck


_SAME_EDGE_ONLY_EK_ENTRY_POINTS = {"_ek_reg_extra", "D_ek_moment"}
_DUAL_BACKEND_EK_ENTRY_POINTS = {"_ek_factor"}


@pytest.mark.parametrize("offedge", ["numpy", "default"])
@pytest.mark.parametrize("same_edge", ["numpy", "default"])
def test_g7b_the_counters_fire_when_ek_is_on(
    ek_call_counts, monkeypatch, same_edge, offedge
):
    """The control for the three gates below: a monkeypatch that silently
    failed to bind would make them pass vacuously.

    Parametrized over BOTH the same-edge and (#270 unit 2) off-edge
    backends rather than narrowed: with a pair of C++ twins forced off,
    every counter their scope covers must fire; with them live, the
    counters exclusive to that scope must be exactly zero. Either way each
    counter is asserted, so a dead monkeypatch is still caught.
    """
    if same_edge == "numpy":
        monkeypatch.setattr(_bk, "_HAVE_BSPLINE_STATIC_EK_ACCEL", False)
        monkeypatch.setattr(_bk, "_HAVE_BSPLINE_REG_SWEPT_EK_ACCEL", False)
    if offedge == "numpy":
        monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_EK_ACCEL", False)
        monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_SWEPT_EK_ACCEL", False)
    cpp_same_edge = (
        _bk._HAVE_BSPLINE_STATIC_EK_ACCEL and _bk._HAVE_BSPLINE_REG_SWEPT_EK_ACCEL
    )
    cpp_offedge = (
        _bk._HAVE_BSPLINE_OFFEDGE_EK_ACCEL and _bk._HAVE_BSPLINE_OFFEDGE_SWEPT_EK_ACCEL
    )
    # "bend" rather than "free space", for the reason given in
    # test_u1_ek_on_same_edge_is_served_by_cpp: `_ek_factor` is a dual-backend
    # counter, so when the off-edge twins are forced to numpy this test needs a
    # deck that actually HAS off-edge work (momwire#759).
    BSplineSolver(
        **_G7_BSPLINE["bend"],
        wavelength=LAM,
        degree=2,
        extended_kernel=True,
    ).compute_impedance()
    for attr, n in ek_call_counts.items():
        if cpp_same_edge and attr in _SAME_EDGE_ONLY_EK_ENTRY_POINTS:
            assert n == 0, f"{attr}: numpy entered despite the C++ same-edge twin"
        elif attr in _DUAL_BACKEND_EK_ENTRY_POINTS:
            if cpp_same_edge and cpp_offedge:
                assert n == 0, f"{attr}: numpy entered despite both C++ twins"
            else:
                assert n > 0, f"{attr} never called with EK on"
        else:
            assert n > 0, f"{attr} never called with EK on"


@pytest.mark.parametrize("name", list(_G7_BSPLINE))
def test_g7b_bspline_ek_off_enters_no_ek_code(ek_call_counts, name):
    BSplineSolver(**_G7_BSPLINE[name], wavelength=LAM, degree=2).compute_impedance()
    assert ek_call_counts == dict.fromkeys(ek_call_counts, 0)


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_g7b_hmatrix_ek_off_enters_no_ek_code(ek_call_counts, eta):
    _hmatrix_pair(aca_eta=eta)[0].compute_impedance()
    assert ek_call_counts == dict.fromkeys(ek_call_counts, 0)


@pytest.mark.parametrize("ground", list(_G7_HM_GROUNDS))
def test_g7b_hmatrix_finite_ground_ek_off_enters_no_ek_code(ek_call_counts, ground):
    _hmatrix_pair(aca_eta=1.0, **_G7_HM_GROUNDS[ground])[0].compute_impedance()
    assert ek_call_counts == dict.fromkeys(ek_call_counts, 0)


def test_g7b_array_block_ek_off_enters_no_ek_code(ek_call_counts):
    for lattice_fft in (True, False):
        reset_array_caches()
        ArrayBlockSolver(
            wires=[_dipole_wire(1.2, z=3.0 * i) for i in range(4)],
            n_per_edge_per_wire=[[8]] * 4,
            wavelength=LAM,
            wire_radius=0.03,
            degree=2,
            feeds=[(i, None, 1.0 + 0j) for i in range(4)],
            lattice_fft=lattice_fft,
        ).build_array_blocks()
    assert ek_call_counts == dict.fromkeys(ek_call_counts, 0)


# ----------------------------------------------------------------------
# Gate 8 — ArrayBlock's block-Toeplitz structure survives EK
# ----------------------------------------------------------------------

# `_build_lattice_operator` fills ONE dense block per lattice displacement
# and reuses it at every site, which is only legitimate if eligibility is
# translation-invariant. It is (#249 §6.5) — but the COLLINEAR array is the
# case where cross-element pairs are genuinely EK-eligible, so it is the one
# that would expose a distance- or position-dependent rule. The broadside
# 4x4 is the complementary case (no cross-element pair is eligible) and
# exercises the 2-D lattice bookkeeping.
#
# The per-pair path compresses each coupling block with ACA while the
# lattice path keeps its displacement blocks exact, so the two agree only to
# the ACA truncation — at the default `aca_tol=1e-4` that is 3.6e-9, with EK
# off and on ALIKE (measured: 3.74e-9 / 3.64e-9 on the 4x1). This gate is
# about structure, not compression, so it runs at `aca_tol=1e-9`, where the
# two paths agree to 9.2e-19 (4x1) and 2.1e-17 (4x4) with EK on.

_G8_DECKS = {
    "4x1 collinear": [_dipole_wire(1.2, z=3.0 * i) for i in range(4)],
    "4x4 broadside": [
        _dipole_wire(1.2, x=4.0 * i, y=4.0 * j) for i in range(4) for j in range(4)
    ],
}


@pytest.mark.parametrize("name", list(_G8_DECKS))
def test_g8_lattice_dense_matches_the_per_pair_path_with_ek_on(name):
    wires = _G8_DECKS[name]
    kw = dict(
        wires=wires,
        n_per_edge_per_wire=[[6]] * len(wires),
        wavelength=LAM,
        wire_radius=0.03,
        degree=2,
        feeds=[(0, None, 1.0 + 0j)],
        extended_kernel=True,
        aca_tol=1e-9,
    )
    reset_array_caches()
    lat = ArrayBlockSolver(
        **kw, lattice_fft=True, require_lattice_fft=True
    ).build_array_blocks()
    assert isinstance(lat, LatticeArrayBlock)
    reset_array_caches()
    pairwise = ArrayBlockSolver(**kw, lattice_fft=False).build_array_blocks()
    assert not isinstance(pairwise, LatticeArrayBlock)
    A, B = lat.to_dense(), pairwise.to_dense()
    rel = float(np.abs(A - B).max() / np.abs(B).max())
    assert rel <= 1e-12, f"{name}: lattice vs per-pair {rel:.3e}"


# ----------------------------------------------------------------------
# Gate 15 — the solver family still agrees with itself under EK
# ----------------------------------------------------------------------

# The "nine kernel sites" risk: EK wired into some fills and not others
# would leave each solver internally consistent and the family split. The
# tolerances are the ones these pairs already hold at with EK OFF, measured
# on this deck (two fat dipoles, Δ/a = 2.44, NS = 41, degree 2):
#
#   pair                                 EK off     EK on
#   HMatrix (ACA, aca_tol=1e-7) / dense  3.30e-7    3.15e-7
#   HMatrix (dense blocks)     / dense   5.17e-12   2.81e-12
#   ArrayBlock                 / dense   3.26e-7    3.11e-7
#
# Both columns are asserted against the same bars, so the test is also its
# own control: a bar that only the EK-off column could clear would mean the
# family had drifted, not that the bar was loose.

_G15_TOL = {"hmatrix-aca": 1e-6, "hmatrix-dense": 1e-10, "array-block": 1e-6}


def _g15_common(extended_kernel):
    return dict(
        wires=[_dipole_wire(2.5, x=-6.0), _dipole_wire(2.5, x=6.0)],
        n_per_edge_per_wire=[[41], [41]],
        wavelength=LAM,
        wire_radius=0.05,
        degree=2,
        feeds=[(0, None, 1.0 + 0j), (1, None, 1.0 + 0j)],
        extended_kernel=extended_kernel,
    )


@pytest.mark.parametrize("extended_kernel", [False, True])
def test_g15_solver_family_agrees_on_a_fat_deck(extended_kernel):
    common = _g15_common(extended_kernel)
    z_dense = np.atleast_1d(BSplineSolver(**common).compute_impedance()[0])

    def rel(z):
        return float(np.abs(np.atleast_1d(z) - z_dense).max() / np.abs(z_dense).max())

    aca = HMatrixSolver(**common, aca_tol=1e-7)
    assert len(aca.build_hmatrix().far) > 0  # the ACA fill really ran
    got = {
        "hmatrix-aca": rel(aca.compute_impedance()[0]),
        "hmatrix-dense": rel(
            HMatrixSolver(**common, aca_tol=1e-7, aca_eta=0.0).compute_impedance()[0]
        ),
    }
    reset_array_caches()
    got["array-block"] = rel(ArrayBlockSolver(**common).compute_impedance()[0])
    for name, r in got.items():
        assert r <= _G15_TOL[name], f"EK={extended_kernel} {name}: {r:.3e}"


# ----------------------------------------------------------------------
# The refusals (#249 §5) — and what is NOT refused
# ----------------------------------------------------------------------


def _refusal_kw(**over):
    return dict(
        wires=[_dipole_wire(2.5)],
        n_per_edge_per_wire=[[8]],
        wavelength=LAM,
        wire_radius=0.02,
        degree=2,
        extended_kernel=True,
        **over,
    )


def test_extended_kernel_refuses_singular_enrichment():
    with pytest.raises(NotImplementedError, match="use_singular_enrichment"):
        BSplineSolver(**_refusal_kw(use_singular_enrichment=True))


def test_extended_kernel_serves_plain_pec_ground():
    z, _ = BSplineSolver(**_refusal_kw(ground_z=-3.0)).compute_impedance()
    assert np.isfinite(z)


# momwire#269 lifted #249's second refusal. What replaces the "raises" test
# is the weakest possible statement — that both models construct and answer
# a finite number on every solver in the family. The gates that say the
# answer is RIGHT are G16-G19 below.
@pytest.mark.parametrize("cls", [BSplineSolver, HMatrixSolver, ArrayBlockSolver])
@pytest.mark.parametrize("ground_model", ["refl-coef", "sommerfeld"])
def test_extended_kernel_serves_finite_ground(cls, ground_model):
    z, _ = cls(
        **_refusal_kw(
            ground_z=-3.0, ground_eps=(13.0, 0.005), ground_model=ground_model
        )
    ).compute_impedance()
    assert np.isfinite(np.atleast_1d(z)).all()


def test_extended_kernel_enrichment_refusal_reaches_the_subclasses():
    """The refusal that STAYS (#249 follow-up C / momwire#271)."""
    for cls in (BSplineSolver, HMatrixSolver, ArrayBlockSolver):
        with pytest.raises(NotImplementedError, match="use_singular_enrichment"):
            cls(**_refusal_kw(use_singular_enrichment=True))


# ----------------------------------------------------------------------
# `extended_kernel` is part of both ArrayBlock cache keys (the #240 class)
# ----------------------------------------------------------------------

# The extended kernel is a different operator on the same geometry, radius,
# degree, quadrature and k — every field the module-scope caches already
# key on. Without joining the keys, an EK-on solve in the same process
# would silently hand back the reduced-kernel operator (or self-block) that
# an earlier EK-off solve built. Same staleness class as issue #240's
# junction composition, so the same treatment and the same test.


def _cache_key_kw(extended_kernel):
    return dict(
        wires=[_dipole_wire(1.2, z=3.0 * i) for i in range(2)],
        n_per_edge_per_wire=[[8]] * 2,
        wavelength=LAM,
        wire_radius=0.03,
        degree=2,
        feeds=[(i, None, 1.0 + 0j) for i in range(2)],
        extended_kernel=extended_kernel,
    )


def test_extended_kernel_joins_the_array_operator_cache_key():
    reset_array_caches()
    n0 = cache_stats()["operator_build"]
    ops = [
        ArrayBlockSolver(**_cache_key_kw(ek))._build_operator() for ek in (False, True)
    ]
    assert cache_stats()["operator_build"] - n0 == 2, "the EK build hit a stale entry"
    A, B = (op.to_dense() for op in ops)
    assert np.abs(A - B).max() / np.abs(A).max() > 1e-4, "EK did not move the operator"


def test_extended_kernel_joins_the_self_block_cache_key():
    reset_array_caches()
    n0 = cache_stats()["self_block_build"]
    blocks = []
    for ek in (False, True):
        sim = ArrayBlockSolver(**_cache_key_kw(ek))
        ctx = sim._context()
        part = sim.array_partition()
        sim.build_array_blocks()
        key = sim._self_block_key(ctx, part.seg_groups[0], part.groups[0], sim.k)
        blocks.append(key)
    assert cache_stats()["self_block_build"] - n0 == 2
    assert blocks[0] != blocks[1], "the EK self-block would alias the reduced one"


# ----------------------------------------------------------------------
# The end-to-end shift, measured but NOT gated here
# ----------------------------------------------------------------------

# The engine-level shift gates (δZ against nec2c's own δZ column, against
# the sinusoidal basis at the same mesh, the PEC-ground parity) are unit 3's
# and belong with the LADDER pins in tests/test_extended_kernel.py. What is
# recorded here is the one measurement that says the wiring above produced
# a shift of the right size and sign before anyone formalises it — the
# #233 ladder deck at Δ/a = 2.44 (L = 5 m, NS = 41, a = 0.05, degree 2),
# against nec2c's pinned EK-OFF 101.010 + 50.114j and EK-ON 98.660 + 48.113j:
#
#   feed_model   Z(EK off)             Z(EK on)              δZ(bspline)
#   point        100.676 + 46.943j     98.128 + 45.123j      -2.548 - 1.820j
#   segment       98.824 + 48.489j     96.727 + 46.339j      -2.097 - 2.150j
#
#   nec2c                                                     -2.350 - 2.001j
#
# Both components have nec2c's sign, |δ| is within 3 % of nec2c's, and the
# vector mismatch |δ_bsp - δ_nec|/|δ_nec| is 8.7 % (point) / 9.5 % (segment)
# — comfortably inside the 30 % cross-basis bar #249 §7.2 derived for the
# shift gate. The absolute-Z residual is essentially unmoved (2.83 % → 2.77 %
# against the matching nec2c column), which is the design's point: at this
# mesh the EK shift and the bspline-vs-NEC basis difference are the same
# size, so only the SHIFT is a measurement of the kernel.


# ----------------------------------------------------------------------
# The swept paths serve EK too (#249 §5, the swept-k row)
# ----------------------------------------------------------------------

# EK adds only k-independent geometry to the reg-geometry dict (the a-powers
# of R) plus a k-dependent factor on the phase step, so nothing about the
# swept hoists changes — but there are three separate swept routes to get
# wrong: the fully batched chunk builder (`_swept_batched_z_chunks`, which
# has its own offedge and image fills), the per-k `compute_impedance` loop,
# and `_port_solutions_swept`. Each is checked against its own per-k solve
# with EK on, so a spec dropped on any one of them shows up as a mismatch
# rather than as a silently reduced sweep.

_SWEPT_DECKS = {
    "free space": dict(wires=[_dipole_wire(2.5)], n_per_edge_per_wire=[[16]]),
    "PEC ground": dict(
        wires=[np.array([[0.0, 0.0, 0.2], [0.0, 0.0, 2.6]])],
        n_per_edge_per_wire=[[14]],
        ground_z=0.0,
    ),
}


@pytest.mark.parametrize("name", list(_SWEPT_DECKS))
def test_swept_impedance_with_ek_matches_the_per_k_solves(name):
    kw = dict(
        _SWEPT_DECKS[name],
        wavelength=LAM,
        wire_radius=0.05,
        degree=2,
        extended_kernel=True,
    )
    ks = 2 * np.pi / LAM * np.array([0.85, 1.0, 1.15])
    swept = BSplineSolver(**kw).compute_impedance_swept(ks)
    sim = BSplineSolver(**kw)
    with sim._k_restored():
        per_k = []
        for kk in ks:
            sim._set_k(kk)
            per_k.append(sim.compute_impedance()[0])
    rel = np.abs(swept - np.asarray(per_k)) / np.abs(per_k)
    assert rel.max() <= 1e-11, f"{name}: swept vs per-k {rel.max():.3e}"


def test_swept_port_solutions_with_ek_match_the_per_k_solves():
    kw = dict(
        _SWEPT_DECKS["free space"],
        wavelength=LAM,
        wire_radius=0.05,
        degree=2,
        extended_kernel=True,
    )
    ks = 2 * np.pi / LAM * np.array([0.9, 1.1])
    swept = [ps.y[0, 0] for ps in BSplineSolver(**kw)._port_solutions_swept(ks)]
    sim = BSplineSolver(**kw)
    with sim._k_restored():
        per_k = []
        for kk in ks:
            sim._set_k(kk)
            per_k.append(sim.compute_port_solution().y[0, 0])
    rel = np.abs(np.asarray(swept) - np.asarray(per_k)) / np.abs(per_k)
    assert rel.max() <= 1e-11, f"swept ports vs per-k {rel.max():.3e}"


# ----------------------------------------------------------------------
# The IMAGE side is wired, and stays wired
# ----------------------------------------------------------------------

# Review finding on unit 2: setting `_build_J_image_blocks`' `ek=` argument
# to None left all 84 gates green. Two reasons, and both are worth naming.
#
# 1. G6 gates the LABELS and G15's deck is free space, so nothing above
#    connected a mirrored spec to an actual image fill.
# 2. `_build_J_image_blocks` is not even on the default solve path.
#    `_compute_Z_operator` picks `_accumulate_Z_image_chunked` whenever the
#    weighted-windowed C++ assembler is present — which is independent of
#    the dense-tensor budget — so the tensor-route image fill is the
#    NO-ACCELERATOR fallback and a mutation there is invisible to any test
#    that just calls `compute_impedance()` on this box.
#
# So there are four image wiring points, not one: the two BSplineSolver
# routes, `HMatrixSolver._zblock_image`, and the batched swept builder's
# `ek_img`. The last is already pinned (the swept-vs-per-k PEC-ground gate
# above compares a batched fill against a chunked one, so a spec dropped on
# either side shows up as a mismatch). The other three are pinned here,
# structurally by a spy and numerically by the mutation itself.

_IMAGE_DECK = dict(
    # Two wires with OPPOSITE mirror eligibility, so a spec built without
    # mirror=True is distinguishable from one built with it: the vertical
    # monopole is coaxial with its own image, the horizontal wire is not.
    wires=[
        np.array([[0.0, 0.0, 0.05], [0.0, 0.0, 2.5]]),
        np.array([[-2.0, 3.0, 1.4], [2.0, 3.0, 1.4]]),
    ],
    n_per_edge_per_wire=[[12], [12]],
    wavelength=LAM,
    wire_radius=0.05,
    degree=2,
    ground_z=0.0,
)


def _image_solver(cls, extended_kernel=True, **over):
    return cls(**_IMAGE_DECK, extended_kernel=extended_kernel, **over)


def _assert_mirrored_spec(ek, sim, rows=None, cols=None):
    """The spec really is the JOINT real+image labelling of `_IMAGE_DECK`.

    A free-space spec (`mirror=False`) would pass a naive "is not None"
    check and even a "the monopole extends" check, because there group_i is
    group_j — but it would ALSO mark the horizontal wire eligible against
    its own image, which is wrong by twice the height. Both halves are
    asserted, so only the mirrored spec survives.
    """
    assert ek is not None, "the image fill got no EK spec"
    gi = np.asarray(ek.group_i)
    gj = np.asarray(ek.group_j)
    mask = (gi[:, None] == gj[None, :]) & (gi[:, None] >= 0)
    n_mono = _IMAGE_DECK["n_per_edge_per_wire"][0][0]
    seg_rows = np.arange(sim._build_geometry()["n_segs_total"])
    r = seg_rows if rows is None else np.asarray(rows)
    c = seg_rows if cols is None else np.asarray(cols)
    mono_r, mono_c = r < n_mono, c < n_mono
    if mono_r.any() and mono_c.any():
        assert mask[np.ix_(mono_r, mono_c)].all(), (
            "the vertical monopole must extend against its own image "
            "(NEC's IND = 0 ground-contact branch)"
        )
    if (~mono_r).any() and (~mono_c).any():
        assert not mask[np.ix_(~mono_r, ~mono_c)].any(), (
            "the horizontal wire is parallel to its image, not coaxial — a "
            "free-space (mirror=False) spec would wrongly extend it"
        )


@pytest.fixture
def offedge_ek_spy(monkeypatch):
    """Record the `ek=` kwarg of every off-edge moment fill."""
    seen = []
    original = _bk._seg_seg_full_moments_offedge

    def spy(*args, ek=None, **kwargs):
        seen.append(ek)
        return original(*args, ek=ek, **kwargs)

    for mod in (_bs, _hm):
        monkeypatch.setattr(mod, "_seg_seg_full_moments_offedge", spy)
    return seen


def test_image_tensor_route_fills_with_a_mirrored_ek_spec(offedge_ek_spy):
    """`_build_J_image_blocks` — the no-accelerator fallback route."""
    sim = _image_solver(BSplineSolver)
    geom = sim._build_geometry()
    offedge_ek_spy.clear()
    sim._build_J_image_blocks(geom, sim.k)
    assert len(offedge_ek_spy) == 1
    _assert_mirrored_spec(offedge_ek_spy[0], sim)


def test_image_chunked_route_fills_with_a_mirrored_ek_spec(offedge_ek_spy):
    """`_accumulate_Z_image_chunked` — the route a default solve takes."""
    sim = _image_solver(BSplineSolver)
    geom = sim._build_geometry()
    supp_seg, polys, _kcl, _wk, _wbg = sim._build_basis_polynomials(geom)
    w_A = sim._image_tangent_dot(geom["tangents"]).astype(np.complex128)
    w_Phi = np.ones_like(w_A)
    Z = np.zeros((supp_seg.shape[0],) * 2, dtype=np.complex128, order="F")
    offedge_ek_spy.clear()
    sim._accumulate_Z_image_chunked(
        Z,
        geom,
        sim.k,
        supp_seg,
        polys,
        # The accumulator takes a per-chunk window producer (#323); these
        # explicit PEC tables are the thing under test, so slice them here
        # rather than route through `_image_weight_window_fn`.
        lambda i0, i1: (w_A[i0:i1], w_Phi[i0:i1]),
    )
    assert offedge_ek_spy, "no image fill happened"
    n_segs = geom["n_segs_total"]
    row0 = 0
    for ek in offedge_ek_spy:
        rows = np.arange(row0, row0 + len(ek.group_i))
        _assert_mirrored_spec(ek, sim, rows=rows)
        row0 += len(ek.group_i)
    assert row0 == n_segs, "the observer chunks did not cover the mesh"


def test_hmatrix_image_block_fills_with_a_mirrored_ek_spec(offedge_ek_spy):
    """`HMatrixSolver._zblock_image` — the block form of the same fill."""
    sim = _image_solver(HMatrixSolver, aca_tol=1e-9)
    ctx = sim._context()
    idx = np.arange(ctx["n_basis"], dtype=np.int64)
    offedge_ek_spy.clear()
    sim._zblock_image(idx, idx)
    assert len(offedge_ek_spy) == 1
    seg = np.unique(ctx["supp_seg"][idx].ravel())
    _assert_mirrored_spec(offedge_ek_spy[0], sim, rows=seg, cols=seg)


# --- and the mutation itself, as a numeric gate ------------------------


@pytest.fixture
def image_ek_disabled(monkeypatch):
    """THE mutation, as a fixture: the free-space blocks stay extended and
    the image blocks quietly drop to the reduced kernel."""
    original = BSplineSolver._ek_spec

    def free_space_only(self, geom, mirror=False):
        return None if mirror else original(self, geom, mirror)

    monkeypatch.setattr(BSplineSolver, "_ek_spec", free_space_only)


# Measured on this box: extending the image side moves the grounded EK-on
# impedance by 3.3e-3 relative on the monopole deck — three orders above the
# 3e-7 the accelerated and numpy fills of the SAME operator differ by, so
# the bar below is a signal test, not a noise test.
_IMAGE_EK_SIGNAL = 1e-4


def _grounded_z(cls, **over):
    return _image_solver(cls, **over).compute_impedance()[0]


@pytest.mark.parametrize("weighted_accel", [True, False])
def test_bspline_grounded_ek_uses_the_extended_image(
    request, monkeypatch, weighted_accel
):
    """Both BSplineSolver image routes: `weighted_accel=True` is the chunked
    route a default solve takes, False forces the tensor-route fallback
    (`_build_J_image_blocks`), which is otherwise unreachable on a build
    that has the C++ weighted-windowed assembler."""
    monkeypatch.setattr(_bs, "_HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL", weighted_accel)
    z_wired = _grounded_z(BSplineSolver)
    request.getfixturevalue("image_ek_disabled")
    z_reduced_image = _grounded_z(BSplineSolver)
    rel = abs(z_wired - z_reduced_image) / abs(z_wired)
    assert rel >= _IMAGE_EK_SIGNAL, (
        f"weighted_accel={weighted_accel}: the image fill answers the same "
        f"whether or not it is handed an EK spec (moved {rel:.3e}) — the "
        f"mirrored wiring is dead"
    )


def test_hmatrix_grounded_ek_uses_the_extended_image(request):
    z_wired = _grounded_z(HMatrixSolver, aca_tol=1e-9)
    request.getfixturevalue("image_ek_disabled")
    z_reduced_image = _grounded_z(HMatrixSolver, aca_tol=1e-9)
    rel = abs(z_wired - z_reduced_image) / abs(z_wired)
    assert rel >= _IMAGE_EK_SIGNAL, f"_zblock_image is not EK-wired ({rel:.3e})"


# ======================================================================
# UNIT 3 — the engine-level shift gates against nec2c (#249 §7.2)
# ======================================================================
#
# Everything above stays inside momwire: kernel-level oracles (unit 1) and
# solver self-consistency (unit 2). This section is the one place an
# external, independent implementation — nec2c — gets to check the answer,
# via the SHIFT form #249 §0 item 2 derived to replace a mesh-extrapolated
# absolute-Z comparison (measured not decisive: the EK shift and the
# bspline-vs-nec2c basis gap are the same size at any one mesh).
#
# THE ORACLE is the #233 ladder deck's own oracle: `LADDER` below is
# `tests/test_extended_kernel.py`'s dict of the same name (radius -> (Δ/a,
# nec2c EK-OFF, nec2c EK-ON)), copied rather than imported — the two files
# are independent oracle consumers and a cross-test-module import would
# make one file's collection depend on the other's.
#
# FEED MODEL: the ladder deck's nec2c source is a segment-wide gap at the
# centre segment (module docstring of test_extended_kernel.py), which is
# `feed_model="segment"` on this side, not this solver's own
# `feed_model="point"` default. All ladder-deck gates below use
# `feed_model="segment"` for that reason. `"point"` was measured too (see
# each gate's comment): it tracks "segment" closely rather than rescuing
# any rung "segment" misses, so it is noted, not gated.

LADDER = {
    0.001: (121.951, 80.146 + 46.432j, 80.146 + 46.430j),
    0.0048780: (25.000, 83.291 + 48.053j, 83.281 + 48.016j),
    0.005: (24.390, 83.364 + 48.092j, 83.353 + 48.053j),
    0.02: (6.0976, 90.040 + 50.737j, 89.774 + 50.190j),
    0.05: (2.4390, 101.010 + 50.114j, 98.660 + 48.113j),
    0.1: (1.2195, 123.470 + 34.444j, 111.020 + 37.528j),
    0.15: (0.81301, 130.460 - 30.093j, 122.240 + 18.377j),
    0.3: (0.40650, 0.40269 - 7.5778j, 37.990 - 56.910j),
}

_U3_NS = 41  # the #233 ladder deck's own mesh


@functools.lru_cache(maxsize=None)
def _u3_ladder_z(radius, extended_kernel, feed_model="segment", ns=_U3_NS):
    """Module-scope memoised solve — G9/G10/G12/G13 all read the same
    (radius, on/off) impedances off the identical ladder deck, so caching
    here means each is solved once across the whole file rather than once
    per gate."""
    z, _ = BSplineSolver(
        wires=[_dipole_wire(LEN / 2)],
        n_per_edge_per_wire=[[ns]],
        degree=2,
        wavelength=LAM,
        wire_radius=radius,
        nsegs=ns,
        feed_arclength=LEN / 2,
        feed_model=feed_model,
        extended_kernel=extended_kernel,
    ).compute_impedance()
    return z


# ----------------------------------------------------------------------
# Gate 9 — the shift vs nec2c's own δZ column
# ----------------------------------------------------------------------
#
# Only two ladder rungs clear the "nec2c's own shift >= 0.5%" floor #249
# picks (a = 0.1 and thinner are all < 0.5%, a = 0.1 itself sits below the
# Δ/a >= 2 usable floor and is excluded on that ground too):
#
#   a       Δ/a      nec2c shift    δ_bsp             δ_nec            mismatch
#   0.02    6.098    0.589%         -0.354-0.795j     -0.266-0.547j    43.2%
#   0.05    2.439    2.737%         -2.097-2.149j     -2.350-2.001j    9.51%
#
# a = 0.05 lands well inside the design's proposed 30% cross-basis bar, so
# it is gated tighter (15%, ~1.6x the measured 9.51%) to keep the gate
# decisive rather than merely satisfied.
#
# a = 0.02 does NOT clear 30% — measured 43.2% (43.5% with feed_model=
# "point", so the feed model is not the cause). This is a real #249 §0 item
# 2 finding, not a bug: at Δ/a = 6.1 the bspline-vs-nec2c EK-OFF gap is
# already 1.87% of |Z| (G13 below) against a nec2c EK shift of only 0.59%
# of |Z| — the basis gap this rung's SHIFT was supposed to cancel is
# *larger* than the shift signal itself, unlike a = 0.05 where the shift
# (2.74%) dominates the basis gap (2.42%). The shift form only escapes the
# basis-noise floor #249 §0 item 2 diagnosed once the shift itself is the
# bigger of the two quantities. G11 and G14 below independently confirm the
# same Δ/a-dependent floor on a completely different oracle (SinusoidalSolver
# instead of nec2c), so this is deferred to a per-rung tolerance rather than
# a single 30% bar — see the mismatch column above for what "wide" means
# here: 50%, just above the measured 43.2%, still excludes a "no EK at all"
# answer (that would sit at 100% mismatch: δ_bsp = 0).

_G9_TOL = {0.02: 0.50, 0.05: 0.15}


@pytest.mark.parametrize("radius", list(_G9_TOL))
def test_g9_shift_matches_nec2c(radius):
    da, z_off_nec, z_on_nec = LADDER[radius]
    delta_bsp = _u3_ladder_z(radius, True) - _u3_ladder_z(radius, False)
    delta_nec = z_on_nec - z_off_nec
    mismatch = abs(delta_bsp - delta_nec) / abs(delta_nec)
    assert mismatch <= _G9_TOL[radius], (
        f"Δ/a={da}: δ_bsp={delta_bsp} vs nec2c δ={delta_nec}, "
        f"mismatch {mismatch:.3f} > {_G9_TOL[radius]}"
    )


# ----------------------------------------------------------------------
# Gate 10 — shift direction (mirrors test_monopole_ek_shift_direction_
# matches_nec2c in test_extended_kernel.py)
# ----------------------------------------------------------------------
#
# Adds a = 0.005 to G9's two rungs. Signs match nec2c at all three; the
# magnitude ratio is well inside [0.5, 2] at a = 0.02 (1.43) and a = 0.05
# (0.97), but at a = 0.005 nec2c's own shift is only 0.042% of |Z| — deep in
# the same basis-noise floor G9's comment describes, one order of magnitude
# below the a = 0.02 rung already found wide there — so its ratio is 3.61,
# not something [0.5, 2] can hold. The direction (sign) claim, which does
# not degrade with the noise floor the way the magnitude ratio does, is
# still gated at all three rungs unconditionally.

_G10_RADII = [0.02, 0.05, 0.005]
_G10_RATIO_BOUNDS = {
    0.02: (0.5, 2.0),
    0.05: (0.5, 2.0),
    0.005: (0.3, 4.2),  # measured ratio 3.61 — see the comment above
}


@pytest.mark.parametrize("radius", _G10_RADII)
def test_g10_shift_direction_matches_nec2c(radius):
    da, z_off_nec, z_on_nec = LADDER[radius]
    delta_bsp = _u3_ladder_z(radius, True) - _u3_ladder_z(radius, False)
    delta_nec = z_on_nec - z_off_nec
    assert delta_bsp.real * delta_nec.real > 0, f"{delta_bsp} vs {delta_nec}"
    assert delta_bsp.imag * delta_nec.imag > 0, f"{delta_bsp} vs {delta_nec}"
    ratio = abs(delta_bsp) / abs(delta_nec)
    lo, hi = _G10_RATIO_BOUNDS[radius]
    assert lo < ratio < hi, f"Δ/a={da}: ratio {ratio:.3f} not in ({lo}, {hi})"


# ----------------------------------------------------------------------
# Gate 11 — cross-basis shift vs SinusoidalSolver(extended_kernel=True)
# ----------------------------------------------------------------------
#
# `SinusoidalSolver` is segment-fed by construction (its `feed_model`
# defaults to `"segment"` and it REFUSES `"point"` —
# `_reject_point_feed_model`), so the bspline side uses `feed_model=
# "segment"` here too, matching both G9's choice and the oracle's own.
#
# #249's design picked "a in {0.02, 0.01, 0.005}, NS with Δ/a >= 2" without
# pinning NS per radius. Measured at NS = 41 fixed (Δ/a = 6.1 / 12.2 / 24.4
# for a = 0.02 / 0.01 / 0.005): the mismatch is 14.4% at a = 0.02 but 61.3%
# at a = 0.01 and 220.9% at a = 0.005 — i.e. Δ/a >= 2 alone is not
# sufficient, exactly the G9 floor (shift signal vs basis-gap noise) again,
# now on a THIRD independent oracle pairing. Holding Δ/a FIXED at 6.098
# instead (NS = 41 / 82 / 164 for a = 0.02 / 0.01 / 0.005) gives a mismatch
# that is flat across all three radii regardless of the absolute a:
#
#   a       NS    Δ/a      δ_bsp             δ_sin             mismatch
#   0.02    41    6.098    -0.354-0.795j     -0.409-0.930j     14.43%
#   0.01    82    6.098    -0.171-0.467j     -0.196-0.546j     14.30%
#   0.005   164   6.098    -0.082-0.267j     -0.094-0.312j     14.22%
#
# confirming the mismatch is a function of Δ/a, not of a itself — the
# honest reading of #249's "NS with Δ/a >= 2" is "NS chosen to land in the
# window where the shift is measurable", not "NS = 41 always". Gated at 18%
# (~1.25x the measured ~14.3%, tighter than the design's blanket 30% since
# the real, now-implemented kernel is far more consistent here than the
# radius-perturbation proxy #249 §7.2 built the 30% bar from).

_G11_MESH = {0.02: 41, 0.01: 82, 0.005: 164}  # all Δ/a = 6.098
_G11_TOL = 0.18


@functools.lru_cache(maxsize=None)
def _u3_sin_z(radius, ns, extended_kernel):
    z, _ = SinusoidalSolver(
        wires=[_dipole_wire(LEN / 2)],
        n_per_edge_per_wire=[[ns]],
        wavelength=LAM,
        wire_radius=radius,
        nsegs=ns,
        feed_arclength=LEN / 2,
        extended_kernel=extended_kernel,
    ).compute_impedance()
    return z


@pytest.mark.parametrize("radius", list(_G11_MESH))
def test_g11_cross_basis_shift_matches_sinusoidal(radius):
    ns = _G11_MESH[radius]
    delta_bsp = _u3_ladder_z(radius, True, ns=ns) - _u3_ladder_z(radius, False, ns=ns)
    delta_sin = _u3_sin_z(radius, ns, True) - _u3_sin_z(radius, ns, False)
    mismatch = abs(delta_bsp - delta_sin) / abs(delta_sin)
    assert mismatch <= _G11_TOL, (
        f"a={radius} NS={ns}: δ_bsp={delta_bsp} vs sinusoidal δ={delta_sin}, "
        f"mismatch {mismatch:.3f} > {_G11_TOL}"
    )


# ----------------------------------------------------------------------
# Gate 12 — no-op on ordinary wire
# ----------------------------------------------------------------------
#
# #249's own radii (a in {0.005, 0.001} — NOT test_extended_kernel.py's
# {0.0048780, 0.001}), same ladder deck. Measured: 1.53e-3 at a = 0.005,
# 2.32e-4 at a = 0.001. The sinusoidal file's "no-op" bar is 1e-3, pinned to
# NEC's OWN EK shift there (<=0.04%); it does not transfer, because this is
# a same-EDGE Galerkin moment correction, not a per-end collocation one, and
# unit 1's own moment-level table (#249 §3) already shows the same-edge
# diagonal moment moves 0.34% at h/a = 24 — an order above nec2c's number,
# consistent with what is measured here at the engine level (Δ/a = 24.39,
# a = 0.005 rung). a = 0.001 (Δ/a = 121.95, comfortably in the "hundreds"
# ordinary-HF-wire band) clears 1e-3 with room to spare.

_G12_TOL = {0.005: 2e-3, 0.001: 5e-4}


@pytest.mark.parametrize("radius", list(_G12_TOL))
def test_g12_extended_kernel_is_a_no_op_on_ordinary_wire(radius):
    z_off = _u3_ladder_z(radius, False)
    z_on = _u3_ladder_z(radius, True)
    rel = abs(z_on - z_off) / abs(z_off)
    assert rel <= _G12_TOL[radius], f"a={radius}: {rel:.3e} > {_G12_TOL[radius]:.3e}"


# ----------------------------------------------------------------------
# Gate 13 — EK does not break the reduced answer (the control for G9)
# ----------------------------------------------------------------------
#
# EK-OFF is byte-identical pre- vs post-#249 (unit 2's G7 armor), so this is
# a re-measurement of the SAME pre-#249 Δ/a-floor tolerances the class
# docstring already states (Δ/a >= 24: <=0.7%, Δ/a >= 6: <=2.1%,
# Δ/a >= 2.4: <=2.9%), not a new claim — it is what makes G9's shift a
# measurement of the kernel rather than of a coincidentally-changed EK-off
# path. Measured (feed_model="segment", matching G9/G11):
#
#   a           Δ/a       rel vs nec2c EK-OFF
#   0.001       121.951   0.196%
#   0.0048780    25.000   0.471%
#   0.005        24.390   0.490%
#   0.02          6.098   1.872%
#   0.05          2.439   2.416%

_G13_TOL = {
    0.001: 0.007,
    0.0048780: 0.007,
    0.005: 0.007,
    0.02: 0.021,
    0.05: 0.029,
}


@pytest.mark.parametrize("radius", list(_G13_TOL))
def test_g13_ek_off_still_matches_nec2c_ek_off(radius):
    da, z_off_nec, _z_on_nec = LADDER[radius]
    z_off = _u3_ladder_z(radius, False)
    rel = abs(z_off - z_off_nec) / abs(z_off_nec)
    assert rel <= _G13_TOL[radius], f"Δ/a={da}: {z_off} vs nec2c EK-OFF {z_off_nec}"


# ----------------------------------------------------------------------
# Gate 14 — PEC-ground parity (the mirrored-source eligibility, end to end)
# ----------------------------------------------------------------------
#
# Same geometry convention as test_extended_kernel.py's Gate 9 monopole
# (base at the origin, tip at z = H, fed at the base segment, PEC ground at
# z = 0) but with the radius chosen for Δ/a >= 2: that file's own fixture
# (a = 0.09, NS = 21) sits at Δ/a = 1.32, below the usable floor, so it is
# not reused verbatim. a = 0.02 at the same NS = 21 gives Δ/a = 5.95 — the
# same regime G11 measured a flat ~14% mismatch at on the free-space deck.
# Measured here: δ_bsp = -0.156-0.412j, δ_sin = -0.185-0.487j, mismatch =
# 15.3%. Gated at 20% (~1.3x measured) rather than the design's blanket
# 30%: this is the one gate that reaches the mirrored-source path end to
# end, and the mutation probe (see the unit's final report) found it far
# more decisive against a wrong EK coefficient than the free-space rungs
# above — a coefficient bug that a 30% bar would miss on G9/G11 moved this
# gate's mismatch to 96.5%.

MONO_H = LAM / 4
MONO_NS = 21
MONO_A = 0.02
MONO_FEED = (MONO_H / MONO_NS) / 2
_G14_TOL = 0.20


@functools.lru_cache(maxsize=None)
def _u3_monopole_z(cls, extended_kernel):
    kw = dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_H]])],
        n_per_edge_per_wire=[[MONO_NS]],
        wavelength=LAM,
        wire_radius=MONO_A,
        nsegs=MONO_NS,
        ground_z=0.0,
        feed_arclength=MONO_FEED,
        extended_kernel=extended_kernel,
    )
    if cls is BSplineSolver:
        kw.update(degree=2, feed_model="segment")
    z, _ = cls(**kw).compute_impedance()
    return z


def test_g14_pec_ground_shift_matches_sinusoidal():
    delta_bsp = _u3_monopole_z(BSplineSolver, True) - _u3_monopole_z(
        BSplineSolver, False
    )
    delta_sin = _u3_monopole_z(SinusoidalSolver, True) - _u3_monopole_z(
        SinusoidalSolver, False
    )
    mismatch = abs(delta_bsp - delta_sin) / abs(delta_sin)
    assert mismatch <= _G14_TOL, (
        f"δ_bsp={delta_bsp} vs sinusoidal δ={delta_sin}, "
        f"mismatch {mismatch:.3f} > {_G14_TOL}"
    )


# ======================================================================
# momwire#269 — FINITE GROUND under the extended kernel
# ======================================================================
#
# #249 refused `extended_kernel=True` + `ground_eps` on both models with one
# boundary line, because the refl-coef image would have ridden the EK-aware
# moment blocks while the Sommerfeld remainder stayed reduced — "probably
# harmless, but a claim with no gate behind it". These are the gates.
#
# WHAT ACTUALLY CHANGED IN THE SOLVER, and therefore what is worth gating:
#
# * The refl-coef image needed NO new physics on the dense path. Its Fresnel
#   dyad / image-charge tables are per-segment-pair WEIGHTS the assembler
#   applies after the moment fill (`_accumulate_Z_image_chunked` hands them
#   to `assemble_Z_bspline_weighted_windowed`), so the moment kernel choice
#   is orthogonal to them and the #270 unit-2 EK twins already served it.
#   The H-matrix's FUSED far-block assembler is the one place where the two
#   are welded into a single C++ kernel, so that one needed a new twin —
#   `bspline_assemble_offedge_block_refl_ek`, gated in U4 below.
# * The Sommerfeld remainder (`_Z_sommerfeld_remainder`,
#   `_sommerfeld_global_lowrank`) deliberately stays REDUCED. G18 measures
#   what that costs instead of assuming it.
#
# THE ORACLE is `SinusoidalSolver(extended_kernel=True)` + `ground_eps`,
# fully C++-served for both models since momwire#259 — G14's PEC-ground
# parity gate with a ground constant added. Gate shape is G11/G14's: the
# cross-basis comparison is of the SHIFT δZ = Z(EK on) − Z(EK off), because
# at any one mesh the absolute bspline-vs-sinusoidal basis gap is the same
# size as the EK shift itself, so only the shift measures the kernel.
#
# THE CONTROL is the same deck's PEC-ground mismatch, and it carries more
# weight than the absolute bar. Every deck below has a cross-basis EK shift
# mismatch that is a property of the DECK (14% on a straight wire, 44% at a
# bend — #249 §4.3: this Galerkin fill declines to extend cross-arm pairs
# where NEC still extends them). Adding a ground constant must not move
# that number. Measured, it moves it by at most 0.0073 absolute.

_G16_EPS = {"soil": (13.0, 0.005), "sea": (81.0, 5.0)}
_G16_H = LAM / 4


def _g16_decks():
    """Four grounded decks, each at Δ/a ≥ 2.4 and fed at a segment CENTRE so
    both bases excite the same cell (the sinusoidal side is segment-fed by
    construction; the bspline side uses `feed_model="segment"` to match, as
    G11/G14 do)."""
    ns = 21
    bend = np.array([[0.0, 0.0, 0.3], [2.0, 0.0, 1.5], [3.6, 1.1, 1.5]])
    h_bend = float(np.linalg.norm(bend[1] - bend[0])) / 6
    return {
        # A quarter-wave monopole STANDING IN the plane — the ground-contact
        # branch (NEC's IND = 0), and G14's own deck with ε̃ added.
        "mono_contact": dict(
            wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, _G16_H]])],
            n_per_edge_per_wire=[[ns]],
            nsegs=ns,
            wire_radius=0.02,
            feed_arclength=(_G16_H / ns) * 0.5,
            ground_z=0.0,
        ),
        # The same wire LIFTED clear of the plane: the image is a real
        # distance away and both ends are ordinary IND = 1.
        "mono_lifted": dict(
            wires=[np.array([[0.0, 0.0, 0.4], [0.0, 0.0, 2.9]])],
            n_per_edge_per_wire=[[ns]],
            nsegs=ns,
            wire_radius=0.02,
            feed_arclength=(2.5 / ns) * 10.5,
            ground_z=0.0,
        ),
        # Horizontal at height: the wire whose image is PARALLEL, not
        # coaxial, so the joint mirror labelling must decline it — and the
        # deck where the Fresnel dyad's −(ρ_v+ρ_h)(t·p̂) half carries real
        # weight rather than cancelling.
        "horizontal": dict(
            wires=[np.array([[-2.5, 0.0, 1.5], [2.5, 0.0, 1.5]])],
            n_per_edge_per_wire=[[ns]],
            nsegs=ns,
            wire_radius=0.05,
            feed_arclength=(5.0 / ns) * 10.5,
            ground_z=0.0,
        ),
        # Bent above the plane: two arms, a p̂ that rotates pair by pair, and
        # the fat radius (Δ/a = 4.9) that makes the EK shift large.
        "bent": dict(
            wires=[bend],
            n_per_edge_per_wire=[[6, 5]],
            nsegs=11,
            wire_radius=0.08,
            feed_arclength=h_bend * 3.5,
            ground_z=0.0,
        ),
    }


_G16_DECKS = _g16_decks()

# Ground pairings, as (ground_model, eps_name) -> solver kwargs. "pec" is
# the control: no ground constant at all.
_G16_GROUNDS = {"pec": {}}
for _m in ("refl-coef", "sommerfeld"):
    for _e, _v in _G16_EPS.items():
        _G16_GROUNDS[f"{_m} {_e}"] = dict(ground_eps=_v, ground_model=_m)


@functools.lru_cache(maxsize=None)
def _g16_z(deck, ground, cls, ek):
    kw = dict(_G16_DECKS[deck], wavelength=LAM, extended_kernel=ek)
    kw.update(_G16_GROUNDS[ground])
    if cls is BSplineSolver:
        kw.update(degree=2, feed_model="segment")
    z, _ = cls(**kw).compute_impedance()
    return complex(z)


def _g16_mismatch(deck, ground):
    """|δ_bsp − δ_sin| / |δ_sin| on one (deck, ground) pairing."""
    d_bsp = _g16_z(deck, ground, BSplineSolver, True) - _g16_z(
        deck, ground, BSplineSolver, False
    )
    d_sin = _g16_z(deck, ground, SinusoidalSolver, True) - _g16_z(
        deck, ground, SinusoidalSolver, False
    )
    return abs(d_bsp - d_sin) / abs(d_sin), d_bsp, d_sin


# ----------------------------------------------------------------------
# Gate 16 — the parity table, over every ground the oracle can serve
# ----------------------------------------------------------------------
#
# Measured on this box (mismatch |δ_bsp − δ_sin|/|δ_sin|):
#
#   deck            PEC      refl soil  refl sea  somm soil  somm sea
#   mono_contact    0.1532   withdrawn  withdrawn 0.0808     0.1529
#   mono_lifted     0.0738   0.0715     0.0736    0.0721     0.0737
#   horizontal      0.2769   0.2829     0.2773    0.2792     0.2770
#   bent            0.4389   0.4444     0.4391    0.4463     0.4392
#
# Read the columns, not the rows: every deck answers the SAME number under
# every ground it can be measured under. The row-to-row spread is the
# deck's own cross-basis EK gap (G11's ~14% class on a straight wire, more
# at a bend where this fill is deliberately more conservative than NEC —
# #249 §4.3); the column-to-column spread is what finite ground adds, and
# it is ≤ 0.0073 absolute. G16b gates that directly and is the stronger
# statement; the absolute bars here are #249 §7.2's 0.30 where the deck
# meets it in PEC too, and the deck's own PEC value + margin where it does
# not.
#
# The contact row is the whole of momwire#282 and momwire#292, and it used
# to have three cells excluded outright. A wire end lying in the plane left
# an uncancelled point charge there (the image carries only ρ of the wire
# current, so the two end charges no longer cancel) and the sinusoidal
# answer walked away under refinement. #282 subtracted that charge on the
# REDUCED kernel's end-charge bracket and EK-OFF settled; EK-ON kept
# walking — 32.30+38.50j → 21.28+116.34j over NS = 11/21/41, spread 1.56 —
# because an EK-on fill carries EKSCX's bracket instead, an O(a²/R²) ≈ 10%
# different quantity at these meshes. #292 builds the subtraction from the
# same per-end GXX quantities the fill uses, and the oracle converges with
# EK on. Measured, a = 0.02, feed at the base segment, NS = 11/21/41:
#
#   ground            sinusoidal EK-OFF          EK-ON             spreads
#   refl-coef soil    33.67→29.54 +21j    33.64→29.76 +20j     0.129 / 0.127
#   sommerfeld soil   53.64→52.84 +26j    53.59→52.70 +25j     0.020 / 0.027
#
# so all five contact cells were measured. `test_g16c_*` pins those ladders
# directly, on both kernels, so neither half can rot.
#
# momwire#282 stage 1 (2026-08-18) WITHDREW the two `mono_contact` x
# refl-coef cells: ground contact under `ground_model="refl-coef"` is
# refused at construction now, so those two decks cannot be built.
# `docs/design/contact-over-finite-ground.md` §3.6 is why — that row sat
# 27 Ω from the Sommerfeld answer on the same deck.
#
# It removed one piece of machinery with them, and the reason is worth
# keeping. `refl soil` was the only pairing whose RATIO was not a usable
# statistic: over average soil the reflection-coefficient ground very nearly
# cancelled the sinusoidal solver's EK shift in the real part (δ_sin =
# −0.026 − 0.309j at NS = 21, against −0.185 − 0.487j under PEC), so |δ_sin|
# was a small denominator, the ratio inflated to 0.456, and the cell had to
# be scored on |δ_bsp − δ_sin| = 0.141 instead. That small denominator was
# itself a symptom: the ground whose EK shift nearly cancels the other
# trunk's at a contact is the ground that is not modelling the contact. With
# the cell gone the whole table is scored on one metric again.

# Absolute bar per deck: #249 §7.2's 0.30 where the deck's own PEC control
# meets it, the PEC value + ~15% headroom where it does not.
_G16_TOL = {
    "mono_contact": 0.30,
    "mono_lifted": 0.30,
    "horizontal": 0.30,
    "bent": 0.50,
}

_G16_FINITE = [g for g in _G16_GROUNDS if g != "pec"]

_G16_CASES = [
    (deck, ground)
    for deck in _G16_DECKS
    for ground in _G16_FINITE
    if not (deck == "mono_contact" and ground.startswith("refl-coef"))
]


def _g16_score(deck, ground):
    """The number G16/G16b gate on this pairing: |δ_bsp − δ_sin|/|δ_sin|."""
    ratio, d_bsp, d_sin = _g16_mismatch(deck, ground)
    return ratio, d_bsp, d_sin


@pytest.mark.parametrize("deck,ground", _G16_CASES)
def test_g16_finite_ground_shift_matches_sinusoidal(deck, ground):
    score, d_bsp, d_sin = _g16_score(deck, ground)
    tol = _G16_TOL[deck]
    assert score <= tol, (
        f"{deck} / {ground}: δ_bsp={d_bsp} vs sinusoidal δ={d_sin}, "
        f"mismatch {score:.4f} > {tol}"
    )


# ----------------------------------------------------------------------
# Gate 16b — and the ground constant did not move that mismatch
# ----------------------------------------------------------------------
#
# The gate the headline claim actually rests on. A deck's cross-basis EK
# mismatch is a property of its geometry and mesh; if the finite-ground
# image were extended differently on the two sides (wrong eligibility, a
# weight applied before the kernel instead of after, the reduced refl
# assembler silently serving an EK block), this number would move. Measured
# worst |mismatch_ground − mismatch_PEC| over the 12 non-contact pairings:
# 0.0073 (bent / sommerfeld soil). Gated at 0.03, ~4x the worst.
#
# The two surviving contact pairings get their own looser bound. Their δ
# real parts are small (−0.16 bspline, −0.30 sinusoidal on sommerfeld sea),
# so the ratio is noisy even where the vectors agree — measured 0.0724
# (somm soil), 0.0003 (somm sea) against a 0.153 PEC control. The two
# refl-coef contact cells that used to sit beside them (0.0336 and, scored
# absolutely, 0.0616) went with momwire#282 stage 1's withdrawal.

_G16B_TOL = 0.03
_G16B_TOL_CONTACT = 0.15


@pytest.mark.parametrize("deck,ground", _G16_CASES)
def test_g16b_finite_ground_does_not_move_the_pec_mismatch(deck, ground):
    mm_g = _g16_score(deck, ground)[0]
    mm_pec = _g16_score(deck, "pec")[0]
    tol = _G16B_TOL_CONTACT if deck == "mono_contact" else _G16B_TOL
    assert abs(mm_g - mm_pec) <= tol, (
        f"{deck} / {ground}: mismatch {mm_g:.4f} vs PEC control {mm_pec:.4f} "
        f"— the ground constant moved the cross-basis EK gap by "
        f"{abs(mm_g - mm_pec):.4f} > {tol}"
    )


# ----------------------------------------------------------------------
# Gate 16c — the contact ladders both kernels now converge on
# ----------------------------------------------------------------------

_G16C_NS = (11, 21, 41)


def _g16c_contact_z(cls, ns, ek=False, **ground):
    kw = dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, _G16_H]])],
        n_per_edge_per_wire=[[ns]],
        nsegs=ns,
        wire_radius=0.02,
        feed_arclength=(_G16_H / ns) * 0.5,
        ground_z=0.0,
        wavelength=LAM,
        extended_kernel=ek,
        **ground,
    )
    if cls is BSplineSolver:
        kw.update(degree=2, feed_model="segment")
    return complex(cls(**kw).compute_impedance()[0])


def _g16c_spread(cls, **ground):
    z = [_g16c_contact_z(cls, ns, **ground) for ns in _G16C_NS]
    return abs(z[-1] - z[0]) / abs(z[0])


def _g16c_spread_ek(cls, **ground):
    z = [_g16c_contact_z(cls, ns, ek=True, **ground) for ns in _G16C_NS]
    return abs(z[-1] - z[0]) / abs(z[0])


def test_g16c_the_contact_oracle_converges_on_both_kernels():
    """The contact row of G16, as a measurement rather than a tolerance —
    and the armor under momwire#282 and momwire#292 both.

    Over average soil at ground CONTACT the sinusoidal oracle used to walk
    away under mesh refinement at ~100% over NS = 11 → 41, EK on or off,
    which is why three pairings were excluded from G16/G16b outright. #282
    fixed the EK-OFF half — the uncancelled contact-node charge — taking it
    to 0.13 (refl-coef) / 0.02 (Sommerfeld) against this solver's 0.04 and
    the PEC controls' 0.03-0.04. #292 fixed the EK-ON half, which was still
    at 1.56 (refl-coef) / 0.42 (Sommerfeld) because #282's subtraction
    cancelled the REDUCED kernel's end-charge bracket while an EK-on fill
    carries EKSCX's; built from the same per-end GXX quantities it lands at
    0.127 / 0.027, i.e. ON the EK-off numbers rather than merely near them.

    Both halves are pinned here because both are load-bearing for G16: δ_sin
    on the contact row is an EK-vs-no-EK DIFFERENCE, so an instability in
    either kernel is an instability in the gated quantity. Reverting either
    fix fails this test long before it fails a tolerance elsewhere — the
    EK-on bars are set at 2x the measured EK-off spread, which the pre-#292
    numbers miss by an order of magnitude.

    The refl-coef half of the loop went with momwire#282 stage 1's
    withdrawal (2026-08-18) — the deck cannot be built. The historical
    numbers above are left in place because they are the record of what #282
    and #292 fixed, and that record does not become untrue when the ground
    stops being served. Note in passing what it says: the refl-coef contact
    ladder converged at 0.129 the whole time, six times worse than the
    Sommerfeld one's 0.020 and stable at it. Converging is not agreeing.
    """
    soil = dict(ground_eps=_G16_EPS["soil"])
    somm = dict(soil, ground_model="sommerfeld")
    assert _g16c_spread(BSplineSolver) < 0.05, "the bspline PEC control is unstable"
    assert _g16c_spread(SinusoidalSolver) < 0.05, (
        "the sinusoidal PEC control is unstable"
    )
    assert _g16c_spread(BSplineSolver, **somm) < 0.10, (
        "the bspline contact answer over soil stopped converging"
    )
    for name, ground in (("sommerfeld", somm),):
        # #282: the EK-OFF oracle converges, within 4x of this solver's own
        # spread. Measured 0.129 / 0.020.
        off = _g16c_spread(SinusoidalSolver, **ground)
        assert off < 0.20, (
            f"the EK-OFF sinusoidal contact answer over {name} soil regressed "
            f"to spread {off:.3f} — momwire#282 is undone"
        )
        # #292: and so does the EK-ON one, to within 2x of it. Measured
        # 0.127 / 0.027 against pre-#292's 1.565 / 0.423.
        on = _g16c_spread_ek(SinusoidalSolver, **ground)
        assert on < 2.0 * off, (
            f"the EK-ON sinusoidal contact answer over {name} soil is at "
            f"spread {on:.3f} against EK-OFF's {off:.3f} — momwire#292's "
            f"EKSCX end-charge bracket is undone"
        )


# ----------------------------------------------------------------------
# Gate 17 — the family agrees with itself over finite ground under EK
# ----------------------------------------------------------------------
#
# G15's fat-deck cross-solver gate, with the two finite grounds. The dense
# path, the H-matrix (near blocks via `_zblock_image_refl`, far blocks via
# the fused refl twin) and the Sommerfeld global low-rank term are three
# different assemblies of the same operator; under EK they must still land
# on one answer. Measured worst on this box: 1.3e-7 (sommerfeld soil — the
# global remainder's own ACA at tol 1e-9 is the floor), 1e-13 class on the
# refl-coef pairings, which have no low-rank term.
#
# `leaf_size` is a PARAMETER, not a detail. This deck is 11 segments, so at
# the default 32 the whole mesh is one leaf and the H-matrix is all near
# blocks — the far-block dispatch (and therefore the fused refl EK twin)
# never runs, and a mutation that sent it to the reduced refl assembler
# would leave this gate green. leaf_size = 4 gives two admissible blocks
# and closes that hole; the mutation run for #269 confirms it (the reduced-
# assembler dispatch mutation fails this gate only in the leaf_size = 4
# rungs).

_G17_TOL = 1e-6


@pytest.mark.parametrize("leaf_size,want_far", [(32, False), (4, True)])
@pytest.mark.parametrize("ground", _G16_FINITE)
def test_g17_hmatrix_agrees_with_the_dense_path_over_finite_ground(
    ground, leaf_size, want_far
):
    kw = dict(
        _G16_DECKS["bent"],
        wavelength=LAM,
        degree=2,
        feed_model="segment",
        extended_kernel=True,
        **_G16_GROUNDS[ground],
    )
    z_dense, _ = BSplineSolver(**kw).compute_impedance()
    sim = HMatrixSolver(**kw, aca_tol=1e-9, aca_leaf_size=leaf_size)
    assert (len(sim.build_partition()["far"]) > 0) == want_far
    z_hm, _ = sim.compute_impedance()
    rel = abs(z_hm - z_dense) / abs(z_dense)
    assert rel <= _G17_TOL, (
        f"{ground} / leaf {leaf_size}: dense {z_dense} vs H-matrix {z_hm} ({rel:.2e})"
    )


# ----------------------------------------------------------------------
# Gate 18 — the Sommerfeld remainder stays reduced, and that is measured
# ----------------------------------------------------------------------
#
# THE CLAIM #249 refused to ship ungated. The remainder Q (theory manual
# eqs 143-147) is the smooth ground-wave correction left after the C2-scaled
# exact image; the image is extended under EK and Q is not.
#
# THE ARITHMETIC. EK is the azimuthal average of the source over a tube of
# radius a, an O((a/R)²) correction where R is the source-observer distance
# — precisely, |fac − 1| ≈ (a²/2R²)|1 + jkR| for a ≪ R. Every term in Q has
# the ground reflection as its source, so its R is the IMAGE distance
# r1 = √(ρ² + (z+z')²) ≥ 2h for a wire at height h: the un-applied
# correction is O((a/2h)²). Measured min r1 over the remainder quadrature
# confirms the ≥ 2h floor is tight (mono_lifted 0.827 vs 2h = 0.8;
# horizontal 3.000 vs 3.000; bent 0.645 vs 0.600).
#
# THE MEASUREMENT, because the arithmetic alone is an estimate and the
# estimate degenerates at ground contact (r1 → 0, where the (a/2h)² form
# says nothing and only the SMOOTHNESS of Q — the whole point of NEC's
# decomposition, the C2 image absorbs the singular part — bounds it). So
# this gate builds the extended remainder outright: the same
# `remainder_field_proj`, evaluated on a ring of `n_phi` source points at
# radius a about each source axis point and averaged, which IS the tube
# average EK applies. The cost of shipping the reduced one is then the
# difference in the solved Z:
#
#   deck            a       min r1    2h      |ΔZ|/|Z| soil   sea
#   mono_contact    0.020   0.0268    0.000   3.03e-03    3.83e-03
#   mono_lifted     0.020   0.8268    0.800   9.60e-07    9.30e-08
#   horizontal      0.050   3.0000    3.000   3.94e-05    4.15e-06
#   bent            0.080   0.6451    0.600   1.09e-04    2.12e-05
#
# Converged in n_phi (4, 8 and 16 agree to the digits shown), and O(a²) as
# the expansion says — on mono_contact/soil, a = 0.04/0.02/0.01/0.005 gives
# 7.89e-3 / 3.03e-3 / 9.29e-4 / 2.51e-4, ratios 2.6 / 3.3 / 3.7 → 4.
#
# HOW TO READ IT. For any wire clear of the plane the un-modelled term is
# ≤ 1.1e-4 relative — three orders below the EK shift the image blocks DO
# carry on the same decks (3.6e-2 on bent) and two orders below this
# basis's own accuracy at Δ/a ≥ 2 (~3%, class docstring). AT GROUND CONTACT
# it is 3-4e-3, which is ~45% of that deck's own EK shift (7.1e-3) — the
# one place the mixture is visible. It is still an order below the basis
# error, and refl-coef is invalid at contact (#153) so Sommerfeld is the
# only model there; refusing it would cost the consumer more than 0.3% on
# |Z|. Recorded, gated, and named in the class docstring rather than
# hidden.

_G18_TOL = {
    "mono_contact": 6e-3,
    "mono_lifted": 5e-6,
    "horizontal": 1e-4,
    "bent": 3e-4,
}
_G18_NPHI = 8


def _tube_averaged_proj(a, n_phi=_G18_NPHI):
    """`_sommerfeld.remainder_field_proj` with the SOURCE points spread onto
    a ring of radius `a` about the source axis and averaged — NEC Eq 89's
    tube average, applied to the smooth remainder field the shipped code
    leaves on the axis."""
    from momwire import _sommerfeld as _sm

    original = _sm.remainder_field_proj

    def ring(obs, t_obs, src, t_src, gz, k, grid, cancel_flag=0):
        ref = np.tile(np.array([0.0, 0.0, 1.0]), (t_src.shape[0], 1))
        ref[np.abs(t_src[:, 2]) > 0.9] = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(t_src, ref)
        e1 /= np.linalg.norm(e1, axis=1)[:, None]
        e2 = np.cross(t_src, e1)
        total = None
        for i in range(n_phi):
            phi = 2.0 * np.pi * i / n_phi
            s = src + a * (np.cos(phi) * e1 + np.sin(phi) * e2)
            v = original(obs, t_obs, s, t_src, gz, k, grid, cancel_flag)
            total = v if total is None else total + v
        return total / n_phi

    return original, ring


@pytest.mark.parametrize("eps_name", list(_G16_EPS))
@pytest.mark.parametrize("deck", list(_G16_DECKS))
def test_g18_reduced_sommerfeld_remainder_is_negligible(
    monkeypatch, deck, eps_name, request
):
    from momwire import _sommerfeld as _sm

    kw = dict(
        _G16_DECKS[deck],
        wavelength=LAM,
        degree=2,
        feed_model="segment",
        extended_kernel=True,
        ground_eps=_G16_EPS[eps_name],
        ground_model="sommerfeld",
    )
    a = float(np.max(np.atleast_1d(kw["wire_radius"])))
    # The fused C++ remainder never calls `remainder_field_proj`, so the
    # tube average has to be installed on the numpy route — and BOTH sides
    # of the comparison are measured there, so the route is not a variable.
    monkeypatch.delattr(_bs._acc, "sommerfeld_remainder_bspline_Q", raising=False)
    z_axis, _ = BSplineSolver(**kw).compute_impedance()
    original, ring = _tube_averaged_proj(a)
    monkeypatch.setattr(_sm, "remainder_field_proj", ring)
    z_tube, _ = BSplineSolver(**kw).compute_impedance()
    rel = abs(z_tube - z_axis) / abs(z_axis)
    assert rel <= _G18_TOL[deck], (
        f"{deck} / {eps_name}: extending the Sommerfeld remainder moves Z by "
        f"{rel:.3e} > {_G18_TOL[deck]:.3e} — the reduced-remainder mixture is "
        f"no longer negligible on this deck"
    )
    assert original is not ring


def test_g18_the_tube_average_is_a_real_perturbation():
    """The control for G18: with the ring installed at a radius the solve
    can actually feel, the numbers above must MOVE. Otherwise G18 would pass
    on a no-op monkeypatch (e.g. a route change that stopped calling
    `remainder_field_proj` at all) and prove nothing."""
    from momwire import _sommerfeld as _sm

    kw = dict(
        _G16_DECKS["bent"],
        wavelength=LAM,
        degree=2,
        feed_model="segment",
        extended_kernel=True,
        ground_eps=_G16_EPS["soil"],
        ground_model="sommerfeld",
    )
    saved = getattr(_bs._acc, "sommerfeld_remainder_bspline_Q", None)
    if saved is not None:
        delattr(_bs._acc, "sommerfeld_remainder_bspline_Q")
    original, ring = _tube_averaged_proj(0.5)  # 6x the real radius
    _sm.remainder_field_proj = ring
    try:
        z_fat, _ = BSplineSolver(**kw).compute_impedance()
    finally:
        _sm.remainder_field_proj = original
        if saved is not None:
            setattr(_bs._acc, "sommerfeld_remainder_bspline_Q", saved)
    z_axis, _ = BSplineSolver(**kw).compute_impedance()
    rel = abs(z_fat - z_axis) / abs(z_axis)
    assert rel > 1e-3, f"the tube average is not reaching the solve ({rel:.3e})"


# ----------------------------------------------------------------------
# Gate 19 — the refl-coef image routes are EK-wired, and stay wired
# ----------------------------------------------------------------------
#
# The #249/#270 image-wiring pattern (spy for the structure, mutation for
# the numbers) applied to the routes `ground_eps` adds. There are FOUR, and
# they are not the same four the PEC image has:
#
#   1. `_accumulate_Z_image_chunked` with Fresnel weights — the route a
#      default dense grounded solve takes.
#   2. `_ground_finite_Z` / `_build_J_image_blocks` — the no-accelerator
#      tensor fallback (dead on this box unless forced; #273).
#   3. `HMatrixSolver._zblock_image_refl` — every near block of a grounded
#      H-matrix solve.
#   4. The fused far-block `bspline_assemble_offedge_block_refl_ek` — the
#      new C++ twin, gated numerically in U4 below.
#
# The deck is `_IMAGE_DECK`'s (a vertical monopole, coaxial with its own
# image, PLUS a horizontal wire that is NOT) with a ground constant added,
# for the reason its own comment gives: an all-vertical deck cannot tell a
# joint mirror labelling from a free-space one.

_IMAGE_DECK_REFL = dict(_IMAGE_DECK, ground_eps=(13.0, 0.005))


def _refl_image_solver(cls, extended_kernel=True, **over):
    return cls(**_IMAGE_DECK_REFL, extended_kernel=extended_kernel, **over)


def test_g19_refl_chunked_route_fills_with_a_mirrored_ek_spec(offedge_ek_spy):
    """Route 1: the Fresnel-weighted chunked image fill, driven with the
    solver's OWN weight tables (`PotentialGround.weight_tables`) rather
    than the PEC ones the #270 gate above uses, so the route under test is
    the one a `ground_eps` solve takes."""
    sim = _refl_image_solver(BSplineSolver)
    geom = sim._build_geometry()
    supp_seg, polys, _kcl, _wk, _wbg = sim._build_basis_polynomials(geom)
    ground = _potential_ground.potential_ground_for(sim, geom, sim.k, sim.omega)
    w_A, w_Phi = ground.weight_tables()
    Z = np.zeros((supp_seg.shape[0],) * 2, dtype=np.complex128, order="F")
    offedge_ek_spy.clear()
    sim._accumulate_Z_image_chunked(
        Z, geom, sim.k, supp_seg, polys, lambda i0, i1: (w_A[i0:i1], w_Phi[i0:i1])
    )
    assert offedge_ek_spy, "no image fill happened"
    row0 = 0
    for ek in offedge_ek_spy:
        rows = np.arange(row0, row0 + len(ek.group_i))
        _assert_mirrored_spec(ek, sim, rows=rows)
        row0 += len(ek.group_i)
    assert row0 == geom["n_segs_total"], "the observer chunks did not cover the mesh"
    assert np.abs(Z).max() > 0.0, "the weighted image fill produced nothing"


def test_g19_refl_tensor_route_fills_with_a_mirrored_ek_spec(offedge_ek_spy):
    """Route 2 — the no-accelerator fallback, reached explicitly."""
    sim = _refl_image_solver(BSplineSolver)
    geom = sim._build_geometry()
    offedge_ek_spy.clear()
    sim._build_J_image_blocks(geom, sim.k)
    assert len(offedge_ek_spy) == 1
    _assert_mirrored_spec(offedge_ek_spy[0], sim)


def test_g19_hmatrix_refl_near_block_fills_with_a_mirrored_ek_spec(offedge_ek_spy):
    """Route 3."""
    sim = _refl_image_solver(HMatrixSolver, aca_tol=1e-9)
    ctx = sim._context()
    idx = np.arange(ctx["n_basis"], dtype=np.int64)
    offedge_ek_spy.clear()
    sim._zblock_image_refl(idx, idx)
    assert len(offedge_ek_spy) == 1
    seg = np.unique(ctx["supp_seg"][idx].ravel())
    _assert_mirrored_spec(offedge_ek_spy[0], sim, rows=seg, cols=seg)


# --- and the mutation itself, as a numeric gate ------------------------


def _grounded_refl_z(cls, **over):
    return complex(_refl_image_solver(cls, **over).compute_impedance()[0])


@pytest.mark.parametrize("weighted_accel", [True, False])
def test_g19_bspline_refl_ek_uses_the_extended_image(
    request, monkeypatch, weighted_accel
):
    """Routes 1 and 2 numerically: `image_ek_disabled` leaves the free-space
    blocks extended and quietly drops the IMAGE blocks to the reduced
    kernel. Measured move on this deck: 3.1e-3 (chunked) / 3.1e-3 (tensor),
    against the 1e-4 bar #249 set from the accelerated-vs-numpy noise floor
    of the same operator."""
    monkeypatch.setattr(_bs, "_HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL", weighted_accel)
    z_wired = _grounded_refl_z(BSplineSolver)
    request.getfixturevalue("image_ek_disabled")
    z_reduced = _grounded_refl_z(BSplineSolver)
    rel = abs(z_wired - z_reduced) / abs(z_wired)
    assert rel >= _IMAGE_EK_SIGNAL, (
        f"weighted_accel={weighted_accel}: the Fresnel image fill answers the "
        f"same with and without an EK spec (moved {rel:.3e})"
    )


@pytest.mark.parametrize("eta,want_far", [(0.0, False), (1.0, True)])
def test_g19_hmatrix_refl_ek_uses_the_extended_image(request, eta, want_far):
    """Routes 3 and 4: eta = 0 admits nothing so every block is a dense near
    block (`_zblock_image_refl`); eta = 1 gives far blocks and therefore the
    fused refl EK twin."""
    sim = _refl_image_solver(HMatrixSolver, aca_tol=1e-9, aca_eta=eta, aca_leaf_size=8)
    # The partition, not `HMatrix.far`: this deck's clusters are small
    # enough that an admissible block's ACA factors cost as much as the
    # dense block, so `build_hmatrix` stores them densely — but the FILL
    # still went through `_offedge_aca_evaluators`, which is the dispatch
    # under test.
    assert (len(sim.build_partition()["far"]) > 0) == want_far
    hm = dict(aca_tol=1e-9, aca_eta=eta, aca_leaf_size=8)
    z_wired = _grounded_refl_z(HMatrixSolver, **hm)
    request.getfixturevalue("image_ek_disabled")
    z_reduced = _grounded_refl_z(HMatrixSolver, **hm)
    rel = abs(z_wired - z_reduced) / abs(z_wired)
    assert rel >= _IMAGE_EK_SIGNAL, (
        f"eta={eta}: the H-matrix Fresnel image is not EK-wired ({rel:.3e})"
    )


# ----------------------------------------------------------------------
# U4 — the fused refl-coef block assembler's EK twin
# ----------------------------------------------------------------------
#
# `bspline_assemble_offedge_block_refl_ek` is #270 unit 3's twin composed
# with #259's Fresnel tail: the coaxial factor multiplies G pair by pair
# BEFORE the Galerkin contraction, the dyad weights the contracted terms
# AFTER it. Both halves are already gated in their own right, so what U4
# has to say is that the composition is the composition — against the numpy
# route (`_zblock_image_refl`, C++ moments + einsum combine) on the
# discriminating image deck. Measured: dense 3.4e-13, row 3.5e-13, col
# 4.2e-13 — unit 3's own 1e-12 class.
#
# The DISPATCH matters as much as the arithmetic: a build lacking the twin
# must fall back to the numpy closures, never to the reduced
# `bspline_assemble_offedge_block_refl` (which would be silently wrong, not
# merely slow — the same trap `_HAVE_OFFEDGE_BLOCK_EK_ACCEL` already guards
# on the free-space side).

_U4_HAVE_ACCEL = _hm._HAVE_OFFEDGE_BLOCK_REFL_EK_ACCEL
pytestmark_u4 = pytest.mark.skipif(
    not _U4_HAVE_ACCEL, reason="extension built without the #269 fused refl EK twin"
)


def _u4_refl_block(extended_kernel=True):
    sim = HMatrixSolver(**_IMAGE_DECK_REFL, extended_kernel=extended_kernel)
    ctx = sim._context()
    idx = np.arange(ctx["n_basis"], dtype=np.int64)
    return sim, ctx, idx


@pytestmark_u4
def test_u4_fused_refl_ek_block_matches_the_numpy_route():
    sim, ctx, idx = _u4_refl_block()
    row_f, col_f, dense_f = sim._offedge_block_evaluators(
        ctx, idx, idx, sim.k, mirror_J=True, refl=True
    )
    D_ref = sim._zblock_image_refl(idx, idx, k=sim.k)
    scale = float(np.abs(D_ref).max())
    for label, got, ref in (
        ("dense", dense_f(), D_ref),
        ("row", row_f(0), D_ref[0]),
        ("col", col_f(0), D_ref[:, 0]),
    ):
        rel = float(np.abs(np.asarray(got).ravel() - ref.ravel()).max() / scale)
        assert rel <= U3_AGREEMENT, f"{label}: {rel:.3e}"


@pytestmark_u4
def test_u4_fused_refl_ek_block_declines_a_noncoaxial_mirror():
    """The discriminating half: the horizontal wire's image is parallel, not
    coaxial, so the joint mirror labelling must decline it. A twin wired
    with a free-space (`mirror=False`) spec would extend it and land off
    `_zblock_image_refl`'s answer — the rows/cols the horizontal wire owns
    are checked separately so a whole-block norm cannot dilute it."""
    sim, ctx, idx = _u4_refl_block()
    cen = ctx["basis_centroid"]
    horiz = np.flatnonzero(np.abs(cen[:, 1] - 3.0) < 1e-6)
    assert horiz.size, "the discriminating deck degenerated"
    _row, _col, dense_f = sim._offedge_block_evaluators(
        ctx, idx, idx, sim.k, mirror_J=True, refl=True
    )
    D = dense_f()
    D_ref = sim._zblock_image_refl(idx, idx, k=sim.k)
    sub, sub_ref = D[np.ix_(horiz, horiz)], D_ref[np.ix_(horiz, horiz)]
    rel = float(np.abs(sub - sub_ref).max() / np.abs(sub_ref).max())
    assert rel <= U3_AGREEMENT, f"horizontal self-image block: {rel:.3e}"


@pytestmark_u4
def test_u4_the_reduced_refl_assembler_would_have_been_visibly_wrong():
    """The control that gives U4a its teeth: the block the dispatch must NOT
    fall back to. `bspline_assemble_offedge_block_refl` on the same inputs
    (i.e. what an EK solve would have silently got before #269 added the
    twin) differs from the extended answer by ~2e-3 — four orders above the
    1e-12 U4a asserts, so U4a is a signal test."""
    sim, ctx, idx = _u4_refl_block()
    _r, _c, dense_ek = sim._offedge_block_evaluators(
        ctx, idx, idx, sim.k, mirror_J=True, refl=True
    )
    D_ek = dense_ek()
    sim_off = HMatrixSolver(**_IMAGE_DECK_REFL, extended_kernel=False)
    _r0, _c0, dense_red = sim_off._offedge_block_evaluators(
        ctx, idx, idx, sim.k, mirror_J=True, refl=True
    )
    D_red = dense_red()
    rel = float(np.abs(D_ek - D_red).max() / np.abs(D_red).max())
    assert rel > 1e-6, f"the EK twin and the reduced refl assembler agree ({rel:.3e})"


def _u4_count_block_calls(monkeypatch, names):
    calls = dict.fromkeys(names, 0)
    for name in names:
        original = getattr(_hm._acc, name)

        def trip(*a, _n=name, _o=original, **kw):
            calls[_n] += 1
            return _o(*a, **kw)

        monkeypatch.setattr(_hm._acc, name, trip)
    return calls


_U4_BLOCK_KERNELS = (
    "bspline_assemble_offedge_block",
    "bspline_assemble_offedge_block_ek",
    "bspline_assemble_offedge_block_refl",
    "bspline_assemble_offedge_block_refl_ek",
)


@pytestmark_u4
def test_u4_the_twin_is_what_serves_a_grounded_far_block(monkeypatch):
    """The positive control for the dispatch gate below: with the capability
    present, an EK far-block fill over `ground_eps` uses the two EK twins and
    NEITHER reduced assembler. Without this, the "no calls" gate below could
    pass on a build where the far blocks never reached C++ at all."""
    calls = _u4_count_block_calls(monkeypatch, _U4_BLOCK_KERNELS)
    _refl_image_solver(
        HMatrixSolver, aca_tol=1e-9, aca_eta=1.0, aca_leaf_size=8
    ).build_hmatrix()
    assert calls["bspline_assemble_offedge_block_ek"] > 0, "no free-space EK fill"
    assert calls["bspline_assemble_offedge_block_refl_ek"] > 0, "no refl EK fill"
    assert calls["bspline_assemble_offedge_block"] == 0
    assert calls["bspline_assemble_offedge_block_refl"] == 0


@pytestmark_u4
def test_u4_a_build_without_the_twin_uses_the_numpy_closures(monkeypatch):
    """Never the reduced fused assembler. Pinned by counting calls to both
    C++ entry points while the capability flag is off."""
    calls = []
    for name in (
        "bspline_assemble_offedge_block_refl",
        "bspline_assemble_offedge_block_refl_ek",
    ):
        original = getattr(_hm._acc, name)

        def trip(*a, _n=name, _o=original, **kw):
            calls.append(_n)
            return _o(*a, **kw)

        monkeypatch.setattr(_hm._acc, name, trip)

    monkeypatch.setattr(_hm, "_HAVE_OFFEDGE_BLOCK_REFL_EK_ACCEL", False)
    sim = _refl_image_solver(HMatrixSolver, aca_tol=1e-9, aca_eta=1.0, aca_leaf_size=8)
    assert len(sim.build_partition()["far"]) > 0, "nothing was dispatched"
    sim.build_hmatrix()
    assert calls == [], f"a build without the twin called {set(calls)}"


@pytestmark_u4
def test_u4_ek_off_never_reaches_the_refl_ek_twin(monkeypatch):
    calls = []
    original = _hm._acc.bspline_assemble_offedge_block_refl_ek

    def trip(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(_hm._acc, "bspline_assemble_offedge_block_refl_ek", trip)
    _refl_image_solver(
        HMatrixSolver,
        extended_kernel=False,
        aca_tol=1e-9,
        aca_eta=1.0,
        aca_leaf_size=8,
    ).compute_impedance()
    assert calls == []


@pytestmark_u4
def test_u4_refl_ek_twin_aborts_as_solve_aborted():
    """The new kernel polls `cancel_flag`, so its abort must reach callers as
    the shared `SolveAborted` — which means being listed in
    `_accel._CANCELLABLE_KERNELS`. Python-level checkpoints are neutralized
    so the only thing that can observe the tripped token is the C++ poll."""
    from momwire import CancelToken, SolveAborted

    token = CancelToken()
    token.cancel()
    sim = _refl_image_solver(
        HMatrixSolver, cancel=token, aca_tol=1e-9, aca_eta=1.0, aca_leaf_size=8
    )
    sim._checkpoint = lambda: None
    with pytest.raises(SolveAborted):
        sim.compute_impedance()


# ======================================================================
# momwire#270 UNIT 1 — the C++ same-edge extended-kernel twins
# ======================================================================
#
# #249 shipped EK as numpy only: every C++ dispatch in `_bspline_kernels`
# was guarded by `ek is None`, so an EK-on fill paid the numpy penalty.
# #270 adds the accelerated twins. Unit 1 is the SAME-EDGE pair —
# `seg_seg_static_moments_bspline_uniform_ek` and
# `seg_seg_reg_moments_bspline_swept_ek`. Off-edge is unit 2, the fused
# assemblers and the ACA re-enable are unit 3.
#
# WHAT IS AND IS NOT COMPARED. Everything here is a CROSS-BACKEND
# comparison, so nothing here asks for bit equality — that is the finding
# unit 1 of #249 recorded higher up in this file and it applies with
# knobs on: the C++ (q, r) reduction is not the einsum's, and the closed
# forms differ in the last bits between `np.arcsinh` and `std::asinh`.
# Gates are relative-tolerance. Measured worsts on this box are in each
# gate's comment; the tolerances sit ~20x above them.
#
# The one place bit equality IS still demanded is EK-OFF, and it is
# demanded WITHIN one backend: `test_g7*` above pin that a defaulted
# solver and `extended_kernel=False` produce the same bits and enter no EK
# code. Those tests are unmodified by #270 and still pass. What #270 does
# NOT claim is that the reduced C++ kernel's absolute output is unchanged
# against a pre-#270 BUILD: it moves by 1-3 ulp, because D_ek_pq_*_2 gives
# J_static_pq_*_0 a second call site and GCC then inlines those header
# forms differently under `-mfma`. That is measured, bisected and
# explained at the kernel in _accelerators.cpp, and it is deliberately not
# pinned — cross-build bit equality is the same trap as the cross-machine
# one antennaknobs#253 hotfixed.

_U1_STATIC_EK_ACCEL = _bk._HAVE_BSPLINE_STATIC_EK_ACCEL
_U1_REG_EK_ACCEL = _bk._HAVE_BSPLINE_REG_SWEPT_EK_ACCEL
_U1_ACCEL = _U1_STATIC_EK_ACCEL and _U1_REG_EK_ACCEL

# The 1e-13 class this file's own cross-backend note calls for. Measured
# worsts: 4.5e-15 (static), 1.8e-14 (reg, single k), 3.2e-15 (reg, swept).
ACCEL_AGREEMENT = 1e-13

# Solve-level, where the moment differences are amplified by the Z
# inverse. Measured worst 6.9e-13 over the five decks below.
ACCEL_AGREEMENT_SOLVE = 5e-12

pytestmark_u1 = pytest.mark.skipif(
    not _U1_ACCEL, reason="extension built without the #270 same-edge EK twins"
)

# Δ/a ladder: the fat end of the usable window (#248's Δ/a >= 2 floor),
# the middle, and an ordinary wire.
_U1_H = H
_U1_RADII = {"fat d/a=2": _U1_H / 2, "d/a=6": _U1_H / 6, "ordinary d/a=24": _U1_H / 24}
_U1_KS = {"k/4": 0.25 * K, "k": K, "4k": 4.0 * K}
_U1_N = 9
_U1_NQP = 4

# A deliberately uneven edge: the uniform-h Toeplitz fast path must not
# claim it.
_U1_NONUNIFORM = np.array([0.0, 0.1, 0.25, 0.31, 0.5, 0.72])


@pytest.fixture
def numpy_same_edge(monkeypatch):
    """Force the same-edge EK kernels back onto their numpy paths."""
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_STATIC_EK_ACCEL", False)
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_REG_SWEPT_EK_ACCEL", False)


_U1_ACCEL_ENTRY_POINTS = (
    "seg_seg_static_moments_bspline_uniform",
    "seg_seg_static_moments_bspline_uniform_ek",
    "seg_seg_reg_moments_bspline_swept",
    "seg_seg_reg_moments_bspline_swept_ek",
    # The off-edge pair (#270 unit 2): both the reduced symbols (so a
    # dispatch probe can pin they stay idle under EK) and the EK twins.
    "seg_seg_full_moments_bspline",
    "seg_seg_full_moments_bspline_swept",
    "seg_seg_full_moments_bspline_ek",
    "seg_seg_full_moments_bspline_swept_ek",
    # ... and the reduced symbols' TIERED twins (momwire#906/#907). Since the
    # free-space pair-order ladder came on, a reduced off-edge block may be
    # served by these instead, so a probe that counts only the plain symbol
    # can no longer tell "EK stayed idle" from "nothing ran".
    "seg_seg_full_moments_bspline_tiered",
    "seg_seg_full_moments_bspline_cplx_tiered",
)

# The reduced (non-EK) off-edge entries, whichever tier serves the block.
_REDUCED_OFFEDGE_ENTRIES = (
    "seg_seg_full_moments_bspline",
    "seg_seg_full_moments_bspline_swept",
    "seg_seg_full_moments_bspline_tiered",
    "seg_seg_full_moments_bspline_cplx_tiered",
)


class _AccelSpy:
    """Counting proxy over the accelerator module.

    `_bspline_kernels` reaches the extension as `_acc.<name>`, so swapping
    the module-level `_acc` for this counts calls without touching the
    extension object (whose attributes are not all rebindable). Everything
    not in the counted set passes straight through, so the rest of the
    accelerated fill is unaffected.
    """

    def __init__(self, real):
        self._real = real
        self.counts = dict.fromkeys(_U1_ACCEL_ENTRY_POINTS, 0)

    def __getattr__(self, name):
        target = getattr(self._real, name)
        if name not in self.counts:
            return target

        def counted(*args, **kwargs):
            self.counts[name] += 1
            return target(*args, **kwargs)

        return counted


@pytest.fixture
def accel_spy(monkeypatch):
    spy = _AccelSpy(_bk._acc)
    monkeypatch.setattr(_bk, "_acc", spy)
    return spy


def _u1_rel(got, ref):
    return float(np.abs(got - ref).max() / np.abs(ref).max())


def _u1_ends(n=_U1_N, h=_U1_H):
    return np.arange(n + 1) * h


# ----------------------------------------------------------------------
# U1a — the static twin agrees with numpy's D_ek_moment path
# ----------------------------------------------------------------------
#
# Measured relative worsts (degree 2):
#
#   Δ/a      2         6         24
#   static   3.7e-15   4.5e-15   1.9e-15


@pytestmark_u1
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("radius", list(_U1_RADII))
def test_u1_static_ek_cpp_matches_numpy(numpy_same_edge, degree, radius):
    a = _U1_RADII[radius]
    ends = _u1_ends()
    ref = _seg_seg_static_moments(ends, a, degree, ek=WHOLE_BLOCK)  # numpy (fixture)
    _bk._HAVE_BSPLINE_STATIC_EK_ACCEL = _U1_STATIC_EK_ACCEL
    got = _seg_seg_static_moments(ends, a, degree, ek=WHOLE_BLOCK)
    rel = _u1_rel(got, ref)
    assert rel <= ACCEL_AGREEMENT, f"degree {degree}, {radius}: {rel:.3e}"


@pytestmark_u1
def test_u1_static_ek_honours_an_explicit_radius_override(numpy_same_edge):
    """`_EK.a` is not the kernel's own `a`, and the C++ takes it as its own
    argument rather than assuming the two are equal."""
    a = _U1_RADII["d/a=6"]
    spec = _EK(a=0.7 * a, group_i=None, group_j=None)
    ends = _u1_ends()
    ref = _seg_seg_static_moments(ends, a, 2, ek=spec)
    _bk._HAVE_BSPLINE_STATIC_EK_ACCEL = _U1_STATIC_EK_ACCEL
    got = _seg_seg_static_moments(ends, a, 2, ek=spec)
    assert _u1_rel(got, ref) <= ACCEL_AGREEMENT
    # And the override is not being ignored: the two radii disagree.
    plain = _seg_seg_static_moments(ends, a, 2, ek=WHOLE_BLOCK)
    assert _u1_rel(got, plain) > 1e-6


@pytestmark_u1
@pytest.mark.parametrize("degree", [1, 2])
def test_u1_static_ek_leaves_non_uniform_edges_to_numpy(accel_spy, monkeypatch, degree):
    """The Toeplitz kernel exists only for a uniform-h edge — under EITHER
    kernel flavour. A non-uniform edge returns on the dense numpy branch
    above the fast paths, so the dispatch must not reach C++ at all and the
    answer must be bit-identical to the flags-off one."""
    got = _seg_seg_static_moments(
        _U1_NONUNIFORM, _U1_RADII["d/a=6"], degree, ek=WHOLE_BLOCK
    )
    assert accel_spy.counts["seg_seg_static_moments_bspline_uniform_ek"] == 0
    assert accel_spy.counts["seg_seg_static_moments_bspline_uniform"] == 0
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_STATIC_EK_ACCEL", False)
    ref = _seg_seg_static_moments(
        _U1_NONUNIFORM, _U1_RADII["d/a=6"], degree, ek=WHOLE_BLOCK
    )
    assert np.array_equal(got, ref)


# ----------------------------------------------------------------------
# U1b — the reg twin agrees with numpy's `_ek_reg_kernel` path
# ----------------------------------------------------------------------
#
# Measured relative worsts: 1.8e-14 single k, 3.2e-15 swept, over the full
# (degree, Δ/a, k) cross-product below.


@pytestmark_u1
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("radius", list(_U1_RADII))
@pytest.mark.parametrize("kname", list(_U1_KS))
def test_u1_reg_ek_cpp_matches_numpy_single_k(numpy_same_edge, degree, radius, kname):
    a = _U1_RADII[radius]
    k = _U1_KS[kname]
    ends = _u1_ends()
    ref = _seg_seg_reg_moments(ends, a, k, degree, _U1_NQP, ek=WHOLE_BLOCK)
    _bk._HAVE_BSPLINE_REG_SWEPT_EK_ACCEL = _U1_REG_EK_ACCEL
    got = _seg_seg_reg_moments(ends, a, k, degree, _U1_NQP, ek=WHOLE_BLOCK)
    rel = _u1_rel(got, ref)
    assert rel <= ACCEL_AGREEMENT, f"degree {degree}, {radius}, {kname}: {rel:.3e}"


@pytestmark_u1
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("radius", list(_U1_RADII))
def test_u1_reg_ek_cpp_matches_numpy_swept(numpy_same_edge, degree, radius):
    a = _U1_RADII[radius]
    ks = np.array(list(_U1_KS.values()))
    geo = _seg_seg_reg_geometry(_u1_ends(), a, degree, _U1_NQP, ek=WHOLE_BLOCK)
    ref = _seg_seg_reg_moments_from_geometry_swept(geo, ks)
    _bk._HAVE_BSPLINE_REG_SWEPT_EK_ACCEL = _U1_REG_EK_ACCEL
    got = _seg_seg_reg_moments_from_geometry_swept(geo, ks)
    rel = _u1_rel(got, ref)
    assert rel <= ACCEL_AGREEMENT, f"degree {degree}, {radius}: {rel:.3e}"


@pytestmark_u1
def test_u1_reg_ek_cpp_does_serve_a_non_uniform_edge(numpy_same_edge, accel_spy):
    """Unlike the static twin, the reg twin has no uniformity precondition:
    it consumes the precomputed (N·n_qp, N·n_qp) R table, which already
    carries whatever geometry the edge has. Pinned as a positive claim so
    the asymmetry between the two same-edge kernels is not read as an
    oversight in either."""
    a = _U1_RADII["d/a=6"]
    ref = _seg_seg_reg_moments(_U1_NONUNIFORM, a, K, 2, _U1_NQP, ek=WHOLE_BLOCK)
    _bk._HAVE_BSPLINE_REG_SWEPT_EK_ACCEL = _U1_REG_EK_ACCEL
    got = _seg_seg_reg_moments(_U1_NONUNIFORM, a, K, 2, _U1_NQP, ek=WHOLE_BLOCK)
    assert accel_spy.counts["seg_seg_reg_moments_bspline_swept_ek"] == 1
    assert _u1_rel(got, ref) <= ACCEL_AGREEMENT


# ----------------------------------------------------------------------
# U1c — end to end: the same EK-on solve, both same-edge backends
# ----------------------------------------------------------------------
#
# Measured relative worsts on Z:
#
#   dipole fat 6.9e-13 | dipole thin 4.5e-13 | bend 9.4e-16
#   mixed radii 2.2e-15 | PEC ground 1.5e-14 | swept 5.8e-13
#
# The two dipoles are the loose end because they are the two decks whose
# whole matrix IS one same-edge block, so nothing else dilutes the moment
# difference before the Z inverse amplifies it.

_U1_DECKS = {name: dict(kw, extended_kernel=True) for name, kw in _G7_BSPLINE.items()}
_U1_DECKS["dipole thin"] = dict(
    wires=[_dipole_wire(2.5)],
    n_per_edge_per_wire=[[21]],
    wire_radius=0.005,
    extended_kernel=True,
)


@pytestmark_u1
@pytest.mark.parametrize("name", list(_U1_DECKS))
def test_u1_end_to_end_solve_agrees_with_the_numpy_same_edge_path(monkeypatch, name):
    kw = dict(_U1_DECKS[name], wavelength=LAM, degree=2)
    z_cpp, c_cpp = BSplineSolver(**kw).compute_impedance()
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_STATIC_EK_ACCEL", False)
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_REG_SWEPT_EK_ACCEL", False)
    z_npy, c_npy = BSplineSolver(**kw).compute_impedance()
    rel = abs(z_cpp - z_npy) / abs(z_npy)
    assert rel <= ACCEL_AGREEMENT_SOLVE, f"{name}: {z_cpp!r} vs {z_npy!r} ({rel:.3e})"
    assert _u1_rel(c_cpp, c_npy) <= ACCEL_AGREEMENT_SOLVE


@pytestmark_u1
def test_u1_end_to_end_swept_solve_agrees_with_the_numpy_same_edge_path(monkeypatch):
    kw = dict(_U1_DECKS["free space"], wavelength=LAM, degree=2)
    ks = 2 * np.pi / LAM * np.array([0.9, 1.0, 1.1])
    z_cpp = np.asarray(BSplineSolver(**kw).compute_impedance_swept(ks))
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_STATIC_EK_ACCEL", False)
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_REG_SWEPT_EK_ACCEL", False)
    z_npy = np.asarray(BSplineSolver(**kw).compute_impedance_swept(ks))
    assert _u1_rel(z_cpp, z_npy) <= ACCEL_AGREEMENT_SOLVE


# ----------------------------------------------------------------------
# U1d — EK-OFF armor: the twins are unreachable and inert with EK off
# ----------------------------------------------------------------------
#
# `test_g7*` above already pin the numeric and no-EK-code-entered halves
# within a backend, unmodified by #270. These two add what is specific to
# having a second C++ entry point in the file at all.


@pytestmark_u1
@pytest.mark.parametrize("name", list(_G7_BSPLINE))
def test_u1_ek_off_never_reaches_the_ek_entry_points(accel_spy, name):
    """A defaulted (EK-off) solve must call the reduced C++ symbols and
    neither EK one — the dispatch guard, checked at the C++ boundary rather
    than at the numpy helpers `test_g7b` watches."""
    BSplineSolver(**_G7_BSPLINE[name], wavelength=LAM, degree=2).compute_impedance()
    assert accel_spy.counts["seg_seg_static_moments_bspline_uniform_ek"] == 0
    assert accel_spy.counts["seg_seg_reg_moments_bspline_swept_ek"] == 0
    assert accel_spy.counts["seg_seg_reg_moments_bspline_swept"] > 0
    assert accel_spy.counts["seg_seg_static_moments_bspline_uniform"] > 0


@pytestmark_u1
@pytest.mark.parametrize("degree", [1, 2])
def test_u1_static_ek_twin_adds_exactly_d_and_stays_toeplitz(degree):
    """The twin's two structural claims, both checked inside one backend so
    both can be exact where they should be.

    1. It adds D and nothing else: the EK-minus-reduced difference is the
       numpy `D_ek_moment` family on the same corners, over 1/(4π).
    2. D rides the SAME 2N-1 Toeplitz table: every entry on a given
       diagonal of the EK block is bit-identical, which would not survive a
       twin that evaluated D per (i, j) with drifting arguments.

    (There is no zero-radius bit-collapse gate here, unlike `test_g4_...`
    above: on a uniform edge the Δ ∈ {-1, 0, 1} corners make one of D's
    `1/√(a² + ζ²)` terms infinite at a = 0, so the explicit a² prefactor
    gives 0·inf = nan rather than the exact zero it gives off-diagonal.)
    """
    h, a, n = _U1_H, _U1_RADII["d/a=6"], _U1_N
    reduced = _bk._acc.seg_seg_static_moments_bspline_uniform(h, a, n, degree)
    ek = _bk._acc.seg_seg_static_moments_bspline_uniform_ek(h, a, n, degree, a)

    delta = np.arange(-(n - 1), n, dtype=float)
    alpha = np.zeros_like(delta)
    beta = np.full_like(delta, h)
    A = delta * h
    B = (delta + 1.0) * h
    gather = (np.arange(n)[None, :] - np.arange(n)[:, None]) + (n - 1)
    for p in range(degree + 1):
        for q in range(degree + 1):
            want = D_ek_moment(p, q, alpha, beta, A, B, a)[gather] / (4 * np.pi)
            got = ek[p, q] - reduced[p, q]
            assert _u1_rel(got, want) <= ACCEL_AGREEMENT, f"({p}, {q})"

    for d in range(-(n - 1), n):
        diag = np.diagonal(ek, offset=d, axis1=2, axis2=3)
        assert np.all(diag == diag[..., :1]), f"EK block is not Toeplitz at Δ={d}"


# ----------------------------------------------------------------------
# U1e — dispatch probes: EK-on same-edge really is served by C++
# ----------------------------------------------------------------------
#
# Numeric agreement alone would be satisfied by a dispatch that quietly
# never left numpy, so each direction is pinned from both sides: the C++
# entry point is entered, and the numpy closed form it replaces is not.
# The probe is scoped to the SAME-EDGE helpers: `_ek_factor` is still
# entered in this unit, from the off-edge fill that unit 2 will move.


@pytestmark_u1
def test_u1_ek_on_same_edge_is_served_by_cpp(accel_spy, ek_call_counts, monkeypatch):
    # Off-edge gained its own C++ EK twins in #270 unit 2; force them off so
    # this probe still tests what unit 1 built it to test — that the
    # SAME-EDGE twins are what is entered, with the off-edge numpy
    # `_ek_factor` call as the scoping control below. Unit 2's own probes
    # (`test_u2_*`) cover the off-edge C++ path this monkeypatch disables
    # here.
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_EK_ACCEL", False)
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_SWEPT_EK_ACCEL", False)
    # "bend", not "free space": the off-edge `_ek_factor` call below is this
    # probe's scoping control, and a single-edge deck has no off-edge work for
    # it to scope — since momwire#759 the discarded pre-pass is not computed at
    # all, so the control would read zero for a reason that has nothing to do
    # with what this test asserts. "bend" is one wire, two edges, free space:
    # same EK path, genuine cross-edge pairs.
    BSplineSolver(
        **_G7_BSPLINE["bend"],
        wavelength=LAM,
        degree=2,
        extended_kernel=True,
    ).compute_impedance()
    assert accel_spy.counts["seg_seg_static_moments_bspline_uniform_ek"] > 0
    assert accel_spy.counts["seg_seg_reg_moments_bspline_swept_ek"] > 0
    # ... and the numpy same-edge closed forms were not touched.
    assert ek_call_counts["D_ek_moment"] == 0
    assert ek_call_counts["_ek_reg_extra"] == 0
    # The reduced C++ same-edge symbols are idle too: EK-on same-edge
    # blocks go to the EK twins, not to the reduced kernel plus a fixup.
    assert accel_spy.counts["seg_seg_static_moments_bspline_uniform"] == 0
    assert accel_spy.counts["seg_seg_reg_moments_bspline_swept"] == 0
    # Off-edge is forced to numpy above — the control that this probe is
    # scoped, not merely lucky.
    assert ek_call_counts["_ek_factor"] > 0


@pytestmark_u1
def test_u1_missing_symbols_fall_back_to_numpy(numpy_same_edge, accel_spy):
    """Graceful degradation: an extension built before #270 has neither
    twin, the `_HAVE_*` flags are False, and the EK fill is the #249 numpy
    path — same answer, no AttributeError."""
    z, _ = BSplineSolver(
        **_G7_BSPLINE["free space"],
        wavelength=LAM,
        degree=2,
        extended_kernel=True,
    ).compute_impedance()
    assert np.isfinite(z)
    assert accel_spy.counts["seg_seg_static_moments_bspline_uniform_ek"] == 0
    assert accel_spy.counts["seg_seg_reg_moments_bspline_swept_ek"] == 0


# ======================================================================
# momwire#270 UNIT 2 — the C++ OFF-EDGE extended-kernel twins
# ======================================================================
#
# Unit 1 landed the SAME-EDGE pair, where a block is eligible in its
# entirety. Off-edge eligibility is a property of the (i, j) SEGMENT PAIR
# (`group_i[i] == group_j[j]`, momwire#249 §4.1), evaluated once per pair
# and applied to every quadrature sub-pair inside it. The twins here —
# `seg_seg_full_moments_bspline_ek` / `_swept_ek` — take the per-segment
# group labels as int64 arrays in addition to the reduced kernel's own
# contract, and apply NEC Eq 89's coaxial factor pair by pair rather than
# block-wide. The fused H-matrix assemblers and the ACA re-enable are
# unit 3's.
#
# WHAT IS AND IS NOT COMPARED — same rule as unit 1: cross-backend (C++ vs
# numpy) comparisons are relative-tolerance, never bit equality; EK-off
# stays bit-exact within one backend, which `test_g7*` / `test_g7b*` above
# already cover.
#
# ADAPTED TESTS (dispatch rerouted by this unit — both listed here so the
# change is traceable from one place):
#
#   test_g7b_the_counters_fire_when_ek_is_on — was parametrized only over
#     the same-edge backend; now parametrized over the off-edge backend
#     too, because `_ek_factor` stopped being same-edge-exclusive (the
#     off-edge fill's own `ek is not None` branch calls it too, and now has
#     a C++ alternative).
#
#   test_u1_ek_on_same_edge_is_served_by_cpp — forces the new off-edge
#     flags off so its "off-edge is still numpy" scoping control keeps
#     meaning what unit 1 wrote it to mean.
#
# Nothing else needed adapting: every other unit-1/#249 test either does
# not touch the off-edge fill, or already parametrizes over "numpy vs
# whatever the box has" in a way this unit's new C++ path falls naturally
# into (e.g. `test_swept_offedge_moments_with_ek_match_the_per_k_path`,
# which stayed green at its ORIGINAL bit-exact tolerance once both the
# single-k and swept calls started reaching the same C++ backend).

_U2_OFFEDGE_EK_ACCEL = _bk._HAVE_BSPLINE_OFFEDGE_EK_ACCEL
_U2_OFFEDGE_SWEPT_EK_ACCEL = _bk._HAVE_BSPLINE_OFFEDGE_SWEPT_EK_ACCEL
_U2_ACCEL = _U2_OFFEDGE_EK_ACCEL and _U2_OFFEDGE_SWEPT_EK_ACCEL

# Measured worsts on this box (degree in {1, 2}, k in _U2_KS, scalar and
# per-row radius, over the mixed-eligibility fixture below): 3.7e-16 single
# k, 3.7e-16 swept, 1.8e-16 all-eligible — a full order tighter than unit
# 1's same-edge twins (no static/reg split here, so fewer summed terms),
# but the bar stays at the 1e-13 class this file's own cross-backend note
# calls for rather than chasing this box's measurement.
U2_AGREEMENT = 1e-13

pytestmark_u2 = pytest.mark.skipif(
    not _U2_ACCEL, reason="extension built without the #270 off-edge EK twins"
)

_U2_NQP = 4
_U2_KS = {"k/2": 0.5 * K, "k": K, "2k": 2.0 * K}


@pytest.fixture
def numpy_offedge(monkeypatch):
    """Force the off-edge EK kernels back onto their numpy paths."""
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_EK_ACCEL", False)
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_SWEPT_EK_ACCEL", False)


def _u2_mixed_geo(n_i=4, n_co=3, n_perp=2, radius=0.05, gap=5.0):
    """Observer run + a coaxial continuation + a perpendicular run — "a
    collinear pair of edges plus a perpendicular edge", the mixed-
    eligibility fixture shape the design calls for. One call exercises both
    mask outcomes: `lo_j[:n_co]` is coaxial with the observer run and
    extends; `lo_j[n_co:]` is offset and does not.
    """
    lo_i, hi_i, _, _ = _straight(n_i, radius)
    lo_co, hi_co, _, _ = _straight(n_co, radius, origin=(0.0, 0.0, gap))
    lo_off, hi_off, _, _ = _straight(n_perp, radius, origin=(1.0, 0.0, 0.0))
    lo_j = np.vstack([lo_co, lo_off])
    hi_j = np.vstack([hi_co, hi_off])
    group_i = np.zeros(n_i, dtype=np.int64)
    group_j = np.concatenate(
        [np.zeros(n_co, dtype=np.int64), np.ones(n_perp, dtype=np.int64)]
    )
    return lo_i, hi_i, lo_j, hi_j, group_i, group_j


def _u2_gl():
    gl_xi, gl_w = np.polynomial.legendre.leggauss(_U2_NQP)
    return 0.5 * (gl_xi + 1.0), 0.5 * gl_w


# ----------------------------------------------------------------------
# U2a — the off-edge twins agree with numpy's `ek is not None` branch
# ----------------------------------------------------------------------


@pytestmark_u2
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("kname", list(_U2_KS))
@pytest.mark.parametrize("radius_kind", ["scalar", "per-row"])
def test_u2_offedge_ek_cpp_matches_numpy_single_k(
    numpy_offedge, degree, kname, radius_kind
):
    lo_i, hi_i, lo_j, hi_j, gi, gj = _u2_mixed_geo()
    a = (
        0.05
        if radius_kind == "scalar"
        # Two runs on the observer side (momwire#147): the C++ path
        # dispatches one call per constant-radius run and slices `group_i`
        # the same way — this is what proves that slice is correct.
        else np.array([0.05, 0.05, 0.02, 0.02])
    )
    spec = _EK(a=None, group_i=gi, group_j=gj)
    k = _U2_KS[kname]
    ref = _seg_seg_full_moments_offedge(
        lo_i, hi_i, lo_j, hi_j, a, k, degree, _U2_NQP, ek=spec
    )
    _bk._HAVE_BSPLINE_OFFEDGE_EK_ACCEL = _U2_OFFEDGE_EK_ACCEL
    _bk._HAVE_BSPLINE_OFFEDGE_SWEPT_EK_ACCEL = _U2_OFFEDGE_SWEPT_EK_ACCEL
    got = _seg_seg_full_moments_offedge(
        lo_i, hi_i, lo_j, hi_j, a, k, degree, _U2_NQP, ek=spec
    )
    rel = _u1_rel(got, ref)
    assert rel <= U2_AGREEMENT, f"degree {degree}, {kname}, {radius_kind}: {rel:.3e}"


@pytestmark_u2
@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("radius_kind", ["scalar", "per-row"])
def test_u2_offedge_ek_cpp_matches_numpy_swept(numpy_offedge, degree, radius_kind):
    lo_i, hi_i, lo_j, hi_j, gi, gj = _u2_mixed_geo()
    a = 0.05 if radius_kind == "scalar" else np.array([0.05, 0.05, 0.02, 0.02])
    spec = _EK(a=None, group_i=gi, group_j=gj)
    ks = np.array(list(_U2_KS.values()))
    ref = _seg_seg_full_moments_offedge_swept(
        lo_i, hi_i, lo_j, hi_j, a, ks, degree, _U2_NQP, ek=spec
    )
    _bk._HAVE_BSPLINE_OFFEDGE_EK_ACCEL = _U2_OFFEDGE_EK_ACCEL
    _bk._HAVE_BSPLINE_OFFEDGE_SWEPT_EK_ACCEL = _U2_OFFEDGE_SWEPT_EK_ACCEL
    got = _seg_seg_full_moments_offedge_swept(
        lo_i, hi_i, lo_j, hi_j, a, ks, degree, _U2_NQP, ek=spec
    )
    rel = _u1_rel(got, ref)
    assert rel <= U2_AGREEMENT, f"degree {degree}, {radius_kind}: {rel:.3e}"


# ----------------------------------------------------------------------
# U2b — the MASK is the risk: mutation checks from both ends
# ----------------------------------------------------------------------
#
# `test_offedge_ek_extends_only_the_coaxial_pairs` above pins the numpy
# mask; these two pin the C++ one, calling the extension directly so the
# claim is about the kernel itself rather than about Python dispatch (that
# is U2e's job). Bracketing from both ends catches mutations a single
# direction would miss: a mask that is always True passes an all-eligible
# check vacuously; a mask that is always False passes an all-ineligible one
# the same way. Only both together pin `(gi == gj) && (gi >= 0)` exactly.


# Why these gates can assert EXACT equality (momwire#781).
#
# They could not, briefly. The reduced and EK kernels run textually identical
# accumulation loops, but each carried `#pragma omp simd reduction(+:sr,si)`,
# and a reduction clause LICENSES the compiler to reassociate the sum. The
# reduction tree then follows the vectorization factor chosen per FUNCTION, and
# the two functions differ in register pressure. x86-64 picked the same shape
# for both; arm64 did not, and momwire#762's tiling changed the trip count
# enough to expose it -- 1 ulp at the kernel, 148-615 ulp after the assembler.
#
# The clause was measured worth -0.2%/+0.4% on the bspline fills (single
# threaded, pinned, passive wait, min of 5 over 3 alternating rounds) -- i.e.
# nothing -- so it was removed from _accel_bspline.cpp rather than weakening
# these gates. Without it the compiler may not reassociate, both kernels sum in
# source order, and exact equality is a property of the SOURCE instead of a
# coincidence of one target's codegen. _accel_razor.cpp keeps its clause: worth
# ~4.6% there, and it has no cross-kernel equality gate to protect.
#
# A tolerance would also have worked and is the wrong trade: it would have let
# a genuine reassociation regression through for a benefit of zero percent.
# `_bit_report` below stays, because it is what made this diagnosable without
# an arm64 machine.
_EK_ELIGIBLE_MIN_SEPARATION = 1e-6


def _bit_report(got, ref, label):
    """Diagnostic for the exact-agreement gates (momwire#781).

    `array_equal` says only True/False, which on a platform-specific failure
    leaves you guessing whether the gap is one ulp or structural. This reports
    the size and the shape of the disagreement so a CI log is actionable.
    """
    got = np.asarray(got)
    ref = np.asarray(ref)
    if got.shape != ref.shape:
        return f"{label}: SHAPE {got.shape} vs {ref.shape}"
    diff = np.abs(got - ref)
    n_bad = int((diff != 0).sum())
    if n_bad == 0:
        return f"{label}: identical"
    den = np.abs(ref).max()
    rel = diff.max() / den if den else float("inf")
    # ulp distance at the offending entry, on the real part's magnitude.
    idx = np.unravel_index(int(np.argmax(diff)), diff.shape)
    scale = max(abs(ref[idx].real), abs(ref[idx].imag), np.finfo(float).tiny)
    ulps = diff[idx] / np.spacing(scale)
    return (
        f"{label}: {n_bad} of {diff.size} entries differ; "
        f"max|d|={diff.max():.3e} rel={rel:.3e} ~{ulps:.1f} ulp at {idx}; "
        f"got={got[idx]!r} ref={ref[idx]!r}"
    )


@pytestmark_u2
@pytest.mark.parametrize("degree", [1, 2])
def test_u2_the_reduction_gate_still_catches_an_eligible_pair(degree):
    """The tolerance in `_EK_REDUCTION_RTOL` must not have hollowed the gate.

    The gates above assert exact equality, which is only meaningful if an
    actual gating bug would move the answer far more than rounding. This is the
    red half: flip the labels so every pair IS eligible, and require the
    difference to be enormous by comparison.

    If this test ever fails, the reduction gates above have become decorative.
    """
    lo_i, hi_i, lo_j, hi_j, _gi, _gj = _u2_mixed_geo()
    a = 0.05
    n_i, n_j = lo_i.shape[0], lo_j.shape[0]
    t01, w01 = _u2_gl()
    reduced = _bk._acc.seg_seg_full_moments_bspline(
        lo_i, hi_i, lo_j, hi_j, a * a, K, degree, t01, w01
    )
    eligible = _bk._acc.seg_seg_full_moments_bspline_ek(
        lo_i,
        hi_i,
        lo_j,
        hi_j,
        a * a,
        K,
        degree,
        t01,
        w01,
        np.zeros(n_i, dtype=np.int64),
        np.zeros(n_j, dtype=np.int64),
        a,
    )
    rel = np.abs(eligible - reduced).max() / np.abs(reduced).max()
    assert rel > _EK_ELIGIBLE_MIN_SEPARATION, (
        f"an all-eligible EK fill sits only {rel:.3e} from the reduced kernel; "
        "the reduction gates above would not notice a pair wrongly taking the "
        "EK path"
    )


@pytest.mark.parametrize("degree", [1, 2])
def test_u2_all_ineligible_labels_reduce_to_the_reduced_kernel(degree):
    """Every pair declared ineligible (`group_i`/`group_j` all -1, never
    equal to each other by the `>= 0` guard) must match the C++ REDUCED
    kernel: with `eligible` false the twin executes the exact same
    G_re/G_im/wuwu sequence the reduced kernel does.

    Bit for bit. The old docstring said "measured: 0.0 relative ON THIS BOX",
    which was the honest caveat — it WAS an x86 observation. momwire#781 made it
    a source property by dropping the `omp simd reduction` clause that licensed
    the compiler to reassociate differently per function."""
    lo_i, hi_i, lo_j, hi_j, _gi, _gj = _u2_mixed_geo()
    a = 0.05
    n_i, n_j = lo_i.shape[0], lo_j.shape[0]
    t01, w01 = _u2_gl()
    reduced = _bk._acc.seg_seg_full_moments_bspline(
        lo_i, hi_i, lo_j, hi_j, a * a, K, degree, t01, w01
    )
    ineligible = _bk._acc.seg_seg_full_moments_bspline_ek(
        lo_i,
        hi_i,
        lo_j,
        hi_j,
        a * a,
        K,
        degree,
        t01,
        w01,
        np.full(n_i, -1, dtype=np.int64),
        np.full(n_j, -1, dtype=np.int64),
        a,
    )
    assert np.array_equal(ineligible, reduced), _bit_report(
        ineligible, reduced, f"u2 degree {degree}"
    )


@pytestmark_u2
@pytest.mark.parametrize("degree", [1, 2])
def test_u2_all_eligible_labels_apply_the_full_coaxial_factor(degree):
    """Every pair declared eligible (`group_i`/`group_j` all 0) must match
    the numpy oracle computed with the SAME-EDGE convention for "whole
    block eligible" (`group_i=group_j=None`) — the off-edge fill's version
    of a fully-extended block."""
    lo_i, hi_i, lo_j, hi_j, _gi, _gj = _u2_mixed_geo()
    a = 0.05
    n_i, n_j = lo_i.shape[0], lo_j.shape[0]
    whole = _EK(a=None, group_i=None, group_j=None)
    oracle = _seg_seg_full_moments_offedge(
        lo_i, hi_i, lo_j, hi_j, a, K, degree, _U2_NQP, ek=whole
    )
    t01, w01 = _u2_gl()
    got = _bk._acc.seg_seg_full_moments_bspline_ek(
        lo_i,
        hi_i,
        lo_j,
        hi_j,
        a * a,
        K,
        degree,
        t01,
        w01,
        np.zeros(n_i, dtype=np.int64),
        np.zeros(n_j, dtype=np.int64),
        a,
    )
    rel = _u1_rel(got, oracle)
    assert rel <= U2_AGREEMENT, f"degree {degree}: {rel:.3e}"


# ----------------------------------------------------------------------
# U2c — end to end: the same EK-on solve, both off-edge backends
# ----------------------------------------------------------------------
#
# Measured relative worsts on Z (both well inside `ACCEL_AGREEMENT_SOLVE`,
# unit 1's own solve-level bar): T-junction 1.5e-15, T-junction swept
# 5.3e-16, PEC ground (image route) 3.6e-16.

# A "T" junction: two collinear arms through the joint (coaxial, eligible)
# plus a perpendicular third arm (not coaxial) — the bent-deck-with-
# junction shape the design calls for, exercised through the solver's own
# KCL/basis-polynomial machinery rather than through a hand-built mask.
_U2_JUNCTION_DECK = dict(
    wires=[
        np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    ],
    n_per_edge_per_wire=[[8], [8], [8]],
    wire_radius=0.02,
    junctions=[[(0, "end"), (1, "start"), (2, "start")]],
)


@pytestmark_u2
def test_u2_end_to_end_solve_agrees_with_the_numpy_offedge_path(monkeypatch):
    kw = dict(_U2_JUNCTION_DECK, wavelength=LAM, degree=2, extended_kernel=True)
    z_cpp, c_cpp = BSplineSolver(**kw).compute_impedance()
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_EK_ACCEL", False)
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_SWEPT_EK_ACCEL", False)
    z_npy, c_npy = BSplineSolver(**kw).compute_impedance()
    rel = abs(z_cpp - z_npy) / abs(z_npy)
    assert rel <= ACCEL_AGREEMENT_SOLVE, f"{z_cpp!r} vs {z_npy!r} ({rel:.3e})"
    assert _u1_rel(c_cpp, c_npy) <= ACCEL_AGREEMENT_SOLVE


@pytestmark_u2
def test_u2_end_to_end_swept_solve_agrees_with_the_numpy_offedge_path(monkeypatch):
    kw = dict(_U2_JUNCTION_DECK, wavelength=LAM, degree=2, extended_kernel=True)
    ks = 2 * np.pi / LAM * np.array([0.9, 1.0, 1.1])
    z_cpp = np.asarray(BSplineSolver(**kw).compute_impedance_swept(ks))
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_EK_ACCEL", False)
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_SWEPT_EK_ACCEL", False)
    z_npy = np.asarray(BSplineSolver(**kw).compute_impedance_swept(ks))
    assert _u1_rel(z_cpp, z_npy) <= ACCEL_AGREEMENT_SOLVE


@pytestmark_u2
def test_u2_end_to_end_pec_ground_solve_agrees_with_the_numpy_offedge_path(monkeypatch):
    """The IMAGE route (`_IMAGE_DECK`, #249's own fixture): the mirrored
    spec's labels flow through `_ek_slice` into the same off-edge fill, so
    this is also the PEC-ground case the design calls out."""
    kw = dict(_IMAGE_DECK, extended_kernel=True)
    z_cpp, c_cpp = BSplineSolver(**kw).compute_impedance()
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_EK_ACCEL", False)
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_SWEPT_EK_ACCEL", False)
    z_npy, c_npy = BSplineSolver(**kw).compute_impedance()
    rel = abs(z_cpp - z_npy) / abs(z_npy)
    assert rel <= ACCEL_AGREEMENT_SOLVE, f"{z_cpp!r} vs {z_npy!r} ({rel:.3e})"
    assert _u1_rel(c_cpp, c_npy) <= ACCEL_AGREEMENT_SOLVE


# --- and that the C++ kernel really does receive the JOINT image labels ---


@pytestmark_u2
def test_u2_image_tensor_route_serves_the_joint_labels_to_cpp(
    offedge_ek_spy, accel_spy
):
    """`_build_J_image_blocks` — the no-accelerator-assembler fallback
    route (#249's own fixture: `_IMAGE_DECK` has opposite mirror
    eligibility on its two wires, so a free-space spec is distinguishable
    from a mirrored one). `offedge_ek_spy` already pins that the JOINT
    real+image labels reach `_seg_seg_full_moments_offedge`'s `ek=`
    kwarg (#249); this pins that they then reach the C++ symbol rather
    than silently staying on numpy."""
    sim = _image_solver(BSplineSolver)
    geom = sim._build_geometry()
    offedge_ek_spy.clear()
    sim._build_J_image_blocks(geom, sim.k)
    assert len(offedge_ek_spy) == 1
    _assert_mirrored_spec(offedge_ek_spy[0], sim)
    assert accel_spy.counts["seg_seg_full_moments_bspline_ek"] > 0


@pytestmark_u2
def test_u2_image_chunked_route_serves_the_joint_labels_to_cpp(
    offedge_ek_spy, accel_spy
):
    """`_accumulate_Z_image_chunked` — the route a DEFAULT grounded EK-on
    solve actually takes."""
    sim = _image_solver(BSplineSolver)
    geom = sim._build_geometry()
    supp_seg, polys, _kcl, _wk, _wbg = sim._build_basis_polynomials(geom)
    w_A = sim._image_tangent_dot(geom["tangents"]).astype(np.complex128)
    w_Phi = np.ones_like(w_A)
    Z = np.zeros((supp_seg.shape[0],) * 2, dtype=np.complex128, order="F")
    offedge_ek_spy.clear()
    sim._accumulate_Z_image_chunked(
        Z, geom, sim.k, supp_seg, polys, lambda i0, i1: (w_A[i0:i1], w_Phi[i0:i1])
    )
    assert offedge_ek_spy, "no image fill happened"
    n_segs = geom["n_segs_total"]
    row0 = 0
    for ek in offedge_ek_spy:
        rows = np.arange(row0, row0 + len(ek.group_i))
        _assert_mirrored_spec(ek, sim, rows=rows)
        row0 += len(ek.group_i)
    assert row0 == n_segs, "the observer chunks did not cover the mesh"
    assert accel_spy.counts["seg_seg_full_moments_bspline_ek"] > 0


# ----------------------------------------------------------------------
# U2d — EK-OFF armor: the off-edge twins are unreachable and inert
# ----------------------------------------------------------------------


@pytestmark_u2
@pytest.mark.parametrize("name", list(_G7_BSPLINE))
def test_u2_ek_off_never_reaches_the_offedge_ek_entry_points(accel_spy, name):
    BSplineSolver(**_G7_BSPLINE[name], wavelength=LAM, degree=2).compute_impedance()
    assert accel_spy.counts["seg_seg_full_moments_bspline_ek"] == 0
    assert accel_spy.counts["seg_seg_full_moments_bspline_swept_ek"] == 0
    # Vacuity guard: the zeros above mean nothing unless the non-EK off-edge
    # path was actually entered. Which it is — except on a deck that has no
    # off-edge work at all, where momwire#759 skips the pre-pass rather than
    # computing it and overwriting every block. Asserted both ways so this
    # stays a guard rather than becoming a tolerance.
    #
    # Counted over the reduced entries as a SET since momwire#907: with the
    # free-space ladder on, one of these decks is served by the tiered twin
    # rather than the plain symbol, and the guard is about whether the reduced
    # path ran at all, not about which tier it picked.
    reduced = sum(accel_spy.counts[e] for e in _REDUCED_OFFEDGE_ENTRIES)
    if _has_offedge_work(_G7_BSPLINE[name]):
        assert reduced > 0, accel_spy.counts
    else:
        assert reduced == 0, accel_spy.counts


@pytestmark_u2
def test_u2_missing_symbols_fall_back_to_numpy(numpy_offedge, accel_spy):
    """Graceful degradation: an extension built before #270 unit 2 has
    neither off-edge twin, the `_HAVE_*` flags are False, and the EK fill
    is the numpy path — same answer, no AttributeError."""
    z, _ = BSplineSolver(
        **_G7_BSPLINE["mixed radii"],
        wavelength=LAM,
        degree=2,
        extended_kernel=True,
    ).compute_impedance()
    assert np.isfinite(z)
    assert accel_spy.counts["seg_seg_full_moments_bspline_ek"] == 0
    assert accel_spy.counts["seg_seg_full_moments_bspline_swept_ek"] == 0


# ----------------------------------------------------------------------
# U2e — dispatch probes: EK-on off-edge really is served by C++
# ----------------------------------------------------------------------


@pytestmark_u2
def test_u2_ek_on_offedge_is_served_by_cpp(accel_spy, ek_call_counts):
    """ "mixed radii" (#249's own G7 deck: two offset dipoles at different
    wire_radius) exercises both the off-edge C++ path AND its mixed-radius
    row-splitting in one solve — the two dipoles are parallel and offset,
    so every cross-dipole pair is ineligible and every same-edge block is
    (unit 1's territory), which is exactly why this deck is also this
    unit's negative-scoping control below."""
    BSplineSolver(
        **_G7_BSPLINE["mixed radii"],
        wavelength=LAM,
        degree=2,
        extended_kernel=True,
    ).compute_impedance()
    assert accel_spy.counts["seg_seg_full_moments_bspline_ek"] > 0
    assert accel_spy.counts["seg_seg_full_moments_bspline"] == 0
    # ... and the numpy off-edge closed form was not touched — with BOTH
    # C++ pairs live, nothing anywhere reaches `_ek_factor` any more.
    assert ek_call_counts["_ek_factor"] == 0


@pytestmark_u2
def test_u2_ek_on_offedge_falls_back_to_ek_factor_without_the_twin(
    monkeypatch, ek_call_counts
):
    """The scoping control for the probe above: force JUST the off-edge
    twins off (same-edge stays on C++) and confirm `_ek_factor` fires —
    proving the `== 0` above is a dispatch claim, not a "this deck never
    calls it" coincidence."""
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_EK_ACCEL", False)
    monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_SWEPT_EK_ACCEL", False)
    BSplineSolver(
        **_G7_BSPLINE["mixed radii"],
        wavelength=LAM,
        degree=2,
        extended_kernel=True,
    ).compute_impedance()
    assert ek_call_counts["_ek_factor"] > 0


# ======================================================================
# momwire#270 UNIT 3 — the fused H-matrix off-edge assembler + ACA re-enable
# ======================================================================
#
# Units 1 and 2 gave the MOMENT kernels C++ EK twins. `_offedge_aca_
# evaluators` never used them for the ACA fill: `bspline_assemble_
# offedge_block` fuses the off-edge moment quadrature with the Galerkin
# combine in one pass (no intermediate moment tensor), and unit 2's twins
# are moment-TENSOR entry points that a fused assembler cannot call — so
# `_offedge_aca_evaluators` forced `use_accel = False` under EK outright,
# and every ACA fill (H-matrix far blocks, `ArrayBlockSolver._coupling_aca`,
# `_build_lattice_operator`'s per-displacement dense fills) paid the numpy
# `zblock(..., same_edge=False)` cost under EK. This unit adds
# `bspline_assemble_offedge_block_ek` — the fused assembler's own EK twin,
# transcribing unit 2's eligibility rule and `fac` spelling into the fused
# loop — and replaces the blanket `not self.extended_kernel` gate with a
# capability check: `use_accel` stays on under EK whenever the extension
# has the twin, and only degrades to numpy when it does not (never to the
# REDUCED fused assembler, which would silently be wrong under EK).
#
# Labels flow in per CALL, not per block: `_offedge_block_evaluators_
# uniform`'s `get_row`/`get_col` rebuild their single-basis side's segment
# geometry fresh each call (the existing "C++ never precomputes positions
# for unused segments" design unit 3 inherited unmodified) — so `_call`
# takes the GLOBAL segment ids each call's I-side/J-side arrays were built
# from and slices the whole-mesh `_ek_spec` labels by those, not by the
# block-wide `segI`/`segJ` unions `dense()` uses. Getting this wrong (using
# the block-wide slice inside `get_row`/`get_col` too) is exactly the
# mutation exercised below.
#
# WHAT IS AND IS NOT COMPARED — same discipline as units 1/2: cross-backend
# (C++ vs numpy) is relative-tolerance; EK-off is untouched (G7/G7b already
# pin that, and this unit adds no new EK-off code path — only a new EK
# sibling of the fused assembler and a new dispatch decision governing
# which of the two closures is called).

_U3_HAVE_ACCEL = _hm._HAVE_OFFEDGE_BLOCK_EK_ACCEL
pytestmark_u3 = pytest.mark.skipif(
    not _U3_HAVE_ACCEL, reason="extension built without the #270 unit 3 fused EK twin"
)

# Measured on this box: mixed-eligibility fixture below, free-space dense
# 2.2e-14 / row 3.3e-14 / col 8.1e-14; PEC-ground (mirror_J) dense 2.4e-13 /
# row 2.8e-13 / col 3.6e-13 — the tighter end of unit 2's own 1e-13 class
# (this twin shares unit 2's `fac` arithmetic almost verbatim, one fewer
# reduction step than the moment-tensor-then-combine route).
U3_AGREEMENT = 1e-12
# End-to-end solves: unit 1's own solve-level bar (ACA truncation and the
# fused-vs-tensor reassociation both stay well inside it — measured worsts
# below each gate's comment).
U3_AGREEMENT_SOLVE = 5e-12


def _u3_mixed_deck_kw(radius=0.02, ground_z=None):
    """An observer wire (A), a coaxial continuation (B — eligible against
    A) and a parallel offset wire (C — not eligible): the fused-assembler's
    own mixed-eligibility fixture. Unit 2's own `_u2_mixed_geo` builds this
    shape at the raw segment level; this one goes through a real
    `HMatrixSolver` mesh because unit 3's fusion operates at the BASIS
    level (`supp_I`/`polys_I`), not the segment level unit 2's twins do.
    `ground_z` (well below every wire — its value is not otherwise
    meaningful here) is only for the `mirror_J` image-block gate, which
    needs `_image_positions` to have a plane to mirror across."""
    kw = dict(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]]),  # A: observer (I)
            np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 12.5]]),  # B: coaxial with A
            np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 2.5]]),  # C: parallel, offset
        ],
        n_per_edge_per_wire=[[10], [8], [8]],
        wavelength=LAM,
        wire_radius=radius,
    )
    if ground_z is not None:
        kw["ground_z"] = ground_z
    return kw


def _u3_mixed_block(degree=2, radius=0.02, extended_kernel=True, ground_z=None):
    sim = HMatrixSolver(
        **_u3_mixed_deck_kw(radius, ground_z=ground_z),
        degree=degree,
        extended_kernel=extended_kernel,
    )
    ctx = sim._context()
    cen = ctx["basis_centroid"]
    idx_a = np.flatnonzero((np.abs(cen[:, 0]) < 1e-6) & (cen[:, 2] < 5.0))
    idx_b = np.flatnonzero((np.abs(cen[:, 0]) < 1e-6) & (cen[:, 2] > 5.0))
    idx_c = np.flatnonzero(np.abs(cen[:, 0] - 5.0) < 1e-6)
    assert idx_a.size and idx_b.size and idx_c.size, "the mixed fixture degenerated"
    I = idx_a.astype(np.int64)
    J = np.concatenate([idx_b, idx_c]).astype(np.int64)
    return sim, ctx, I, J


# ----------------------------------------------------------------------
# U3a — the fused assembler agrees with the numpy zblock path
# ----------------------------------------------------------------------


@pytestmark_u3
@pytest.mark.parametrize("degree", [1, 2])
def test_u3_fused_offedge_block_matches_zblock_free_space(degree):
    sim, ctx, I, J = _u3_mixed_block(degree=degree)
    row_f, col_f, dense_f = sim._offedge_block_evaluators(ctx, I, J, sim.k)
    D = dense_f()
    D_ref = sim.zblock(I, J, k=sim.k, same_edge=False)
    rel = float(np.abs(D - D_ref).max() / np.abs(D_ref).max())
    assert rel <= U3_AGREEMENT, f"degree {degree}: dense {rel:.3e}"

    r0 = row_f(0)
    r0_ref = sim.zblock(I[0:1], J, k=sim.k, same_edge=False).ravel()
    rel_r = float(np.abs(r0 - r0_ref).max() / np.abs(r0_ref).max())
    assert rel_r <= U3_AGREEMENT, f"degree {degree}: row {rel_r:.3e}"

    c0 = col_f(0)
    c0_ref = sim.zblock(I, J[0:1], k=sim.k, same_edge=False).ravel()
    rel_c = float(np.abs(c0 - c0_ref).max() / np.abs(c0_ref).max())
    assert rel_c <= U3_AGREEMENT, f"degree {degree}: col {rel_c:.3e}"


def _u3_image_block(extended_kernel=True):
    """momwire#249's own `_IMAGE_DECK` (a vertical monopole coaxial with its
    own image, plus a horizontal wire that is NOT), returned as an
    HMatrixSolver self-block spanning the whole mesh.

    REVIEW FINDING (caught in review, before merge): `_u3_mixed_block`
    above is all-VERTICAL geometry, so mirroring it across a horizontal
    ground plane is a no-op on every eligibility label — a vertical wire's
    image lies on the exact same infinite line as the wire itself, so the
    (buggy) free-space-sliced labels and the (correct) joint mirror labels
    produce the IDENTICAL mask. Mutating `_offedge_block_evaluators_
    uniform`'s `self._ek_spec(ctx["geom"], mirror=mirror_J)` to
    `mirror=False` left every unit 3 gate on `_u3_mixed_block` green —
    exactly the "vertical monopole" coincidence momwire#249 unit 2's own
    module docstring already warned about. A HORIZONTAL wire discriminates:
    its image is a PARALLEL, offset line, never coaxial with the real wire,
    so a mirror-wiring bug that quietly fell back to real-vs-real labels
    would wrongly mark the horizontal wire's self-image reaction eligible
    (trivially "coaxial with itself") where the correct joint labelling
    declines it — moving the block measurably off `_zblock_image`'s answer.
    Verified: with the `mirror=False` mutation restored, `test_u3_fused_
    offedge_block_image_declines_a_noncoaxial_mirror` below fails (both the
    dense and the horizontal-wire-owned row/col checks); on a clean tree it
    passes. Kept as its own deck/helper rather than folded into
    `_u3_mixed_block`: that helper's all-vertical shape is still the right
    fixture for the FREE-SPACE mask tests (U3a's free-space gate, U3b's
    brackets), where mirroring never enters."""
    sim = HMatrixSolver(**_IMAGE_DECK, extended_kernel=extended_kernel)
    ctx = sim._context()
    idx = np.arange(ctx["n_basis"], dtype=np.int64)
    return sim, ctx, idx


@pytestmark_u3
def test_u3_fused_offedge_block_image_declines_a_noncoaxial_mirror():
    """The mirror_J path: `_offedge_block_evaluators(..., mirror_J=True)`
    against `_zblock_image`, on the discriminating `_u3_image_block` deck
    (see its docstring for why `_u3_mixed_block` could not catch a mirror-
    wiring bug here). Row/col checks use the LAST basis, which by
    construction (`_IMAGE_DECK`'s wire order) belongs to the horizontal
    wire — the discriminating half of the deck — not the monopole."""
    sim, ctx, idx = _u3_image_block()
    row_f, col_f, dense_f = sim._offedge_block_evaluators(
        ctx, idx, idx, sim.k, mirror_J=True
    )
    D = dense_f()
    D_ref = sim._zblock_image(idx, idx, k=sim.k)
    rel = float(np.abs(D - D_ref).max() / np.abs(D_ref).max())
    assert rel <= U3_AGREEMENT, f"dense: {rel:.3e}"

    j_last = int(idx.size - 1)
    r0 = row_f(j_last)
    r0_ref = sim._zblock_image(idx[j_last : j_last + 1], idx, k=sim.k).ravel()
    rel_r = float(np.abs(r0 - r0_ref).max() / np.abs(r0_ref).max())
    assert rel_r <= U3_AGREEMENT, f"row: {rel_r:.3e}"

    c0 = col_f(j_last)
    c0_ref = sim._zblock_image(idx, idx[j_last : j_last + 1], k=sim.k).ravel()
    rel_c = float(np.abs(c0 - c0_ref).max() / np.abs(c0_ref).max())
    assert rel_c <= U3_AGREEMENT, f"col: {rel_c:.3e}"

    # Sanity: the deck really does carry EK content on this block (a fused
    # twin that silently never applied Eq 89's factor at all would pass the
    # comparisons above too, for the wrong reason — it would just agree
    # with an equally-EK-blind `_zblock_image`... except `_zblock_image` is
    # NOT EK-blind, so this would already be caught above; this assertion
    # instead pins that the EK-on answer differs from the EK-OFF one, i.e.
    # that SOME pair in this block really is eligible).
    sim_off, _ctx_off, idx_off = _u3_image_block(extended_kernel=False)
    D_off = sim_off._zblock_image(idx_off, idx_off, k=sim_off.k)
    rel_off = float(np.abs(D - D_off).max() / np.abs(D_off).max())
    assert rel_off > 1e-6, (
        f"EK-on image block == EK-off ({rel_off:.3e}) — no EK content"
    )


@pytestmark_u3
def test_u3_fused_offedge_block_image_labels_match_the_joint_spec(monkeypatch):
    """Structural companion to the numeric gate above — the #249 pattern of
    pinning wiring "structurally by a spy and numerically by the mutation
    itself" (see the IMAGE-side-is-wired section higher up this file).
    Captures the `group_I`/`group_J` arrays the fused EK twin is actually
    called with for `_u3_image_block`'s self-block and checks them against
    the same joint-mirror mask `_assert_mirrored_spec` pins for the numpy
    route: the monopole (bases 0..11) must extend to its own image, the
    horizontal wire (bases 12..23) must not.
    """
    sim, ctx, idx = _u3_image_block()
    calls = []
    real = _hm._acc

    class _Capture:
        def __getattr__(self, name):
            target = getattr(real, name)
            if name != "bspline_assemble_offedge_block_ek":
                return target

            def wrapped(*args, **kwargs):
                calls.append(args)
                return target(*args, **kwargs)

            return wrapped

    monkeypatch.setattr(_hm, "_acc", _Capture())
    _row_f, _col_f, dense_f = sim._offedge_block_evaluators(
        ctx, idx, idx, sim.k, mirror_J=True
    )
    dense_f()
    assert len(calls) == 1, f"expected exactly one fused-twin call, got {len(calls)}"
    # _call appends (..., group_i, group_j, a_ek, cancel_flag) after the 18
    # positional geometry args (hmatrix.py's `_offedge_block_evaluators_
    # uniform`), so group_i/group_j are the 4th/3rd-from-last positionals.
    group_i, group_j = calls[0][-4], calls[0][-3]
    group_i = np.asarray(group_i)
    group_j = np.asarray(group_j)
    mask = (group_i[:, None] == group_j[None, :]) & (group_i[:, None] >= 0)
    n_mono = _IMAGE_DECK["n_per_edge_per_wire"][0][0]
    mono = np.arange(mask.shape[0]) < n_mono
    assert mask[np.ix_(mono, mono)].all(), (
        "the monopole must extend to its own image (NEC's IND=0 branch)"
    )
    assert not mask[np.ix_(~mono, ~mono)].any(), (
        "the horizontal wire must NOT extend to its own image"
    )


# ----------------------------------------------------------------------
# U3b — the ELIGIBILITY MASK is the risk: bracket checks from both ends
# ----------------------------------------------------------------------
#
# Same discipline as unit 2's U2b: a mask that is always True passes an
# all-eligible check vacuously and a mask that is always False passes an
# all-ineligible one the same way — only both together pin the rule. Forced
# via `_ek_spec`, so both `dense_ek()` (the fused C++ call) and its numpy
# comparison target (which also calls `self._ek_spec` internally) see the
# SAME forced labels.


@pytestmark_u3
@pytest.mark.parametrize("degree", [1, 2])
def test_u3_all_ineligible_labels_reduce_to_the_reduced_assembler(monkeypatch, degree):
    """Every pair declared ineligible must match the REDUCED fused
    assembler exactly: with `eligible` false the EK kernel executes the
    identical G_re/G_im/wuwu sequence the reduced one does (measured: 0.0
    relative on this box, every degree — mirrors unit 2's own
    `test_u2_all_ineligible_labels_reduce_to_the_reduced_kernel`, one level
    up the fusion)."""
    sim_on, ctx_on, I, J = _u3_mixed_block(degree=degree, extended_kernel=True)

    def all_ineligible(self, geom, mirror=False):
        labels = np.full(geom["seg_l"].shape[0], -1, dtype=np.int64)
        return _EK(a=None, group_i=labels, group_j=labels)

    monkeypatch.setattr(HMatrixSolver, "_ek_spec", all_ineligible)
    _, _, dense_ek = sim_on._offedge_block_evaluators(ctx_on, I, J, sim_on.k)
    ek_out = dense_ek()

    sim_off, ctx_off, I2, J2 = _u3_mixed_block(degree=degree, extended_kernel=False)
    assert np.array_equal(I, I2), _bit_report(I, I2, "swept I")
    assert np.array_equal(J, J2), _bit_report(J, J2, "swept J")
    _, _, dense_reduced = sim_off._offedge_block_evaluators(ctx_off, I2, J2, sim_off.k)
    _ref_dense = dense_reduced()
    assert np.array_equal(ek_out, _ref_dense), _bit_report(
        ek_out, _ref_dense, f"u3 degree {degree}"
    )


@pytestmark_u3
@pytest.mark.parametrize("degree", [1, 2])
def test_u3_all_eligible_labels_apply_the_full_coaxial_factor(monkeypatch, degree):
    """Every pair declared eligible must match the numpy zblock oracle
    computed under the SAME forced labels — pinning that the eligible
    branch really does apply Eq 89's factor, not merely that the
    ineligible branch above skips it."""
    sim, ctx, I, J = _u3_mixed_block(degree=degree, extended_kernel=True)

    def all_eligible(self, geom, mirror=False):
        labels = np.zeros(geom["seg_l"].shape[0], dtype=np.int64)
        return _EK(a=None, group_i=labels, group_j=labels)

    monkeypatch.setattr(HMatrixSolver, "_ek_spec", all_eligible)
    _, _, dense_ek = sim._offedge_block_evaluators(ctx, I, J, sim.k)
    ek_out = dense_ek()
    ref = sim.zblock(I, J, k=sim.k, same_edge=False)
    rel = float(np.abs(ek_out - ref).max() / np.abs(ref).max())
    assert rel <= U3_AGREEMENT, f"degree {degree}: {rel:.3e}"


# ----------------------------------------------------------------------
# U3c — end to end: HMatrix + ArrayBlock EK-on solves, accel vs numpy
# ----------------------------------------------------------------------


@pytestmark_u3
def test_u3_end_to_end_hmatrix_solve_agrees_accel_vs_numpy(monkeypatch):
    kw = dict(
        wires=[_dipole_wire(2.5, x=-4.0), _dipole_wire(2.5, x=4.0)],
        n_per_edge_per_wire=[[24], [24]],
        wavelength=LAM,
        wire_radius=0.05,
        degree=2,
        feeds=[(0, None, 1.0 + 0j), (1, None, 1.0 + 0j)],
        aca_tol=1e-7,
        aca_eta=1.0,
        extended_kernel=True,
    )
    sim = HMatrixSolver(**kw)
    assert len(sim.build_hmatrix().far) > 0
    z_accel, _ = sim.compute_impedance()
    monkeypatch.setattr(_hm, "_HAVE_OFFEDGE_BLOCK_EK_ACCEL", False)
    z_numpy, _ = HMatrixSolver(**kw).compute_impedance()
    za, zn = np.atleast_1d(z_accel), np.atleast_1d(z_numpy)
    rel = float(np.abs(za - zn).max() / np.abs(zn).max())
    assert rel <= U3_AGREEMENT_SOLVE, f"{rel:.3e}"


@pytestmark_u3
def test_u3_end_to_end_hmatrix_pec_ground_solve_agrees_accel_vs_numpy(monkeypatch):
    """PEC-ground deck with genuinely eligible image far blocks (two
    monopoles, each coaxial with its own image via the joint real+image
    labelling) — the case the free-space two-dipole deck above cannot
    exercise (those dipoles are parallel but offset, never coaxial)."""
    kw = dict(
        wires=[
            np.array([[0.0, 0.0, 0.02], [0.0, 0.0, 2.42]]),
            np.array([[6.0, 0.0, 0.02], [6.0, 0.0, 2.42]]),
        ],
        n_per_edge_per_wire=[[24], [24]],
        wavelength=LAM,
        wire_radius=0.02,
        degree=2,
        feeds=[(0, None, 1.0 + 0j), (1, None, 1.0 + 0j)],
        ground_z=0.0,
        aca_tol=1e-7,
        aca_eta=1.0,
        extended_kernel=True,
    )
    sim = HMatrixSolver(**kw)
    assert len(sim.build_hmatrix().far) > 0
    z_accel, _ = sim.compute_impedance()
    monkeypatch.setattr(_hm, "_HAVE_OFFEDGE_BLOCK_EK_ACCEL", False)
    z_numpy, _ = HMatrixSolver(**kw).compute_impedance()
    za, zn = np.atleast_1d(z_accel), np.atleast_1d(z_numpy)
    rel = float(np.abs(za - zn).max() / np.abs(zn).max())
    assert rel <= U3_AGREEMENT_SOLVE, f"{rel:.3e}"


@pytestmark_u3
@pytest.mark.parametrize("lattice_fft", [False, True])
def test_u3_end_to_end_array_block_solve_agrees_accel_vs_numpy(
    monkeypatch, lattice_fft
):
    """The 4x1 COLLINEAR deck (G8's own fixture): every element shares the
    same axis, so cross-element pairs are genuinely EK-eligible on both the
    per-pair ACA route (`_coupling_aca`, `lattice_fft=False`) and the
    lattice route (`_build_lattice_operator`'s dense per-displacement
    fills, `lattice_fft=True`)."""
    kw = dict(
        wires=[_dipole_wire(1.2, z=3.0 * i) for i in range(4)],
        n_per_edge_per_wire=[[8]] * 4,
        wavelength=LAM,
        wire_radius=0.03,
        degree=2,
        feeds=[(i, None, 1.0 + 0j) for i in range(4)],
        extended_kernel=True,
        lattice_fft=lattice_fft,
        require_lattice_fft=lattice_fft,
    )
    reset_array_caches()
    op_accel = ArrayBlockSolver(**kw).build_array_blocks()
    assert isinstance(op_accel, LatticeArrayBlock) == lattice_fft
    monkeypatch.setattr(_hm, "_HAVE_OFFEDGE_BLOCK_EK_ACCEL", False)
    reset_array_caches()
    op_numpy = ArrayBlockSolver(**kw).build_array_blocks()
    A, B = op_accel.to_dense(), op_numpy.to_dense()
    rel = float(np.abs(A - B).max() / np.abs(B).max())
    assert rel <= U3_AGREEMENT_SOLVE, f"lattice_fft={lattice_fft}: {rel:.3e}"


# ----------------------------------------------------------------------
# U3d — dispatch probes: EK-on ACA really is served by the fused twin
# ----------------------------------------------------------------------

_U3_ACCEL_ENTRY_POINTS = (
    "bspline_assemble_offedge_block",
    "bspline_assemble_offedge_block_refl",
    "bspline_assemble_offedge_block_ek",
)


class _HMAccelSpy:
    """`_AccelSpy`'s counterpart over `hmatrix`'s OWN `_acc` reference.

    `_bspline_kernels` and `hmatrix` each bind `_acc = ._accel.acc` at
    import time as separate module-level names, so the `accel_spy` fixture
    above (which patches `_bk._acc`) does not see calls `hmatrix.py` makes
    through its own `_acc` — the fused block assemblers are reached only
    from here, never from `_bspline_kernels`.
    """

    def __init__(self, real):
        self._real = real
        self.counts = dict.fromkeys(_U3_ACCEL_ENTRY_POINTS, 0)

    def __getattr__(self, name):
        target = getattr(self._real, name)
        if name not in self.counts:
            return target

        def counted(*args, **kwargs):
            self.counts[name] += 1
            return target(*args, **kwargs)

        return counted


@pytest.fixture
def hm_accel_spy(monkeypatch):
    spy = _HMAccelSpy(_hm._acc)
    monkeypatch.setattr(_hm, "_acc", spy)
    return spy


@pytestmark_u3
def test_u3_g15_now_exercises_the_fused_ek_twin(hm_accel_spy):
    """momwire#249's `test_g15_solver_family_agrees_on_a_fat_deck` builds
    `HMatrixSolver(**common, aca_tol=1e-7)` with `extended_kernel=True` —
    before this unit that ACA fill was forced onto the numpy `zblock` path
    (`_offedge_aca_evaluators`'s old blanket `not self.extended_kernel`
    gate); confirm it now reaches the fused EK twin instead, and that the
    REDUCED fused symbol stays idle (an EK-on fill must not go to the
    kernel that would silently drop the EK correction)."""
    common = _g15_common(True)
    aca = HMatrixSolver(**common, aca_tol=1e-7)
    assert len(aca.build_hmatrix().far) > 0  # the ACA fill really ran
    aca.compute_impedance()
    assert hm_accel_spy.counts["bspline_assemble_offedge_block_ek"] > 0
    assert hm_accel_spy.counts["bspline_assemble_offedge_block"] == 0


@pytestmark_u3
def test_u3_ek_on_aca_fill_never_touches_zblock(monkeypatch, hm_accel_spy):
    calls = []
    orig_zblock = HMatrixSolver.zblock

    def spy_zblock(self, *a, **kw):
        calls.append(1)
        return orig_zblock(self, *a, **kw)

    monkeypatch.setattr(HMatrixSolver, "zblock", spy_zblock)
    sim, ctx, I, J = _u3_mixed_block()
    get_row, get_col, dense = sim._offedge_aca_evaluators(ctx, I, J, sim.k, True)
    dense()
    get_row(0)
    get_col(0)
    assert hm_accel_spy.counts["bspline_assemble_offedge_block_ek"] > 0
    assert len(calls) == 0, "the ACA fill fell back to zblock despite the twin"


@pytestmark_u3
def test_u3_dispatch_control_without_the_twin_uses_zblock(monkeypatch, hm_accel_spy):
    """The scoping control for the probe above: force the fused EK twin
    off and confirm the SAME ACA fill now goes through `zblock` instead —
    proving the `== 0` above is a dispatch claim, not a "this fill never
    needed the fallback" coincidence."""
    monkeypatch.setattr(_hm, "_HAVE_OFFEDGE_BLOCK_EK_ACCEL", False)
    calls = []
    orig_zblock = HMatrixSolver.zblock

    def spy_zblock(self, *a, **kw):
        calls.append(1)
        return orig_zblock(self, *a, **kw)

    monkeypatch.setattr(HMatrixSolver, "zblock", spy_zblock)
    sim, ctx, I, J = _u3_mixed_block()
    get_row, get_col, dense = sim._offedge_aca_evaluators(ctx, I, J, sim.k, True)
    dense()
    get_row(0)
    get_col(0)
    assert hm_accel_spy.counts["bspline_assemble_offedge_block_ek"] == 0
    assert len(calls) > 0, "the control never hit zblock — the probe proves nothing"


@pytestmark_u3
def test_u3_ek_off_never_reaches_the_fused_ek_twin(hm_accel_spy):
    """EK-OFF armor: a defaulted (EK-off) HMatrix ACA fill must reach the
    REDUCED fused assembler and never the EK one — unmodified by this unit
    (G7's own bit-identity gate already pins the numeric half; this is the
    dispatch half, specific to the new symbol this unit adds)."""
    a, _b = _hmatrix_pair(aca_eta=1.0)
    assert len(a.build_hmatrix().far) > 0
    a.compute_impedance()
    assert hm_accel_spy.counts["bspline_assemble_offedge_block_ek"] == 0
    assert hm_accel_spy.counts["bspline_assemble_offedge_block"] > 0


@pytestmark_u3
def test_u3_missing_symbol_falls_back_to_numpy(monkeypatch):
    """Graceful degradation: an extension built before #270 unit 3 has no
    fused EK twin, `_HAVE_OFFEDGE_BLOCK_EK_ACCEL` is False, and the ACA
    fill is the numpy `zblock` path — same answer, no AttributeError."""
    monkeypatch.setattr(_hm, "_HAVE_OFFEDGE_BLOCK_EK_ACCEL", False)
    sim, _, _, _ = _u3_mixed_block()
    z, _ = sim.compute_impedance()
    assert np.isfinite(z)
