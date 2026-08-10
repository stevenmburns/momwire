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
    R = np.geomspace(1e-3, 10.0, 41)
    assert np.all(_ek_factor(R, 0.0, K) == 1.0)
    assert np.all(_ek_reg_extra(R, 0.0, K) == 0.0)
    reduced = (np.exp(-1j * K * R) - 1.0) / (4 * np.pi * R)
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
}


@pytest.mark.parametrize("name", list(_G7_BSPLINE))
def test_g7_bspline_ek_off_is_bit_identical_to_the_default(name):
    kw = dict(_G7_BSPLINE[name], wavelength=LAM, degree=2)
    z_def, c_def = BSplineSolver(**kw).compute_impedance()
    z_off, c_off = BSplineSolver(**kw, extended_kernel=False).compute_impedance()
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
# entered. The other three are still reached from the off-edge fill (which is
# unit 2's) and from the solver itself.
_SAME_EDGE_ONLY_EK_ENTRY_POINTS = {"_ek_reg_extra", "D_ek_moment"}


@pytest.mark.parametrize("same_edge", ["numpy", "default"])
def test_g7b_the_counters_fire_when_ek_is_on(ek_call_counts, monkeypatch, same_edge):
    """The control for the three gates below: a monkeypatch that silently
    failed to bind would make them pass vacuously.

    Parametrized over the same-edge backend rather than narrowed (#270 unit
    1): with the C++ twins forced off every counter must fire, which is the
    original claim; with them live the two same-edge-only counters must be
    exactly zero, which is the dispatch claim. Either way each counter is
    asserted, so a dead monkeypatch is still caught.
    """
    if same_edge == "numpy":
        monkeypatch.setattr(_bk, "_HAVE_BSPLINE_STATIC_EK_ACCEL", False)
        monkeypatch.setattr(_bk, "_HAVE_BSPLINE_REG_SWEPT_EK_ACCEL", False)
    cpp_same_edge = (
        _bk._HAVE_BSPLINE_STATIC_EK_ACCEL and _bk._HAVE_BSPLINE_REG_SWEPT_EK_ACCEL
    )
    BSplineSolver(
        **_G7_BSPLINE["free space"],
        wavelength=LAM,
        degree=2,
        extended_kernel=True,
    ).compute_impedance()
    for attr, n in ek_call_counts.items():
        if cpp_same_edge and attr in _SAME_EDGE_ONLY_EK_ENTRY_POINTS:
            assert n == 0, f"{attr}: numpy entered despite the C++ same-edge twin"
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


@pytest.mark.parametrize("ground_model", ["refl-coef", "sommerfeld"])
def test_extended_kernel_refuses_finite_ground(ground_model):
    with pytest.raises(NotImplementedError, match="ground_eps"):
        BSplineSolver(
            **_refusal_kw(
                ground_z=0.0, ground_eps=(13.0, 0.005), ground_model=ground_model
            )
        )


def test_extended_kernel_serves_plain_pec_ground():
    z, _ = BSplineSolver(**_refusal_kw(ground_z=-3.0)).compute_impedance()
    assert np.isfinite(z)


@pytest.mark.parametrize("cls", [HMatrixSolver, ArrayBlockSolver])
def test_extended_kernel_refusals_reach_the_subclasses(cls):
    with pytest.raises(NotImplementedError, match="ground_eps"):
        cls(**_refusal_kw(ground_z=0.0, ground_eps=(13.0, 0.005)))


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
    Z = np.zeros((supp_seg.shape[0],) * 2, dtype=np.complex128, order="F")
    offedge_ek_spy.clear()
    sim._accumulate_Z_image_chunked(
        Z, geom, sim.k, supp_seg, polys, w_A, np.ones_like(w_A)
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
def test_u1_ek_on_same_edge_is_served_by_cpp(accel_spy, ek_call_counts):
    BSplineSolver(
        **_G7_BSPLINE["free space"],
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
    # Off-edge is still numpy in unit 1 — the control that this probe is
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
