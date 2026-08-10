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
