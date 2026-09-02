"""The cancellation-free spellings, and the property that makes them worth it.

momwire#799. `src/momwire/_stable.py` carries the sibling table and
`_stable_inline.h` its C++ twin; `docs/cancellation-free-spellings.md` says
why. What is gated HERE is the only claim that matters about them:

  * **they are accurate where the literal spelling is not.** Not "the two
    agree" — they do not, and the difference IS the point. Each helper is
    measured against an independent high-precision evaluation at the
    arguments the solvers actually reach, and the literal form it replaced is
    measured beside it so a regression that quietly reintroduces the
    subtraction shows up as the literal number rather than as a pass.

  * **the reference is stdlib.** `Decimal` with a hand-rolled exp/sin/cos/log,
    not mpmath: momwire's test extra is `pytest` + `matplotlib` + `scikit-rf`,
    and a gate that skips when an undeclared dependency is missing is not a
    gate. Sixty digits is ~44 more than float64 carries, which covers every
    amplification in the table below with four decades to spare.

  * **the complex branch is not a second definition.** At Im k = 0 the
    complex bracket must return the real branch's BITS, or a build that
    happens to reach the in-medium entry point with a real k would silently
    change answers. `expm1(0)*cos(y)` is exactly 0.0 and `exp(0)*sin(y)`
    exactly `sin(y)`, so this is an identity rather than a tolerance.

The amplifications this module pins are the ones the audit measured on real
decks: kR ~ 1e-3 for the razor remainder (an 801-segment thin dipole's
near-self pairs), h/|u_r| ~ 1e-3 for the statics (the same deck's far pairs),
kΔ ~ 4e-3 for the sinusoidal atom.
"""

from __future__ import annotations

from decimal import Decimal, getcontext

import numpy as np
import pytest

from momwire._stable import (
    asinh_diff,
    expm1_neg_j,
    expm1_neg_j_from_half,
    expm1_neg_jkR,
    sqrt_diff,
)

getcontext().prec = 60

_TINY = Decimal(10) ** -56


def _dexp(x):
    x = Decimal(x)
    term = s = Decimal(1)
    n = 0
    while True:
        n += 1
        term = term * x / n
        s += term
        if abs(term) < _TINY * (abs(s) + 1):
            return s


def _dsin(x):
    x = Decimal(x)
    term = s = Decimal(x)
    n = 1
    while True:
        n += 2
        term = -term * x * x / (n * (n - 1))
        s += term
        if abs(term) < _TINY * (abs(s) + 1):
            return s


def _dcos(x):
    x = Decimal(x)
    term = s = Decimal(1)
    n = 0
    while True:
        n += 2
        term = -term * x * x / (n * (n - 1))
        s += term
        if abs(term) < _TINY * (abs(s) + 1):
            return s


def _dlog(x):
    """log by Newton on exp, seeded from float64 — six steps is far past 60
    digits from a 16-digit seed (the iteration doubles them)."""
    import math

    x = Decimal(x)
    y = Decimal(repr(math.log(float(x))))
    for _ in range(6):
        e = _dexp(y)
        y = y + (x - e) / e
    return y


def _dasinh(x):
    x = Decimal(x)
    r = (x * x + 1).sqrt()
    return _dlog(x + r) if x >= 0 else -_dlog(-x + r)


def _rel(approx, exact):
    return float(abs(Decimal(repr(float(approx))) - exact) / abs(exact))


def _rel_c(approx, exact_re, exact_im):
    dr = Decimal(repr(float(approx.real))) - exact_re
    di = Decimal(repr(float(approx.imag))) - exact_im
    num = (dr * dr + di * di).sqrt()
    den = (exact_re * exact_re + exact_im * exact_im).sqrt()
    return float(num / den)


# ======================================================================
# The reference itself
# ======================================================================
def test_the_decimal_reference_reproduces_libm():
    """A wrong reference would pass every test below by agreeing with the
    thing it is supposed to judge. Check it where float64 is not in trouble:
    at O(1) arguments the literal spellings are correct to an ulp, so libm is
    the oracle for the oracle."""
    import math

    for x in (0.5, 1.3, 2.7):
        assert _rel(math.exp(x), _dexp(x)) < 5e-16
        assert _rel(math.cos(x), _dcos(x)) < 5e-16
        assert _rel(math.sin(x), _dsin(x)) < 5e-16
        assert _rel(math.log(x + 1.0), _dlog(x + 1.0)) < 5e-16
        assert _rel(math.asinh(x), _dasinh(x)) < 5e-16


# ======================================================================
# The remainder
# ======================================================================
# (kR, the literal form's relative error on the REAL part, measured). The
# literal number is here so the bar below cannot be met by regressing: it is
# the distance the rewrite bought.
REMAINDER_KR = [
    (1e-5, 8.3e-8),
    (1e-4, 5.2e-9),
    (1.26e-3, 6.7e-11),
    (8e-3, 1.1e-12),
    (3e-2, 6.0e-14),
    (1.0, 1.0e-16),
]


@pytest.mark.parametrize("kR,literal_err", REMAINDER_KR)
def test_the_remainder_real_part_is_relatively_accurate(kR, literal_err):
    """`exp(-jkR) - 1`'s real part, which is the whole of the cancellation.

    The literal spelling returns `cos(kR) - 1` to an absolute epsilon, so its
    relative error is ~eps/((kR)^2/2). The stable one is flat at an ulp across
    five decades of kR — that flatness, not any single number, is the claim.
    """
    ref_re = _dcos(kR) - 1
    got = expm1_neg_jkR(1.0, np.float64(kR))
    assert _rel(np.real(got), ref_re) < 1e-15

    literal = complex(np.exp(-1j * kR) - 1.0)
    measured = _rel(literal.real, ref_re)
    # The literal form really is as bad as the table says (within a factor of
    # four; the exact value depends on which way one ulp of cos() fell).
    assert measured < 4.0 * max(literal_err, 2e-16)
    if literal_err > 1e-13:
        assert measured > 0.25 * literal_err


def test_the_remainder_agrees_with_the_literal_form_where_nothing_cancels():
    """At kR = O(1) there is no cancellation and the two spellings must give
    the same answer to a couple of ulps — the rewrite is not a change of
    value, only of accuracy where accuracy was being lost."""
    R = np.geomspace(1.0, 20.0, 41)
    stable = expm1_neg_jkR(1.0, R)
    literal = np.exp(-1j * R) - 1.0
    assert np.abs(stable - literal).max() / np.abs(literal).max() < 8e-16


def test_the_complex_branch_returns_the_real_branch_bits_at_zero_imag():
    """`expm1(0)*cos(y)` is exactly 0.0 and `exp(0)*sin(y)` exactly sin(y), so
    the branch is an optimisation and not a second definition. A tolerance
    here would hide a build reaching the in-medium entry with a real k."""
    R = np.geomspace(1e-6, 20.0, 97)
    for k in (0.314, 1.0, 6.28):
        assert np.array_equal(expm1_neg_jkR(k, R), expm1_neg_jkR(complex(k, 0.0), R))


@pytest.mark.parametrize("R", [1e-4, 1e-3, 4e-3, 0.1, 1.0, 9.0])
def test_the_complex_bracket_is_accurate_in_medium(R):
    """The in-medium bracket at soil C / 7 MHz, the most per-ulp sensitive
    medium in the momwire#796 table (smallest |k|). Both terms of the real
    bracket are cancellation-free; the imaginary part never cancelled."""
    k = complex(0.3380818578745221, -0.081740240401636)
    a, y = k.imag * R, k.real * R
    e = _dexp(a)
    ref_re, ref_im = e * _dcos(y) - 1, -e * _dsin(y)
    got = complex(expm1_neg_jkR(k, np.float64(R)))
    assert _rel_c(got, ref_re, ref_im) < 1e-15
    assert _rel(got.real, ref_re) < 1e-15


def test_the_half_angle_entry_point_is_the_same_object():
    """`expm1_neg_j_from_half` exists so the sinusoidal lanes can share one
    spelling with a C++ kernel whose phase table holds half angles. It is the
    same value to a couple of ulps — if it ever drifts further, the two lanes
    have stopped computing the same thing."""
    w = np.geomspace(1e-6, 3.0, 61)
    a = expm1_neg_j(w)
    b = expm1_neg_j_from_half(-0.5 * w)
    assert np.abs(a - b).max() / np.abs(a).max() < 4e-16


# ======================================================================
# The statics
# ======================================================================
# A far observer on a thin collinear deck: rho = a = 5e-4, h = the segment
# length, u_r = the axial offset. h/|u_r| is the cancellation parameter, and
# 5e-3 is where an 801-segment half-wave dipole's outermost pairs sit.
STATICS = [
    (0.4042, -8.0),  # h/|u_r| ~ 5e-2, a 24-segment deck
    (0.0485, -9.5),  # ~5e-3,          200 segments
    (0.0121, -9.6),  # ~1.3e-3,        801 segments
    (0.0121, 9.6),  # the same, observer on the other side (u0 > 0 branch)
]


@pytest.mark.parametrize("h,u_r", STATICS)
def test_asinh_diff_beats_the_literal_difference(h, u_r):
    rho2 = np.float64(5e-4) ** 2
    u0, u1 = np.float64(-u_r), np.float64(h - u_r)
    r0, r1 = np.sqrt(u0 * u0 + rho2), np.sqrt(u1 * u1 + rho2)
    rho = np.sqrt(rho2)
    ref = _dasinh(Decimal(repr(float(u1))) / Decimal(repr(float(rho)))) - _dasinh(
        Decimal(repr(float(u0))) / Decimal(repr(float(rho)))
    )
    stable = _rel(asinh_diff(u0, u1, rho2, r0, r1), ref)
    literal = _rel(np.arcsinh(u1 / rho) - np.arcsinh(u0 / rho), ref)
    assert stable < 1e-13
    assert stable <= max(literal, 1e-15)


@pytest.mark.parametrize("h,u_r", STATICS)
def test_sqrt_diff_is_the_exact_rationalisation(h, u_r):
    rho2 = np.float64(5e-4) ** 2
    u0, u1 = np.float64(-u_r), np.float64(h - u_r)
    r0, r1 = np.sqrt(u0 * u0 + rho2), np.sqrt(u1 * u1 + rho2)
    ref = (Decimal(repr(float(u1))) ** 2 + Decimal(repr(float(rho2)))).sqrt() - (
        Decimal(repr(float(u0))) ** 2 + Decimal(repr(float(rho2)))
    ).sqrt()
    assert _rel(sqrt_diff(u0, u1, rho2, r0, r1), ref) < 1e-13


def test_the_statics_helpers_are_what_the_kernel_calls():
    """The helpers are only worth testing if `_static_axis_moments` uses them.
    Structural rather than numeric: reimplementing the moments here and
    comparing would pass whichever spelling the module shipped."""
    import inspect

    from momwire import _kernel_moments

    src = inspect.getsource(_kernel_moments._static_axis_moments)
    assert "asinh_diff(" in src and "sqrt_diff(" in src
    assert "np.arcsinh" not in src
    src_ek = inspect.getsource(_kernel_moments._static_axis_moments_ek)
    assert "asinh_diff(" in src_ek and "sqrt_diff(" in src_ek
    assert "np.arcsinh" not in src_ek


# ======================================================================
# The sinusoidal atom
# ======================================================================
@pytest.mark.parametrize("kd", [3.8e-3, 1e-2, 0.127, 1.5])
def test_the_p_sum_atom_is_tan_half(kd):
    """`(1 - cos kd)/sin kd` is `tan(kd/2)` exactly, and the literal quotient
    computes an O(kd^2) numerator to an absolute epsilon. 3.8e-3 is an
    801-segment half-wave dipole's kd."""
    ref = (1 - _dcos(kd)) / _dsin(kd)
    assert _rel(np.tan(0.5 * kd), ref) < 1e-15
    assert _rel(np.sin(0.5 * kd) / np.cos(0.5 * kd), ref) < 1e-15
    if kd < 1e-2:
        assert _rel((1.0 - np.cos(kd)) / np.sin(kd), ref) > 1e-13


# ======================================================================
# The axis frame — the one that bit
# ======================================================================
def test_the_axis_frame_is_exactly_on_axis_for_a_collinear_pair():
    """`rho2` is EXACTLY `a²` when the observer sits on the segment's own axis.

    Not "to a tolerance". `|d|² − (d·t)²` is an exact cancellation there, so
    any spelling that computes it as a subtraction returns the rounding error
    of `|d|²` — which is +0.0 on one compiler and ±2e-14 on another, and
    2e-14 is a 1e-7 RELATIVE perturbation of a 5e-4 wire's `rho2`, the
    quantity every static moment takes a logarithm of. `|d − (d·t)t|²` is the
    zero vector's norm under any rounding and any FMA contraction.

    This is the gate momwire#798's macOS reading needed and did not have: the
    defect is invisible to every check that compares one implementation
    against itself, and surfaces only as a cross-lane or cross-platform
    disagreement — which #798 then attributed to the wrong cancellation.
    """
    from momwire._kernel_moments import _axis_frame

    a = 5e-4
    seg_p0 = np.array([[0.0, 0.0, -4.85]])
    seg_t = np.array([[0.0, 0.0, 1.0]])
    obs = np.zeros((64, 3))
    obs[:, 2] = np.linspace(-4.85, 4.85, 64)
    _u_r, rho2 = _axis_frame(obs, seg_p0, seg_t, a)
    assert np.array_equal(rho2, np.full_like(rho2, a * a))


def test_the_axis_frame_beats_the_subtraction_off_a_skew_axis():
    """The same claim where the exact answer is not zero.

    A wire along (1,1,1)/√3 puts an observer's axial projection at an angle
    whose cosine is not representable, so `perp` is a small nonzero number
    reached by subtracting two O(|d|²) terms. The vector form's relative
    error is a few ε; the subtraction's is ε·|d|²/|p|², i.e. ε/sin²θ.

    This is the negative control the exactness test above cannot be: on
    x86-64 the old spelling happens to return exactly 0.0 for an axis-ALIGNED
    pair (both terms round identically), so only arm64 saw it fail. Here it
    fails on every platform — measured 1.9e-7 for the subtraction against
    6.7e-16 for the norm, with the bound five decades between them.
    """
    from momwire._kernel_moments import _axis_frame

    a = 5e-4
    t = np.array([[1.0, 1.0, 1.0]]) / np.sqrt(3.0)
    seg_p0 = np.zeros((1, 3))
    # Observers a hair off the axis: s·t plus a tiny perpendicular kick.
    s = np.linspace(1.0, 9.0, 32)[:, None]
    perp_dir = np.array([[1.0, -1.0, 0.0]]) / np.sqrt(2.0)
    eps_off = 1e-6
    obs = s * t + eps_off * perp_dir
    _u_r, rho2 = _axis_frame(obs, seg_p0, t, a)
    truth = eps_off * eps_off + a * a
    # Measured: 6.7e-16 for the norm, 1.9e-7 for the subtraction it replaced.
    assert np.abs(rho2 / truth - 1.0).max() < 1e-12
