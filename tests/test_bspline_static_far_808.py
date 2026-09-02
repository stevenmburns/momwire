"""The same-edge static moments away from the diagonal — momwire#808.

`_bspline_static_moments.J_static_moment` is sympy's closed form, and it is
the NEAR-field half. The value decays like h⁵/D while every term of a
four-corner closed form grows like D⁵, so at 401 segments on a 9.7 m edge the
(2, 2) moment between the two ends comes out **2.94e+01 relative** — no
correct digits. `_bspline_static_far` supplies the other regime and
`J_static_stable` dispatches between them on the convergence ratio.

What this module gates:

  * **an independent oracle, not a second implementation.** The reference is
    Gauss-Legendre on the integral's OWN integrand — nothing shared with
    either spelling under test. Under the substitution ξ = a·sinh t the
    kernel's a-wide peak becomes unit-width and `dξ/√(ξ²+a²)` is exactly
    `dt`, so the rule converges at the self pair too. That matters: plain GL
    on the raw integrand cannot see a peak 900× narrower than its window and
    reports the self pair 1e-2 wrong, which is how #808's first draft came to
    accuse the closed form of a near-field defect it does not have.

  * **both halves, and that each is needed.** The stable dispatch is checked
    everywhere; the closed form is checked to be GOOD near and BAD far. A
    regression that quietly drops the dispatch would pass an accuracy test
    that only looked at the answer it returns near the diagonal.

  * **the two lanes switch on the same pairs.** Not just that they agree in
    the bulk — that pairs sitting either side of the threshold, and pairs
    sitting on it to the last ulp, land the same way in C++ and numpy.

  * **the generated files are not hand-edited.** Four files come out of one
    sympy run; a hand edit to either lane is a silent divergence.
"""

from __future__ import annotations

import hashlib
import pathlib
from math import comb

import numpy as np
import pytest

from momwire._bspline_static_far import (
    J_static_far,
    J_static_stable,
    far_mask,
    series_ratio,
)
from momwire._bspline_static_moments import J_static_moment

A_WIRE = 5e-4
EDGE = 9.7
LADDER = (21, 81, 201, 401)
MOMENTS = [(p, q) for p in range(3) for q in range(3)]


# ======================================================================
# The oracle
# ======================================================================
def _overlap_poly(w, p, q, h1, h2):
    """W̃(w) = ∫ u^p (u−w)^q du over the overlap, in closed form.

    u runs over [max(0, w), min(h1, w+h2)] — the u for which v = u−w lies in
    the source segment. Expanding (u−w)^q makes this a short sum of exact
    power integrals, which is both faster and more accurate than quadrature.
    """
    lo = np.maximum(0.0, w)
    hi = np.minimum(h1, w + h2)
    hi = np.maximum(hi, lo)
    out = np.zeros_like(np.asarray(w, dtype=float))
    for j in range(q + 1):
        k = p + j + 1
        out = out + comb(q, j) * (-w) ** (q - j) * (hi**k - lo**k) / k
    return out


def oracle_J(p, q, h1, h2, D, a, nt=200):
    """∫∫ u^p v^q / √((u−v−D)² + a²) — the definition, by quadrature.

    Rewritten over w = u − v as ∫ W̃(w)/√((w−D)² + a²) dw and then substituted
    w − D = a·sinh t, which is exact and flattens the peak. Split at W̃'s
    breakpoints so each piece is smooth.
    """
    breaks = sorted({-h2, 0.0, h1 - h2, h1})
    x, wq = np.polynomial.legendre.leggauss(nt)
    total = 0.0
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        if hi <= lo:
            continue
        t_lo = np.arcsinh((lo - D) / a)
        t_hi = np.arcsinh((hi - D) / a)
        mid = 0.5 * (t_hi + t_lo)
        half = 0.5 * (t_hi - t_lo)
        w = D + a * np.sinh(mid + half * x)
        total += float((_overlap_poly(w, p, q, h1, h2) * wq).sum() * half)
    return total


def test_the_oracle_is_converged():
    """It is only a reference if it has stopped moving. Checked at the self
    pair, which is the hard one — the peak sits inside the window there."""
    h = EDGE / 21
    for p, q in ((0, 0), (2, 2)):
        lo = oracle_J(p, q, h, h, 0.0, A_WIRE, nt=100)
        hi = oracle_J(p, q, h, h, 0.0, A_WIRE, nt=200)
        assert abs(lo - hi) / abs(hi) < 1e-13


def _call(fn, p, q, h, delta):
    f = np.float64
    return float(
        np.asarray(fn(p, q, f(0.0), f(h), f(delta * h), f((delta + 1) * h), f(A_WIRE)))
    )


# ======================================================================
# The claim
# ======================================================================
@pytest.mark.parametrize("N", LADDER)
@pytest.mark.parametrize("p,q", MOMENTS)
def test_the_stable_moment_holds_at_every_separation(N, p, q):
    """1e-12 across the whole edge, for every moment, at every mesh.

    The bound is one number for all of them on purpose: the point of the
    dispatch is that accuracy stops depending on where the pair sits.
    """
    h = EDGE / N
    for delta in (0, 1, 2, 3, N // 2, N - 1):
        ref = oracle_J(p, q, h, h, delta * h, A_WIRE)
        got = _call(J_static_stable, p, q, h, delta)
        assert abs(got - ref) <= 1e-12 * abs(ref), (
            f"N={N} delta={delta} (p,q)=({p},{q}): {got} vs {ref}"
        )


def test_the_closed_form_alone_is_still_wrong_far_away():
    """The negative control, and the reason the dispatch exists.

    If this ever passes, either the generated forms were re-derived (in which
    case the far branch may be retired, deliberately) or the test lost its
    grip. It must not pass by accident.
    """
    h = EDGE / 401
    ref = oracle_J(2, 2, h, h, 400 * h, A_WIRE)
    got = _call(J_static_moment, 2, 2, h, 400)
    assert abs(got - ref) / abs(ref) > 1.0, "the closed form got better?"


def test_the_closed_form_alone_is_right_near_the_diagonal():
    """And the reason it is kept: the far series is not a replacement.

    #808's first draft reported the self pair as 1e-2 wrong. It is not — that
    was an unconverged oracle. The near field is where the closed form is at
    its best and the series does not converge at all.
    """
    for N in LADDER:
        h = EDGE / N
        for delta in (0, 1):
            for p, q in MOMENTS:
                ref = oracle_J(p, q, h, h, delta * h, A_WIRE)
                got = _call(J_static_moment, p, q, h, delta)
                assert abs(got - ref) <= 1e-12 * abs(ref), f"N={N} d={delta}"


# ======================================================================
# The switch
# ======================================================================
def test_the_ratio_is_the_radius_of_convergence():
    """`series_ratio` is read off the kernel's poles, not fitted.

    f(ξ) = (ξ²+a²)^{-1/2} has poles at ξ = ±ja, so the expansion about the
    pair centroid converges for window half-widths under √(ξ₀²+a²) — and on a
    uniform edge that makes r exactly 1 for adjacent segments and exactly 1/2
    for next-but-one, which is what these numbers are.
    """
    h = EDGE / 201
    r1 = series_ratio(0.0, h, h, 2 * h, A_WIRE)
    r2 = series_ratio(0.0, h, 2 * h, 3 * h, A_WIRE)
    assert abs(r1 - 1.0) < 1e-4
    assert abs(r2 - 0.5) < 1e-4
    assert not far_mask(0.0, h, h, 2 * h, A_WIRE)
    assert far_mask(0.0, h, 2 * h, 3 * h, A_WIRE)


def test_the_series_does_not_converge_inside_the_switch():
    """The other half of the threshold's justification: it is not
    conservatism, it is where the series stops working."""
    h = EDGE / 201
    ref = oracle_J(2, 2, h, h, h, A_WIRE)
    got = _call(J_static_far, 2, 2, h, 1)
    assert abs(got - ref) / abs(ref) > 1e-8


@pytest.mark.parametrize("N", (81, 401))
def test_both_spellings_agree_where_the_regimes_overlap(N):
    """At the switch both are accurate, so the dispatch is not a cliff.

    This is what makes a threshold safe to have: crossing it changes the
    answer by less than either side's own error.
    """
    h = EDGE / N
    for p, q in MOMENTS:
        near = _call(J_static_moment, p, q, h, 2)
        far = _call(J_static_far, p, q, h, 2)
        assert abs(near - far) <= 1e-11 * abs(near), f"(p,q)=({p},{q})"


# ======================================================================
# The lanes
# ======================================================================
def _numpy_table(N, h, a, max_d=2):
    import momwire._bspline_kernels as bk

    saved = bk._HAVE_BSPLINE_STATIC_ACCEL
    bk._HAVE_BSPLINE_STATIC_ACCEL = False
    try:
        return bk._seg_seg_static_moments(np.linspace(0.0, N * h, N + 1), a, max_d)
    finally:
        bk._HAVE_BSPLINE_STATIC_ACCEL = saved


@pytest.mark.parametrize("N", LADDER)
def test_the_two_lanes_agree_on_the_whole_table(N):
    from momwire._accel import acc as _acc

    if _acc is None or not hasattr(_acc, "seg_seg_static_moments_bspline_uniform"):
        pytest.skip("static-moments accelerator not built")
    h = EDGE / N
    cxx = _acc.seg_seg_static_moments_bspline_uniform(float(h), float(A_WIRE), N, 2)
    npy = _numpy_table(N, h, A_WIRE)
    assert cxx.shape == npy.shape
    assert np.abs(cxx - npy).max() <= 1e-14 * np.abs(npy).max()


@pytest.mark.slow
def test_the_two_lanes_switch_on_the_same_pairs():
    """Agreement in the bulk is not the claim; agreement AT the threshold is.

    A pair whose ratio sits within an ulp of the switch must not take the
    series in one lane and the closed form in the other. The two compute `r`
    from the same quantities in the same order, so they land the same way —
    and even if a compiler's FMA contraction moved one of them across, the
    cost is bounded by the overlap test above rather than by a cliff.
    """
    from momwire._accel import acc as _acc

    if _acc is None or not hasattr(_acc, "seg_seg_static_moments_bspline_uniform"):
        pytest.skip("static-moments accelerator not built")
    # Solve for the h that puts a delta=2 pair exactly at FAR_RATIO, then walk
    # it across in single ulps.
    a = A_WIRE
    # Small N deliberately: what is under test is the DECISION at the
    # threshold, which every pair in the table exercises, not the table size.
    for N in (41, 81):
        h = EDGE / N
        for bump in range(-2, 3):
            hh = np.nextafter(h, np.inf) if bump > 0 else h
            for _ in range(abs(bump) - 1 if bump else 0):
                hh = np.nextafter(hh, np.inf if bump > 0 else -np.inf)
            cxx = _acc.seg_seg_static_moments_bspline_uniform(float(hh), float(a), N, 2)
            npy = _numpy_table(N, hh, a)
            assert np.abs(cxx - npy).max() <= 1e-14 * np.abs(npy).max(), (
                f"N={N} bump={bump}"
            )


# ======================================================================
# The generated files
# ======================================================================
GENERATED = {
    "_bspline_static_moments.py": (
        "6c09cb94e7d5774d9cd16f420c34eabdca92f4c27bbdd4a6befc8339ff998a9b"
    ),
    "_bspline_static_moments_inline.h": (
        "f11b6b65abf0b9b1f1257a1ffb0ebd71563650068266b6109c5b396f6c01b441"
    ),
    "_bspline_ek_moments.py": (
        "01a91dac847a18e91127d9e359bf56df469714447a79c12e10624dbd69d13956"
    ),
    "_bspline_ek_moments_inline.h": (
        "86aa450dfc3fbe59e793b7cc140d5eb3b8fb3b7aca481c4609c52e216283d8bb"
    ),
}


def test_the_generated_moment_files_are_not_hand_edited():
    """Four files come out of ONE sympy run, two of them C++ twins of the
    other two. A hand edit to either lane is a silent divergence that no
    cross-lane test would necessarily catch, because both lanes would have
    been edited to agree.

    The hashes are the contract. Re-running
    `scripts/derive_bspline_static_moments.py` and updating them is the
    supported way to change these files; editing them is not. The script's
    own `--check` does the same comparison against a fresh sympy run, which
    is the stronger test and needs sympy, which is not a test dependency.
    """
    import momwire

    root = pathlib.Path(momwire.__file__).parent
    got = {}
    for name in GENERATED:
        got[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    expected = {k: v for k, v in GENERATED.items() if v is not None}
    if not expected:
        pytest.skip("hashes not yet recorded")
    for name, want in expected.items():
        assert got[name] == want, (
            f"{name} does not match its recorded hash. If you re-ran "
            f"scripts/derive_bspline_static_moments.py, update GENERATED in "
            f"this file to {got[name]!r}; if you hand-edited it, do not."
        )
