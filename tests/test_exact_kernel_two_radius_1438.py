"""The exact ring kernel across a radius step (momwire#1438).

Coaxial pairs at two radii take K₂(z) = (1/2π)∮e^{-jkR}/(4πR)dφ with
R² = z² + a₁² + a₂² - 2a₁a₂ cos φ, each ring on its own wire's surface
(`_exact_kernel.ring_kernel2`). Gates:

* the kernel against direct adaptive φ quadrature (no AGM, no series), at
  radius ratios 16:1 to 1.00001:1 and |z| from 0 to 300√(a₁a₂), ≤ 1e-12
  relative (measured 7e-15);
* continuity onto the one-radius kernel as a₂ → a₁, first order in δ;
* the pair and point moments against nested adaptive quadrature of K₂,
  near-touching pairs and near-equal radii included, ≤ 1e-11 relative;
* the row windows against the whole block, zero on same-radius pairs;
* the MECHANISM on a stepped dipole: with the fill's own kernel swapped in,
  `exact_kernel=True` reproduces the fill, so the cross-radius base is
  removed and the correction lands on the right bases;
* on a straight stepped wire every pair is on one line, so the exact-kernel
  Z cannot depend on the fill's kernel (reduced vs EK);
* bs2's Galerkin Z is symmetric with the step present;
* a deck without a radius step never enters the two-radius path;
* on main's seams (the v2 rebuild): bs2's cross job through `BaseRows`'
  cross mode equals the per-pair base masked to the cross pairs, gathered
  blocks included, and produces nothing on a same-radius pair; the row
  windows equal the whole route; razor's cross jobs join the plan cache and
  a hit adds the uncached route's bits; the k-series never serves a cross
  job, and Z through it is the direct route's.
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from momwire import BSplineSolver, RazorSolver
from momwire import _exact_kernel as EK
from momwire import _exact_kernel_base as EKB
from momwire import _exact_kernel_seam as SEAM
from momwire import _exact_kernel_series as ES
from momwire import _razor_exact_plan as PLAN
from momwire import bspline as _bspline
from momwire._bspline_kernels import _ek_factor, _ek_factor_floored

K = 2 * np.pi / 21.4  # 14 MHz, the study's band


def _cquad(f, lo, hi, pts=()):
    pts = sorted(p for p in pts if lo < p < hi)
    kw = dict(points=pts or None, epsabs=0, epsrel=1e-12, limit=400)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        re = quad(lambda x: f(x).real, lo, hi, **kw)[0]
        im = quad(lambda x: f(x).imag, lo, hi, **kw)[0]
    return re + 1j * im


def _phi_reference(z, a1, a2, k=K):
    """K₂ by adaptive quadrature over φ, graded toward φ = 0 where a
    near-equal pair's integrand peaks."""

    def R(ph):
        return np.sqrt(z * z + (a1 - a2) ** 2 + 4 * a1 * a2 * np.sin(ph / 2) ** 2)

    br = [0.0, *np.geomspace(1e-9, 1.0, 10), np.pi]
    tot = 0j
    for lo, hi in zip(br[:-1], br[1:]):
        tot += _cquad(lambda p: np.exp(-1j * k * R(p)) / (4 * np.pi * R(p)), lo, hi)
    return tot / np.pi


_RADII = [
    (0.008, 0.0005),  # 16 mm -> 1 mm
    (0.0005, 0.008),  # transposed
    (0.008, 0.004),
    (0.008, 0.0079),
    (0.008, 0.00799992),  # 1.00001:1
    (0.05, 0.001),
]


@pytest.mark.parametrize(("a1", "a2"), _RADII)
def test_kernel_matches_phi_quadrature(a1, a2):
    t = np.array([0.0, 1e-4, 1e-2, 0.3, 1.0, 3.0, 5.999, 6.0, 6.001, 20.0, 300.0])
    z = t * np.sqrt(a1 * a2)
    got = EK.ring_kernel2(z, a1, a2, K)
    ref = np.array([_phi_reference(zi, a1, a2) for zi in z])
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= 1e-12


def test_kernel_is_symmetric_in_the_radii():
    z = np.geomspace(1e-5, 1.0, 40)
    a = EK.ring_kernel2(z, 0.008, 0.0005, K)
    b = EK.ring_kernel2(z, 0.0005, 0.008, K)
    assert np.max(np.abs(a - b) / np.abs(a)) <= 1e-15


def test_kernel_reduces_to_one_radius_first_order():
    """|K₂ - K| is first order in δ = a₂ - a₁ at every z ≠ 0, so the two-
    radius kernel joins the one-radius one continuously."""
    a = 0.008
    z = np.array([1e-3, 4e-3, 2e-2, 0.1])
    one = EK.ring_kernel(z, a, K)
    errs = [
        np.abs(EK.ring_kernel2(z, a, a * (1 + r), K) - one) / np.abs(one)
        for r in (1e-3, 1e-5, 1e-7)
    ]
    assert np.all(errs[2] <= 1e-6)
    for e0, e1 in zip(errs[:-1], errs[1:]):
        assert np.all(e1 / e0 == pytest.approx(1e-2, rel=0.05))


# --------------------------------------------------------------------------
# moments
# --------------------------------------------------------------------------
# pair0/1/5/7 joined the slow lane at momwire 0.82.0 release prep: CI medians
# 6.8/6.8/6.1/7.6 s against the 5 s default-lane budget. pair6 stays as the
# PR-lane guard.
_PAIRS = [  # (h_i, h_j, c, sigma_i, sigma_j, a_i, a_j)
    pytest.param(  # touching across 16:1
        (0.01, 0.01, 0.01, 1, 1, 0.008, 0.0005), marks=pytest.mark.slow
    ),
    pytest.param(  # the transposed radii
        (0.01, 0.01, 0.01, 1, 1, 0.0005, 0.008), marks=pytest.mark.slow
    ),
    # The nested reference takes 10-35 s on these three: the push lane.
    pytest.param(  # touching, near-equal radii
        (0.01, 0.01, 0.01, 1, 1, 0.008, 0.0079), marks=pytest.mark.slow
    ),
    pytest.param(  # nearer still
        (0.01, 0.01, 0.01, 1, 1, 0.008, 0.00799), marks=pytest.mark.slow
    ),
    pytest.param(  # antiparallel source
        (0.005, 0.012, 0.005, 1, -1, 0.008, 0.004), marks=pytest.mark.slow
    ),
    pytest.param(  # across a 2 mm gap
        (0.01, 0.01, 0.012, 1, 1, 0.008, 0.0005), marks=pytest.mark.slow
    ),
    (0.01, 0.01, 0.05, 1, 1, 0.008, 0.0005),  # far
    pytest.param(  # overlapping spans
        (0.003, 0.02, -0.004, 1, 1, 0.008, 0.001), marks=pytest.mark.slow
    ),
]


def _K2(z, a1, a2):
    return EK.ring_kernel2(np.array([z]), a1, a2, K)[0]


def _pair_reference(hi, hj, c, si, sj, a1, a2, p, q):
    """∫∫ u^p v^q K₂(c + σ_j v - σ_i u) du dv by nested adaptive quadrature,
    breakpoints at the near-singular point ±δ: independent of the W_pq
    reduction and of the piece rule."""
    dl = abs(a1 - a2)

    def inner(u):
        vs = sj * (si * u - c)
        return _cquad(
            lambda v: u**p * v**q * _K2(c + sj * v - si * u, a1, a2),
            0.0,
            hj,
            (vs - dl, vs, vs + dl),
        )

    us = [si * c, si * (c + sj * hj)]
    return _cquad(inner, 0.0, hi, [x + e for x in us for e in (-dl, 0.0, dl)])


@pytest.mark.parametrize("pair", _PAIRS)
def test_pair_moments_match_nested_quadrature(pair):
    J = EK.pair_moments2(*pair, K, 3)
    for p, q in ((0, 0), (2, 1), (1, 2)):
        ref = _pair_reference(*pair, p, q)
        assert abs(J[p, q] - ref) <= 1e-11 * abs(ref)


def test_far_vectorised_matches_scalar():
    pair = (0.01, 0.01, 0.05, 1, 1, 0.008, 0.0005)
    far = EK.pair_moments2_far(*([v] for v in pair), K, 3)[:, :, 0]
    near = EK.pair_moments2(*pair, K, 3)
    assert np.max(np.abs(far - near) / np.abs(near)) <= 1e-13


@pytest.mark.parametrize(
    ("c", "h", "a1", "a2"),
    [
        (-0.005, 0.01, 0.008, 0.0005),  # observer at the centre
        (0.0, 0.01, 0.0005, 0.008),  # at the start
        (-0.01, 0.01, 0.008, 0.0079),  # at the end, near-equal radii
        (0.0001, 0.01, 0.008, 0.00799),
        (0.03, 0.01, 0.008, 0.0005),  # far
        (-0.0001, 0.02, 0.008, 0.004),
    ],
)
def test_point_moments_match_adaptive_quadrature(c, h, a1, a2):
    dl = abs(a1 - a2)
    got = EK.point_moments2(c, h, a1, a2, K, 3)
    for q in range(3):
        ref = _cquad(
            lambda v, q=q: v**q * _K2(c + v, a1, a2), 0.0, h, (-c - dl, -c, -c + dl)
        )
        assert abs(got[q] - ref) <= 1e-11 * abs(ref)


def test_moments_reduce_to_one_radius():
    """Touching and self-overlapping pairs, where the one-radius moments
    carry the log singularity: the two-radius ones approach them as
    a₂ → a₁."""
    a = 0.008
    for pair in ((0.01, 0.01, 0.01, 1, 1), (0.01, 0.01, 0.0, 1, 1)):
        one = EK.pair_moments(*pair, a, K, 2)
        errs = [
            np.max(np.abs(EK.pair_moments2(*pair, a, a * (1 + r), K, 2) - one))
            / np.max(np.abs(one))
            for r in (1e-3, 1e-5, 1e-7)
        ]
        assert errs[0] > errs[1] > errs[2]
        assert errs[2] <= 1e-6
    one = EK.point_moments(-0.005, 0.01, a, K, 2)
    two = EK.point_moments2(-0.005, 0.01, a, a * (1 + 1e-7), K, 2)
    assert np.max(np.abs(two - one) / np.abs(one)) <= 1e-6


def _line(n_fat=8, n_thin=12, h=0.006):
    h_ = np.full(n_fat + n_thin, h)
    x0 = np.arange(n_fat + n_thin) * h
    a = np.where(np.arange(n_fat + n_thin) < n_fat, 0.008, 0.0005)
    lab = (a == 0.0005).astype(np.int64)
    return x0, np.ones_like(x0), h_, a, lab


def test_cross_rows_are_the_block_and_zero_on_same_radius_pairs():
    x0, sgn, h, a, lab = _line()
    whole = EK.coaxial_cross_block(x0, sgn, h, a, lab, K, 2)
    rows = EK.CoaxialCrossRows(x0, sgn, h, a, lab, K, 2)
    parts = np.concatenate([rows.rows(r0, min(r0 + 7, 20)) for r0 in (0, 7, 14)], 2)
    assert np.array_equal(parts, whole)
    same = lab[:, None] == lab[None, :]
    assert np.all(whole[:, :, same] == 0)
    assert np.all(whole[0, 0, ~same] != 0)
    # dedup: the two uniform runs' cross pairs cost O(n), not n_fat·n_thin
    assert rows.evaluated <= 2 * 20


def test_point_cross_rows_match_the_scalar_primitive():
    x0, sgn, h, a, lab = _line()
    xo = x0 + 0.5 * h
    got = EK.PointCrossRows(x0, sgn, h, a, lab, K, 2).rows(xo, a, lab)
    for o, j in ((7, 8), (8, 7), (2, 15)):
        want = EK.point_moments2(x0[j] - xo[o], h[j], a[o], a[j], K, 2)
        assert np.max(np.abs(got[:, o, j] - want)) <= 1e-15 * np.max(np.abs(want))
    assert np.all(got[:, lab[:, None] == lab[None, :]] == 0)


_CROSS_ACCEL = pytest.mark.skipif(
    not EK._HAVE_POINT_CROSS_ACCEL, reason="accelerator without the two-radius rows"
)


@_CROSS_ACCEL
def test_the_cxx_two_radius_kernel_is_the_numpy_kernel():
    """`_exact_kernel_cross_inline.h`'s K2 against `ring_kernel2` across the
    near/far switch and through z = 0 (measured <= 1.1e-16 relative)."""
    from momwire._accel import acc

    for a1, a2 in ((0.004, 0.0015), (0.0015, 0.004), (0.02, 0.019)):
        z = np.concatenate([[0.0], np.geomspace(1e-7, 2.0, 400)])
        want = EK.ring_kernel2(z, a1, a2, K)
        got = acc.exact_kernel_ring_kernel2(z, a1, a2, K)
        assert np.max(np.abs(got - want) / np.abs(want)) <= 1e-14


@_CROSS_ACCEL
def test_the_cxx_cross_rows_are_the_numpy_rows():
    """`PointCrossRowsAccel` against the numpy `PointCrossRows`: the same
    pairs, zero on the same-radius ones, the values to the near series'
    roundoff (measured 2.9e-16 of the block's largest), the same
    deduplication, and bits independent of the thread count and of how the
    observers are cut into windows."""
    x0, sgn, h, a, lab = _line()
    xo = x0 + 0.5 * h
    want = EK.PointCrossRows(x0, sgn, h, a, lab, K, 2).rows(xo, a, lab)
    src = EK.PointCrossRowsAccel(x0, sgn, h, a, lab, K, 2)
    got = src.rows(xo, a, lab)
    same = lab[:, None] == lab[None, :]
    assert np.all(got[:, same] == 0) and np.all(got[0, ~same] != 0)
    assert np.max(np.abs(got - want)) <= 1e-14 * np.max(np.abs(want))
    assert src.evaluated <= 2 * 20
    for nt in (1, 4):
        one = EK.PointCrossRowsAccel(x0, sgn, h, a, lab, K, 2, n_threads=nt)
        parts = [
            one.rows(xo[r0 : r0 + 7], a[r0 : r0 + 7], lab[r0 : r0 + 7])
            for r0 in (0, 7, 14)
        ]
        assert np.concatenate(parts, 1).tobytes() == got.tobytes()
    # A capped cache recomputes what it declined, to the same bits.
    capped = EK.PointCrossRowsAccel(x0, sgn, h, a, lab, K, 2, max_cached=3)
    again = [capped.rows(xo, a, lab) for _ in range(2)]
    assert again[0].tobytes() == again[1].tobytes() == got.tobytes()


@_CROSS_ACCEL
def test_the_cxx_scalar_primitive_is_the_numpy_one():
    from momwire._accel import acc

    rules = EK._point_rules()
    for c, h, a1, a2 in (
        (-0.003, 0.006, 0.004, 0.0015),
        (0.0, 0.006, 0.0015, 0.004),
        (-0.006, 0.006, 0.004, 0.0015),
        (0.02, 0.006, 0.004, 0.0015),
        (-0.0031, 0.0062, 0.004, 0.00399),
    ):
        want = EK.point_moments2(c, h, a1, a2, K, 2)
        got = acc.exact_kernel_point_moments2(c, h, a1, a2, K, 2, rules)
        assert np.max(np.abs(got - want)) <= 1e-14 * np.max(np.abs(want)), (c, a1, a2)


def test_the_solvers_take_the_cxx_cross_rows_where_it_serves(monkeypatch):
    built = []
    for name in ("PointCrossRows", "PointCrossRowsAccel"):
        orig = getattr(EK, name)

        def spy(*a, _orig=orig, _name=name, **kw):
            built.append(_name)
            return _orig(*a, **kw)

        monkeypatch.setattr(EK, name, spy)
    _z(RazorSolver, exact_kernel=True)
    want = "PointCrossRowsAccel" if EK._HAVE_POINT_CROSS_ACCEL else "PointCrossRows"
    assert built and set(built) == {want}


_COAXIAL_CROSS_ACCEL = pytest.mark.skipif(
    not EK._HAVE_COAXIAL_CROSS_ACCEL,
    reason="accelerator without the two-radius Galerkin rows",
)


@_COAXIAL_CROSS_ACCEL
@pytest.mark.parametrize("nd", [2, 3])
def test_the_cxx_coaxial_cross_rows_are_the_numpy_rows(nd):
    """`CoaxialCrossRowsAccel` (momwire#1615, the text the standalone bs2
    route compiles) against the numpy `CoaxialCrossRows`: the same pairs,
    zero on the same-radius ones, the values to the einsums' roundoff
    (measured 9.5e-17 of the block's largest), the same deduplication, and
    bits independent of the thread count, of how the rows are cut into
    windows and of a capped cache."""
    x0, sgn, h, a, lab = _line()
    want = EK.CoaxialCrossRows(x0, sgn, h, a, lab, K, nd).rows(0, 20)
    src = EK.CoaxialCrossRowsAccel(x0, sgn, h, a, lab, K, nd)
    got = src.rows(0, 20)
    same = lab[:, None] == lab[None, :]
    assert np.all(got[:, :, same] == 0) and np.all(got[0, 0, ~same] != 0)
    assert np.max(np.abs(got - want)) <= 1e-14 * np.max(np.abs(want))
    assert src.evaluated <= 2 * 20
    for nt in (1, 4):
        one = EK.CoaxialCrossRowsAccel(x0, sgn, h, a, lab, K, nd, n_threads=nt)
        parts = [one.rows(r0, min(r0 + 7, 20)) for r0 in (0, 7, 14)]
        assert np.concatenate(parts, 2).tobytes() == got.tobytes()
    capped = EK.CoaxialCrossRowsAccel(x0, sgn, h, a, lab, K, nd, max_cached=3)
    again = [capped.rows(0, 20) for _ in range(2)]
    assert again[0].tobytes() == again[1].tobytes() == got.tobytes()


@_COAXIAL_CROSS_ACCEL
def test_the_cxx_coaxial_cross_rows_handle_reversed_and_touching_pairs():
    """A line walked in both directions (σ = -1 on the thin run) and a step
    with the thin run's first segment touching the fat run's last: the
    near rule's pieces through the step against the numpy primitive."""
    x0, sgn, h, a, lab = _line()
    n_fat = int(np.sum(lab == 0))
    sgn = sgn.copy()
    sgn[n_fat:] = -1.0
    x0 = x0.copy()
    x0[n_fat:] = x0[n_fat:] + h[n_fat:]
    want = EK.CoaxialCrossRows(x0, sgn, h, a, lab, K, 2).rows(0, 20)
    got = EK.CoaxialCrossRowsAccel(x0, sgn, h, a, lab, K, 2).rows(0, 20)
    assert np.max(np.abs(got - want)) <= 1e-14 * np.max(np.abs(want))


def test_bs2_takes_the_cxx_coaxial_cross_rows_where_it_serves(monkeypatch):
    built = []
    for name in ("CoaxialCrossRows", "CoaxialCrossRowsAccel"):
        orig = getattr(EK, name)

        def spy(*a, _orig=orig, _name=name, **kw):
            built.append(_name)
            return _orig(*a, **kw)

        monkeypatch.setattr(EK, name, spy)
    _z(BSplineSolver, exact_kernel=True)
    want = (
        "CoaxialCrossRowsAccel" if EK._HAVE_COAXIAL_CROSS_ACCEL else "CoaxialCrossRows"
    )
    assert built and set(built) == {want}


# --------------------------------------------------------------------------
# the solvers
# --------------------------------------------------------------------------
_L, _S = 0.47, 0.06
STEP = dict(
    wires=[
        np.array([(0.0, 0.0, -_L / 2), (0.0, 0.0, -_S)]),
        np.array([(0.0, 0.0, -_S), (0.0, 0.0, _S)]),
        np.array([(0.0, 0.0, _S), (0.0, 0.0, _L / 2)]),
    ],
    n_per_edge_per_wire=[[17], [11], [17]],
    wire_radius=[0.0025, 0.01, 0.0025],
    junctions=[[(0, "end"), (1, "start")], [(1, "end"), (2, "start")]],
    feeds=[(1, _S, 1.0)],
)
DIPOLE = dict(
    wires=[np.array([(0.0, 0.0, -_L / 2), (0.0, 0.0, _L / 2)])],
    n_per_edge_per_wire=[[41]],
    wire_radius=0.00425,
)


def _solver(cls, deck=STEP, **kw):
    extra = (
        dict(nec5_quadrature=True)
        if cls is RazorSolver
        else dict(n_qp_pair_same_edge=8, n_qp_pair=16)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(wavelength=1.0, **deck, **extra, **kw)


def _z(cls, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(_solver(cls, **kw).compute_impedance()[0])


def _reduced(z, a, k):
    R = np.sqrt(np.asarray(z) ** 2 + a * a)
    return np.exp(-1j * k * R) / (4 * np.pi * R)


def _extended(z, a, k, b=None):
    R = np.sqrt(np.asarray(z) ** 2 + a * a)
    b = None if b is None else np.broadcast_to(b, np.shape(R))
    return np.exp(-1j * k * R) / (4 * np.pi * R) * _ek_factor(R, a, k, b)


# The fill's own kernels between two radii, in `ring_kernel2`'s argument
# order (observer, source). Razor regularises R by the SOURCE radius and
# takes the observer's as Eq 89's b; bs2 regularises by the OBSERVER's.
_FILL2 = {
    (RazorSolver, False): lambda z, a1, a2, k: _reduced(z, a2, k),
    (RazorSolver, True): lambda z, a1, a2, k: _extended(z, a2, k, a1),
    (BSplineSolver, False): lambda z, a1, a2, k: _reduced(z, a1, k),
}


@pytest.mark.parametrize(("cls", "ek"), sorted(_FILL2, key=str))
def test_swapped_kernel_reproduces_the_fill(monkeypatch, cls, ek):
    """Hand both corrections the fill's own kernels: Z must not move beyond
    the fill's quadrature accuracy (measured <= 9.5e-10). bs2's EK floors
    its factor at R = b across a step (`_ek_factor_floored`), a kink its
    fixed-order pair rule integrates to only ~1e-6 (falling with
    n_qp_pair), so that kernel is held by the next test instead."""
    base = _z(cls, extended_kernel=ek)
    k1 = _extended if ek else _reduced
    monkeypatch.setattr(EK, "_USE_ACCEL", False)
    monkeypatch.setattr(EK, "ring_kernel", k1)
    monkeypatch.setattr(
        EK,
        "ring_kernel_split",
        lambda z, a, k: (np.zeros(np.shape(z), complex), k1(z, a, k)),
    )
    monkeypatch.setattr(EK, "ring_kernel2", _FILL2[cls, ek])
    built = []
    for name in ("CoaxialCrossRows", "PointCrossRows"):
        orig = getattr(EK, name)

        def spy(*a, _orig=orig, **kw):
            built.append(1)
            return _orig(*a, **kw)

        monkeypatch.setattr(EK, name, spy)
    swapped = _z(cls, extended_kernel=ek, exact_kernel=True)
    assert built, "the two-radius correction did not run"
    assert np.max(np.abs(swapped - base)) <= 1e-8 * np.max(np.abs(base))


@pytest.mark.parametrize("cls", [RazorSolver, BSplineSolver])
def test_one_line_does_not_depend_on_the_fill_kernel(cls):
    """Every pair of a straight stepped wire is on one line, so the exact-
    kernel Z replaces every kernel the fill put there: reduced and EK fills
    must give the same Z to roundoff (they differed by 0.1-0.4 Ω before
    the cross-radius pairs were covered)."""
    zr = _z(cls, extended_kernel=False, exact_kernel=True)
    ze = _z(cls, extended_kernel=True, exact_kernel=True)
    assert np.max(np.abs(zr - ze)) <= 1e-12 * np.max(np.abs(zr))


@pytest.mark.parametrize("ek", [False, True])
@pytest.mark.parametrize("degree", [1, 2])
def test_bs2_galerkin_symmetry_with_a_step(degree, ek):
    """The fill regularises R by the observer's radius, so a stepped wire's
    Z is not symmetric (|Z - Zᵀ|/|Z| 7e-3 to 0.28 here); the two-radius
    kernel is symmetric in the radii and covers every cross pair of the
    line, so the corrected Z is (measured <= 3.5e-16)."""
    s = _solver(BSplineSolver, degree=degree, extended_kernel=ek, exact_kernel=True)
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = np.array(s._compute_Z_operator(geom, supp_seg, polys))
    assert np.abs(Z - Z.T).max() <= 1e-14 * np.abs(Z).max()


@pytest.mark.parametrize("cls", [RazorSolver, BSplineSolver])
def test_no_step_never_enters_the_two_radius_path(monkeypatch, cls):
    def boom(*_a, **_k):
        raise AssertionError("two-radius code ran on a deck without a step")

    for name in ("CoaxialCrossRows", "PointCrossRows", "ring_kernel2"):
        monkeypatch.setattr(EK, name, boom)
    _z(cls, deck=DIPOLE, exact_kernel=True)


def test_lines_list_only_stepped_lines():
    s = _solver(BSplineSolver)
    geom = s._build_geometry()
    seg_a = s._seg_radius(geom)
    args = (geom["seg_l"], geom["seg_r"], geom["tangents"], seg_a)
    labels, groups = SEAM.coaxial_groups(*args)
    lines = SEAM.coaxial_lines(*args, labels)
    assert len(groups) == 2 and len(lines) == 1
    S, lab = lines[0]
    assert S.size == seg_a.size and np.unique(lab).size == 2


def test_bs2_ek_swap_converges_with_the_pair_rule(monkeypatch):
    """bs2's EK fill across the step, swapped in: the residual is the fill's
    own pair rule on the floored factor's kink, so it falls as n_qp_pair
    rises (measured 4.4e-6, 4.9e-7 at 16, 32)."""

    def k2(z, a1, a2, k):
        R = np.sqrt(np.asarray(z) ** 2 + a1 * a1)
        b = np.broadcast_to(a2, np.shape(R))
        return np.exp(-1j * k * R) / (4 * np.pi * R) * _ek_factor_floored(R, a1, k, b)

    monkeypatch.setattr(EK, "_USE_ACCEL", False)
    monkeypatch.setattr(EK, "ring_kernel", _extended)
    monkeypatch.setattr(
        EK,
        "ring_kernel_split",
        lambda z, a, k: (np.zeros(np.shape(z), complex), _extended(z, a, k)),
    )
    monkeypatch.setattr(EK, "ring_kernel2", k2)
    res = []
    for nqp in (16, 32):
        kw = dict(extended_kernel=True, n_qp_pair_same_edge=8, n_qp_pair=nqp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = BSplineSolver(wavelength=1.0, **STEP, **kw).compute_impedance()[0]
            sw = BSplineSolver(
                wavelength=1.0, **STEP, exact_kernel=True, **kw
            ).compute_impedance()[0]
        res.append(abs(complex(np.ravel(sw)[0]) - complex(np.ravel(base)[0])))
    assert res[1] < 0.3 * res[0]
    assert res[1] <= 1e-6 * abs(complex(np.ravel(base)[0]))


# --------------------------------------------------------------------------
# main's seams (the v2 rebuild): bs2's deduplicated base, razor's plan cache
# and its k-series
# --------------------------------------------------------------------------

# One pitch on both sides of the step (h = 0.01 everywhere), so the fat and
# thin runs share a length class and the cross blocks gather; and the same
# with the upper arm reversed and bent at its tip, so a Hankel block, a
# second line and an off-line edge are in it too.
_H = 0.01
STEP_UNIFORM = dict(
    wires=[
        np.array([(0.0, 0.0, -0.24), (0.0, 0.0, -0.06)]),
        np.array([(0.0, 0.0, -0.06), (0.0, 0.0, 0.06)]),
        np.array([(0.0, 0.0, 0.06), (0.0, 0.0, 0.24)]),
    ],
    n_per_edge_per_wire=[[18], [12], [18]],
    wire_radius=[0.0025, 0.01, 0.0025],
    junctions=[[(0, "end"), (1, "start")], [(1, "end"), (2, "start")]],
    feeds=[(1, 0.06, 1.0)],
)
STEP_FLIPPED = dict(
    wires=[
        np.array([(0.0, 0.0, -0.24), (0.0, 0.0, -0.06)]),
        np.array([(0.0, 0.0, -0.06), (0.0, 0.0, 0.06)]),
        np.array([(0.04, 0.0, 0.28), (0.0, 0.0, 0.24), (0.0, 0.0, 0.06)]),
    ],
    n_per_edge_per_wire=[[18], [12], [4, 18]],
    wire_radius=[0.0025, 0.01, 0.0025],
    junctions=[[(0, "end"), (1, "start")], [(1, "end"), (2, "end")]],
    feeds=[(1, 0.06, 1.0)],
)
_STEPS = {"step": STEP, "uniform": STEP_UNIFORM, "flipped": STEP_FLIPPED}


def _bs2(deck, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BSplineSolver(wavelength=1.0, exact_kernel=True, **deck, **kw)


def _Zop(s):
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.array(s._compute_Z_operator(geom, supp_seg, polys))


def _rel(got, want):
    return float(np.max(np.abs(got - want)) / np.max(np.abs(want)))


def _base_spy(monkeypatch):
    made = []
    init = EKB.BaseRows.__init__

    def spy(self, **kw):
        init(self, **kw)
        made.append(self)

    monkeypatch.setattr(EKB.BaseRows, "__init__", spy)
    return made


@pytest.mark.parametrize("rows", [None, 1, 5])
@pytest.mark.parametrize("ek", [False, True])
@pytest.mark.parametrize("deck", sorted(_STEPS))
def test_bs2_cross_base_dedup_matches_the_masked_per_pair_base(
    monkeypatch, deck, ek, rows
):
    """The cross job's base through `BaseRows`' cross mode (the gather forced
    onto every block) against the per-pair `_exact_kernel_base_J` masked to
    the cross pairs, #1440's route. The tolerance is #1421's bar: these
    decks sit far inside its 16·eps·X/h_min, and the gather may only move a
    moment by coordinate roundoff."""
    monkeypatch.setattr(EKB, "TOEPLITZ_MIN_PAIRS", 0)
    if rows is not None:
        monkeypatch.setattr(_bspline, "_EXACT_KERNEL_ROWS_OVERRIDE", rows)
    s = _bs2(_STEPS[deck], extended_kernel=ek)
    monkeypatch.setattr(_bspline, "_EXACT_KERNEL_BASE_DEDUP", False)
    want = _Zop(s)
    made = _base_spy(monkeypatch)
    monkeypatch.setattr(_bspline, "_EXACT_KERNEL_BASE_DEDUP", True)
    got = _Zop(s)
    cross = [b for b in made if b.cross]
    assert cross, "the cross job's BaseRows was not built"
    assert _rel(got, want) <= 1e-13
    if deck != "step":
        # the fat and thin runs share a pitch: the cross blocks gathered
        assert sum(b.toeplitz_blocks for b in cross) > 0


@pytest.mark.parametrize("ek", [False, True])
def test_bs2_cross_base_rows_produce_only_cross_pairs(monkeypatch, ek):
    """`BaseRows(cross_lab=...)` is the per-pair base on every cross pair and
    writes nothing on a same-radius pair (those are the groups')."""
    monkeypatch.setattr(EKB, "TOEPLITZ_MIN_PAIRS", 0)
    s = _bs2(STEP_UNIFORM, extended_kernel=ek)
    geom = s._build_geometry()
    ((S, lab, g_edges),) = s._exact_kernel_lines(geom)
    ek_spec, ladder = s._exact_kernel_fill_spec(geom, s.k)
    base = EKB.BaseRows(
        seg_l=geom["seg_l"],
        seg_r=geom["seg_r"],
        seg_a=s._seg_radius(geom),
        tangents=geom["tangents"],
        S=S,
        g_edges=g_edges,
        k=s.k,
        d=s.degree,
        n_qp_pair=s.n_qp_pair,
        n_qp_same_edge=s.n_qp_pair_same_edge,
        ek=ek_spec,
        ladder=ladder,
        ek_se=_bspline._EK_SAME_EDGE if ek else None,
        n_segs_total=int(geom["n_segs_total"]),
        cross_lab=lab,
    )
    n = S.size
    got = np.concatenate([base.rows(r0, min(n, r0 + 7)) for r0 in range(0, n, 7)], 2)
    want = s._exact_kernel_base_J(geom, S, g_edges, s.k)
    same = lab[:, None] == lab[None, :]
    assert np.all(got[:, :, same] == 0)
    scale = np.max(np.abs(want[:, :, ~same]))
    assert np.max(np.abs(got[:, :, ~same] - want[:, :, ~same])) <= 1e-13 * scale
    assert base.toeplitz_blocks > 0


@pytest.mark.parametrize("deck", sorted(_STEPS))
def test_bs2_row_windows_match_the_whole_route(monkeypatch, deck):
    s = _bs2(_STEPS[deck], extended_kernel=True)
    monkeypatch.setattr(_bspline, "_EXACT_KERNEL_ROW_WINDOWS", False)
    whole = _Zop(s)
    monkeypatch.setattr(_bspline, "_EXACT_KERNEL_ROW_WINDOWS", True)
    monkeypatch.setattr(_bspline, "_EXACT_KERNEL_ROWS_OVERRIDE", 3)
    assert _rel(_Zop(s), whole) <= 1e-13


def _razor_z(deck, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = RazorSolver(
            wavelength=1.0,
            exact_kernel=True,
            nec5_quadrature=True,
            extended_kernel=True,
            **deck,
            **kw,
        )
        return np.asarray(s._assemble_Z(s._build_geometry(), s.k))


@pytest.fixture
def _plans():
    saved = (ES.ENABLED, PLAN.ENABLED)
    PLAN.clear()
    yield
    ES.ENABLED, PLAN.ENABLED = saved
    PLAN.clear()


def _bits(a):
    return np.ascontiguousarray(a).tobytes()


@pytest.mark.parametrize("deck", sorted(_STEPS))
def test_razor_cross_jobs_join_the_plan_and_a_hit_adds_the_same_bits(_plans, deck):
    """The cross jobs' tables are in the plan (keyed on the same arrays),
    their moment source is rebuilt each call: a repeat call hits the plan
    and adds the bits of the uncached route; the one-radius groups' C++
    sources are still kept (a cross job does not spoil the rows cache)."""
    ES.ENABLED = False
    PLAN.ENABLED = False
    want = _razor_z(_STEPS[deck])
    PLAN.ENABLED = True
    PLAN.clear()
    first = _razor_z(_STEPS[deck])
    again = _razor_z(_STEPS[deck])
    assert PLAN.STATS["plan_hit"] == 1
    assert _bits(first) == _bits(want) and _bits(again) == _bits(want)
    (plan,) = PLAN._PLANS.values()
    assert sum(g.lab is not None for g in plan.groups) == 1
    if EK.point_accel_serves(K, 2):
        assert PLAN.STATS["rows_hit"] == 1


@pytest.mark.skipif(not ES._HAVE_ACCEL, reason="accelerator not built")
@pytest.mark.parametrize("deck", sorted(_STEPS))
def test_razor_series_never_serves_a_cross_pair(_plans, monkeypatch, deck):
    """`MOMWIRE_EXACT_KERNEL_SERIES=1` with cross jobs present: the series
    serves the one-radius groups (it ran), never a cross job (every series
    source is built for ONE radius), the cross pairs keep the direct rule,
    and Z is the direct route's to the series' drift budget."""
    ES.ENABLED = False
    want = _razor_z(_STEPS[deck])
    radii, cross = [], []
    init = ES.PointSeriesAccel.__init__

    def spy(self, x0, sgn, h, a, *rest, **kw):
        radii.append(a)
        init(self, x0, sgn, h, a, *rest, **kw)

    monkeypatch.setattr(ES.PointSeriesAccel, "__init__", spy)
    orig = EK.point_cross_rows

    def cspy(*a, **kw):
        cross.append(1)
        return orig(*a, **kw)

    monkeypatch.setattr(EK, "point_cross_rows", cspy)
    ES.ENABLED = True
    PLAN.clear()
    got = _razor_z(_STEPS[deck])
    assert radii and all(np.ndim(a) == 0 for a in radii)
    assert cross
    assert _rel(got, want) <= 1e-13
