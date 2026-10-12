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
* a deck without a radius step never enters the two-radius path.
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import quad

from momwire import BSplineSolver, RazorSolver
from momwire import _exact_kernel as EK
from momwire import _exact_kernel_seam as SEAM
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
_PAIRS = [  # (h_i, h_j, c, sigma_i, sigma_j, a_i, a_j)
    (0.01, 0.01, 0.01, 1, 1, 0.008, 0.0005),  # touching across 16:1
    (0.01, 0.01, 0.01, 1, 1, 0.0005, 0.008),  # the transposed radii
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
    (0.01, 0.01, 0.012, 1, 1, 0.008, 0.0005),  # across a 2 mm gap
    (0.01, 0.01, 0.05, 1, 1, 0.008, 0.0005),  # far
    (0.003, 0.02, -0.004, 1, 1, 0.008, 0.001),  # overlapping spans
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
