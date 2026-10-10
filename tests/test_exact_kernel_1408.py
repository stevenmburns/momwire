"""The exact ring kernel on coaxial pairs, `exact_kernel=True` (momwire#1408).

Gates, in the order the correction is built:

* the kernel against an independent reference (elliptic K by scipy plus a
  320-point θ quadrature of the smooth dynamic part) at t = |z|/a in
  [1e-4, 1e3], ≤ 1e-11 relative, and its log split K = A ln|z| + B;
* the product-integration log rule, exact on polynomials;
* the pair moments against adaptive 1-D quadrature of the closed-form
  weight, and that weight against a 2-D tensor rule;
* the MECHANISM: with the kernel swapped for the reduced (or extended) one,
  `exact_kernel=True` must reproduce the ordinary fill — which is what proves
  the base moments are removed and the correction assembled on the right
  bases, junctions and port wires included;
* default off: the correction never runs and Z is bit-identical;
* the open-tube dipole converges monotonically to the exact-kernel reference
  (antennaknobs scratch/1396-exact-kernel REPORT.md, segment gap, avg
  readout), and the thin wire stays within the EK lane's spread;
* refusals.
"""

import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ellipkm1

from momwire import BSplineSolver, HMatrixSolver
from momwire import _exact_kernel as EK
from momwire import bspline as _bspline

K0 = 2 * np.pi  # wavelength 1


def _reference_kernel(z, a, k):
    """Independent of `_exact_kernel`: K = static elliptic part (scipy) plus
    the smooth dynamic part (e^{-jkR} - 1)/R by 2 x 160-point Gauss in θ."""
    z = np.abs(np.asarray(z, float))
    rho2 = z * z + 4 * a * a
    ks = ellipkm1(z * z / rho2) / (2 * np.pi**2 * np.sqrt(rho2))
    x, w = np.polynomial.legendre.leggauss(160)
    out = np.zeros(z.shape, complex)
    for lo, hi in ((0, np.pi / 2), (np.pi / 2, np.pi)):
        th = 0.5 * (hi - lo) * (x + 1) + lo
        wt = 0.5 * (hi - lo) * w
        R = np.sqrt(z[..., None] ** 2 + 4 * a * a * np.sin(th) ** 2)
        out += (np.expm1(-1j * k * R) / R * wt).sum(-1)
    return ks + out / (4 * np.pi**2)


@pytest.mark.parametrize("d_over_lam", [0.001, 0.0085, 0.05])
def test_kernel_matches_reference(d_over_lam):
    a = d_over_lam / 2
    t = np.concatenate([np.geomspace(1e-4, 1e3, 71), [5.999999, 6.0, 6.000001]])
    got = EK.ring_kernel(t * a, a, K0)
    ref = _reference_kernel(t * a, a, K0)
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= 1e-11


def test_kernel_log_split_reproduces_kernel():
    a = 0.00425
    t = np.geomspace(1e-6, 5.99, 50)
    A, B = EK.ring_kernel_split(t * a, a, K0)
    K = EK.ring_kernel(t * a, a, K0)
    assert np.max(np.abs(A * np.log(t * a) + B - K) / np.abs(K)) <= 1e-13
    # the log coefficient's leading term: K ~ -ln|z| / (4 pi^2 a)
    assert A[0] == pytest.approx(-1 / (4 * np.pi**2 * a), rel=1e-6)
    with pytest.raises(ValueError):
        EK.ring_kernel_split(np.array([6.0 * a]), a, K0)


def test_log_rule_exact_on_polynomials():
    x, _w = EK._gl01(20)
    wl = EK._gl01_log(20)
    for m in range(20):
        assert np.dot(wl, x**m) == pytest.approx(-1.0 / (m + 1) ** 2, rel=1e-13)


_A = 0.00425
_PAIRS = [  # (h_i, h_j, c, sigma_i, sigma_j)
    (6 * _A, 6 * _A, 0.0, 1, 1),  # self
    (0.7 * _A, 0.7 * _A, 0.0, 1, 1),  # self, Δ < a
    (3 * _A, 1.3 * _A, 3 * _A, 1, 1),  # adjacent, unequal
    (3 * _A, 2 * _A, 1 * _A, 1, 1),  # overlapping spans
    (4 * _A, 4 * _A, 4.5 * _A, 1, 1),  # across a 0.5a gap
    (4 * _A, 4 * _A, 8.5 * _A, 1, -1),  # antiparallel, across the gap
    (4 * _A, 3 * _A, -0.5 * _A, -1, 1),
    (2 * _A, 2 * _A, 30 * _A, 1, 1),  # far
    (100 * _A, 100 * _A, 100 * _A, 1, 1),  # thin-wire adjacent
]


@pytest.mark.parametrize("pair", _PAIRS)
def test_weight_matches_2d_rule(pair):
    hi, hj, c, si, sj = pair
    x, w = np.polynomial.legendre.leggauss(40)
    x, w = (x + 1) / 2, w / 2
    u, v = hi * x, hj * x
    f = np.exp(3.0 * (c + sj * v[None, :] - si * u[:, None]) / hi)
    up = np.array([u**p for p in range(3)])
    vq = np.array([v**q for q in range(3)])
    ref = np.einsum("i,j,ij,pi,qj->pq", w * hi, w * hj, f, up, vq)
    corners = np.sort([0.0, sj * hj, -si * hi, sj * hj - si * hi])
    got = np.zeros((3, 3))
    for y0, y1 in zip(corners[:-1], corners[1:]):
        if y1 > y0:
            yy = y0 + (y1 - y0) * x
            ww = (y1 - y0) * w * np.exp(3.0 * (c + yy) / hi)
            got += np.einsum("i,ipq->pq", ww, EK._weight(yy, hi, hj, si, sj, 3))
    assert np.max(np.abs(got - ref) / np.abs(ref)) <= 1e-12


@pytest.mark.parametrize("pair", _PAIRS)
def test_pair_moments_match_adaptive_quadrature(pair):
    hi, hj, c, si, sj = pair
    J = EK.pair_moments(hi, hj, c, si, sj, _A, K0, 3)
    corners = np.sort([0.0, sj * hj, -si * hi, sj * hj - si * hi])
    for p, q in ((0, 0), (2, 1)):
        tot = 0j
        for y0, y1 in zip(corners[:-1], corners[1:]):
            if y1 <= y0:
                continue
            pts = [-c] if y0 < -c < y1 else None

            def g(y, p=p, q=q):
                Ky = EK.ring_kernel(np.array([c + y]), _A, K0)[0]
                return Ky * EK._weight(np.array([y]), hi, hj, si, sj, 3)[0, p, q]

            for part, unit in ((np.real, 1), (np.imag, 1j)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    val = quad(
                        lambda y, part=part: part(g(y)),
                        y0,
                        y1,
                        points=pts,
                        epsabs=0,
                        epsrel=1e-13,
                    )[0]
                tot += unit * val
        assert abs(J[p, q] - tot) <= 1e-11 * abs(tot)


def test_far_vectorised_matches_scalar():
    hi, hj, c, si, sj = _PAIRS[7]
    far = EK.pair_moments_far([hi], [hj], [c], [si], [sj], _A, K0, 3)[:, :, 0]
    near = EK.pair_moments(hi, hj, c, si, sj, _A, K0, 3)
    assert np.max(np.abs(far - near) / np.abs(near)) <= 1e-13


# --------------------------------------------------------------------------
# the solver
# --------------------------------------------------------------------------
L = 0.47
DIPOLE = [np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)])]


def _halves(w):
    return [
        np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, -w / 2)]),
        np.array([(0.0, 0.0, w / 2), (0.0, 0.0, L / 2)]),
    ]


def _endpair_kw(w):
    return dict(
        wires=_halves(w),
        feeds=[(0, (L - w) / 4, 0j)],
        junctions=[[(0, "end")], [(1, "start")]],
        junction_ports=[(0, 0j), (1, 0j)],
    )


def _solve(n, a, wires=DIPOLE, y=False, **kw):
    s = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[n]] * len(wires),
        wavelength=1.0,
        wire_radius=a,
        **kw,
    )
    if y:
        return s.compute_y_matrix()
    return complex(np.atleast_1d(s.compute_impedance()[0])[0])


def _reduced(z, a, k):
    R = np.sqrt(np.asarray(z) ** 2 + a * a)
    return np.exp(-1j * k * R) / (4 * np.pi * R)


def _extended(z, a, k):
    R = np.sqrt(np.asarray(z) ** 2 + a * a)
    kr = k * R
    c1 = 1 + 1j * kr
    c2 = 3 * c1 - kr * kr
    fac = 1 + 0.25 * a**4 / R**4 * c2 - 0.5 * a * a / R**2 * c1
    return np.exp(-1j * kr) / (4 * np.pi * R) * fac


_DECKS = {
    "dipole": dict(),
    "dipole-segment-gap": dict(feed_model="segment"),
    "end-port pair": dict(_endpair_kw(L / 75), y=True),
    "collinear junction + bend": dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.2)]),
            np.array([(0.0, 0.0, 0.2), (0.0, 0.0, 0.3), (0.1, 0.0, 0.4)]),
        ],
        n_edges=[[19], [9, 13]],
        junctions=[[(0, "end"), (1, "start")]],
    ),
}


@pytest.mark.parametrize("ek", [False, True])
@pytest.mark.parametrize("deck", sorted(_DECKS))
def test_swapped_kernel_reproduces_the_fill(monkeypatch, deck, ek):
    """The mechanism gate. Hand the correction the fill's OWN kernel: the
    exact-kernel solve must then equal the ordinary one, to the fill's
    quadrature accuracy (n_qp_pair_same_edge=8 holds that below 1e-9)."""
    kern = _extended if ek else _reduced
    # The swap reaches the numpy route only: the C++ route (momwire#1410)
    # compiles its kernel in. That route is held to this one by
    # tests/test_exact_kernel_accel_1410.py.
    monkeypatch.setattr(EK, "_USE_ACCEL", False)
    monkeypatch.setattr(EK, "ring_kernel", kern)
    monkeypatch.setattr(
        EK,
        "ring_kernel_split",
        lambda z, a, k: (np.zeros(np.shape(z), complex), kern(z, a, k)),
    )
    kw = dict(_DECKS[deck])
    n_edges = kw.pop("n_edges", None)
    wires = kw.pop("wires", DIPOLE)
    common = dict(extended_kernel=ek, n_qp_pair_same_edge=8, n_qp_pair=16, **kw)

    def go(**extra):
        s = BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=n_edges or [[37]] * len(wires),
            wavelength=1.0,
            wire_radius=0.00425,
            **common,
            **extra,
        )
        if kw.get("y"):
            return s.compute_y_matrix()
        return s.compute_impedance()[0]

    common.pop("y", None)
    base = np.asarray(go())
    calls = []
    orig = BSplineSolver._add_exact_kernel_correction

    def spy(self, *args):
        calls.append(1)
        return orig(self, *args)

    monkeypatch.setattr(BSplineSolver, "_add_exact_kernel_correction", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        swapped = np.asarray(go(exact_kernel=True))
    assert calls, "the correction did not run"
    assert np.max(np.abs(swapped - base)) <= 1e-9 * np.max(np.abs(base))


def test_swapped_kernel_detects_a_wrong_kernel(monkeypatch):
    """The same gate is not vacuous: a kernel 1e-6 off moves Z."""
    monkeypatch.setattr(EK, "_USE_ACCEL", False)
    monkeypatch.setattr(EK, "ring_kernel", lambda z, a, k: _reduced(z, a, k) * 1.000001)
    monkeypatch.setattr(
        EK,
        "ring_kernel_split",
        lambda z, a, k: (np.zeros(np.shape(z), complex), _reduced(z, a, k) * 1.000001),
    )
    base = _solve(37, 0.00425, n_qp_pair_same_edge=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        moved = _solve(37, 0.00425, n_qp_pair_same_edge=8, exact_kernel=True)
    assert abs(moved - base) > 1e-8 * abs(base)


@pytest.mark.parametrize("ek", [False, True])
def test_default_off_is_untouched(monkeypatch, ek):
    """exact_kernel=False never enters the correction, and Z is bit-identical
    to a solver built without the argument."""

    def boom(*_a, **_k):
        raise AssertionError("exact-kernel code ran with exact_kernel=False")

    monkeypatch.setattr(BSplineSolver, "_add_exact_kernel_correction", boom)
    monkeypatch.setattr(EK, "coaxial_block", boom)
    for kw in (dict(), _endpair_kw(L / 75)):
        wires = kw.pop("wires", DIPOLE)
        y = "junction_ports" in kw
        z0 = _solve(19, 0.00425, wires=wires, y=y, extended_kernel=ek, **kw)
        z1 = _solve(
            19, 0.00425, wires=wires, y=y, extended_kernel=ek, exact_kernel=False, **kw
        )
        assert np.array_equal(np.asarray(z0), np.asarray(z1))


# REPORT.md's exact-kernel reference, L = 0.47 lambda, segment gap w = L/n,
# gap-averaged readout (Richardson extrapolated; spread <= 0.011 ohm here).
_REF_FAT = {19: 76.0210 + 9.7860j, 37: 76.1928 + 8.6393j, 75: 76.4060 + 7.5038j,
            151: 76.6058 + 6.3722j}  # fmt: skip
_REF_THIN = {151: 68.2082 - 16.5203j, 301: 68.1498 - 16.6296j, 601: 68.0983 - 16.7267j}


def test_fat_tube_converges_monotonically_to_the_reference():
    """d/lambda 0.0085, Delta/a 5.8 -> 0.73: the error is negative in R and X
    at every rung, and shrinks by at least 1.6x per halving of Delta (it is
    first order, as the Galerkin reference's own ladder is)."""
    a = 0.0085 / 2
    err = {
        n: _solve(n, a, exact_kernel=True, feed_model="segment") - ref
        for n, ref in _REF_FAT.items()
    }
    ns = sorted(err)
    for n in ns:
        assert err[n].real < 0 and err[n].imag < 0
    for n0, n1 in zip(ns, ns[1:]):
        assert abs(err[n1]) * 1.6 < abs(err[n0])
    assert abs(err[151]) < 0.45  # Delta/a 0.73: -0.21-0.33j measured


def test_thin_wire_stays_within_the_ek_lane():
    """d/lambda 0.001 at Delta/a >= 10 (n <= 47 on this dipole): exact and EK
    agree to within the EK lane's own mesh spread."""
    a = 0.0005
    for n in (19, 37):
        z_ek = _solve(n, a, extended_kernel=True, feed_model="segment")
        z_ex = _solve(n, a, exact_kernel=True, feed_model="segment")
        spread = abs(
            _solve(2 * n + 1, a, extended_kernel=True, feed_model="segment") - z_ek
        )
        assert abs(z_ex - z_ek) < 0.2 * spread


def test_thin_wire_converges_to_the_reference():
    """d/lambda 0.001, Delta/a 6.2 -> 1.6: monotone, from below, first order."""
    a = 0.0005
    err = {
        n: _solve(n, a, exact_kernel=True, feed_model="segment") - ref
        for n, ref in _REF_THIN.items()
    }
    ns = sorted(err)
    for n in ns:
        assert err[n].real < 0 and err[n].imag < 0
    for n0, n1 in zip(ns, ns[1:]):
        assert abs(err[n1]) * 1.6 < abs(err[n0])


def test_point_gap_warns():
    with pytest.warns(RuntimeWarning, match="no limiting reactance"):
        BSplineSolver(wires=DIPOLE, n_per_edge_per_wire=[[9]], exact_kernel=True)


def test_refusals():
    with pytest.raises(NotImplementedError, match="1408"):
        BSplineSolver(
            wires=DIPOLE,
            n_per_edge_per_wire=[[9]],
            exact_kernel=True,
            use_singular_enrichment=True,
        )
    with pytest.raises(NotImplementedError, match="HMatrixSolver"):
        HMatrixSolver(wires=DIPOLE, n_per_edge_per_wire=[[9]], exact_kernel=True)
    s = BSplineSolver(
        wires=DIPOLE, n_per_edge_per_wire=[[9]], exact_kernel=True, feed_model="segment"
    )
    with pytest.raises(NotImplementedError, match="restricted-row"):
        s._refuse_exact_kernel_route(np.arange(3), False)


def test_swept_takes_the_per_k_loop():
    s = BSplineSolver(
        wires=DIPOLE,
        n_per_edge_per_wire=[[19]],
        wavelength=1.0,
        wire_radius=0.00425,
        exact_kernel=True,
        feed_model="segment",
    )
    assert not s._swept_batched_available()
    ks = np.array([0.98, 1.02]) * K0
    zs = np.asarray(s.compute_impedance_swept(ks))
    for i, kk in enumerate(ks):
        s2 = BSplineSolver(
            wires=DIPOLE,
            n_per_edge_per_wire=[[19]],
            wavelength=2 * np.pi / kk,
            wire_radius=0.00425,
            exact_kernel=True,
            feed_model="segment",
        )
        z2 = complex(np.atleast_1d(s2.compute_impedance()[0])[0])
        assert complex(np.ravel(zs[i])[0]) == pytest.approx(z2, rel=1e-12)


def test_module_reachable_from_bspline():
    assert _bspline._exact_kernel is EK
