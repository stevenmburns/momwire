"""The exact ring kernel's point moments as a k-series on geometry-only
coefficients (momwire#1454 step 3, `_exact_kernel_series`).

What is held here:

* the series reproduces the direct rule (`_exact_kernel.point_moments`) on
  every one of its three pair rules and on both sides of each switch, and
  its static term agrees with an independent adaptive quadrature;
* the C++ route equals the numpy reference to a derived tolerance, and is
  bit-stable in the thread count;
* the coefficients are built once per geometry: a new wavenumber reuses
  them, a repeat at one wavenumber evaluates nothing, and the counters that
  say so move;
* razor's correction through the series equals the direct route's to the
  drift budget, takes the series only where its bounds hold, and with the
  switch off is the direct route.
"""

import warnings

import numpy as np
import pytest
from scipy import integrate, special

from momwire import (
    RazorSolver,
    _exact_kernel as EK,
    _exact_kernel_series as ES,
    _razor_exact_plan as P,
)

needs_accel = pytest.mark.skipif(not ES._HAVE_ACCEL, reason="accelerator not built")

# Both routes sum the same O(30) positive-and-alternating terms in different
# orders; the reference check against 30-digit arithmetic
# (scratch/1454-split/reference.py) puts each within 3e-15 of the truth on
# this grid, relative to the pair's static moment.
ROUTE_TOL = 2e-14
# The same sums in C++ and in numpy: only the order of additions differs.
TWIN_TOL = 5e-15

RADII = (0.001, 0.01, 0.1)
H_OVER_A = (0.05, 0.2, 1.0, 3.0, 6.0, 10.0, 100.0)
C_OVER_H = (
    -0.5,  # the self term
    -0.3,
    -1.0,  # the observer on the far end
    0.0,  # ... and on the near end
    1e-7,
    -0.999,
    0.5,  # the neighbour, from a centroid
    0.999,  # NEAR, just inside the switch
    1.0,  # the switch itself: at least its own length away
    1.5,
    4.0,
    5.9,
    6.1,
    30.0,
    300.0,
    -2.5,
    -80.0,
)


def _grid():
    for a in RADII:
        for hr in H_OVER_A:
            for cr in C_OVER_H:
                yield a, hr * a, cr * hr * a


def _direct(c, h, a, k):
    if ES.pair_is_near(c, h):
        return EK.point_moments(c, h, a, k, 2)
    return EK.point_moments_far(np.array([c]), np.array([h]), a, k, 2)[:, 0]


def test_the_three_rules_are_all_on_the_grid():
    kinds = {ES.pair_kind(c, h, a) for a, h, c in _grid()}
    assert kinds == {ES.NEAR, ES.CENTRED, ES.PLAIN_FAR}


@pytest.mark.parametrize("k", [0.0, 0.02, 0.15, 1.0, 3.0])
def test_the_series_is_the_direct_rule(k):
    """Every pair kind, on both sides of the near/far switch (c/h = 0.999 and
    1) and of the 6a switch, at every wavenumber the bounds admit."""
    seen = 0
    for a, h, c in _grid():
        if not ES.serves(k, ES.group_reach(np.array([h]), a)):
            continue
        Rc, mu = ES.point_series(c, h, a)
        # the bound the term count was sized for holds
        cap = ES.X_CENTRED if Rc else ES.X_MAX
        assert k * ES.pair_reach(c, h, a) <= cap
        got, want = ES.evaluate(Rc, mu, k), _direct(c, h, a, k)
        scale = abs(want[0]) * np.array([1.0, h])
        # the phase of a pair kR away is only known to kR ulps, in any rule
        tol = ROUTE_TOL + 4e-16 * k * (abs(c) + h)
        assert np.all(np.abs(got - want) <= tol * scale), (a, h / a, c / h)
        seen += 1
    assert seen > 100


@pytest.mark.parametrize(
    "a, h_over_a, c_over_h",
    [
        (0.01, 0.05, -0.5),  # a fat wire's self term
        (0.01, 0.2, -0.999),
        (0.01, 1.0, 0.0),
        (0.001, 100.0, -0.5),  # a thin wire's
        (0.01, 1.0, 0.999),  # NEAR, at the switch
        (0.01, 1.0, 1.0),  # PLAIN FAR, at the switch
        (0.001, 30.0, 0.999),
        (0.001, 30.0, 1.0),  # CENTRED, at the switch
        (0.01, 3.0, 1.9),
        (0.01, 3.0, 2.1),
    ],
)
def test_the_static_term_against_adaptive_quadrature(a, h_over_a, c_over_h):
    """μ_{0,q} = ∫ v^q K_static(c + v) dv, K_static = K(m)/(2π²ρ): scipy's
    adaptive rule with the singular point given, independent of every rule
    here."""
    h = h_over_a * a
    c = c_over_h * h

    def static(v, q):
        z = c + v
        rho2 = z * z + 4.0 * a * a
        return v**q * special.ellipkm1(z * z / rho2) / (2.0 * np.pi**2 * np.sqrt(rho2))

    pts = [-c] if 0.0 < -c < h else None
    _rc, mu = ES.point_series(c, h, a)
    for q in (0, 1):
        want, _err = integrate.quad(
            static, 0.0, h, args=(q,), points=pts, epsabs=0.0, epsrel=1e-13, limit=400
        )
        assert mu[0, q] == pytest.approx(want, rel=5e-12)


def test_complex_k_is_served_by_the_arithmetic():
    """No real-k assumption in the coefficients or the evaluation: a lossy
    wavenumber agrees with the ring average done by brute force."""
    a, h, c = 0.002, 0.05, 0.4
    k = 2.0 - 0.7j
    Rc, mu = ES.point_series(c, h, a)
    assert Rc > 0.0
    xg, wg = EK._gl01(40)
    phi = (np.arange(64) + 0.5) * (2.0 * np.pi / 64)
    z = c + h * xg
    R = np.sqrt(z[:, None] ** 2 + 4.0 * a * a * np.sin(phi[None, :] / 2.0) ** 2)
    K = (np.exp(-1j * k * R) / (4.0 * np.pi * R)).mean(axis=1)
    want = np.array([np.sum(h * wg * (h * xg) ** q * K) for q in (0, 1)])
    got = ES.evaluate(Rc, mu, k)
    assert np.all(np.abs(got - want) <= 1e-13 * abs(want[0]) * np.array([1.0, h]))


@needs_accel
def test_the_cpp_coefficients_are_the_numpy_ones():
    rules = ES._rules()
    for a, h, c in _grid():
        rc_py, py = ES.point_series(c, h, a)
        rc_cc, cc = ES._acc.exact_kernel_point_series(
            c, h, a, ES.X_MAX, ES.X_CENTRED, ES.SERIES_TOL, rules
        )
        assert rc_cc == rc_py
        assert cc.shape == py.shape
        n = np.arange(py.shape[0])[:, None]
        scale = abs(py[0, 0]) * ES.pair_reach(c, h, a) ** n * np.array([1.0, h])
        assert np.all(np.abs(py - cc) <= TWIN_TOL * scale), (a, h / a, c / h)


def _run(n, a, length):
    h = np.full(n, length / n)
    x0 = np.arange(n) * length / n
    return x0, np.ones(n), h, x0 + 0.5 * h


@needs_accel
@pytest.mark.parametrize(
    "n, a, length",
    [(59, 0.002, 59.4), (20, 0.05, 2.0), (3, 0.001, 0.5)],
    ids=["long-thin", "fat", "short"],
)
def test_rows_are_the_direct_rows_and_the_coefficients_are_built_once(n, a, length):
    """A uniform run seen from its centroids, the long one many wavelengths
    at the top wavenumber: the centred rule needs no more terms out there."""
    x0, sgn, h, obs = _run(n, a, length)
    ser = ES.PointSeriesAccel(x0, sgn, h, a, 2, max_cached=4 * n + 1024)
    ref = ES.PointSeriesRows(x0, sgn, h, a, 2)
    k_top = min(ES.X_MAX / ser.reach[0], ES.X_CENTRED / ser.reach[1])
    built = None
    for i, k in enumerate((0.02, 0.153, k_top)):
        want = EK.PointRowsAccel(x0, sgn, h, a, k, 2).rows(obs)
        got = ser.rows(obs, k)
        scale = np.abs(want[0]).max() * np.array([1.0, length / n])[:, None, None]
        assert np.max(np.abs(got - want) / scale) <= ROUTE_TOL
        assert np.max(np.abs(got - ref.rows(obs, k)) / scale) <= TWIN_TOL
        if built is None:
            built = ser.built
            assert built == n  # the mirror key: n unique pairs
        assert ser.built == built  # a new k integrates nothing
        assert ser.reused == i * built
        assert ser.evaluated == (i + 1) * built
    assert sum(ser.kinds) == built and ser.kinds[ES.NEAR] == 2
    # a repeat at one k evaluates nothing, and returns the same bits
    again = ser.rows(obs, k_top)
    assert ser.evaluated == 3 * built
    assert np.array_equal(again, got)
    assert 0 < ser.nbytes <= ES.PAIR_NBYTES * built


@needs_accel
def test_rows_are_bit_stable_in_threads_and_in_the_cache_bound():
    x0, sgn, h, obs = _run(31, 0.004, 3.0)
    want = ES.PointSeriesAccel(x0, sgn, h, 0.004, 2, n_threads=1).rows(obs, 0.4)
    for kw in ({"n_threads": 4}, {"max_cached": 0}, {"max_cached": 7}):
        ser = ES.PointSeriesAccel(x0, sgn, h, 0.004, 2, **kw)
        for _ in range(2):
            assert np.array_equal(ser.rows(obs, 0.4), want), kw
    assert ser.built > 31  # past the bound a pair is rebuilt, not remembered


@needs_accel
def test_a_wavenumber_beyond_the_bounds_is_refused_not_approximated():
    x0, sgn, h, obs = _run(4, 0.001, 4.0)
    ser = ES.PointSeriesAccel(x0, sgn, h, 0.001, 2)
    k = 1.01 * ES.X_MAX / ser.reach[0]
    assert not ES.serves(k, ser.reach)
    with pytest.raises(RuntimeError, match="series bound"):
        ser.rows(obs, k)
    # and the object still serves afterwards
    k = 0.5 * ES.X_MAX / ser.reach[0]
    want = EK.PointRowsAccel(x0, sgn, h, 0.001, k, 2).rows(obs)
    assert np.allclose(ser.rows(obs, k), want, rtol=1e-12, atol=0.0)


# --------------------------------------------------------------------------
# razor's correction through the series
# --------------------------------------------------------------------------
BENT = [np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.25), (0.15, 0.0, 0.40)])]
DIPOLE = [np.array([(0.0, 0.0, -0.235), (0.0, 0.0, 0.235)])]


def _solver(wires, npe, wavelength=1.0, a=0.0125):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RazorSolver(
            wires=wires,
            n_per_edge_per_wire=npe,
            wavelength=wavelength,
            wire_radius=a,
            exact_kernel=True,
            nec5_quadrature=True,
            extended_kernel=True,
        )


def _z(wires, npe, **kw):
    s = _solver(wires, npe, **kw)
    return np.asarray(s._assemble_Z(s._build_geometry(), s.k))


@pytest.fixture
def series_switch():
    saved = ES.ENABLED
    P.clear()
    yield
    ES.ENABLED = saved
    P.clear()


@needs_accel
@pytest.mark.parametrize(
    "wires, npe", [(DIPOLE, [[37]]), (BENT, [[9, 7]])], ids=["dipole", "bent"]
)
def test_razor_z_through_the_series_is_the_direct_z(series_switch, wires, npe):
    ES.ENABLED = False
    want = _z(wires, npe)
    assert P.SERIES_STATS["miss"] == 0 and P.SERIES_STATS["built"] == 0
    ES.ENABLED = True
    P.clear()
    got = _z(wires, npe)
    assert P.SERIES_STATS["built"] > 0  # the series ran
    assert P.STATS["rows_miss"] == 0  # and the direct rule did not
    # the drift budget: 1e-12 of the matrix (measured ~1e-15)
    assert np.max(np.abs(got - want)) <= 1e-13 * np.max(np.abs(want))


@needs_accel
def test_a_sweep_pays_the_coefficients_once(series_switch):
    ES.ENABLED = True
    lams = (1.00, 1.02, 1.04, 1.06, 1.08)
    for lam in lams:
        _z(BENT, [[9, 7]], wavelength=lam)
    st = dict(P.SERIES_STATS)
    assert (P.STATS["plan_miss"], P.STATS["plan_hit"]) == (1, 4)
    assert (st["miss"], st["hit"]) == (1, 4)
    built = st["built"]
    assert built > 0
    assert st["reused"] == 4 * built  # every later frequency found them
    assert st["evaluated"] == 5 * built
    # the same frequency again: nothing integrated, nothing evaluated
    _z(BENT, [[9, 7]], wavelength=lams[-1])
    assert P.SERIES_STATS["built"] == built
    assert P.SERIES_STATS["evaluated"] == 5 * built


@needs_accel
def test_the_sweep_is_history_free(series_switch):
    """A frequency's Z does not depend on what the cache held: the sweep's
    third point equals the same point solved first in an empty cache."""
    ES.ENABLED = True
    for lam in (1.00, 1.02):
        _z(BENT, [[9, 7]], wavelength=lam)
    warm = _z(BENT, [[9, 7]], wavelength=1.04)
    P.clear()
    cold = _z(BENT, [[9, 7]], wavelength=1.04)
    assert np.array_equal(warm, cold)


@needs_accel
def test_segments_too_long_for_the_bounds_keep_the_direct_rule(series_switch):
    """`serves` is the gate: one segment of 0.47 λ per wire is beyond the
    centred bound, so the group's moments come from the direct rule, and
    are the direct rule's bits."""
    wires = [np.array([(0.0, 0.0, -0.94), (0.0, 0.0, 0.94)])]
    s = _solver(wires, [[4]], a=0.2)
    reach = ES.group_reach(np.array([0.47]), 0.2)
    assert not ES.serves(s.k, reach)
    ES.ENABLED = False
    want = _z(wires, [[4]], a=0.2)
    ES.ENABLED = True
    P.clear()
    got = _z(wires, [[4]], a=0.2)
    assert P.SERIES_STATS["built"] == 0 and P.STATS["rows_miss"] == 1
    assert np.array_equal(got, want)


def test_the_series_is_opt_in():
    """Off unless asked for: the default route is the direct rule's bits
    (see `ENABLED`'s comment for why)."""
    import importlib
    import os

    assert "MOMWIRE_EXACT_KERNEL_SERIES" not in os.environ or ES.ENABLED
    env = os.environ.pop("MOMWIRE_EXACT_KERNEL_SERIES", None)
    try:
        assert importlib.reload(ES).ENABLED is False
        os.environ["MOMWIRE_EXACT_KERNEL_SERIES"] = "1"
        assert importlib.reload(ES).ENABLED is True
    finally:
        os.environ.pop("MOMWIRE_EXACT_KERNEL_SERIES", None)
        if env is not None:
            os.environ["MOMWIRE_EXACT_KERNEL_SERIES"] = env
        importlib.reload(ES)
