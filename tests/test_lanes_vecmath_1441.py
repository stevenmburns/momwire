"""The lanes' vector-math route (momwire#1441 step 4), OPT-IN.

`_sommerfeld._LANES_VECMATH` (a bit mask; `MOMWIRE_LANES_VECMATH` for a whole
run) routes the Sommerfeld lanes' transcendental functions through glibc's
libmvec instead of libm per lane (`_somm_lanes_inline.h`, MW_SL_VM_*). It is
off by default, and off is the lanes exactly as before. On is a different
table, so these gates are tolerances, not bit equalities:

- G-1441-V1  off is the default, and a call at the default is the call with
             `vecmath=0`, bit for bit, on all three lane kernels; no table key
             changes at the default.
- G-1441-V2  on, each family alone and all together, the above-ground and
             below/below six integrals stay within VM_TOL of the libm lanes,
             relative to each node's largest component.
- G-1441-V3  the switch reaches the kernels and the keys: a grid filled with
             it on is not the grid served with it off, and the binding is
             asked for the mode the build honours.
- G-1441-V4  a build that cannot honour the route (no libmvec: MSVC, macOS,
             musl) is asked for nothing and keys on nothing.
"""

import numpy as np
import pytest

from momwire import _near_interface as near
from momwire import _sommerfeld as somm
from momwire import _sommerfeld_below as below

ACC = somm._acc
NIA = near._nia
BUILT = int(getattr(ACC, "lanes_vecmath_1441", 0)) if ACC is not None else 0

pytestmark = pytest.mark.skipif(
    ACC is None
    or not hasattr(ACC, "lanes_vecmath_1441")
    or NIA is None
    or not hasattr(NIA, "lanes_vecmath_1441"),
    reason="the .so predates the lanes' vector-math switch: rebuild",
)
needs_libmvec = pytest.mark.skipif(
    BUILT == 0, reason="this build has no libmvec route (not glibc x86-64)"
)

K2 = 2.0 * np.pi
SOILS = [13.0 - 14.420993610649957j, 4.0 - 0.05j, 5.0e4 - 9.0e4j]
# The deviation the census measured is at most ~5e-12 of a node's scale
# (above ground at rtol 1e-6, where the quadrature's own error is ~1e-7);
# below/below ~1e-15. A tolerance two orders above the census and three
# below the tables' rtol.
VM_TOL = 1e-9
MODES = [1, 2, 4, 8, 15]


def _nodes():
    rho = np.array([1e-7, 0.003, 0.02, 0.1, 0.17, 0.3, 0.5, 1.4, 3.0, 0.25])
    h = np.array([0.0, 0.0, 0.004, 0.2, 0.06, 0.7, 0.0, 0.3, 0.01, 0.125])
    return rho, h


def _below_case():
    f_hz, er, sigma = 6.0078e6, 13.0, 0.005
    w = 2.0 * np.pi * f_hz
    eps_t = er - 1j * sigma / (w * 8.8541878128e-12)
    k_p = w / 299792458.0
    k_m = below.k_medium(eps_t, k_p)
    rho = np.array([0.0, 1e-9, 0.01, 0.3, 2.0, 7.5, 25.0, 60.0, 3.0, 0.5])
    h = np.array([0.05, 0.2, 0.002, 0.05, 0.2, 0.02, 1.0, 0.4, 3.0, 0.0001])
    return k_p, k_m, rho, h


def _assert_near(got, ref, tol):
    got, ref = np.asarray(got), np.asarray(ref)
    assert np.all(np.isfinite(got))
    scale = np.max(np.abs(ref), axis=1, keepdims=True)
    rel = np.max(np.abs(got - ref) / np.where(scale > 0, scale, 1.0))
    assert rel <= tol, rel


@pytest.mark.skipif(
    bool(__import__("os").environ.get("MOMWIRE_LANES_VECMATH")),
    reason="the run asked for the route by environment",
)
def test_g1441_v1_off_is_the_default_and_the_lanes():
    assert somm._LANES_VECMATH == 0
    assert somm._vecmath_key() == ()
    assert somm._vecmath_kw(ACC) == {}
    rho, h = _nodes()
    for eps_t in SOILS:
        a = ACC.somm_six_integrals_batch(eps_t, K2, rho, h, 1e-9, 0)
        b = ACC.somm_six_integrals_batch(eps_t, K2, rho, h, 1e-9, 0, 0, True, 0)
        assert a.tobytes() == b.tobytes()
    k_p, k_m, rho, h = _below_case()
    a = below._six_below_accel(k_p, k_m, rho, h, 1e-9, False)
    b = ACC.below_six_integrals_batch(
        float(k_p),
        complex(k_m),
        rho,
        h,
        1e-11,
        below._ADAPT_DEPTH,
        below._DETOUR,
        below._GX,
        below._GW,
        False,
        1e-9,
        below._ADAPT_DEPTH_COARSE,
        below._DETOUR_COARSE,
        below._GXC,
        below._GWC,
        below._MAX_TAIL_PANELS,
        **below._wa_accel_kw(),
        vecmath=0,
    )
    assert a[0].tobytes() == b[0].tobytes()


@needs_libmvec
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("eps_t", SOILS)
@pytest.mark.parametrize("rtol", [1e-6, 1e-9])
def test_g1441_v2_above_ground_within_tolerance(mode, eps_t, rtol):
    rho, h = _nodes()
    ref = ACC.somm_six_integrals_batch(eps_t, K2, rho, h, rtol, 0)
    got = ACC.somm_six_integrals_batch(eps_t, K2, rho, h, rtol, 0, vecmath=mode)
    _assert_near(got, ref, VM_TOL)


@needs_libmvec
@pytest.mark.parametrize("mode", MODES)
def test_g1441_v2_below_within_tolerance(monkeypatch, mode):
    k_p, k_m, rho, h = _below_case()
    ref = below._six_below_accel(k_p, k_m, rho, h, 1e-9, True)
    monkeypatch.setattr(somm, "_LANES_VECMATH", mode)
    got = below._six_below_accel(k_p, k_m, rho, h, 1e-9, True)
    _assert_near(got[0], ref[0], VM_TOL)


@needs_libmvec
def test_g1441_v3_the_switch_reaches_kernels_and_keys(monkeypatch):
    seen = []
    real = ACC.somm_six_integrals_batch

    def spy(*a, **kw):
        seen.append(kw.get("vecmath", 0))
        return real(*a, **kw)

    monkeypatch.setattr(ACC, "somm_six_integrals_batch", spy)
    monkeypatch.setenv("MOMWIRE_SOMM_CACHE", "0")
    monkeypatch.setattr(somm, "_LANES_VECMATH", 0)
    saved = [dict(somm._GRID_CACHE), dict(somm._NORM_CACHE)]
    somm._GRID_CACHE.clear()
    somm._NORM_CACHE.clear()
    try:
        args = (13.0 - 14.42j, K2, 0.3, K2 * 299792458.0)
        off = somm.get_grid(*args)
        assert set(seen) == {0}
        monkeypatch.setattr(somm, "_LANES_VECMATH", 15)
        seen.clear()
        on = somm.get_grid(*args)
        assert on is not off
        assert seen and set(seen) == {15 & BUILT}
        assert somm._vecmath_key() == (("vecmath-1441", 15 & BUILT),)
        monkeypatch.setattr(somm, "_LANES_VECMATH", 0)
        assert somm.get_grid(*args) is off
    finally:
        somm._GRID_CACHE.clear()
        somm._NORM_CACHE.clear()
        somm._GRID_CACHE.update(saved[0])
        somm._NORM_CACHE.update(saved[1])


def test_g1441_v4_a_build_without_libmvec_asks_for_nothing(monkeypatch):
    class NoRoute:
        lanes_vecmath_1441 = 0

    monkeypatch.setattr(somm, "_LANES_VECMATH", 15)
    assert somm._vecmath_kw(NoRoute) == {}
    assert somm._vecmath_key(NoRoute) == ()
    assert somm._vecmath_kw(object()) == {}
    # the scalar walk runs libm whatever the switch says
    assert somm._vecmath_kw(ACC, lanes=False) == {}
