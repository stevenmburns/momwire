"""The C++ column twin of the point-observer keys — momwire#1224 unit 1.

`_near_interface_point_accel.near_interface_point_columns` against its
reference, the numpy `point_keys_columns` route, through the one caller
(`point_radius_tables`). The house twin rule: relative, never bit (libmvec,
Amos and the ordered lane sum differ in the last bits).

The bar is 1e-9 of each key's largest value, set from two platforms' numbers
rather than copied from the six twin's 1e-12:
  * Linux/Skylake: worst 2.1e-15, every key, every soil;
  * macOS arm64 (CI, 2026-09-28): 9.2e-11, on dRhoW over soil A at the
    near-corner member (rho_eff = the 1 mm wire radius, s = 1.5 mm). dRhoW is
    the difference-type key (W = (gamma_p - gamma_m) V), and the corner's
    integral cancels hard, so Apple's libm and numpy's differing in the last
    bits is amplified ~1e5 there. Every other key and soil met 1e-12 on macOS.
A real defect (a wrong weight, a sign) is an O(1) error, far above either.

Also the exact zeros the numpy route has: the rho keys on the axis (J1(0) = 0)
and dRhoW at eps~ = 1 (W~ = 0), and the refusals, which stay the numpy walk's
because the caller raises them before the twin runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _ground_refl
from momwire import _near_interface as ni

pytestmark = pytest.mark.skipif(
    not ni._HAVE_POINT_COLUMNS_ACCEL, reason="the point twin is not built"
)

C0 = 299792458.0
F7 = 7e6
K7 = 2.0 * np.pi * F7 / C0
OM7 = 2.0 * np.pi * F7
EPS0 = 8.8541878128e-12
SOILS = {
    "eps1": 1.0 + 0j,
    "soilA": _ground_refl.eps_tilde((13.0, 0.005), OM7, EPS0),
    "highsigma": _ground_refl.eps_tilde((13.0, 5.0), OM7, EPS0),
    "sea": _ground_refl.eps_tilde((80.0, 5.0), OM7, EPS0),
}


_BAR = 1e-9  # see the module docstring: macOS measures 9.2e-11 at the corner


def _triples(seed=1224, n=300):
    rng = np.random.default_rng(seed)
    # The U3a triples (ordinary, near-corner, the corner class), the axis, and
    # a spread of columns with several members each.
    rho = np.concatenate([[0.3, 1.5, 0.01, 0.002, 0.0, 0.0], rng.uniform(0, 30, n)])
    z = np.concatenate([[0.2, 0.8, 0.005, 0.001, 0.2, 0.001], rng.uniform(0, 4, n)])
    zp = np.concatenate(
        [[-0.1, -0.4, -0.002, -0.0005, -0.1, -0.0005], -rng.uniform(1e-4, 2, n)]
    )
    return rho, z, zp


def _both(monkeypatch, eps, rho, z, zp, a=0.001):
    monkeypatch.setattr(ni, "_FORCE_NUMPY", True)
    ref = ni.point_radius_tables(eps, K7, rho, z, zp, a)
    monkeypatch.setattr(ni, "_FORCE_NUMPY", False)
    got = ni.point_radius_tables(eps, K7, rho, z, zp, a)
    return ref, got


@pytest.mark.parametrize("soil", sorted(SOILS))
def test_the_twin_is_the_numpy_route(soil, monkeypatch):
    ref, got = _both(monkeypatch, SOILS[soil], *_triples())
    for key in ni.POINT_KEYS:
        scale = np.max(np.abs(ref[key]))
        if scale == 0.0:
            assert np.max(np.abs(got[key])) == 0.0, key
            continue
        diff = np.abs(got[key] - ref[key])
        i = int(np.argmax(diff))
        rho, z, zp = (a.ravel()[i] for a in _triples())
        assert diff.max() <= _BAR * scale, (
            f"{key}: max |twin - numpy| = {diff.max():.3e} = {diff.max() / scale:.3e} "
            f"of max |{key}| ({scale:.3e}); worst at rho={rho:g} z={z:g} zp={zp:g}, "
            f"numpy {ref[key].ravel()[i]!r} twin {got[key].ravel()[i]!r}"
        )


def test_on_the_axis_the_rho_keys_are_an_exact_zero(monkeypatch):
    """wire_radius 0 keeps rho_eff = 0: J1(0) = 0 on every node, so the two
    rho keys are 0.0 exactly, and the z keys are not."""
    rho = np.zeros(3)
    z = np.array([0.2, 0.5, 1.0])
    zp = np.array([-0.1, -0.3, -0.05])
    _ref, got = _both(monkeypatch, SOILS["soilA"], rho, z, zp, a=0.0)
    assert np.all(got["gRhoV"] == 0) and np.all(got["gRhoW"] == 0)
    assert np.all(got["gzV"] != 0) and np.all(got["gzpV"] != 0)


def test_at_eps_one_the_w_gradient_vanishes(monkeypatch):
    _ref, got = _both(monkeypatch, SOILS["eps1"], *_triples(n=20))
    assert np.all(got["gRhoW"] == 0)


def test_the_refusals_are_the_numpy_walks(monkeypatch):
    monkeypatch.setattr(ni, "_FORCE_NUMPY", False)
    with pytest.raises(ValueError, match="need z >= 0 >= zp"):
        ni.point_radius_tables(
            SOILS["soilA"],
            K7,
            np.array([0.3]),
            np.array([-0.1]),
            np.array([-0.1]),
            0.001,
        )
