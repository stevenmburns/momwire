"""The point-observer family's far columns on the steepest-descent path
(momwire#1653): the D1b route of momwire#1595, on the same path, rules and
crossover (`_sdp_theta`), for the four keys the buried near-field readout
reads (dRhoV, dzV, dRhoW, dz'V).

Measured on a 7 MHz soil-A column at 45 deg, one thread: 0.20 ms a pair at
10 m and 0.18 ms at 16 km after, against 0.44 ms at 1 km and 5.1 ms at
16 km before -- the cost of a far point no longer grows with its range.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _near_interface as ni

C0 = 299792458.0

# (MHz, sigma, eps_r): the SPEC-class soils and the HF band ends.
SOILS = [(7.0, 0.005, 13.0), (14.0, 0.05, 13.0), (3.5, 1e-4, 5.0), (1.8, 0.001, 4.0)]


def _medium(f_mhz, sigma, eps_r):
    k = 2 * np.pi * f_mhz * 1e6 / C0
    w = 2 * np.pi * f_mhz * 1e6
    return k, eps_r - 1j * sigma / (w * 8.854187817e-12)


def _far_cases(k):
    """(rho, z) across the crossover k rho = 20 and out to 16 km."""
    out = []
    for r in (20.2 / k, 40.0 / k, 100.0, 1000.0, 4000.0, 16000.0):
        for th in (20.0, 45.0, 70.0, 85.0, 89.5):
            rho, z = r * np.sin(np.radians(th)), r * np.cos(np.radians(th))
            if ni._sdp_theta(rho, np.array([z]), k) is not None:
                out.append((rho, z))
    return out


@pytest.mark.parametrize("soil", SOILS, ids=[f"{s[0]}MHz-{s[1]}" for s in SOILS])
def test_the_far_route_is_the_column_rule_within_its_rtol(soil, monkeypatch):
    """Past the crossover the steepest-descent rule reads what the head /
    mid / tail rule reads, to the contour kernels' rtol (1e-10): measured
    <= 9.4e-11 over these soils (numpy walk against numpy walk)."""
    k, eps = _medium(*soil)
    cases = _far_cases(k)
    assert cases, "no far column in the ladder"
    worst = 0.0
    for rho, z in cases:
        zs, zps = np.array([z, 1.0001 * z]), np.array([-0.15, -0.3])
        far = ni.point_keys_columns(eps, k, rho, zs, zps)
        monkeypatch.setattr(ni, "_SDP", False)
        near = ni.point_keys_columns(eps, k, rho, zs, zps)
        monkeypatch.setattr(ni, "_SDP", True)
        worst = max(worst, float(np.max(np.abs(far - near) / np.abs(near))))
    assert worst < 1e-10, worst


@pytest.mark.slow  # the reference walk at rtol 1e-13 is ~4 s
def test_the_far_route_is_the_adaptive_walk_on_sea_water():
    """Where the column rule itself is off at grazing over sea water (2e-5 at
    89.5 deg, 28 MHz, eps 81 / sigma 4: the six keys' D1b rule measured 1e-5
    there too), the far route agrees with the tight adaptive walk
    (`point_keys_point` at rtol 1e-13): it is the better of the two."""
    k, eps = _medium(28.0, 4.0, 81.0)
    for r, th in ((100.0, 89.5), (68.16, 85.0), (1000.0, 60.0)):
        rho, z = r * np.sin(np.radians(th)), r * np.cos(np.radians(th))
        far = ni.point_keys_columns(eps, k, rho, np.array([z]), np.array([-0.15]))[0]
        walk = ni.point_keys_point(eps, k, rho, z, -0.15, rtol=1e-13)
        assert float(np.max(np.abs(far - walk) / np.abs(walk))) < 1e-10


@pytest.mark.skipif(not ni._HAVE_POINT_COLUMNS_ACCEL, reason="no point twin")
@pytest.mark.parametrize("soil", SOILS, ids=[f"{s[0]}MHz-{s[1]}" for s in SOILS])
def test_the_cxx_far_route_is_its_numpy_walk(soil):
    """The C++ twin takes the far route at the same crossover and reads the
    numpy walk's numbers on both sides of it, to the point twin's own bar."""
    k, eps = _medium(*soil)
    worst = 0.0
    for rho, z in _far_cases(k) + [(15.0 / k, 3.0), (5.0 / k, 1.0)]:
        twin = ni._point_columns_exact(
            eps, k, np.array([rho]), np.array([z]), np.array([-0.15])
        )[0]
        ref = ni.point_keys_columns(eps, k, rho, np.array([z]), np.array([-0.15]))[0]
        worst = max(worst, float(np.max(np.abs(twin - ref)) / np.max(np.abs(ref))))
    # The point twin's own bar (tests/test_near_interface_point_twin_1224.py,
    # derived on two platforms: Apple's libm and numpy's last bits): measured
    # here 2.2e-13 on Linux, 1.8e-12 on macOS arm64.
    assert worst < 1e-9, worst
