"""momwire#1595 D1b — far columns on the steepest-descent path.

`six_columns` takes a FAR column (k_p rho >= 20, every member's k_p R >= 30,
its angles within two Gaussian widths of one another) on a fixed rule along
the path cos(w - theta) = 1 - j s^2: 96 Gauss nodes on s, a pseudo-node for
the subtracted Zenneck pole, and the keyhole down k_m's cut when the path
passes left of k_m. The C++ column twin builds the same rule
(`mw1595::build`), so the gates here are the house twin gates:

  * the twin against ITS OWN walk at 1e-12 relative (the vector scale), on
    rows that exercise each piece: the bare path (eps 1), the keyhole alone,
    and the pole with the keyhole at grazing;
  * the walk against the point walk (`six_point`, the reference every route
    answers to) at the route's own 1e-10;
  * the switch: `sdp=False` is the head / mid / tail rule, near columns are
    untouched bit for bit, and a far column never touches the column cache.

The measured worst of the twin gate over the full study grid (AK
scratch/d1b-sdp, 492 far columns, six soils) is 2.2e-13; a one-ulp nudge to
k_p moves the walk's own answer by the same amount there (k_p R = 300 at
89.9 deg), so that is the conditioning, not a defect.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from momwire import _near_interface as ni
from momwire._sommerfeld_below import k_medium

pytestmark = pytest.mark.skipif(
    ni._nia is None or not hasattr(ni._nia, "six_columns_sdp_1595"),
    reason="the near-interface twin is not built",
)

C0 = 299792458.0
EPS0 = 8.8541878128e-12


def _ground(f, er, sig):
    w = 2.0 * math.pi * f
    k2 = w / C0
    eps_t = er - 1j * sig / (w * EPS0)
    return eps_t, k2, k_medium(eps_t, k2)


def _rel(got, ref):
    """Relative on the VECTOR scale, as `test_near_interface_columns_895`."""
    return float(np.max(np.abs(got - ref))) / max(float(np.max(np.abs(ref))), 1e-300)


def _twin(k_p, k_m, rho, zs, zps, **kw):
    zs = np.ascontiguousarray(zs, dtype=float)
    zps = np.ascontiguousarray(np.broadcast_to(zps, zs.shape), dtype=float)
    return ni._nia.near_interface_six_columns(
        k_p,
        k_m,
        np.array([rho]),
        np.array([0, zs.size], dtype=np.intp),
        zs,
        zps,
        float(ni._LAM_MULT),
        int(ni._COLUMN_P),
        float(ni._DETOUR),
        1,
        ni._GX,
        ni._GW,
        **kw,
    )


def _point(f, k_p_r, theta_deg):
    k_p = 2.0 * math.pi * f / C0
    r = k_p_r / k_p
    th = math.radians(theta_deg)
    return r * math.sin(th), r * math.cos(th)


# (soil, MHz, k_p R, theta, the pieces the rule must carry there)
ROWS = [
    ((1.0, 0.0), 14.0, 300, 60.0, set()),
    ((3.0, 1e-4), 3.6, 3000, 89.9, {"keyhole"}),
    ((13.0, 5e-3), 29.7, 300, 89.9, {"pole", "keyhole"}),
    ((80.0, 5.0), 3.6, 3000, 89.9, {"pole", "keyhole"}),
    ((30.0, 0.03), 14.0, 60, 25.0, set()),  # near the k_p rho >= 20 edge
]


def _pieces(k_p, k_m, rho, zs):
    theta, kr = ni._sdp_theta(rho, zs, k_p)
    got = set()
    pole = ni._sdp_pole(k_p, k_m, theta, rho)
    if pole is not None and abs(np.sqrt(kr) * pole[0]) <= ni._SDP_A_MAX:
        got.add("pole")
    if k_m.imag < 0.0 and ni._sdp_encloses(k_p, k_m, theta, np.sqrt(ni._SDP_TAIL / kr)):
        got.add("keyhole")
    return got


@pytest.mark.parametrize(("soil", "mhz", "kr", "theta", "pieces"), ROWS)
@pytest.mark.parametrize("zp", [-0.1, -1.0])
def test_the_twin_is_its_walk_on_a_far_column(soil, mhz, kr, theta, pieces, zp):
    eps_t, k_p, k_m = _ground(mhz * 1e6, *soil)
    rho, z = _point(mhz * 1e6, kr, theta)
    zs = np.array([z])
    assert ni._sdp_theta(rho, zs, k_p) is not None, "the row must be a far column"
    assert _pieces(k_p, k_m, rho, zs) == pieces, "the row no longer exercises its piece"
    ref = ni.six_columns(eps_t, k_p, rho, zs, zp)
    assert _rel(_twin(k_p, k_m, rho, zs, zp)[0], ref[0]) <= 1e-12


@pytest.mark.parametrize("soil", [(4.0, 1e-3), (13.0, 5e-3), (80.0, 5.0)])
def test_a_many_member_column_with_mixed_sources(soil):
    """Ward's far probe column (momwire#1582's deck: 643 m out, 760-775 m
    up) at 14.182 MHz, with two source depths in one column."""
    eps_t, k_p, k_m = _ground(14.182e6, *soil)
    zs = np.linspace(760.0, 775.0, 9)
    zps = np.where(np.arange(9) % 3 == 0, -0.025, -0.05)
    assert ni._sdp_theta(643.0, zs, k_p) is not None
    ref = ni.six_columns(eps_t, k_p, 643.0, zs, zps)
    got = _twin(k_p, k_m, 643.0, zs, zps)
    assert max(_rel(got[i], ref[i]) for i in range(9)) <= 1e-12


# The point walk at k_p R = 3000 takes seconds: those rows are slow-lane.
POINT_ROWS = [
    pytest.param(*r[:4], marks=pytest.mark.slow) if r[2] >= 3000 else r[:4]
    for r in ROWS[:4]
]


@pytest.mark.parametrize(("soil", "mhz", "kr", "theta"), POINT_ROWS)
def test_the_walk_answers_to_the_point_walk(soil, mhz, kr, theta):
    eps_t, k_p, _ = _ground(mhz * 1e6, *soil)
    rho, z = _point(mhz * 1e6, kr, theta)
    got = ni.six_columns(eps_t, k_p, rho, np.array([z]), -0.1)[0]
    ref = ni.six_point(eps_t, k_p, rho, z, -0.1, rtol=1e-12)
    assert _rel(got, ref) <= 1e-10


def test_the_switch_off_is_the_head_mid_tail_rule(monkeypatch):
    eps_t, k_p, k_m = _ground(29.7e6, 13.0, 5e-3)
    rho, z = _point(29.7e6, 300, 89.9)
    zs = np.array([z])
    monkeypatch.setattr(ni, "_SDP", False)
    assert ni._sdp_theta(rho, zs, k_p) is None
    ref = ni.six_columns(eps_t, k_p, rho, zs, -0.1)
    assert _rel(_twin(k_p, k_m, rho, zs, -0.1, sdp=False)[0], ref[0]) <= 1e-12
    assert ni._columns_lanes_kw() == {"sdp": False}


def test_a_near_column_is_untouched_bit_for_bit():
    _, k_p, k_m = _ground(14e6, 13.0, 5e-3)
    zs = np.linspace(0.1, 10.0, 7)
    assert ni._sdp_theta(2.0, zs, k_p) is None
    on = _twin(k_p, k_m, 2.0, zs, -0.3)
    off = _twin(k_p, k_m, 2.0, zs, -0.3, sdp=False)
    assert np.array_equal(on.view(np.float64), off.view(np.float64))


def test_a_far_column_never_touches_the_column_cache():
    _, k_p, k_m = _ground(14.182e6, 13.0, 5e-3)
    cache = ni._nia.ColumnCache(1 << 24)
    _twin(k_p, k_m, 643.0, np.linspace(760.0, 775.0, 9), -0.05, cache=cache)
    assert (cache.hits, cache.misses, cache.entries) == (0, 0, 0)
