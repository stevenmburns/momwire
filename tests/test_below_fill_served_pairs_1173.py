"""momwire#1173 (the Beverage cost map): the below/below grid's LAZY fill is
sized from the pairs that are SERVED, i.e. those inside the R1 cap.

`remainder_field_proj_below` serves a pair past `_SOMM_BELOW_R1_CAP_LAMBDA_M`
as exactly zero (momwire#1053). Its grid query nevertheless sized the lazy
fill from the kernel's extremes over EVERY pair: a zeroed pair's grazing angle
and its R1 clamped to the cap. On antennaknobs' Beverage (a ground rod at each
end, 246 m = 10.7 lambda_m apart) the two rods' cross pairs therefore filled
the floor, low and mid bands of all three R1 zones -- 36 s of a 54 s cold
razor-2p solve, for values that are discarded. momwire#1187 had already
switched to the capped pairs' angles, but only when a past-cap pair lay UNDER
the grazing floor (and it kept R1 at the cap, so the far annulus still
filled). Whether a deck paid therefore flipped on a few hundredths of a degree:
at nseg 21 the cross pairs sat at 0.037 deg (just over the 0.0167 deg floor,
full cost), at nseg 84 under it (0.6 s).

These gates pin the fill EXTENT, not a time, on both branches of the
projection and on both sides of that cliff:

1. with one rod's pairs inside the cap and the other rod past it, the query
   materializes NO deferred region (the served pairs need only the eager
   inner steep band), every served entry is finite and every past-cap entry
   is exactly zero;
2. a block with no served pair at all fills nothing and is all zeros;
3. the negative control: with the cap disabled every pair is served, and the
   same query reaches the grazing bands and the far annulus -- so gate 1 can
   fail, and does on the pre-#1173 sizing.

The refusals are unchanged (the grid's `_refuse_out_of_domain` reads the same
extremes as before); `test_grazing_past_cap_1187.py` gates those.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _ground_refl
from momwire import _sommerfeld_below as below

C0 = 299792458.0
EPS0 = 8.8541878128e-12
FREQ_HZ = 1.83e6
AVERAGE = (13.0, 0.005)
SEP_M = 245.7  # the Beverage's rod separation
EAGER = frozenset(
    i for i in range(3 * below._N_BANDS) if i not in below._DEFERRED_REGIONS
)

_GRID = {}


def _smooth_surfaces(eps_t, k2, rr, tt, **_kw):
    """A stand-in for the direct evaluator: finite, smooth, nonzero. These
    gates read which regions are asked for and whether an entry is finite or
    zero, never a surface's value, so the real ~1-3 s eager fill buys nothing
    here."""
    base = (1.0 + rr) * (1.0 + np.cos(tt)) + 0.5j
    return {key: base * (i + 1) for i, key in enumerate(below._SURF_KEYS)}


def _grid():
    """One grid for the module: the fill is stubbed below (it RECORDS which
    deferred region was asked for and fills nothing), so no gate mutates it."""
    if "g" not in _GRID:
        k2 = 2.0 * np.pi * FREQ_HZ / C0
        om = 2.0 * np.pi * FREQ_HZ
        eps_t = _ground_refl.eps_tilde(AVERAGE, om, EPS0)
        k_m = below.k_medium(eps_t, k2)
        cap = below.below_r1_cap(k_m)
        real = below.iv_surfaces_direct_below
        below.iv_surfaces_direct_below = _smooth_surfaces
        try:
            g = below.SommerfeldGridBelow(eps_t, k2, cap, omega=om)
        finally:
            below.iv_surfaces_direct_below = real
        _GRID["g"] = (g, k2, k_m)
    return _GRID["g"]


@pytest.fixture
def asked(monkeypatch):
    """The deferred regions the projection asks for, filled with nothing."""
    g, _k2, _k_m = _grid()
    assert all(r["filled"] == (i in EAGER) for i, r in enumerate(g._regions))
    seen = set()

    def record(idx):
        # `_fill_region` is idempotent and `_ensure_for` asks it of every
        # region in reach, filled or not; only an UNFILLED one costs a fill.
        if not g._regions[idx]["filled"]:
            seen.add(int(idx))

    monkeypatch.setattr(g, "_fill_region", record)
    return seen


def _rods(top_depth, sep=SEP_M):
    """Two vertical 1.8 m rods `sep` apart: observers and sources on each
    axis, the shallowest `top_depth` under the interface. Same-rod pairs are
    steep (rho = 0) and short; cross pairs graze at ~2*top_depth/sep rad."""
    d_obs = np.array([top_depth, 0.3, 0.8, 1.5])
    d_src = np.array([top_depth * 0.9, 0.45, 1.0, 1.75])
    obs = np.vstack([[(x, 0.0, -d) for d in d_obs] for x in (0.0, sep)])
    src = np.vstack([[(x, 0.0, -d) for d in d_src] for x in (0.0, sep)])
    t = np.tile([0.0, 0.0, 1.0], (obs.shape[0], 1))
    ts = np.tile([0.0, 0.0, 1.0], (src.shape[0], 1))
    return obs, t, src, ts


def _branch(monkeypatch, accel):
    if not accel:
        monkeypatch.setattr(below, "_use_below_accel", lambda: False)
    elif not below._use_below_accel():
        pytest.skip("no below/below accelerator")


def _counts(monkeypatch):
    # Read tolerantly, so that on the pre-#1173 code this module fails on the
    # fill-extent assertions (what it gates) rather than on a missing name.
    routes = getattr(below, "_SERVED_FILL_ROUTES", {})
    for k in routes:
        monkeypatch.setitem(routes, k, 0)
    return routes


# top depth 0.02 m: cross pairs at ~0.009 deg -> under the floor (#1187's
# branch); 0.05 m: ~0.021 deg -> over it, the side that paid in full.
CLIFF = {"under-floor": 0.02, "over-floor": 0.05}


@pytest.mark.parametrize("side", list(CLIFF))
@pytest.mark.parametrize("accel", [True, False], ids=["cpp", "numpy"])
def test_the_fill_reaches_only_what_the_served_pairs_read(
    monkeypatch, asked, accel, side
):
    _branch(monkeypatch, accel)
    routes = _counts(monkeypatch)
    g, k2, k_m = _grid()
    obs, t, src, ts = _rods(CLIFF[side])
    d_obs, d_src = -obs[:, 2], -src[:, 2]
    rho = np.abs(obs[:, 0][:, None] - src[:, 0][None, :])
    hh = d_obs[:, None] + d_src[None, :]
    th = np.degrees(np.arctan2(hh, rho))
    past = np.sqrt(rho * rho + hh * hh) > g.r1_cap
    assert past.any() and (~past).any()
    floor = np.degrees(g.th_min)
    assert (th[past].min() < floor) == (side == "under-floor"), th[past].min()
    out = below.remainder_field_proj_below(obs, t, src, ts, 0.0, k2, k_m, g)
    assert asked == set(), f"deferred regions filled for zeroed pairs: {sorted(asked)}"
    assert np.all(out[past] == 0.0)
    assert np.all(np.isfinite(out[~past])) and np.any(out[~past] != 0.0)
    # The served sizing is what ran. On the under-floor side the numpy
    # branch is #1187's own (it already reads past-cap pairs at a served
    # pair's point); the C++ branch runs #1187's angle swap, then this sizing.
    if accel or side == "over-floor":
        assert routes.get("cpp" if accel else "numpy", 0) >= 1, routes


@pytest.mark.parametrize("accel", [True, False], ids=["cpp", "numpy"])
def test_a_block_with_no_served_pair_fills_nothing(monkeypatch, asked, accel):
    _branch(monkeypatch, accel)
    g, k2, k_m = _grid()
    obs, t, src, ts = _rods(0.05)
    # observers on rod A only, sources on rod B only: every pair is past it
    out = below.remainder_field_proj_below(
        obs[:4], t[:4], src[4:], ts[4:], 0.0, k2, k_m, g
    )
    assert asked == set()
    assert out.shape == (4, 4) and np.all(out == 0.0)


@pytest.mark.parametrize("accel", [True, False], ids=["cpp", "numpy"])
def test_negative_control_with_every_pair_served_the_fill_reaches_far(
    monkeypatch, asked, accel
):
    """The same over-floor query with the cap lifted (every pair served):
    the grazing bands and the far annulus ARE asked for. This is the extent
    the pre-#1173 sizing paid for a zeroed pair, so gate 1 can fail."""
    _branch(monkeypatch, accel)
    g, k2, k_m = _grid()
    monkeypatch.setattr(g, "r1_cap", np.inf)
    obs, t, src, ts = _rods(0.02, sep=0.9 * g.r1_max)
    below.remainder_field_proj_below(obs, t, src, ts, 0.0, k2, k_m, g)
    far = below.region_index(below._ZONE_FAR, below._BAND_FLOOR)
    assert far in asked, sorted(asked)


def _fresh_grid(monkeypatch):
    """A grid of its own whose deferred regions really fill (smoothly)."""
    monkeypatch.setattr(below, "iv_surfaces_direct_below", _smooth_surfaces)
    g0, k2, k_m = _grid()
    g = below.SommerfeldGridBelow(g0.eps_t, k2, g0.r1_cap, omega=g0.omega)
    return g, k2, k_m


def test_a_served_read_the_served_extremes_missed_falls_back_to_the_kernels(
    monkeypatch,
):
    """The C++ kernel routes each pair on its OWN hypot; the served extremes
    are numpy's. Should the two disagree across a seam, a served entry reads
    an unfilled region (NaN). The projection then sizes the fill from the
    kernel's extremes, as before #1173, and reads again. Forced here by
    handing the served sizing extremes that miss the served grazing pair."""
    if not below._use_below_accel():
        pytest.skip("no below/below accelerator")
    routes = _counts(monkeypatch)
    # One grazing pair INSIDE the cap in the far annulus (50 m, 0.11 deg),
    # plus a rod past the cap, so the served sizing runs.
    obs = np.array([(0.0, 0.0, -0.05), (0.0, 0.0, -1.0)])
    src = np.array([(50.0, 0.0, -0.05), (SEP_M, 0.0, -0.05)])
    t = np.tile([0.0, 0.0, 1.0], (2, 1))
    g, k2, k_m = _fresh_grid(monkeypatch)
    ref = below.remainder_field_proj_below(obs, t, src, t, 0.0, k2, k_m, g)
    assert routes["fallback"] == 0 and np.all(np.isfinite(ref))
    g, k2, k_m = _fresh_grid(monkeypatch)
    monkeypatch.setattr(
        below, "_served_extremes", lambda *a: (1.0, 0.5 * np.pi, 0.5 * np.pi)
    )
    got = below.remainder_field_proj_below(obs, t, src, t, 0.0, k2, k_m, g)
    assert routes["fallback"] == 1, routes
    assert np.array_equal(got, ref)
