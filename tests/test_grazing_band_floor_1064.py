"""momwire#1064: the fifth theta band -- the FLOOR band -- for below/below.

antennaknobs plan U9 extended the low band down to a 0.016667 deg floor. A band
fills as one region per R1 zone, so every deck reaching under 0.1 deg started
paying for the two new nodes under 0.05 deg (2.3-2.9x its cold solve). This
moves those cells into a band of their own, routed over [floor, 0.05) deg, so
that only a deck reaching under 0.05 deg pays for them.

Three things about it are invisible in the code, and each has its own gate:

  * The floor band's LATTICE runs one node past its routing EDGE. A 4-point
    stencil needs four nodes and [floor, 0.05] holds three, so the lattice is
    U9's own cells, 0.016667 ... 0.066667 deg. Routing reads the edge and
    interpolation reads the lattice, so the two are gated separately.
  * The top two nodes are the LOW band's first two, and each is filled ONCE,
    by the low band, which owns them. Its values must be the same bits
    whichever band a deck reaches first, or everything in [0.05, 0.1] deg
    would move with fill order.
  * theta = 0.05 deg routes to the LOW band (strict `<`), and nothing at or
    above it may touch the floor band.
"""

import math
import sys

import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from momwire import _sommerfeld_below as below  # noqa: E402
from momwire._sommerfeld import _SURF_KEYS  # noqa: E402
from test_below_fills_568 import force_numpy  # noqa: E402
from test_grazing_band_lo_935 import FREQS, SOILS, _deck, _grid, _record_fills  # noqa: E402

RI = below.region_index
ZONES = (below._ZONE_INNER, below._ZONE_NEAR, below._ZONE_FAR)

# The band's own interpolation bar, the 4.7e-4 every grazing band has matched
# since momwire#553 U2. U9 measured 3.9e-9 over these same cells, with this same
# stencil (`scratch/u9-multi-crossing/s_interp_L2.json`), so the gate is set
# 100x over that and still ~1000x under the bar.
FLOOR_BAND_BAR = 3.9e-7


@pytest.fixture(scope="module", autouse=True)
def _warm_the_inner_floor_band():
    """Fill the INNER zone's floor band on the shared deck, once.

    The floor band's nodes are the most expensive in the grid (22,239 tail
    panels at the floor), and filling it fills the low band's inner zone too.
    Only the inner zone: the dispatch gate below stays inside it, and the far
    zones' fill is what made #935's dispatch gate run 84 s when its sweep
    reached the floor.
    """
    g, *_ = _grid("A", 7e6)
    g._fill_region(RI(below._ZONE_INNER, below._BAND_FLOOR))


# ---------------------------------------------------------------------------
# the lattice and the edge
# ---------------------------------------------------------------------------


def test_the_floor_band_lattice_runs_one_node_past_its_edge():
    """Four nodes on U9's cells, and the top two ARE the low band's first two.

    Asserted on the constructed regions in every zone: the far zone builds its
    own lattices for the bands above, so "the inner zone is right" is not the
    whole question.
    """
    g, *_ = _grid("A", 7e6)
    edge = g.th_band_floor_hi
    assert below._SOMM_BELOW_BAND_FLOOR_NODES == 4
    for zone in ZONES:
        floor = g._regions[RI(zone, below._BAND_FLOOR)]
        lo = g._regions[RI(zone, below._BAND_LO)]
        assert floor["n_th"] == 4, f"zone {zone}: {floor['n_th']} floor-band nodes"
        assert floor["th0"] == g.th_min, f"zone {zone}: floor band starts off the floor"
        assert floor["dth"] == lo["dth"], f"zone {zone}: the two bands' dtheta differ"
        assert lo["th0"] == edge, (
            f"zone {zone}: the low band starts at {lo['th0']!r}, not at the "
            f"floor band's edge {edge!r}"
        )
        f_nodes, lo_nodes = floor["th_nodes"], lo["th_nodes"]
        assert f_nodes[1] < edge < f_nodes[-1], (
            f"zone {zone}: the edge is not inside the floor band's lattice"
        )
        for f, j in below._SOMM_BELOW_BAND_FLOOR_SHARED:
            assert abs(f_nodes[f] - lo_nodes[j]) <= 1e-15, (
                f"zone {zone}: floor node {f} ({f_nodes[f]!r}) is not low node "
                f"{j} ({lo_nodes[j]!r})"
            )
        assert f_nodes[2] == edge, "the 0.05 deg node is not the edge itself"


def test_theta_at_the_floor_edge_routes_to_the_low_band():
    """The strict `<` at the new edge, asserted on the FILL STATE (a region is
    materialized if and only if a query routes into it)."""
    g, eps_t, k2, om, lam_m = _grid("A", 7e6)
    reg = g._regions[RI(below._ZONE_INNER, below._BAND_LO)]
    r_node = reg["r0"] + reg["dr"] * 6
    floor_inner = RI(below._ZONE_INNER, below._BAND_FLOOR)
    lo_inner = RI(below._ZONE_INNER, below._BAND_LO)

    fresh = below.SommerfeldGridBelow(eps_t, k2, g.r1_max, omega=om)
    with _record_fills(fresh):
        fresh.eval(np.array([r_node]), np.array([g.th_band_floor_hi]))
        assert fresh._regions[lo_inner]["filled"], (
            "theta = th_band_floor_hi did not reach the LOW band"
        )
        assert not fresh._regions[floor_inner]["filled"], (
            "theta = th_band_floor_hi routed into the FLOOR band; the edge is the "
            "low band's, which is what keeps [0.05, 0.1] deg on 0.55.0's lattice"
        )
        fresh.eval(
            np.array([r_node]), np.array([np.nextafter(g.th_band_floor_hi, 0.0)])
        )
        assert fresh._regions[floor_inner]["filled"], (
            "one ulp below th_band_floor_hi did not reach the floor band"
        )


def test_the_old_domain_never_touches_the_floor_band():
    """Nothing at or above 0.05 deg may read the floor band's lattice."""
    g, eps_t, k2, om, lam_m = _grid("A", 7e6)
    floor_all = [RI(z, below._BAND_FLOOR) for z in ZONES]
    fresh = below.SommerfeldGridBelow(eps_t, k2, g.r1_max, omega=om)
    with _record_fills(fresh) as seen:
        for th_deg in (0.05, 0.0500001, 0.06, 0.0666667, 0.09, 0.1, 0.5, 1.0, 45.0):
            for r1l in (0.1, 0.5, 1.0, 2.0, 3.0):
                fresh.eval(np.array([r1l * lam_m]), np.radians([th_deg]))
        assert not any(fresh._regions[i]["filled"] for i in floor_all), (
            "a query at or above 0.05 deg materialized a floor-band region. "
            f"Regions reached: {sorted(seen)}"
        )
        assert seen, "no region was reached at all; the sweep is inert"


# ---------------------------------------------------------------------------
# each shared node filled once, owned by the low band
# ---------------------------------------------------------------------------


def _counting_surfaces(calls):
    """A stand-in for `iv_surfaces_direct_below` that records every theta it
    is asked for and returns a value that is a pure function of (R1, theta),
    so a copied column can be told from a re-evaluated one by value."""

    def fake(eps_t, k2, R1, theta, **_kw):
        th = np.asarray(theta, dtype=float)
        calls.append(np.unique(th))
        base = np.asarray(R1, dtype=float) + 1j * th
        return {key: base * (i + 1) for i, key in enumerate(_SURF_KEYS)}

    return fake


@pytest.mark.parametrize("order", ["floor_first", "low_first"])
@pytest.mark.parametrize("zone", ZONES)
def test_each_shared_node_is_filled_once_and_owned_by_the_low_band(
    monkeypatch, zone, order
):
    """The fill counter, per zone and in both fill orders.

    Each of the floor and low bands' distinct theta nodes is evaluated in
    exactly one call; the floor band's own call carries only its two lower
    nodes; and the floor band's top two columns hold exactly the low band's
    values. Cheap: the direct surfaces are replaced by a counter, so nothing
    here integrates.
    """
    calls = []
    monkeypatch.setattr(below, "iv_surfaces_direct_below", _counting_surfaces(calls))
    eps_t, k2, om, lam_m = _deck("A", 7e6)
    g = below.SommerfeldGridBelow(
        eps_t, k2, below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m, omega=om
    )
    floor_idx, lo_idx = RI(zone, below._BAND_FLOOR), RI(zone, below._BAND_LO)
    floor, lo = g._regions[floor_idx], g._regions[lo_idx]
    calls.clear()  # construction evaluates the eager regions; not counted

    if order == "floor_first":
        g._fill_region(floor_idx)
        g._fill_region(lo_idx)  # already filled by the floor band's fill
    else:
        g._fill_region(lo_idx)
        g._fill_region(floor_idx)
    g._fill_region(floor_idx)  # and neither refills

    assert len(calls) == 2, f"{len(calls)} evaluation calls, expected 2 (one per band)"
    by_band = {len(c): c for c in calls}
    assert np.array_equal(
        np.sort(np.concatenate(calls)), np.unique(np.concatenate(calls))
    ), "a theta node was evaluated twice"
    assert sorted(by_band) == [2, 4], f"call sizes {sorted(by_band)}, expected [2, 4]"
    assert np.array_equal(by_band[4], np.unique(lo["th_nodes"]))
    assert np.array_equal(by_band[2], np.unique(floor["th_nodes"][:2]))

    for f, j in below._SOMM_BELOW_BAND_FLOOR_SHARED:
        assert np.array_equal(floor["vals"][:, :, f], lo["vals"][:, :, j]), (
            f"floor column {f} is not low column {j}: the shared node was not copied"
        )
    rr, tt = np.meshgrid(floor["r_nodes"], lo["th_nodes"][[0, 1]], indexing="ij")
    owner = np.stack([(rr + 1j * tt) * (i + 1) for i in range(len(_SURF_KEYS))])
    assert np.array_equal(lo["vals"][:, :, :2], owner), (
        "the low band's shared columns are not its own evaluation"
    )


def test_the_floor_band_is_deferred_until_something_reaches_it():
    """A deck that reaches 0.07 deg pays nothing for the floor band."""
    g, eps_t, k2, om, lam_m = _grid("B", 7e6)
    fresh = below.SommerfeldGridBelow(eps_t, k2, g.r1_max, omega=om)
    floor_all = [RI(z, below._BAND_FLOOR) for z in ZONES]
    lo_all = [RI(z, below._BAND_LO) for z in ZONES]
    assert not any(fresh._regions[i]["filled"] for i in floor_all + lo_all), (
        "a grazing band was filled at construction"
    )
    with _record_fills(fresh):
        fresh.eval(np.array([1.0 * lam_m]), np.radians([0.07]))
        assert any(fresh._regions[i]["filled"] for i in lo_all), (
            "a 0.07 deg query did not fill the low band, so this gate is not "
            "measuring what it claims"
        )
        assert not any(fresh._regions[i]["filled"] for i in floor_all), (
            "a 0.07 deg query filled the FLOOR band, whose nodes cost up to three "
            "times a low-band node -- the deferral is per band"
        )
        fresh.eval(np.array([1.0 * lam_m]), np.radians([0.03]))
        assert any(fresh._regions[i]["filled"] for i in floor_all), (
            "a 0.03 deg query did not fill the floor band"
        )


# ---------------------------------------------------------------------------
# the two dispatches across the new edge
# ---------------------------------------------------------------------------


def test_both_dispatches_agree_across_the_floor_edge():
    """numpy `_interp` and C++ `proj_one_below` carry two copies of the band
    routing and the stride, and #1064 changed both: a 5-way selector and a zone
    stride of 5. The C++ side takes all three edges as explicit arguments, so
    this gates one layout described twice.

    Inside the INNER zone (R1 ~ 1 m, under `r_break`), which is what the module
    fixture warmed; the low-band points above the edge ride the same fill.
    """
    g, eps_t, k2, om, lam_m = _grid("A", 7e6)
    assert 1.0 < g.r_break, "the sweep must stay in the warmed inner zone"
    k_m = below.k_medium(eps_t, k2)
    rho = 1.0
    edge = math.degrees(g.th_band_floor_hi)
    floor = math.degrees(g.th_min)
    for thd in (floor, 0.02, 0.0333, edge - 1e-9, edge, edge + 1e-9, 0.06, 0.0666):
        hh = rho * math.tan(math.radians(thd))
        obs = np.array([[0.0, 0.0, -0.5 * hh]])
        src = np.array([[rho, 0.0, -0.5 * hh]])
        t = np.array([[1.0, 0.0, 0.0]])
        cpp = below.remainder_field_proj_below(obs, t, src, t, 0.0, k2, k_m, g)
        with force_numpy():
            npy = below.remainder_field_proj_below(obs, t, src, t, 0.0, k2, k_m, g)
        rel = abs(cpp[0, 0] - npy[0, 0]) / max(abs(npy[0, 0]), 1e-300)
        assert rel < 1e-12, f"theta = {thd}: cpp {cpp[0, 0]!r} vs numpy {npy[0, 0]!r}"


# ---------------------------------------------------------------------------
# accuracy and the panel cap (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_floor_band_interpolates_to_the_bar():
    """The floor band against the direct surfaces, at cell midpoints and thirds
    of [floor, 0.05] deg: where the queries it can receive are, and where a
    4-point stencil is at its worst."""
    worst, where = 0.0, None
    for soil in SOILS:
        for f in FREQS:
            g, eps_t, k2, om, lam_m = _grid(soil, f)
            g._ensure_band_floor()
            nodes = g._regions[RI(below._ZONE_INNER, below._BAND_FLOOR)]["th_nodes"][:3]
            th_q = np.concatenate(
                [nodes[:-1] + 0.5 * np.diff(nodes), nodes[:-1] + 0.33 * np.diff(nodes)]
            )
            for r1l in (0.2, 1.0, 1.9):
                r1 = np.full(th_q.shape, r1l * lam_m)
                got = g.eval(r1, th_q)
                ref = below.iv_surfaces_direct_below(
                    eps_t, k2, r1, th_q, rtol=1e-9, omega=om
                )
                for k in _SURF_KEYS:
                    a = np.asarray(got[k], dtype=complex)
                    b = np.asarray(ref[k], dtype=complex)
                    rel = np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))
                    if rel > worst:
                        worst, where = rel, (soil, f, r1l, k)
    assert worst < FLOOR_BAND_BAR, f"{worst:.3e} at {where}"


@pytest.mark.slow
def test_every_floor_band_node_converges_under_the_panel_cap():
    """The floor is placed by the panel cap, and a capped node is SILENT
    (momwire#841), so every floor-band node is asserted converged on every SPEC
    soil, with the headroom as an early warning. Measured worst 22239/24000 =
    92.7 % at (C, 21 MHz, R1 = 0.05 lambda_m), at the floor itself."""
    for soil in SOILS:
        for f in FREQS:
            eps_t, k2, om, lam_m = _deck(soil, f)
            h = below.Health()
            th = np.radians(
                below._SOMM_BELOW_TH_MIN_DEG
                + below._SOMM_BELOW_DTH_BAND_LO_DEG
                * np.arange(below._SOMM_BELOW_BAND_FLOOR_NODES)
            )
            for r1l in (0.0, 0.02, 0.05, 0.2, 1.0, 2.0):
                below.iv_surfaces_direct_below(
                    eps_t,
                    k2,
                    np.full(th.shape, r1l * lam_m),
                    th,
                    rtol=1e-9,
                    omega=om,
                    health=h,
                )
            assert h.nonconvergent == 0, (
                f"{soil}/{f}: {h.nonconvergent} floor-band nodes hit the panel cap "
                "and were silently extrapolated"
            )
            frac = h.max_tail_panels / below._MAX_TAIL_PANELS
            assert frac < 0.98, (
                f"{soil}/{f}: worst node used {h.max_tail_panels} of "
                f"{below._MAX_TAIL_PANELS} panels ({frac:.1%})"
            )
