"""momwire#935: the fourth theta band, [0.05, 0.1] deg, for below/below.

momwire#838 put a third band under the grazing one and moved the floor from
1 deg to 0.1. This does it once more, one band lower, for the same reason and
by the same construction -- and the reason the SECOND addition needed its own
issue is that the blocker was not the lattice. Interpolation had six orders of
headroom at 0.1 deg already; what refused everything below it was
`_MAX_TAIL_PANELS`, which #935 raised 4000 -> 8000.

Two constraints are invisible in the code and would fail silently, so each has
its own gate here rather than being folded into a surface comparison. They are
#838's two, restated at the new seam:

  * dtheta must DIVIDE the band exactly, or the new band and the mid band
    share no node and every cell in the OLD domain moves.
  * theta = `th_band_lo_hi` must route to the MID band, i.e. a STRICT `<`,
    the coarser side of the seam owning the shared node.

A third is specific to this one: the new band must be lazy SEPARATELY from the
mid band. A low-band node is about twice the tail cost of a mid-band node
(7610 panels against 3868), so a deck that reaches 0.5 deg and not 0.05 must
not pay for it.
"""

import contextlib
import math
import sys

import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from momwire import _ground_refl  # noqa: E402
from momwire import _sommerfeld_below as below  # noqa: E402
from momwire._sommerfeld import _SURF_KEYS  # noqa: E402
from test_below_fills_568 import force_numpy  # noqa: E402

RI = below.region_index

C0 = 299792458.0
EPS0 = 8.8541878128e-12
SOILS = {"A": (13.0, 0.005), "B": (20.0, 0.03), "C": (5.0, 0.001)}
FREQS = (7e6, 21e6)

# The band's own interpolation bar, the same 4.7e-4 the #838 band had to match
# (momwire#553 U2, grazing band at dtheta = 1 deg / dr = 0.05 lambda_m).
#
# The standalone lattice probe measured 6.834e-10 for four uniform nodes over
# this range. On the REAL grid the R1 axis is in it too, so the bar here is
# set from a grid measurement rather than from the probe: worst 9.786e-10,
# at (C, 7 MHz, R1 = 1.9 lambda_m, IphiH), across soils A/B/C x 7/21 MHz x
# R1/lambda_m in {0.2, 1.0, 1.9} x cell midpoints and thirds
# (`scratch/935-study/probe_lo_band_grid_error.py`).
#
# Worth noting what that comparison says: adding the R1 axis moved the error
# from 6.8e-10 to 9.8e-10, i.e. theta interpolation is essentially the whole
# of it and the band is nowhere near being the limiting term. The blocker at
# these angles is the tail budget, not the lattice -- which is the finding
# that made #935 a cap change with a band attached rather than the reverse.
#
# Set 100x over the measurement, still 4800x under the 4.7e-4 it had to match.
LO_BAND_BAR = 9.8e-8


def _deck(soil, f):
    k2 = 2.0 * np.pi * f / C0
    om = 2.0 * np.pi * f
    eps_t = _ground_refl.eps_tilde(SOILS[soil], om, EPS0)
    return eps_t, k2, om, below.lambda_medium(eps_t, k2)


_GRIDS = {}


def _grid(soil, f):
    eps_t, k2, om, lam_m = _deck(soil, f)
    if (soil, f) not in _GRIDS:
        r1_max = below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m
        _GRIDS[(soil, f)] = below.SommerfeldGridBelow(eps_t, k2, r1_max, omega=om)
    return _GRIDS[(soil, f)], eps_t, k2, om, lam_m


def _lo_nodes(g, zone=None):
    """The LOW band's ACTUAL theta nodes in one R1 zone.

    Read from the region, not rebuilt as `th0 + dth*k`. The two are not the
    same float: the constructor builds nodes as `radians(th0_deg + dth_deg*k)`
    while `th0`/`dth` are stored as `radians(th0_deg)` / `radians(dth_deg)`,
    and re-accumulating in RADIAN space lands an ulp low for this band
    (measured: 0.0017453292519943294 against radians(0.1) =
    ...96). #838's helper rebuilds and gets away with it because
    dtheta = 0.225 happens to accumulate exactly; 0.05/3 does not.

    That distinction matters beyond the helper. The SEAM is exact -- the
    stored last node IS radians(0.1) -- so nothing in the old domain moves.
    What is an ulp off is `_interp`'s cell coordinate `(theta - th0)/dth`,
    which reads 3.0000000000000004 rather than 3.0 at the last node, shifting
    the Lagrange weights by ~1e-16. Harmless, and pre-existing on every band;
    worth knowing before someone re-derives nodes the convenient way.
    """
    reg = g._regions[RI(below._ZONE_INNER if zone is None else zone, below._BAND_LO)]
    return reg["th_nodes"]


@contextlib.contextmanager
def _record_fills(grid):
    """Answer routing questions WITHOUT evaluating the regions they reach.

    A routing gate asks which region a query lands in, and the lazy fill
    answers that exactly -- a region is materialized if and only if something
    routed into it. Actually computing it is a pure side effect that tells the
    gate nothing, and in this band it is the most expensive fill in the grid
    (a low-band node is ~7600 tail panels). Three gates here ran 27-40 s
    against the 20 s hard ceiling for exactly that reason.

    So `_fill_region` is swapped for a recorder that marks the region filled
    and notes the index. Values become NaN, which is fine because no gate
    using this reads one -- and a gate that did would fail loudly rather than
    quietly, since NaN fails every comparison.
    """
    seen = []

    def _fake(idx):
        reg = grid._regions[int(idx)]
        if not reg["filled"]:
            seen.append(int(idx))
            reg["filled"] = True

    orig = grid._fill_region
    grid._fill_region = _fake
    try:
        yield seen
    finally:
        grid._fill_region = orig


@pytest.fixture(scope="module", autouse=True)
def _warm_the_decks():
    """Fill both grazing bands on the shared decks ONCE.

    The low band is the expensive one -- its nodes sit at twice the tail cost
    of the mid band's -- so paying it per test, or per xdist worker, is what
    this and the `_FIXTURE_GROUP_FILES` entry in `conftest.py` prevent.
    """
    g, *_ = _grid("A", 7e6)
    g._ensure_band()
    g._ensure_band_lo()


# ---------------------------------------------------------------------------
# the seam
# ---------------------------------------------------------------------------


def test_the_low_band_divides_the_interval_exactly():
    """dtheta must divide [th_min, th_band_lo_hi], or the bands share no node.

    #838's gate, restated at the new seam. The low band's LAST node and the
    mid band's FIRST have to be the same float, because the mid band's first
    node is what every old-domain query at theta = 0.1 deg reads -- and the
    whole promise of adding a band below is that the old domain does not move.
    """
    span = below._SOMM_BELOW_TH_BAND_LO_HI_DEG - below._SOMM_BELOW_TH_MIN_DEG
    cells = span / below._SOMM_BELOW_DTH_BAND_LO_DEG
    assert abs(cells - round(cells)) < 1e-12, (
        f"_SOMM_BELOW_DTH_BAND_LO_DEG = {below._SOMM_BELOW_DTH_BAND_LO_DEG} "
        f"does not divide the band [{below._SOMM_BELOW_TH_MIN_DEG}, "
        f"{below._SOMM_BELOW_TH_BAND_LO_HI_DEG}] deg: {cells} cells. The last "
        "node then overshoots th_band_lo_hi, the low band and the mid band "
        "share no node, and every old-domain cell at theta = th_band_lo_hi "
        "moves. Pick a dtheta that divides the span exactly (0.05/3, 0.05/4, "
        "0.05/5)."
    )
    # And the constructed lattice agrees with that arithmetic, on a real grid,
    # in EVERY R1 zone -- the far zone builds its own theta lattice for the
    # bands above, so "the inner zone divides" is not the whole question.
    g, *_ = _grid("A", 7e6)
    for zone in (below._ZONE_INNER, below._ZONE_NEAR, below._ZONE_FAR):
        nodes = _lo_nodes(g, zone)
        assert nodes[-1] == g.th_band_lo_hi, (
            f"zone {zone}: low band ends at {math.degrees(nodes[-1]):.6g} deg, "
            f"not {math.degrees(g.th_band_lo_hi):.6g}"
        )
        mid = g._regions[RI(zone, below._BAND_MID)]
        assert mid["th0"] == g.th_band_lo_hi, (
            f"zone {zone}: the mid band starts at {mid['th0']!r}, not at the "
            f"shared node {g.th_band_lo_hi!r}"
        )
        assert nodes[-1] == mid["th0"], (
            f"zone {zone}: the two bands' shared node is not the same float: "
            f"{nodes[-1]!r} vs {mid['th0']!r}"
        )


def test_theta_at_the_low_band_edge_routes_to_the_mid_band():
    """The strict `<` at the new seam, asserted on the FILL STATE.

    Routing is a discrete question and the fill state answers it exactly;
    float equality does not, and reddened `test-macos` three times when #838
    tried it (see the long note in `test_grazing_band_838.py`). A region is
    materialized if and only if a query routes into it.
    """
    g, eps_t, k2, om, lam_m = _grid("A", 7e6)
    reg = g._regions[RI(below._ZONE_INNER, below._BAND_MID)]
    r_node = reg["r0"] + reg["dr"] * 6
    lo_inner = RI(below._ZONE_INNER, below._BAND_LO)

    fresh = below.SommerfeldGridBelow(eps_t, k2, g.r1_max, omega=om)
    with _record_fills(fresh):
        fresh.eval(np.array([r_node]), np.array([g.th_band_lo_hi]))
        assert not fresh._regions[lo_inner]["filled"], (
            "theta = th_band_lo_hi routed into the LOW band; it belongs to "
            "the mid band, which is what keeps the old domain unmoved there"
        )
        fresh.eval(np.array([r_node]), np.array([np.nextafter(g.th_band_lo_hi, 0.0)]))
        assert fresh._regions[lo_inner]["filled"], (
            "one ulp below th_band_lo_hi did not reach the low band, so the "
            "band edge is not a boundary at all"
        )


def test_the_old_domain_is_unmoved_by_the_new_band():
    """Nothing at or above 0.1 deg may change. This is the whole contract.

    Asserted two ways, because they fail differently. First structurally: no
    query in the old domain may even TOUCH a low-band region -- if one did,
    the value would come from a different lattice. Then numerically, at the
    seam node itself, where the bicubic collapses onto a single node in both
    axes and the reading must equal `iv_surfaces_direct_below` BIT for bit.
    """
    g, eps_t, k2, om, lam_m = _grid("A", 7e6)
    lo_all = [
        RI(z, below._BAND_LO)
        for z in (below._ZONE_INNER, below._ZONE_NEAR, below._ZONE_FAR)
    ]

    fresh = below.SommerfeldGridBelow(eps_t, k2, g.r1_max, omega=om)
    with _record_fills(fresh) as seen:
        for th_deg in (0.1, 0.11, 0.2, 0.5, 0.99, 1.0, 2.0, 45.0, 89.0):
            for r1l in (0.1, 0.5, 1.0, 2.0, 3.0):
                fresh.eval(np.array([r1l * lam_m]), np.radians([th_deg]))
        assert not any(fresh._regions[i]["filled"] for i in lo_all), (
            "an OLD-domain query (theta >= 0.1 deg) materialized a low-band "
            f"region, so its value no longer comes from the lattice it used "
            f"to. Regions reached: {sorted(seen)}"
        )
        # and the sweep did reach SOMETHING, or it proves nothing
        assert seen, "no region was reached at all; the sweep is inert"

    reg = g._regions[RI(below._ZONE_INNER, below._BAND_MID)]
    r_node = reg["r0"] + reg["dr"] * 6
    th = np.array([g.th_band_lo_hi])
    got = g.eval(np.array([r_node]), th)
    ref = below.iv_surfaces_direct_below(
        eps_t, k2, np.array([r_node]), th, rtol=1e-9, omega=om
    )
    scale = max(abs(complex(ref[k][0])) for k in _SURF_KEYS)
    for k in _SURF_KEYS:
        assert abs(complex(got[k][0]) - complex(ref[k][0])) / scale < 1e-9


# ---------------------------------------------------------------------------
# accuracy
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_low_band_interpolates_to_the_bar():
    """The new band against the direct surfaces, at OFF-NODE points.

    On-node agreement is trivially exact and would measure nothing. The
    queries are deliberately at cell midpoints and thirds, i.e. where a
    4-point Lagrange stencil is at its worst.
    """
    worst, where = 0.0, None
    for soil in SOILS:
        for f in FREQS:
            g, eps_t, k2, om, lam_m = _grid(soil, f)
            g._ensure_band_lo()
            nodes = _lo_nodes(g)
            th_q = np.concatenate(
                [
                    nodes[:-1] + 0.5 * np.diff(nodes),
                    nodes[:-1] + 0.33 * np.diff(nodes),
                ]
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
                    scale = np.maximum(np.abs(b), 1e-300)
                    rel = np.max(np.abs(a - b) / scale)
                    if rel > worst:
                        worst, where = rel, (soil, f, r1l, k)
    assert worst < LO_BAND_BAR, f"{worst:.3e} at {where}"


def test_both_dispatches_agree_on_the_low_band():
    """numpy `_interp` and C++ `proj_one_below` carry two copies of the band
    routing and the stride. #935 changed BOTH -- a 4-way selector and a zone
    stride of 4 -- so a drift shows up here first.

    The C++ side takes both band edges as explicit arguments rather than
    inferring them from `len(reg_vals)`, precisely so that this gate is
    testing one layout described twice and not two layouts.
    """
    # The module fixture already warmed BOTH bands on this deck; filling
    # again here would add ~50 s for nothing.
    g, eps_t, k2, om, lam_m = _grid("A", 7e6)
    k_m = below.k_medium(eps_t, k2)
    rho = 3.0
    edge = math.degrees(g.th_band_lo_hi)
    floor = math.degrees(g.th_min)
    for thd in (floor, 0.06, 0.08, edge - 1e-9, edge, edge + 1e-9, 0.5):
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
# cost
# ---------------------------------------------------------------------------


def test_the_low_band_is_deferred_until_something_reaches_it():
    """A deck that reaches 0.5 deg pays nothing for the 0.05 deg band.

    This is the reason the two grazing bands are deferred separately rather
    than as one set. Fold them together and every deck under 1 deg -- which is
    every buried screen momwire#838 was for -- pays twice the tail cost for
    nodes it never reads.
    """
    g, eps_t, k2, om, lam_m = _grid("B", 7e6)
    fresh = below.SommerfeldGridBelow(eps_t, k2, g.r1_max, omega=om)
    zones = (below._ZONE_INNER, below._ZONE_NEAR, below._ZONE_FAR)
    lo_all = [RI(z, below._BAND_LO) for z in zones]
    mid_all = [RI(z, below._BAND_MID) for z in zones]

    assert not any(fresh._regions[i]["filled"] for i in lo_all), (
        "the low band was filled at construction"
    )
    with _record_fills(fresh):
        # R1 = lambda_m is past `r_break`, so this lands in the NEAR zone --
        # which is why the assertion is over the zones rather than on the
        # inner one. Pinning the wrong zone made this gate fail while the
        # deferral it tests was working correctly.
        fresh.eval(np.array([1.0 * lam_m]), np.radians([0.5]))
        assert any(fresh._regions[i]["filled"] for i in mid_all), (
            "a 0.5 deg query did not fill the mid band in any zone, so this "
            "gate is not measuring what it claims"
        )
        assert not any(fresh._regions[i]["filled"] for i in lo_all), (
            "a 0.5 deg query filled the LOW band, which costs about twice "
            "per node -- the deferral is per band, not per grazing region"
        )
        fresh.eval(np.array([1.0 * lam_m]), np.radians([0.07]))
        assert any(fresh._regions[i]["filled"] for i in lo_all), (
            "a 0.07 deg query did not fill the low band"
        )


@pytest.mark.slow
def test_every_low_band_node_converges_under_the_panel_cap():
    """The floor is placed by the panel cap, and the margin is thin.

    `_MAX_TAIL_PANELS` binds before convergence does (momwire#841) and the
    failure past it is SILENT -- the contour extrapolates and returns. So
    every node of the new band is asserted converged, on every SPEC soil, plus
    the headroom as an early warning. Measured worst 7610/8000 = 95.1 % at
    (C, 21 MHz, R1 = 0.05 lambda_m).
    """
    for soil in SOILS:
        for f in FREQS:
            eps_t, k2, om, lam_m = _deck(soil, f)
            h = below.Health()
            th = np.radians(
                np.arange(
                    below._SOMM_BELOW_TH_MIN_DEG,
                    below._SOMM_BELOW_TH_BAND_LO_HI_DEG + 1e-12,
                    below._SOMM_BELOW_DTH_BAND_LO_DEG,
                )
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
                f"{soil}/{f}: {h.nonconvergent} low-band nodes hit the panel "
                "cap and were silently extrapolated"
            )
            frac = h.max_tail_panels / below._MAX_TAIL_PANELS
            assert frac < 0.98, (
                f"{soil}/{f}: worst node used {h.max_tail_panels} of "
                f"{below._MAX_TAIL_PANELS} panels ({frac:.1%})"
            )


# ---------------------------------------------------------------------------
# what the floor means as a geometry
# ---------------------------------------------------------------------------


def _h_min(length_m, floor_deg):
    """Shallowest depth a flat buried wire of this span can be solved at.

    The worst pair on a flat wire is its two ends: horizontal separation is
    the full span and the depth sum is twice the depth, so the smallest angle
    the grid is asked for is atan(2h / L). Setting that to the floor gives
    h_min = L * tan(floor) / 2.
    """
    return length_m * math.tan(math.radians(floor_deg)) / 2.0


@pytest.mark.parametrize("length_m", [5.9114, 20.0])
def test_the_floor_sets_a_minimum_depth_that_scales_with_length(length_m):
    """The floor is an ANGLE; what a user has is a length and a depth.

    This gate ties the two together and is the reason #935 is worth doing at
    all: halving the floor halves the shallowest depth every buried geometry
    can be solved at, in proportion to its own span.

        span      h_min at 0.1 deg   h_min at 0.05 deg
        5.9114 m       5.16 mm            2.58 mm
        20 m          17.45 mm            8.73 mm

    Asserted by driving the grid at the angle such a wire produces, just
    inside and just outside the floor.
    """
    floor_deg = below._SOMM_BELOW_TH_MIN_DEG
    h_min = _h_min(length_m, floor_deg)

    # it really is linear in the span, so one measurement carries every length
    assert _h_min(2 * length_m, floor_deg) == pytest.approx(2 * h_min, rel=1e-12)
    # and halving the floor halved it (tan is linear to well under a part in
    # 1e6 at these angles, so this is a real 2x and not a coincidence)
    assert _h_min(length_m, 2 * floor_deg) / h_min == pytest.approx(2.0, rel=1e-4)

    g, eps_t, k2, om, lam_m = _grid("A", 7e6)
    with _record_fills(g):
        for depth, served in ((h_min * 1.02, True), (h_min * 0.98, False)):
            th = math.atan2(2.0 * depth, length_m)
            r1 = np.array([math.hypot(length_m, 2.0 * depth)])
            if served:
                g.eval(r1, np.array([th]))
            else:
                with pytest.raises(ValueError, match="grazing floor"):
                    g.eval(r1, np.array([th]))


# ---------------------------------------------------------------------------
# the shape check: a real geometry, end to end, against NEC-5
# ---------------------------------------------------------------------------

_SHAPE_HALF, _SHAPE_RAD, _SHAPE_FREQ = 2.9557, 5.0e-4, 7.1e6
_SHAPE_WL = C0 / _SHAPE_FREQ
_SHAPE_SOIL = (13.0, 0.005)


def _shape_z(depth_m, n):
    from momwire.bspline import BSplineSolver

    wire = np.array([[-_SHAPE_HALF, 0.0, -depth_m], [_SHAPE_HALF, 0.0, -depth_m]])
    z, _ = BSplineSolver(
        wires=[wire],
        n_per_edge_per_wire=[[n]],
        wire_radius=_SHAPE_RAD,
        wavelength=_SHAPE_WL,
        degree=2,
        feed_model="segment",
        feed_wire_index=0,
        feed_arclength=_SHAPE_HALF,
        ground_z=0.0,
        ground_eps=_SHAPE_SOIL,
        ground_model="sommerfeld",
    ).compute_impedance()
    return complex(z)


@pytest.mark.parametrize("depth_mm", [0.5, 1.0])
def test_the_floor_still_refuses_what_it_cannot_reach(depth_mm):
    """Half the shape check is what #935 does NOT buy.

    A 5.9114 m dipole at 0.5 mm sees theta = 0.0097 deg and at 1 mm 0.0194
    deg, both far under even the new floor. They must still refuse BY NAME --
    a floor that quietly started serving everything would pass a one-sided
    gate just as well as a correct one.

    Cheap despite the geometry: the refusal is raised off the query extremes
    before any region is filled.
    """
    with pytest.raises(ValueError, match="below/below pair elevation"):
        _shape_z(depth_mm / 1000.0, 41)


@pytest.mark.slow
def test_the_new_floor_serves_a_3mm_dipole_and_matches_nec5():
    """The other half: the geometry #935 buys, against the laddered oracle.

    A 5.9114 m dipole 3 mm down sees theta = 0.0582 deg -- inside the new
    floor, outside the old one. Before #935 this deck refused; it now solves,
    and the value is right.

    THE ORACLE IS LADDERED (see `golden_grazing_shape_nec5`): NEC-5 is first
    order in the far mesh, so its x8 print carries a Richardson correction of
    0.158 ohm. AND SO IS MOMWIRE -- measured
    (`scratch/935-study/probe_shape_check_vs_nec5.py`):

        n=41   166.480 +10.801j     n=241  165.589 +11.668j
        n=81   166.429 +12.516j     n=321  165.414 +11.434j
        n=161  165.882 +12.051j

    which is still drifting at n=321 (steps -0.175 R, -0.234 X) toward roughly
    165.2 +11.1j against the oracle's 164.43 +10.08j -- about 0.8 % of |Z|.

    So momwire's own mesh convergence, not the oracle's 0.16 ohm bar, is the
    limiting term in this comparison, and the tolerance says so. The gate runs
    at n=161, where the measured separation is 1.49 % of |Z|; the bar is 2.5 %.
    This is a cross-code shape check -- it catches a floor that serves the
    wrong VALUE -- not a convergence gate on either side.
    """
    from golden_grazing_shape_nec5 import BAR, CONVERGED

    ref = CONVERGED[3.0]
    got = _shape_z(3.0 / 1000.0, 161)
    rel = abs(got - ref) / abs(ref)
    assert rel < 0.025, (
        f"momwire {got!r} against the laddered NEC-5 limit {ref!r} "
        f"(oracle bar {BAR[3.0]:.4f} ohm): {rel:.2%} of |Z|"
    )
