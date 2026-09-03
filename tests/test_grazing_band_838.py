"""momwire#838 part 1: the sub-1 deg grazing band for the below/below family.

`_SOMM_BELOW_TH_MIN_DEG` was 1.0 deg, and every realistic buried screen is
below it -- BLE 1937's 45 ft radial tip is 0.64 deg and its 135 ft tip
0.21 deg. The floor is now 0.1 deg, carried by a THIRD uniform theta band
over [0.1, 1.0] deg at `_SOMM_BELOW_DTH_BAND_DEG`.

Two constraints in that construction are invisible in the code and would
fail silently, so each has its own gate here rather than being folded into
the surface comparison:

  * dtheta must DIVIDE the band exactly. At 0.25 the last node overshoots to
    1.1 deg, the two bands share no node, and every old-domain cell at
    theta = 1 deg moves (measured 1.2e-07 when this was first written that
    way -- the `test_the_old_domain_is_unmoved` gate below is what caught it).
  * theta = `th_band_hi` must route to the OLD band, i.e. a STRICT `<`.
    `th_min + dth*n` need not reproduce `th_band_hi` in the last bit, so a
    query landing on the new band's copy of that node would differ in low
    bits. The comparison must be the same in numpy and in C++.
"""

import math
import sys

import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import golden_below_old_domain_838 as gold_old  # noqa: E402

from momwire import _ground_refl  # noqa: E402
from momwire import _sommerfeld_below as below  # noqa: E402
from momwire._sommerfeld import _SURF_KEYS  # noqa: E402
from test_below_fills_568 import force_numpy  # noqa: E402

C0 = 299792458.0
EPS0 = 8.8541878128e-12
SOILS = {"A": (13.0, 0.005), "B": (20.0, 0.03), "C": (5.0, 0.001)}
FREQS = (7e6, 21e6)

# The band's own interpolation bar. #553 U2 measured 4.7e-4 for the grazing
# band at dtheta = 1 deg and dr = 0.05 lambda_m, and that is the accuracy the
# new band was required to match, not beat. Measured worst on the real grid
# (so R1 interpolation is in it too, unlike the standalone probe): 1.771e-07
# at dtheta = 0.225, across soils A/B/C x 7/21 MHz x R1/lambda_m in
# {0.2, 1.0, 1.9} x 23 off-node theta. Set 100x over that, still 265x under
# the bar it had to match.
BAND_BAR = 1.8e-5

# Panel-cap headroom for every node the band fills. The floor sits where it
# does because `_MAX_TAIL_PANELS` binds before convergence does (momwire#841),
# and the margin is THIN: measured worst across SPEC soils A/B/C x 7/21 MHz x
# R1/lambda_m in {0, 0.02, 0.05, 0.2, 1, 2} is 3868 of 4000 = 96.7%, at
# (C, 21 MHz, R1 = 0.05 lambda_m). An early reading of 3542 (90%) was one deck
# only -- soil A at 7 MHz -- and understated it.
#
# So this bound is 0.98, not a comfortable round number, because 0.98 is what
# the measurement leaves room for. `nonconvergent == 0` below is the hard
# requirement; this one is the early warning.
CAP_HEADROOM = 0.98


def _deck(soil, f):
    k2 = 2.0 * np.pi * f / C0
    om = 2.0 * np.pi * f
    eps_t = _ground_refl.eps_tilde(SOILS[soil], om, EPS0)
    lam_m = below.lambda_medium(eps_t, k2)
    return eps_t, k2, om, lam_m


_GRIDS = {}


def _grid(soil, f):
    """Cached: a below grid costs ~2.6 s to fill since the band landed, and
    these tests want several decks each."""
    eps_t, k2, om, lam_m = _deck(soil, f)
    if (soil, f) not in _GRIDS:
        r1_max = below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m
        _GRIDS[(soil, f)] = below.SommerfeldGridBelow(eps_t, k2, r1_max, omega=om)
    return _GRIDS[(soil, f)], eps_t, k2, om, lam_m


def _band_theta_nodes(g):
    reg = g._regions[0]
    return reg["th0"] + reg["dth"] * np.arange(reg["n_th"])


@pytest.fixture(scope="module", autouse=True)
def _warm_the_decks():
    """Build the two decks these gates share, band included, ONCE.

    The band fill is this unit's headline cost (~2 s, 4.2x the rest of the
    grid). Paying it in module setup rather than inside whichever test touches
    the band first keeps the time-budget guardrail measuring what each gate
    actually does — the guardrail reads the CALL phase — and `conftest`'s
    `_FIXTURE_GROUP_FILES` entry keeps xdist from paying it once per worker.
    """
    for soil, f in (("A", 7e6), ("C", 21e6)):
        g, *_ = _grid(soil, f)
        g._ensure_band()


def test_the_band_divides_the_interval_exactly():
    """dtheta must divide [th_min, th_band_hi], or the bands share no node.

    This is the whole seam. The new band's LAST node and the old grazing
    band's FIRST node have to be the same float, because the old band's
    first node is what every old-domain query at theta = 1 deg reads.
    """
    span = below._SOMM_BELOW_TH_BAND_HI_DEG - below._SOMM_BELOW_TH_MIN_DEG
    cells = span / below._SOMM_BELOW_DTH_BAND_DEG
    assert abs(cells - round(cells)) < 1e-12, (
        f"_SOMM_BELOW_DTH_BAND_DEG = {below._SOMM_BELOW_DTH_BAND_DEG} does not "
        f"divide the band [{below._SOMM_BELOW_TH_MIN_DEG}, "
        f"{below._SOMM_BELOW_TH_BAND_HI_DEG}] deg: {cells} cells. The last node "
        "then overshoots th_band_hi, the new band and the old grazing band "
        "share no node, and every old-domain cell at theta = th_band_hi moves. "
        "Pick a dtheta that divides the span exactly (0.9/4 = 0.225, 0.9/6 = "
        "0.15, 0.9/9 = 0.1)."
    )
    # And the constructed lattice agrees with that arithmetic, on a real grid.
    g, *_ = _grid("A", 7e6)
    band_nodes = _band_theta_nodes(g)
    assert band_nodes[-1] == g.th_band_hi, (
        f"band ends at {math.degrees(band_nodes[-1]):.6g} deg, not "
        f"{math.degrees(g.th_band_hi):.6g}"
    )
    # regions 1 and 4 are the old grazing band, per R1 zone.
    for idx in (1, 4):
        assert g._regions[idx]["th0"] == g.th_band_hi
        assert g._regions[idx]["dth"] == math.radians(below._SOMM_BELOW_DTH_GRAZE_DEG)


def test_theta_at_the_band_edge_routes_to_the_old_band():
    """The strict `<`, and why it has to be strict.

    At theta = th_band_hi exactly, with R1 on a lattice row, the bicubic
    collapses onto a single node in both axes -- so the reading must equal
    `iv_surfaces_direct_below` at that point BIT for bit. That is only true
    if the query routes to the old band, whose node sits at exactly
    `radians(1.0)`; the new band's copy is `th_min + dth*n`, which need not
    be the same float.
    """
    g, eps_t, k2, om, lam_m = _grid("A", 7e6)
    reg = g._regions[4]  # outer R1 zone, old grazing band
    r_node = reg["r0"] + reg["dr"] * 6
    th = np.array([g.th_band_hi])
    got = g.eval(np.array([r_node]), th)
    ref = below.iv_surfaces_direct_below(
        eps_t, k2, np.array([r_node]), th, rtol=1e-9, omega=om
    )
    for k in _SURF_KEYS:
        assert complex(got[k][0]) == complex(ref[k][0]), (
            f"{k}: theta = th_band_hi did not collapse onto the old band's "
            f"node -- got {got[k][0]!r} vs {ref[k][0]!r}"
        )

    # The invariant that actually carries the seam: the new band's LAST theta
    # node and the old band's FIRST are the same float. Measured -- with
    # dtheta = 0.225, `radians(0.1 + 0.225*4)` IS `radians(1.0)` exactly. So a
    # query one ulp below th_band_hi, which routes to the NEW band and whose
    # stencil collapses onto that node, legitimately reads the same value.
    #
    # That makes the strict `<` belt-and-braces rather than load-bearing --
    # and it is worth keeping precisely because the float coincidence is a
    # consequence of the divisibility rule above, not an independent fact. If
    # someone widens the band so the arithmetic no longer lands exactly, the
    # `<` is what keeps the old domain's readings coming from the old band.
    band_nodes = _band_theta_nodes(g)
    assert band_nodes[-1] == g._regions[4]["th0"], (
        f"the two bands' shared node is not the same float: "
        f"{band_nodes[-1]!r} vs {g._regions[4]['th0']!r}"
    )
    th_lo = np.array([np.nextafter(g.th_band_hi, 0.0)])
    got_lo = g.eval(np.array([r_node]), th_lo)
    ref_lo = below.iv_surfaces_direct_below(
        eps_t, k2, np.array([r_node]), th_lo, rtol=1e-9, omega=om
    )
    scale = max(abs(complex(ref_lo[k][0])) for k in _SURF_KEYS)
    for k in _SURF_KEYS:
        assert abs(complex(got_lo[k][0]) - complex(ref_lo[k][0])) / scale < BAND_BAR


def test_both_dispatches_route_the_seam_the_same_way():
    """numpy and C++ carry two copies of the band routing; a drift in the
    `<` would show first here, exactly at the edge."""
    g, eps_t, k2, om, lam_m = _grid("A", 7e6)
    k_m = below.k_medium(eps_t, k2)
    rho = 3.0
    edge = math.degrees(g.th_band_hi)
    for thd in (0.11, 0.5, edge - 1e-9, edge, edge + 1e-9, 2.0, 45.0):
        hh = rho * math.tan(math.radians(thd))
        obs = np.array([[0.0, 0.0, -0.5 * hh]])
        src = np.array([[rho, 0.0, -0.5 * hh]])
        t = np.array([[1.0, 0.0, 0.0]])
        cpp = below.remainder_field_proj_below(obs, t, src, t, 0.0, k2, k_m, g)
        with force_numpy():
            npy = below.remainder_field_proj_below(obs, t, src, t, 0.0, k2, k_m, g)
        rel = abs(cpp[0, 0] - npy[0, 0]) / abs(npy[0, 0])
        assert rel < 1e-13, f"theta {thd}: dispatches disagree at {rel:.3e}"


def test_the_old_domain_is_unmoved():
    """Nothing that was already served moved. Off-node readings captured on
    main `79911c5`, the commit before the band existed.

    This is the gate that caught dtheta = 0.25 (it read 1.2e-07 at theta =
    1 deg, where it must read zero), so it has been shown to bite rather
    than merely being asserted to.
    """
    for (soil, f), rows in gold_old.OLD_DOMAIN.items():
        g, *_ = _grid(soil, f)
        for r_frac, th_deg, *vals in rows:
            got = g.eval(np.array([r_frac * g.r1_max]), np.radians([th_deg]))
            for k, want in zip(gold_old.KEYS, vals, strict=True):
                assert complex(got[k][0]) == want, (
                    f"{soil}/{f:g} R1/r1_max={r_frac} theta={th_deg}: {k} "
                    f"moved, {got[k][0]!r} != {want!r}"
                )


def test_every_band_node_converges_under_the_panel_cap():
    """Part 1's own assumption, pinned: the floor is where it is because
    `_MAX_TAIL_PANELS` binds before convergence does (momwire#841), and the
    cap is reached SILENTLY. So assert the headroom, not just the pass.

    Deliberately NOT `slow`-marked: `test-slow` is push-only, so a slow gate
    on this assumption would first read AFTER the merge that broke it.
    """
    eps_t, k2, om, lam_m = _deck("A", 7e6)
    g, *_ = _grid("A", 7e6)
    th_nodes = _band_theta_nodes(g)
    h = below.Health()
    for r1l in (0.05, 1.0, 2.0):
        below.iv_surfaces_direct_below(
            eps_t,
            k2,
            np.full(th_nodes.shape, r1l * lam_m),
            th_nodes,
            rtol=1e-9,
            omega=om,
            health=h,
        )
    d = h.as_dict()
    assert d["nonconvergent"] == 0, (
        f"{d['nonconvergent']} of {d['evaluations']} band nodes hit "
        f"_MAX_TAIL_PANELS and returned a TRUNCATED value (momwire#841 -- the "
        f"cap is silent). The floor cannot be this low."
    )
    assert d["max_tail_panels"] <= CAP_HEADROOM * below._MAX_TAIL_PANELS, (
        f"the band's worst node used {d['max_tail_panels']} of "
        f"{below._MAX_TAIL_PANELS} tail panels, past the "
        f"{CAP_HEADROOM:.0%} headroom this floor was chosen for. Panels grow "
        f"like 6.4/tan(theta), so this is what lowering _SOMM_BELOW_TH_MIN_DEG "
        f"costs -- see the table beside that constant."
    )


@pytest.mark.slow
def test_every_band_node_converges_on_every_soil():
    """The headroom sweep the PR-lane gate above samples: all SPEC soils,
    both frequencies, R1 out to the cap."""
    for soil in SOILS:
        for f in FREQS:
            eps_t, k2, om, lam_m = _deck(soil, f)
            g, *_ = _grid(soil, f)
            th_nodes = _band_theta_nodes(g)
            h = below.Health()
            for r1l in (0.0, 0.05, 0.5, 1.0, 2.0):
                below.iv_surfaces_direct_below(
                    eps_t,
                    k2,
                    np.full(th_nodes.shape, r1l * lam_m),
                    th_nodes,
                    rtol=1e-9,
                    omega=om,
                    health=h,
                )
            d = h.as_dict()
            assert d["nonconvergent"] == 0, (soil, f, d)
            assert d["max_tail_panels"] <= CAP_HEADROOM * below._MAX_TAIL_PANELS, (
                soil,
                f,
                d,
            )


@pytest.mark.slow
def test_the_band_interpolates_to_the_bar():
    """The band matches the accuracy the 1 deg band already had (4.7e-4),
    measured against the direct surfaces at off-node theta."""
    worst = 0.0
    for soil in SOILS:
        for f in FREQS:
            g, eps_t, k2, om, lam_m = _grid(soil, f)
            th = np.radians(np.linspace(0.105, 0.995, 23))
            for r1l in (0.2, 1.0, 1.9):
                r1 = np.full(th.shape, r1l * lam_m)
                got = g.eval(r1, th)
                ref = below.iv_surfaces_direct_below(
                    eps_t, k2, r1, th, rtol=1e-9, omega=om
                )
                G = np.stack([got[k] for k in _SURF_KEYS])
                R = np.stack([ref[k] for k in _SURF_KEYS])
                worst = max(
                    worst, float((np.abs(G - R) / np.abs(R).max(axis=0)[None, :]).max())
                )
    assert worst < BAND_BAR, f"{worst:.3e}"


def test_ble_geometries_are_served():
    """What momwire#838 is for: BLE 1937's radial tips stop refusing.

    45 ft radials 6 in deep is theta = 0.64 deg at the tip, 135 ft is
    0.21 deg -- both refused at the old 1 deg floor. The Validation Manual
    screen's 0.023 deg is still refused, deliberately (see momwire#841).
    """
    g, eps_t, k2, om, lam_m = _grid("A", 7e6)
    r1 = np.array([0.5 * lam_m])
    for th_deg in (0.64, 0.21, 0.1):
        g.eval(r1, np.radians([th_deg]))
    with pytest.raises(ValueError, match="grazing floor"):
        g.eval(r1, np.radians([0.023]))
