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

from momwire import _ground_refl  # noqa: E402
from momwire import _sommerfeld_below as below  # noqa: E402
from momwire._sommerfeld import _SOMM_TH_SPLIT_DEG, _SURF_KEYS  # noqa: E402
from momwire._sommerfeld import SommerfeldGrid  # noqa: E402
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
    # Routing on the FILL STATE, which is exact and portable -- see the long
    # note in `test_the_far_zone_seam_at_r_near_routes_the_old_domain_inward`
    # for why float equality is the wrong instrument for this question.
    # Region 3 is the NEAR zone's sub-1 deg band and fills lazily; region 4
    # is its old grazing band.
    fresh = below.SommerfeldGridBelow(eps_t, k2, g.r1_max, omega=om)
    fresh.eval(np.array([r_node]), th)
    assert not fresh._regions[3]["filled"], (
        "theta = th_band_hi routed into the sub-1 deg band; it belongs to the "
        "OLD grazing band, which is what keeps the old domain unmoved there"
    )
    fresh.eval(np.array([r_node]), np.array([np.nextafter(g.th_band_hi, 0.0)]))
    assert fresh._regions[3]["filled"], (
        "one ulp below th_band_hi did not reach the sub-1 deg band, so the "
        "band edge is not a boundary at all"
    )

    got = g.eval(np.array([r_node]), th)
    ref = below.iv_surfaces_direct_below(
        eps_t, k2, np.array([r_node]), th, rtol=1e-9, omega=om
    )
    scale = max(abs(complex(ref[k][0])) for k in _SURF_KEYS)
    for k in _SURF_KEYS:
        assert abs(complex(got[k][0]) - complex(ref[k][0])) / scale < 1e-9

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


def _legacy_eval(g, r1, th):
    """The pre-momwire#838 routing, driven off THIS grid's own tables.

    Before the band there were two theta bands per R1 zone, split at
    `_SOMM_TH_SPLIT_DEG`, and those two are still here untouched as regions
    1/2 (inner) and 4/5 (outer) — same node sets, same spacings, same fill
    call. So replaying the old routing over them reproduces exactly what the
    old grid returned, and comparing `eval` against it isolates the ONE thing
    momwire#838 could have broken in the old domain: which region a query is
    sent to, and with what local index.

    Deliberately computed IN PROCESS rather than compared against committed
    values. A golden captured on one box asserts bit-equality of the contour
    fill across toolchains, which is not portable and not this unit's claim —
    macOS reads ~1.7e-12 from Linux on these surfaces, and a golden here
    reddened `test-macos` for exactly that reason (the momwire#839 class).
    Both sides of this comparison are filled by the same box, so `==` means
    what it says.
    """
    th_split = math.radians(_SOMM_TH_SPLIT_DEG)
    r1 = np.atleast_1d(np.asarray(r1, dtype=float))
    th = np.atleast_1d(np.asarray(th, dtype=float))
    out = np.empty((len(_SURF_KEYS), r1.size), dtype=np.complex128)
    for n in range(r1.size):
        idx = (0 if r1[n] <= g.r_break else 3) + (1 if th[n] <= th_split else 2)
        reg = g._regions[idx]
        fr = (r1[n] - reg["r0"]) / reg["dr"]
        ft = (th[n] - reg["th0"]) / reg["dth"]
        i0 = int(np.clip(np.floor(fr) - 1, 0, reg["n_r"] - 4))
        j0 = int(np.clip(np.floor(ft) - 1, 0, reg["n_th"] - 4))
        wr = SommerfeldGrid._lagrange4(np.array([fr - i0]))[0]
        wt = SommerfeldGrid._lagrange4(np.array([ft - j0]))[0]
        block = reg["vals"][:, i0 : i0 + 4, j0 : j0 + 4]
        out[:, n] = np.einsum("sij,i,j->s", block, wr, wt)
    return {k: out[i] for i, k in enumerate(_SURF_KEYS)}


def test_the_old_domain_is_unmoved():
    """Nothing that was already served moved -- bit for bit.

    Shown to bite rather than asserted to, by mutation. It catches the bug
    this unit actually shipped in draft -- a non-dividing dtheta = 0.25
    together with a non-strict band edge, which let a band whose last node
    overshoots to 1.1 deg steal the theta = 1 deg query from the old band --
    and it catches a plain misroute (the band claiming everything under
    `_SOMM_TH_SPLIT_DEG`), which moves the first cells above 1 deg.

    Note it does NOT fire on dtheta = 0.25 alone: the strict `<` added later
    protects theta = 1 deg independently, and
    `test_the_band_divides_the_interval_exactly` is what names that one. The
    two gates cover different halves of the seam, which is why both are here.

    Swept OFF-node on purpose, densely through the seam's own first cell and
    across the 30 deg split -- a sweep that only landed on lattice nodes would
    pass even if the stencil changed underneath it. R1 is swept as a fraction
    of `r_near` (the old cap), which is what "the old domain" means once
    part 2 moved `r1_max` out to 4 lambda_m.
    """
    th_deg = np.concatenate(
        [
            np.linspace(1.0, 1.999, 40),  # the seam's first cell
            np.linspace(2.0, 29.9, 60),
            np.linspace(30.0, 89.9, 40),
        ]
    )
    for soil, f in (("A", 7e6), ("C", 21e6)):
        g, *_ = _grid(soil, f)
        # Fractions of `r_near`, NOT of `r1_max`. momwire#838 part 2 doubled
        # the cap to 4 lambda_m, and R1 past `r_near` = 2 lambda_m is the new
        # far annulus -- domain that REFUSED before, so there is no old value
        # of it to preserve. `r_near` is exactly the old cap, so this sweep is
        # the old domain however the cap moves again later.
        for r_frac in (0.02, 0.35, 0.68, 0.98):
            r1 = np.full(th_deg.shape, r_frac * g.r_near)
            th = np.radians(th_deg)
            got = g.eval(r1, th)
            want = _legacy_eval(g, r1, th)
            for k in _SURF_KEYS:
                bad = np.nonzero(got[k] != want[k])[0]
                assert bad.size == 0, (
                    f"{soil}/{f:g} R1/r1_max={r_frac}: {k} moved at "
                    f"theta = {th_deg[bad[:4]]} deg, first "
                    f"{got[k][bad[0]]!r} != {want[k][bad[0]]!r}"
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


# ---------------------------------------------------------------------------
# momwire#838 part 2 -- the far annulus [2, 4) lambda_m
# ---------------------------------------------------------------------------
#
# The cap was 2 lambda_m because the far annulus interpolated at 2.4e-3, 12x
# worse than everything inside it. Measured one axis at a time
# (`scratch/probe838_r1_cap.py`), that is entirely the THETA axis: dr = 0.05
# lambda_m already reads 1.6e-5 out there, while the steep band at dtheta 2.5
# reads 1.5e-3. So the far zone is the same lattice in R1 with a finer theta
# one, and the cap moves to 4 lambda_m.

# Worst measured over soils A/B/C at the far-zone lattice: 3.6e-5, against the
# 2e-4 the inner domain is gated at. Set 4x over the measurement, still 5x
# under the inner bar.
FAR_BAR = 1.5e-4


def _far_band_bounds(g, band):
    """(lo, hi) in radians for one theta band, and its far-zone region."""
    reg = g._regions[6 + band]
    lo = reg["th0"]
    return lo, lo + reg["dth"] * (reg["n_th"] - 1), reg


def test_the_far_zone_dtheta_divides_its_band_exactly():
    """Part 1's rule, applied to the two NEW far-zone lattices.

    Spelled as cell counts in the source precisely so this cannot go wrong --
    neither of the decimals originally proposed divides (60/0.833 = 72.03,
    29/0.4 = 72.5). This gate is what keeps that true if someone edits the
    counts into decimals later.
    """
    g, *_ = _grid("A", 7e6)
    for band, hi_deg in ((1, below._SOMM_TH_SPLIT_DEG), (2, 90.0)):
        lo, hi, reg = _far_band_bounds(g, band)
        assert abs(hi - math.radians(hi_deg)) < 1e-12, (
            f"far band {band} ends at {math.degrees(hi):.6f} deg, not {hi_deg}"
        )
        # and it meets the inner zone's band at the SAME theta nodes
        inner = g._regions[3 + band]
        assert reg["th0"] == inner["th0"], (band, reg["th0"], inner["th0"])


def test_the_far_zone_seam_at_r_near_routes_the_old_domain_inward():
    """R1 = r_near belongs to the NEAR zone, the way theta = 1 deg belongs to
    the old grazing band -- the same strict-inequality rule, on the other
    axis. With R1 on a lattice row of both zones the bicubic collapses to a
    node, so the reading must be the direct surface bit for bit.
    """
    g, eps_t, k2, om, lam_m = _grid("A", 7e6)
    near, far = g._regions[4], g._regions[7]
    # the shared boundary is a node of both R1 axes
    assert any(
        abs(near["r0"] + near["dr"] * i - g.r_near) < 1e-9 for i in range(near["n_r"])
    )
    assert any(
        abs(far["r0"] + far["dr"] * i - g.r_near) < 1e-9 for i in range(far["n_r"])
    )
    # theta is `th_band_hi`, which is region 4's own `th0` -- so the stencil
    # collapses EXACTLY on both axes and bit-identity is a fair thing to ask.
    # (A "nice" angle like 45 deg is not a fair ask: the theta node is built
    # as radians(30 + 2.5*k) while `_interp` forms (theta - th0)/dth, and
    # those disagree in the last bit, which reads as 2.2e-16 rather than 0.)
    # ROUTING IS ASSERTED ON THE FILL STATE, not on float equality.
    #
    # Three attempts at this gate compared values with `==` -- eval against a
    # fresh contour solve, then eval against the stored node -- and both
    # reddened `test-macos` while passing on Linux. The second one is the
    # instructive failure: even reading a node back is only bit-exact if
    # `(r_near - r0)/dr` lands exactly on an integer, so the cubic weights
    # come out exactly [0,1,0,0]. That is a property of one platform's
    # division, not of this grid. Bit-exactness is simply the wrong
    # instrument for a routing question.
    #
    # The lazy fill gives an exact, integer-valued one instead: a region is
    # materialized if and only if a query routes into it. theta =
    # `th_band_hi` is band 1 (the old grazing band, by the strict `<`), so
    # the far-zone region it would reach is 6 + 1 = 7.
    th = np.array([g.th_band_hi])
    fresh = below.SommerfeldGridBelow(eps_t, k2, g.r1_max, omega=om)
    fresh.eval(np.array([fresh.r_near]), th)
    assert not fresh._regions[7]["filled"], (
        "R1 = r_near routed OUTWARD into the far annulus; it belongs to the "
        "near zone, the way theta = th_band_hi belongs to the old grazing band"
    )
    fresh.eval(np.array([np.nextafter(fresh.r_near, np.inf)]), th)
    assert fresh._regions[7]["filled"], (
        "a query one ulp past r_near did not reach the far zone, so the seam "
        "is not a boundary at all"
    )

    # And the value is right, to interpolation precision rather than to the bit.
    got = g.eval(np.array([g.r_near]), th)
    ref = below.iv_surfaces_direct_below(
        eps_t, k2, np.array([g.r_near]), th, rtol=1e-9, omega=om
    )
    scale = max(abs(complex(ref[k][0])) for k in _SURF_KEYS)
    for k in _SURF_KEYS:
        assert abs(complex(got[k][0]) - complex(ref[k][0])) / scale < 1e-9


def test_both_dispatches_agree_across_the_r_near_seam():
    """numpy and C++ carry two copies of the three-zone routing."""
    g, eps_t, k2, om, lam_m = _grid("A", 7e6)
    k_m = below.k_medium(eps_t, k2)
    t = np.array([[1.0, 0.0, 0.0]])
    for r1_l, th_deg in (
        (1.5, 45.0),
        (1.999, 45.0),
        (2.0, 45.0),
        (2.001, 45.0),
        (3.0, 45.0),
        (3.6, 0.21),
        (3.99, 5.0),
    ):
        R = r1_l * lam_m
        rho, hh = R * math.cos(math.radians(th_deg)), R * math.sin(math.radians(th_deg))
        obs = np.array([[0.0, 0.0, -0.5 * hh]])
        src = np.array([[rho, 0.0, -0.5 * hh]])
        cpp = below.remainder_field_proj_below(obs, t, src, t, 0.0, k2, k_m, g)
        with force_numpy():
            npy = below.remainder_field_proj_below(obs, t, src, t, 0.0, k2, k_m, g)
        rel = abs(cpp[0, 0] - npy[0, 0]) / abs(npy[0, 0])
        assert rel < 1e-13, f"R1 {r1_l} lam, theta {th_deg}: {rel:.3e}"


def test_the_far_zone_is_deferred_until_something_reaches_it():
    """Nothing under 2 lambda_m pays for the far annulus.

    It is ~4x the panels of the whole rest of the grid, and 68 % of THAT is
    its own sub-1 deg band (panels go as 6.4/tan theta), which is why it is
    deferred per theta BAND rather than as a block.
    """
    g, eps_t, k2, om, lam_m = _grid("B", 7e6)
    assert not any(g._regions[i]["filled"] for i in (6, 7, 8)), (
        "the far annulus was filled at construction"
    )
    g.eval(np.array([1.0 * lam_m]), np.array([math.radians(45.0)]))
    assert not any(g._regions[i]["filled"] for i in (6, 7, 8)), (
        "an inner-domain query filled the far annulus"
    )
    g.eval(np.array([3.0 * lam_m]), np.array([math.radians(45.0)]))
    assert g._regions[8]["filled"], "the far steep band was not filled on demand"
    assert not g._regions[6]["filled"], (
        "a steep far query filled the far zone's sub-1 deg band, which is 68 % "
        "of its panels -- the deferral is per theta band, not per zone"
    )


@pytest.mark.slow
def test_the_far_zone_interpolates_to_the_bar():
    """The far annulus against the direct surfaces, at off-node points."""
    worst = 0.0
    for soil in SOILS:
        for f in FREQS:
            g, eps_t, k2, om, lam_m = _grid(soil, f)
            r1 = np.linspace(2.05, 3.95, 11) * lam_m
            for th_deg in (0.3, 2.0, 17.0, 45.0, 80.0):
                th = np.radians(np.full(r1.shape, th_deg))
                got = g.eval(r1, th)
                ref = below.iv_surfaces_direct_below(
                    eps_t, k2, r1, th, rtol=1e-9, omega=om
                )
                G = np.stack([got[k] for k in _SURF_KEYS])
                R = np.stack([ref[k] for k in _SURF_KEYS])
                worst = max(
                    worst, float((np.abs(G - R) / np.abs(R).max(axis=0)[None, :]).max())
                )
    assert worst < FAR_BAR, f"{worst:.3e}"
