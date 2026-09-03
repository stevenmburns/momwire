"""BLE 1937 Fig. 37 — the first buried gate with a MEASUREMENT behind it.

Brown, Lewis and Epstein, "Ground Systems as a Factor in Antenna
Efficiency", Proc. IRE 25(6), June 1937. Fig. 37 is base resistance of a
77-degree mast against buried-radial count N at 3 MHz, radials 45 ft long
plowed in 6 in. It is the only measured R-against-N series with a fully
specified buried geometry, which is why momwire#838 made it the gate: every
other buried check in this tree compares momwire against another MODEL.

Readings and the reading method are momwire#838's, not re-derived here:
a 260 dpi crop with the axes calibrated off the printed gridlines, giving
>= 50 ohm at N = 2 (the curve leaves the 50 ohm scale), ~33 at N = 15, then
flat at ~31 through N = 113, with ~+-1 ohm of reading error on the plateau.

WHAT IS GATED IS THE SHAPE, NOT THE READING. A figure read off a 1937
halftone is not an oracle to three digits, and the paper does not state
permittivity at all. So this pins: monotone decrease in N, a large total
drop, and a plateau inside a stated envelope. The envelope is built from
the measured uncertainties below and NOT from whatever made the test pass.

## The deck, and every assumption in it

  3 MHz (lambda = 99.93 m); mast 21.4 m = 77.1 deg; radials 13.716 m
  (45 ft) at depth 0.1524 m (6 in); sigma = 2e-3 S/m (the paper's
  0.2e-4 mho/cm^3); hub spelling -- N radials meeting a buried hub, one
  rise to the node at z = 0, the mast above it.

* **eps_r = 15, ASSUMED.** The paper states conductivity only. Measured
  spread over eps_r 5 / 10 / 15 / 20 / 30 at sigma 2e-3: **3.00 ohm at
  N = 15, 3.60 ohm at N = 30** (and non-monotone -- R falls to ~eps 20
  then rises again). This is the single largest term in the envelope.

* **ONE wire radius, 1.63 mm (No. 8), for the mast TOO.** The crossing
  serve's corner regularization is defined for a single radius, so the
  paper's 2.5 in mast cannot be spelled at its own radius here. No. 8 is
  the honest choice because the screen, not the mast, sets the loss --
  measured at N = 15: R = 31.57 / 30.71 / 29.93 ohm at a = 1.63 / 4 / 10
  mm, i.e. R moves under 1.7 ohm across a 6x radius change. Pushing the
  shared radius to the mast's own 31.75 mm instead reads 43.50 ohm, but
  that deck has 31.75 mm RADIALS, which is not BLE's screen.

* **Node grading, #674-matched.** Ungraded, this deck raises
  `CoarseCrossingNode` (~1427 mm of mesh within 150 mm of the node
  against a 25 mm bar). Graded as below it is silent.

* **The feed is pinned to a polyline VERTEX at z = 0.05 m**, not to a
  fraction of the mast mesh. A feed at `H - 0.5*H/n_far` moves as the mesh
  refines, and that confounds the very convergence it is being used to
  measure: it read a 0.75 ohm "mast mesh drift" over n_far 12 -> 60 that
  was entirely the feed moving. Pinned, the same sweep is FLAT to 0.014
  ohm (31.565 / 31.571 / 31.575 / 31.578 / 31.579).

## Convergence, all three axes

  quadrature  n_qp_pair 4 / 8 / 16 / 32 / 64 -> R 91.75 / 91.65 / 91.6487
              / 91.6487 / 91.6487 at N = 2. Converged by 16, which is the
              order used here. momwire#674/#760's lesson is that mesh
              convergence at fixed quadrature converges to the WRONG limit;
              on this deck the quadrature axis is far better behaved than
              the soil-A fan's (0.16 ohm from q=4, against that deck's 6.8).
  radial mesh n_rad 6 / 10 / 16 / 24 -> 32.258 / 32.264 / 32.265 / 32.265.
  mast mesh   flat to 0.014 ohm once the feed is pinned (above).

## The measured ladder (this deck, q = 16, n_rad = 10, n_far = 19)

      N       2       4       8      15      30      60     113
      R   90.28   56.21   38.68   31.57   28.51   27.40   26.97
   cost    2.9s    1.5s    2.3s    3.3s    5.5s   27.1s  241.8s

N = 113 is BANKED rather than gated: 242 s is past what any lane should
carry, and the plateau is already established by N = 30. The gate runs
N = 2 / 15 / 30, ~20 s for the file.

MARKED `crossgate` AND DELIBERATELY NOT `slow`. Those two markers place a
test in opposite lanes here. `test-crossgate` is push-to-main only, but the
PR lane runs `-m "not slow and not memgate and not integration"` -- which
does NOT exclude `crossgate` -- so a crossgate test runs before a merge and
again after it. Adding `slow` would remove it from the PR lane (excluded
there) AND from `test-slow` (which excludes crossgate), leaving it to the
push-only crossgate job, i.e. first read AFTER the merge that broke it.
~20 s on the PR lane is the price of this gate failing in front of a
reviewer instead of behind one.

Against Fig. 37: the shape matches (steep fall, knee near N = 15, plateau),
and momwire's plateau sits ~4 ohm BELOW the figure's ~31. Roughly 3.6 of
that is inside the eps_r assumption alone. That is a real residual and it
is not papered over -- it is why the envelope is +-6 and why this gate pins
a shape rather than a number.

THE ENVELOPE IS FALSIFIABLE, checked rather than asserted: the same deck at
sigma = 2e-4 (a decade worse soil) reads R(30) = 24.76 ohm and is REJECTED.
So +-6 is wide enough to carry the stated assumptions and narrow enough
that getting the ground wrong by a decade fails.

## Fig. 36, the 135 ft rung (momwire#838 part 2)

That screen is 82.3 m across = 3.61 lambda_m at this soil, so it refused at
the old 2 lambda_m below/below cap; part 2 moved the cap to 4 and it serves.

    N          2      15      30      60     113
    momwire  84.11  35.37   30.52   27.30   25.22
    Fig. 36   >=50   34      30      26      24.3

**Agreement here is markedly better than Fig. 37's** -- within 1.4 ohm at
every rung, against the 45 ft series sitting ~4 ohm low. Both are gated at
N = 2 / 15 / 30 with the same shape claims and the same +-6 envelope.

ONE HONEST RESIDUAL, recorded rather than smoothed over. The measurement's
whole point is that longer radials help: Fig. 36 and Fig. 37 separate by
6.7 ohm at N = 113 (24.3 against 31), and momwire separates them by only
1.75 (25.22 against 26.97). It reproduces the ORDERING and the crossing --
the curves are within 0.1 ohm at N = 60 and the 135 ft screen is lower by
N = 113 -- but it under-states how much the longer screen buys. Most of
that is the 45 ft series being low rather than the 135 ft series being
high. Not gated, because a gate on a 1.75-vs-6.7 residual would be pinning
a known disagreement; recorded so the next person does not rediscover it.

THE CAP IS SOIL-DEPENDENT, which is why it is expressed in lambda_m. The
SAME 45 ft screen is 1.20 lambda_m at sigma = 2e-3 and 3.02 at sigma = 2e-2
(lambda_m 22.81 m against 9.09 m) -- inside the old cap at one soil, past
it at the other, without changing size. That is momwire#838 part 2's second
target and it has its own gate below.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from momwire import BSplineSolver  # noqa: E402

C0 = 299792458.0
F_HZ = 3.0e6
WL = C0 / F_HZ
H_MAST = 21.4  # 77 deg
L_RADIAL = 45 * 0.3048  # 13.716 m (Fig. 37)
L_RADIAL_135 = 135 * 0.3048  # 41.148 m (Fig. 36) -- needs momwire#838 part 2
DEPTH = 6 * 0.0254  # 0.1524 m
A_WIRE = 1.63e-3  # No. 8 copper, shared (see docstring)
SIGMA = 2e-3
EPS_R = 15.0  # ASSUMED; the paper states sigma only
N_QP_PAIR = 16

# momwire#674 matched node grading. The absolute node scale is what matters,
# and this depth (0.1524) is within a millimetre of the fan's 0.15, so the
# fan's vertices carry over.
RISE_Z = [-DEPTH, -0.05, -0.0125, 0.0]
RISE_NPE = [2, 2, 2]
MAST_Z = [H_MAST, 0.5, 0.05, 0.0125, 0.0]
MAST_NPE_TAIL = [2, 3, 2]
Z_FEED = 0.05  # a vertex of MAST_Z, so the feed cannot move with the mesh

# Fig. 37, as read in momwire#838.
FIG37_OFF_SCALE_AT_N2 = 50.0
FIG37_PLATEAU = 31.0

# Envelope on the plateau. Built from measurement, not from the answer:
#   +-1.0  reading error on a 260 dpi halftone crop (momwire#838)
#   +-1.8  the eps_r assumption (measured spread 3.60 ohm over eps_r 5..30)
#   +-3.2  everything else this deck assumes -- the one-radius rule, the hub
#          spelling, and momwire's own residual against a 1937 measurement
# Total +-6.0. momwire reads 28.51 at N = 30, which sits 3.5 ohm inside the
# low edge; a model wrong by a factor of 1.5 either way still fails.
FIG37_ENVELOPE = 6.0

# Banked, not gated (242 s each). See the module docstring's ladders.
BANKED_R = {60: 27.401, 113: 26.972}
BANKED_R_135 = {60: 27.297, 113: 25.216}

# Fig. 36 (135 ft radials), read the same way as Fig. 37 in momwire#838:
# >= 50 at N = 2, then 34 / 30 / 26 / 24.3 at N = 15 / 30 / 60 / 113.
FIG36_PLATEAU_AT_30 = 30.0


def ble_deck(n_radials, n_rad=10, n_far=19, l_radial=L_RADIAL, **override):
    """BLE's geometry, hub-spelled with a graded node.

    `l_radial` selects the figure: the default 45 ft is Fig. 37, and
    `L_RADIAL_135` is Fig. 36 -- which only serves since momwire#838 part 2
    moved the below/below R1 cap to 4 lambda_m (that screen is 82.3 m across,
    3.61 lambda_m at this soil, and refused at the old 2 lambda_m).
    """
    ang = 2.0 * np.pi * np.arange(n_radials) / n_radials
    wires = [
        np.array(
            [
                (l_radial * np.cos(a), l_radial * np.sin(a), -DEPTH),
                (0.0, 0.0, -DEPTH),
            ]
        )
        for a in ang
    ]
    npe = [[n_rad] for _ in ang]
    rise_i = len(wires)
    wires.append(np.array([(0.0, 0.0, z) for z in RISE_Z]))
    npe.append(list(RISE_NPE))
    mast_i = rise_i + 1
    wires.append(np.array([(0.0, 0.0, z) for z in MAST_Z]))
    npe.append([n_far] + list(MAST_NPE_TAIL))
    build = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[
            [(i, "end") for i in range(n_radials)] + [(rise_i, "start")],
            [(rise_i, "end"), (mast_i, "end")],
        ],
        feeds=[(mast_i, H_MAST - Z_FEED, 1 + 0j)],
        wavelength=WL,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=(EPS_R, SIGMA),
        ground_model="sommerfeld",
        n_qp_pair=N_QP_PAIR,
    )
    build.update(override)
    return build


def _r_of(n, **kw):
    z, _ = BSplineSolver(**ble_deck(n, **kw)).compute_impedance()
    return z.real


@pytest.mark.crossgate
def test_ble_fig37_shape_against_the_1937_measurement():
    """The gate momwire#838 asked for: BLE Fig. 37's SHAPE, on bspline.

    Three claims, each read off the published figure rather than off
    momwire: the two-radial case leaves the 50 ohm scale, R falls
    monotonically in N, and it plateaus near 31 ohm. Bars carry margin over
    the measured ladder; see the module docstring for how the envelope was
    built.
    """
    rungs = (2, 15, 30)
    r = {n: _r_of(n) for n in rungs}

    assert r[2] > FIG37_OFF_SCALE_AT_N2, (
        f"Fig. 37's N = 2 point runs off the 50 ohm scale; momwire reads {r[2]:.2f}"
    )
    vals = [r[n] for n in rungs]
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False)), (
        f"R must fall monotonically with radial count: {r}"
    )
    assert r[2] - r[30] >= 15.0, (
        f"Fig. 37 falls from off-scale (>50) to ~31 -- at least 19 ohm, "
        f"gated at 15 with margin. momwire falls {r[2] - r[30]:.2f}"
    )
    assert abs(r[30] - FIG37_PLATEAU) < FIG37_ENVELOPE, (
        f"the plateau is {r[30]:.2f} ohm against Fig. 37's ~{FIG37_PLATEAU} "
        f"+-{FIG37_ENVELOPE}. The eps_r = {EPS_R} assumption alone is worth "
        f"~3.6 ohm here (the paper states sigma only), so a miss this large "
        f"is a model finding, not a reading error."
    )


@pytest.mark.crossgate
def test_the_ble_node_needs_no_coarse_crossing_warning():
    """The graded node is graded ENOUGH.

    Ungraded, this deck raises `CoarseCrossingNode` -- ~1427 mm of mesh
    within 150 mm of the node against a 25 mm bar, which momwire#674 values
    at several ohm on a comparable fan. The grading in `RISE_Z` / `MAST_Z`
    silences it, and this gate is what keeps the ladder above honest: an
    edit that coarsens the node would otherwise move every number in this
    file with nothing to say so.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        BSplineSolver(**ble_deck(4)).compute_impedance()
    names = sorted({w.category.__name__ for w in caught})
    assert "CoarseCrossingNode" not in names, names


@pytest.mark.crossgate
def test_the_ble_plateau_is_converged_in_quadrature():
    """momwire#674/#760: mesh convergence at fixed quadrature converges to
    the WRONG limit, so the ladder's order is pinned rather than assumed.

    On this deck q = 8 -> 16 moves R by under 0.01 ohm at N = 15, which is
    what licenses gating at 16 -- 600x inside the envelope.
    """
    lo = _r_of(15, n_qp_pair=8)
    hi = _r_of(15, n_qp_pair=32)
    assert abs(hi - lo) < 0.05, (
        f"quadrature is not converged at the gated order: q=8 {lo:.4f} vs q=32 {hi:.4f}"
    )


@pytest.mark.crossgate
def test_ble_fig36_shape_against_the_1937_measurement():
    """Fig. 36, the 135 ft rung -- the geometry momwire#838 part 2 unlocked.

    This screen is 82.3 m across = 3.61 lambda_m at BLE's soil, so it
    REFUSED at the old 2 lambda_m below/below cap and is the reason part 2
    exists. Same reading method and the same shape claims as Fig. 37, with
    the plateau envelope built the same way.

    Measured ladder (q = 16, n_rad = 10, n_far = 19), against the figure:

        N          2      15      30      60     113
        momwire  84.11  35.37   30.52   27.30   25.22
        Fig. 36   >=50   34      30      26      24.3
        cost      1.4s   3.5s    5.6s   26.1s  246.6s

    Agreement is markedly better than Fig. 37's -- within 1.4 ohm at every
    rung, against the 45 ft series sitting ~4 ohm low. N = 60 and 113 are
    banked in `BANKED_R_135`; the gate runs 2 / 15 / 30.
    """
    rungs = (2, 15, 30)
    r = {n: _r_of(n, l_radial=L_RADIAL_135) for n in rungs}
    assert r[2] > FIG37_OFF_SCALE_AT_N2, r
    vals = [r[n] for n in rungs]
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False)), r
    assert r[2] - r[30] >= 15.0, f"{r[2] - r[30]:.2f}"
    assert abs(r[30] - FIG36_PLATEAU_AT_30) < FIG37_ENVELOPE, (
        f"Fig. 36 at N = 30 is {r[30]:.2f} against the figure's "
        f"~{FIG36_PLATEAU_AT_30} +-{FIG37_ENVELOPE}"
    )


@pytest.mark.crossgate
def test_the_135_ft_screen_needs_the_part_2_cap():
    """Why part 2 was needed at all, pinned rather than asserted in prose.

    The 135 ft screen's opposite tips are 2 x 41.148 m apart, which is 3.61
    in-medium wavelengths at BLE's soil -- past the 2 lambda_m the
    below/below remainder used to be tabulated to, and comfortably inside
    the 4 lambda_m it is tabulated to now.
    """
    from momwire import _ground_refl
    from momwire import _sommerfeld_below as below

    om = 2.0 * np.pi * F_HZ
    k2 = 2.0 * np.pi * F_HZ / C0
    eps_t = _ground_refl.eps_tilde((EPS_R, SIGMA), om, 8.8541878128e-12)
    lam_m = below.lambda_medium(eps_t, k2)
    span = 2.0 * L_RADIAL_135 / lam_m
    assert 2.0 < span < below._SOMM_BELOW_R1_CAP_LAMBDA_M, (
        f"the 135 ft screen spans {span:.2f} lambda_m; it must be past the "
        f"old 2 lambda_m cap (or this gate proves nothing) and inside the "
        f"current {below._SOMM_BELOW_R1_CAP_LAMBDA_M}"
    )
    # and it actually solves
    assert np.isfinite(_r_of(2, l_radial=L_RADIAL_135))


@pytest.mark.crossgate
def test_the_45_ft_screen_survives_a_high_conductivity_soil():
    """momwire#838 part 2's second target, and the reason the cap had to move
    in LAMBDA_M rather than in metres.

    lambda_m shrinks as conductivity rises, so a screen that is comfortably
    inside the cap at one soil can refuse at another WITHOUT changing size.
    BLE's own 45 ft screen is 1.20 lambda_m at sigma = 2e-3 and 3.02 at
    sigma = 2e-2 -- past the old cap, inside the new one. This was measured
    while building the Fig. 37 gate (a sigma sweep there hit the refusal),
    and it is what put a soil axis into part 2's scope.
    """
    from momwire import _ground_refl
    from momwire import _sommerfeld_below as below

    om = 2.0 * np.pi * F_HZ
    k2 = 2.0 * np.pi * F_HZ / C0
    hi_sigma = 2e-2
    eps_t = _ground_refl.eps_tilde((EPS_R, hi_sigma), om, 8.8541878128e-12)
    span = 2.0 * L_RADIAL / below.lambda_medium(eps_t, k2)
    assert 2.0 < span < below._SOMM_BELOW_R1_CAP_LAMBDA_M, f"{span:.2f} lambda_m"
    r = _r_of(15, ground_eps=(EPS_R, hi_sigma))
    assert np.isfinite(r) and 5.0 < r < 100.0, r


# ----------------------------------------------------------------------
# The permittivity BAND (G1-A, 2026-09-03). The paper states sigma only.
# Rather than assert one assumed eps_r, this gate asks the question the
# measurement can actually answer: does SOME permittivity in the plausible
# band for a 2e-3 S/m soil put BOTH screens on their figures at once, with no
# member of the band breaking the shape? Measured (momwire#838, the G1-A
# comment): at N = 30 and sigma 2e-3, 45 ft reads 31.53 / 28.51 / 27.94 over
# eps_r 5 / 15 / 30 and 135 ft reads 29.72 / 30.52 / refused (eps_r 30 puts the
# 135 ft screen past the 4 lambda_m cap). eps_r 5 puts
# both within 0.6 ohm of the figures; the assumed 15 puts the 45 ft screen
# 2.5 ohm low. The N = 113 separation goes 1.75 -> 5.51 ohm against the
# measured 6.7 at eps_r 5.
#
# This does not adopt eps_r 5: envelopes come from the measurement, not from
# what fits, and the site's permittivity is unknown. It records that the
# model is consistent with the figures for a physically plausible soil and
# fails if it stops being so anywhere in the band.
# ----------------------------------------------------------------------

EPS_BAND = (5.0, 15.0, 30.0)
# Joint bar at N = 30: the figure reading error (+-1.0, momwire#838) plus a
# model allowance of 1.5. The best band member measured 0.53 / 0.28 ohm off.
JOINT_BAR_AT_30 = 2.5


def _r_both_at_30(eps):
    """(R_45ft, R_135ft) at N = 30 for this eps_r, or None where the 135 ft
    screen refuses (the 4 lambda_m cap: lambda_m grows as |eps~| falls)."""
    r45 = _r_of(30, ground_eps=(eps, SIGMA))
    try:
        r135 = _r_of(30, l_radial=L_RADIAL_135, ground_eps=(eps, SIGMA))
    except ValueError:
        return r45, None
    return r45, r135


@pytest.mark.crossgate
def test_some_plausible_permittivity_puts_both_screens_on_their_figures(
    record_property,
):
    fits = {}
    ran = 0
    for eps in EPS_BAND:
        r45, r135 = _r_both_at_30(eps)
        record_property(f"eps{eps:g}_45ft_N30", f"{r45:.2f}")
        # Shape, every band member: off the 50 ohm scale at N = 2, then
        # falling to the plateau.
        r2 = _r_of(2, ground_eps=(eps, SIGMA))
        assert r2 > FIG37_OFF_SCALE_AT_N2 > r45, (eps, r2, r45)
        if r135 is None:
            record_property(f"eps{eps:g}_135ft_N30", "refused (4 lambda_m cap)")
            continue
        ran += 1
        record_property(f"eps{eps:g}_135ft_N30", f"{r135:.2f}")
        miss = max(abs(r45 - FIG37_PLATEAU), abs(r135 - FIG36_PLATEAU_AT_30))
        fits[eps] = miss
    assert ran >= 2, (
        f"the band must have at least two members that serve both screens: {fits}"
    )
    best = min(fits, key=fits.get)
    record_property("best_eps_r", best)
    record_property("best_joint_miss_ohm", fits[best])
    assert fits[best] <= JOINT_BAR_AT_30, (
        f"no permittivity in {EPS_BAND} puts both BLE screens within "
        f"{JOINT_BAR_AT_30} ohm of their N = 30 figures at once (joint misses "
        f"{ {k: round(v, 2) for k, v in fits.items()} }). The model has stopped "
        "being consistent with the 1937 measurement for any plausible soil; "
        "do NOT fix this by widening the band or adopting the best eps_r."
    )
