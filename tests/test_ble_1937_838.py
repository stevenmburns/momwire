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

WHY 45 FT AND NOT 135. momwire#838 part 2 is the R1 cap, and this is where
it bites: the below/below remainder is tabulated to 2 lambda_m, and a
screen's opposite tips are 2 x L apart. At BLE's sigma = 2e-3, lambda_m =
22.81 m, so the cap is 45.6 m against this screen's 27.4 m diameter --
comfortable. The 135 ft screen is 82.3 m across and refuses. The margin is
soil-dependent, not fixed: the SAME 45 ft screen refuses at sigma = 2e-2
(lambda_m 9.09 m, cap 18.2 m), which is worth knowing before part 2 picks a
target.
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
L_RADIAL = 45 * 0.3048  # 13.716 m
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

# Banked, not gated (242 s). See the module docstring's ladder.
BANKED_R = {60: 27.401, 113: 26.972}


def ble_deck(n_radials, n_rad=10, n_far=19, **override):
    """BLE's Fig. 37 geometry, hub-spelled with a graded node."""
    ang = 2.0 * np.pi * np.arange(n_radials) / n_radials
    wires = [
        np.array(
            [
                (L_RADIAL * np.cos(a), L_RADIAL * np.sin(a), -DEPTH),
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
