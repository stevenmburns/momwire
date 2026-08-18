"""The ground-CONTACT-over-a-finite-ground lane (momwire#282 stage 1).

momwire has solved a base-fed vertical standing in lossy earth since
momwire#151 and, until `docs/design/contact-over-finite-ground.md`, nothing
in the tree had ever compared that answer to a reference engine. The gates
that existed (`test_282_*`, `test_291_*`, `test_g16c`) are SELF-CONSISTENCY
gates: they check that the answer stops walking and that the two trunks
agree with each other. Both trunks agreeing on a wrong answer would pass
every one of them, and on the reflection-coefficient ground that is exactly
what was happening — which is why that row was withdrawn by the same stage.

This file is the external half. `tests/golden_contact_nec5.py` holds the
impedances the licensed NEC-5 binary PRINTED for the same decks;
`scripts/capture_contact_nec5_lane.py` regenerated them and carries the deck
cards, the run recipe and the reasoning behind the two bar shapes. No binary
is needed here.

WHAT IS COMPARED
----------------
The ground-induced shift `delta = Z(ground) - Z(PEC)` at matched geometry
and matched N — not `Z`. At PEC contact the two codes already sit ~1.3 Ω
apart in X on pure basis difference; differencing the columns cancels that
and leaves what the ground did.

THE TWO BAR SHAPES (maintainer decision D2, 2026-08-18)
--------------------------------------------------------
The table has two behaviours and they are two different claims:

  DECAY, on the high-|eps~| grounds (sea water, "very good"). The residual
  SHRINKS down the ladder. Gated at the finest rung's measured level + 25 %,
  with the finest rung required to beat the coarsest — a real agreement
  claim, and the shape `docs/design/solver-architecture.md` §6.6 calls the
  production-lane rule.

  ENVELOPE, on the low-eps_r grounds (average, poor soil). The residual
  GROWS with mesh and then flattens. That is a difference of LIMITS, not a
  discretization artifact, and there is no honest tight bar for it. Gated by
  an envelope pin at the measured saturation PLUS a check that it really has
  saturated, which is what keeps the envelope a claim rather than a shrug.

The envelope rows are momwire#282 stage 2's subject: momwire under-predicts
the ground-loss resistance of a grounded vertical over poor soil by ~2.7 Ω
(~6 points of efficiency on a full-size 40 Ω monopole), the cause is not
known, and study §5.4 names the candidates. If stage 2 closes it, these pins
are what should fail.
"""

import functools

import numpy as np
import pytest

from golden_contact_nec5 import CONTACT_LADDERS, GROUND_EPS
from momwire import BSplineSolver

C = 299792458.0
FREQ_MHZ = 14.0
WL = C / (FREQ_MHZ * 1e6)
RAD = 0.005
MONO_H = 5.3535
INVL_RISER = MONO_H / 2.0
INVL_TOP = MONO_H / 2.0

# Geometry, in momwire's spelling. The deck cards live in the capture
# script; these two must stay the same antennas, which is what makes the
# difference of columns a comparison rather than a coincidence.
GEOMS = {
    "monopole": (
        lambda: [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_H]])],
        lambda n: [[n]],
    ),
    "invl": (
        lambda: [
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, INVL_RISER],
                    [INVL_TOP, 0.0, INVL_RISER],
                ]
            )
        ],
        lambda n: [[n // 2, n - n // 2]],
    ),
}

DECAY_GROUNDS = ("sea", "vgood")
ENVELOPE_GROUNDS = ("avg", "poor")


@functools.lru_cache(maxsize=None)
def _momwire_z(geom, n, ground):
    """`BSplineSolver(degree=2, feed_model="segment")`, base-fed at the
    grounded knot — the mixed-potential trunk, which is the one that serves
    contact over a finite ground and the one this lane gates."""
    wires, split = GEOMS[geom]
    kw = dict(
        wires=wires(),
        n_per_edge_per_wire=split(n),
        wire_radius=RAD,
        wavelength=WL,
        degree=2,
        feed_model="segment",
        feed_wire_index=0,
        feed_arclength=0.0,
        ground_z=0.0,
    )
    if ground != "pec":
        kw.update(ground_eps=GROUND_EPS[ground], ground_model="sommerfeld")
    z, _ = BSplineSolver(**kw).compute_impedance()
    return complex(z)


def _residuals(geom, ground):
    """`|delta_momwire - delta_nec5|` at every rung of the ladder, coarse
    to fine, with `delta = Z(ground) - Z(PEC)` on each side."""
    nec = dict(CONTACT_LADDERS[geom][ground])
    pec = dict(CONTACT_LADDERS[geom]["pec"])
    out = []
    for n in sorted(nec):
        d_n5 = nec[n] - pec[n]
        d_mw = _momwire_z(geom, n, ground) - _momwire_z(geom, n, "pec")
        out.append(abs(d_mw - d_n5))
    return out


# --------------------------------------------------------------------------
# 1. The DECAY rows — sea water and very good ground
# --------------------------------------------------------------------------
#
# Measured 2026-08-18, |delta_mw - delta_n5| coarse -> fine:
#
#   monopole  sea    0.7039 0.5248 0.3721 0.3056 0.2703   (N = 11/21/41/61/81)
#   monopole  vgood  0.9772 0.3498 0.0053 0.1166 0.1766
#   invl      sea    0.6866 0.4882 0.3395 0.2798 0.2484   (N = 12/24/48/72/96)
#   invl      vgood  0.9011 0.3119 0.1528 0.2021 0.2398
#
# The `vgood` rows are not monotone — they pass THROUGH agreement (0.005 on
# the monopole at N = 41, which is the two codes crossing rather than
# meeting) and come back up slightly. So the claim is net decay, coarsest to
# finest, not rung-by-rung descent: pretending otherwise would be pinning a
# crossing point.

_DECAY_FINEST = {
    ("monopole", "sea"): 0.2703,
    ("monopole", "vgood"): 0.1766,
    ("invl", "sea"): 0.2484,
    ("invl", "vgood"): 0.2398,
}


@pytest.mark.parametrize("geom,ground", sorted(_DECAY_FINEST))
def test_contact_lane_decays_on_the_high_eps_grounds(geom, ground):
    """Sea water and very good ground: momwire's contact answer CONVERGES
    onto the binary's, and the finest rung is pinned at its measured level
    with 25 % of headroom.

    This is the honest half of the table. Where |eps~| is large the ground
    is close to the perfect conductor both formulations agree about, the
    residual comes down with mesh, and the two codes are saying the same
    thing about the same antenna to a quarter of an ohm.
    """
    res = _residuals(geom, ground)
    bar = 1.25 * _DECAY_FINEST[(geom, ground)]
    assert res[-1] <= bar, (
        f"{geom}/{ground}: finest-rung residual {res[-1]:.4f} > {bar:.4f} "
        f"(ladder {['%.4f' % r for r in res]})"
    )
    assert res[-1] < res[0], (
        f"{geom}/{ground}: residual did not decay down the ladder: "
        f"{['%.4f' % r for r in res]}"
    )


# --------------------------------------------------------------------------
# 2. The ENVELOPE rows — average and poor soil
# --------------------------------------------------------------------------
#
# Measured 2026-08-18, same shape:
#
#   monopole  avg    0.6715 0.9627 1.1630 1.2355 1.2712
#   monopole  poor   2.9152 3.1511 3.2691 3.3086 3.3274
#   invl      avg    0.4169 0.9018 1.1454 1.2256 1.2646
#   invl      poor   2.8454 3.1238 3.2590 3.3033 3.3244
#
# Nearly all of the poor-soil miss is in RESISTANCE: momwire's ground adds
# 1.30 Ω of R where the binary adds 3.98 Ω at the finest monopole rung. On a
# full-size 40 Ω monopole that is ~6 points of efficiency (90.1 % against
# 96.5 %) and much more on the short loaded verticals this class of user
# actually builds.

_ENVELOPE = {"avg": 1.5, "poor": 4.0}


@pytest.mark.parametrize("geom", sorted(GEOMS))
@pytest.mark.parametrize("ground", ENVELOPE_GROUNDS)
def test_contact_lane_is_inside_its_envelope_on_the_low_eps_grounds(geom, ground):
    """Average and poor soil: gated at the measured saturation, openly
    loose, with the issue number on it.

    Do not tighten this by hand. The number it is holding is a known,
    unexplained, reproducible gap — momwire#282 stage 2 — and the way it
    should come down is by someone closing it, at which point this pin
    fails and gets re-derived.
    """
    res = _residuals(geom, ground)
    bar = _ENVELOPE[ground]
    assert max(res) <= bar, (
        f"{geom}/{ground}: residual {max(res):.4f} escaped its "
        f"{bar} ohm envelope: {['%.4f' % r for r in res]}"
    )


@pytest.mark.parametrize("geom", sorted(GEOMS))
@pytest.mark.parametrize("ground", ENVELOPE_GROUNDS)
def test_the_low_eps_residual_saturates_rather_than_diverging(geom, ground):
    """What makes the envelope a claim instead of a shrug.

    An envelope pin on a residual that is still GROWING is worthless — the
    next rung escapes it. So the growth has to be shown to be stopping: the
    last rung's increment must be under a quarter of the first's. Measured
    ratios are 0.08-0.13, i.e. the growth has largely finished by the middle
    of the ladder, which is the evidence that this is a difference of limits
    and not a discretization artifact that refinement would remove.
    """
    res = _residuals(geom, ground)
    first = res[1] - res[0]
    last = res[-1] - res[-2]
    assert first > 0.0, f"{geom}/{ground}: residual did not grow at all: {res}"
    assert last <= 0.25 * first, (
        f"{geom}/{ground}: residual still climbing — last increment "
        f"{last:.4f} against the first's {first:.4f}: "
        f"{['%.4f' % r for r in res]}"
    )


def test_the_low_eps_gap_belongs_to_the_ground_and_not_to_the_geometry():
    """The question the headline deck could not answer, answered.

    The study measured the low-eps_r gap on ONE geometry (a straight
    vertical) and could not say whether it was a property of the ground or
    of that antenna. The lane carries a second: the same quarter wave BENT
    into an inverted-L, with an interior bend and a second edge between the
    contact node and the far end. The saturated residuals land on top of
    each other — 1.271 against 1.265 over average soil, 3.327 against 3.324
    over poor — to under 1 % of the number itself.

    So it is the ground. Stage 2 should look at the shared half-space
    machinery (study §5.4 candidate 1, the remainder's near-interface
    behaviour) and not at the vertical's contact node, and this test is why.
    """
    for ground in ENVELOPE_GROUNDS:
        mono = _residuals("monopole", ground)[-1]
        invl = _residuals("invl", ground)[-1]
        assert abs(mono - invl) < 0.05 * mono, (
            f"{ground}: the two geometries' saturated residuals parted — "
            f"monopole {mono:.4f}, inverted-L {invl:.4f}. If that is real, "
            "the gap is geometry-dependent after all and stage 2's scope "
            "changes."
        )


# --------------------------------------------------------------------------
# 3. The eps~ -> infinity limit, pinned at the floor it actually has
# --------------------------------------------------------------------------


def test_the_pec_limit_at_contact_floors_on_the_sommerfeld_grid():
    """A Sommerfeld ground at enormous conductivity must return the PEC
    contact answer. It returns it to ~0.55 Ω, and the floor RISES with mesh
    refinement instead of falling — 0.477 / 0.550 / 0.589 at N = 21/41/81.

    This is momwire#443 and it is not a contact defect. Study §3.7 measured
    the mechanism directly: in the near-PEC regime the true Sommerfeld
    surfaces are ~1/sqrt(eps~) small, the interpolation grid's ABSOLUTE
    error does not shrink with them, and the remainder therefore stops
    vanishing. Ground contact is simply the only geometry that queries the
    grid at the small R1 where that happens — a wire clear of the plane has
    a floor under R1 and converges to PEC at the textbook 10x per decade,
    reaching 3e-5 Ω.

    Pinned WHERE IT IS, deliberately, with room in both directions. Do not
    tighten it: fixing the grid is momwire#443's unit, and when it lands
    this pin is what should fail.
    """
    floors = []
    for n in (21, 41, 81):
        z_pec = _momwire_z("monopole", n, "pec")
        wires, split = GEOMS["monopole"]
        z_big, _ = BSplineSolver(
            wires=wires(),
            n_per_edge_per_wire=split(n),
            wire_radius=RAD,
            wavelength=WL,
            degree=2,
            feed_model="segment",
            feed_arclength=0.0,
            ground_z=0.0,
            ground_eps=(13.0, 1e8),
            ground_model="sommerfeld",
        ).compute_impedance()
        floors.append(abs(complex(z_big) - z_pec))
    assert all(0.2 < f < 1.0 for f in floors), (
        f"the near-PEC contact floor moved: {['%.4f' % f for f in floors]} "
        "(measured 0.477 / 0.550 / 0.589 at N = 21/41/81)"
    )
    assert floors[0] < floors[-1], (
        f"the floor stopped rising with mesh: {['%.4f' % f for f in floors]} "
        "— if momwire#443 was fixed, celebrate and re-derive this pin"
    )


# --------------------------------------------------------------------------
# 4. The stubbed-limit gate — no binary needed
# --------------------------------------------------------------------------
#
# Study §5.3, and the study's most useful accidental find: a contact deck
# must equal the same deck with its contact replaced by a VANISHING grounded
# stub. It is formulation-agnostic, needs no licensed binary, and catches
# every class of contact-node bookkeeping error including the model-(a) 1/Δ
# point charge that momwire#282 was originally about.
#
# One correction to how the study framed it, and it makes the instrument an
# order of magnitude sharper. The study's ladder B fed the deck AT the stub's
# grounded base, which is what NEC's `EX` on segment 1 does. But then the
# feed segment shrinks with the stub, so the ladder measures the delta-gap
# source model as much as the contact node: fed that way momwire's own PEC
# ladder is 53 Ω out at a 0.1 mm stub, which is a statement about a
# delta-gap over a 0.1 mm gap and nothing at all about the ground. Move the
# feed onto the radiator, where the segment length is fixed, and the same
# ladder holds to 3.2e-4 Ω at the same stub. The binary's own ~0.19 Ω wobble
# down its ladder is very likely the same effect seen through its own source
# model.
#
# Measured 2026-08-18 with the feed at 0.5 m up the radiator,
# |Z_stubbed(h) - Z_contact|, h = 0.1 / 1 / 10 mm:
#
#   N = 21   PEC   0.00036  0.00316  0.01185
#            avg   0.00671  0.01503  0.08483
#            poor  0.01105  0.01431  0.05118
#   N = 41   PEC   0.00032  0.00281  0.02719
#            avg   0.00726  0.01934  0.08015
#            poor  0.01056  0.01697  0.05678
#   N = 81   PEC   0.00038  0.00350  0.03127
#            avg   0.00718  0.01864  0.06380
#            poor  0.00968  0.01620  0.04800

_STUB_FEED = 0.5
_STUB_HEIGHTS = (1e-4, 1e-3, 1e-2)
_STUB_BAR = {"pec": 0.002, "avg": 0.02, "poor": 0.03}


def _stub_z(n, h, ground):
    """The monopole with its bottom `h` metres split off as a one-segment
    contacting stub — same total length, same radiator mesh, same feed
    position, one extra knot."""
    kw = dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, h], [0.0, 0.0, MONO_H]])],
        n_per_edge_per_wire=[[1, n]],
        wire_radius=RAD,
        wavelength=WL,
        degree=2,
        feed_model="segment",
        feed_arclength=_STUB_FEED,
        ground_z=0.0,
    )
    if ground != "pec":
        kw.update(ground_eps=GROUND_EPS[ground], ground_model="sommerfeld")
    z, _ = BSplineSolver(**kw).compute_impedance()
    return complex(z)


def _plain_z(n, ground):
    kw = dict(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_H]])],
        n_per_edge_per_wire=[[n]],
        wire_radius=RAD,
        wavelength=WL,
        degree=2,
        feed_model="segment",
        feed_arclength=_STUB_FEED,
        ground_z=0.0,
    )
    if ground != "pec":
        kw.update(ground_eps=GROUND_EPS[ground], ground_model="sommerfeld")
    z, _ = BSplineSolver(**kw).compute_impedance()
    return complex(z)


@pytest.mark.parametrize("ground", ("pec", "avg", "poor"))
@pytest.mark.parametrize("n", (21, 41, 81))
def test_the_contact_answer_is_the_limit_of_its_own_stubbed_ladder(n, ground):
    """The contact node's bookkeeping, checked against itself.

    A grounded end and a vanishingly short grounded stub carrying the same
    current into the same plane are the same antenna. If the contact node
    were double-counting charge — the momwire#282 pathology, model (a) of
    the study's §2.3 — the two would part company like 1/Δ as the stub
    shrank. They converge instead, to 3e-4 Ω over PEC and ~1e-2 Ω over the
    two soils, at every mesh.
    """
    plain = _plain_z(n, ground)
    ds = [abs(_stub_z(n, h, ground) - plain) for h in _STUB_HEIGHTS]
    bar = _STUB_BAR[ground]
    assert ds[0] <= bar, (
        f"N={n}/{ground}: the 0.1 mm stub sits {ds[0]:.5f} ohm from the "
        f"contact deck (bar {bar}): {['%.5f' % d for d in ds]}"
    )
    # ...and it is a LIMIT: the shorter the stub, the closer the agreement.
    assert ds[0] < ds[-1], (
        f"N={n}/{ground}: shortening the stub did not close the gap: "
        f"{['%.5f' % d for d in ds]}"
    )


def test_the_stubbed_limit_kills_the_contact_node_as_a_stage_2_candidate():
    """Study §5.4 candidate 2, and its verdict.

    The study named three possible causes for the 2.6-3.3 Ω the lane's
    envelope rows are holding, and proposed this exact experiment to
    separate one of them: *"if momwire's own stubbed limit disagrees with
    its own contact deck by ~3 Ω over poor soil while the binary's agree to
    0.19 Ω, the disagreement is in momwire's contact node, not in its
    ground."*

    It disagrees by 0.011 Ω — three hundred times smaller than the gap it
    would have to explain, and the same size as the PEC and average-soil
    figures, which carry no such gap at all. Candidate 2 is dead: whatever
    momwire is missing over poor soil, it is not in the contact node's
    bookkeeping. That leaves candidate 1 (the remainder's near-interface
    behaviour, in shared half-space machinery) and candidate 3, and it is
    the same conclusion `test_the_low_eps_gap_belongs_to_the_ground_and_
    not_to_the_geometry` reaches from the other direction.
    """
    poor = abs(_stub_z(41, 1e-4, "poor") - _plain_z(41, "poor"))
    envelope = max(_residuals("monopole", "poor"))
    assert poor < 0.05 * envelope, (
        f"the stubbed limit is {poor:.4f} ohm out against a {envelope:.4f} "
        "ohm gap to explain — if the contact node is back in the frame, "
        "stage 2's scope widens"
    )
