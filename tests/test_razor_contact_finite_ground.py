"""Razor's grounded tent over a FINITE ground — the lane momwire#624 opened.

`RazorSolver` refused ground contact over any finite ground from momwire#398
until momwire#624. `docs/design/contact-over-finite-ground.md` §7 **D6** asked
whether it should have the capability at all and was the one decision point
the 2026-08-18 record never answered; §5.5 named the experiment that would
settle it and said "until that experiment runs, Stage 3 has no schedule".

**What the experiment found, because it is not what the study predicted.**
§4.3 diagnosed the refusal's subject as a missing term: over a finite ground
the T2 drop discards `(1 − w_Φ)·M0(plane)` rather than zero, so restoring it
was to be the fix. Implemented behind a flag and measured, it does not
survive:

* against the binary it is WORSE at full strength — poor soil 3.384 → 3.906 Ω
  at N = 61. One coefficient (≈ 0.4) is the argmin on every lossy ground, but
  at a fixed mesh, which is a fit;
* on the STUBBED LADDER — momwire against momwire, no reference at all, where
  a self-consistent contact node must give an answer independent of a
  vanishing grounded stub — coefficient 0 is flattest on every row by an
  order of magnitude, on both soils and both ground models. At 0.4 the ladder
  slides 42.18+25.82j → 33.58+16.50j as the stub shrinks, converging back
  onto the coefficient-0 answer. No SCALE for the term is self-consistent.

So the refusal was not protecting a defect that a term would repair, and what
it was costing was five of the 62 EZNEC captures and the most common HF model
there is. This file is the row it opened, gated.

THE BAR IS D1'S, LITERALLY
--------------------------
The issue's second ask was to hold razor to the bar bspline's contact row
already ships under rather than to "match the binary" — the asymmetry that
kept these decks refused was shipping an unexplained 3.3 Ω gap on one trunk
while refusing over a diagnosed term on the other. So the soil and dielectric
bars here are **imported from** `test_contact_nec5_lane`, not copied: razor
passes the same constants, and a maintainer who re-derives bspline's envelope
moves razor's in the same edit.

WHERE THE TWO TRUNKS DIFFER IN SHAPE, AND WHY THAT IS NOT HIDDEN
----------------------------------------------------------------
D2 gave bspline two bar shapes because one bar "would be a lie on one half of
the table". The same is true across trunks, and measured (2026-08-25, both
geometries, N = 11…81 / 12…96):

    ground   razor, coarse → fine              bspline finest   shape
    sea      0.0025 0.0033 0.0049 0.0052 0.0076   0.1614        razor GROWS
    vgood    0.3913 0.4006 0.4042 0.4052 0.4036   0.1757        razor saturates
    avg      1.3525 1.3793 1.3925 1.3966 1.3970   1.2712        both saturate
    poor     3.2897 3.3464 3.3746 3.3843 3.3885   3.3274        both saturate
    diel     4.2211 4.2867 4.3196 4.3322 4.3376   4.3410        both flat

razor's residual grows-then-flattens on every ground, including the two where
bspline's DECAYS. So razor's high-|ε̃| rows cannot take bspline's decay bar
and are not given a forged one: `sea` is pinned on LEVEL only and says out
loud that it is still growing, and `vgood` gets an envelope with the same
saturation check the soil rows carry.

On sea water razor is twenty times closer to the binary than bspline
(0.008 Ω against 0.161), which is the twin claim doing what it is for. On the
low-ε_r rows the two trunks agree to within 0.07 Ω of each other's miss —
they share the gap, which is study §5.4's open question and not razor's.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from golden_contact_nec5 import CONTACT_LADDERS, GROUND_EPS
from momwire import RazorSolver

# The bars, and the geometry, from the trunk lane that already owns them.
# Importing rather than restating is the whole content of "the bar is D1's":
# two copies of 1.5 and 4.0 would be two numbers to keep equal, and the point
# of the issue was that razor is held to the SAME one.
from test_contact_nec5_lane import (
    _DIEL_ENVELOPE,
    _DIEL_FLATNESS,
    _ENVELOPE,
    ENVELOPE_GROUNDS,
    GEOMS,
    RAD,
    WL,
)
from test_contact_nec5_lane import _residuals as _bspline_residuals

# razor's own two rows. `sea` carries no saturation claim — see the module
# docstring's table and `test_the_sea_row_is_pinned_on_level_not_on_decay`.
_RAZOR_LEVEL = {"sea": 0.02, "vgood": 0.50}


@functools.lru_cache(maxsize=None)
def _razor_z(geom, n, ground):
    """`RazorSolver(nec5_quadrature=True)`, base-fed at the grounded knot.

    The same antenna, mesh and feed site `test_contact_nec5_lane._momwire_z`
    hands `BSplineSolver`, so the two trunks' columns are differenced against
    one shared set of printed NEC-5 numbers rather than against two captures
    that would have to be kept identical.
    """
    wires, split = GEOMS[geom]
    kw = dict(
        wires=wires(),
        n_per_edge_per_wire=split(n),
        wire_radius=RAD,
        wavelength=WL,
        ground_z=0.0,
        feed_arclength=0.0,
        nec5_quadrature=True,
    )
    if ground != "pec":
        kw.update(ground_eps=GROUND_EPS[ground], ground_model="sommerfeld")
    z, _ = RazorSolver(**kw).compute_impedance()
    return complex(z)


def _residuals(geom, ground):
    """`|delta_razor - delta_nec5|` at every rung, coarse to fine."""
    nec = dict(CONTACT_LADDERS[geom][ground])
    pec = dict(CONTACT_LADDERS[geom]["pec"])
    out = []
    for n in sorted(nec):
        d_n5 = nec[n] - pec[n]
        d_mw = _razor_z(geom, n, ground) - _razor_z(geom, n, "pec")
        out.append(abs(d_mw - d_n5))
    return out


# --------------------------------------------------------------------------
# 1. The parity gate momwire#624 actually turns on
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("geom", sorted(GEOMS))
@pytest.mark.parametrize("ground", ("sea", "vgood", "avg", "poor", "diel"))
def test_the_contact_residual_is_bounded_on_every_ground(geom, ground):
    """BOUNDED, not accurate, is what decides whether this row can ship.

    The direct-field trunk's contact residual DIVERGES under refinement
    (momwire#282) and cannot be served at any bar; the mixed-potential
    trunk's grows and then flattens, which D1 accepted under an envelope.
    §4.3 predicted razor sits on the second side — its contact charge is a
    bounded doublet, −1/h on the real segment and +1/h on its image, not a
    1/Δ point charge — and this is that prediction gated on all five
    grounds at once rather than only on the two the envelope rows watch.

    The bar is deliberately crude (an ohm of headroom over the worst row,
    which is `diel` at 4.34) because a divergence is not a subtle effect:
    the shape this excludes doubles.
    """
    res = _residuals(geom, ground)
    assert max(res) < 6.0, f"{geom}/{ground}: unbounded-looking ladder {res}"
    assert all(np.isfinite(r) for r in res)
    # The tail is flat compared with the head on every ground: the last
    # increment cannot exceed the first one's magnitude. That holds even on
    # `sea`, where the growth does not stop — it is slow, not accelerating.
    incs = [b - a for a, b in zip(res, res[1:])]
    assert abs(incs[-1]) <= abs(incs[0]) + 5e-3, (
        f"{geom}/{ground}: increments accelerating down the ladder: {incs}"
    )


# --------------------------------------------------------------------------
# 2. The soil rows — bspline's own envelope, unmodified
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("geom", sorted(GEOMS))
@pytest.mark.parametrize("ground", ENVELOPE_GROUNDS)
def test_razor_ships_inside_the_spline_trunks_own_envelope(geom, ground):
    """The issue's ask, as an assert: `_ENVELOPE` is imported, not restated.

    Measured maxima are 1.3996 (avg) and 3.3931 (poor) against pins of 1.5
    and 4.0 — 7 % and 18 % of headroom, the same order bspline's own rows
    carry. Do not tighten by hand for the same reason the trunk lane says
    not to: the number being held is momwire#282's known gap, and the way it
    comes down is by someone changing the contact node, at which point both
    trunks' pins fail together and get re-derived together.

    Average soil's 7 % is the tight one. It is also the row where the two
    trunks are closest (1.397 against 1.271), so a shift that trips this
    without tripping the trunk lane is more likely razor's than the ground's.
    """
    res = _residuals(geom, ground)
    bar = _ENVELOPE[ground]
    assert max(res) <= bar, (
        f"{geom}/{ground}: razor's residual {max(res):.4f} escaped the "
        f"{bar} ohm envelope bspline ships under: {['%.4f' % r for r in res]}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("geom", sorted(GEOMS))
@pytest.mark.parametrize("ground", ENVELOPE_GROUNDS + ("vgood", "diel"))
def test_the_residual_saturates_rather_than_diverging(geom, ground):
    """What makes each envelope a claim instead of a shrug, on four rows.

    Same rule as the trunk lane's — the last increment under a quarter of
    the first's — and it reaches further here, because razor saturates on
    `vgood` and `diel` too where bspline decays and is flat respectively.
    Measured ratios are 0.01-0.09.

    `sea` is excluded, and that exclusion is the honest part: see the next
    test, which pins it on level and says why it gets no saturation claim.
    """
    res = _residuals(geom, ground)
    first = res[1] - res[0]
    last = res[-1] - res[-2]
    assert first > 0.0, f"{geom}/{ground}: residual did not grow at all: {res}"
    assert last <= 0.25 * first, (
        f"{geom}/{ground}: residual still climbing — last increment "
        f"{last:.4f} against the first's {first:.4f}: {['%.4f' % r for r in res]}"
    )


# --------------------------------------------------------------------------
# 3. The two rows where razor's SHAPE differs from the trunk's
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("geom", sorted(GEOMS))
def test_the_sea_row_is_pinned_on_level_not_on_decay(geom):
    """razor is 20x closer than bspline here — and it is still growing.

    Both halves matter. bspline's sea row DECAYS (0.6503 → 0.1614) and is
    gated as a convergence claim; razor's climbs (0.0025 → 0.0076) and would
    fail that gate, so it does not get one. What it gets is a LEVEL: at the
    finest rung it sits three orders below the soil rows, and the pin is
    2-3x that level rather than the trunk lane's tight 25 % because there is
    real growth to leave room for.

    If this ever reaches 0.02 Ω that is a finding and not a re-baseline —
    it would mean the near-PEC limit is drifting on the trunk whose whole
    claim is faithfulness to the reference near it.
    """
    res = _residuals(geom, "sea")
    assert max(res) <= _RAZOR_LEVEL["sea"], (
        f"{geom}/sea: {max(res):.4f} over the {_RAZOR_LEVEL['sea']} ohm "
        f"level pin: {['%.4f' % r for r in res]}"
    )
    # The comparative claim, on the same instrument: this is the row where
    # being NEC-5's own formulation shows, so it is asserted rather than
    # only recorded in the docstring.
    assert res[-1] < _bspline_residuals(geom, "sea")[-1] / 10.0, (
        f"{geom}/sea: razor {res[-1]:.4f} no longer an order below bspline's "
        f"{_bspline_residuals(geom, 'sea')[-1]:.4f}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("geom", sorted(GEOMS))
def test_the_very_good_row_saturates_where_the_trunk_decays(geom):
    """The one ground where razor is the worse of the two, stated plainly.

    0.405 against bspline's 0.176 at the finest monopole rung. It is also
    the row where bspline is not monotone — it passes THROUGH agreement at
    N = 41 and comes back up — so "worse" here is worth less than it looks;
    what is gated is razor's own level and its saturation, not a comparison.
    """
    res = _residuals(geom, "vgood")
    assert max(res) <= _RAZOR_LEVEL["vgood"], (
        f"{geom}/vgood: {max(res):.4f} over the {_RAZOR_LEVEL['vgood']} ohm "
        f"envelope: {['%.4f' % r for r in res]}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("geom", sorted(GEOMS))
def test_the_dielectric_row_is_the_same_flat_limit_difference(geom):
    """The lossless half-space, on the trunk lane's own two constants.

    This is the row that killed study §5.4's candidate 3 — a missing LOSS
    term cannot be at its maximum over a ground with no loss in it — and
    razor lands on it within 0.01 Ω of bspline (4.3376 against 4.3410).
    Two trunks with different bases and different testing agreeing that
    closely about the size of a limit difference is the strongest single
    piece of evidence that the gap belongs to neither of them.
    """
    res = _residuals(geom, "diel")
    assert max(res) <= _DIEL_ENVELOPE, (
        f"{geom}/diel: {max(res):.4f} over the {_DIEL_ENVELOPE} ohm envelope: "
        f"{['%.4f' % r for r in res]}"
    )
    assert max(res) - min(res) <= _DIEL_FLATNESS, (
        f"{geom}/diel: moved {max(res) - min(res):.4f} ohm down the ladder, "
        f"which is not the flat limit difference this row claims: {res}"
    )


# --------------------------------------------------------------------------
# 4. PEC is untouched, which is what makes the change safe to make
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_the_new_row_recovers_pec_as_epsilon_grows():
    """The analytic limit (§3.3), on the path momwire#624 opened.

    Everything above is referenced to captured NEC-5 printouts. This one is
    not referenced to anything: a half-space whose permittivity runs away
    IS a perfect conductor, so the finite-ground contact answer has to walk
    onto razor's own PEC contact answer, and razor computes both. That makes
    it the one gate here that would still work if every capture were lost —
    and the one that exercises the newly-served code path against a limit
    rather than against a level.

    Measured at N = 21, ε_r = 10³…10⁶ with σ = 0: 5.47 → 2.20 → 0.74 →
    0.236 Ω from PEC, a ratio near √10 per decade. That rate is the study's
    own reading of the binary at contact (§3.3, "recovers PEC, at C₂'s
    rate"), so what is gated is the DECAY and the final level, not a
    convergence order fitted here.

    momwire#624 changed no PEC arithmetic, which is why this limit is
    reachable at all: the lifted refusal fired on `ground_eps is not None`
    and never guarded a PEC solve, and at PEC there is no `w_Φ` table, so
    the branch a finite-ground correction would live in does not exist.
    """
    wires, split = GEOMS["monopole"]
    kw = dict(
        wires=wires(),
        n_per_edge_per_wire=split(21),
        wire_radius=RAD,
        wavelength=WL,
        ground_z=0.0,
        feed_arclength=0.0,
        nec5_quadrature=True,
    )
    z_pec, _ = RazorSolver(**kw).compute_impedance()
    gaps = []
    for eps_r in (1e3, 1e4, 1e5, 1e6):
        z, _ = RazorSolver(
            **kw, ground_eps=(eps_r, 0.0), ground_model="sommerfeld"
        ).compute_impedance()
        gaps.append(abs(complex(z) - complex(z_pec)))

    assert gaps[-1] <= 0.35, (
        f"contact over eps_r=1e6 sits {gaps[-1]:.4f} ohm from its own PEC "
        f"answer: {['%.4f' % g for g in gaps]}"
    )
    for coarse, fine in zip(gaps, gaps[1:]):
        assert fine <= coarse / 2.0, (
            f"the PEC limit stopped converging: {['%.4f' % g for g in gaps]}"
        )


# --------------------------------------------------------------------------
# 5. The residual the spike leaves, with a target
# --------------------------------------------------------------------------

# The stubbed ladder (§3.8 ladder B), which needs no binary: replace the
# contacting element with a vanishing grounded stub and the antenna is
# unchanged, so a self-consistent contact node must give an h-independent
# answer. `scripts/spike_contact_stub_ladder.py` is the full sweep; this is
# its coefficient-0 column, which is what shipped.
#
# Two corrections from the stage-2 record are built in and neither is
# optional: the feed sits on the RADIATOR (fed at the shrinking base, the
# ladder measures the delta-gap source model instead — 53 Ω out at a 0.1 mm
# stub), and the mesh above the stub is held FIXED on the original knots
# (re-meshing every rung drifts the PEC control 2.5 Ω, a mesh artefact with
# no contact node in it).
_STUB_HEIGHTS = (0.1, 0.03, 0.01, 0.003, 0.001)
_STUB_MONO_H = 5.3535
_STUB_N = 20
_STUB_PEC_FLATNESS = 0.01
_STUB_FINITE_SPREAD = 0.8


def _stub_z(h, soil):
    d = _STUB_MONO_H / _STUB_N
    knots = [h] + [(i + 1) * d for i in range(_STUB_N)]
    kw = dict(
        wires=[
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, h]]),
            np.array([[0.0, 0.0, z] for z in knots]),
        ],
        n_per_edge_per_wire=[[1], [1] * _STUB_N],
        wire_radius=0.005,
        wavelength=WL,
        ground_z=0.0,
        nec5_quadrature=True,
        feeds=[(1, (_STUB_N // 2) * d - h, 1.0 + 0j)],
    )
    if soil is not None:
        kw.update(ground_eps=soil, ground_model="sommerfeld")
    z, _ = RazorSolver(**kw).compute_impedance()
    return complex(z)


@pytest.mark.slow
def test_the_contact_node_is_self_consistent_to_under_an_ohm():
    """What momwire#624 leaves behind, pinned so that closing it is visible.

    The PEC control first, because it certifies the instrument: the term
    under test is identically zero there, so a flat PEC ladder means any
    finite-ground spread is the ground's and not the harness's. Measured
    2.15e-03 Ω across a hundredfold range of stub heights.

    Then the finite grounds, which spread 0.21-0.55 Ω — two orders worse.
    So the contact node IS internally inconsistent over a finite ground, by
    about half an ohm, and that is the honest description of what shipping
    this row costs. It is also the best thing the spike produced: a residual
    with a TARGET, on an instrument needing no licensed binary and no
    capture, which is what stage 3 did not have. The number to drive to zero
    is this one, not the 3.3 Ω of §5.4 — that gap is shared with bspline and
    is a different question.
    """
    pec = [_stub_z(h, None) for h in _STUB_HEIGHTS]
    pec_spread = max(abs(z - pec[-1]) for z in pec[:-1])
    assert pec_spread <= _STUB_PEC_FLATNESS, (
        f"the PEC control is not flat ({pec_spread:.2e} ohm), so no "
        f"finite-ground reading below can be attributed to the ground: {pec}"
    )

    for name, soil in (("average", (13.0, 0.005)), ("poor", (5.0, 0.001))):
        zs = [_stub_z(h, soil) for h in _STUB_HEIGHTS]
        spread = max(abs(z - zs[-1]) for z in zs[:-1])
        assert spread <= _STUB_FINITE_SPREAD, (
            f"{name}: the contact node's self-consistency residual "
            f"{spread:.4f} ohm exceeded its {_STUB_FINITE_SPREAD} ohm pin: {zs}"
        )
        assert spread > pec_spread, (
            f"{name}: spread {spread:.4f} is at the PEC control's level — "
            "if the contact node became consistent over a finite ground, "
            "this pin has been closed and wants re-deriving, not deleting"
        )
