"""The ground field evaluator at a point in space (momwire#545 U1).

`momwire._field_point` is `_sommerfeld.remainder_field_proj`'s azimuth
combination kept as a VECTOR, so that an observer which is a point in space
rather than a wire element can be asked what the finite ground does to the
field there.  This module gates it, and the gates split four ways.

**The envelopes.**  The composition ``direct + C₂·image + remainder`` is
scored cell for cell against the licensed engine's captured near-field grids
over ``GN 0`` — 0110 and 0113 electric, 0111 magnetic, the corpus's only
magnetic oracle.  Each bar is the MEASURED worst live cell plus 25 %, the
same discipline `test_eznec_serve.py`'s impedance and near-field envelopes
use, with the measurements written down below so the next person can see what
moved rather than only that something did.  A pin that tightens is a finding;
a pin that loosens is a regression.

**The convention.**  The remainder ADDS, and that is a measurement rather than
a reading of a manual's sign.  ``- remainder`` is gated to VIOLATE 0110's
envelope, so the composition sign cannot drift back without a red test.

**The singular cell.**  0112 asks for the field at exactly (0, 0, 0), the base
of a contact-fed monopole, where the composition does not converge.  Its
subdiv ladder is gated to keep GROWING — a gate that FAILS IF FIXED, because
a ladder that converged would mean the composition changed and U3's refusal
sentence needs re-measuring.

**The bare `GD` ground (momwire#545 U2).**  A MININEC-type ground solves its
CURRENTS over a perfect image — the `Z ≡ GN 1` identity, gated on the
impedance rungs elsewhere — but its NEAR FIELD in the medium: 0108 (`GD`)
agrees with 0110 (`GN 0`) to 4.4 % worst-cell in the captures, symmetry-dead
`EY` included.  So a `GD` deck's composition is the SAME `direct + C₂·image +
remainder` this file already gates, with ε̃ taken from the deck's own medium
rather than from the PEC row `_ground_spec.ground_config` necessarily returns
for a `ground_z`-alone solver.  No new physics lands here — `_composed` grows
one branch, and the new gates are an envelope (0108), a reproduction of the
captures' own cross-agreement (0108 against 0110), an anti-aliasing gate that
FAILS if a `GD` near field is ever composed as a plain PEC image, and the
`GD` flavor of the singular-cell ladder (0107, `GD`'s own contact point).

**The collapses and the identities.**  ε̃ → 1e16 collapses the pair
``C₂·image + remainder`` onto the plain PEC image; ε̃ = 1 makes the remainder
exactly zero; the point dyad projected on an observer tangent reproduces the
matrix fill's own numbers; and the finite-difference curl the magnetic
evaluator is built on is measured against an ANALYTIC magnetic field, away
from any ground, so its own error is pinned separately from the physics it is
applied to.

The decks are built through the eznec seam's internals rather than through
`serve()`, because `serve()` refuses every capture in this file until #545 U3
routes them.  What is under test is the evaluator, not the routing.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import numpy as np
import pytest

from momwire import _field_point, _ground_refl, _ground_spec, _sommerfeld
from momwire.deck._nec5 import Nec5MininecGround, Nec5NearFieldRequest, parse_nec5
from momwire.eznec._serve import (
    SPEED_OF_LIGHT_MHZ_M,
    _NEAR_FIELD_SUBDIV,
    _cards,
    _medium,
    _port_state,
    _solver_for,
    _transform,
    build_mesh,
    structure_of,
)
from momwire.portal._portal import _element_fields, _image_moments
from test_eznec_printout import FIXTURE_DIR, extract, printout_text

DECKS = Path(FIXTURE_DIR) / "decks"


# --------------------------------------------------------------------------
# the measured tables the pins come from
# --------------------------------------------------------------------------
#
# Measured 2026-08-21 on this tree with `_field_point.reflected_field_at` (and
# `reflected_h_field_at` for 0111), composed as
#
#     direct + C₂·image + remainder
#
# with direct and image from `_element_fields` over
# `element_currents(subdiv=_NEAR_FIELD_SUBDIV)`, the image being the mirrored
# moments over z-flipped nodes with the continuity charge NEGATED (the GN 1
# image `_serve._near_field` already builds), and C₂ = (ε̃−1)/(ε̃+1) from
# `_ground_spec.ground_config` — the one owner of that coefficient — applied
# `np.multiply(coef, block, out=block)`-style, coefficient on the LEFT, per
# `_field_ground.FieldGround.image_coefficient`'s contract.
#
# A cell is LIVE when its captured magnitude exceeds 1e-4 of the table's own
# maximum.  Errors are quoted against that TABLE SCALE rather than per cell: a
# near-field grid spans decades and the composition's error is an absolute
# quantity set by the table's biggest cells, so a per-cell bar would be
# measuring the grid's dynamic range instead of the physics.  The per-cell
# readings are quoted beside them for context and are not pinned.
#
#   capture  card  column  live  worst |E| of scale  of cell   worst phase
#   0110     NE    EX        25        4.2469 %      5.3071 %    0.637 deg
#   0110     NE    EZ        25        1.2240 %      2.4101 %    1.262 deg
#   0113     NE    EX        25        0.2251 %      1.5277 %    0.241 deg
#   0113     NE    EY        25        1.7964 %      1.8126 %    0.304 deg
#   0113     NE    EZ        25        1.1156 %      1.5163 %    0.260 deg
#   0111     NH    HY        25        1.0144 %      6.7000 %    0.328 deg
#   0108     NE    EX        25        4.3732 %      5.4533 %    0.604 deg
#   0108     NE    EZ        25        1.1693 %      2.3030 %    1.235 deg
#
# 0110's worst cell is EX at (1, 0, 10) — a third of a metre below the top of
# a 10.3 m monopole and one metre out from it, the hardest place in this
# corpus to ask either formulation about, and the same cell `test_eznec_serve`
# measures its own worst GN 1 near-field reading at.  0113's is EY at
# (0.3048, 0, 3.048), and 0111's is HY at (1, 0, 8).  Everything else in all
# three tables is well inside.
#
# 0108 is `GD` (momwire#545 U2), its `NE` card identical to 0110's — same
# grid, same worst cell (1, 0, 10) — but composed from the PEC-image solve's
# currents evaluated in the deck's OWN medium rather than from a Sommerfeld
# solve: `_composed` reads ε̃ from `_medium(deck.ground, wavelength)` for a
# `Nec5MininecGround` deck instead of from `_ground_spec.ground_config`,
# which would hand back the PEC row (`eps_tilde=None`) that a `ground_z`-alone
# solver always constructs.  The envelope lands in the same class as 0110's
# own, which is the point: two different current solves read out at the same
# medium sit close together, and gate 3 below reproduces that agreement
# directly rather than trusting this table's resemblance to say so.
#
# The bars are those worst readings plus 25 %.
NEAR_POINT_BAR = {
    "0110": (0.0531, 1.58),
    "0113": (0.0225, 0.38),
    "0111": (0.0127, 0.41),
    "0108": (0.0547, 1.54),
}

# The columns a live gate is drawn on, per capture.  The rest are
# SYMMETRY-DEAD (see `DUST_BOUND`).
LIVE_COLUMNS = {
    "0110": (0, 2),  # EX, EZ — a vertical monopole makes EY exactly zero
    "0113": (0, 1, 2),  # the elevated dipole lights every column
    "0111": (1,),  # HY — the same symmetry kills HX and HZ
    "0108": (0, 2),  # EX, EZ — same vertical, now over `GD`
}

# The columns whose true value is exactly zero, gated by MAGNITUDE alone.
#
# A vertical monopole on the z axis, sampled on the y = 0 plane, has no EY and
# no HX/HZ: in the combination those components carry the factors d̂y and the
# bilinear p_x·d̂y − p_y·d̂x, both structurally zero here, and the mixed-
# potential direct and image terms kill them the same way.  The served values
# below are not small, they are EXACTLY 0.0.
#
# The engine's are not, and that is the whole reason this is a magnitude gate:
# its Sommerfeld interpolation leaves dust there (measured against each
# table's own scale, 2026-08-21)
#
#   capture  column  captured dust   served
#   0110     EY        0.0219 %      0.0 (exact)
#   0111     HX        0.2216 %      0.0 (exact)
#   0111     HZ        0.0000 %      0.0 (exact)
#   0108     EY        0.0219 %      0.0 (exact)
#
# — and the ANGLE of that dust is a random number.  0110's EY phase disagrees
# with ours by 46 deg and 0111's HX by 65 deg, which is exactly what comparing
# the arguments of two independent zeros looks like.  0108's EY phase
# disagrees by the same 46.88 deg (its captured dust is 1.7359E-02 against
# 0110's 1.7342E-02 — the two engine tables' own dust, not ours), which is the
# same "angle of zero" observation one grid over: the GD and GN 0 currents
# differ, but whatever residual their shared symmetry cannot kill lands at an
# unrelated phase on both.  So the bar is the dust bound, 1 % of table scale,
# applied to BOTH sides, and no phase is compared: the angle of zero is not a
# number.
DUST_BOUND = 0.01
DEAD_COLUMNS = {"0110": (1,), "0111": (0, 2), "0108": (1,)}

# The contact cell's subdiv ladder, measured 2026-08-21 (see
# `test_the_contact_cell_ladder_does_not_converge_on_purpose`).
#
#   subdiv     1      4       8       16      32      64
#   |E| V/m  36.560 141.029 298.246 588.130 977.790 1229.987
#
# The engine prints 8.2521E+02 ∠162.15 deg, which sits BETWEEN subdiv 16 and
# subdiv 32.
CONTACT_LADDER_SUBDIVS = (1, 4, 8, 16, 32, 64)

# 0107 is 0112's deck under `GD` instead of `GN 0` — same single point at
# (0, 0, 0), same singularity, now composed from the PEC-image solve's
# currents read out in the medium (see
# `test_the_gd_contact_cell_ladder_does_not_converge_on_purpose_either`).
# Measured 2026-08-21:
#
#   subdiv     1       4        8        16       32       64
#   |E| V/m  34.137 115.425  237.162  461.816  763.860  959.384
#
# The engine prints 6.6673E+02 for this cell's `EZ` component, which sits
# BETWEEN subdiv 16 and subdiv 32 — the same pair of rungs 0112's engine
# number sits between, though nothing about the composition guarantees that;
# it was measured, not assumed.  Not pinned as a python constant, the same
# discipline `CONTACT_LADDER_SUBDIVS`'s own rungs get above: printing a rung
# would be publishing an artefact of `_NEAR_FIELD_SUBDIV`, so only the
# structural properties (monotone, ≥3x spread, engine between subdiv 16 and
# 32) are gated below.

# 0108's own worst live cells against 0110's, both through `_composed` — the
# seam's reproduction of the captures' 4.4 %-worst-cell cross-agreement
# (momwire#516).  Measured 2026-08-21 over the live columns (EX, EZ):
#
#   column  worst distance of scale
#   EX          1.5379 %
#   EZ          0.7294 %
#
# well under the 4.4 % the captures themselves sit apart at — the currents
# differ (PEC-image solve vs Sommerfeld solve) but the evaluation medium is
# identical, so the two composed tables land closer to each other than either
# does to its own capture.  The smallest live-cell distance measured is
# 0.0121 % of scale.  That floor is the gate's other half: it is never
# exactly zero, because the two current solves are genuinely different — a
# routing bug that fed 0108 the SAME currents as 0110 (rather than its own
# PEC-image solve) would collapse this distance to 0.0 identically, and the
# floor bar catches that the envelope and alias gates do not.
CROSS_AGREEMENT_BAR = 0.0192
CROSS_AGREEMENT_FLOOR = 0.00009


# --------------------------------------------------------------------------
# the solves, through the seam's internals
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _solved(cid: str):
    """`(deck, mesh, solver, coeffs)` for one capture.

    The steps `_serve.serve` takes up to the point where it would read a
    result out, spelled here because `serve()` itself refuses every one of
    this file's decks — an ``NE``/``NH`` over a ``GN 0`` or a bare ``GD`` —
    until #545 U3 routes them.  Cached because eleven of the thirteen gates
    below want the same solves.
    """
    deck = parse_nec5(next(DECKS.glob(f"{cid}_*.nec")).read_text())
    structure = structure_of(deck)
    mesh = build_mesh(deck, structure)
    for source in deck.sources:
        for site in mesh.sites:
            if site.at == source.at:
                site.driven = True
    wavelength = SPEED_OF_LIGHT_MHZ_M / float(deck.frequency_mhz)
    medium = _medium(deck.ground, wavelength)
    cards = _cards(deck, structure, mesh)
    solver = _solver_for(deck, mesh, wavelength, medium)
    solution = solver.compute_port_solution()
    state = _port_state(deck, mesh, cards, solution.y, wavelength)
    coeffs = solution.coeffs @ (_transform(mesh) @ state.v_gap)
    return deck, mesh, solver, coeffs


def _request_points(deck):
    """`(request, points)` — the ``NE``/``NH`` card and its grid, walked X
    fastest then Y then Z, which is `_serve._near_field`'s own walk."""
    request = next(r for r in deck.requests if isinstance(r, Nec5NearFieldRequest))
    n_x, n_y, n_z = request.counts
    start = np.asarray(request.origin, dtype=float)
    step = np.asarray(request.step, dtype=float)
    return (
        request,
        np.array(
            [
                start + np.array([ix, iy, iz]) * step
                for iz in range(n_z)
                for iy in range(n_y)
                for ix in range(n_x)
            ]
        ),
    )


def _composed(cid, *, subdiv=_NEAR_FIELD_SUBDIV, sign=1.0, eps_t=None, pec_alias=False):
    """``direct + C₂·image + sign·remainder`` at the capture's own grid.

    `sign` exists for the anti-regression gate and is +1 everywhere else;
    `eps_t` overrides the deck's medium for the two collapse gates.
    `pec_alias` collapses the whole composition to ``direct + 1.0·image`` —
    no medium, no remainder — the plain PEC image a `GD` near field must NOT
    become (momwire#545 U2 gate 2, `test_the_gd_alias_...`).  Returns
    `(points, total, magnetic)`.

    For a `Nec5MininecGround` (`GD`) deck, ε̃ does not come from
    `_ground_spec.ground_config`: that function reads `solver.ground_eps`,
    which `_solver_for` never sets for `GD` — it hands the solver the SAME
    ``ground_z``-alone kwargs `GN 1` gets, the mechanical origin of the
    ``Z ≡ GN 1`` identity — so `ground_config` would hand back the PEC row
    (`eps_tilde=None`, `image_coefficient=1.0`).  That row is right for the
    CURRENTS (the identity above) and wrong for the near field: `GD` solves
    its near field IN THE MEDIUM (module docstring, "the bare `GD` ground").
    ε̃ instead comes from `_medium(deck.ground, wavelength)` — the same
    medium `_far_ground` reads for the pattern — folded by
    `_ground_refl.eps_tilde`, the one function every other ground-consuming
    solve already calls to do it.
    """
    deck, mesh, solver, coeffs = _solved(cid)
    request, points = _request_points(deck)
    magnetic = bool(request.magnetic)
    radius = min(piece.radius for piece in mesh.pieces)
    mid, moment, nodes, delta = solver.element_currents(coeffs, subdiv=subdiv)

    k = solver.k
    direct = _element_fields(points, (mid, moment, nodes, delta), k, radius, magnetic)
    mid_img, moment_img = _image_moments(mid, moment, 0.0)
    nodes_img = nodes.copy()
    nodes_img[:, 2] = -nodes[:, 2]
    block = _element_fields(
        points, (mid_img, moment_img, nodes_img, -delta), k, radius, magnetic
    )

    if pec_alias:
        return points, direct + block, magnetic

    if eps_t is None:
        if isinstance(deck.ground, Nec5MininecGround):
            wavelength = SPEED_OF_LIGHT_MHZ_M / float(deck.frequency_mhz)
            medium = _medium(deck.ground, wavelength)
            eps_t = _ground_refl.eps_tilde(
                (medium.eps_r, medium.sigma), solver.omega, solver.eps
            )
            coef = (eps_t - 1.0) / (eps_t + 1.0)
        else:
            config = _ground_spec.ground_config(solver, solver.omega)
            eps_t, coef = config.eps_tilde, config.image_coefficient
    else:
        eps_t = complex(eps_t)
        coef = (eps_t - 1.0) / (eps_t + 1.0)
    evaluate = (
        _field_point.reflected_h_field_at
        if magnetic
        else _field_point.reflected_field_at
    )
    remainder = evaluate(points, mid, moment, eps_t, 0.0, k, solver.omega)

    # Coefficient on the LEFT and the pair associated before any outer sum —
    # `_field_ground.FieldGround`'s "compose" contract, which is a contract
    # about float64 evaluation order and not a style.
    np.multiply(coef, block, out=block)
    return points, direct + (block + sign * remainder), magnetic


def _captured(cid):
    """`(magnitudes, phases, magnetic)` of a capture's near-field table."""
    (block,) = extract(printout_text(cid)).near_fields
    return (
        np.array([row.magnitudes for row in block.rows]),
        np.array([row.phases_deg for row in block.rows]),
        block.magnetic,
    )


def _score(cid, total, magnetic):
    """`(worst magnitude of table scale, worst phase degrees)` of an already-
    composed `(total, magnetic)` pair against `cid`'s capture, over the live
    columns' live cells.

    Split out of `_worst` (momwire#545 U2) so the anti-aliasing gate can
    score a composition that never went through `_composed` — the plain PEC
    image has no `sign`, no `eps_t`, nothing `_worst`'s signature has a slot
    for — without a parallel scoring loop.
    """
    want_magnitude, want_phase, want_magnetic = _captured(cid)
    assert magnetic == want_magnetic
    scale = want_magnitude.max()
    live = want_magnitude > 1e-4 * scale
    got_magnitude = np.abs(total)
    got_phase = np.degrees(np.arctan2(total.imag, total.real))
    worst_magnitude = 0.0
    worst_phase = 0.0
    for column in LIVE_COLUMNS[cid]:
        selected = live[:, column]
        assert selected.any(), f"{cid} column {column} has no live cell"
        error = np.abs(got_magnitude - want_magnitude)[:, column][selected] / scale
        drift = np.abs(
            (got_phase[:, column] - want_phase[:, column] + 180.0) % 360.0 - 180.0
        )[selected]
        worst_magnitude = max(worst_magnitude, float(error.max()))
        worst_phase = max(worst_phase, float(drift.max()))
    return worst_magnitude, worst_phase


def _worst(cid, *, sign=1.0):
    """`(worst magnitude of table scale, worst phase degrees)` over the live
    columns' live cells."""
    _, total, magnetic = _composed(cid, sign=sign)
    return _score(cid, total, magnetic)


# --------------------------------------------------------------------------
# gate 1 — the envelopes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cid", sorted(NEAR_POINT_BAR))
def test_the_composed_point_field_sits_inside_its_measured_envelope(cid):
    """Every live cell of the composition, magnitude and phase, against the
    capture (see the measured table above).

    An ENVELOPE and not an agreement claim.  momwire's B-spline formulation
    and NEC-5's are two discretizations of one physics and the gap between
    them does not shrink with anything this unit controls — it is the same
    gap `test_eznec_serve.py` prices as an impedance, read out at a point in
    space instead of at a feed.
    """
    magnitude_bar, phase_bar = NEAR_POINT_BAR[cid]
    worst_magnitude, worst_phase = _worst(cid)
    assert worst_magnitude <= magnitude_bar, (
        f"{cid}: worst live cell {100 * worst_magnitude:.4f} % of table scale, "
        f"bar {100 * magnitude_bar:.4f} %"
    )
    assert worst_phase <= phase_bar, (
        f"{cid}: worst live phase {worst_phase:.3f} deg, bar {phase_bar} deg"
    )


@pytest.mark.parametrize("cid", sorted(DEAD_COLUMNS))
def test_the_symmetry_dead_columns_are_dust_on_both_sides(cid):
    """The columns a vertical's symmetry kills, gated by MAGNITUDE only.

    Both engines have to leave the column under the dust bound and neither
    engine's phase there means anything — see the `DUST_BOUND` note.  Ours is
    exactly 0.0 and is asserted to be, because "exactly zero" is a stronger
    and more falsifiable statement than "small": it says the combination's
    d̂y and (p_x·d̂y − p_y·d̂x) factors are structurally absent rather than
    numerically lucky.
    """
    _, total, _ = _composed(cid)
    want_magnitude, _, _ = _captured(cid)
    scale = want_magnitude.max()
    got_magnitude = np.abs(total)
    for column in DEAD_COLUMNS[cid]:
        assert got_magnitude[:, column].max() == 0.0, f"{cid} column {column} is lit"
        assert want_magnitude[:, column].max() <= DUST_BOUND * scale, (
            f"{cid} column {column}: captured dust "
            f"{100 * want_magnitude[:, column].max() / scale:.4f} % of scale is "
            "past the dust bound — it is a real component, not dust"
        )


# --------------------------------------------------------------------------
# gate 2 — the convention, as a standing anti-regression
# --------------------------------------------------------------------------


def test_subtracting_the_remainder_breaks_the_envelope():
    """``direct + C₂·image − remainder`` must FAIL 0110's envelope.

    The composition sign was settled by this measurement and not by a reading
    of the theory manual's minus signs, so it is gated by the measurement.
    With the remainder subtracted 0110 reads 7.73 % of table scale and 20.6°
    against bars of 5.31 % and 1.58° — a factor of six in magnitude and a
    factor of thirteen in phase, which is what a wrong sign looks like when
    the term it is applied to is a third of the answer.

    If this test ever passes, the remainder is being applied with the wrong
    sign somewhere and the envelope gate above is passing for the wrong
    reason.
    """
    magnitude_bar, phase_bar = NEAR_POINT_BAR["0110"]
    worst_magnitude, worst_phase = _worst("0110", sign=-1.0)
    assert worst_magnitude > magnitude_bar
    assert worst_phase > phase_bar


# --------------------------------------------------------------------------
# gate 3 — the singular cell, which fails if it is ever fixed
# --------------------------------------------------------------------------


def test_the_contact_cell_ladder_does_not_converge_on_purpose():
    """0112's single grid point is (0, 0, 0) — the base of a contact-fed
    monopole — and the composition there does not converge.  Gated so.

    The point is SINGULAR.  The image cancels the base node's continuity
    charge only by ``1 − C₂ = 2/(1 + ε̃)``, so what is left of a charge that
    sits exactly at the observer is a residual 1/R² with nothing to cancel it,
    and refining the sampling only puts more of the current's charge closer to
    the point.  The ladder therefore GROWS without limit; the engine's printed
    8.2521E+02 ∠162.15° is not the converged answer either — it sits between
    our subdiv-16 and subdiv-32 rungs and is its own discretization's
    regularization of the same singularity.

    Printing any rung would be publishing an artefact of
    ``_NEAR_FIELD_SUBDIV``; printing the engine's would be tuning a sampling
    constant against an oracle.  Both are forbidden (the seam rule, and the
    no-pin discipline), which is why #545 U3 NARROWS the near-field refusal to
    this observation point instead of deleting it.

    **This gate fails if the ladder is ever fixed**, and that failure is the
    signal to re-measure U3's refusal sentence: a ladder that converged would
    mean the composition changed.
    """
    rungs = []
    for subdiv in CONTACT_LADDER_SUBDIVS:
        _, total, _ = _composed("0112", subdiv=subdiv)
        rungs.append(float(np.linalg.norm(total[0])))
    assert all(b > a for a, b in zip(rungs, rungs[1:])), rungs
    assert rungs[-1] >= 3.0 * rungs[2], rungs

    # The engine's own number sits inside the ladder rather than at its end,
    # which is the sharpest way to say neither party has converged.
    (block,) = extract(printout_text("0112")).near_fields
    captured = block.rows[0].magnitudes[2]
    assert rungs[3] < captured < rungs[4], (rungs, captured)


def test_the_gd_contact_cell_ladder_does_not_converge_on_purpose_either():
    """0107 is 0112's deck under `GD` instead of `GN 0` — the same single
    grid point at (0, 0, 0), the same contact-fed base, now through the
    PEC-image-solved-currents / in-medium composition U2 adds.

    The singularity above does not care which ground supplied the currents:
    the base node's continuity charge sits exactly at the observer either
    way, and the image cancels it only by ``1 − C₂`` — a finite factor at a
    point where the residual charge's own field is infinite.  The ladder
    therefore grows here too (see the measured-table comment following
    `CONTACT_LADDER_SUBDIVS` above for the rungs), and the engine's
    printed 6.6673E+02 sits between the same pair of rungs (subdiv 16 and
    subdiv 32) 0112's engine number sits between — measured, not assumed to
    follow from that coincidence.

    **This gate fails if the ladder is ever fixed**, the same signal
    `test_the_contact_cell_ladder_does_not_converge_on_purpose` above sends
    for `GN 0`: a `GD` ladder that converged would mean the composition
    changed and U3's refusal sentence for this observation point needs
    re-measuring on this ground too.
    """
    rungs = []
    for subdiv in CONTACT_LADDER_SUBDIVS:
        _, total, _ = _composed("0107", subdiv=subdiv)
        rungs.append(float(np.linalg.norm(total[0])))
    assert all(b > a for a, b in zip(rungs, rungs[1:])), rungs
    assert rungs[-1] >= 3.0 * rungs[2], rungs

    (block,) = extract(printout_text("0107")).near_fields
    captured = block.rows[0].magnitudes[2]
    assert rungs[3] < captured < rungs[4], (rungs, captured)


# --------------------------------------------------------------------------
# gate 4 — the collapses and the two identities
# --------------------------------------------------------------------------


def test_a_pec_limit_ground_collapses_onto_the_plain_image():
    """ε̃ → 1e16: ``C₂·image + remainder`` becomes the plain PEC image.

    NEC's decomposition (theory manual eqs 136-147) is the exact image scaled
    by C₂ plus a smooth remainder, so the PEC limit has to take C₂ → 1 and
    the remainder → 0 — the same limit `_ground_spec.ground_config` documents
    for the fill, now checked on the point evaluator.  Measured at 4.1e-9 of
    the table's own scale against a 1e-3 bar; what the bar is guarding is the
    SHAPE of the limit, and a remainder that failed to vanish would miss it by
    decades rather than by a factor.
    """
    points, total, magnetic = _composed("0110", eps_t=1e16)
    deck, mesh, solver, coeffs = _solved("0110")
    radius = min(piece.radius for piece in mesh.pieces)
    elements = solver.element_currents(coeffs, subdiv=_NEAR_FIELD_SUBDIV)
    mid, moment, nodes, delta = elements
    mid_img, moment_img = _image_moments(mid, moment, 0.0)
    nodes_img = nodes.copy()
    nodes_img[:, 2] = -nodes[:, 2]
    pec = _element_fields(
        points, elements, solver.k, radius, magnetic
    ) + _element_fields(
        points, (mid_img, moment_img, nodes_img, -delta), solver.k, radius, magnetic
    )
    assert np.abs(total - pec).max() <= 1e-3 * np.abs(pec).max()


def test_free_space_leaves_the_remainder_exactly_zero():
    """ε̃ = 1: the remainder is EXACTLY zero, and so is its curl.

    Not "small" — `_sommerfeld._six_integrals` short-circuits at ε̃ = 1
    because D₁ = D₂ = 0 identically there, and the analytic R₁ → 0 limits of
    eqs 169-172 carry the same factor, so every node of the interpolation grid
    is a hard zero.  A free-space "ground" that left ulp noise behind would be
    a ground the free-space path could not be compared against.
    """
    deck, mesh, solver, coeffs = _solved("0110")
    _, points = _request_points(deck)
    mid, moment, _, _ = solver.element_currents(coeffs, subdiv=_NEAR_FIELD_SUBDIV)
    electric = _field_point.reflected_field_at(
        points, mid, moment, 1.0, 0.0, solver.k, solver.omega
    )
    magnetic = _field_point.reflected_h_field_at(
        points, mid, moment, 1.0, 0.0, solver.k, solver.omega
    )
    assert np.count_nonzero(electric) == 0
    assert np.count_nonzero(magnetic) == 0


def test_the_point_dyad_and_the_fill_dyad_are_one_algebra():
    """The vector evaluator projected on an observer tangent IS
    `remainder_field_proj`.

    Same grid, same pairs, one source at a time so the fill's (M, S) table can
    be reproduced column by column — the fill sums nothing and the point
    evaluator sums over sources, which is the only structural difference
    between them.  The bar is 1e-10 relative; measured 3.9e-16, i.e. a couple
    of ulp, against the ACCELERATED fill path as well as the numpy one.

    This is #545's byte-stability statement in its testable form.  The fill
    keeps its bytes because nothing in it was touched; that the two spellings
    are one algebra is a gate rather than a refactor, and this is the gate.

    The DIAGONAL is included on purpose.  A self-pair has ρ = 0, where the
    incidence azimuth degenerates and the two spellings choose DIFFERENT
    fallback directions — the fill takes the source tangent's horizontal part,
    the point evaluator takes (1, 0), because a complex moment has no
    direction to take.  They agree anyway, which is the numerical statement of
    the identity I_ρ^H(90°) = −I_φ^H that licenses either choice.
    """
    deck, mesh, solver, coeffs = _solved("0113")
    mid, _, _, _ = solver.element_currents(coeffs, subdiv=1)
    tangents = np.concatenate(
        [
            np.repeat(
                ((piece.points[1] - piece.points[0]) / piece.length)[None, :],
                piece.n_elements,
                axis=0,
            )
            for piece in mesh.pieces
        ]
    )
    assert len(tangents) == len(mid)

    eps_t = _ground_spec.ground_config(solver, solver.omega).eps_tilde
    endpoints = np.concatenate([mid, mid])
    grid = _sommerfeld.get_grid(
        eps_t,
        solver.k,
        _sommerfeld.max_image_distance(endpoints, endpoints, 0.0),
        solver.omega,
    )
    fill = _sommerfeld.remainder_field_proj(
        mid, tangents, mid, tangents, 0.0, solver.k, grid
    )
    ours = np.stack(
        [
            np.sum(
                tangents
                * _field_point.reflected_field_at(
                    mid,
                    mid[j : j + 1],
                    tangents[j : j + 1],
                    eps_t,
                    0.0,
                    solver.k,
                    solver.omega,
                    grid=grid,
                ),
                axis=1,
            )
            for j in range(len(mid))
        ],
        axis=1,
    )
    assert np.abs(ours - fill).max() <= 1e-10 * np.abs(fill).max()


def test_the_finite_difference_curl_floor_is_measured_away_from_any_ground():
    """The magnetic evaluator's own error, pinned without a ground in sight.

    ``reflected_h_field_at`` differences an INTERPOLATED remainder, so its
    error has two parts and only one of them is the interpolation's.  This
    gate isolates the other: the same central differences at the same step,
    applied to `_element_fields`' ANALYTIC electric field, scored against
    `_element_fields`' analytic magnetic field at 0113's grid.  The identity
    is exact in the continuum — the mixed-potential scalar term is a true
    gradient of the regularized ``e^{−jkr}/r`` and contributes nothing to a
    curl, so what is left is ``∇×(−jωA) = −jωμ₀H`` — and every part per
    million between them is the differencing.

    Measured 1.06e-5 of the table's scale against the budget's predicted
    ``(k·step)²/6 ≈ 6.6e-6``: the same order, the excess being the near-zone
    terms' faster R-dependence, which is what confirms the budget's arithmetic
    rather than merely its conclusion.  The bar is that plus 25 % — two orders
    inside the 0.1 % the docstring's error budget claims.
    """
    deck, mesh, solver, coeffs = _solved("0113")
    _, points = _request_points(deck)
    radius = min(piece.radius for piece in mesh.pieces)
    elements = solver.element_currents(coeffs, subdiv=_NEAR_FIELD_SUBDIV)
    step = _field_point.CURL_STEP_WAVELENGTHS * (2.0 * math.pi / solver.k)

    fields = []
    for axis in range(3):
        for sign in (+1.0, -1.0):
            shift = np.zeros(3)
            shift[axis] = sign * step
            fields.append(
                _element_fields(points + shift, elements, solver.k, radius, False)
            )
    d_e = [(fields[2 * a] - fields[2 * a + 1]) / (2.0 * step) for a in range(3)]
    curl = np.stack(
        [
            d_e[1][:, 2] - d_e[2][:, 1],
            d_e[2][:, 0] - d_e[0][:, 2],
            d_e[0][:, 1] - d_e[1][:, 0],
        ],
        axis=-1,
    )
    got = curl / (-1j * solver.omega * _field_point._MU0)
    want = _element_fields(points, elements, solver.k, radius, True)
    assert np.abs(got - want).max() <= 1.33e-5 * np.abs(want).max()


# --------------------------------------------------------------------------
# gate 5 — the bare `GD` ground (momwire#545 U2)
# --------------------------------------------------------------------------


def test_the_gd_alias_as_a_plain_pec_image_breaks_its_envelope():
    """``direct + 1.0·image``, no medium, no remainder, must FAIL 0108's
    envelope — the near-field twin of the far-field anti-aliasing gate.

    The module docstring's own arithmetic is the shape of this bug: a `GD`
    near field composed as though the ground were `GN 1` prints half the
    field the medium actually has there (0107's origin cell 6.6673E+02
    against an image's 2.34E+02, 0110's `EZ` at (1, 0, 2) 1.5290E+01 against
    7.88E+00).  Composing 0108 the same wrong way reads 9.26 % of table scale
    and 30.7° against bars of 5.47 % and 1.54° — the magnitude misses by
    under 2x here (the grid's other cells dilute a factor-of-two singular
    cell down), but the PHASE misses by a factor of twenty, because forcing
    C₂ = 1 and dropping the remainder does not just scale the answer, it
    removes the entire smooth part NEC's decomposition (theory manual eqs
    136-147) puts there.

    If this test ever passes, `_composed`'s `Nec5MininecGround` branch has
    stopped reading the deck's medium somewhere and a `GD` near field is
    landing on the PEC image again.
    """
    magnitude_bar, phase_bar = NEAR_POINT_BAR["0108"]
    _, total, magnetic = _composed("0108", pec_alias=True)
    worst_magnitude, worst_phase = _score("0108", total, magnetic)
    assert worst_magnitude > magnitude_bar
    assert worst_phase > phase_bar


def test_the_gd_and_gn0_composed_tables_cross_agree_like_the_captures_do():
    """0108's composed table (`GD`, PEC-image currents, in-medium eval)
    against 0110's (`GN 0`, Sommerfeld currents, same eval) — the seam's own
    reproduction of the captures' 4.4 %-worst-cell agreement (module
    docstring, "One near field, two grounds that cannot have it"), not a
    trust that resemblance in the measured-table comment implies it.

    Both tables are read out at the SAME medium — 0108's `_composed` branch
    and 0110's take the identical ε̃ path once the PEC-image/Sommerfeld
    current split is behind them — so the two tables sit far closer to each
    other (see `CROSS_AGREEMENT_BAR` above) than either sits to its own
    capture.  `EY` is included at magnitude zero on both sides: the same
    vertical-monopole symmetry kills it identically regardless of which
    solve produced the currents underneath.

    The floor half of the gate is the aliasing check this unit's OTHER
    aliasing gate cannot see: composing 0108 with the wrong medium fails
    LOUD (the envelope gate above), but a bug that fed 0108 `_composed`
    0110's OWN currents — same medium, same evaluator, wrong solve — would
    pass the envelope gate by definition (the tables would be identical) and
    only this floor would notice.
    """
    _, total_gd, magnetic_gd = _composed("0108")
    _, total_gn0, magnetic_gn0 = _composed("0110")
    assert magnetic_gd == magnetic_gn0
    assert magnetic_gd is False

    want_magnitude, _, _ = _captured("0108")
    scale = want_magnitude.max()
    live = want_magnitude > 1e-4 * scale
    distance = np.abs(total_gd - total_gn0) / scale

    worst = 0.0
    floor = None
    for column in LIVE_COLUMNS["0108"]:
        selected = live[:, column]
        cell = distance[:, column][selected]
        worst = max(worst, float(cell.max()))
        floor = float(cell.min()) if floor is None else min(floor, float(cell.min()))
    assert worst <= CROSS_AGREEMENT_BAR, (
        f"worst live-cell distance {100 * worst:.4f} % of scale, "
        f"bar {100 * CROSS_AGREEMENT_BAR:.4f} %"
    )
    assert floor >= CROSS_AGREEMENT_FLOOR, (
        f"smallest live-cell distance {100 * floor:.6f} % of scale, "
        f"floor {100 * CROSS_AGREEMENT_FLOOR:.6f} % — the two routes have "
        "collapsed onto the same currents"
    )

    for column in DEAD_COLUMNS["0108"]:
        assert np.abs(total_gd[:, column]).max() == 0.0
        assert np.abs(total_gn0[:, column]).max() == 0.0


# --------------------------------------------------------------------------
# gate 6 — the regime registry
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "regime", [("below", "below"), ("below", "above"), ("above", "below")]
)
def test_the_three_buried_regimes_refuse_by_name(regime):
    """The regimes momwire#524 owns refuse, and say so in words.

    The keying is the module's DISPATCH SHAPE and not a docstring promise:
    a buried-wire arc lands by filling in three entries of `_REGIMES`, and
    every caller of `reflected_field_at` is already spelling which media it
    means.  A refusal that named neither the regime nor the work it belongs
    to would send the next reader to the wrong module.
    """
    source_medium, observer_medium = regime
    points = np.array([[1.0, 0.0, 2.0]])
    with pytest.raises(ValueError) as excinfo:
        _field_point.reflected_field_at(
            points,
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 1.0 + 0j]]),
            complex(13.0, -1.2),
            0.0,
            1.0,
            3e8,
            source_medium=source_medium,
            observer_medium=observer_medium,
        )
    message = str(excinfo.value)
    assert repr(regime) in message
    assert "momwire#524" in message
    assert "('above', 'above')" in message


def test_an_unknown_medium_name_is_not_silently_the_default():
    """A typo in a medium name is a ValueError and not the reflected wave."""
    with pytest.raises(ValueError, match="unknown ground regime"):
        _field_point.reflected_field_at(
            np.array([[1.0, 0.0, 2.0]]),
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 1.0 + 0j]]),
            complex(13.0, -1.2),
            0.0,
            1.0,
            3e8,
            source_medium="ABOVE",
        )
