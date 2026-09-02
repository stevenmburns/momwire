"""Which MEDIUM each wire sits in — the per-segment medium label, and the two
geometries a buried wire is still refused for (momwire#553 U5).

`_ground_spec` answers "what ground is this"; this layer answers the question
that only exists once the ground below the interface is a real half-space with
a wavenumber of its own: **which side of the interface is this wire on**. It
is a pure geometry-plus-ground-decision question, asked identically by the
fill and by the EZNEC seam, so — like `_ground_spec.contact_ends` — it is
answered once here and quoted by both.

Three labels and nothing else
-----------------------------
* ``ABOVE`` — every polyline point at or above ``ground_z`` within the wire's
  own `_ground_spec.ground_touch_tol`. A wire END exactly IN the plane is
  ABOVE and is the ground-CONTACT case momwire#151 already serves; the tol is
  the solver's own so this layer and the contact tagging cannot disagree.
* ``BELOW`` — every polyline point STRICTLY below the plane by more than that
  tolerance. Detached buried wires and fed buried wires are both this. Since
  momwire#524 phase 2, so is a wholly-below wire whose plane-touching ANCHOR
  is a crossing-junction member (`crossing_ends`): its current reaches the
  interface through the crossing junction the solver owns, and the matching
  contact end above is exempt from the contact-with-buried refusal for the
  same reason.
* refused — anything else. A wire with points on both sides of the plane
  crosses it mid-span, and where it pierces would be momwire's guess: the
  served spelling is the SPLIT one, a below wire ending in the plane joined
  there to an above wire.

There is no per-SEGMENT crossing to worry about after that, which is the point
of labelling per WIRE: a wire that is wholly on one side has every segment on
that side, so `segment_media` is a broadcast rather than a second test with
its own tolerance to drift.

The medium a buried wire is buried IN
-------------------------------------
`BELOW` is only a label if there is a lower medium to label. Two of momwire's
four grounds do not have one at all, and the refusals below say WHICH rather
than saying "not served": a PEC plane has no interior, and the reflection-
coefficient ground is a boundary condition on the upper half-space that never
solves the lower one. Only ``ground_model='sommerfeld'`` with a `ground_eps`
carries ε̃ and therefore k_m, and that is the ground momwire#553 widens.
"""

from __future__ import annotations

import numpy as np

from . import _ground_spec

ABOVE = "above"
BELOW = "below"


# The engine's print for the phase-0 crossing deck (a 2 m buried vertical
# joined at z = 0 to a 10 m monopole over eps_r 13 / sigma 0.005 S/m soil at
# 7 MHz). Once the waiting gate for phase 2; ADJUDICATED 2026-08-26 as a
# DIFFERENT EXPERIMENT, not a target: the engine's junction there is two
# independent contact ends plus a point-electrode sink (its own printed
# junction currents violate its AGARD condition divergently, with a KCL
# deficit of ~2 A vanishing into the interface point), while the crossing
# serve's exact-EM answer for the same deck is 138.77 - 102.99j ohm, with
# continuity and the AGARD slope emerging from the fill. The two conventions
# collapse onto each other exactly where the contact fiction becomes
# physical (sigma -> inf) and only there. Kept for the record and for the
# convention-difference documentation; NEVER gate the crossing serve
# against it (the house rule: never gate cross-formulation agreement).
ENGINE_CROSSING_PRINT = "74.761 - 57.730j ohm"

# Each of the four geometry refusals below is a per-deck PREAMBLE naming the
# offending wire and the numbers that made it offending, followed by a CONSTANT
# reason. The split is what lets a `Capabilities` row declare the reason
# (momwire#792): a matrix cell is written before any deck exists, so it
# can carry the second half and not the first, and `tests/
# test_refusals_are_declared.py` gates every raise here as
# preamble-plus-declared-reason. Before the split these were declared by
# filling the placeholders with `("<wire>", 0.0, 0.0, 0.0)`, which read as
# prose in an exception and as noise in a generated document.

CROSSING_REFUSAL = (
    "momwire serves "
    "wires wholly at or above the interface, wires strictly below it, and "
    "current CROSSING it only through a crossing junction (momwire#524 "
    "phase 2): split the wire AT the interface into a below wire whose end "
    "stands in the plane and an above wire starting there, and declare the "
    "junction between them - that deck is served, with continuity of "
    "current and the interface slope condition emerging from the fill "
    "itself. A single polyline with points on both sides is not, because "
    "which point the wire pierces the plane at would be momwire's guess "
    "where it must be the model's statement. Alternatively leave the "
    "buried part DETACHED (a buried radial screen under a base-fed "
    "vertical is served that way, momwire#553), or raise the whole wire "
    "clear of the interface"
)

_REFUSE_CROSSING = (
    "wire {w} crosses the ground interface mid-span (polyline z runs "
    "{zmin:.6g} to {zmax:.6g} across ground_z = {gz:g}): "
) + CROSSING_REFUSAL

BURIED_PEC_REFUSAL = (
    "over a PERFECTLY CONDUCTING ground, which has no lower medium "
    "to put it in: the field inside a perfect conductor is identically zero, "
    "so a wire there is not buried, it is shorted out. A buried wire is "
    "served only under ground_model='sommerfeld' with a ground_eps, where "
    "the half-space below the interface is a real medium with a wavenumber "
    "k_m = k0*sqrt(eps_tilde) of its own (momwire#553). Raise the wire to or "
    "above the plane, or give the solve a Sommerfeld ground"
)

_REFUSE_BURIED_PEC = (
    "wire {w} runs below the ground plane (min z = {zmin:.6g} < ground_z = {gz:g}) "
) + BURIED_PEC_REFUSAL

BURIED_REFL_REFUSAL = (
    "under ground_model='refl-coef', which has no lower medium to "
    "put it in: the reflection-coefficient ground is a plane-wave boundary "
    "condition applied on the UPPER half-space alone - it multiplies an "
    "image by a Fresnel coefficient and never solves the field inside the "
    "ground at all, so there is nothing there for a wire to be buried in. A "
    "buried wire is served only under ground_model='sommerfeld' with a "
    "ground_eps, where the half-space below the interface is a real medium "
    "with a wavenumber k_m = k0*sqrt(eps_tilde) of its own (momwire#553). "
    "Raise the wire to or above the plane, or ask for the Sommerfeld ground"
)

_REFUSE_BURIED_REFL = (
    "wire {w} runs below the ground plane (min z = {zmin:.6g} < ground_z = {gz:g}) "
) + BURIED_REFL_REFUSAL


# The one combination momwire#553 U5 measured itself OUT of, and the number
# that measured it. See `contact_with_buried_refusal`.
CONTACT_WITH_BURIED_REFUSAL = (
    "that COMBINATION is not served, though each "
    "half is. momwire's contact model continues the wire's current into the "
    "ground as the C2-scaled IMAGE (momwire#151), which is a fiction that "
    "works because no observer is ever inside the ground to look at it - and "
    "a buried wire is exactly such an observer. What it should see is the "
    "contact current SPREADING in the lower medium - a real soil current "
    "this deck has NO CONDUCTOR for. momwire#524 phase 2 measured every "
    "consistent spelling of the missing cross-medium physics against our "
    "licensed NEC-5 engine's prints for exactly these decks (92.130 - "
    "70.141j ohm for a 10 m contact monopole over one detached 5 m radial "
    "15 cm down, 90.051 - 70.731j ohm for the four-radial fan, eps_r 13 / "
    "sigma 0.005 S/m, 7 MHz): at matched feed and converged meshes on BOTH "
    "sides the continuation-consistent spelling still lands ~3 ohm (lone) / "
    "~6 ohm (fan) away (momwire#567 re-derivation, 2026-08-28) and the "
    "residual is the spreading current itself, "
    "which the engine carries as a point-electrode stake - the same "
    "fiction the phase-2 adjudication measured violating its own junction "
    "physics. There is no honest sub-ohm serve of this deck class in "
    "either convention, so it stays refused rather than answered wrong. "
    "Serve it by giving the spreading current its conductor: respell the "
    "radial - or the whole SCREEN of them - to RISE to the surface and "
    "junction-join the monopole at z = 0; that crossing junction is served "
    "for one above wire over N below wires (momwire#524 phase 2, fan "
    "widening), including the screen's buried-hub spelling (one rise, N "
    "radials joined at depth). Or raise the above-ground wire clear of "
    "the interface (an elevated feed over a buried counterpoise is "
    "served), or solve the buried structure on its own"
)

_REFUSE_CONTACT_WITH_BURIED = (
    "wire {cw} stands an END in the ground plane (ground CONTACT) and wire "
    "{bw} is buried below it: "
) + CONTACT_WITH_BURIED_REFUSAL


def contact_with_buried_refusal(contact_wire, buried_wire):
    """The ground-contact-plus-buried sentence.

    Refusing this is momwire#553 U5's own scope decision and it costs the
    unit its two headline decks, so the reasoning is worth stating once:

    The buried fill mixes two testing conventions, because momwire always
    has. The direct and image blocks are MIXED-POTENTIAL — the
    integrate-by-parts rewrite of ⟨f, E⟩ — and the Sommerfeld remainder
    block is FIELD-form. The rewrite drops a boundary term `[f·Φ]` at each
    end of a basis's support, and for every basis that vanishes at its own
    ends (which is every basis except a junction's or a ground contact's)
    that term is zero and the two conventions agree exactly. At a ground
    CONTACT the basis has value 1 in the plane and the term is not zero; the
    shipped path gets away with it because its field-form block is a small
    REMAINDER, so a boundary term on a small block is a small error.

    That is the fifth of momwire#553's inversions and it is the same shape as
    the other four: **a ±=+ convenience whose licence is "the remainder is
    small", used where the field-form block is not a remainder at all.** The
    cross-medium block is the WHOLE interaction between the two media, so its
    boundary term is O(1) of the block, and the measurement in the refusal
    text says so.

    Fixing it needs the transmitted family's scalar POTENTIALS, so that the
    cross block can be written mixed-potential like its neighbours — or the
    crossing basis, which removes the contact fiction altogether. Both are
    momwire#524 phase 2 / a recorded follow-up, and neither is this unit.
    """
    return _REFUSE_CONTACT_WITH_BURIED.format(cw=contact_wire, bw=buried_wire)


def crossing_refusal(w, zmin, zmax, ground_z):
    """The mid-span interface-crossing sentence for wire `w`."""
    return _REFUSE_CROSSING.format(w=w, zmin=zmin, zmax=zmax, gz=ground_z)


def buried_no_medium_refusal(w, zmin, ground_z, *, pec):
    """The no-lower-medium sentence for wire `w`: PEC or refl-coef."""
    template = _REFUSE_BURIED_PEC if pec else _REFUSE_BURIED_REFL
    return template.format(w=w, zmin=zmin, gz=ground_z)


_REFUSE_BURIED_FAR_FIELD = (
    "RP asks for the far field of a deck with a wire below the ground plane, "
    "and a buried deck's radiation pattern is not served. The pattern of a "
    "buried source is the transmitted field's FAR-ZONE asymptotics - a "
    "saddle-point evaluation of the same integral, with its own lateral-wave "
    "and critical-angle structure - and momwire#553 built the transmitted "
    "family over a NEAR-zone tabulation only (2 free-space wavelengths of "
    "range), so there is nothing here to take a limit of (momwire#570). "
    "This deck's IMPEDANCE, its CURRENTS and its CHARGES are all served: "
    "drop the RP "
    "card, or lift the wire above z = 0"
)


def buried_far_field_refusal():
    """The buried-radiation-pattern sentence — read by BOTH front ends.

    It lives here rather than in either dialect because the obstruction is
    the PHYSICS's and not the printout's: momwire#553 tabulated the
    transmitted family over a near-zone range only, so neither seam has a
    far-zone limit to take and neither can acquire one without the other.
    Two copies of that sentence would be two sentences to keep equal, which
    is the defect momwire#567 found in a pair of banked impedances.

    The NEC-5 seam has refused this since momwire#553; the NEC-2 portal
    SERVED it — summing the upper medium's wavenumber with a
    Fresnel-reflected image, so a buried element contributed as though it
    radiated in air from a point underground.  That printed a plausible
    pattern rather than failing: -20.55 dB at theta 60 and -25.72 dB at 75
    for a 5 m wire 0.5 m under a ``GN 2`` interface at 14 MHz, only the
    grazing sample floored to -999.99, and a POWER BUDGET claiming 100 %
    efficiency for a wire dissipating in soil.  momwire#570 is where the
    real answer is tracked.
    """
    return _REFUSE_BURIED_FAR_FIELD


def wire_media(polylines, ground_z, *, lower_medium, pec, crossing_ends=()):
    """One label per wire — `ABOVE` or `BELOW` — or a named `ValueError`.

    `lower_medium` is True exactly when the solve's ground carries a medium
    below the interface (`ground_model='sommerfeld'` with a `ground_eps`);
    `pec` picks which of the two no-lower-medium sentences a buried wire gets.

    `crossing_ends` is the crossing-junction exemption (momwire#524 phase 2):
    the `(wire, "start"|"end")` pairs that participate in a junction whose
    shared point lies IN the ground plane. A wholly-below wire whose
    plane-touching anchor is such a junction member is `BELOW` — its current
    reaches the interface through the crossing junction the caller owns —
    and a contact end that is such a member is exempt from the
    contact-with-buried refusal for the same reason. A caller with no
    crossing basis (razor) passes nothing and keeps the refusals verbatim.

    Over free space (`ground_z is None`) every wire is `ABOVE`: z < 0 is legal
    geometry with no interface under it, and this layer must not invent one.
    """
    if ground_z is None:
        return tuple(ABOVE for _ in polylines)
    gz = float(ground_z)
    crossing_ends = frozenset(crossing_ends)
    labels = []
    for w, pl in enumerate(polylines):
        pl_arr = np.asarray(pl, dtype=np.float64)
        tol = _ground_spec.ground_touch_tol(pl_arr)
        zmin = float(pl_arr[:, 2].min())
        zmax = float(pl_arr[:, 2].max())
        if zmin >= gz - tol:
            labels.append(ABOVE)
            continue
        if zmax >= gz - tol and not _is_crossing_below(
            pl_arr, w, gz, tol, crossing_ends
        ):
            raise ValueError(crossing_refusal(w, zmin, zmax, gz))
        if not lower_medium:
            raise ValueError(buried_no_medium_refusal(w, zmin, gz, pec=pec))
        labels.append(BELOW)
    labels = tuple(labels)
    if BELOW in labels:
        contacts = [
            c
            for c in _ground_spec.contact_ends(polylines, gz)
            if c not in crossing_ends
        ]
        if contacts:
            raise ValueError(
                contact_with_buried_refusal(contacts[0][0], labels.index(BELOW))
            )
    return labels


def _is_crossing_below(pl_arr, w, gz, tol, crossing_ends):
    """Whether wire `w` is a legitimate crossing-junction BELOW wire: it
    touches the plane ONLY at anchor(s), nothing pokes above, and every
    touching anchor is an exempted junction member."""
    z = pl_arr[:, 2]
    if float(z.max()) > gz + tol:
        return False
    touch_start = abs(float(z[0]) - gz) <= tol
    touch_end = abs(float(z[-1]) - gz) <= tol
    if not (touch_start or touch_end):
        return False
    interior = z[1:-1]
    if interior.size and float(interior.max()) >= gz - tol:
        return False
    if touch_start and (w, "start") not in crossing_ends:
        return False
    if touch_end and (w, "end") not in crossing_ends:
        return False
    return True


def segment_media(labels, seg_offsets):
    """`(n_segs,)` boolean "this segment is BELOW the interface".

    A broadcast of the per-wire labels over `geom["seg_offsets"]` — see the
    module docstring for why there is no per-segment test.
    """
    n = int(seg_offsets[-1])
    below = np.zeros(n, dtype=bool)
    for w, label in enumerate(labels):
        if label == BELOW:
            below[seg_offsets[w] : seg_offsets[w + 1]] = True
    return below
