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
  tolerance. Detached buried wires and fed buried wires are both this.
* refused — anything else. A wire with points on both sides, and a buried
  wire whose end stands IN the plane, are the same case: current crossing the
  interface, which needs the crossing basis momwire#524 phase 2 owns.

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


# The waiting gate for phase 2, banked from the momwire#524 phase-0 capture
# campaign. Quoted in the refusal so the sentence names a NUMBER the crossing
# basis has to meet rather than an open-ended "later".
CROSSING_ANCHOR = "74.761 - 57.730j ohm"

_REFUSE_CROSSING = (
    "wire {w} crosses the ground interface (polyline z runs {zmin:.6g} to "
    "{zmax:.6g} across ground_z = {gz:g}): momwire serves wires wholly at or "
    "above the interface and wires STRICTLY below it, and a wire with points "
    "on both sides - including a buried wire with an END standing in the "
    "plane - is neither. Current crossing the interface needs the crossing "
    "basis momwire#524 phase 2 owns: the two media meet along the wire, so "
    "the continuity of current there is a boundary condition, not a "
    "junction. The gate phase 2 has to meet is already banked - the phase-0 "
    "crossing anchor (a 2 m buried vertical joined at z = 0 to a 10 m "
    "monopole over eps_r 13 / sigma 0.005 S/m soil at 7 MHz) is "
    "{anchor} from our licensed NEC-5 engine's printout. Until then: leave "
    "the buried part DETACHED from the part above the plane (a buried radial "
    "screen under a base-fed vertical is served that way, momwire#553), or "
    "raise the whole wire clear of the interface"
)

_REFUSE_BURIED_PEC = (
    "wire {w} runs below the ground plane (min z = {zmin:.6g} < ground_z = "
    "{gz:g}) over a PERFECTLY CONDUCTING ground, which has no lower medium "
    "to put it in: the field inside a perfect conductor is identically zero, "
    "so a wire there is not buried, it is shorted out. A buried wire is "
    "served only under ground_model='sommerfeld' with a ground_eps, where "
    "the half-space below the interface is a real medium with a wavenumber "
    "k_m = k0*sqrt(eps_tilde) of its own (momwire#553). Raise the wire to or "
    "above the plane, or give the solve a Sommerfeld ground"
)

_REFUSE_BURIED_REFL = (
    "wire {w} runs below the ground plane (min z = {zmin:.6g} < ground_z = "
    "{gz:g}) under ground_model='refl-coef', which has no lower medium to "
    "put it in: the reflection-coefficient ground is a plane-wave boundary "
    "condition applied on the UPPER half-space alone - it multiplies an "
    "image by a Fresnel coefficient and never solves the field inside the "
    "ground at all, so there is nothing there for a wire to be buried in. A "
    "buried wire is served only under ground_model='sommerfeld' with a "
    "ground_eps, where the half-space below the interface is a real medium "
    "with a wavenumber k_m = k0*sqrt(eps_tilde) of its own (momwire#553). "
    "Raise the wire to or above the plane, or ask for the Sommerfeld ground"
)


# The one combination momwire#553 U5 measured itself OUT of, and the number
# that measured it. See `contact_with_buried_refusal`.
_REFUSE_CONTACT_WITH_BURIED = (
    "wire {cw} stands an END in the ground plane (ground CONTACT) and wire "
    "{bw} is buried below it: that COMBINATION is not served, though each "
    "half is. momwire's contact model continues the wire's current into the "
    "ground as the C2-scaled IMAGE (momwire#151), which is a fiction that "
    "works because no observer is ever inside the ground to look at it - and "
    "a buried wire is exactly such an observer. What it should see is the "
    "contact current SPREADING in the lower medium, which is the crossing "
    "physics momwire#524 phase 2 owns; what momwire can currently hand it is "
    "the transmitted field of the above-ground wire alone, with the contact "
    "node's charge unbalanced. That is not a small error and it was "
    "MEASURED: at eps_tilde = 1, where the whole buried fill must reproduce "
    "the free-space mixed-potential fill exactly, a 10 m contact monopole "
    "over a detached 5 m radial 15 cm down disagrees by 2.5 RELATIVE on the "
    "contact basis's cross-medium entries and by 1e-8 on every other basis; "
    "lift the same monopole 1 m clear of the plane and the whole fill agrees "
    "to 1.0e-5. Two banked gates are waiting for phase 2 to land this: our "
    "licensed NEC-5 engine prints 92.130 - 70.141j ohm for that lone-radial "
    "deck and 89.985 - 71.401j ohm for the four-radial fan, both at eps_r 13 "
    "/ sigma 0.005 S/m and 7 MHz. Until then: raise the above-ground wire "
    "clear of the interface (an elevated feed over a buried counterpoise is "
    "served), or solve the buried structure on its own"
)


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
    """The interface-crossing sentence for wire `w`."""
    return _REFUSE_CROSSING.format(
        w=w, zmin=zmin, zmax=zmax, gz=ground_z, anchor=CROSSING_ANCHOR
    )


def buried_no_medium_refusal(w, zmin, ground_z, *, pec):
    """The no-lower-medium sentence for wire `w`: PEC or refl-coef."""
    template = _REFUSE_BURIED_PEC if pec else _REFUSE_BURIED_REFL
    return template.format(w=w, zmin=zmin, gz=ground_z)


def wire_media(polylines, ground_z, *, lower_medium, pec):
    """One label per wire — `ABOVE` or `BELOW` — or a named `ValueError`.

    `lower_medium` is True exactly when the solve's ground carries a medium
    below the interface (`ground_model='sommerfeld'` with a `ground_eps`);
    `pec` picks which of the two no-lower-medium sentences a buried wire gets.

    Over free space (`ground_z is None`) every wire is `ABOVE`: z < 0 is legal
    geometry with no interface under it, and this layer must not invent one.
    """
    if ground_z is None:
        return tuple(ABOVE for _ in polylines)
    gz = float(ground_z)
    labels = []
    for w, pl in enumerate(polylines):
        pl_arr = np.asarray(pl, dtype=np.float64)
        tol = _ground_spec.ground_touch_tol(pl_arr)
        zmin = float(pl_arr[:, 2].min())
        zmax = float(pl_arr[:, 2].max())
        if zmin >= gz - tol:
            labels.append(ABOVE)
            continue
        if zmax >= gz - tol:
            raise ValueError(crossing_refusal(w, zmin, zmax, gz))
        if not lower_medium:
            raise ValueError(buried_no_medium_refusal(w, zmin, gz, pec=pec))
        labels.append(BELOW)
    labels = tuple(labels)
    if BELOW in labels:
        contacts = _ground_spec.contact_ends(polylines, gz)
        if contacts:
            raise ValueError(
                contact_with_buried_refusal(contacts[0][0], labels.index(BELOW))
            )
    return labels


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
