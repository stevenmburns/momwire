"""Below-plane wires at the seam: what LANDED, and what each of the three
remaining refusals now names.

momwire#525 gave these decks a refusal grammar instead of an ``INTERNAL
ERROR`` frame; momwire#553 lands the CAPABILITY behind most of that grammar.
A wire strictly below a ``GN 0`` / ``GN 2`` interface is now served — the
solver labels it with the lower medium and fills its pairs through the two
buried Sommerfeld families — so the old single sentence ("buried wires are
not served") is gone and what replaced it is four narrower ones, each naming
a DIFFERENT missing thing:

* a wire with points on BOTH sides of the interface — the crossing basis,
  momwire#524 phase 2, with its banked anchor quoted;
* a buried wire over ``GN 1`` or a bare ``GD`` — neither card has a lower
  medium at all, and the sentence says which;
* a buried wire on a deck that ALSO stands a wire end in the plane — the
  combination momwire#553 U5 measured itself out of, with both phase-0
  anchors quoted as the gates phase 2 has to meet;
* the OUTPUTS a buried deck cannot answer: its near field (phase 3) and its
  far field (the transmitted far-zone follow-up). Impedance, currents and
  charges serve — that is the serve matrix, and these two refusals are its
  other half.

Free space stays exempt on all of them: z < 0 is legal geometry with no
interface under it.
"""

import pytest

from momwire.deck._nec5 import parse_nec5
from momwire.eznec._serve import refusal
from momwire.eznec._shell import render

# The anchors, their envelope and the printed-Z reader all live somewhere
# else already; two copies of 92.130 - 70.141j would be two numbers to keep
# equal, which is the defect momwire#567 found in the first place.
from test_buried_serve_553 import (
    ANCHOR_ENVELOPE_OHM,
    ANCHOR_FOUR_RADIAL,
    ANCHOR_LONE_RADIAL,
)
from test_eznec_drive_spelling import input_impedance

GN0 = "GN 0,0,0,0,13.,.005"


def deck(radial_z, ge, ground, extra="", requests="PQ 0\nXQ 0\n", mono_bottom="0."):
    return (
        "CM buried probe\n"
        "CE\n"
        f"GW 1,15,0.,0.,10.,0.,0.,{mono_bottom},.001\n"
        f"GW 2,10,0.,0.,{radial_z},5.,0.,{radial_z},.001\n"
        f"{extra}"
        f"GE {ge}\n"
        "FR 0,1,0,0,7.\n"
        f"{ground}\n"
        "EX 4,1,7,0,1.,0.\n"
        f"{requests}"
        "EN\n"
    )


def reason(text):
    out = render(text)
    assert " ***** NEC ERROR - " in out
    return out.split(" ***** NEC ERROR - ")[1]


def why(text):
    """The seam's own refusal string, without paying for a solve."""
    return refusal(parse_nec5(text))


# ----------------------------------------------------------------------
# what LANDED
# ----------------------------------------------------------------------


def test_a_buried_wire_under_gn0_is_no_longer_refused_for_being_buried():
    """The rung landed: an ELEVATED feed over a buried counterpoise carries
    no refusal at all. (The solve itself is gated in
    ``tests/test_buried_serve_553.py``; what is pinned here is that the seam
    stopped saying no.)"""
    text = deck(-0.15, "1,-1", GN0, mono_bottom="1.")
    assert why(text) is None


def test_the_old_sentence_is_gone():
    text = deck(-0.15, "1,-1", GN0, mono_bottom="1.")
    out = render(text)
    assert "buried wires are not served" not in out


# ----------------------------------------------------------------------
# the crossing wire
# ----------------------------------------------------------------------


def test_a_crossing_wire_refuses_naming_phase_two_and_its_anchor():
    text = (
        "CM crossing probe\n"
        "CE\n"
        "GW 1,4,0.,0.,-2.,0.,0.,0.,.001\n"
        "GW 2,15,0.,0.,0.,0.,0.,10.,.001\n"
        "GE 1,-1\n"
        "FR 0,1,0,0,7.\n"
        f"{GN0}\n"
        "EX 4,2,7,0,1.,0.\n"
        "PQ 0\nXQ 0\nEN\n"
    )
    r = why(text)
    assert r is not None
    assert "crosses the ground interface" in r
    assert "phase 2" in r
    assert "74.761 - 57.730j" in r
    assert "INTERNAL ERROR" not in r


# ----------------------------------------------------------------------
# no lower medium
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "ground,card,needle",
    [
        ("GN 1", "GN 1", "perfect conductor"),
        ("GD 0,0,0,0,13.,.005", "GD", "PERFECT image"),
    ],
)
def test_a_buried_wire_over_a_ground_with_no_lower_medium_refuses(ground, card, needle):
    r = why(deck(-0.15, "1,-1", ground, mono_bottom="1."))
    assert r is not None
    assert "below the ground plane" in r
    assert f"under a {card} card" in r
    assert needle in r
    assert "GN 0 / GN 2" in r


# ----------------------------------------------------------------------
# the combination momwire#553 U5 measured itself out of
# ----------------------------------------------------------------------


def test_a_contact_wire_plus_a_buried_wire_refuses_with_both_anchors():
    r = why(deck(-0.15, "1,-1", GN0))
    assert r is not None
    assert "stands an END in the ground plane" in r
    assert "92.130 - 70.141j" in r
    assert "90.051 - 70.731j" in r
    assert "phase 2" in r
    assert "elevated feed over a buried counterpoise is served" in r


def test_ground_contact_alone_still_serves():
    contact = (
        "CM contact control\n"
        "CE\n"
        "GW 1,15,0.,0.,10.,0.,0.,0.,.001\n"
        "GE 1,-1\n"
        "FR 0,1,0,0,7.\n"
        f"{GN0}\n"
        "EX 4,1,7,0,1.,0.\n"
        "PQ 0\nXQ 0\nEN\n"
    )
    assert "NEC ERROR" not in render(contact)


# ----------------------------------------------------------------------
# the serve matrix: impedance / currents / charges, and nothing else
# ----------------------------------------------------------------------


def test_a_buried_decks_near_field_refuses_naming_phase_three():
    text = deck(
        -0.15,
        "1,-1",
        GN0,
        requests="PQ 0\nNE 0,1,1,1,2.,0.,1.,0.,0.,0.\nXQ 0\n",
        mono_bottom="1.",
    )
    r = why(text)
    assert r is not None
    assert "buried deck's near field is not served" in r
    assert "phase 3" in r
    assert "IMPEDANCE, its CURRENTS and its CHARGES are all served" in r


def test_a_buried_decks_far_field_refuses_naming_the_far_zone_followup():
    text = deck(
        -0.15,
        "1,-1",
        GN0,
        requests="PQ 0\nRP 0,19,1,1000,0.,0.,5.,0.\n",
        mono_bottom="1.",
    )
    r = why(text)
    assert r is not None
    assert "radiation pattern is not served" in r
    assert "FAR-ZONE asymptotics" in r
    assert "IMPEDANCE, its CURRENTS and its CHARGES are all served" in r


def test_an_above_ground_deck_still_gets_its_near_and_far_fields():
    """The two output refusals key on the DECK having a buried wire, not on
    the ground card, so an all-above deck over the same ground is untouched."""
    text = deck(
        0.5,
        "1,-1",
        GN0,
        requests="PQ 0\nNE 0,1,1,1,2.,0.,1.,0.,0.,0.\nRP 0,19,1,1000,0.,0.,5.,0.\n",
    )
    assert why(text) is None


# ----------------------------------------------------------------------
# unchanged
# ----------------------------------------------------------------------


def test_in_plane_wire_refuses_by_name_not_internal_error():
    r = reason(deck(0.0, "1,-1", GN0))
    assert "wire 2" in r
    assert "in the ground plane" in r
    assert "INTERNAL ERROR" not in r


def test_free_space_serves_the_same_wires():
    out = render(deck(-0.15, "0,-1", "GN -1"))
    assert "NEC ERROR" not in out


def test_a_slight_standoff_above_the_plane_serves():
    """1 cm above ground is a legal elevated radial, not a refusal."""
    out = render(deck(0.01, "1,-1", GN0))
    assert "NEC ERROR" not in out


# ----------------------------------------------------------------------
# the two banked anchors, through the DECK route
# ----------------------------------------------------------------------
#
# `test_buried_serve_553`'s G-U5-12 gates the same two decks at the solver
# API. This is the route a user actually takes, and it can drift from the
# solver's on its own — a wrong feed segment or a mis-mapped GN card would
# leave the solver gate green. The constants are imported rather than
# restated so there is one place to move them.


def fan_deck_text():
    """The four-radial anchor as cards. The radials share (0, 0, -.15), so
    the deck route joins them itself and no junction is declared here."""
    radials = "".join(
        f"GW {i + 2},10,0.,0.,-.15,{5.0 * dx:g},{5.0 * dy:g},-.15,.001\n"
        for i, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1)))
    )
    return (
        "CM four-radial anchor\n"
        "CE\n"
        "GW 1,15,0.,0.,10.,0.,0.,0.,.001\n"
        f"{radials}"
        "GE 1,-1\n"
        "FR 0,1,0,0,7.\n"
        f"{GN0}\n"
        "EX 4,1,7,0,1.,0.\n"
        "PQ 0\nXQ 0\nEN\n"
    )


@pytest.mark.parametrize(
    "text,anchor",
    [
        (lambda: deck(-0.15, "1,-1", GN0), ANCHOR_LONE_RADIAL),
        (fan_deck_text, ANCHOR_FOUR_RADIAL),
    ],
    ids=["lone-radial", "four-radial"],
)
def test_the_anchor_deck_answers_what_the_engine_printed(text, anchor):
    """Self-arming, the same shape G-U5-12 has: today the deck route prints
    a ``NEC ERROR`` and there is no impedance row to read, so the gate
    checks the refusal is still the contact+buried one and xfails. The day
    the refusal lifts, ``ANTENNA INPUT PARAMETERS`` appears and the gate
    scores it — with no edit here."""
    out = render(text())
    z = input_impedance(out)
    if z is None:
        assert "stands an END in the ground plane" in out
        pytest.xfail("the deck route refuses this anchor today — momwire#567")
    assert abs(z - anchor) <= ANCHOR_ENVELOPE_OHM, (
        f"the deck route answers {z:.4f} where the engine prints "
        f"{anchor:.4f} — {abs(z - anchor):.4f} ohm apart"
    )
