"""The ``nec2`` deck dialect, section by section.

Every test names the section of the normative spec it measures:
``site/src/content/docs/reference/deck-grammar-nec2.md``, whose headings are
stable citation targets.  A refusal is asserted VERBATIM — the page publishes
the exact text, and a caller frames it — so the page cannot drift from the
code without one of these going red.
"""

from __future__ import annotations

import math

import pytest

from momwire.deck import DeckError, parse, parse_card, tokenize
from momwire.deck._nec2 import (
    _ARMING_CARDS,
    _GEOMETRY_CARDS,
    _OPERATOR_CARDS,
    _REFUSED_BY_NAME,
)
from momwire.deck._nec2_geometry import build_geometry
from momwire.deck.model import (
    Environment,
    FarFieldRequest,
    LoadSpec,
    NearFieldRequest,
    SecondMedium,
)

# A minimal runnable deck: one wire, driven, so `parse` gets past the
# structural "nothing drives the structure" refusal.
DIPOLE = """CM a dipole
CE
GW 1 5 -0.5 0. 0. 0.5 0. 0. 1.E-3
GE 0
EX 0 1 3 0 1. 0.
FR 0 1 0 0 14.
XQ
NX
"""


def geometry(text: str):
    """The deck's geometry section, built.  Cards that are not geometry are
    dropped, so a section can be measured without a whole runnable deck."""
    return build_geometry([c for c in tokenize(text) if c.mnemonic in _GEOMETRY_CARDS])


# ---------------------------------------------------------------------------
# #deck-framing and #field-numbering
# ---------------------------------------------------------------------------


def test_cards_are_free_format():
    """§#deck-framing: spaces and/or commas separate fields."""
    spaced = parse_card("GW 1 5 -0.5 0. 0. 0.5 0. 0. 1.E-3")
    commas = parse_card("GW,1,5,-0.5,0.,0.,0.5,0.,0.,1.E-3")
    assert spaced is not None and commas is not None
    assert spaced.values == commas.values


def test_a_fused_mnemonic_splits():
    """§#deck-framing: ``GE1`` and ``GD 2,0,...`` split correctly."""
    assert parse_card("GE1").mnemonic == "GE"
    assert parse_card("GE1").i(0) == 1
    assert parse_card("GD2,0,0,0").values == (2.0, 0.0, 0.0, 0.0)
    # A third LETTER is not a fused field — it is a mnemonic error.
    with pytest.raises(DeckError):
        parse_card("GWX 1 5")


def test_fortran_d_exponents_are_numbers():
    """§#deck-framing: numeric fields accept ``1.0D-3``."""
    assert parse_card("GW 1 5 0 0 0 1 0 0 1.0D-3").f(8) == 1.0e-3
    assert parse_card("GW 1 5 0 0 0 1 0 0 1.0d-3").f(8) == 1.0e-3


def test_blank_lines_are_skipped():
    """§#deck-framing."""
    assert parse_card("   ") is None
    assert parse_card("") is None
    assert len(tokenize("GW 1 5 0 0 0 1 0 0 1E-3\n\n\nGE 0\n")) == 2


def test_mnemonic_and_field_errors_are_verbatim():
    """§#deck-framing: the two mnemonic-level errors, raised before any card
    is interpreted."""
    with pytest.raises(DeckError) as exc:
        parse_card("X 1 2")
    assert str(exc.value) == "CARD'S MNEMONIC CODE TOO SHORT OR MISSING: 'X 1 2'"
    with pytest.raises(DeckError) as exc:
        parse_card("GW 1 five 0 0 0 1 0 0 1E-3")
    assert str(exc.value) == (
        "NON-NUMERICAL CHARACTER IN FIELD: 'five' on 'GW 1 five 0 0 0 1 0 0 1E-3'"
    )


def test_an_integer_field_written_as_a_real_reads_as_an_integer():
    """§#field-numbering: fields are read positionally and converted on
    demand, so ``1.`` reads as ``1``."""
    assert parse_card("GW 1. 5. 0 0 0 1 0 0 1E-3").i(1) == 5


def test_a_field_the_page_does_not_name_is_read_and_discarded():
    """§#field-numbering: it never changes an answer, and the parser never
    refuses a deck because of one — here ``GE``'s ``I2`` and reals."""
    flagged = geometry("GW 1 5 0 0 0 1 0 0 1E-3\nGE 1 7 9. 9. 9.\n")
    assert flagged.ground_plane_flag is True
    assert len(flagged.wires) == 1


def test_an_absent_field_reads_as_zero():
    """§#field-numbering: a card's zero-filled image."""
    card = parse_card("GE")
    assert card.i(0) == 0 and card.f(5) == 0.0


def test_comment_text_and_its_two_directives():
    """§#cm--ce--comments: free text in card order, ``QQ n`` and ``FF n``
    consumed by the parser rather than passed through."""
    model = parse("CM hello QQ 1\nCE and FF 3\n" + DIPOLE)
    assert model.comments[:2] == (" hello QQ 1", " and FF 3")
    assert model.quiet is True
    assert model.reduced_field == 3


def test_nx_ends_the_deck():
    """§#nx--end-of-deck: everything after ``NX`` is a new deck."""
    model = parse(DIPOLE + "GW 9 5 0 0 0 1 0 0 1E-3\nGE 0\n")
    assert len(model.wires) == 1


def test_en_terminates_a_deck_too():
    """§#en--end-of-run."""
    assert len(parse(DIPOLE.replace("NX", "EN")).wires) == 1


# ---------------------------------------------------------------------------
# #gw--straight-wire
# ---------------------------------------------------------------------------


def test_gw_becomes_a_two_vertex_polyline():
    """§#gw--straight-wire: two vertices, ``NS`` elements between them, and a
    radius carried per wire with no averaging across the structure."""
    model = parse(
        "GW 1 5 0. 0. 0. 1. 0. 0. 1.E-3\n"
        "GW 2 3 0. 1. 0. 1. 1. 0. 5.E-3\n"
        "GE 0\nEX 0 1 3 0 1.\nXQ\nNX\n"
    )
    assert [w.vertices for w in model.wires] == [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 1.0, 0.0), (1.0, 1.0, 0.0)),
    ]
    assert [w.edge_elements for w in model.wires] == [(5,), (3,)]
    assert [w.radius for w in model.wires] == [1.0e-3, 5.0e-3]


@pytest.mark.parametrize("n_seg", ["0", "-2"])
def test_gw_refuses_a_segment_count_below_one(n_seg):
    """§#gw--straight-wire."""
    with pytest.raises(DeckError) as exc:
        geometry(f"GW 1 {n_seg} 0 0 0 1 0 0 1E-3\nGE 0\n")
    assert str(exc.value) == f"segment count must be >= 1, got {int(n_seg)}"


@pytest.mark.parametrize("radius", ["0.", "-1E-3"])
def test_gw_refuses_a_non_positive_radius(radius):
    """§#gw--straight-wire: NEC's announcement of a tapered wire continued by
    a ``GC`` card, which is out of dialect."""
    with pytest.raises(DeckError) as exc:
        geometry(f"GW 1 5 0 0 0 1 0 0 {radius}\nGE 0\n")
    assert str(exc.value) == (
        "GW with a non-positive radius announces a tapered wire "
        "(GW + GC continuation), which is not part of this engine's nec2 dialect"
    )


# ---------------------------------------------------------------------------
# #gm--move-and-replicate
# ---------------------------------------------------------------------------


def test_gm_with_nrpt_zero_transforms_the_block_in_place():
    """§#gm--move-and-replicate: rotation about X, then Y, then Z, applied
    before the translation."""
    built = geometry(
        "GW 1 2 0. 1. 0. 0. 2. 0. 1.E-3\nGM 0 0 90. 0. 0. 0. 0. 0. 0\nGE 0\n"
    )
    (wire,) = built.wires
    assert wire.p1 == pytest.approx([0.0, 0.0, 1.0], abs=1e-15)
    assert wire.p2 == pytest.approx([0.0, 0.0, 2.0], abs=1e-15)
    # Translation is applied after the rotation, not rotated with it.
    shifted = geometry(
        "GW 1 2 0. 1. 0. 0. 2. 0. 1.E-3\nGM 0 0 90. 0. 0. 5. 0. 0. 0\nGE 0\n"
    )
    assert shifted.wires[0].p1 == pytest.approx([5.0, 0.0, 1.0], abs=1e-15)


def test_gm_repetitions_compound_and_increment_the_tag():
    """§#gm--move-and-replicate: each repetition transforms the PREVIOUS
    copy, so one loop side replicates into a polygon."""
    built = geometry(
        "GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3\nGM 1 3 0. 0. 90. 1. 0. 0. 0\nGE 0\n"
    )
    assert [w.tag for w in built.wires] == [1, 2, 3, 4]
    assert len(built.wires) == 4
    # A square: the last replica's far end is back at the origin.
    assert built.wires[-1].p2 == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_gm_keeps_tag_zero_at_zero():
    """§#gm--move-and-replicate: a wire with tag 0 keeps tag 0 in every
    replica; every other tag gains ``I1`` per repetition."""
    built = geometry(
        "GW 0 2 0. 0. 0. 1. 0. 0. 1.E-3\n"
        "GW 7 2 0. 1. 0. 1. 1. 0. 1.E-3\n"
        "GM 5 2 0. 0. 0. 0. 0. 1. 0\nGE 0\n"
    )
    assert [w.tag for w in built.wires] == [0, 7, 0, 12, 0, 17]


def test_gm_its_selects_the_block_and_is_rounded():
    """§#gm--move-and-replicate: the block is every wire from the first one
    carrying tag ``ITS`` to the end; ``ITS`` is read as a real and rounded."""
    built = geometry(
        "GW 1 2 0. 0. 0. 1. 0. 0. 1.E-3\n"
        "GW 2 2 0. 1. 0. 1. 1. 0. 1.E-3\n"
        "GM 0 0 0. 0. 0. 0. 0. 3. 1.6\nGE 0\n"
    )
    assert built.wires[0].p1[2] == 0.0
    assert built.wires[1].p1[2] == 3.0


def test_gm_refuses_an_unknown_tag():
    """§#gm--move-and-replicate."""
    with pytest.raises(DeckError) as exc:
        geometry("GW 1 2 0 0 0 1 0 0 1E-3\nGM 0 0 0 0 0 0 0 0 9\nGE 0\n")
    assert str(exc.value) == "no wire has tag 9"


# ---------------------------------------------------------------------------
# #gs--scale
# ---------------------------------------------------------------------------


def test_gs_scales_endpoints_and_radii_but_not_segment_counts():
    """§#gs--scale."""
    built = geometry("GW 1 5 0. 0. 0. 2. 0. 0. 1.E-3\nGS 0 0 3.\nGE 0\n")
    (wire,) = built.wires
    assert wire.p2 == [6.0, 0.0, 0.0]
    assert wire.radius == 3.0e-3
    assert wire.n_seg == 5


def test_gs_ranged_form_only_applies_when_the_range_is_real():
    """§#gs--scale: the xnec2c extension applies only when ``I1 > 0`` and
    ``I2 >= I1``; any other pair scales the whole structure."""
    ranged = geometry(
        "GW 1 2 0. 0. 0. 1. 0. 0. 1.E-3\n"
        "GW 5 2 0. 1. 0. 1. 1. 0. 1.E-3\n"
        "GS 1 2 10.\nGE 0\n"
    )
    assert ranged.wires[0].p2[0] == 10.0
    assert ranged.wires[1].p2[0] == 1.0
    # I2 < I1 is not a range: everything scales.
    whole = geometry(
        "GW 1 2 0. 0. 0. 1. 0. 0. 1.E-3\n"
        "GW 5 2 0. 1. 0. 1. 1. 0. 1.E-3\n"
        "GS 3 1 10.\nGE 0\n"
    )
    assert [w.p2[0] for w in whole.wires] == [10.0, 10.0]


def test_gs_refuses_a_non_positive_factor():
    """§#gs--scale."""
    with pytest.raises(DeckError) as exc:
        geometry("GW 1 2 0 0 0 1 0 0 1E-3\nGS 0 0 -2.\nGE 0\n")
    assert str(exc.value) == "scale factor must be > 0, got -2.0"


# ---------------------------------------------------------------------------
# #ge--end-of-geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag,expected", [("0", False), ("1", True), ("-1", True)])
def test_ge_records_the_ground_plane_flag(flag, expected):
    """§#ge--end-of-geometry: ``0`` = no ground plane, anything else = ground
    plane present."""
    assert (
        geometry(f"GW 1 2 0 0 0 1 0 0 1E-3\nGE {flag}\n").ground_plane_flag is expected
    )


def test_ge_does_not_specify_a_ground():
    """§#ge--end-of-geometry: a deck with ``GE -1`` and no ``GN`` solves in
    free space — the flag drives the structure report and nothing else."""
    model = parse(DIPOLE.replace("GE 0", "GE -1"))
    assert model.ground_plane_flag is True
    assert model.ground is None


def test_a_deck_with_no_ge_parses_as_though_it_wrote_ge_zero():
    """§#deck-framing."""
    model = parse("GW 1 5 0 0 0 1 0 0 1E-3\nEX 0 1 3 0 1.\nXQ\nNX\n")
    assert model.ground_plane_flag is False
    assert len(model.wires) == 1


# ---------------------------------------------------------------------------
# #connections
# ---------------------------------------------------------------------------


def test_endpoint_clustering_fuses_a_near_miss_corner():
    """§#connections, pass 1: endpoints within an L1 separation of ``1e-3``
    times the connecting segment's length are rewritten to one exact shared
    coordinate."""
    built = geometry(
        "GW 1 10 0. 0. 0. 1. 0. 0. 1.E-3\n"
        "GW 2 10 1.000000001 0. 0. 1. 1. 0. 1.E-3\nGE 0\n"
    )
    assert tuple(built.wires[1].p1) == tuple(built.wires[0].p2)


def test_a_gap_wider_than_the_tolerance_is_left_open():
    """§#connections: deliberately gapped geometry is unaffected."""
    built = geometry(
        "GW 1 10 0. 0. 0. 1. 0. 0. 1.E-3\nGW 2 10 1.00015 0. 0. 1. 1. 0. 1.E-3\nGE 0\n"
    )
    assert tuple(built.wires[1].p1) != tuple(built.wires[0].p2)


def test_the_tolerance_is_the_minimum_of_the_pair():
    """§#connections: for segments of different lengths the parser takes the
    MINIMUM — the conservative intersection, which never connects what a NEC
    build would leave open."""
    # A 0.25 m segment and a 0.0025 m one, 1e-4 m apart: inside 1e-3 of the
    # long segment, outside 1e-3 of the short one.
    built = geometry(
        "GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3\nGW 2 400 1.0001 0. 0. 2. 0. 0. 1.E-3\nGE 0\n"
    )
    assert tuple(built.wires[1].p1) != tuple(built.wires[0].p2)


def test_an_end_onto_a_wire_snaps_and_becomes_a_cut():
    """§#connections, pass 2: an endpoint within tolerance of another wire's
    interior segment boundary is snapped onto it, and the crossing becomes a
    cut so it carries current."""
    built = geometry(
        "GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3\nGW 2 4 0.5 0. 1.E-7 0.5 1. 0. 1.E-3\nGE 0\n"
    )
    assert tuple(built.wires[1].p1) == (0.5, 0.0, 0.0)
    assert built.cuts == {0: frozenset({2})}
    # The cut wire becomes two model wires meeting at the shared node, so
    # every connection in the structure is a shared wire END.
    assert len(built.pieces) == 3
    assert built.pieces[0].p2 == built.pieces[1].p1 == (0.5, 0.0, 0.0)
    assert [p.n_seg for p in built.pieces] == [2, 2, 4]


def test_a_shattered_wire_keeps_its_tag_radius_and_element_boundaries():
    """§#connections: the split is lossless — same segments, same
    boundaries."""
    model = parse(
        "GW 1 4 0. 0. 0. 1. 0. 0. 2.E-3\n"
        "GW 2 4 0.5 0. 0. 0.5 1. 0. 3.E-3\n"
        "GE 0\nEX 0 1 3 0 1.\nXQ\nNX\n"
    )
    assert [w.edge_elements for w in model.wires] == [(2,), (2,), (4,)]
    assert [w.radius for w in model.wires] == [2.0e-3, 2.0e-3, 3.0e-3]
    assert sum(w.n_elements for w in model.wires) == 8


# ---------------------------------------------------------------------------
# #addressing
# ---------------------------------------------------------------------------


TWO_TAGS = (
    "GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3\n"
    "GW 2 6 0. 1. 0. 1. 1. 0. 1.E-3\n"
    "GW 1 2 0. 2. 0. 1. 2. 0. 1.E-3\nGE 0\n"
)


def test_a_nonzero_tag_counts_only_that_tags_segments_in_card_order():
    """§#addressing: several wires may share a tag."""
    built = geometry(TWO_TAGS)
    assert built.locate(1, 1) == (0, 1)
    assert built.locate(1, 4) == (0, 4)
    assert built.locate(1, 5) == (2, 1)
    assert built.locate(2, 3) == (1, 3)


def test_tag_zero_is_an_absolute_segment_number():
    """§#addressing: with ``tag = 0`` the count runs over every segment of
    the structure, in card order."""
    built = geometry(TWO_TAGS)
    assert built.locate(0, 5) == (1, 1)
    assert built.locate(0, 11) == (2, 1)


def test_a_segment_number_of_zero_or_below_reads_as_one():
    """§#addressing."""
    built = geometry(TWO_TAGS)
    assert built.locate(2, 0) == built.locate(2, -4) == built.locate(2, 1)


def test_an_out_of_range_request_refuses_verbatim():
    """§#addressing."""
    built = geometry(TWO_TAGS)
    with pytest.raises(DeckError) as exc:
        built.locate(2, 7)
    assert str(exc.value) == "segment 7 is out of range for tag 2"
    with pytest.raises(DeckError) as exc:
        built.locate(0, 13)
    assert str(exc.value) == "segment 13 is out of range for the structure"
    with pytest.raises(DeckError) as exc:
        built.locate(9, 1)
    assert str(exc.value) == "segment 1 is out of range for tag 9"


def test_arclength_is_the_segment_centre():
    """§#addressing: local segment ``k`` of a wire's ``NS`` occupies
    ``[(k-1)L/NS, kL/NS]`` and its centre is ``(k - 1/2)L/NS`` metres from
    the wire's first vertex."""
    built = geometry("GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3\nGE 0\n")
    for k in range(1, 5):
        wire, arclength = built.resolve(1, k)
        assert wire == 0
        assert arclength == pytest.approx((k - 0.5) / 4.0)


def test_addressing_a_shattered_wire_lands_on_the_right_piece():
    """§#addressing + §#connections: the resolver's "wire" is a model wire,
    so an address on a cut wire resolves against the piece that holds it."""
    built = geometry(
        "GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3\nGW 2 4 0.5 0. 0. 0.5 1. 0. 1.E-3\nGE 0\n"
    )
    assert built.resolve(1, 1) == (0, 0.125)
    assert built.resolve(1, 2) == (0, 0.375)
    assert built.resolve(1, 3) == (1, 0.125)
    assert built.resolve(1, 4) == (1, 0.375)
    # ...and the point it lands on is the flat wire's own segment centre.
    model = parse(
        "GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3\n"
        "GW 2 4 0.5 0. 0. 0.5 1. 0. 1.E-3\n"
        "GE 0\nEX 0 1 3 0 1.\nXQ\nNX\n"
    )
    wire, arclength = built.resolve(1, 3)
    assert math.dist(model.wires[wire].point_at(arclength), (0.625, 0.0, 0.0)) < 1e-15


# ---------------------------------------------------------------------------
# #execution and #arming
# ---------------------------------------------------------------------------


BODY = "GW 1 5 -0.5 0. 0. 0.5 0. 0. 1.E-3\nGE 0\nEX 0 1 3 0 1.\nFR 0 1 0 0 14.\n"
# BODY without its FR / EX card, for tests that supply their own.
BODY_NO_FR = "GW 1 5 -0.5 0. 0. 0.5 0. 0. 1.E-3\nGE 0\nEX 0 1 3 0 1.\n"
BODY_NO_EX = "GW 1 5 -0.5 0. 0. 0.5 0. 0. 1.E-3\nGE 0\nFR 0 1 0 0 14.\n"


def test_the_first_execute_card_always_runs():
    """§#arming."""
    model = parse(BODY + "XQ\nNX\n")
    assert len(model.groups) == 1 and model.groups[0] is not None


def test_a_second_execute_card_with_nothing_new_produces_no_output():
    """§#arming: an execute card runs only if something has changed since the
    last one."""
    model = parse(BODY + "XQ\nXQ\nNX\n")
    assert len(model.groups) == 2
    assert model.groups[1] is None


@pytest.mark.parametrize(
    "card",
    ["EX 0 1 2 0 1.", "FR 0 1 0 0 21.", "LD 0 1 3 3 50. 0. 0.", "GN 1", "EK"],
)
def test_the_arming_cards_re_arm_the_next_execute_card(card):
    """§#arming: the arming cards are ``EX``, ``FR``, ``LD``, ``GN`` and
    ``EK`` — the cards that move the operator or the drive.  ``GN``'s
    membership is the oracle-verified one."""
    model = parse(BODY + f"XQ\n{card}\nXQ\nNX\n")
    assert model.groups[1] is not None


@pytest.mark.parametrize("card", ["GD 0 0 0 0 5. .001 20. -2.", "MP 4 8", "PT -1"])
def test_gd_mp_and_pt_are_explicitly_not_arming(card):
    """§#arming: ``GD`` moves nothing outside the far field's cliff modes,
    ``MP`` is advisory, and ``PT`` changes what a run prints, not what it
    computes.  None of them can turn a bare ``XQ`` into a fresh run."""
    model = parse(BODY + f"XQ\n{card}\nXQ\nNX\n")
    assert model.groups[1] is None


def test_the_arming_set_is_exactly_the_pages_five_cards():
    """§#arming, as a set."""
    assert _ARMING_CARDS == {"EX", "FR", "LD", "GN", "EK"}


# -- the operator cards, and the partial refill they produce -----------------


def test_the_operator_set_is_exactly_the_pages_two_cards():
    """§#arming: of the five arming cards, ``GN`` and ``EK`` move the
    OPERATOR; the rest move the drive, the loading table or the frequency
    list.  A subset of the arming set by construction."""
    assert _OPERATOR_CARDS == {"GN", "EK"}
    assert _OPERATOR_CARDS <= _ARMING_CARDS


@pytest.mark.parametrize("card", ["GN 1", "EK"])
def test_an_operator_card_between_execute_cards_refills_partially(card):
    """§#arming: an operator card between two execute cards rebuilds the
    operator without a new frequency list — ``refilled_partial``, not
    ``refilled``.  ``GN`` alone is enough: no ``EK``, no fresh ``FR``."""
    model = parse(BODY + f"XQ\n{card}\nXQ\nNX\n")
    assert model.groups[0].refilled is True
    assert model.groups[0].refilled_partial is False
    assert model.groups[1] is not None
    assert model.groups[1].refilled is False
    assert model.groups[1].refilled_partial is True


@pytest.mark.parametrize("card", ["EX 0 1 2 0 1.", "LD 0 1 3 3 50. 0. 0."])
def test_an_arming_card_that_is_not_an_operator_card_refills_neither_way(card):
    """§#arming: the run is real — the card armed it — but nothing announces
    a rebuilt operator, because ``EX`` moves the drive and ``LD`` is stamped
    outside the fill."""
    model = parse(BODY + f"XQ\n{card}\nXQ\nNX\n")
    assert model.groups[1] is not None
    assert model.groups[1].refilled is False
    assert model.groups[1].refilled_partial is False


def test_a_fresh_fr_beats_a_partial_refill():
    """§#arming: an operator card AND a fresh ``FR`` between two execute cards
    is a whole refill — the frequency list moved too, so the group reports
    ``refilled`` and not the partial form."""
    model = parse(BODY + "XQ\nGN 1\nFR 0 1 0 0 21.\nXQ\nNX\n")
    assert model.groups[1] is not None
    assert model.groups[1].refilled is True
    assert model.groups[1].refilled_partial is False


def test_an_operator_card_before_the_first_execute_card_refills_whole():
    """§#arming: the first execute card of a deck always refills, so the
    partial flag is never set on it."""
    model = parse(BODY + "GN 1\nEK\nXQ\nNX\n")
    assert model.groups[0].refilled is True
    assert model.groups[0].refilled_partial is False


def test_the_operator_test_is_the_card_not_the_value_it_carries():
    """§#arming: NEC rebuilds because a kernel card arrived, not because the
    kernel differed.  ``EK -1`` on a deck already using the standard kernel
    refills exactly as a change does."""
    model = parse(BODY + "XQ\nEK -1\nXQ\nNX\n")
    assert model.groups[1] is not None
    assert model.groups[1].extended_kernel is False
    assert model.groups[1].refilled_partial is True


# -- the environment each group ran under ------------------------------------


def test_a_group_carries_the_environment_in_force_at_its_execute_card():
    """§#the-deckmodel: the environment is a per-execute-group quantity.  A
    ``GN`` between two execute cards arms, so the deck runs once in free
    space and once over ground and each group says which."""
    model = parse(BODY + "XQ\nGN 1\nXQ\nNX\n")
    assert model.groups[0].environment == Environment()
    assert model.groups[1] is not None
    assert model.groups[1].environment.ground == "pec"
    assert model.groups[1].environment.ground_z == 0.0


def test_a_ground_to_ground_transition_moves_the_groups_environment_too():
    """§#the-deckmodel: not only free space to ground — a second ``GN`` over
    a different half-space is the same per-group move."""
    model = parse(
        BODY + "GN 0 0 0 0 13. .005\nXQ\nGN 2 0 0 0 5. .001\nXQ\nNX\n",
    )
    assert model.groups[0].environment.ground == ("finite-fast", 13.0, 0.005)
    assert model.groups[1] is not None
    assert model.groups[1].environment.ground == ("finite", 5.0, 0.001)


def test_the_deck_level_ground_is_the_decks_last_environment():
    """§#the-deckmodel: ``DeckModel.ground`` keeps meaning what it always
    meant — the environment the cards had reached at the END of the deck —
    which for a multi-environment deck is the last group's, not every
    group's."""
    model = parse(BODY + "XQ\nGN 1\nXQ\nNX\n")
    assert model.ground == "pec"
    assert model.environment == model.groups[-1].environment
    assert model.groups[0].environment.ground is None


def test_a_gd_after_an_execute_card_reaches_no_group():
    """§#gd--additional-ground-medium: ``GD`` does not arm, so a card that
    arrives after the last execute card moves the DECK's second medium and no
    group's — there is no further group for it to reach."""
    model = parse(BODY + "XQ\nGD 0 0 0 0 5. .001 20. -2.\nXQ\nNX\n")
    assert model.groups[1] is None
    assert model.second_medium == SecondMedium(5.0, 0.001, 20.0, -2.0)
    assert model.groups[0].environment.second_medium is None


def test_a_gd_before_an_execute_card_rides_that_groups_environment():
    """§#gd--additional-ground-medium: the cliff a far-field request reads is
    the one in force at its own execute card."""
    model = parse(BODY + "GD 0 0 0 0 5. .001 20. -2.\nXQ\nNX\n")
    assert model.groups[0].environment.second_medium == SecondMedium(
        5.0, 0.001, 20.0, -2.0
    )


def test_every_group_of_a_single_environment_deck_carries_the_decks_own():
    """§#the-deckmodel: the common case, stated so the per-group field is not
    read as a change of meaning.  Most decks state their ground once before
    the first execute card, and then every group's environment IS the
    deck's."""
    model = parse(BODY + "GN 2 0 0 0 13. .005\nXQ\nEX 0 1 2 0 1.\nXQ\nNX\n")
    assert [g.environment for g in model.groups] == [model.environment] * 2


def test_a_fresh_fr_rebuilds_the_operator():
    """§#frequency-groups: an execute card that follows a fresh ``FR`` runs
    the whole list; one with no new ``FR`` re-runs at the last frequency."""
    model = parse(BODY + "XQ\nFR 0 3 0 0 14. 1.\nXQ\nEK\nXQ\nNX\n")
    assert [g.refilled for g in model.groups if g] == [True, True, False]


def test_every_execute_card_is_an_execute_card():
    """§#execution: ``RP``, ``NE`` and ``NH`` run the pending group and then
    report; a plain ``XQ`` leaves the request empty."""
    model = parse(
        BODY + "RP 0 19 1 1000 0. 0. 5. 0. 0.\nEK\nNE 0 3 1 1 0. 0. 1. .1 0. 0.\nNX\n"
    )
    assert isinstance(model.groups[0].request, FarFieldRequest)
    assert isinstance(model.groups[1].request, NearFieldRequest)
    assert parse(BODY + "XQ\nNX\n").groups[0].request is None


def test_a_far_field_request_carries_the_cards_geometry():
    """§#rp--radiation-pattern."""
    model = parse(BODY + "RP 0 19 2 1000 10. 20. 5. 90. 1000.\nNX\n")
    request = model.groups[0].request
    assert request == FarFieldRequest(0, 19, 2, 10.0, 20.0, 5.0, 90.0, 1000.0)
    # NTH/NPH below 1 read as 1.
    assert (
        parse(BODY + "RP 0 0 0 0 0. 0. 0. 0. 0.\nNX\n").groups[0].request.n_theta == 1
    )


@pytest.mark.parametrize("mode", [1, 4, 5, 6])
def test_rp_refuses_the_modes_this_engine_does_not_compute(mode):
    """§#rp--radiation-pattern."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + f"RP {mode} 19 1 1000 0. 0. 5. 0. 0.\nNX\n")
    assert str(exc.value) == (
        f"RP mode {mode} is not supported by this engine (modes 0, 2, 3 only)"
    )


def test_a_near_field_request_carries_its_grid():
    """§#ne--near-electric-field: samples vary X fastest, then Y, then Z."""
    model = parse(BODY + "NE 0 3 4 5 1. 2. 3. .1 .2 .3\nNX\n")
    assert model.groups[0].request == NearFieldRequest(
        magnetic=False, counts=(3, 4, 5), origin=(1.0, 2.0, 3.0), step=(0.1, 0.2, 0.3)
    )
    assert parse(BODY + "NH 0 1 1 1 0. 0. 1. 0. 0. 0.\nNX\n").groups[0].request.magnetic


@pytest.mark.parametrize("mnemonic", ["NE", "NH"])
def test_near_field_refuses_a_spherical_grid(mnemonic):
    """§#ne--near-electric-field: rectangular (0) only."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + f"{mnemonic} 1 1 1 1 0. 0. 0. 0. 0. 0.\nNX\n")
    assert str(exc.value) == (
        f"{mnemonic} coordinate system 1 (spherical) is not supported by this "
        f"engine; rectangular (0) only"
    )


def test_pt_is_a_toggle_that_holds_across_execute_cards():
    """§#pt--current-print-control."""
    model = parse(BODY + "PT -1\nXQ\nEK\nXQ\nPT -2\nEK\nXQ\nNX\n")
    assert [g.print_control.suppressed for g in model.groups if g] == [
        True,
        True,
        False,
    ]


def test_pt_zero_restricts_the_report_to_a_resolved_element_range():
    """§#pt--current-print-control: addressed exactly as an ``EX`` addresses a
    segment, and carried in the MODEL's own (wire, element) terms."""
    model = parse(BODY + "PT 0 1 2 4\nXQ\nNX\n")
    assert model.groups[0].print_control.elements == ((0, 1), (0, 2), (0, 3))
    # An all-zero range means "no restriction" rather than "no rows".
    assert parse(BODY + "PT 0 1 0 0\nXQ\nNX\n").groups[0].print_control.elements is None
    # 1, 2 and 3 are no restriction too.
    assert parse(BODY + "PT 2 0 0 0\nXQ\nNX\n").groups[0].print_control.elements is None


def test_mp_is_recorded_per_group_and_changes_nothing():
    """§#mp--multiprocessing-hint."""
    model = parse(BODY + "MP 4 8\nXQ\nNX\n")
    assert model.groups[0].multiprocessing == (4, 8)


def test_mp_refuses_a_fractional_field():
    """§#mp--multiprocessing-hint: matching NEC's integer-field reader."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + "MP 4.5 8\nXQ\nNX\n")
    assert str(exc.value) == "MP field 1 must be an integer, not 4.5"


def test_a_deck_with_no_ex_card_is_refused():
    """§#excitation-retention."""
    with pytest.raises(DeckError) as exc:
        parse("GW 1 5 0 0 0 1 0 0 1E-3\nGE 0\nFR 0 1 0 0 14.\nXQ\nNX\n")
    assert str(exc.value) == "deck has no EX card — nothing drives the structure"


# ---------------------------------------------------------------------------
# #gn--ground-parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,expected",
    [
        (-1, None),
        (1, "pec"),
        (0, ("finite-fast", 13.0, 0.005)),
        (2, ("finite", 13.0, 0.005)),
    ],
)
def test_gn_type_maps_to_momwires_own_ground_spelling(code, expected):
    """§#gn--ground-parameters: the four-row table."""
    model = parse(BODY + f"GN {code} 0 0 0 13. .005\nXQ\nNX\n")
    assert model.ground == expected


def test_gn_free_and_pec_ignore_epsr_and_sig_but_still_read_them():
    """§#gn--ground-parameters: ``GN -1`` and ``GN 1`` take no parameters;
    F1/F2 are read but move nothing."""
    assert parse(BODY + "GN -1 0 0 0 99. 99.\nXQ\nNX\n").ground is None
    assert parse(BODY + "GN 1 0 0 0 99. 99.\nXQ\nNX\n").ground == "pec"


def test_gn_refuses_an_unknown_type():
    """§#gn--ground-parameters."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + "GN 3\nXQ\nNX\n")
    assert str(exc.value) == "GN type 3 is not supported by this engine"


@pytest.mark.parametrize("code", [0, 2])
def test_gn_refuses_a_radial_ground_screen_on_the_reflection_coefficient_types(code):
    """§#gn--ground-parameters: NEC folds a screen into the reflection
    coefficient, so this is refused for the two ground types that HAVE one."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + f"GN {code} 4\nXQ\nNX\n")
    assert str(exc.value) == (
        f"GN {code} with a 4-wire radial ground screen is not supported by this engine"
    )


@pytest.mark.parametrize("code", [-1, 1])
def test_gn_pec_and_free_space_do_not_check_nradl(code):
    """§#gn--ground-parameters, oracle-verified (probe
    ``gn_pec_radial_probe.nec``): nec2c's PEC and free-space branches never
    look at NRADL — only the reflection-coefficient types (0, 2) fold a
    screen into a reflection coefficient at all — so a nonzero radial count
    on ``GN -1``/``GN 1`` runs rather than refusing."""
    model = parse(BODY + f"GN {code} 4\nXQ\nNX\n")
    assert model.ground == (None if code == -1 else "pec")


def test_gn_second_medium_when_nradl_is_zero():
    """§#gn--ground-parameters: F3-F6 carry a whole second medium in the
    same four slots a GD card sets.  Stated on a ``GN 2`` because the same
    four slots under a ``GN 1`` are the MININEC-type ground idiom, refused
    at the execute card (§#gd--additional-ground-medium)."""
    model = parse(BODY + "GN 2 0 0 0 13. .005 20. -2.\nXQ\nNX\n")
    assert model.second_medium == SecondMedium(20.0, -2.0, 0.0, 0.0)


def test_a_bare_gn_clears_an_earlier_cliff():
    """§#gn--ground-parameters: a GN that reaches the parser always rewrites
    those four slots, so a bare ``GN 1`` clears an earlier cliff."""
    model = parse(BODY + "GN 1 0 0 0 5. .001 20. -2.\nGN 1\nXQ\nNX\n")
    assert model.second_medium == SecondMedium()


def test_a_radial_screen_count_keeps_the_second_medium_slots():
    """§#gn--ground-parameters: a nonzero NRADL takes F3/F4 as the screen's
    own geometry and leaves any earlier cliff alone — measured through the
    free-space escape hatch, since 0/2 refuse outright with NRADL set and a
    cliff under ``GN 1`` is the MININEC-type ground idiom."""
    model = parse(BODY + "GN -1 0 0 0 5. .001 20. -2.\nGN -1 4\nXQ\nNX\n")
    assert model.second_medium == SecondMedium(20.0, -2.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# #gd--additional-ground-medium
# ---------------------------------------------------------------------------


def test_gd_sets_the_second_medium():
    """§#gd--additional-ground-medium."""
    model = parse(BODY + "GD 0 0 0 0 5. .001 20. -2.\nXQ\nNX\n")
    assert model.second_medium == SecondMedium(5.0, 0.001, 20.0, -2.0)


def test_a_bare_gd_runs_exactly_as_a_fully_populated_one_does():
    """§#gd--additional-ground-medium: ``I1``-``I4``, ``F5`` and ``F6`` are
    ignored."""
    model = parse(BODY + "GD\nXQ\nNX\n")
    assert model.second_medium == SecondMedium()


def test_a_later_gd_or_gn_overwrites_an_earlier_one():
    """§#gd--additional-ground-medium: a later ``GD`` (or a later ``GN`` with
    ``NRADL = 0``) overwrites an earlier one."""
    model = parse(
        BODY + "GD 0 0 0 0 5. .001 20. -2.\nGD 0 0 0 0 9. .002 30. -3.\nXQ\nNX\n"
    )
    assert model.second_medium == SecondMedium(9.0, 0.002, 30.0, -3.0)
    model = parse(
        BODY
        + "GD 0 0 0 0 5. .001 20. -2.\nGN 2 0 0 0 13. .005 9. .002 30. -3.\nXQ\nNX\n"
    )
    assert model.second_medium == SecondMedium(9.0, 0.002, 30.0, -3.0)


def test_gd_is_not_an_arming_card():
    """§#gd--additional-ground-medium and §#arming: it moves nothing outside
    the far field's cliff modes."""
    model = parse(BODY + "XQ\nGD 0 0 0 0 5. .001 20. -2.\nXQ\nNX\n")
    assert model.groups[1] is None


# -- the MININEC-type ground idiom (#458) -----------------------------------

MININEC_GROUND = (
    "GD with a perfect ground (GN 1) in force is the MININEC-type ground "
    "idiom (4nec2 GN 3; EZNEC 'MININEC-type'), which this engine does not "
    "implement (momwire#456): use GN 2 for a finite Sommerfeld ground, or "
    "drop the GD for perfect ground"
)


@pytest.mark.parametrize(
    "cards",
    [
        # 4nec2's manufactured form, written out of its own ``GN 3``.
        "GN 1\nGD 0 0 0 0 13. .005\nRP 0 1 1 1000 0. 0. 0. 10.\n",
        # The hand-written form 4 bundled models carry: ``I1`` is ignored by
        # the GD reader, so it reaches exactly the same state.
        "GN 1\nGD 2 0 0 0 13. .005\nRP 0 1 1 1000 0. 0. 0. 10.\n",
        # The same pair on an XQ: the idiom is the environment, not the
        # request, so which execute card fires does not change the answer.
        "GN 1\nGD 0 0 0 0 13. .005\nXQ\n",
        # A near-field request reads the record no more than ``RP 0`` does.
        "GN 1\nGD 0 0 0 0 13. .005\nNE 0 1 1 1 0. 0. 1. 0. 0. 0.\n",
        # The whole cliff stated on the GN card itself (F3-F6), with no GD
        # anywhere: the same state, so the same refusal.
        "GN 1 0 0 0 0. 0. 13. .005\nXQ\n",
    ],
)
def test_a_second_medium_under_a_perfect_ground_refuses_by_name(cards):
    """§#gd--additional-ground-medium: ``GD`` under a ``GN 1`` is how both
    frontends spell MININEC-type ground, and NEC-2 answers it as plain
    perfect ground — 34 % in R, silently."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + cards + "NX\n")
    assert str(exc.value) == MININEC_GROUND


def test_a_bare_perfect_ground_still_serves():
    """§#gd--additional-ground-medium: an all-zero second medium is no
    medium — it is what a bare ``GN 1`` writes into the four slots — so the
    commonest ground card in the corpus does not trip the gate."""
    model = parse(BODY + "GN 1\nXQ\nNX\n")
    assert model.ground == "pec"
    assert model.groups[0].environment.second_medium == SecondMedium()


def test_a_perfect_ground_cliff_under_rp2_and_rp3_still_serves():
    """§#gd--additional-ground-medium: modes 2 and 3 READ the record, so
    nothing is silent — those groups are answered, not refused (fixtures
    ``dipole_rp2_linear_cliff`` / ``dipole_rp3_circular_cliff``)."""
    for mode in (2, 3):
        model = parse(
            BODY
            + "GN 1\nGD 0 0 0 0 5. .001 10. -2.\n"
            + f"RP {mode} 9 5 1001 0. 0. 10. 45. 1000.\nNX\n"
        )
        assert model.groups[0].environment.second_medium == SecondMedium(
            5.0, 0.001, 10.0, -2.0
        )


def test_a_genuine_cliff_over_a_finite_ground_still_serves():
    """§#gd--additional-ground-medium: the gate is perfect ground's alone —
    ``GN 0`` and ``GN 2`` answer the second medium in every mode."""
    for code in (0, 2):
        for request in ("XQ", "RP 2 9 5 1001 0. 0. 10. 45. 1000."):
            model = parse(
                BODY
                + f"GN {code} 0 0 0 13. .005\nGD 0 0 0 0 5. .001 10. -2.\n"
                + request
                + "\nNX\n"
            )
            assert model.groups[0].environment.second_medium == SecondMedium(
                5.0, 0.001, 10.0, -2.0
            )


def test_free_space_with_a_second_medium_still_serves():
    """§#gd--additional-ground-medium: ``GD`` is read by the cliff modes only,
    and free space has no reflection at all — there is no substitution to
    make, so ``GN -1`` is untouched."""
    model = parse(BODY + "GN -1\nGD 0 0 0 0 13. .005\nXQ\nNX\n")
    assert model.groups[0].environment.ground is None


def test_the_idiom_is_judged_per_execute_group():
    """§#the-environment-is-per-execute-group: a deck that states the pair
    and then switches to ``GN 2`` before its only execute card never runs the
    idiom, so it is not refused for having written it."""
    model = parse(BODY + "GN 1\nGD 0 0 0 0 13. .005\nGN 2 0 0 0 13. .005\nXQ\nNX\n")
    assert model.groups[0].environment.ground == ("finite", 13.0, 0.005)


# ---------------------------------------------------------------------------
# #fr--frequency
# ---------------------------------------------------------------------------


def test_fr_linear_sweep():
    """§#fr--frequency: ``F1 + i*F2`` for ``i`` in ``0..NFRQ-1``."""
    model = parse(BODY_NO_FR + "FR 0 4 0 0 14. 0.5\nXQ\nNX\n")
    assert model.groups[0].frequencies == (14.0, 14.5, 15.0, 15.5)


def test_fr_multiplicative_sweep():
    """§#fr--frequency: ``F1 * F2**i``."""
    model = parse(BODY_NO_FR + "FR 1 3 0 0 10. 2.\nXQ\nNX\n")
    assert model.groups[0].frequencies == (10.0, 20.0, 40.0)


@pytest.mark.parametrize("ratio", ["0.", "-1."])
def test_fr_multiplicative_sweep_degenerates_on_a_non_positive_ratio(ratio):
    """§#fr--frequency: a multiplicative sweep whose ratio is zero or
    negative degenerates to NFRQ copies of F1."""
    model = parse(BODY_NO_FR + f"FR 1 3 0 0 10. {ratio}\nXQ\nNX\n")
    assert model.groups[0].frequencies == (10.0, 10.0, 10.0)


def test_fr_nfrq_below_one_reads_as_one():
    """§#fr--frequency."""
    model = parse(BODY_NO_FR + "FR 0 0 0 0 14. 1.\nXQ\nNX\n")
    assert model.groups[0].frequencies == (14.0,)


def test_every_fr_is_read_and_a_later_one_replaces_the_list():
    """§#fr--frequency: every FR in the deck is read; a later one replaces
    the list entirely."""
    model = parse(BODY_NO_FR + "FR 0 2 0 0 14. 1.\nXQ\nFR 0 3 0 0 20. 2.\nXQ\nNX\n")
    assert model.groups[0].frequencies == (14.0, 15.0)
    assert model.groups[1].frequencies == (20.0, 22.0, 24.0)


def test_an_execute_card_with_no_new_fr_runs_at_the_last_frequency_only():
    """§#frequency-groups."""
    model = parse(BODY_NO_FR + "FR 0 3 0 0 14. 1.\nXQ\nEK\nXQ\nNX\n")
    assert model.groups[0].frequencies == (14.0, 15.0, 16.0)
    assert model.groups[1].frequencies == (16.0,)


def test_a_deck_with_no_fr_card_runs_at_nec2cs_own_default():
    """§#fr--frequency, oracle-verified (probe ``no_fr_probe.nec``): the spec
    is silent on a deck with no FR card at all — every corpus deck sends
    one — but nec2c's own FMHZ starts at 299.8 (a 1 m wavelength), not 0."""
    model = parse(BODY_NO_FR + "XQ\nNX\n")
    assert model.groups[0].frequencies == (299.8,)


# ---------------------------------------------------------------------------
# #ex--voltage-source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ex_type", [1, 2, 3, 4, 5, 6])
def test_ex_refuses_every_type_but_zero(ex_type):
    """§#ex--voltage-source."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + f"EX {ex_type} 1 3 0 1.\nXQ\nNX\n")
    assert str(exc.value) == (
        f"EX type {ex_type} is not a voltage source; this engine drives EX 0 only"
    )


def test_ex_places_the_source_at_the_segment_centre():
    """§#ex--voltage-source: the same addressing EX/LD/IS/PT all share."""
    model = parse(BODY + "XQ\nNX\n")
    wire, arclength, volts = model.feeds[0]
    assert wire == 0
    # A 1 m wire, 5 segments: segment 3's centre is (3 - 1/2)/5 = 0.5 m
    # from the wire's first vertex.
    assert arclength == pytest.approx(0.5)
    assert volts == complex(1.0, 0.0)


def test_the_first_ex_after_an_execution_replaces_the_source_list():
    """§#excitation-retention."""
    model = parse(
        BODY_NO_EX + "EX 0 1 1 0 1.\nEX 0 1 4 0 2.\nXQ\nEX 0 1 5 0 3.\nXQ\nNX\n"
    )
    # Group 0 drives both ports 0 and 1; group 1's fresh EX replaces the
    # list, so only the new port (2) is driven.
    assert model.groups[0].voltages == (complex(1.0), complex(2.0), 0j)
    assert model.groups[1].voltages == (0j, 0j, complex(3.0))


def test_an_execute_card_with_no_new_ex_redrives_the_previous_set():
    """§#excitation-retention: an execute card with no EX since the previous
    one re-drives the previous set unchanged.  Armed via ``EK`` rather than a
    new ``EX``, so the second group is a real run rather than a no-op, and
    the retained voltage is what it is measured against."""
    model = parse(BODY_NO_EX + "EX 0 1 3 0 1.\nXQ\nEK\nXQ\nNX\n")
    assert len(model.groups) == 2
    assert model.groups[1] is not None
    assert model.groups[1].voltages == model.groups[0].voltages == (complex(1.0),)


def test_every_ex_segment_across_every_group_becomes_one_union_port_set():
    """§#one-geometry-one-port-set: in discovery order, the same segment may
    be driven in different groups at different voltages."""
    model = parse(
        BODY_NO_EX + "EX 0 1 1 0 1.\nXQ\nEX 0 1 4 0 2.\nXQ\nEX 0 1 1 0 9.\nXQ\nNX\n"
    )
    # Discovery order: segment 1 first (group 0), then segment 4 (group 1).
    # Group 2 redrives segment 1 at a NEW voltage without adding a new port.
    assert len(model.feeds) == 2
    assert model.groups[0].voltages == (complex(1.0), 0j)
    assert model.groups[1].voltages == (0j, complex(2.0))
    assert model.groups[2].voltages == (complex(9.0), 0j)


# ---------------------------------------------------------------------------
# #ld--loading
# ---------------------------------------------------------------------------


def test_ld_minus_one_nullifies_every_load_but_not_is_insulation():
    """§#ld--loading: ``LD -1`` clears the whole list — and only loads."""
    model = parse(
        BODY + "IS 0 1 0 0 5. 0. 0.002\nLD 0 1 2 2 50. 0. 0.\nLD -1\nXQ\nNX\n"
    )
    assert model.loads == ()
    assert model.wires[0].material.insulation_radius == 0.002


def test_ld_series_and_parallel_rlc():
    """§#ld--loading: types 0 and 1."""
    model = parse(BODY + "LD 0 1 3 3 50. 1.e-6 2.e-9\nXQ\nNX\n")
    (load,) = model.loads
    assert load[2] == LoadSpec("series", r=50.0, l=1e-6, c=2e-9)
    model = parse(BODY + "LD 1 1 3 3 50. 1.e-6 2.e-9\nXQ\nNX\n")
    (load,) = model.loads
    assert load[2] == LoadSpec("parallel", r=50.0, l=1e-6, c=2e-9)


def test_ld_fixed_impedance():
    """§#ld--loading: type 4, R + jX."""
    model = parse(BODY + "LD 4 1 3 3 100. -75.\nXQ\nNX\n")
    (load,) = model.loads
    assert load[2] == LoadSpec("fixed", r=100.0, x=-75.0)


def test_zero_valued_loads_are_dropped_as_no_ops():
    """§#ld--loading."""
    assert parse(BODY + "LD 0 1 3 3 0. 0. 0.\nXQ\nNX\n").loads == ()
    assert parse(BODY + "LD 4 1 3 3 0. 0.\nXQ\nNX\n").loads == ()


@pytest.mark.parametrize("ldtyp", [2, 3, 6, 7, 8, 99])
def test_ld_refuses_the_types_this_engine_does_not_support(ldtyp):
    """§#ld--loading: 2/3 (per-metre) and 6/7 (4nec2 extensions) refuse, and
    so does any type this engine does not recognise."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + f"LD {ldtyp} 1 3 3 1. 1. 1.\nXQ\nNX\n")
    assert str(exc.value) == f"LD type {ldtyp} is not supported by this engine"


def test_ld_expands_a_whole_tag_when_the_range_is_zero():
    """§#ld--loading: I3 = 0 loads every segment of the tag (matching the
    whole-structure form's own I2 = 0, I3 = 0 sentinel), oracle-verified
    (probe ``ld_range_probe.nec``/``ld_range_probe4.nec``): I4 is ignored
    entirely once I3 = 0, not treated as an upper bound."""
    model = parse(BODY + "LD 0 1 0 0 50. 0. 0.\nXQ\nNX\n")
    assert len(model.loads) == 5
    # I4 = 3 does not restrict the range once I3 = 0 (oracle-verified).
    model = parse(BODY + "LD 0 1 0 3 50. 0. 0.\nXQ\nNX\n")
    assert len(model.loads) == 5


def test_ld_over_eight_segments_is_refused():
    """§#ld--loading: a range wider than 8 segments is refused rather than
    silently truncated."""
    body = "GW 1 20 -0.5 0. 0. 0.5 0. 0. 1.E-3\nGE 0\nEX 0 1 3 0 1.\nFR 0 1 0 0 14.\n"
    with pytest.raises(DeckError) as exc:
        parse(body + "LD 0 1 1 9 50. 0. 0.\nXQ\nNX\n")
    assert str(exc.value) == (
        "LD over 9 segments is not supported by this engine — at most 8 "
        "segments expand into per-segment loads"
    )


def test_ld_refuses_a_second_load_on_the_same_segment():
    """§#ld--loading."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + "LD 0 1 2 2 50. 0. 0.\nLD 4 1 2 2 10. 5.\nXQ\nNX\n")
    assert str(exc.value) == (
        "LD on a segment that already carries a load is not supported by "
        "this engine — a second load on one segment is not merged"
    )


def test_ld5_whole_structure_form():
    """§#ld--loading: type 5, ``I2 = 0, I3 = 0`` sets every wire."""
    model = parse(
        "GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3\n"
        "GW 2 4 0. 1. 0. 1. 1. 0. 1.E-3\n"
        "GE 0\nEX 0 1 3 0 1.\nFR 0 1 0 0 14.\nLD 5 0 0 0 5.8e7\nXQ\nNX\n"
    )
    assert [w.material.conductivity for w in model.wires] == [5.8e7, 5.8e7]


def test_ld5_ranged_form_sets_it_per_wire():
    """§#ld--loading: a ranged form sets it per wire."""
    model = parse(
        "GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3\n"
        "GW 2 4 0. 1. 0. 1. 1. 0. 1.E-3\n"
        "GE 0\nEX 0 1 3 0 1.\nFR 0 1 0 0 14.\nLD 5 1 0 0 5.8e7\nXQ\nNX\n"
    )
    assert model.wires[0].material.conductivity == 5.8e7
    assert model.wires[1].material is None


def test_ld5_refuses_a_partial_wire_range():
    """§#ld--loading: the range must cover each touched wire in full."""
    with pytest.raises(DeckError) as exc:
        parse(
            "GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3\nGE 0\n"
            "EX 0 1 3 0 1.\nFR 0 1 0 0 14.\nLD 5 1 1 2 5.8e7\nXQ\nNX\n"
        )
    assert str(exc.value) == (
        "LD 5 conductivity on a partial-wire segment range is not "
        "supported by this engine — per-wire conductivity covers whole "
        "wires only"
    )


# ---------------------------------------------------------------------------
# #is--insulated-sheath
# ---------------------------------------------------------------------------


def test_is_sets_the_wires_insulation_jacket():
    """§#is--insulated-sheath."""
    model = parse(BODY + "IS 0 1 0 0 5. 0. 0.002\nXQ\nNX\n")
    material = model.wires[0].material
    assert material.insulation_eps_r == 5.0
    assert material.insulation_radius == 0.002


def test_is_after_an_execute_request_is_refused():
    """§#is--insulated-sheath: structural — one geometry, one set of
    per-wire specs, every group."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + "XQ\nIS 0 1 0 0 5. 0. 0.002\nNX\n")
    assert str(exc.value) == (
        "IS after an execute request is not supported by this engine: wire "
        "insulation is part of the structure, so it cannot change between "
        "runs"
    )


def test_is_refuses_a_conductive_sheath():
    """§#is--insulated-sheath."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + "IS 0 1 0 0 5. 0.1 0.002\nXQ\nNX\n")
    assert str(exc.value) == (
        "IS with a conductive sheath (F2 != 0) is not modelled by this "
        "engine — the insulation jacket is a lossless dielectric; set the "
        "sheath conductivity to 0"
    )


@pytest.mark.parametrize(
    "eps_r,radius", [(5.0, 0.0), (5.0, -1.0), (0.5, 0.002), (1.0, 0.002)]
)
def test_is_drops_a_vacuum_jacket_as_a_no_op(eps_r, radius):
    """§#is--insulated-sheath: F3 <= 0 or F1 <= 1 is electrically a vacuum."""
    model = parse(BODY + f"IS 0 1 0 0 {eps_r} 0. {radius}\nXQ\nNX\n")
    assert model.wires[0].material is None


def test_is_refuses_a_partial_wire_range():
    """§#is--insulated-sheath: exactly as a ranged LD 5 must."""
    with pytest.raises(DeckError) as exc:
        parse(
            "GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3\nGE 0\n"
            "EX 0 1 3 0 1.\nFR 0 1 0 0 14.\nIS 0 1 1 2 5. 0. 0.002\nXQ\nNX\n"
        )
    assert str(exc.value) == (
        "IS: insulation on a partial-wire segment range — per-wire specs "
        "cover whole wires only"
    )


def test_is_refuses_a_jacket_that_does_not_clear_the_conductor():
    """§#is--insulated-sheath."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + "IS 0 1 0 0 5. 0. 0.0005\nXQ\nNX\n")
    assert str(exc.value) == (
        "IS: insulation whose outer radius does not exceed the wire's conductor radius"
    )


def test_is_does_not_arm_the_next_execute_card():
    """§#is--insulated-sheath: it does not arm, because a deck that changes
    it between runs is refused outright."""
    model = parse(BODY + "IS 0 1 0 0 5. 0. 0.002\nXQ\nNX\n")
    assert len(model.groups) == 1


# ---------------------------------------------------------------------------
# #ek--extended-thin-wire-kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("card,expected", [
    ("EK", True), ("EK 0", True), ("EK 1", True), ("EK 2", True),
    ("EK -2", True), ("EK -1", False),
])  # fmt: skip
def test_ek_the_test_is_i1_equals_minus_one_and_nothing_else(card, expected):
    """§#ek--extended-thin-wire-kernel: matching NEC's own card reader."""
    model = parse(BODY + f"{card}\nXQ\nNX\n")
    assert model.groups[0].extended_kernel is expected


def test_ek_is_carried_per_execute_group():
    """§#ek--extended-thin-wire-kernel: one deck may answer two groups under
    two kernels with two fills.  ``EK`` arms unconditionally (spec ``#ek--
    extended-thin-wire-kernel``: "EK arms the next execute card"), so the
    second XQ is a real run even though ``EK -1`` did not change the value
    (it was already off)."""
    model = parse(BODY + "XQ\nEK\nXQ\nNX\n")
    assert model.groups[0].extended_kernel is False
    assert model.groups[1] is not None
    assert model.groups[1].extended_kernel is True
    model = parse(BODY + "XQ\nEK -1\nXQ\nNX\n")
    assert model.groups[0].extended_kernel is False
    assert model.groups[1] is not None  # EK still arms, even at its own value
    assert model.groups[1].extended_kernel is False


def test_a_kernel_change_between_execute_cards_rearms_without_a_new_fr():
    """§#ek--extended-thin-wire-kernel: a kernel change between two execute
    cards re-arms without a new FR."""
    model = parse(BODY + "XQ\nEK\nXQ\nNX\n")
    assert model.groups[1] is not None
    assert model.groups[1].refilled is False
    assert model.groups[1].refilled_partial is True


# ---------------------------------------------------------------------------
# #refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mnemonic", sorted(_REFUSED_BY_NAME))
def test_cards_refused_by_name(mnemonic):
    """§#cards-refused-by-name: every message, verbatim."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + f"{mnemonic} 1 2 3 4\nXQ\nNX\n")
    assert str(exc.value) == _REFUSED_BY_NAME[mnemonic]


def test_the_refusal_table_is_the_pages_table():
    """§#cards-refused-by-name, as a set: nothing in this dialect is accepted
    and silently ignored."""
    assert set(_REFUSED_BY_NAME) == {
        "TL", "NT", "GA", "GH", "GX", "GR", "GC", "GF", "SY",
        "SP", "SM", "SC", "KH", "PQ", "CP", "PL", "WG", "ZO",
    }  # fmt: skip


def test_an_unknown_card_is_refused_by_name():
    """§#cards-refused-by-name."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + "QW 1 2\nXQ\nNX\n")
    assert str(exc.value) == "unrecognised NEC card 'QW'"


# ---------------------------------------------------------------------------
# #dialects
# ---------------------------------------------------------------------------


def test_nec2_is_the_only_dialect_this_release_ships():
    """§#dialects: a second dialect is a second parser, not a second
    pipeline — and the refusal names the ones that exist."""
    with pytest.raises(DeckError) as exc:
        parse(DIPOLE, dialect="nec5")
    assert str(exc.value) == "unknown deck dialect 'nec5'; known dialects: 'nec2'"


def test_the_model_speaks_no_nec():
    """§#the-deckmodel: the model carries no tags, no segment numbers, no
    mnemonics and no card ordinals."""
    model = parse(DIPOLE)
    fields = set(type(model).__dataclass_fields__)
    assert not fields & {"tags", "segments", "cards", "geometry", "data_cards"}
    assert not hasattr(model.wires[0], "tag")


def test_a_finished_dialect_defers_nothing():
    """§#the-deckmodel: ``deferred`` "exists so an unfinished [dialect] is
    honest rather than silent" — every card this dialect reads now has real
    semantics (momwire#359 unit C), so a parsed deck defers nothing."""
    assert parse(DIPOLE).deferred == ()
