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
from momwire.deck._nec2 import _ARMING_CARDS, _GEOMETRY_CARDS, _REFUSED_BY_NAME
from momwire.deck._nec2_geometry import build_geometry
from momwire.deck.model import FarFieldRequest, NearFieldRequest

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
        "SP", "SM", "KH", "PQ", "CP", "PL", "WG", "ZO",
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
