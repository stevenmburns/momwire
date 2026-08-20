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
from momwire.deck._nec2_geometry import Symmetry, build_geometry
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
# #gx--structure-reflection
# ---------------------------------------------------------------------------

# The gating fixture of momwire#415, k9ay_orig's geometry: two wires meeting a
# reflection plane at one end each, mirrored in X=0 with a tag increment of 2,
# so the cell is tags 1 and 2 and its image is tags 3 and 4.
K9AY = (
    "GW 1 8 0. 0. 8.5 5. 0. 1.5 1.5E-3\nGW 2 5 0. 0. 0.  5. 0. 1.5 1.5E-3\nGX 2 100\n"
)


def test_gx_reflects_the_whole_structure_so_far():
    """§#gx--structure-reflection: one plane copies every wire built so far,
    negating one coordinate and adding the increment to every nonzero tag."""
    built = geometry(K9AY + "GE 0\n")
    assert [w.tag for w in built.wires] == [1, 2, 3, 4]
    assert built.wires[2].p1 == [0.0, 0.0, 8.5]
    assert built.wires[2].p2 == [-5.0, 0.0, 1.5]
    assert built.wires[3].p2 == [-5.0, 0.0, 1.5]
    # The image is a copy in every other respect.
    assert [w.n_seg for w in built.wires] == [8, 5, 8, 5]
    assert [w.radius for w in built.wires] == [1.5e-3] * 4


def test_gx_fires_z_then_y_then_x_and_doubles_the_tag_increment():
    """§#gx--structure-reflection: the planes fire in NEC's own order, and
    the increment doubles after each one, so one wire tagged 1 under all
    three planes becomes eight wires tagged 1 through 8."""
    built = geometry("GW 1 2 1. 1. 1. 2. 2. 2. 1.E-3\nGX 1 111\nGE 0\n")
    assert [w.tag for w in built.wires] == [1, 2, 3, 4, 5, 6, 7, 8]
    # Z first (tag 2), then Y over both (tags 3, 4), then X over all four
    # (tags 5-8): the sign pattern is the firing order read backwards.
    assert [w.p1 for w in built.wires] == [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
    ]


def test_the_k9ay_orig_cell_is_the_one_nec2c_prints():
    """§#gx--structure-reflection, the gating fixture of momwire#415.

    nec2c on the deck this geometry is taken from:

        TOTAL SEGMENTS USED: 26   SEGMENTS IN A SYMMETRIC CELL: 13
        SYMMETRY FLAG: 1   STRUCTURE HAS 1 PLANES OF SYMMETRY
    """
    built = geometry(K9AY + "GE 0\n")
    assert built.n_segments == 26
    assert built.symmetry == Symmetry(cell_wires=2, cell_segments=13)


def test_gx_images_are_reachable_by_their_own_tags():
    """§#gx--structure-reflection + §#addressing: the doubling exists so that
    every image keeps a unique tag, which is what makes a deck's ``EX``/``LD``
    able to name one — k9ay_orig drives tag 4, an image."""
    built = geometry(K9AY + "GE 0\n")
    assert built.locate(3, 1) == (2, 1)
    assert built.locate(4, 5) == (3, 5)
    # Two planes, increment 1: the Y image of tag 1 is tag 2, and the X
    # images are tags 3 and 4 — the increment having doubled once.
    two = geometry("GW 1 2 1. 1. 0. 2. 1. 0. 1.E-3\nGX 1 110\nGE 0\n")
    assert [w.tag for w in two.wires] == [1, 2, 3, 4]
    assert two.locate(4, 1) == (3, 1)


def test_gx_keeps_tag_zero_at_zero():
    """§#gx--structure-reflection: a wire tagged 0 is tagged 0 in every
    image, exactly as under ``GM``."""
    built = geometry(
        "GW 0 2 1. 0. 0. 2. 0. 0. 1.E-3\n"
        "GW 7 2 1. 1. 0. 2. 1. 0. 1.E-3\n"
        "GX 5 100\nGE 0\n"
    )
    assert [w.tag for w in built.wires] == [0, 7, 0, 12]


@pytest.mark.parametrize(
    "code,plane",
    [("001", "Z=0"), ("010", "Y=0"), ("100", "X=0")],
)
def test_gx_refuses_a_wire_lying_in_a_firing_plane(code, plane):
    """§#gx--structure-reflection: both ends on the plane — the image would
    land on top of the wire."""
    with pytest.raises(DeckError) as exc:
        geometry(f"GW 1 2 0. 0. 0. 0. 0. 0. 1.E-3\nGX 1 {code}\nGE 0\n")
    assert str(exc.value) == f"a wire lies in or crosses the {plane} symmetry plane"


def test_gx_refuses_a_wire_crossing_a_firing_plane():
    """§#gx--structure-reflection: the two ends on opposite sides — the image
    would land inside the wire.

    This wire crosses at a SEGMENT BOUNDARY, which is the one shape where the
    per-wire test is stronger than NEC's per-segment one: nec2c accepts this
    deck and reports ``TOTAL SEGMENTS USED: 4 ... SYMMETRIC CELL: 2``, two of
    those four segments being duplicates lying on top of the other two.  The
    page says this dialect refuses the degenerate structure instead.
    """
    with pytest.raises(DeckError) as exc:
        geometry("GW 1 2 -1. 1. 0. 1. 1. 0. 1.E-3\nGX 1 100\nGE 0\n")
    assert str(exc.value) == "a wire lies in or crosses the X=0 symmetry plane"


def test_gx_only_tests_the_planes_that_fire():
    """§#gx--structure-reflection: a wire lying in Y=0 is fine under a
    reflection that only selects X=0 — the error is per firing plane."""
    built = geometry("GW 1 2 1. 0. 0. 2. 0. 0. 1.E-3\nGX 1 100\nGE 0\n")
    assert len(built.wires) == 2


def test_gx_allows_a_wire_touching_a_firing_plane_at_one_end():
    """§#gx--structure-reflection: legal and common — that is how a symmetric
    loop or an inverted-L is built, and it is what k9ay_orig does."""
    built = geometry("GW 1 2 0. 1. 0. 1. 1. 0. 1.E-3\nGX 1 100\nGE 0\n")
    assert len(built.wires) == 2
    assert built.wires[1].p1 == [0.0, 1.0, 0.0]
    assert built.wires[1].p2 == [-1.0, 1.0, 0.0]


def test_gx_declares_a_symmetric_cell_of_the_structure_so_far():
    """§#gx--structure-reflection: the cell is the structure as it stood when
    the card fired — nec2c's ``SEGMENTS IN A SYMMETRIC CELL``."""
    built = geometry("GW 1 5 1. 0. 0. 1. 0. 1. 1.E-3\nGX 1 100\nGE 0\n")
    assert built.symmetry == Symmetry(cell_wires=1, cell_segments=5)
    assert built.n_segments == 10


def test_a_two_plane_gx_still_declares_one_cell():
    """§#gx--structure-reflection: there is no hierarchy — two planes give
    one cell and four copies of it, not a cell of a cell."""
    built = geometry("GW 1 5 1. 1. 0. 1. 1. 1. 1.E-3\nGX 1 110\nGE 0\n")
    assert built.symmetry == Symmetry(cell_wires=1, cell_segments=5)
    assert len(built.wires) == 4 and built.n_segments == 20


def test_a_second_gx_resets_the_cell_to_the_whole_prior_structure():
    """§#gx--structure-reflection: two cards do not nest."""
    built = geometry("GW 1 5 1. 1. 0. 1. 1. 1. 1.E-3\nGX 1 100\nGX 4 010\nGE 0\n")
    assert built.symmetry == Symmetry(cell_wires=2, cell_segments=10)
    assert len(built.wires) == 4 and built.n_segments == 20


def test_a_gx_selecting_no_plane_retires_the_symmetry():
    """§#gx--structure-reflection: NEC resets the cell before it reads the
    plane code, so ``GX n 0`` is not a no-op — it ends the symmetry."""
    built = geometry("GW 1 5 1. 0. 0. 1. 0. 1. 1.E-3\nGX 1 100\nGX 9 0\nGE 0\n")
    assert built.symmetry is None
    assert len(built.wires) == 2


LIVE = "GW 1 5 1. 1. 0. 1. 1. 1. 1.E-3\nGX 1 100\n"


@pytest.mark.parametrize(
    "card",
    [
        "GM 0 0 0. 0. 90. 0. 0. 0. 0",  # whole-structure rotate
        "GM 0 0 0. 0. 0. 0. 0. 3. 0",  # whole-structure translate
        "GM 0 0 0. 0. 0. 0. 0. 3. 1",  # ITS naming the FIRST wire's tag
        "GS 0 0 2.",  # whole-structure scale
        "GS 1 2 2.",  # a ranged scale is still a scale
    ],
)
def test_a_congruence_of_the_whole_structure_preserves_the_symmetry(card):
    """§#gx--structure-reflection: symmetry survives a transform of the
    ENTIRE structure — nec2c's ``move`` returns before touching ``NP``, and
    its ``GS`` never touches it at all."""
    built = geometry(LIVE + card + "\nGE 0\n")
    assert built.symmetry == Symmetry(cell_wires=1, cell_segments=5)


@pytest.mark.parametrize(
    "card",
    [
        "GW 9 3 5. 5. 0. 5. 5. 1. 1.E-3",  # anything ADDED
        "GM 0 0 0. 0. 0. 0. 0. 3. 2",  # ITS restricting to the image
        "GM 1 1 0. 0. 0. 0. 0. 3. 0",  # NRPT replicating
    ],
)
def test_adding_or_selectively_transforming_destroys_the_symmetry(card):
    """§#gx--structure-reflection: symmetry dies the moment anything is added
    or transformed selectively — nec2c's ``NP = N``."""
    assert geometry(LIVE + card + "\nGE 0\n").symmetry is None


# ---------------------------------------------------------------------------
# #gr--cylindrical-structure-rotation
# ---------------------------------------------------------------------------


def test_the_issues_flat_layout_example():
    """§#gr--cylindrical-structure-rotation: momwire#415's own worked
    example — one cell and three copies, laid out flat and contiguous, each
    copy's tag one more than the last."""
    built = geometry("GW 1 5 0. 0. 0. 1. 0. 0. 1.E-3\nGR 1 4\nGE 0\n")
    assert [w.tag for w in built.wires] == [1, 2, 3, 4]
    assert [w.n_seg for w in built.wires] == [5, 5, 5, 5]
    assert built.n_segments == 20
    # segments 1-5 the cell, 6-10 / 11-15 / 16-20 the three copies in order.
    assert built.locate(1, 5) == (0, 5)
    assert built.locate(2, 1) == (1, 1)
    assert built.locate(3, 1) == (2, 1)
    assert built.locate(4, 5) == (3, 5)


def test_gr_rotates_each_copy_from_the_previous_ones_coordinates():
    """§#gr--cylindrical-structure-rotation: each copy is built by rotating
    the PREVIOUS copy about Z, not the original structure, so the rotations
    compound around the cylinder — a 90-degree step run four times returns
    to the start."""
    built = geometry("GW 1 5 0. 0. 0. 1. 0. 0. 1.E-3\nGR 1 4\nGE 0\n")
    assert built.wires[0].p2 == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    assert built.wires[1].p2 == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)
    assert built.wires[2].p2 == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9)
    assert built.wires[3].p2 == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)


def test_gr_keeps_tag_zero_at_zero():
    """§#gr--cylindrical-structure-rotation: a wire tagged 0 is tagged 0 in
    every copy, exactly as under ``GX``/``GM``."""
    built = geometry(
        "GW 0 2 1. 0. 0. 2. 0. 0. 1.E-3\nGW 7 2 1. 1. 0. 2. 1. 0. 1.E-3\nGR 5 3\nGE 0\n"
    )
    assert [w.tag for w in built.wires] == [0, 7, 0, 12, 0, 17]


def test_gr_0_shares_the_original_tag_across_all_copies():
    """§#gr--cylindrical-structure-rotation: ``I1 = 0`` is legal and
    common — every copy then carries the ORIGINAL tag, and a tag-addressed
    card resolves across all of them, in structure order."""
    built = geometry("GW 1 5 0. 0. 0. 1. 0. 0. 1.E-3\nGR 0 4\nGE 0\n")
    assert [w.tag for w in built.wires] == [1, 1, 1, 1]
    assert built.n_segments == 20
    assert built.locate(1, 6) == (1, 1)
    assert built.locate(1, 20) == (3, 5)


@pytest.mark.parametrize("nop", [0, -1, -4])
def test_gr_refuses_a_nonpositive_structure_count(nop):
    """§#gr--cylindrical-structure-rotation: ``nop < 1`` is a geometry
    error, not a silent no-op."""
    with pytest.raises(DeckError) as exc:
        geometry(f"GW 1 2 0. 0. 0. 1. 0. 0. 1.E-3\nGR 1 {nop}\nGE 0\n")
    assert str(exc.value) == f"structure count must be >= 1, got {nop}"


def test_gr_declares_a_symmetric_cell_of_the_structure_so_far():
    """§#gr--cylindrical-structure-rotation: the cell is the structure as it
    stood when the card fired — the same rule ``GX`` follows."""
    built = geometry("GW 1 5 1. 0. 0. 2. 0. 0. 1.E-3\nGR 1 4\nGE 0\n")
    assert built.symmetry == Symmetry(cell_wires=1, cell_segments=5)
    assert built.n_segments == 20


def test_a_degenerate_gr_of_one_structure_still_declares_symmetry():
    """§#gr--cylindrical-structure-rotation: nec2c sends ``GR`` through the
    same routine as ``GX`` and sets the symmetry flag before it ever looks
    at ``nop``, so ``nop = 1`` — no copies at all — still declares a cell,
    equal to the (unchanged) whole structure. This is a deliberate reading,
    not a fall-through: unlike ``GX n 0``, there is no ``GR`` spelling that
    retires symmetry."""
    built = geometry("GW 1 5 1. 0. 0. 2. 0. 0. 1.E-3\nGR 3 1\nGE 0\n")
    assert len(built.wires) == 1
    assert built.symmetry == Symmetry(cell_wires=1, cell_segments=5)


def test_a_gr_then_gx_resets_the_cell_to_the_whole_prior_structure():
    """§#gr--cylindrical-structure-rotation + §#gx--structure-reflection:
    two separate symmetry-creating cards do not nest — the later one resets
    the cell to the whole prior structure. momwire#415's own worked
    example: ``GR 1 4`` then ``GX 4 1`` gives cell 20 / total 40."""
    built = geometry("GW 1 5 1. 0. 0. 1. 0. 1. 1.E-3\nGR 1 4\nGX 4 1\nGE 0\n")
    assert built.symmetry == Symmetry(cell_wires=4, cell_segments=20)
    assert built.n_segments == 40


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


@pytest.mark.parametrize(
    "card", ["GD 0 0 0 0 5. .001 20. -2.", "MP 4 8", "PT -1", "PQ -1"]
)
def test_gd_mp_and_pt_are_explicitly_not_arming(card):
    """§#arming: ``GD`` moves nothing outside the far field's cliff modes,
    ``MP`` is advisory, and ``PT``/``PQ`` change what a run prints, not what
    it computes.  None of them can turn a bare ``XQ`` into a fresh run."""
    model = parse(BODY + f"XQ\n{card}\nXQ\nNX\n")
    assert model.groups[1] is None


def test_the_arming_set_is_exactly_the_pages_seven_cards():
    """§#arming, as a set.

    ``TL`` and ``NT`` joined it in momwire#456 phase C, oracle-measured: a
    network card alone between two execute cards produces a whole second
    answer.  They arm without refilling, which is ``EX``'s shape — see
    ``#network-retention``."""
    assert _ARMING_CARDS == {"EX", "FR", "LD", "GN", "EK", "TL", "NT"}


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


def test_pq_negative_parses_and_solves_identically_to_no_pq_at_all():
    """§#pq--charge-print-control: momwire's printout never had a charge
    report to suppress, so ``PQ -1`` (the suppression form) is a no-op —
    the parsed model is identical with or without it."""
    with_pq = parse(BODY + "PQ -1\nXQ\nNX\n")
    without_pq = parse(BODY + "XQ\nNX\n")
    assert with_pq == without_pq


def test_pq_nonnegative_refuses_the_missing_charge_report():
    """§#pq--charge-print-control: ``I1 >= 0`` is a print REQUEST, and this
    engine produces no charge-density report to print."""
    with pytest.raises(DeckError) as exc:
        parse(BODY + "PQ 0\nXQ\nNX\n")
    assert str(exc.value) == (
        "PQ 0 requests a charge-density report this engine does not produce; "
        "PQ -1 (suppress) is the only form served"
    )


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


_K9AY_DECK = K9AY + "GE 0\nEX 0 4 1 0 1. 0.\nFR 0 1 0 0 14.\n{ld}XQ\nNX\n"

# The same structure with the reflection written out by hand — the twin the
# cell rule is gated against.  Tags 3 and 4 are the images of 1 and 2, so
# there is no symmetry at `GE` and every tag addresses its own segments.
_K9AY_TWIN = (
    "GW 1 8 0. 0. 8.5  5. 0. 1.5 1.5E-3\n"
    "GW 2 5 0. 0. 0.   5. 0. 1.5 1.5E-3\n"
    "GW 3 8 0. 0. 8.5 -5. 0. 1.5 1.5E-3\n"
    "GW 4 5 0. 0. 0.  -5. 0. 1.5 1.5E-3\n"
    "GE 0\nEX 0 4 1 0 1. 0.\nFR 0 1 0 0 14.\n{ld}XQ\nNX\n"
)

_OUT_OF_CELL_REFUSAL = (
    "LD addresses {n} segment{s} outside the GX/GR symmetric cell in force "
    "(the cell is segments 1-{cell} of {total}), which this engine does not "
    "serve (momwire#415): while the symmetry is live the matrix is the "
    "cell's, so NEC silently drops such a card rather than loading the copy "
    "— address the cell instead, where a load applies to every copy of it, "
    "or write the GX/GR out as explicit GW cards"
)


def test_a_load_inside_a_live_cell_lands_on_every_copy():
    """§#the-symmetric-cell: ``k9ay_orig``'s own card. Tag 2 is in the cell,
    so the 470 Ω lands on tag 2 AND on tag 4 — the image, which is also the
    driven wire, which is why the deck reads 960.79 Ω rather than 490.79 on
    the oracle (momwire#415)."""
    model = parse(_K9AY_DECK.format(ld="LD 4 2 1 1 470. 0.\n"))
    assert [wire for wire, _, _ in model.loads] == [1, 3]
    assert {spec for _, _, spec in model.loads} == {LoadSpec("fixed", r=470.0, x=0.0)}


def test_the_cell_rule_reproduces_the_hand_expanded_twin_exactly():
    """§#the-symmetric-cell, the gating property of momwire#415: a cell load
    is *identical* to writing one card per copy by hand.

    The issue states the transferable form — "each engine's cell-load result
    equals its own all-copies-loaded expansion exactly" — and momwire can
    gate it one level below a solve: the two decks produce the same wires,
    the same feed and the same load list, term for term, so no solver is
    needed to know the answers agree."""
    written = parse(_K9AY_DECK.format(ld="LD 4 2 1 1 470. 0.\n"))
    expanded = parse(_K9AY_TWIN.format(ld="LD 4 2 1 1 470. 0.\nLD 4 4 1 1 470. 0.\n"))
    assert written.wires == expanded.wires
    assert written.feeds == expanded.feeds
    assert written.loads == expanded.loads


def test_the_cell_rule_differs_from_the_naive_single_tag_twin():
    """§#the-symmetric-cell: the other half of the gate — putting the load
    only where the card wrote it is a DIFFERENT model, one load short, and
    the missing one sits on the driven wire (53 % on the oracle)."""
    written = parse(_K9AY_DECK.format(ld="LD 4 2 1 1 470. 0.\n"))
    naive = parse(_K9AY_TWIN.format(ld="LD 4 2 1 1 470. 0.\n"))
    assert written.wires == naive.wires
    assert len(naive.loads) == 1
    assert written.loads != naive.loads


def test_without_the_load_the_two_forms_already_agree():
    """§#the-symmetric-cell: delete the ``LD`` and the as-written deck and
    its hand-expanded twin are the same model — which is what isolates the
    cell rule to LOADING rather than to the geometry expansion."""
    written = parse(_K9AY_DECK.format(ld=""))
    expanded = parse(_K9AY_TWIN.format(ld=""))
    assert written.wires == expanded.wires
    assert written.feeds == expanded.feeds
    assert written.loads == expanded.loads == ()


def test_a_load_outside_a_live_cell_refuses_by_name():
    """§#the-symmetric-cell: NEC drops this card in silence and still echoes
    it in its ``DATA CARD`` list; this engine says so instead."""
    with pytest.raises(DeckError) as exc:
        parse(_K9AY_DECK.format(ld="LD 4 3 1 1 470. 0.\n"))
    assert str(exc.value) == _OUT_OF_CELL_REFUSAL.format(n=1, s="", cell=13, total=26)


def test_the_out_of_cell_refusal_counts_the_segments_it_names():
    """§#the-symmetric-cell: the count is the dropped segments, not the
    card's whole range — here all eight of tag 3."""
    with pytest.raises(DeckError) as exc:
        parse(_K9AY_DECK.format(ld="LD 5 3 0 0 5.8e7\n"))
    assert str(exc.value) == _OUT_OF_CELL_REFUSAL.format(n=8, s="s", cell=13, total=26)


def test_a_zero_valued_ld_under_a_live_symmetry_is_still_a_no_op():
    """§#the-symmetric-cell: a card byte-identical to omitting it is not
    refused for a rule it cannot trip — inside the cell or outside it."""
    assert parse(_K9AY_DECK.format(ld="LD 4 2 1 1 0. 0.\n")).loads == ()
    assert parse(_K9AY_DECK.format(ld="LD 4 3 1 1 0. 0.\n")).loads == ()


def test_a_deck_whose_symmetry_is_dead_at_ge_serves_its_ld():
    """§#gx--structure-reflection: the common corpus case (all but four of the
    36 decks that use these cards) — a feed wire or mast after the ``GX``
    collapses the symmetry, and ordinary per-tag addressing comes back."""
    model = parse(
        K9AY
        + "GW 9 1 0. 0. 0. 0. 0. 1. 1.5E-3\n"
        + "GE 0\nEX 0 4 1 0 1. 0.\nFR 0 1 0 0 14.\n"
        + "LD 4 2 1 1 470. 0.\nXQ\nNX\n"
    )
    assert len(model.loads) == 1
    assert model.loads[0][2] == LoadSpec("fixed", r=470.0, x=0.0)


def test_a_live_symmetry_with_no_ld_serves_in_full():
    """§#gx--structure-reflection: expansion is exact, so the two decks with
    live symmetry and no ``LD`` need nothing further."""
    model = parse(_K9AY_DECK.format(ld=""))
    assert len(model.wires) == 4
    assert sum(w.edge_elements[0] for w in model.wires) == 26


_GR_LIVE_DECK = (
    "GW 1 5 1. 0. 0. 2. 0. 0. .001\nGR 1 4\nGE 0\n"
    "EX 0 1 3 0 1. 0.\nFR 0 1 0 0 14.\n{ld}XQ\nNX\n"
)


def test_a_load_inside_a_live_gr_cell_lands_on_every_copy():
    """§#gr--cylindrical-structure-rotation: the cell rule reads
    ``structure.symmetry`` alone, so a ``GR``-declared cell replicates
    exactly as a ``GX``-declared one does — segment 3 of the cell becomes
    segment 3 of all four copies, at the same arclength on each."""
    model = parse(_GR_LIVE_DECK.format(ld="LD 4 1 3 3 50. 0.\n"))
    assert [wire for wire, _, _ in model.loads] == [0, 1, 2, 3]
    assert len({arclength for _, arclength, _ in model.loads}) == 1


def test_a_load_outside_a_live_gr_cell_refuses_by_name():
    """§#gr--cylindrical-structure-rotation: and the refusal is the one
    ``GX`` gets, from the same code path."""
    with pytest.raises(DeckError) as exc:
        parse(_GR_LIVE_DECK.format(ld="LD 4 2 3 3 50. 0.\n"))
    assert str(exc.value) == _OUT_OF_CELL_REFUSAL.format(n=1, s="", cell=5, total=20)


def test_a_deck_whose_gr_symmetry_is_dead_at_ge_serves_its_ld():
    """§#gr--cylindrical-structure-rotation: a feed wire or mast added
    after ``GR`` collapses the symmetry, same as it does after ``GX``, and
    ordinary per-tag addressing comes back."""
    model = parse(
        "GW 1 5 1. 0. 0. 2. 0. 0. .001\nGR 1 4\n"
        "GW 9 1 0. 0. 0. 0. 0. 1. .001\n"
        "GE 0\nEX 0 1 3 0 1. 0.\nFR 0 1 0 0 14.\n"
        "LD 4 1 3 3 50. 0.\nXQ\nNX\n"
    )
    assert len(model.loads) == 1
    assert model.loads[0][2] == LoadSpec("fixed", r=50.0, x=0.0)


# One cell wire of two segments and one copy of it: the smallest structure
# that can tell "replicated onto the copy" from "written on the copy".
_CELL_PAIR = (
    "GW 1 2 1. 0. 0. 2. 0. 0. .001\nGR 1 2\nGE 0\n"
    "EX 0 1 1 0 1. 0.\nFR 0 1 0 0 14.\n{ld}XQ\nNX\n"
)


def test_whole_structure_addressing_under_a_live_cell_does_not_double_stamp():
    """§#the-symmetric-cell: ``LD ... 0 0 0`` is the whole structure, which
    under a live symmetry IS cell plus copies — so the cell rule and
    whole-structure addressing name the same set (the oracle prints tag 0 and
    tag 1 identically, momwire#415), and the replication must not stamp the
    copies a second time."""
    model = parse(_CELL_PAIR.format(ld="LD 4 0 0 0 50. 0.\n"))
    assert [wire for wire, _, _ in model.loads] == [0, 0, 1, 1]


def test_a_global_segment_range_inside_the_cell_replicates_like_a_tag():
    """§#the-symmetric-cell: the rule is read off the SEGMENTS a card
    resolves to, not off how the tag field spelled them — so tag 0 segment 1
    and tag 1 segment 1 are the same card, exactly as the oracle's probe
    table shows."""
    by_structure = parse(_CELL_PAIR.format(ld="LD 4 0 1 1 50. 0.\n"))
    by_tag = parse(_CELL_PAIR.format(ld="LD 4 1 1 1 50. 0.\n"))
    assert by_structure.loads == by_tag.loads
    assert [wire for wire, _, _ in by_tag.loads] == [0, 1]


def test_replication_does_not_trip_the_doubled_load_refusal():
    """§#the-symmetric-cell: two cell cards on DIFFERENT segments replicate
    into four loads without colliding — the dedup set sees each copy's
    segment once."""
    model = parse(_CELL_PAIR.format(ld="LD 4 1 1 1 50. 0.\nLD 4 1 2 2 60. 0.\n"))
    assert len(model.loads) == 4


def test_two_cell_cards_on_one_segment_still_refuse():
    """§#ld--loading: replication widens the dedup set, it does not weaken
    it — a deck that genuinely double-loads a cell segment refuses as it
    always did."""
    with pytest.raises(DeckError) as exc:
        parse(_CELL_PAIR.format(ld="LD 4 1 1 1 50. 0.\nLD 0 1 1 1 10. 0. 0.\n"))
    assert str(exc.value) == (
        "LD on a segment that already carries a load is not supported by "
        "this engine — a second load on one segment is not merged"
    )


def test_the_expansion_limit_counts_the_card_as_written_not_the_replicas():
    """§#the-symmetric-cell: the ≤ 8-segment expansion limit is read against
    the range the deck TYPED, because NEC's own reader never sees the
    replicas as a range — they are stamped on afterwards, one segment at a
    time.  Eight typed segments over a two-copy cell serve as sixteen loads;
    a ninth typed segment refuses."""
    wide = "GW 1 8 1. 0. 0. 2. 0. 0. .001\nGR 1 2\nGE 0\nEX 0 1 1 0 1. 0.\n{ld}XQ\nNX\n"
    assert len(parse(wide.format(ld="LD 4 1 1 8 50. 0.\n")).loads) == 16
    with pytest.raises(DeckError) as exc:
        parse(wide.format(ld="LD 4 0 1 9 50. 0.\n"))
    assert str(exc.value) == (
        "LD over 9 segments is not supported by this engine — at most 8 "
        "segments expand into per-segment loads"
    )


def test_ld5_conductivity_inside_a_live_cell_reaches_every_copy():
    """§#the-symmetric-cell: a type-5 conductivity lands in the same matrix
    diagonal a lumped load does, so it is cell-scoped the same way."""
    model = parse(_CELL_PAIR.format(ld="LD 5 1 0 0 5.8e7\n"))
    assert [w.material.conductivity for w in model.wires] == [5.8e7, 5.8e7]


def test_ld5_whole_structure_conductivity_serves_under_a_live_cell():
    """§#the-symmetric-cell: the whole-structure form already names every
    copy, so it never reaches the cell rule at all."""
    model = parse(_CELL_PAIR.format(ld="LD 5 0 0 0 5.8e7\n"))
    assert [w.material.conductivity for w in model.wires] == [5.8e7, 5.8e7]


def test_is_under_a_live_cell_refuses_by_name():
    """§#the-symmetric-cell: ``IS`` is this dialect's own card, absent from
    NEC-2, so the cell rule that was measured for ``LD`` has no oracle here —
    it refuses rather than guessing."""
    with pytest.raises(DeckError) as exc:
        parse(_CELL_PAIR.format(ld="IS 0 1 0 0 2.3 0. .002\n"))
    assert str(exc.value) == (
        "IS while a GX/GR symmetric cell is in force is not supported by "
        "this engine (momwire#415): a sheath moves the matrix the way a load "
        "does, so the cell rule applies to it, but NEC-2 has no IS card to "
        "measure that rule against — write the GX/GR out as explicit GW cards"
    )


def test_is_under_a_dead_symmetry_serves():
    """§#the-symmetric-cell: the ``IS`` refusal is scoped to a LIVE cell, so
    a deck that collapses its symmetry insulates its wires as usual."""
    model = parse(
        "GW 1 2 1. 0. 0. 2. 0. 0. .001\nGR 1 2\n"
        "GW 9 1 0. 0. 3. 0. 0. 4. .001\nGE 0\n"
        "EX 0 1 1 0 1. 0.\nFR 0 1 0 0 14.\nIS 0 1 0 0 2.3 0. .002\nXQ\nNX\n"
    )
    assert model.wires[0].material.insulation_eps_r == 2.3


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
# #tl--transmission-line and #nt--two-port-network
# ---------------------------------------------------------------------------
#
# A two-wire body, because a network needs two places to attach and the
# one-wire BODY above can only give it two segments of the same wire (which is
# legal, and used below where the point is the addressing rather than the
# topology).

PAIR = (
    "GW 1 5 -0.5 0. 0. 0.5 0. 0. 1.E-3\n"
    "GW 2 5 -0.5 1. 0. 0.5 1. 0. 1.E-3\n"
    "GE 0\nEX 0 1 3 0 1.\nFR 0 1 0 0 14.\n"
)


def test_tl_and_nt_are_read_rather_than_refused_by_name():
    """§#tl--transmission-line, §#nt--two-port-network: the dialect's identity
    change.  Both cards used to refuse by name as out of an antenna-only
    grammar; they are part of it now."""
    model = parse(PAIR + "TL 1 3 2 3 450. 2.\nNT 1 3 2 3 0. .02 0. 0. 0. .02\nXQ\nNX\n")
    assert [c.kind for c in model.networks] == ["TL", "NT"]


def test_a_network_endpoint_resolves_the_way_a_feed_does():
    """§#tl--transmission-line: an endpoint is a ``(tag, seg)`` address like
    any other, and it lands on the SEGMENT CENTRE (§#addressing) — the same
    point ``EX 0 1 3`` drives, arclength for arclength."""
    model = parse(PAIR + "TL 1 3 2 3 450. 2.\nXQ\nNX\n")
    (card,) = model.networks
    feed_wire, feed_arc, _volts = model.feeds[0]
    assert card.end_a == (feed_wire, feed_arc)
    assert card.end_b == (1, feed_arc)


def test_a_network_card_keeps_its_nec_addressing_alongside_the_resolution():
    """§#nt--two-port-network: the ``NETWORK DATA`` printout is addressed in
    tags and segments, and a deck whose tags repeat cannot have them
    reconstructed from an arclength — so the card's own pair is recorded."""
    model = parse(PAIR + "NT 2 4 1 2 0. .02 0. 0. 0. .02\nXQ\nNX\n")
    (card,) = model.networks
    assert (card.address_a, card.address_b) == ((2, 4), (1, 2))
    assert card.end_a[0] == 1 and card.end_b[0] == 0


def test_the_payload_is_recorded_verbatim_and_uninterpreted():
    """§#tl--transmission-line: F1-F6 as the deck wrote them.  A crossed line
    is spelled with a NEGATIVE Z0 and the sign is kept — normalising it here
    would decide a semantics question this reader does not own."""
    model = parse(PAIR + "TL 1 3 2 3 -450. 2. 1.E-3 2.E-3 3.E-3 -4.E-3\nXQ\nNX\n")
    (card,) = model.networks
    assert card.payload == (-450.0, 2.0, 1.0e-3, 2.0e-3, 3.0e-3, -4.0e-3)


@pytest.mark.parametrize(
    "card,expected",
    [
        ("TL 1 3 2 3 450.", (450.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ("TL 1 3 2 3 450. 2.", (450.0, 2.0, 0.0, 0.0, 0.0, 0.0)),
        ("NT 1 3 2 3 .02", (0.02, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ],
)
def test_a_short_network_card_zero_fills(card, expected):
    """§#field-numbering: a field the deck did not write reads as 0, exactly
    as NEC's zero-filled card image does.  Measured on the oracle: an ``NT``
    with its six reals omitted entirely and the same card with six explicit
    zeros produce byte-identical printouts."""
    model = parse(PAIR + card + "\nXQ\nNX\n")
    assert model.networks[0].payload == expected


def test_an_all_zero_nt_is_read_and_not_dropped():
    """§#nt--two-port-network: the all-zero admittance matrix is NOT a no-op,
    which is the trap this pins.  An ``LD`` of zero value is dropped as one
    (§#ld--loading), and antennaknobs' importer skips an all-zero ``NT`` too —
    but the oracle does neither: it attaches a network of zero admittance,
    which OPEN-CIRCUITS both addressed segments.  Measured: the deck answers
    0.68923 - j4651.8 against the no-card control's 0.10161 + j514.86, with
    both connection-point currents collapsed to ~1e-20."""
    model = parse(PAIR + "NT 1 3 2 3 0. 0. 0. 0. 0. 0.\nXQ\nNX\n")
    (card,) = model.networks
    assert card.payload == (0.0,) * 6


# -- the by-value refusals ---------------------------------------------------


@pytest.mark.parametrize("mnemonic", ("TL", "NT"))
@pytest.mark.parametrize("seg", (0, -3))
def test_a_nonpositive_segment_number_refuses(mnemonic, seg):
    """§#fields-refused-by-value: NEC halts the whole run on this card, and
    the guard exists because ``locate`` would otherwise read segment 0 as
    segment 1 and attach a deck NEC refuses outright to the wrong place."""
    with pytest.raises(DeckError) as exc:
        parse(PAIR + f"{mnemonic} 1 3 2 {seg} 450. 2.\nXQ\nNX\n")
    assert str(exc.value) == (
        f"{mnemonic} addresses segment {seg} of tag 2, which is not a segment: "
        "NEC halts on this card with CHECK DATA, PARAMETER SPECIFYING SEGMENT "
        "POSITION IN A GROUP OF EQUAL TAGS MUST NOT BE ZERO — a network "
        "endpoint names one segment, so its segment number must be 1 or more"
    )


def test_the_segment_guard_reads_the_segment_field_alone():
    """§#fields-refused-by-value: it fires with a ZERO tag too, where the
    segment number is a global index rather than a position within a group of
    equal tags.  NEC's own check is on the segment field and runs before the
    tag is looked at, so a deck written ``NT 0 0 ...`` gets the same halt —
    measured (probes ``c_tag0_zero``, ``c_tag0_neg``)."""
    with pytest.raises(DeckError) as exc:
        parse(PAIR + "NT 0 0 0 8 .02 0. 0. 0. .02 0.\nXQ\nNX\n")
    assert "addresses segment 0 of the structure" in str(exc.value)


def test_a_zero_characteristic_impedance_refuses():
    """§#fields-refused-by-value: NEC aborts reading the deck on this card.
    A short ``TL`` reaches the same guard, and should — a card that names two
    endpoints and no line is not a transmission line."""
    expected = (
        "TL with a zero characteristic impedance is not a transmission line — "
        "NEC aborts reading the deck on this card; Z0 must be nonzero, and its "
        "SIGN, not its magnitude, is what selects a crossed line"
    )
    with pytest.raises(DeckError) as exc:
        parse(PAIR + "TL 1 3 2 3 0. 2.\nXQ\nNX\n")
    assert str(exc.value) == expected
    with pytest.raises(DeckError) as exc:
        parse(PAIR + "TL 1 3 2 3\nXQ\nNX\n")
    assert str(exc.value) == expected


def test_an_nt_of_zero_admittance_is_not_the_z0_refusal():
    """§#nt--two-port-network: the zero test is ``TL``'s alone.  ``NT``'s F1
    is an admittance, not an impedance, and zero is a legal value for it —
    the oracle runs the card."""
    assert len(parse(PAIR + "NT 1 3 2 3 0. 0. 0. 0. 0. 0.\nXQ\nNX\n").networks) == 1


# -- contiguity (§#network-contiguity) ---------------------------------------


def _contiguity_message(mnemonic: str, destroyer: str) -> str:
    return (
        f"{mnemonic} with an interposed {destroyer} card between it and an "
        f"earlier network card is not supported by this engine (momwire#456): "
        f"NEC silently DESTROYS every TL/NT read before such a card, so this "
        f"deck's earlier network cards would vanish with no diagnostic while "
        f"still being echoed in its DATA CARD list as read — keep a deck's "
        f"TL/NT cards contiguous (only PT, PQ and MP may sit between them)"
    )


@pytest.mark.parametrize(
    "destroyer,line",
    [
        ("LD", "LD 4 1 2 2 10. 0."),
        ("EX", "EX 0 2 3 0 1."),
        ("FR", "FR 0 1 0 0 14."),
        ("GN", "GN 1"),
        ("EK", "EK"),
        ("GD", "GD 0 0 0 0 5. 0.001"),
        ("XQ", "XQ"),
    ],
)
def test_an_interposed_card_of_the_destroy_class_refuses(destroyer, line):
    """§#network-contiguity: every member of the measured destroy class.

    Each row was run on the oracle as ``TL / <card> / NT`` and read back off
    the network printout block: in every one the ``TL`` is simply gone, with
    no diagnostic and the card still echoed in the ``DATA CARD`` list as read.
    Silence of that kind is what this dialect refuses rather than matches."""
    deck = (
        PAIR + f"TL 1 3 2 3 450. 2.\n{line}\nNT 1 3 2 3 0. .02 0. 0. 0. .02\nXQ\nNX\n"
    )
    with pytest.raises(DeckError) as exc:
        parse(deck)
    assert str(exc.value) == _contiguity_message("NT", destroyer)


@pytest.mark.parametrize("line", ("PT -1", "PQ -1", "MP 16 32"))
def test_the_print_control_cards_may_sit_between_network_cards(line):
    """§#network-contiguity: the transparent half, and equally measured — PT,
    PQ and MP change what a run REPORTS, not what it computes, and the oracle
    keeps both network cards across them.  Serving these matters: SimNEC
    appends ``MP`` on structure size alone."""
    model = parse(
        PAIR + f"TL 1 3 2 3 450. 2.\n{line}\nNT 1 3 2 3 0. .02 0. 0. 0. .02\nXQ\nNX\n"
    )
    assert [c.kind for c in model.networks] == ["TL", "NT"]


def test_the_contiguity_message_names_the_first_interposed_card():
    """§#network-contiguity: with several cards between the two networks it is
    the FIRST that is the deck's mistake, so that is the one named."""
    deck = (
        PAIR + "TL 1 3 2 3 450. 2.\nFR 0 1 0 0 14.\nGN 1\n"
        "NT 1 3 2 3 0. .02 0. 0. 0. .02\nXQ\nNX\n"
    )
    with pytest.raises(DeckError) as exc:
        parse(deck)
    assert str(exc.value) == _contiguity_message("NT", "FR")


def test_a_deck_with_one_network_card_never_meets_the_contiguity_rule():
    """§#network-contiguity: the rule is about a card DESTROYED by a later
    one, so a deck with a single network card is unaffected wherever the card
    sits and whatever follows it."""
    model = parse(PAIR + "TL 1 3 2 3 450. 2.\nGN 1\nFR 0 1 0 0 14.\nXQ\nNX\n")
    assert len(model.networks) == 1


# -- retention (§#network-retention) -----------------------------------------


def test_networks_are_retained_across_execute_groups():
    """§#network-retention: a contiguous network group persists to the end of
    the deck the way an ``EX`` set does — the oracle prints the same two rows
    in both groups' network blocks (probe ``persist_xq``).  So the model
    carries them once, deck-level, rather than per group."""
    model = parse(
        PAIR + "TL 1 3 2 3 450. 2.\nNT 1 3 2 3 0. .02 0. 0. 0. .02\n"
        "XQ\nFR 0 1 0 0 21.\nXQ\nNX\n"
    )
    assert len(model.groups) == 2
    assert [c.kind for c in model.networks] == ["TL", "NT"]
    assert [c.first_group for c in model.networks] == [0, 0]


def test_a_network_card_arms_the_next_execute_card_without_refilling():
    """§#arming, §#network-retention: a bare ``XQ`` after a lone network card
    is a real second run, so ``TL``/``NT`` arm — but they arm the way ``EX``
    does, moving the composition on top of the operator rather than the
    operator, so the group is not marked refilled.

    Measured (probe ``c_xq_only``): the oracle prints two ANTENNA INPUT
    PARAMETERS blocks and exactly one each of MATRIX TIMING, FREQUENCY and
    STRUCTURE IMPEDANCE LOADING."""
    model = parse(PAIR + "XQ\nTL 1 3 2 3 450. 2.\nXQ\nNX\n")
    assert model.groups[1] is not None
    assert not model.groups[1].refilled
    assert not model.groups[1].refilled_partial


def test_a_network_stated_after_an_execute_card_records_where_it_starts():
    """§#network-retention: retention runs FORWARD from the card, not
    backward.  A deck that runs the bare antenna and then attaches a network
    is answered without one in the first group, which the oracle prints
    (probe ``c_after_xq``: group 1's network block carries only the earlier
    card).  ``first_group`` is the whole of that scoping."""
    model = parse(PAIR + "XQ\nTL 1 3 2 3 450. 2.\nXQ\nNX\n")
    (card,) = model.networks
    assert card.first_group == 1


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
        "GA", "GH", "GC", "GF", "SY",
        "SP", "SM", "SC", "KH", "CP", "PL", "WG", "ZO",
    }  # fmt: skip
    # TL and NT left this set in momwire#456 phase C: the dialect reads them
    # now, and what they refuse is by VALUE (§#fields-refused-by-value) —
    # nonpositive segment addressing, a zero TL characteristic impedance, and
    # the non-contiguous destroy pattern.
    assert not {"TL", "NT"} & set(_REFUSED_BY_NAME)


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
