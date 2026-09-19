"""momwire#1134: a gyrator-driven source reports at the segment it DRIVES.

NEC-2 has no current-source ``EX`` card, so EZNEC saving a current-driven
model to that dialect writes a GYRATOR instead: a phantom wire parked ~100
lambda away whose segments are circuit nodes, an ``EX 0`` on one of them, and
an ``NT`` with ``Y11 = Y22 = 0``, ``Y12 = Y21 = jB`` tying it to the segment it
really drives. This seam solved that circuit correctly and then read the
driving point off the PHANTOM, where a gyrator's inversion makes it 1/Z.

Six gates:

1. the controlled pair - one antenna, one drive point, one delivered current,
   spelled natively and through a gyrator in the SAME dialect. Pinned as
   agreement between the two impedances, never as a ratio;
2. the reporting location, on Dan AC6LA's Cardioid as EZNEC wrote it: tags 1
   and 2 with the segment each gyrator drives, not tag 3 at 100 lambda;
3. the sign, adversarially - ``-Y12*V`` against ``+Y12*V``;
4. detection stays narrow: nothing in the 80-deck corpus is a gyrator, and
   the two shapes that wear the same clothes (a source behind a TRANSFORMER,
   a source behind a LINE) are left alone;
5. the constants that separate the phantom from the structure, re-measured on
   the corpus rather than taken on trust;
6. the one guarantee the detector leans on without checking - that a tag
   names exactly one wire.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

import momwire.eznec._serve as _serve
from momwire.deck._cards import DeckError
from momwire.deck._nec5 import parse_nec5
from momwire.eznec import serve
from momwire.eznec._serve import (
    SPEED_OF_LIGHT_MHZ_M,
    _gyrator_drives,
    _phantom_tags,
)

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_gyrator_1134"
CORPUS = Path(__file__).parent / "fixtures" / "eznec"

# The currents the NEC-4.2 twin's `EX 6` cards ask for, card for card
# (Cardioidmodnec4.nec). They are the OTHER dialect's statement of the same
# drive, so they are what the NEC-2 deck's gyrators have to deliver.
CARDIOID_EX6 = (1.414214 + 0j, -1.414214j)


def run(name: str):
    return serve(parse_nec5((FIXTURES / name).read_text()))


def corpus_decks():
    manifest = json.loads((CORPUS / "manifest.json").read_text())
    for entry in manifest["captures"]:
        text = (CORPUS / entry["deck"]).read_bytes().decode("latin-1")
        yield entry["id"], parse_nec5(text)


def wavelength_of(deck) -> float:
    return SPEED_OF_LIGHT_MHZ_M / deck.frequency_mhz


# --------------------------------------------------------------------------
# gate 1 - the controlled pair
# --------------------------------------------------------------------------


def test_a_gyrator_drive_answers_the_native_deck_s_driving_point():
    """The decisive oracle: one antenna, two spellings of one drive.

    The three Cardioid dialects cannot do this job. NEC-5 addresses knots and
    NEC-2 segment centres, so the same model's feeds AND loads land half a
    segment apart in the two files and the served impedances sit 7-9 % apart
    even when both are read correctly - "NEC-2 now equals NEC-5" would be a
    gate on a coincidence. So the pair is built instead: the same 10-segment
    dipole, the same node 5, the same 1.414214 A, written once as a native
    ``EX 4`` and once as an ``EX 0`` of 1.414214j V through an ``NT`` with
    Y12 = j.

    Pinned as AGREEMENT. A test that divided one impedance by the other, or
    compared one to the other's reciprocal, would pass just as happily with
    the readout still on the phantom - that is the shape AK#1595 shipped.
    """
    (native,) = run("native_current.nec").sources
    (gyrator,) = run("gyrator_current.nec").sources

    assert gyrator.tag == native.tag == 1
    assert gyrator.segment == native.segment == 5
    assert gyrator.end_index == native.end_index == 1

    relative = abs(gyrator.impedance - native.impedance) / abs(native.impedance)
    assert relative < 1e-9, (
        f"native {native.impedance!r} vs gyrator {gyrator.impedance!r}, "
        f"relative {relative:.3e}"
    )
    # Measured 1.6e-13 (README). The 1e-9 bar is three decades of headroom
    # over the phantom's own coupling at 173 lambda and the three extra
    # unknowns it adds, which is the only thing that differs between the two
    # solves - and still six decades under the 2.4e-7 at which the UNFIXED
    # readout matches 1/Z, so no reciprocal can creep through it.
    assert native.impedance.real > 0.0


def test_the_unfixed_readout_really_did_report_the_reciprocal():
    """The defect, reproduced by disarming exactly one condition.

    Raising the susceptance floor above the card's ``|B| = 1`` turns the
    detector off and nothing else, so what comes back is the pre-#1134 row
    verbatim. It has to be 1/Z, or gate 1 above is measuring something other
    than the fix.
    """
    (native,) = run("native_current.nec").sources
    floor = _serve._GYRATOR_MIN_B
    _serve._GYRATOR_MIN_B = 1e9
    try:
        (unfixed,) = run("gyrator_current.nec").sources
    finally:
        _serve._GYRATOR_MIN_B = floor

    assert unfixed.tag == 2, "the unfixed row sits on the phantom wire"
    reciprocal = 1.0 / native.impedance
    assert abs(unfixed.impedance - reciprocal) / abs(reciprocal) < 1e-5


# --------------------------------------------------------------------------
# gate 2 - where the row reports
# --------------------------------------------------------------------------


def test_the_cardioid_reports_at_its_real_wires_not_at_the_phantom():
    """Dan AC6LA's Cardioid, as EZNEC wrote it to NEC-2.

    Tag 3 is the phantom, 100 lambda away; tags 1 and 2 are the verticals.
    Both the TAG and the SEGMENT are asserted, because a row that named the
    right wire and the wrong node would still be a wrong driving point.
    """
    rows = run("Cardioidmodnec2.nec").sources
    assert len(rows) == 2
    assert [row.tag for row in rows] == [1, 2]
    # Global element numbers: tag 1 starts at 1 and tag 2 at 7, and each
    # gyrator's real end is that wire's node 1.
    assert [row.segment for row in rows] == [1, 7]
    assert [row.end_index for row in rows] == [1, 1]
    # ... and the numbers are the antenna's, not the reciprocal of them.
    assert rows[0].impedance.real == pytest.approx(36.62353145, rel=1e-6)
    assert rows[0].impedance.imag == pytest.approx(-20.05467077, rel=1e-6)
    assert rows[1].impedance.real == pytest.approx(69.07445964, rel=1e-6)
    assert rows[1].impedance.imag == pytest.approx(21.15496860, rel=1e-6)


def test_the_gyrator_row_is_the_connection_point_the_solve_already_carried():
    """No second solve, and no second opinion.

    The real node is an ``NT`` endpoint, so the reducer already solved it and
    :class:`~momwire.eznec._serve._PortState` already carried it. The moved
    row must therefore be the STRUCTURE EXCITATION DATA row at that same site,
    to the round-off the restore rule removes from the current - which is what
    says this change touched the readout and not the circuit.
    """
    data = run("Cardioidmodnec2.nec")
    by_site = {(row.tag, row.segment): row for row in data.network_excitation}
    for row in data.sources:
        point = by_site[(row.tag, row.segment)]
        assert row.voltage == point.voltage
        assert abs(row.current - point.current) < 1e-15
        assert abs(row.impedance - point.impedance) < 1e-12 * abs(point.impedance)


# --------------------------------------------------------------------------
# gate 3 - the sign
# --------------------------------------------------------------------------


def test_the_delivered_current_is_the_ex6_phasor_in_both_parts():
    """``I = -Y12*V``, and a flip cannot hide.

    A magnitude-only check catches nothing: |-I| = |I|. A single-phase check
    catches nothing either - flipping the sign of one source's current flips
    the sign of the voltage the solve answers with it, so V/I is unmoved, and
    input power and a dBi pattern are invariant with it. That is why the
    Cardioid is the deck for this: its two ``EX 0`` cards carry DIFFERENT
    phases (1.414214j and 1.414214), so the two forced currents come out
    1.414214 and -1.414214j - a real number and a negative imaginary one. No
    single global convention maps that pair onto its own negation, so the
    assertion below is on the two currents SEPARATELY and in both parts.

    The right-hand side is the NEC-4.2 twin's own ``EX 6`` fields, which is an
    oracle from outside this engine: EZNEC wrote the two files from one model,
    and the dialect that HAS a current-source card says in plain numbers what
    the dialect that does not is asking for.
    """
    rows = run("Cardioidmodnec2.nec").sources
    for row, wanted in zip(rows, CARDIOID_EX6, strict=True):
        assert row.current.real == pytest.approx(wanted.real, abs=1e-12)
        assert row.current.imag == pytest.approx(wanted.imag, abs=1e-12)


def test_flipping_the_sign_moves_the_controlled_pair_off_its_native_deck():
    """The adversary, run rather than imagined.

    ``+Y12*V`` negates the reported current while leaving the solved voltage
    alone, so the driving point comes back as -Z: a NEGATIVE resistance on a
    passive dipole. Gate 1 would fail on it, and this test says so by
    constructing the flipped row directly rather than trusting that it would.
    """
    (native,) = run("native_current.nec").sources
    (gyrator,) = run("gyrator_current.nec").sources
    flipped = gyrator.voltage / (-gyrator.current)

    assert flipped.real < 0.0 < native.impedance.real
    relative = abs(flipped - native.impedance) / abs(native.impedance)
    assert relative > 1.0


def test_the_forced_current_is_restored_from_the_card_not_read_back():
    """Whatever a card SETS is a boundary condition (the U1 restore rule).

    A gyrator carries a set voltage to a set current, so the current is exact
    going in and reading it back out of the solve only adds the round-off that
    decides the sign of a zero - 4.9e-17 and 1.9e-17 on the Cardioid's two
    ports, the same residue an ``EX 4`` row restores away. The restored value
    must be EXACTLY ``-Y12*V`` and must still agree with the solve.
    """
    deck = parse_nec5((FIXTURES / "Cardioidmodnec2.nec").read_text())
    gyrators = _gyrator_drives(deck, wavelength_of(deck))
    assert len(gyrators) == 2
    for source in deck.sources:
        (card,) = [n for n in deck.networks if source.at in (n.end_a, n.end_b)]
        assert gyrators[source.at].current == -card.y12 * source.drive

    rows = run("Cardioidmodnec2.nec").sources
    solved = {
        (row.tag, row.segment): row.current
        for row in run("Cardioidmodnec2.nec").network_excitation
    }
    for row in rows:
        # Exact, not approximate: the restore is what removes the residue.
        assert row.current.real in (0.0, 1.414214, -1.414214) or row.current.imag in (
            0.0,
            1.414214,
            -1.414214,
        )
        assert abs(row.current - solved[(row.tag, row.segment)]) < 1e-15


# --------------------------------------------------------------------------
# gate 4 - detection stays narrow
# --------------------------------------------------------------------------


def test_no_committed_capture_carries_a_gyrator():
    """Zero churn, stated structurally rather than by re-blessing printouts.

    23 of the 80 committed decks park EZNEC's phantom wire, and every ``NT``
    any of them writes is an L-network or a transformer with a NONZERO
    diagonal. So no captured deck reaches the collapse, and the 75 committed
    printouts cannot move. If this test ever fails, the detector widened - the
    fix is the detector, not the printout.
    """
    parked = 0
    networks = 0
    for cid, deck in corpus_decks():
        lam = wavelength_of(deck)
        if _phantom_tags(deck, lam):
            parked += 1
        for net in deck.networks:
            networks += 1
            assert net.y11 or net.y22, f"{cid}: an NT with a zero diagonal"
        assert not _gyrator_drives(deck, lam), f"{cid}: collapsed a corpus drive"
    assert parked == 23
    assert networks == 17


def test_a_source_behind_a_transformer_or_a_line_is_left_alone():
    """The two shapes that wear the same clothes, from the corpus itself.

    0000 drives its phantom node with an ``EX 4`` and hangs an L-network
    (``NT 3,1,3,2``, all six fields nonzero) plus two ``TL`` off the same
    wire; 0027 drives its phantom node and reaches the antenna through two
    ``TL`` alone. In both, the source-side impedance the row reports is
    precisely what the operator asked for - that is the point of a source
    behind a matching network - so moving the row would be the defect rather
    than the fix.
    """
    for cid in ("0000", "0027"):
        deck = dict(corpus_decks())[cid]
        lam = wavelength_of(deck)
        parked = _phantom_tags(deck, lam)
        assert parked, f"{cid} parks a phantom wire"
        assert any(source.at.tag in parked for source in deck.sources)
        assert not _gyrator_drives(deck, lam)


@pytest.mark.parametrize(
    ("what", "edit"),
    [
        # A lossy line, or anything else with insertion loss: nonzero diagonal.
        (
            "a nonzero diagonal",
            lambda t: t.replace("0.,0.,0.,1.,0.,0.", "1.,0.,0.,1.,0.,0."),
        ),
        # A transformer: an all-real Y, no reactive transfer at all.
        (
            "an all-real Y12",
            lambda t: t.replace("0.,0.,0.,1.,0.,0.", "0.,0.,1.,0.,0.,0."),
        ),
        # A set CURRENT on the phantom node: 0000 and 0027's shape, a source
        # behind a matching network rather than a gyrator's input.
        (
            "an EX 4 on the phantom",
            lambda t: t.replace("EX 0,2,2,0,0.,1.414214", "EX 4,2,2,0,0.,1.414214"),
        ),
        # Both ends on real geometry: a component of the model, not a source.
        (
            "both ends real",
            lambda t: t.replace("NT 2,2,1,5", "NT 1,2,1,5").replace(
                "EX 0,2,2,0,0.,1.414214", "EX 0,1,2,0,0.,1.414214"
            ),
        ),
        # A second two-port on the phantom node: no longer the bare idiom.
        (
            "a second card on the phantom node",
            lambda t: t.replace("PQ 0", "TL 2,2,2,3,50.,1.,0.,0.,0.,0.\nPQ 0"),
        ),
    ],
)
def test_one_condition_at_a_time_turns_the_collapse_off(what, edit):
    """Each condition in the list is load-bearing, checked by breaking it.

    The controlled pair's deck B with exactly one field changed. Every edit
    below keeps a phantom wire, an EX and an NT card - so what refuses the
    collapse is the named condition and not the shape falling apart.
    """
    text = (FIXTURES / "gyrator_current.nec").read_text()
    edited = edit(text)
    assert edited != text, f"{what}: the edit matched nothing"
    deck = parse_nec5(edited)
    assert not _gyrator_drives(deck, wavelength_of(deck)), what


def test_a_zero_emf_is_a_datum_pin_and_no_deck_can_write_one_here():
    """The sixth condition, and the one this dialect reaches differently.

    antennaknobs refuses a zero-EMF source because a ``Driven`` of 0 V is the
    idiom's datum pin rather than a drive. The condition is in
    :func:`_gyrator_drives` for the same reason - but no DECK can present it
    to this seam, because ``parse_nec5`` normalises a zero-volt ``EX 0`` to
    1 V before the deck exists (momwire#1041: NEC-5 x13 drives one at 1 V and
    prints the 1 V impedance). Both halves are asserted here, so the guard is
    not mistaken for something a deck exercises.
    """
    text = (FIXTURES / "gyrator_current.nec").read_text()
    pinned = parse_nec5(text.replace("EX 0,2,2,0,0.,1.414214", "EX 0,2,2,0,0.,0."))
    (written_zero,) = pinned.sources
    assert written_zero.drive == 1 + 0j, "the dialect's own zero-volt rule"

    deck = parse_nec5(text)
    (source,) = deck.sources
    datum = replace(deck, sources=(replace(source, drive=0j),))
    assert not _gyrator_drives(datum, wavelength_of(datum))
    assert _gyrator_drives(deck, wavelength_of(deck)), "the unedited deck still does"


# --------------------------------------------------------------------------
# gate 5 - the constants, re-measured
# --------------------------------------------------------------------------


def test_the_phantom_thresholds_sit_in_a_three_decade_gap():
    """Where ``_PHANTOM_*`` came from, measured here rather than asserted.

    Two populations over the 80 committed decks. The wires EZNEC parks stand
    173.01-173.50 lambda clear of everything else and are 0.0035-0.0087 lambda
    long; the other 286 wires in the corpus stand at most 0.2625 lambda clear.
    The clearance line at 10 lambda sits 38x above the widest gap any real
    structure opens and 17x below the closest parked wire.
    """
    parked, structure = [], []
    for _cid, deck in corpus_decks():
        lam = wavelength_of(deck)
        if len(deck.wires) < 2:
            continue
        ends = [(tuple(w.end1), tuple(w.end2)) for w in deck.wires]
        for index, (a, b) in enumerate(ends):
            extent = math.dist(a, b) / lam
            clearance = (
                min(
                    math.dist(here, there)
                    for other, pair in enumerate(ends)
                    if other != index
                    for here in (a, b)
                    for there in pair
                )
                / lam
            )
            (parked if clearance > 1.0 else structure).append((extent, clearance))

    assert len(parked) == 23
    assert len(structure) == 286
    assert min(c for _e, c in parked) > _serve._PHANTOM_CLEARANCE_LAMBDA * 17.0
    assert max(c for _e, c in structure) < _serve._PHANTOM_CLEARANCE_LAMBDA / 38.0
    assert max(e for e, _c in parked) < _serve._PHANTOM_EXTENT_LAMBDA / 5.0
    assert min(c / e for e, c in parked) > _serve._PHANTOM_CLEARANCE_EXTENTS


def test_the_extent_gate_keeps_a_remote_antenna_out_of_the_circuit():
    """A genuinely remote ELEMENT is geometry, however far away it is.

    Clearance alone would swallow a coupling study's far element whole. Deck B
    with its phantom stretched to a quarter wave - still 173 lambda off, still
    the only thing an NT and an EX touch - must not be read as a circuit node.
    """
    text = (FIXTURES / "gyrator_current.nec").read_text()
    stretched = text.replace(
        "GW 2,4,999.98,999.98,999.98,1000.,1000.,1000.,1.0000E-4",
        "GW 2,4,999.98,999.98,999.98,1000.,1000.,1002.5,1.0000E-4",
    )
    assert stretched != text
    deck = parse_nec5(stretched)
    assert not _phantom_tags(deck, wavelength_of(deck))
    assert not _gyrator_drives(deck, wavelength_of(deck))


def test_the_phantom_set_is_tags_because_a_tag_names_one_wire():
    """What lets :func:`_phantom_tags` answer in TAGS rather than wire indices.

    A tag shared between a parked wire and a real one would put a real node in
    the phantom set, because ``Structure.index_of`` resolves a tag to the
    FIRST wire carrying it. Nothing in the detector guards against that, and
    nothing needs to: the dialect refuses a repeated tag at the parse. That
    guarantee is what this pins, so a later relaxation of it lands here rather
    than silently in a moved drive row.
    """
    text = (FIXTURES / "gyrator_current.nec").read_text()
    shared = text.replace("GE 0,-1", "GW 2,4,0.,1.,-0.5,0.,1.,0.5,.001\nGE 0,-1")
    assert shared != text
    with pytest.raises(DeckError, match="declares tag 2 a second time"):
        parse_nec5(shared)
