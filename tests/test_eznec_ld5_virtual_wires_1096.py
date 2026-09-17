"""A whole-structure ``LD 5`` that stops before a VIRTUAL wire -- momwire#1096.

EZNEC Pro/2+ 7.0.4's NEC-5 export of an OCF dipole with a transformer writes
``LD 5,0,1,271`` on a 273-segment deck: the conductivity spans the three REAL
wires and stops before wire 4, the two-segment virtual anchor the source and
network hang on. "Whole structure" in that writer means every CONDUCTOR. The
tag-0 rule from momwire#1082/#1086 had only single-wire decks to learn from,
where 1..N and the total coincide, and refused this span by name.

Gates: the span is accepted when it is aligned to wire boundaries and the
conductivity lands on exactly those wires; a span that splits a wire still
refuses; the loading-table row prints the range AS WRITTEN, byte-identical
to our licensed materials on a small deck of our own in the same shape; and
``razor-nec5`` on that deck agrees with the licensed binary's input impedance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from momwire.deck import DeckError
from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _serve
from momwire.eznec._shell import render
from test_eznec_ld5_conductivity_1082 import _loading_table_block

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_ld5_virtual_1096"
DECK = (FIXTURES / "synthetic_ld5_virtual.nec").read_text()
# The licensed binary's ANTENNA INPUT PARAMETERS row for this deck (tag 4,
# segment 64, end 1 -- the virtual wire the EX 4 drives through the NT).
LICENSED_Z = complex(22.469, -19.006)


def test_a_span_that_stops_before_the_virtual_wire_parses_onto_the_real_wires():
    deck = parse_nec5(DECK)
    assert [(w.tag, w.segment_count) for w in deck.wires] == [
        (1, 20),
        (2, 3),
        (3, 40),
        (4, 2),
    ]
    assert dict(deck.wire_conductivity) == {1: 5.7471e7, 2: 5.7471e7, 3: 5.7471e7}


@pytest.mark.parametrize("span", ["1,62", "2,63", "1,64"])
def test_a_span_that_splits_a_wire_still_refuses_by_name(span):
    text = DECK.replace("LD 5,0,1,63,", f"LD 5,0,{span},")
    assert text != DECK
    with pytest.raises(DeckError, match="wire boundaries"):
        parse_nec5(text)


def test_a_run_of_whole_wires_in_the_middle_is_a_run_of_whole_wires():
    """Not only a prefix: 21..63 is wires 2 and 3 exactly."""
    deck = parse_nec5(DECK.replace("LD 5,0,1,63,", "LD 5,0,21,63,"))
    assert set(deck.wire_conductivity) == {2, 3}


def test_the_loading_row_prints_the_span_as_written_byte_for_byte():
    """Verified against our licensed materials: ITAG blank, FROM 1, THRU 63
    -- the card's own range, not the wire count it resolves to."""
    oracle_lines = (FIXTURES / "synthetic_ld5_virtual.out").read_text().splitlines()
    rendered_lines = render(DECK).splitlines()
    assert _loading_table_block(rendered_lines) == _loading_table_block(oracle_lines)


def test_razor_nec5_agrees_with_the_licensed_binary_on_the_deck():
    """The formulation twin on the same deck, the conductivity on the real
    wires only. 1 % is looser than the twin's usual bar, because the drive
    reaches the antenna through the NT and the virtual wire, where the two
    formulations place the gap differently by one knot."""
    z = _serve.serve(parse_nec5(DECK), basis="razor-nec5").sources[0].impedance
    assert abs(z - LICENSED_Z) / abs(LICENSED_Z) < 1e-2, z
