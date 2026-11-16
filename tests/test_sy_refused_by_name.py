"""SY (4nec2's symbolic variables) is refused by name in every dialect.

Its fields are assignments (`SY segs=10`, `SY len=3.25, rad=0.0015`), not
numbers, so converting them raised NON-NUMERICAL CHARACTER IN FIELD ahead of
the sentence that says why the deck is refused. The nec2 and nec4 dialects
check refusals by name before tokenizing (#486); the nec5 dialect tokenized
first. `parse_card` now leaves an SY card's fields unconverted, as it does a
comment's, and the C++ deck reader does the same, so the sentence wins on
every route; the C++ printout core is held to these bytes by
`test_razor_core_standalone`'s `test_sy_decks_refuse_by_name_as_python_does`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from momwire.deck import _nec2, _nec4, _nec5
from momwire.deck._cards import DeckError, parse_card
from momwire.dropin.nec5 import _nec4 as _nec4_slot
from momwire.dropin.nec5 import _shell

DECKS = sorted(
    (Path(__file__).resolve().parent / "fixtures" / "sy_refused_by_name").glob("*.nec")
)
NEC5 = "SY (4nec2 symbolic variables) is not part of this engine's nec5 dialect"
NEC2 = "SY (4nec2 symbolic variables) is not part of this engine's nec2 dialect"
NEC4 = "SY (4nec2 symbolic variables) is not part of this engine's nec4 dialect"


def test_the_fixtures_are_there():
    assert [p.name for p in DECKS] == [
        "sy_one_assignment.nec",
        "sy_two_assignments.nec",
    ]


@pytest.mark.parametrize("line", ["SY segs=10", "SY len=3.25, rad=0.0015", "sy a=1"])
def test_an_sy_card_keeps_its_fields_unconverted(line):
    card = parse_card(line)
    assert (card.mnemonic, card.values, card.raw) == ("SY", (), line)


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.stem)
@pytest.mark.parametrize(
    ("parse", "sentence"),
    [(_nec5.parse_nec5, NEC5), (_nec2.parse_nec2, NEC2), (_nec4.parse_nec4, NEC4)],
    ids=["nec5", "nec2", "nec4"],
)
def test_every_dialect_refuses_sy_by_name(deck, parse, sentence):
    with pytest.raises(DeckError) as info:
        parse(deck.read_text())
    assert str(info.value) == sentence


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.stem)
def test_the_nec5_printout_names_sy(deck):
    out = _shell.render(deck.read_text(), basis="razor-2p")
    assert f" ***** NEC ERROR - {NEC5}" in out.splitlines()
    assert "NON-NUMERICAL" not in out


@pytest.mark.parametrize("deck", DECKS, ids=lambda p: p.stem)
def test_the_nec42_printout_names_its_own_dialect(deck):
    """The NEC-4.2 slot (`momwire-nec42-sinusoidal`, `momwire-nec42-bspline`)
    reads the nec4 dialect, and its SY sentence says so rather than naming
    the nec2 dialect it inherits its other refusals from (census 11-15). The
    C++ reader's copy is held to it by test_nec4_deck_core's
    `refused_by_name_sy` case."""
    out = _nec4_slot.render(deck.read_text(), basis="sinusoidal")
    assert f" ***** NEC ERROR - {NEC4}" in out.splitlines()
    assert "nec2 dialect" not in out
