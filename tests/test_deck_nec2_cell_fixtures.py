"""``k9ay_orig`` and ``1MHz_tower``: the symmetric cell's named regressions.

momwire#415 asked for these two decks as fixtures "with their oracle numbers",
and they are the two decks in the whole 36-deck ``GX``/``GR`` slice of the
xnec2c corpus that reach ``GE`` with a symmetric cell still live AND carry an
``LD`` — the only place NEC's cell rule can move an answer.  The six decks
under ``tests/fixtures/nec2_symmetry/`` are those two trimmed to the cards that
matter, plus, for each, the twin the rule is measured against.

``tests/test_deck_nec2.py`` gates the rule at the level of the MODEL — same
wires, same feed, same load list.  This module gates it one level up, at the
answer, which is the form #415 states: "each engine's cell-load result equals
its own all-copies-loaded expansion exactly".

The oracle numbers below are nec2c 5b4az.ae6ty.1.17 (the build SimNEC ships,
the same one behind ``tests/fixtures/nec_portal``), run on these exact deck
bodies on 2026-08-19.  They are literals here rather than committed printouts:
what they are wanted for is three identities, not a table.

``k9ay_orig`` — free space, 1.8 MHz, driven on tag 4 (an image), 470 ohms on
tag 2 (in the cell):

    ====================  =====================  ==========================
    deck                  nec2c                  momwire
    ====================  =====================  ==========================
    k9ay_orig             957.18 + j498.15       960.2857 + j494.0923
    k9ay_orig_expanded    957.18 + j498.15       960.2857 + j494.0923
    k9ay_orig_naive       487.18 + j498.15       490.2857 + j494.0923
    (no LD at all)          0.10161 + j514.86      0.10296 + j514.9059
    ====================  =====================  ==========================

Three things are true of that table on BOTH engines and are asserted below:
the cell form and the all-copies-loaded expansion agree exactly; the naive
single-tag expansion sits **exactly 470 ohms** lower in resistance and
identical in reactance, because the load the cell rule adds lands on the driven
wire itself; and deleting the ``LD`` makes all forms agree, which is what
isolates the whole effect to loading rather than to the geometry expansion.
The two engines are 0.33 % apart on the loaded deck and 0.19 % apart on the
unloaded one — ordinary cross-basis agreement, and the reason the identities
rather than the absolute numbers are what is pinned.

``1MHz_tower`` — perfect ground, 1 MHz, four ``GR`` copies of a leg assembly,
85 uH from each leg to ground, fed at one leg:

    =====================  ====================  ==========================
    deck                   nec2c                 momwire
    =====================  ====================  ==========================
    1MHz_tower              50.61 - j1.6296      29.7616 + j138.4419
    1MHz_tower_dropped      50.61 - j1.6296      29.7616 + j138.4419
    1MHz_tower_expanded     50.61 - j1.6295      29.7616 + j138.4419
    (no LD at all)           0.001692 + j12.007   0.0018139 + j12.0296
    =====================  ====================  ==========================

``1MHz_tower_dropped`` is the corpus deck's own ``LD`` set — four cards, one
per leg, three of them addressing copies OUTSIDE the cell.  Its oracle row is
the headline of this file: NEC drops those three in silence and then replicates
the cell's load onto the same three segments, so the printout is
``1MHz_tower``'s to every digit.

Unit 3 refused that form, on the ground that a silent drop is a defect worth
naming.  momwire#471 serves it instead, and the middle column above is why:
the three dropped cards say exactly what the replication already said, so the
deck MEANS what NEC computes and there is nothing left to warn about.  The
equality is checked per card rather than assumed — a copy card carrying a
DIFFERENT value still refuses, because that is the case where NEC's silence
loses the author's intent.  All three properties are asserted below.

The two engines are far apart on the LOADED tower — the unloaded structure
agrees to 0.2 %, and each leg's +j534 ohm inductor drives the four-leg
assembly to a resonance whose printed impedance is a small residual between
large cancelling terms (the class ``test_portal_differential.py`` documents as
(a)).  That is a basis question and not a #415 one, so the oracle bar here is
asserted on the UNLOADED control, where the ``GR`` expansion is what is being
measured, and the loaded rows are gated against momwire's own twin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from momwire.deck import DeckError, parse
from momwire.deck._nec2_geometry import Symmetry
from momwire.portal._portal import DeckSolver, parse_deck

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "nec2_symmetry"

NAMES = (
    "1MHz_tower",
    "1MHz_tower_dropped",
    "1MHz_tower_expanded",
    "k9ay_orig",
    "k9ay_orig_expanded",
    "k9ay_orig_naive",
)

# The LD cards each fixture writes, so a control can be built by deleting them
# from the committed body rather than by committing a fourth near-duplicate.
_LD_LINES = {
    "k9ay_orig": ("LD 4 2 1 1 470. 0.\n",),
    "k9ay_orig_expanded": ("LD 4 2 1 1 470. 0.\n", "LD 4 4 1 1 470. 0.\n"),
    "1MHz_tower": ("LD 0 3 1 1 0. 8.5E-5 0.\n",),
    "1MHz_tower_expanded": tuple(
        f"LD 0 {tag} 1 1 0. 8.5E-5 0.\n" for tag in (3, 6, 9, 12)
    ),
}


def body(name: str) -> str:
    return (FIXTURE_DIR / f"{name}.deck").read_text().split("\nNX")[0]


def unloaded(name: str) -> str:
    """The fixture with every ``LD`` card deleted — the control that says the
    geometry expansion, not the loading, is what the two forms share."""
    text = body(name)
    for line in _LD_LINES[name]:
        text = text.replace(line, "")
    assert "\nLD" not in text
    return text


_SOLVED: dict[str, complex] = {}


def driving_point(text: str) -> complex:
    """``V / I`` at the deck's one source — the ANTENNA INPUT PARAMETERS
    impedance, taken at full precision instead of the printout's five
    figures."""
    if text in _SOLVED:
        return _SOLVED[text]
    deck = parse_deck(text)
    solver = DeckSolver(deck)
    group = next(g for g in deck.groups if g is not None)
    result = solver.solve_group(group, group.freqs_mhz[0])
    port, _segment, volts = result.driven[0]
    _SOLVED[text] = complex(volts / result.i_source[port])
    return _SOLVED[text]


def load_tags(name: str) -> list[int]:
    """The NEC tag of every wire carrying a load, in model order.

    The model addresses wires by index, so this walks back through the piece
    that carries each load to the flat NEC wire it was cut from — which is how
    "the load landed on tag 4 as well" is stated in the deck's own vocabulary.
    """
    deck = parse_deck(body(name))
    structure = deck.structure
    return [
        structure.wires[structure.pieces[wire].wire].tag
        for wire, _arc, _spec in deck.model.loads
    ]


@pytest.mark.integration
def test_the_fixture_set_is_the_one_this_module_measures():
    """Every ``LD``-carrying deck in the fixture directory is measured here.

    The directory grew a second tenant in momwire#456 phase C — the eight
    ``k9ay_{nt,tl}_*`` decks that ask the same cell question of the NETWORK
    cards, measured by ``test_deck_nec2_network_fixtures.py``.  They carry no
    ``LD``, which is the property this reads: the set is still closed, and a
    ninth network deck cannot quietly land in this module's blind spot.
    """
    on_disk = tuple(sorted(p.stem for p in FIXTURE_DIR.glob("*.deck")))
    assert NAMES == tuple(n for n in on_disk if "\nLD" in body(n))


# ---------------------------------------------------------------------------
# k9ay_orig — GX, cell = tags 1 and 2
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_k9ay_declares_the_cell_nec2c_prints():
    """The corpus deck's own reflection: two wires, 13 segments, reflected in
    X=0 into 26 — and nec2c prints SEGMENTS IN A SYMMETRIC CELL: 13."""
    deck = parse_deck(body("k9ay_orig"))
    assert deck.structure.symmetry == Symmetry(cell_wires=2, cell_segments=13)
    assert deck.structure.n_segments == 26
    assert [w.tag for w in deck.structure.wires] == [1, 2, 3, 4]


def test_k9ay_lands_its_cell_load_on_the_image_too():
    """The parse-level fact this fixture exists for: ``LD 4 2 1 1`` addresses
    tag 2, which is in the cell, so the 470 ohms is carried by tag 2 AND by
    tag 4 — and tag 4 is the wire ``EX`` drives."""
    assert load_tags("k9ay_orig") == [2, 4]
    assert load_tags("k9ay_orig_expanded") == [2, 4]
    assert load_tags("k9ay_orig_naive") == [2]


@pytest.mark.integration
def test_k9ay_as_written_equals_its_all_copies_loaded_expansion():
    """#415's transferable property, at the answer rather than at the model:
    the ``GX`` deck and the four-``GW`` deck that loads every copy are the
    same solve, bit for bit."""
    assert driving_point(body("k9ay_orig")) == driving_point(body("k9ay_orig_expanded"))


@pytest.mark.integration
def test_the_k9ay_naive_expansion_is_short_by_exactly_the_load():
    """The other half of the gate, and the reason the rule is worth the code:
    reading ``LD`` as if the structure were ordinary leaves the 470 ohms off
    the DRIVEN wire, so the answer is 470 ohms low in resistance and identical
    in reactance.  nec2c prints 957.18 against 487.18 for the same pair."""
    written = driving_point(body("k9ay_orig"))
    naive = driving_point(body("k9ay_orig_naive"))
    assert written.real - naive.real == pytest.approx(470.0, abs=1e-9)
    assert written.imag == pytest.approx(naive.imag, abs=1e-9)


@pytest.mark.integration
def test_without_the_load_every_k9ay_form_agrees():
    """Delete the ``LD`` and the three decks are one deck, which is what
    isolates the whole effect to loading.  nec2c: 0.10161 + j514.86 for all
    three."""
    written = driving_point(unloaded("k9ay_orig"))
    assert written == driving_point(unloaded("k9ay_orig_expanded"))
    assert written == driving_point(
        body("k9ay_orig_naive").replace("LD 4 2 1 1 470. 0.\n", "")
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("name", "oracle"),
    [
        ("k9ay_orig", complex(957.18, 498.15)),
        ("k9ay_orig_expanded", complex(957.18, 498.15)),
        ("k9ay_orig_naive", complex(487.18, 498.15)),
    ],
)
def test_k9ay_agrees_with_the_oracle(name, oracle):
    """Cross-engine agreement at the differential battery's own 5 % bar.

    The identities above are what pin the RULE; this pins that the rule is
    being applied to the right structure — an expansion that lost a wire or
    mis-tagged an image would satisfy every identity above and fail here.
    """
    ours = driving_point(body(name))
    assert abs(ours - oracle) / abs(oracle) <= 0.05, f"{ours} against {oracle}"


@pytest.mark.integration
def test_the_tower_declares_a_quarter_of_itself_as_the_cell():
    """``GR 3 4`` fires with 25 wires and 42 segments built, so the cell is
    that prefix and the three copies follow it contiguously."""
    deck = parse_deck(body("1MHz_tower"))
    assert deck.structure.symmetry == Symmetry(cell_wires=25, cell_segments=42)
    assert len(deck.structure.wires) == 100
    assert deck.structure.n_segments == 168
    assert sorted({w.tag for w in deck.structure.wires}) == list(range(1, 13))


def test_the_towers_cell_load_lands_on_every_leg():
    """The parse-level fact: one ``LD`` on tag 3 becomes four inductors, one
    per leg, on the tags ``GR`` gave the copies."""
    assert load_tags("1MHz_tower") == [3, 6, 9, 12]
    assert load_tags("1MHz_tower_expanded") == [3, 6, 9, 12]


@pytest.mark.integration
def test_the_tower_as_written_equals_its_all_copies_loaded_expansion():
    """#415's property again, on the rotation rather than the reflection.

    The twin retires the symmetry with ``GX 0 0`` rather than writing the
    ``GR`` out as 100 ``GW`` cards: a rotation's coordinates cannot be spelled
    on a card without rounding them, and the point of the comparison is that
    nothing but the ADDRESSING differs.  The wire lists are asserted equal
    first, so the twin cannot pass by being a different structure.
    """
    assert parse(body("1MHz_tower")).wires == parse(body("1MHz_tower_expanded")).wires
    assert driving_point(body("1MHz_tower")) == driving_point(
        body("1MHz_tower_expanded")
    )


@pytest.mark.integration
def test_the_towers_own_ld_set_now_answers_the_cell_forms_number():
    """``1MHz_tower_dropped`` is the corpus deck verbatim: four ``LD`` cards,
    three of them addressing copies.

    THIS TEST USED TO PIN A REFUSAL.  momwire#471 replaced it deliberately —
    the refusal was not wrong about NEC, it was wrong about this deck.  NEC
    drops those three cards onto exactly the segments it then replicates the
    surviving one onto, so the four cards together ARE the replication and
    the deck means precisely what NEC computes.  Refusing a deck we can
    answer exactly is a refusal that costs a user a correct answer.

    Gated at the ANSWER rather than at the load list, which is the stronger
    statement and the form #415 asks for: the corpus deck and the one-card
    cell form must reach the same driving point, not merely the same
    bookkeeping.  The oracle behind both is nec2c's 50.61 - j1.6296, equal
    for the two forms to every printed digit.
    """
    assert parse(body("1MHz_tower_dropped")).wires == parse(body("1MHz_tower")).wires
    assert driving_point(body("1MHz_tower_dropped")) == driving_point(
        body("1MHz_tower")
    )


@pytest.mark.integration
def test_a_copy_card_that_disagrees_with_the_cell_still_refuses():
    """The half of momwire#471 that is still a refusal, and the reason the
    widening checks equality instead of assuming it.

    Move ONE of the three copy cards off the cell's value and the deck stops
    being a restatement: NEC would silently keep the cell's 85 uH and discard
    the 95 uH the author wrote, which is the silent-wrong class this rule
    exists to catch.  A widening that only asked "is this segment already
    loaded" would serve this and answer a different antenna.
    """
    text = body("1MHz_tower_dropped")
    disagreeing = text.replace("LD 0 9 1 1 0. 8.5E-5 0.", "LD 0 9 1 1 0. 9.5E-5 0.", 1)
    assert disagreeing != text, "the fixture's LD spelling moved"
    with pytest.raises(DeckError) as exc:
        parse(disagreeing)
    assert "outside the GX/GR symmetric cell" in str(exc.value)


@pytest.mark.integration
def test_a_copy_card_before_its_cell_card_still_refuses():
    """The documented ORDER limit of momwire#471, pinned so it stays a
    decision rather than becoming a surprise.

    The restatement test runs per card against what has already been loaded,
    so the cell card must come first.  Reverse the ``LD`` block and the first
    card read addresses a copy with nothing yet to restate — refused.  NEC
    itself does not care about the order; lifting this would mean deferring
    the whole decision to the execute card, the way the MININEC gate does,
    and no corpus deck has asked for it.
    """
    text = body("1MHz_tower_dropped")
    lines = text.splitlines()
    lds = [ln for ln in lines if ln.startswith("LD ")]
    assert len(lds) == 4, lds
    keep = [ln for ln in lines if not ln.startswith("LD ")]
    cut = max(i for i, ln in enumerate(keep) if ln.startswith(("GE", "GR")))
    reversed_deck = "\n".join(keep[: cut + 1] + lds[::-1] + keep[cut + 1 :])
    with pytest.raises(DeckError) as exc:
        parse(reversed_deck)
    assert "outside the GX/GR symmetric cell" in str(exc.value)


def test_the_unloaded_tower_agrees_with_the_oracle():
    """The oracle bar for this fixture, and the one that measures ``GR``.

    Unloaded, the two engines are 0.2 % apart on a 100-wire rotated structure
    over perfect ground, which is the ``GR`` expansion being right end to end
    — geometry, tags, ground contact and all.  The LOADED rows are not gated
    against the oracle: 85 uH is +j534 ohms at 1 MHz and drives the four-leg
    assembly to a resonance whose impedance is a small residual between large
    cancelling terms, which is a basis question rather than a #415 one.
    """
    oracle = complex(0.001692, 12.007)
    ours = driving_point(unloaded("1MHz_tower"))
    assert abs(ours - oracle) / abs(oracle) <= 0.05, f"{ours} against {oracle}"
    assert ours == driving_point(unloaded("1MHz_tower_expanded"))
