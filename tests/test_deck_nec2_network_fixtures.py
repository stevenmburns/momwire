"""``TL``/``NT`` under a live symmetric cell: the exemption, measured.

momwire#415 established the SYMMETRIC CELL RULE for ``LD`` — while a ``GX``/
``GR`` cell is live the matrix is the cell's, so a load written on the cell
lands on every copy and a load written on a copy is silently dropped.  Whether
the network cards follow that rule was left explicitly *measured-not* by the
phase C design doc (``docs/design/networks-move-into-the-engine.md``, "Symmetry
× networks"), with a by-name refusal as the placeholder.  This module is the
measurement, and the answer is that networks are **exempt**: a ``TL`` or an
``NT`` attaches ONCE, exactly where addressed, resolved against the fully
generated structure including the image tags the cell created.

The argument the spec makes for it is the one that already exempts ``EX``: a
cell replicates what enters the OPERATOR, and NEC's network solve is a
composition on top of the solved matrix rather than a term inside it.  The
argument is not why it is believed, though — the eight decks under
``tests/fixtures/nec2_symmetry/k9ay_{nt,tl}_*.deck`` are.  Each quartet is one
question asked four ways:

``cell``
    the ``GX 2 100`` structure with one network card on cell tags.
``naive``
    the same structure written out as four ``GW`` cards, carrying that ONE
    card as written — the reading an engine produces with no cell rule at all.
``expanded``
    the same four ``GW`` cards with one card PER COPY — the reading a cell
    rule would produce.
``copy``
    the ``GX`` structure with the card addressing the IMAGE tags, which are
    outside the cell — the address an ``LD`` is dropped for.

nec2c 5b4az.ae6ty (the build SimNEC ships), run on these exact deck bodies on
2026-08-20, free space, 1.8 MHz, driven on tag 4 seg 1 (an image):

    ==================  =====================  ==================
    deck                nec2c Z                versus cell
    ==================  =====================  ==================
    (no network card)    0.10161 + j514.86     --
    k9ay_nt_cell        57.233   + j501.94     --
    k9ay_nt_naive       57.233   + j501.94     **equal**
    k9ay_nt_expanded   109.75    + j492.48     +52 ohms R
    k9ay_nt_copy        55.036   + j497.78     resolved, not dropped
    k9ay_tl_cell         0.098596 + j577.58    --
    k9ay_tl_naive        0.098596 + j577.58    **equal**
    k9ay_tl_expanded     0.094537 + j636.49    +59 ohms X
    k9ay_tl_copy         0.096328 + j573.61    resolved, not dropped
    ==================  =====================  ==================

Three things are true of that table and are what the decks are for.  The cell
form equals the naive form to every printed digit — and their whole network
printout blocks are byte-identical, not merely their impedances.  The
replicated form does NOT: it is the falsifier, and an engine that grew a cell
rule for networks by analogy with ``LD`` would match it instead.  And the
image-tag form resolves and answers rather than dropping, which is the precise
point where a network's addressing parts company with a load's.

**What this module pins today, and what unit 2 adds.**  The network solve is
staged: ``momwire#456`` phase C unit 1 lands the PARSER, so what is checkable
here is the MODEL — that the cell and naive decks resolve to the same
endpoints, that the image-tag deck resolves onto the generated wires, that no
cell-rule refusal fires.  The oracle impedances above are recorded for unit 2,
which composes the network with the antenna and can then assert the identity
at the ANSWER (``driving_point(cell) == driving_point(naive)``, the shape
``test_deck_nec2_cell_fixtures.py`` uses for ``LD``).  The helpers below are
laid out for that: :func:`body` and :func:`without_networks` are the same two
verbs that module has, and adding the answer-level tests needs no rework here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from momwire.deck import DeckError, build_solver, parse
from momwire.deck._nec2_geometry import Symmetry

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "nec2_symmetry"

NT_QUARTET = (
    "k9ay_nt_cell",
    "k9ay_nt_naive",
    "k9ay_nt_expanded",
    "k9ay_nt_copy",
)
TL_QUARTET = (
    "k9ay_tl_cell",
    "k9ay_tl_naive",
    "k9ay_tl_expanded",
    "k9ay_tl_copy",
)
NAMES = NT_QUARTET + TL_QUARTET

# The oracle impedances of the table above, for unit 2's answer-level gate.
# Recorded as data rather than prose so the unit that needs them does not have
# to re-derive them from a docstring.
ORACLE_Z = {
    "k9ay_nt_cell": complex(57.233, 501.94),
    "k9ay_nt_naive": complex(57.233, 501.94),
    "k9ay_nt_expanded": complex(109.75, 492.48),
    "k9ay_nt_copy": complex(55.036, 497.78),
    "k9ay_tl_cell": complex(0.098596, 577.58),
    "k9ay_tl_naive": complex(0.098596, 577.58),
    "k9ay_tl_expanded": complex(0.094537, 636.49),
    "k9ay_tl_copy": complex(0.096328, 573.61),
    # The control: the same structure with the network card deleted.
    "control": complex(0.10161, 514.86),
}


def body(name: str) -> str:
    return (FIXTURE_DIR / f"{name}.deck").read_text().split("\nNX", 1)[0]


def without_networks(name: str) -> str:
    """The fixture with every ``TL``/``NT`` card deleted — the control that
    says the network card, not the geometry expansion, is what moves the
    answer.  Built by deletion rather than by committing a ninth deck."""
    kept = [
        line
        for line in body(name).splitlines()
        if line.split()[:1] not in (["TL"], ["NT"])
    ]
    text = "\n".join(kept) + "\n"
    assert "\nTL" not in text and "\nNT" not in text
    return text


def networks(name: str):
    return parse(body(name)).networks


def test_the_fixture_set_is_the_one_this_module_measures():
    """Both quartets, and nothing claimed that is not on disk."""
    on_disk = {p.stem for p in FIXTURE_DIR.glob("*.deck")}
    assert set(NAMES) <= on_disk
    assert set(NAMES) == {n for n in on_disk if n.startswith(("k9ay_nt", "k9ay_tl"))}


# ---------------------------------------------------------------------------
# §#the-cell-rule — the exemption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ("k9ay_nt_cell", "k9ay_tl_cell", "k9ay_nt_copy"))
def test_the_cell_decks_really_do_reach_ge_with_a_live_cell(name):
    """The premise, stated rather than assumed: without a live symmetry at
    ``GE`` these decks measure nothing at all.  nec2c prints ``SEGMENTS IN A
    SYMMETRIC CELL: 13`` for them, against 26 for their expanded twins."""
    from momwire.deck._nec2 import _Nec2Parser

    parser = _Nec2Parser()
    parser.feed(body(name))
    assert parser.structure.symmetry == Symmetry(cell_wires=2, cell_segments=13)
    assert parser.structure.n_segments == 26


@pytest.mark.parametrize(
    "cell,naive", (("k9ay_nt_cell", "k9ay_nt_naive"), ("k9ay_tl_cell", "k9ay_tl_naive"))
)
def test_a_network_under_a_live_cell_parses_the_same_as_its_naive_expansion(
    cell, naive
):
    """§#the-cell-rule: the headline.  One card under a ``GX`` and the same one
    card over the hand-expanded structure resolve to the SAME endpoints — one
    attachment, exactly as addressed, no replication.

    That is the model-level half of an oracle identity: nec2c answers these two
    decks with the same impedance to every printed digit and with byte-identical
    network printout blocks (57.233 + j501.94 for the ``NT`` pair,
    0.098596 + j577.58 for the ``TL`` pair)."""
    assert networks(cell) == networks(naive)


@pytest.mark.parametrize(
    "cell,expanded",
    (
        ("k9ay_nt_cell", "k9ay_nt_expanded"),
        ("k9ay_tl_cell", "k9ay_tl_expanded"),
    ),
)
def test_a_network_under_a_live_cell_is_not_replicated_onto_the_copies(cell, expanded):
    """§#the-cell-rule, the falsifier.  The reading a cell rule WOULD produce
    is one card per copy, and the oracle says that is a different antenna:
    109.75 against 57.233 ohms for the ``NT`` pair, 636.49 against 577.58 ohms
    of reactance for the ``TL`` pair.  So the cell deck must carry exactly one
    network record, not two."""
    assert len(networks(cell)) == 1
    assert len(networks(expanded)) == 2
    assert networks(cell) != networks(expanded)


@pytest.mark.parametrize("name", ("k9ay_nt_copy", "k9ay_tl_copy"))
def test_a_network_may_address_the_image_tags_a_load_may_not(name):
    """§#the-cell-rule: the precise point where networks part company with
    loads.  Tags 3 and 4 exist only because the ``GX`` generated them and lie
    outside the cell, so an ``LD`` written this way is refused (NEC drops it in
    silence).  A network card resolves against the fully generated structure
    and attaches: model wires 2 and 3, the images of 0 and 1."""
    (card,) = networks(name)
    assert (card.address_a, card.address_b) == ((3, 4), (4, 3))
    # Absolute, not merely "the same as the cell deck's": the images are model
    # wires 2 and 3, and the arclengths are segment CENTRES on them — 3.5/8 and
    # 2.5/5 of each wire's length.  Pinned as numbers because a resolver broken
    # to fold every address back into the cell would move the cell deck's
    # endpoints by the same amount and a cross-deck comparison would not see it.
    assert card.end_a[0] == 2
    assert card.end_b[0] == 3
    assert card.end_a[1] == pytest.approx(3.763517304331149, abs=1e-12)
    assert card.end_b[1] == pytest.approx(2.6100766272276377, abs=1e-12)
    # And they ARE the cell deck's, reflected: same wire-local point, one
    # mirror away — which is the geometric content of "no replication".
    (cell_card,) = networks(name.replace("_copy", "_cell"))
    assert cell_card.end_a[0] == 0 and cell_card.end_b[0] == 1
    assert card.end_a[1] == pytest.approx(cell_card.end_a[1], abs=1e-12)
    assert card.end_b[1] == pytest.approx(cell_card.end_b[1], abs=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_no_cell_guard_fires_on_any_of_the_eight(name):
    """§#the-cell-rule: the exemption is documented BEHAVIOUR, not an accident
    of which guard happens to run first.  All eight decks parse — including the
    four under a live cell, and including the two whose card addresses copies
    the way the refused ``LD`` form does."""
    model = parse(body(name))
    assert model.networks, f"{name} parsed no network card"


@pytest.mark.parametrize("name", NAMES)
def test_every_fixture_still_stops_at_the_staged_solve_refusal(name):
    """The unit boundary, pinned so it cannot rot silently: these decks parse
    and do not yet solve.  Unit 2 deletes this test and puts the answer-level
    identities of ``ORACLE_Z`` in its place."""
    with pytest.raises(DeckError, match="momwire#456 phase C"):
        build_solver(parse(body(name)))


@pytest.mark.parametrize("name", NAMES)
def test_the_control_form_of_every_fixture_solves(name):
    """Delete the network card and the deck is an ordinary antenna again —
    which is what says the staged refusal above is about the network and not
    about the geometry these fixtures happen to use.

    All eight controls are the same two structures, so all eight are the
    oracle's 0.10161 + j514.86 (``ORACLE_Z["control"]``); the cell and expanded
    forms of THAT deck agreeing is momwire#415's own result, re-measured here
    for free."""
    model = parse(without_networks(name))
    assert not model.networks
    assert build_solver(model) is not None
