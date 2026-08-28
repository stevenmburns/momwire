"""Every prose statement of a corpus SIZE, pinned to the corpus.

These figures went stale twice without anything failing.  The corpus grew
49 -> 53 -> 62 -> 80 over six days, and the sentences that quote a denominator
live in fourteen files across ``tests/``, ``src/`` and ``docs/`` — including
two user-visible refusal messages and one module docstring that a reader is
meant to trust.  The promotion commit that took it to 80 restated four
baselines deliberately and missed the rest; a destaling sweep before it had
already been "flagged twice and never actioned".

The failure mode is specific: nothing connects the prose to the manifest, so a
number is only ever wrong in a way a HUMAN can notice.  This module makes the
connection.  Each claim is a sentence fragment rendered from the live counts,
asserted to appear in the file that makes it — so growing the corpus fails
here, once per stale sentence, and the failure names the file to edit.

It does NOT pin the physics figures (the razor medians, the dust floor's
1.648e-9).  Those are measurements rather than counts: they do not follow from
the manifest, they move when a solver changes rather than when a deck lands,
and each already has a gate of its own.  What is gated here is only what can
be counted without solving anything.

One live wrongness this sweep turned up, recorded because a count that was
never right is a different bug from a count that went stale: ``_printout.py``
said "all 62 captures write exactly one request card".  47 of the 80 write
one, 33 write none — the ``XQ`` half of the Src Dat / FF Plot capture pair —
and the argument only ever needed "none writes TWO", which is true and gated
below.
"""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "eznec"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())

# The three counts every stale sentence was a statement of.
CORPUS = len(MANIFEST["captures"])
PRINTOUTS = sum(1 for c in MANIFEST["captures"] if c.get("printout"))


def _served() -> int:
    """How many decks the answering bases accept, off the gated per-deck table.

    Read from ``test_eznec_basis_choice.ACCEPTS`` rather than re-rendered:
    that table is already walked against real renders in its own module, so
    taking the count from it keeps this file to counting and out of solving.
    """
    from test_eznec_basis_choice import ACCEPTS, SERVED

    return max(
        sum(1 for row in ACCEPTS.values() if row[basis] is SERVED)
        for basis in next(iter(ACCEPTS.values()))
    )


# Every sentence in the tree that quotes one of the counts above.  The value is
# the fragment as it must READ, with the live number substituted in — so the
# assertion is on the prose a reader sees, not on a constant next to it.
def _claims() -> list[tuple[str, str]]:
    served = _served()
    return [
        # the seam's own record of what it serves
        ("tests/test_eznec_basis_choice.py", f"accepts all {served}"),
        ("tests/test_eznec_basis_choice.py", f"THREE shapes cover all {CORPUS} now"),
        ("tests/test_eznec_basis_choice.py", f"{served}/{served} for every basis"),
        ("tests/test_eznec_basis_choice.py", f"the {PRINTOUTS}\nprintout-carrying"),
        ("tests/test_eznec_serve.py", f"Every one of the {CORPUS} captured decks"),
        ("tests/test_eznec_serve.py", f"does NOT claim is that all {served} are right"),
        # the drive spelling, and the razor ladder it summarises
        ("tests/test_eznec_drive_spelling.py", f"the {PRINTOUTS}\nprintout-carrying"),
        ("tests/test_eznec_drive_spelling.py", f"{CORPUS} committed captures"),
        ("tests/test_eznec_drive_spelling.py", f"and {served} without it"),
        ("src/momwire/eznec/_serve.py", f"It serves {served} today"),
        # the thread-count sweeps
        ("tests/test_eznec_printout.py", f"over all {CORPUS} capture decks"),
        ("tests/test_eznec_shell.py", f"across all {CORPUS} decks"),
        ("src/momwire/eznec/_serve.py", f"over all {CORPUS} capture decks"),
        # the deck front end
        ("tests/test_deck_nec5.py", f"Parsing all {CORPUS} decks"),
        # user-visible refusal text
        ("src/momwire/eznec/_serve.py", f"none of the {CORPUS} captured decks"),
        # the printout's request-card shape
        ("src/momwire/eznec/_printout.py", f"across the {CORPUS} captures"),
        # the design note
        ("docs/design/seam-rule.md", f"Three of the {CORPUS} corpus decks"),
    ]


@pytest.mark.parametrize(
    "path,fragment", _claims(), ids=lambda v: v.replace("\n", " ")[:48]
)
def test_a_stated_corpus_figure_is_the_corpus_s_own(path: str, fragment: str):
    """One sentence, one count, one place to fix it when the corpus grows."""
    text = (ROOT / path).read_text()
    assert fragment in text, (
        f"{path} no longer states this figure as {fragment!r} — the corpus is "
        f"{CORPUS} decks, {PRINTOUTS} with a printout, {_served()} served"
    )


def test_the_manifest_and_the_files_on_disk_agree():
    """The counts above are only worth pinning to if the manifest is honest."""
    assert len(list((FIXTURES / "decks").glob("*.nec"))) == CORPUS
    assert len(list((FIXTURES / "printouts").glob("*.out"))) == PRINTOUTS


@pytest.mark.integration
def test_no_capture_writes_two_request_cards():
    """The claim ``_printout.py`` needs, gated instead of asserted in prose.

    The block-ordering question it records only arises for a deck with TWO
    request cards, and the renderer does not handle that shape.  What keeps
    that safe is a property of the corpus, so measure it here rather than
    trusting a sentence — the sentence was wrong for a year in the other
    direction, claiming every capture writes exactly one.
    """
    counts = []
    for entry in MANIFEST["captures"]:
        deck = (FIXTURES / entry["deck"]).read_text()
        counts.append(
            sum(1 for line in deck.splitlines() if line[:2] in ("NE", "NH", "RP"))
        )
    assert max(counts) <= 1, "a capture now writes two request cards"

    # And the split, pinned to the sentence rather than to a constant beside
    # it — the same rule as every claim in the table above.
    one = sum(1 for n in counts if n == 1)
    none = sum(1 for n in counts if n == 0)
    said = (ROOT / "src/momwire/eznec/_printout.py").read_text()
    assert (
        f"{one} write exactly one\n    request card and the other {none} write NONE"
        in said
    )
