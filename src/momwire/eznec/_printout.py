"""The NEC-5 printout, top down — in U1, the header and the refusal frame.

Layout is the contract; the numbers (when later units add them) are momwire's.
Everything in this module was derived from CAPTURED INPUT/OUTPUT ONLY — the
decks EZNEC wrote and the printouts the Windows engine left behind, kept under
``tests/fixtures/eznec/`` — and from the two interface studies:

* antennaknobs ``docs/status/2026-08-16-eznec-nec5-dialect-capture.md``
  (invocation protocol, the error convention, the stamp echo)
* antennaknobs ``docs/status/2026-08-20-eznec-nec5-scored-matrix.md``
  (the scored ladder and its normalizations)

No NEC-5 source, algorithm, or internal structure is described or relied on
here, and none may be: the binary is user-licensed and export-controlled.
What is reproduced is the shape of bytes a user's own engine already printed.

Why the header alone is worth a unit
------------------------------------
EZNEC does not read the engine's exit status at all; it reads the printout,
and it decides whether the printout belongs to THIS run by looking for the
comment block echoed back at the top.  EZNEC stamps the launch time into the
deck as a ``CM`` card, so that echo is the only thing in the file that
distinguishes one run from another.  A printout without it is rejected as
"written earlier from another calculation" and no message of any kind — not
even a refusal — reaches the user (capture doc, "Error convention").

So the header is not decoration.  It is the envelope every later unit's
results, and every refusal this engine will ever emit, has to travel in.
"""

from __future__ import annotations

__all__ = [
    "STUB_REFUSAL",
    "render_header",
    "render_refusal",
]

# The stub refusal U1 hands to every deck.  U2 replaces it with per-card
# refusals from the dialect parser; U4 replaces the served decks entirely.
STUB_REFUSAL = "NEC-5 DIALECT NOT YET SERVED BY THIS ENGINE"

# --------------------------------------------------------------------------
# the header, column by column
# --------------------------------------------------------------------------
#
# Measured from all ten byte-gate printouts in tests/fixtures/eznec/printouts/
# (capture ids 0010, 0012, 0013, 0014, 0016, 0017, 0019, 0035, 0043, 0044).
# The skeleton is identical in every one of them; only the echoed comment
# lines differ, and they differ exactly as their decks do.

_PROLOGUE: tuple[str, ...] = (
    # A bare "1" — a Fortran carriage-control form feed, printed literally by
    # every capture — then the engine's build tag.
    "1",
    " x13",
    "",
    "",
    "",
)

_BANNER_INDENT = " " * 32
_BANNER_RULE = "*" * 47
_BANNER: tuple[str, ...] = (
    _BANNER_INDENT + _BANNER_RULE,
    _BANNER_INDENT + "*" + " " * 45 + "*",
    _BANNER_INDENT + "*  NUMERICAL ELECTROMAGNETICS CODE (NEC-5)    *",
    _BANNER_INDENT + "*" + " " * 45 + "*",
    _BANNER_INDENT + _BANNER_RULE,
    "",
    "",
    "",
    "",
)

# The comment box is 80 columns wide and starts four columns left of the text
# it wraps — and the text OVERRUNS it on the right, to column 103.  That is
# not a transcription slip: the box rule and the comment lines are written by
# different formats, and the echo's field is wider than the box.  Reproduced
# as observed.
_BOX_RULE = " " * 22 + "*" * 58

# An echoed comment line is 25 blanks followed by the card's columns 3-80,
# blank-padded to that full 78-column field — so an empty ``CM`` echoes as 103
# spaces, trailing blanks and all.  Reading the text POSITIONALLY (columns
# 3-80) rather than by stripping the mnemonic and a space is the one place
# U1 chose a model over a measurement: every captured card is either bare
# ``CM``/``CE`` or ``CM `` + text, so the two readings agree on all 49 decks
# and differ only on a card with two spaces after the mnemonic, which no
# capture contains.  The positional reading is what a fixed-format engine
# does, so it is the safer guess; if a future capture ever disagrees, this
# is the line to change.
_ECHO_INDENT = " " * 25
_ECHO_TEXT_COLUMNS = slice(2, 80)
_ECHO_WIDTH = 78

# Five blank lines separate the comment box from the STRUCTURE SPECIFICATION
# heading, which is where the header ends and U3's renderer begins.
_HEADER_TAIL: tuple[str, ...] = ("", "", "", "", "")

# The refusal frame, verbatim from the fault-injection experiment (capture doc,
# "Refusals can speak"): one leading space, five asterisks, the words, and the
# reason.  That exact line reached the operator through EZNEC's own viewer.
_ERROR_PREFIX = " ***** NEC ERROR - "


def comment_cards(deck_text: str) -> list[str]:
    """The deck's leading comment block, as the text each card echoes.

    NEC's comment block is the run of ``CM`` cards at the top of the deck,
    terminated by ``CE`` — and ``CE`` echoes too, as its own (usually empty)
    line, which is why a five-``CM`` deck prints six comment lines.  Anything
    after ``CE``, and anything that is not a comment card, belongs to U2.
    """
    texts: list[str] = []
    for line in deck_text.splitlines():
        mnemonic = line[:2].upper()
        if mnemonic == "CM":
            texts.append(line[_ECHO_TEXT_COLUMNS])
        elif mnemonic == "CE":
            texts.append(line[_ECHO_TEXT_COLUMNS])
            break
        else:
            break
    return texts


def _comment_box(deck_text: str) -> list[str]:
    """The boxed comment echo — the stamp EZNEC checks a printout's age by."""
    return [
        _BOX_RULE,
        "",
        *(_ECHO_INDENT + text.ljust(_ECHO_WIDTH) for text in comment_cards(deck_text)),
        "",
        _BOX_RULE,
    ]


def render_header(deck_text: str | None) -> str:
    """The printout down to (not including) ``- - - STRUCTURE SPECIFICATION - - -``.

    ``deck_text`` of ``None`` is the one case with no comment box to print:
    the input file could not be read at all, so there are no ``CM`` cards to
    echo.  Such a printout cannot pass EZNEC's belongs-to-this-run check —
    nothing can, without the stamp — but it is still written, because the
    alternative (no file) sends EZNEC down the "check where you installed the
    NEC program" path and blames the user's configuration for a read error.

    Line endings are LF.  The captured printouts are the Windows engine's, so
    they arrived CRLF; the byte-gates normalize before comparing (see the
    fixture manifest's ``normalizations`` key), and this engine writes the
    platform-neutral form.
    """
    lines = [*_PROLOGUE, *_BANNER]
    if deck_text is not None:
        lines += _comment_box(deck_text)
    lines += _HEADER_TAIL
    return "\n".join(lines) + "\n"


def render_refusal(deck_text: str | None, reason: str) -> str:
    """A complete printout that refuses: the header, then one ``NEC ERROR`` line.

    The order is the contract.  With the echo intact and the results missing,
    EZNEC offers to show the operator the file, and whatever stands where the
    results would have been is what they read — so a refusal that names itself
    arrives as a sentence rather than as an unexplained failure.  Put the same
    line BEFORE the echo and the file is discarded as stale instead.
    """
    return render_header(deck_text) + "\n" + _ERROR_PREFIX + reason + "\n"
