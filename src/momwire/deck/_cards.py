"""Card lines: tokenization and the one exception every refusal raises.

Normative spec: ``site/src/content/docs/reference/deck-grammar-nec2.md``
("The nec2 deck dialect"), sections ``#deck-framing`` and ``#field-numbering``.

The two mnemonic-level errors here are raised *before* any card is
interpreted, which is why they live with the tokenizer rather than with the
dialect: a deck that does not tokenize has no cards to refuse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["DeckError", "Card", "gw_field_refusal", "parse_card", "tokenize"]


class DeckError(ValueError):
    """A deck this engine will not run, with the spec's message.

    A ValueError subclass so a caller that only knows "the parser raises"
    still catches it; the dedicated type is what a framing caller (a portal,
    a CLI) matches on to decide it is the *deck's* fault, not the code's.
    """


# A mnemonic glued to its first field (``GE1``, ``GD2,0,0,...``) splits when
# the third character starts a number.  A letter there would be a three-letter
# word, which is a mnemonic error, not a fused field.
_FUSED_FIELD_START = "0123456789.+-"


@dataclass(frozen=True)
class Card:
    """One deck card: mnemonic plus its numeric fields, in card order.

    Fields are read *positionally* and converted on demand, so NEC's split
    into four integers and six reals never has to be modelled: ``i(k)`` is
    ``f(k)`` rounded.  ``values`` is short when the card is — a field the deck
    did not write reads as 0, exactly as NEC's zero-filled card image does.

    ``trailer`` is the one deliberate hole in the "every field is a float"
    rule (momwire#1084): a ``GN`` card's last token is tokenized as a NAME
    rather than refused when it does not parse as a number, because NEC-5's
    ``GN`` carries an optional trailing Sommerfeld-table filename.  ``None``
    for every card whose last token IS numeric, and for every mnemonic but
    ``GN`` — so an existing ``Card(...)`` construction or equality is
    unaffected by this field's addition.
    """

    mnemonic: str
    values: tuple[float, ...]
    raw: str
    trailer: str | None = None

    def f(self, k: int) -> float:
        return self.values[k] if k < len(self.values) else 0.0

    def i(self, k: int) -> int:
        """Field ``k`` as an integer: rounded half to even, exact however
        large.  An infinite or NaN field has no integer, and refuses here
        with the exception type Python's own conversion raises (OverflowError
        / ValueError) but in this module's words, which name the card and
        the field rather than the interpreter's float-to-int conversion: the
        C++ reader (`_nec5_deck_core.cpp`) raises the same text."""
        value = self.f(k)
        if math.isnan(value):
            raise ValueError(
                f"{self.mnemonic} field {k + 1} is NaN; an integer field must be "
                f"a finite number"
            )
        if math.isinf(value):
            raise OverflowError(
                f"{self.mnemonic} field {k + 1} is infinite; an integer field must "
                f"be a finite number"
            )
        return int(round(value))

    @property
    def text(self) -> str:
        """A comment card's free text: everything after the mnemonic."""
        return self.raw[2:]


# The most segments one GW may ask for. A count past it is no wire anything
# can mesh (the matrix alone is its square), and a count in the millions of
# millions would hang the reader building nodes, so it refuses on the card.
# The C++ readers hold the same limit (`_nec5_deck_core.cpp`'s
# `kMaxGeometry`, `_nec4_deck_core.cpp`'s `kMaxSegments`).
MAX_WIRE_SEGMENTS = 10_000_000


def gw_field_refusal(card: Card) -> str | None:
    """Why a ``GW`` card is malformed past serving, or ``None``.

    Every one of a GW's nine fields is read (tag, segment count, the two
    ends, the radius), and none of them means anything as an infinity or
    a NaN: each such deck used to fail deep in the mesh with the
    interpreter's own float-to-int message. A segment count past
    :data:`MAX_WIRE_SEGMENTS` is refused here too. Every dialect's reader
    asks this on the card, before anything else reads it, and the C++
    readers (`_nec5_deck_core.cpp`'s `gw_field_refusal`) say the same
    words."""
    for k, value in enumerate(card.values[:9]):
        if not math.isfinite(value):
            return (
                f"GW field {k + 1} is not a finite number; a wire's tag, segment "
                f"count, end coordinates and radius must all be finite"
            )
    if card.i(1) > MAX_WIRE_SEGMENTS:
        return (
            f"GW tag {card.i(0)} asks for more than {MAX_WIRE_SEGMENTS} segments; "
            f"this engine meshes at most {MAX_WIRE_SEGMENTS} segments on one wire"
        )
    return None


def _to_float(token: str) -> float | None:
    """``token`` as NEC reads a numeric field, or None when it does not."""
    try:
        return float(token.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def parse_card(line: str) -> Card | None:
    """One deck line as a :class:`Card`, or None for a blank line.

    Free format per ``#deck-framing``: the mnemonic is the first two
    characters, fields are separated by spaces and/or commas, Fortran ``D``
    exponents are accepted, and blank lines are skipped.
    """
    stripped = line.strip()
    if not stripped:
        return None
    tokens = stripped.replace(",", " ").split()
    if not tokens:
        # A line of separators alone. IndexError is the type this line has
        # always raised (an empty token list's first element); the message
        # says what the line is.
        raise IndexError(f"card line has no mnemonic, only separators: {stripped}")
    head = tokens[0]
    if len(head) > 2 and head[:2].isalpha() and head[2] in _FUSED_FIELD_START:
        tokens = [head[:2], head[2:], *tokens[1:]]
    mnemonic = tokens[0].upper()
    if len(mnemonic) != 2 or not mnemonic.isalpha():
        raise DeckError(f"CARD'S MNEMONIC CODE TOO SHORT OR MISSING: {stripped}")
    if mnemonic in ("CM", "CE"):
        # Comment bodies are free text; tokenizing them would refuse a
        # perfectly ordinary English comment for containing an apostrophe.
        return Card(mnemonic, (), line.rstrip("\n"))
    fields = tokens[1:]
    trailer = None
    if mnemonic == "GN" and fields and _to_float(fields[-1]) is None:
        # NEC-5's GN carries an optional trailing Sommerfeld-table filename
        # (momwire#1084) — a NAME, not a field this parser widens numeric
        # parsing for. Scoped to the LAST token only: a non-numeric token
        # anywhere else on ANY card, GN included, is still the ordinary
        # refusal below.
        trailer, fields = fields[-1], fields[:-1]
    values = []
    for token in fields:
        parsed = _to_float(token)
        if parsed is None:
            raise DeckError(f"NON-NUMERICAL CHARACTER IN FIELD: {token} on {stripped}")
        values.append(parsed)
    return Card(mnemonic, tuple(values), line.rstrip("\n"), trailer)


def tokenize(text: str) -> list[Card]:
    """Every card in ``text``, blank lines dropped."""
    cards = []
    for line in text.splitlines():
        card = parse_card(line)
        if card is not None:
            cards.append(card)
    return cards
