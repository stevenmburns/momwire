"""Card lines: tokenization and the one exception every refusal raises.

Normative spec: ``site/src/content/docs/reference/deck-grammar-nec2.md``
("The nec2 deck dialect"), sections ``#deck-framing`` and ``#field-numbering``.

The two mnemonic-level errors here are raised *before* any card is
interpreted, which is why they live with the tokenizer rather than with the
dialect: a deck that does not tokenize has no cards to refuse.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DeckError", "Card", "parse_card", "tokenize"]


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
        return int(round(self.f(k)))

    @property
    def text(self) -> str:
        """A comment card's free text: everything after the mnemonic."""
        return self.raw[2:]


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
    head = tokens[0]
    if len(head) > 2 and head[:2].isalpha() and head[2] in _FUSED_FIELD_START:
        tokens = [head[:2], head[2:], *tokens[1:]]
    mnemonic = tokens[0].upper()
    if len(mnemonic) != 2 or not mnemonic.isalpha():
        raise DeckError(f"CARD'S MNEMONIC CODE TOO SHORT OR MISSING: {stripped!r}")
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
            raise DeckError(
                f"NON-NUMERICAL CHARACTER IN FIELD: {token!r} on {stripped!r}"
            )
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
