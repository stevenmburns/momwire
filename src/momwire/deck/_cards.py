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
    """

    mnemonic: str
    values: tuple[float, ...]
    raw: str

    def f(self, k: int) -> float:
        return self.values[k] if k < len(self.values) else 0.0

    def i(self, k: int) -> int:
        return int(round(self.f(k)))

    @property
    def text(self) -> str:
        """A comment card's free text: everything after the mnemonic."""
        return self.raw[2:]


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
    values = []
    for token in tokens[1:]:
        try:
            values.append(float(token.replace("D", "E").replace("d", "e")))
        except ValueError:
            raise DeckError(
                f"NON-NUMERICAL CHARACTER IN FIELD: {token!r} on {stripped!r}"
            ) from None
    return Card(mnemonic, tuple(values), line.rstrip("\n"))


def tokenize(text: str) -> list[Card]:
    """Every card in ``text``, blank lines dropped."""
    cards = []
    for line in text.splitlines():
        card = parse_card(line)
        if card is not None:
            cards.append(card)
    return cards
