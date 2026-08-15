"""Deck parsing: a card deck in, a dialect-neutral :class:`DeckModel` out.

Normative spec: ``site/src/content/docs/reference/deck-grammar-nec2.md``
("The nec2 deck dialect").  That page is the contract for what
``parse(text, dialect="nec2")`` accepts, which fields it reads, and the exact
text of every refusal.

An engine's dialect is part of its identity: nec2c reads NEC-2's deck
language, NEC-5 reads its own, and momwire reads ``nec2`` — a restricted
NEC-2 that describes wire antennas and asks them questions.  Every dialect
front-end parses into ONE model, and ``build_solver(model, basis=...)`` maps
that model onto momwire's solver families, so a second dialect is a second
parser rather than a second pipeline.
"""

from __future__ import annotations

from ._cards import Card, DeckError, parse_card, tokenize
from .model import (
    DeckModel,
    DeckWire,
    ExecuteGroup,
    FarFieldRequest,
    LoadSpec,
    NearFieldRequest,
    PrintControl,
    SecondMedium,
    WireMaterial,
)

__all__ = [
    "DeckError",
    "DeckModel",
    "DeckWire",
    "WireMaterial",
    "LoadSpec",
    "SecondMedium",
    "FarFieldRequest",
    "NearFieldRequest",
    "PrintControl",
    "ExecuteGroup",
    "Card",
    "parse_card",
    "tokenize",
]
