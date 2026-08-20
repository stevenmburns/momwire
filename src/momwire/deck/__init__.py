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

A model's execute groups carry the OPERATING POINTS a deck asks for: a
frequency list, an extended-kernel flag and an :class:`Environment` each.
``build_solver`` takes all three off the selected group and lets a caller
override any of them; :func:`prepare_mesh` freezes what none of them can
move — the geometry — so a sweep translates the structure once.
"""

from __future__ import annotations

from ._cards import Card, DeckError, parse_card, tokenize
from ._nec2 import parse_nec2
from ._solver import (
    BASES,
    BuiltSolver,
    PortPlan,
    PortSite,
    PreparedMesh,
    build_solver,
    prepare_mesh,
)
from .model import (
    DeckModel,
    DeckWire,
    Environment,
    ExecuteGroup,
    FarFieldRequest,
    LoadSpec,
    NearFieldRequest,
    NetworkCard,
    PrintControl,
    SecondMedium,
    WireMaterial,
)

__all__ = [
    # the two verbs
    "parse",
    "build_solver",
    # translate a model's geometry once, for a swept caller
    "prepare_mesh",
    "PreparedMesh",
    # the model
    "DeckModel",
    "DeckWire",
    "WireMaterial",
    "LoadSpec",
    "SecondMedium",
    "Environment",
    "FarFieldRequest",
    "NearFieldRequest",
    "PrintControl",
    "ExecuteGroup",
    "NetworkCard",
    # what build_solver returns, and the roster it chooses from
    "BuiltSolver",
    "PortPlan",
    "PortSite",
    "BASES",
    # the card reader, for a consumer that must echo a deck as written
    "Card",
    "parse_card",
    "tokenize",
    "DeckError",
]

# One entry per shipped dialect.  The mapping is the error message's source
# as well as the dispatch table, so a new front-end cannot be added without
# the refusal learning its name.
_DIALECTS = {"nec2": parse_nec2}


def parse(text: str, dialect: str = "nec2") -> DeckModel:
    """Parse a deck into a :class:`DeckModel`.

    ``dialect="nec2"`` is the only value this release ships.  Raises
    :class:`DeckError` with the spec's message for anything the dialect will
    not run.
    """
    front_end = _DIALECTS.get(dialect)
    if front_end is None:
        known = ", ".join(repr(name) for name in sorted(_DIALECTS))
        raise DeckError(f"unknown deck dialect {dialect!r}; known dialects: {known}")
    return front_end(text)
