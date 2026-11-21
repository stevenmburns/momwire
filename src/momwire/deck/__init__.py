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
from ._nec2 import _Nec2Parser, parse_nec2, parsed_with_structure
from ._nec4 import _Nec4Parser, parse_nec4
from ._nec5 import (
    Nec5Deck,
    Nec5ExecuteRequest,
    Nec5FarFieldRequest,
    Nec5FreeSpace,
    Nec5Load,
    Nec5MininecGround,
    Nec5NearFieldRequest,
    Nec5Network,
    Nec5Node,
    Nec5PerfectGround,
    Nec5SommerfeldGround,
    Nec5Source,
    Nec5TransmissionLine,
    Nec5Wire,
    parse_nec5,
)
from .._basis_names import BASIS_ALIASES, canonical_basis
from ._solver import (
    BASES,
    NEC2_BASES,
    BuiltSolver,
    ExtendedKernelAboveGround,
    ExtendedKernelDefault,
    PortPlan,
    PortSite,
    PreparedMesh,
    build_solver,
    prepare_mesh,
)
from .model import (
    DeckModel,
    DeckWire,
    DistributedRLC,
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
    "DistributedRLC",
    "LoadSpec",
    "SecondMedium",
    "Environment",
    "FarFieldRequest",
    "NearFieldRequest",
    "PrintControl",
    "ExecuteGroup",
    "NetworkCard",
    # the nec5 dialect, which parses into its OWN model (see below)
    "parse_nec4",
    "parse_nec5",
    "Nec5Deck",
    "Nec5Node",
    "Nec5Wire",
    "Nec5FreeSpace",
    "Nec5PerfectGround",
    "Nec5SommerfeldGround",
    "Nec5MininecGround",
    "Nec5Source",
    "Nec5Load",
    "Nec5TransmissionLine",
    "Nec5Network",
    "Nec5FarFieldRequest",
    "Nec5NearFieldRequest",
    "Nec5ExecuteRequest",
    # what build_solver returns, and the roster it chooses from
    "BuiltSolver",
    "ExtendedKernelAboveGround",
    "ExtendedKernelDefault",
    "PortPlan",
    "PortSite",
    "BASES",
    "NEC2_BASES",
    "BASIS_ALIASES",
    "canonical_basis",
    # the card reader, for a consumer that must echo a deck as written
    "Card",
    "parse_card",
    "tokenize",
    "DeckError",
]

# One entry per shipped dialect THAT PARSES INTO `DeckModel`.  The mapping is
# the error message's source as well as the dispatch table, so a new
# front-end cannot be added without the refusal learning its name.
#
# ``nec5`` is deliberately absent.  NEC-5's connection cards address nodes
# through a favored wire, which `DeckModel` has no vocabulary for and must
# not be given one by aliasing (the W7EL gate: two addresses naming one
# geometric point are NOT interchangeable), so :func:`parse_nec5` returns its
# own :class:`Nec5Deck` and is called by name rather than through ``parse``,
# whose contract is a `DeckModel`.
#
# ``nec4`` (momwire#1295) is ``nec2`` plus the NEC-4.2 deltas EZNEC's External
# NEC-4.2 slot writes; it addresses segment centres and parses into
# `DeckModel` like its parent.
_DIALECTS = {"nec2": parse_nec2, "nec4": parse_nec4}


def parse(text: str, dialect: str = "nec2") -> DeckModel:
    """Parse a deck into a :class:`DeckModel`.

    ``dialect`` is ``"nec2"`` or ``"nec4"`` (:mod:`._nec4`).  Raises
    :class:`DeckError` with the spec's message for anything the dialect will
    not run.
    """
    front_end = _DIALECTS.get(dialect)
    if front_end is None:
        known = ", ".join(repr(name) for name in sorted(_DIALECTS))
        raise DeckError(f"unknown deck dialect {dialect!r}; known dialects: {known}")
    return front_end(text)


# The parser behind each of `_DIALECTS`' front ends.
_PARSERS = {"nec2": _Nec2Parser, "nec4": _Nec4Parser}


def parse_with_structure(text: str, dialect: str = "nec2"):
    """`parse`, plus the flat NEC geometry the dialect built while reading
    the deck (a `~momwire.deck._nec2_geometry.Nec2Structure`: the wires after
    every transform, the tags, the ``(tag, segment)`` resolver). The nec2c
    portal's printout reads both, and used to build the geometry a second time
    (momwire#1929). The same refusals as `parse`, in the same order."""
    parser = _PARSERS.get(dialect)
    if parser is None:
        parse(text, dialect)  # the unknown-dialect refusal, in its words
    return parsed_with_structure(parser(), text)
