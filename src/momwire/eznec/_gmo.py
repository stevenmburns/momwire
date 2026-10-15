"""The ``.GMO`` geometry file 4nec2's NEC-5 package reads (momwire#1488).

4nec2 draws its geometry window from a printout's SEGMENTATION DATA table,
which a NEC-5 printout does not carry.  The "NEC-5 with 4nec2" package
rebuilds that table from a file the engine writes beside the printout when
the deck's ``GE`` card carries a third field of 1 (``GE 0 0 1``):
``<printout stem>.GMO``.

Everything here is read off ONE sample: Ward AE6TY's razor-2p port (76.7)
run on Dan AC6LA's 4nec2 Example 3 (``tests/fixtures/4nec2_example3/``),
which this module reproduces byte for byte.  No NEC-5 source or manual was
read.  The layout, in the sample's own terms:

* the node count, ``%6d``;
* one line per node, `` %13.5E%13.5E%13.5E`` (x, y, z in metres);
* the element count, ``%6d``;
* one line per element, ``%7d%7d%13.5E`` (first node, second node, radius);
* CRLF after every line, the last included.

The numbering is the one the printout's ``Number of nodes`` already counts
(:func:`momwire.eznec._serve.structure_of`): a node is a fused point on the
1 µm grid, so a junction is ONE node, and the sample's one-segment centre
wire adds no node of its own because both its ends fuse into the arms'.
Nodes are numbered in order of first appearance, walking the ``GW`` cards in
deck order and each wire from end 1 to end 2; elements are the segments in
the same order, each from its end-1-side node to its end-2-side node.  The
sample cannot tell first appearance from some other order (its x coordinate
rises monotonically along the whole structure), so that choice, the ground
plane's image, and an ``NX`` chain's file are open questions for Ward, listed
in the PR for momwire#1488.
"""

from __future__ import annotations

import momwire_serve_client as _mech

from ..deck import DeckError
from ..deck._cards import parse_card
from ..deck._nec5 import Nec5Deck, parse_nec5, split_structures
from ._serve import _wire_nodes

__all__ = [
    "GMO_SUFFIX",
    "requests_gmo",
    "gmo_path",
    "gmo_text",
    "render_gmo",
]

# The sample is named ``EXAMPLE3.GMO`` beside ``EXAMPLE3.out``: the stem of
# the printout, the suffix upper case whatever the printout's is.
GMO_SUFFIX = _mech.GMO_SUFFIX


def gmo_path(printout_path: str) -> str:
    """``<printout stem>.GMO`` -- :func:`momwire_serve_client.gmo_path`, the
    one Python owner of the rule (the thin client cannot import momwire, and
    the native launcher spells it again in C)."""
    return _mech.gmo_path(printout_path)


def render_gmo(deck: Nec5Deck) -> str:
    """The ``.GMO`` file for one parsed deck, ``\\n`` line ends.

    Line ends are the writer's: the one-shot shell writes CRLF and the
    resident server's socket wrapper translates to CRLF, exactly as both do
    for the printout.
    """
    index: dict[tuple[int, int, int], int] = {}
    points: list[tuple[float, float, float]] = []
    elements: list[tuple[int, int, float]] = []
    for wire in deck.wires:
        wire_points, keys = _wire_nodes(wire)
        numbers: list[int] = []
        for point, key in zip(wire_points, keys, strict=True):
            number = index.get(key)
            if number is None:
                number = len(points) + 1
                index[key] = number
                points.append(point)
            numbers.append(number)
        for first, second in zip(numbers, numbers[1:]):
            elements.append((first, second, wire.radius))
    lines = [f"{len(points):6d}"]
    lines += [" " + "".join(f"{float(c):13.5E}" for c in point) for point in points]
    lines.append(f"{len(elements):6d}")
    lines += [
        f"{first:7d}{second:7d}{float(radius):13.5E}"
        for first, second, radius in elements
    ]
    return "\n".join(lines) + "\n"


def requests_gmo(text: str) -> bool:
    """Whether any ``GE`` card in ``text`` carries a nonzero third field.

    A sniff, so the decks EZNEC and SimNEC send (``GE`` with one or two
    fields, every one of them) never pay a second parse: only a deck that
    asks for a file is read again by :func:`gmo_text`.  A card that does not
    tokenise is left to the deck's own parse to name.
    """
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped[:2].upper() != "GE":
            continue
        try:
            card = parse_card(line)
        except (DeckError, ValueError):
            continue
        if card is not None and card.mnemonic == "GE" and card.f(2) != 0.0:
            return True
    return False


def gmo_text(text: str) -> str | None:
    """The ``.GMO`` file ``text`` asks for, or ``None`` when it asks for none.

    Written whenever the geometry was read, whether or not the solve then
    answers: the file describes the deck's geometry, and the printout says
    on its own whether the run was served.  A deck that does not parse (its
    printout names the card) gets no file.  In an ``NX`` chain each structure
    is read on its own and the LAST one that asks is the file written, as an
    engine writing the file whenever it reads such a ``GE`` would leave it.
    """
    if not requests_gmo(text):
        return None
    structures = split_structures(text)
    texts = [text] if structures is None else [s.deck_text for s in structures]
    answer: str | None = None
    for deck_text in texts:
        try:
            deck = parse_nec5(deck_text)
        except (DeckError, ValueError, OverflowError):
            continue
        if deck.geometry_file:
            answer = render_gmo(deck)
    return answer
