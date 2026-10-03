"""EZNEC's External NEC-4.2 slot: the ``nec4`` dialect behind the same shell.

momwire#1295.  A program NAME carrying ``nec4``
(:func:`~momwire.deck._solver.dialect_from_program_name`) makes the one-shot
shell answer EZNEC's NEC-4.2 slot instead of its NEC-5 one.  The protocol is
the NEC-5 slot's exactly, as captured in 0223-0239: ``"<engine>.exe" "EZ.NEC"
"NEC.OUT"``, two quoted cwd-relative paths, cwd the engine's folder, no stdin,
exit status unread.  Only the deck dialect and the printout differ, and both
are decided here.

The deck is read by :func:`momwire.deck._nec4.parse_nec4`, which addresses a
source by segment CENTRE — so the slot's roster is
:data:`~momwire.deck.NEC2_BASES`, the families whose ``centre_feeds``
capability holds.  A basis outside it (razor-2p, razor-nec5) is refused BY
NAME in the printout before any deck is read: razor places a gap only on a
knot, and moving a centre feed half a cell silently is momwire#821's failure.

PHASE 1 PRINTOUT — NOT YET THE NEC-4 LAYOUT.  The body below the header is the
nec2 portal's printout (:func:`momwire.portal._portal.render_deck`), the NEC-2
block sequence the solve notes say NEC-4.2 shares in outline (structure,
segmentation, frequency, environment, loading, then input parameters,
currents, power budget and pattern).  It is the printout the ``DeckModel``
pipeline already writes, which is why it was chosen over the NEC-5 layout: that
writer is fed by the node-addressed NEC-5 deck, which cannot carry a centre
feed at all.  Phase 2 replaces this body with the layout EZNEC's NEC-4.2 reader
accepts; until then :data:`PHASE1_MARK` stands at the top of every served
printout so nobody mistakes it for that one.  The header (comment-box stamp
and line-2 engine stamp) is the NEC-5 slot's, unchanged, and refusals are its
``NEC ERROR`` line, the channel EZNEC displays.
"""

from __future__ import annotations

from ..deck import BASES, NEC2_BASES
from . import _printout

__all__ = ["PHASE1_MARK", "basis_refusal", "render"]

# The line every phase-1 served printout carries directly under the header.
PHASE1_MARK = (
    " ***** MOMWIRE NEC-4.2 SLOT, PHASE 1 (momwire#1295): NEC-2 LAYOUT BODY,"
    " NOT YET THE NEC-4 PRINTOUT"
)


def basis_refusal(basis: str) -> str | None:
    """The sentence refusing ``basis`` in this slot, or ``None`` if it serves.

    Two cases, told apart so the user can act on the right one: a name the
    roster does not know is a typo and the sentence lists the choices; a
    roster basis that cannot place a centre feed is refused in its own
    family's declared words.
    """
    if basis in NEC2_BASES:
        return None
    if basis in BASES:
        cls, _kwargs = BASES[basis]
        return (
            f"basis {basis!r} does not serve EZNEC's NEC-4.2 slot, which "
            f"addresses sources at segment centres: "
            f"{cls.capabilities.refusal('centre_feeds')}"
        )
    known = ", ".join(repr(name) for name in NEC2_BASES)
    return f"unknown basis {basis!r} for EZNEC's NEC-4.2 slot; known bases: {known}"


def render(text: str, *, basis: str) -> str:
    """One NEC-4.2-dialect deck as a printout: the answer, or a named refusal.

    Every refusal the nec2 portal's renderer would print as an ``ERROR:``
    line — the dialect's, the solver's, a basis that cannot host this deck —
    comes back here as the NEC-5 slot's ``NEC ERROR`` printout instead, the
    first such line naming the reason.  One refused run refuses the deck: a
    partial printout would read as a complete one.
    """
    # Imported on use: the NEC-5 slot's one-shot never pays for the portal.
    from ..portal._portal import _ERROR_TOKEN, _STRUCTURE_HEADER, render_deck

    refused = basis_refusal(basis)
    if refused is not None:
        return _printout.render_refusal(text, refused, basis=basis)
    out, _err = render_deck(text, dialect="nec4", basis=basis)
    for line in out:
        if line.startswith(_ERROR_TOKEN):
            return _printout.render_refusal(
                text, line[len(_ERROR_TOKEN) :], basis=basis
            )
    body = out[out.index(_STRUCTURE_HEADER) :]
    return (
        _printout.render_header(text, basis=basis)
        + PHASE1_MARK
        + "\n"
        + "\n".join(body)
        + "\n"
    )
