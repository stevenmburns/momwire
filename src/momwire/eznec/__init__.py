"""momwire standing in for the NEC-5 console engine EZNEC Pro+ v7 launches.

EZNEC drives its external engine as a one-shot process — deck in, printout
out, one launch per frequency point — so this package is a program, not a
server: the protocol delta against :mod:`momwire.portal`, which is a resident
daemon SimNEC talks to over stdin.  ``python -m momwire.eznec <deck> <printout>``
is the whole interface.

Issue #497 builds this seam in units.  U1 laid down the SEAM itself — the
process shell, the printout header, and a stub that refuses every deck by
name; U2 added the nec5 dialect front-end (:mod:`momwire.deck._nec5`); U3
added the rest of the printout, so this package can lay out a complete NEC-5
result file given the numbers; U4 (:mod:`._serve`) is the physics that
PRODUCES them for rung 1 of the scored ladder — free space and perfect
ground, one source at a node, fixed-impedance loads, ``RP 0``/``XQ``/``PQ 0``
— and wires the renderer into the shell in place of the stub refusal.  U5
added node-addressed ``TL``/``NT`` networks, which is what turns a seam that
serves bare wires into one that serves feed systems: EZNEC parks the whole
network — source included — on a virtual wire 100 λ away and reaches the
antenna through transmission lines, and a deck the user thinks carries only
lumped loads can still arrive as ``NT`` cards.

Issue #504 climbs the ground rungs on top of that, and finishes them.  Its U1
added ``GN 0`` / ``GN 2``, the finite Sommerfeld ground — the environment most
of the corpus actually stands over — taking the served count from twenty of
the forty-nine captured decks to twenty-seven.  Its U2 added the bare ``GD``,
EZNEC's "Real / MININEC type" ground, which is the corpus's second-most-common
and answers with PEC currents under a second-medium far field; that takes the
count to forty-three, and every ground card this dialect writes is now served.
Its U3 renders the ``NETWORK DATA`` table that carries ``TL`` and ``NT`` rows
at once — one heading, one sub-table each, in the order the decks write them —
and takes the count to FORTY-SIX.

The three still refused are two phased multi-``EX`` drives and one ``NE`` near
field, and each names its own card.  Nothing in the corpus is refused for a
ground or for a table layout any more.

What U3 mostly did, though, was land printouts.  The U1/U2 capture errand was
scoped off the fixture manifest rather than off the corpus, and every deck
those rungs served already had a printout waiting in the capture tree:
nineteen of them landed, and the arc's structure gates and envelope pins went
from thirteen captures to thirty-nine.  Two of the nineteen then disagreed —
the elevated radial systems 0033/0034, whose radials stand 1e-4 λ over a
``GN 0`` earth and whose impedance lands 275 Ω and 188 Ω from the engine's.
Their LAYOUT is gated like everything else and their physics is pinned
nowhere; the divergence is momwire's finite-ground solve at grazing height
rather than anything this package can reach, and it is named in
``tests/test_eznec_serve.py`` rather than hidden inside a wide bar.

Everything above the served rungs still refuses, and refuses BY NAME.  That is not a
placeholder: a refusing engine that speaks the protocol correctly tells the
operator, in EZNEC's own viewer, exactly which card it could not stand
behind, which is the one thing a wrong answer cannot do.

The module path is U1's naming decision: ``momwire.eznec`` names the
APPLICATION on the far side of the seam (EZNEC), not the dialect underneath
it (NEC-5) and not the transport, matching how ``momwire.portal`` is named for
SimNEC's portal dialog rather than for nec2c.  ``momwire.deck``'s NEC-5
dialect, when U2 lands it, is a sibling of the nec2 grammar and keeps its own
name; nothing outside this package should have to know that EZNEC is what
asked.

Courtesy stance
---------------
Every behaviour in this package was derived from CAPTURED INPUT/OUTPUT ONLY:
the decks EZNEC wrote, the printouts the user's own licensed engine left
behind (``tests/fixtures/eznec/``), and the two interface studies —
antennaknobs ``docs/status/2026-08-16-eznec-nec5-dialect-capture.md`` and
``docs/status/2026-08-20-eznec-nec5-scored-matrix.md``.  NEC-5 itself is
LLNL-copyrighted, user-licensed and export-controlled: no source, algorithm,
or internal structure of it is described, quoted, or relied on anywhere here,
and none may be added.
"""

from __future__ import annotations

from ._printout import (
    ENVIRONMENT_FINITE_GROUND,
    ENVIRONMENT_FREE_SPACE,
    ENVIRONMENT_PERFECT_GROUND,
    STUB_REFUSAL,
    ChargeRow,
    GroundMedium,
    LineRow,
    LoadRow,
    NetworkRow,
    PatternBlock,
    PatternRow,
    PortRow,
    PowerBudget,
    RunData,
    WireCurrentRow,
    render_header,
    render_printout,
    render_refusal,
)
from ._serve import refusal, serve
from ._shell import main, render, run

__all__ = [
    "ENVIRONMENT_FINITE_GROUND",
    "ENVIRONMENT_FREE_SPACE",
    "ENVIRONMENT_PERFECT_GROUND",
    "STUB_REFUSAL",
    "ChargeRow",
    "GroundMedium",
    "LineRow",
    "LoadRow",
    "NetworkRow",
    "PatternBlock",
    "PatternRow",
    "PortRow",
    "PowerBudget",
    "RunData",
    "WireCurrentRow",
    "main",
    "refusal",
    "render",
    "render_header",
    "render_printout",
    "render_refusal",
    "run",
    "serve",
]
