"""momwire standing in for the NEC-5 console engine EZNEC Pro+ v7 launches.

EZNEC drives its external engine as a one-shot process — deck in, printout
out, one launch per frequency point — so this package is a program, not a
server: the protocol delta against :mod:`momwire.portal`, which is a resident
daemon SimNEC talks to over stdin.  ``python -m momwire.eznec <deck> <printout>``
is the whole interface.

Issue #497 builds this seam in units.  U1 laid down the SEAM itself — the
process shell, the printout header, and a stub that refuses every deck by
name; U2 added the nec5 dialect front-end (:mod:`momwire.deck._nec5`); U3
added the rest of the printout, so this package can now lay out a complete
NEC-5 result file given the numbers.  Rung-1 physics (U4) and node-addressed
networks (U5) — the units that PRODUCE those numbers, and the ones that wire
the renderer into the shell in place of the stub refusal — plug in behind it.
A refusing engine that speaks the protocol correctly is already useful — it
tells the operator, in EZNEC's own viewer, why their model was not served —
and it is the only order in which the later units can be byte-gated at all.

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
    ENVIRONMENT_FREE_SPACE,
    ENVIRONMENT_PERFECT_GROUND,
    STUB_REFUSAL,
    ChargeRow,
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
from ._shell import main, run

__all__ = [
    "ENVIRONMENT_FREE_SPACE",
    "ENVIRONMENT_PERFECT_GROUND",
    "STUB_REFUSAL",
    "ChargeRow",
    "LoadRow",
    "NetworkRow",
    "PatternBlock",
    "PatternRow",
    "PortRow",
    "PowerBudget",
    "RunData",
    "WireCurrentRow",
    "main",
    "render_header",
    "render_printout",
    "render_refusal",
    "run",
]
