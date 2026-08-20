"""momwire standing in for the NEC-5 console engine EZNEC Pro+ v7 launches.

EZNEC drives its external engine as a one-shot process — deck in, printout
out, one launch per frequency point — so this package is a program, not a
server: the protocol delta against :mod:`momwire.portal`, which is a resident
daemon SimNEC talks to over stdin.  ``python -m momwire.eznec <deck> <printout>``
is the whole interface.

Unit 1 (issue #497) builds the SEAM: the process shell, the printout header,
and a stub that refuses every deck by name.  The dialect parser (U2), the
renderer (U3), rung-1 physics (U4) and node-addressed networks (U5) plug in
behind it.  A refusing engine that speaks the protocol correctly is already
useful — it tells the operator, in EZNEC's own viewer, why their model was
not served — and it is the only order in which the later units can be
byte-gated at all.

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

from ._printout import STUB_REFUSAL, render_header, render_refusal
from ._shell import main, run

__all__ = [
    "STUB_REFUSAL",
    "main",
    "render_header",
    "render_refusal",
    "run",
]
