"""momwire's executable front door: the SimNEC ``nec2c`` portal daemon.

``pip install momwire`` puts a ``momwire-nec2c`` command on the path, and
pointing SimNEC's portal dialog at it makes momwire a resident NEC engine —
a peer of ``nec2c`` and ``nec5cl``.  ``python -m momwire.portal`` is the same
program under its long name.  Usage:
https://momwire.dev/reference/portal-usage/

The implementation is one module, :mod:`momwire.portal._portal`; this package
re-exports what a caller may use.  The split is deliberate: the portal speaks
a PROTOCOL (column-exact NEC-2 printout, the ``NX`` framing, the ``-version``
handshake) and momwire speaks physics, and the seam between them is the
subject of the isolation rule below.

The isolation rule (#846 §4)
----------------------------
``momwire.portal`` may import momwire's public solver API and
:mod:`momwire.deck`.  **Nothing outside ``momwire/portal/`` may import from
``momwire.portal``.**  Living inside the engine repo is what deletes portal-
vs-solver version skew, and this one-way rule is what stops it costing the
engine its identity: no SimNEC protocol detail may become load-bearing for a
solver.  It is enforced by ``tests/test_portal.py``'s source grep over
``src/momwire/``, not by convention.
"""

from __future__ import annotations

from ._portal import (
    BANNER_VERSION,
    LEGACY_PROBE_VERSION,
    PROBE_VERSION,
    deck_frame,
    engine_scope,
    main,
    render_deck,
    run_deck,
)

__all__ = [
    "BANNER_VERSION",
    "LEGACY_PROBE_VERSION",
    "PROBE_VERSION",
    "deck_frame",
    "engine_scope",
    "main",
    "render_deck",
    "run_deck",
]
