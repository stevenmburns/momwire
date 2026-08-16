"""``python -m momwire.portal`` — the console script's long spelling.

The ``momwire-nec2c`` entry point is what SimNEC is pointed at (its dialog
accepts an engine on the FILENAME alone, so the name is contract).  This is
the same ``main`` reachable without a console script on the path: what the
selftest spawns, what the tests drive from an unrelated cwd, and what a user
with an odd install layout can always fall back to.

Two flags select something other than the stock daemon, and both are
machinery rather than user interface: ``--serve`` is the resident server the
``momwire-nec2c-shared`` client spawns (issue #379), and
``--shared-selftest`` is that path's deployment smoke.  Neither can reach the
stock daemon's stdin — they are argv, and a deck is not — so the engine
SimNEC talks to is unchanged.  The import is lazy for the same reason: the
stock path must not pay for a module it never uses.
"""

from __future__ import annotations

import sys

from ._portal import main

_SHARED_MODES = ("--serve", "--shared-selftest")

if __name__ == "__main__":  # pragma: no cover - process entry point
    if any(mode in sys.argv[1:] for mode in _SHARED_MODES):
        from ._shared import shared_main

        raise SystemExit(shared_main(sys.argv[1:]))
    raise SystemExit(main())
