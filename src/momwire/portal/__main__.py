"""``python -m momwire.portal`` — the console script's long spelling.

The ``momwire-nec2c`` entry point is what SimNEC is pointed at (its dialog
accepts an engine on the FILENAME alone, so the name is contract).  This is
the same ``main`` reachable without a console script on the path: what the
selftest spawns, what the tests drive from an unrelated cwd, and what a user
with an odd install layout can always fall back to.
"""

from __future__ import annotations

from ._portal import main

if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
