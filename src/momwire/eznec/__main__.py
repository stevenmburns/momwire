"""``python -m momwire.eznec <deck> <printout>`` — the engine EZNEC launches.

The invocation is positional and cwd-relative, exactly as EZNEC spells it for
its own console engine:

    python -m momwire.eznec EZN5.NEC NEC5.OUT

Quoting is the shell's business — EZNEC quotes both paths, and argv arrives
already unquoted, so a path with spaces needs nothing special here.

Two deliberate omissions, both from the capture study (antennaknobs
``docs/status/2026-08-16-eznec-nec5-dialect-capture.md``):

* **No flags on the EZNEC-facing path.**  The real engine's ``-i``/``-o``
  forms fail on their own ("UNABLE TO OPEN FILE -i", exit 0) and EZNEC never
  sends them.  ``--serve`` below is not an exception to this: it is machinery
  the thin client spawns (momwire#532, same shape as the portal's), and
  EZNEC's own spelling — exactly two quoted cwd-relative paths — can never
  produce it.
* **No prompt loop.**  Given no arguments the real engine prompts for a
  filename, but reads the answer from the console device rather than stdin,
  so it cannot be scripted and EZNEC never takes that path.  This engine
  answers a short command line with an error line on stdout and exit 0
  instead, and never touches stdin at all.

The exit status is always 0 and is never a refusal channel: EZNEC does not
read it (fault injection returned a perfect printout with exit 1 and EZNEC
displayed normal results).  Refusals live in the printout.
"""

from __future__ import annotations

import sys

from ._shell import main

if __name__ == "__main__":  # pragma: no cover - process entry point
    if "--serve" in sys.argv[1:]:
        from ._resident import serve_main

        raise SystemExit(serve_main(sys.argv[1:]))
    raise SystemExit(main())
