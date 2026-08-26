"""PyInstaller entry point for the packaged EZNEC drop-in executable.

The frozen program is :mod:`momwire.eznec`'s process shell and nothing else:
``momwire-eznec <deck> <printout>``, the same two-argument protocol EZNEC
Pro+ v7 uses to launch its external NEC-5 console engine.  A separate script
(rather than ``-m momwire.eznec``) exists only because PyInstaller wants a
file to trace from.

**The executable's NAME picks the formulation** (momwire#593).  EZNEC owns
the command line — it sends two positional paths and nothing else — so the
basis cannot be a flag, and `_shell.main`'s own contract says it must not
become one.  What EZNEC does give the user is a file picker for the engine
path, so the choice rides on the filename:

    momwire-eznec.exe                 the default, degree-2 B-spline
    momwire-eznec-razor-nec5.exe      the NEC-5 formulation twin

This is momwire#528's spelling, not a second one — the same rule, through the
same owner (`basis_from_program_name`), with ``eznec-`` as the marker where
the portal uses ``nec2c-``.  So **a copy you make yourself works**: rename or
copy the exe to ``momwire-eznec-<basis>.exe`` beside its ``_internal`` and
that basis is what answers — in whatever casing Explorer or the user gave the
name, because the match is casefolded and Windows filenames are.

An unknown suffix must not fall back to the default silently — that is
momwire#628's failure, an engine answering as a formulation nobody asked
for — and it must not raise either, because EZNEC reads the printout file and
nothing else.  So it is resolved HERE, before any deck is read, and a bad name
is handed to the shell as a basis it will refuse by name in the printout.
"""

import sys

from momwire.deck._solver import basis_from_program_name
from momwire.eznec import _serve
from momwire.eznec._shell import main

MARKER = "eznec-"


def basis_for(prog: str) -> str:
    """The basis this program name selects, defaulting to the seam's own.

    Returns the SUFFIX unvalidated whenever the name carries the marker, and
    the default only when it does not.  The EMPTY suffix of a name ending at
    the marker (``momwire-eznec.exe`` copied to ``momwire-eznec-.exe``) is a
    suffix and travels as one: it asked for a basis and named none, so it must
    reach the refusal rather than be rounded off to the default — `or` here
    would have made that name serve bs2 silently.

    Unvalidated because the refusal is not this module's to write: `_serve.serve`
    resolves the name through `basis_entry` and raises `ServeRefusal` carrying
    its sentence, and `_shell.render` catches that and prints it as the
    ``NEC ERROR`` line naming the basis and listing the known ones — the only
    channel EZNEC reads.  Validating here could only turn that into a traceback
    on a stream nobody sees.
    """
    suffix = basis_from_program_name(prog, MARKER)
    return _serve.BASIS if suffix is None else suffix


if __name__ == "__main__":
    sys.exit(main(basis=basis_for(sys.argv[0])))
