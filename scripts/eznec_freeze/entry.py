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
    momwire-eznec-razor-2p.exe        the tent basis with razor-blade path
                                      testing, at NEC-5's two-point rule
    momwire-eznec-razor-nec5.exe      the deprecated spelling of razor-2p

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

The frozen exe is also the resident daemon (momwire#718 phase 3): in a
deployed bundle there is no system Python, so the native client's only spawn
target is the bundle's own exe, and ``--serve`` dispatches it exactly as
``python -m momwire.eznec --serve`` does.  This is not a flag on the
EZNEC-facing path — EZNEC's spelling, two quoted cwd-relative paths, can
never produce it — it is machinery only the thin client speaks.
"""

import sys

from momwire.deck._solver import basis_from_program_name
from momwire.eznec import _serve
from momwire.eznec._shell import main

MARKER = "eznec-"

# The frozen exe's own name once the native client takes the
# ``momwire-eznec[-<basis>]`` names (momwire#718 phase 3): the bundle ships
# ONE ``momwire-eznec-engine``, and this segment is consumed exactly as the
# thin client consumes ``client`` — the plain name selects nothing, a renamed
# copy still names its basis.
ENGINE_SEGMENT = "engine"


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
    suffix = basis_from_program_name(prog, MARKER, consumed=ENGINE_SEGMENT)
    return _serve.BASIS if suffix is None else suffix


def run(argv: list[str]) -> int:
    """One frozen exe, two duties, told apart by ``--serve``.

    The filename rule holds on both paths: a twin-named copy started with
    ``--serve`` must serve what its name claims, so when the spawner names no
    ``--basis`` the name's basis rides in as the flag — verbatim, so the
    empty suffix still reaches `serve_main` as the basis ``""`` and refuses
    per-deck rather than serving the default.  A spawner that does say
    ``--basis`` is believed: that argv came from our own machinery, never
    from EZNEC, and the flag is its explicit spelling of the same choice.

    The one-shot path takes the same machinery flag, LEADING only:
    ``--basis <name> <deck> <printout>`` is how the native client's fallback
    rung runs a twin through the bundle's single engine exe (momwire#718
    phase 3 — the per-variant frozen stubs are what that client subsumes).
    EZNEC's spelling cannot produce it, and after the flag the contract is
    the untouched two positional paths.
    """
    basis = basis_for(argv[0])
    rest = argv[1:]
    if "--serve" in rest:
        from momwire.eznec._resident import serve_main

        if "--basis" not in rest:
            rest = [*rest, "--basis", basis]
        return serve_main(rest)
    if len(rest) >= 2 and rest[0] == "--basis":
        basis, rest = rest[1], rest[2:]
    return main(rest, basis=basis)


if __name__ == "__main__":
    sys.exit(run(sys.argv))
