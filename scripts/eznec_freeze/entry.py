"""PyInstaller entry point for the packaged EZNEC drop-in executable.

The frozen program is :mod:`momwire.dropin.nec5`'s process shell and nothing else:
``momwire-engine <deck> <printout>``, the same two-argument protocol EZNEC
Pro+ v7 uses to launch its external NEC-5 console engine.  A separate script
(rather than ``-m momwire.dropin.nec5``) exists only because PyInstaller wants a
file to trace from.

**The executable's NAME picks the formulation** (momwire#593).  EZNEC owns
the command line — it sends two positional paths and nothing else — so the
basis cannot be a flag, and `_shell.main`'s own contract says it must not
become one.  What EZNEC does give the user is a file picker for the engine
path, so the choice rides on the filename: ``momwire-eznec-<basis>.exe``
names ``<basis>``, and a name without the marker serves the default.  Those
were the shipped names until the native launchers took them (momwire#718
phase 3), and the launchers themselves were dropped in momwire#1583; the rule
stays, for a renamed copy of this exe.

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

The frozen exe is also the resident daemon's WORKER (momwire#1418): the
daemon solves in a pool of processes, and in a bundle the only program to
start is this one, so ``--serve-worker`` as the FIRST argument makes it a
worker and nothing else.

The frozen exe is also the resident daemon (momwire#718 phase 3): in a
deployed bundle there is no system Python, so the native client's only spawn
target is the bundle's own exe, and ``--serve`` dispatches it exactly as
``python -m momwire.dropin.nec5 --serve`` does.  This is not a flag on the
EZNEC-facing path — EZNEC's spelling, two quoted cwd-relative paths, can
never produce it — it is machinery only the thin client speaks.

**A name carrying ``nec2c`` is the NEC-2 engine** (momwire#2059), and that is
a different PROTOCOL as well as a different dialect: nec2c's command line and
SimNEC's portal -- decks on stdin and printouts on stdout, NX-framed, the
``-version`` probe -- not EZNEC's two paths.  So that name is handed, whole,
to ``momwire.dropin.nec2c.main``: the very function the pip console scripts
``momwire-nec2c[-<basis>]`` run, given the same program name, so the basis is
read past ``nec2c-`` by the pip launcher's own rule and every byte it writes
is the pip launcher's.  Checked FIRST, in SimNEC's order (``nec2c``, then
``nec5``, momwire#1239).  The bundle's NEC-2 launchers are copies of the
native client, which passes its own name as ``--program <name>`` ahead of the
untouched argv; a copy of THIS exe under a ``nec2c`` name reads its own.
"""

import sys

import momwire_serve_client

# AVX-512 off before the imports below bring numpy in: the frozen engine is
# its own interpreter (EZNEC, the native client and the resident pool all
# start THIS exe), so this line is the bundle's one chance to set it.
momwire_serve_client.pin_cpu_env()
# OpenBLAS at one thread as it loads, for the same reason and at the same
# moment: Windows cannot shrink the pool later (momwire#2023).
momwire_serve_client.pin_blas_pool()

from momwire.deck._solver import (  # noqa: E402 - after the AVX-512 pin
    basis_from_program_name,
    dialect_from_program_name,
    nec4_basis_from_program_name,
)
from momwire.dropin.nec5 import _serve  # noqa: E402 - after the AVX-512 pin
from momwire.dropin.nec5._shell import main  # noqa: E402 - after the AVX-512 pin
from momwire.serve._pool import WORKER_FLAG  # noqa: E402 - after the AVX-512 pin

MARKER = "eznec-"

# The frozen exe's own segment once the native client took the launcher
# names (momwire#718 phase 3): the bundle shipped ONE ``momwire-eznec-engine``,
# and this segment is consumed exactly as the thin client consumes ``client``
# — the plain name selects nothing, a renamed copy still names its basis.
# Since momwire#1583 the engine ships as ``momwire-engine``, which carries no
# marker at all and so selects the default the same way; the launchers always
# say ``--basis``.
ENGINE_SEGMENT = "engine"

# The launcher's one-shot rung for a run whose paths came from stdin
# (momwire#1488): spelt again in `momwire_eznec_client.c`.
EN_OPTIONAL_FLAG = "--en-optional"

# The name a native launcher answers as (momwire#2059), LEADING only:
# ``--program <launcher name> <the launcher's own argv>``.  The NEC-2
# launchers forward everything they were given through it, because the NEC-2
# route reads the basis AND the ``-version`` line off the program name and
# must see the launcher's, not this exe's.  Spelt again in
# `momwire_eznec_client.c`.
PROGRAM_FLAG = "--program"

# `python -m momwire.dropin.nec2c`'s spelling, answered as the plain NEC-2
# program (`run` says why).
NEC2C_MODULE = "momwire.dropin.nec2c"
NEC2C_PROGRAM = "momwire-nec2c"


def dialect_for(prog: str) -> str:
    """The dialect this program name answers: ``"nec2"`` when the name
    carries ``nec2c`` (momwire#2059), else ``"nec4"`` when it carries
    ``nec4`` (momwire#1295), else ``"nec5"``.  The bundle's own engine is
    named for none of them, so a deployed EZNEC launcher passes ``--dialect``
    instead, exactly as it passes ``--basis``, and a NEC-2 one passes its
    name (`PROGRAM_FLAG`)."""
    return dialect_from_program_name(prog)


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
    if dialect_for(prog) == "nec4":
        # A nec4 name reads its basis past `nec42` (else `nec4`, momwire#1443)
        # alone: the NEC-5 markers cannot coexist with it in a name that
        # means one slot.
        suffix = nec4_basis_from_program_name(prog)
    else:
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

    The one-shot path takes the same machinery flags, LEADING only:
    ``--basis <name> <deck> <printout>`` is how the native client's fallback
    rung runs a twin through the bundle's single engine exe (momwire#718
    phase 3 — the per-variant frozen stubs are what that client subsumes),
    and ``--dialect nec4`` before or after it is how a NEC-4.2-slot launcher
    does the same (momwire#1295).  ``--en-optional`` is the third: a launcher
    that read its two paths from stdin (4nec2's form, momwire#1488) hands
    them to this one-shot on the command line, and the flag carries the
    allowance that came with them -- the deck may end without ``EN``.
    EZNEC's spelling cannot produce any of them, and after the flags the
    contract is the untouched two positional paths.  An EMPTY command line is
    the engine's own stdin form, which ``main`` reads.

    ``--program <name>`` (momwire#2059) comes before all of them and names the
    program this run answers as, in place of ``argv[0]``.  A ``nec2c`` name --
    the NEC-2 launchers' -- leaves this function for
    ``momwire.dropin.nec2c.main`` with the rest of the argv untouched, so the
    NEC-2 call forms (stdin decks, ``-version``, ``in out``, ``- -``, 4nec2's
    names) are that function's, byte for byte; no flag above applies to it.
    """
    rest = argv[1:]
    if rest[:1] == [WORKER_FLAG]:
        # A worker of the resident server's pool (momwire#1418): the daemon
        # starts THIS exe again, because a frozen bundle has no other
        # interpreter. The seam to build arrives on stdin, so neither the name
        # nor any flag below applies.
        from momwire.serve._worker import worker_main

        return worker_main(rest[1:])
    prog = argv[0]
    if len(rest) >= 2 and rest[0] == PROGRAM_FLAG:
        prog, rest = rest[1], rest[2:]
    elif rest[:2] == ["-m", NEC2C_MODULE]:
        # `python -m momwire.dropin.nec2c`, which is how the NEC-2 route's
        # `--selftest` starts its copies of itself (`sys.executable -m ...`):
        # in a bundle `sys.executable` is THIS exe, so without this the
        # selftest's children would answer as the NEC-5 one-shot.
        prog, rest = NEC2C_PROGRAM, rest[2:]
    if dialect_for(prog) == "nec2":
        # Imported here, not at the top: the EZNEC one-shot and the daemon
        # must not pay for a front end they never reach (the bundle collects
        # every momwire submodule regardless, so the freeze still finds it).
        from momwire.dropin.nec2c import main as nec2c_main

        return nec2c_main(rest, prog=prog)
    basis = basis_for(prog)
    dialect = dialect_for(prog)
    if "--serve" in rest:
        from momwire.dropin.nec5._resident import serve_main

        if "--basis" not in rest:
            rest = [*rest, "--basis", basis]
        if dialect != "nec5" and "--dialect" not in rest:
            rest = [*rest, "--dialect", dialect]
        return serve_main(rest)
    en_optional = False
    while rest:
        if len(rest) >= 2 and rest[0] in ("--basis", "--dialect"):
            if rest[0] == "--basis":
                basis = rest[1]
            else:
                dialect = rest[1]
            rest = rest[2:]
        elif rest[0] == EN_OPTIONAL_FLAG:
            en_optional = True
            rest = rest[1:]
        else:
            break
    if en_optional:
        return main(rest, basis=basis, dialect=dialect, en_optional=True)
    return main(rest, basis=basis, dialect=dialect)


if __name__ == "__main__":
    sys.exit(run(sys.argv))
