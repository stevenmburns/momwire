"""``momwire-eznec-client`` — the thin one-shot in front of the warm engine.

momwire#532, the EZNEC leg of the #718 arc: EZNEC runs its external engine
once per calculation as ``engine.exe <deck> <printout>`` and today's frozen
one-shot pays ~1 s of interpreter + NumPy import per launch. This client is
the same argv contract with the import deleted: forward the deck bytes to
the resident ``python -m momwire.eznec --serve`` daemon (spawning it if
nobody is listening), write the answered bytes to the printout path, exit 0.
The daemon transcodes (latin-1, CRLF), so what arrives IS the file — this
client never inspects, decodes, or frames anything.

The wire protocol is the seam's degenerate session (#719 U4): send the deck,
``shutdown(SHUT_WR)``, and EOF is the frame — the server answers once and
closes. No length prefixes; the native client of the arc's phase 3
reimplements exactly this exchange in C.

The fallback ladder, designed in from the start so the worst case of the
feature is "as slow as today", never "the engine is broken":

1. a warm server answers — the fast path, milliseconds;
2. nobody listening — spawn the server under the lock and wait for it;
3. the resident path fails ANYWHERE (spawn, timeout, socket death, an empty
   answer) — run the stock one-shot ``python -m momwire.eznec`` in-process
   of a child, which is correct at today's speed;
4. even that fails — write a printout carrying a named ``NEC ERROR`` line,
   because the file is the only channel EZNEC reads (rule 2 of the shell's
   three), and exit 0, because the exit code is not a channel (rule 1).

The shell's three rules bind this client exactly as they bind the engine:
exit 0 always, always write the printout, and read stdin only when the
command line is empty — 4nec2's form (momwire#1488), where the two file names
arrive on stdin and the deck may end without ``EN`` — or is ``- -``, SimNEC's
streaming call (momwire#1512), which a child one-shot shell answers on this
process's own stdin and stdout. A deck whose ``GE`` card
asks for a ``.GMO`` geometry file gets it back framed ahead of the printout
(``momwire_serve_client.split_answer``), and this client writes it beside the
printout.

Like ``momwire_nec2c_client``, this module imports stdlib only — the shared
finding-the-server mechanics live in ``momwire_serve_client`` and the tests
hold both to it.
"""

from __future__ import annotations

import os
import sys
import time as _time

import momwire_serve_client as _mech

# momwire#528/#643: the basis rides on the client's own filename —
# ``momwire-eznec-client-razor-nec5`` (or a renamed copy of the frozen exe)
# selects the twin; the ``client`` segment itself selects nothing.
_FILENAME_MARKER = "eznec-"
_CLIENT_SEGMENT = "client"
# momwire#1239: the SimNEC spelling. SimNEC picks its engine CLASS, and so
# the deck syntax it writes, from a substring of the engine's path: `nec2c`
# -> NEC-2 decks, `nec5` -> NEC-5 decks. A NEC-5 command therefore has to
# carry `nec5` in its name, which `momwire-eznec-client-razor-2p` does not.
# `momwire-nec5[-<basis>]` is the family SimNEC's NEC command setting points
# at; the basis rides past this marker exactly as it rides past `eznec-`.
_NEC5_MARKER = "nec5-"
# momwire#1295: a name carrying `nec4` (shipped as `momwire-nec42[-<basis>]`,
# momwire#1443) answers EZNEC's External NEC-4.2 slot. It is read FIRST and
# alone: the dialect is the engine's identity, and the basis then rides past
# `nec42` (else `nec4`) only.

DEFAULT_IDLE_TIMEOUT = 900.0

# EZNEC's own fault paths print these and write nothing (there is no output
# path to write to) — captured behavior, same strings as the shell's.
ARGUMENT_ERROR_INPUT = "UNABLE TO OPEN FILE"
ARGUMENT_ERROR_OUTPUT = "UNABLE TO OPEN SECOND FILE"

_RECV_CHUNK = 65536


def config_key(basis: str | None, idle_timeout: float, dialect: str = "nec5") -> str:
    """One warm server per formulation per install — the identity rule the
    portal's client established (#379): momwire version, interpreter, basis
    and idle policy are the engine; anything resident under a different
    reading must not answer.

    momwire#733: a name that selects nothing and a name that selects the
    EMPTY suffix are different engines — the first spawns the default, the
    second spawns a per-deck refusal — so ``basis or ""`` (collapsing both to
    the same element) is wrong. ``"default"`` and ``f"basis={basis}"`` cannot
    collide for any ``basis`` string, because only the second spelling can
    ever start with ``"basis="``.

    momwire#1295: the EZNEC slot is identity too — a NEC-5 server must never
    answer a NEC-4.2 deck — so a nec4 key carries ``"dialect=nec4"`` ahead of
    the basis element.  A nec5 key carries nothing new and is the key it was.
    """
    major, minor = _mech.dist_version()
    elements = [
        f"eznec.{major}.{minor}",
        os.path.realpath(sys.executable),
        f"{idle_timeout!r}",
    ]
    if dialect != "nec5":
        elements.append(f"dialect={dialect}")
    elements.append("default" if basis is None else f"basis={basis}")
    return _mech.digest(elements)


def _server_command(
    path: str,
    basis: str | None,
    idle_timeout: float,
    log_path: str,
    dialect: str = "nec5",
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "momwire.eznec",
        "--serve",
        "--socket",
        path,
        "--idle-timeout",
        repr(idle_timeout),
        "--log",
        log_path,
    ]
    if basis is not None:
        # An empty ``basis`` (a name that selected the empty suffix) is a
        # NAMED basis, not "no basis" — it must reach ``serve_main``'s own
        # per-deck refusal, not round to the default engine (momwire#733).
        command += ["--basis", basis]
    if dialect != "nec5":
        command += ["--dialect", dialect]
    return command


def _served_bytes(
    deck_bytes: bytes, basis: str | None, idle_timeout: float, dialect: str = "nec5"
) -> bytes:
    """The answer bytes, warm — rungs 1 and 2 of the ladder: the printout,
    or a ``.GMO`` frame and then the printout (``_mech.split_answer``)."""
    path = _mech.socket_path(config_key(basis, idle_timeout, dialect))
    log_path = f"{path}.log"
    conn = _mech.obtain(
        path,
        _server_command(path, basis, idle_timeout, log_path, dialect),
        log_path,
    )
    try:
        conn.sendall(deck_bytes)
        import socket as _socket

        conn.shutdown(_socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            chunk = conn.recv(_RECV_CHUNK)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        conn.close()
    answer = b"".join(chunks)
    if not answer:
        # A server that closed without a byte is a server that died mid-solve;
        # an empty printout must never reach EZNEC as an answer.
        raise ConnectionError("the resident engine closed without answering")
    return answer


def _one_shot(
    deck_path: str,
    printout_path: str,
    basis: str | None,
    dialect: str = "nec5",
    en_optional: bool = False,
) -> None:
    """Rung 3: the stock one-shot, correct at today's speed.

    A child process rather than an import because this module's reason to
    exist is never importing momwire — the slow path pays the interpreter
    twice rather than surrender the fast path's whole premise.
    """
    import subprocess

    command = [sys.executable, "-m", "momwire.eznec", deck_path, printout_path]
    kwargs = []
    if basis is not None:
        # The module entry has no basis flag (EZNEC's argv contract); the
        # frozen twin exes select it by NAME (#643), and this client's
        # subprocess spelling is the -c equivalent of that entry. An empty
        # ``basis`` is a NAMED (empty) basis and must travel verbatim to
        # ``_shell.main``'s own refusal, not round to the default (#733).
        kwargs.append(f"basis={basis!r}")
    if dialect != "nec5":
        # The EZNEC slot rides the same way, for the same reason (#1295).
        kwargs.append(f"dialect={dialect!r}")
    if en_optional:
        # Paths that came from stdin (4nec2, #1488) reach the child on its
        # command line, so the allowance they carry has to ride along too.
        kwargs.append("en_optional=True")
    if kwargs:
        command = [
            sys.executable,
            "-c",
            "import sys; from momwire.eznec._shell import main; "
            f"sys.exit(main(sys.argv[1:], {', '.join(kwargs)}))",
            deck_path,
            printout_path,
        ]
    subprocess.run(command, stdin=subprocess.DEVNULL, timeout=600, check=True)


def _last_ditch(printout_path: str, reason: str) -> None:
    """Rung 4: the file is the only channel there is.

    Without the engine there is no header echo, and EZNEC may reject a
    stampless printout as stale — but a named refusal that MIGHT reach the
    user beats an absent file that certainly reads as a broken install.
    """
    try:
        with open(printout_path, "w", encoding="latin-1", newline="\r\n") as handle:
            handle.write(f" ***** NEC ERROR - {reason}\n")
    except OSError:
        # No printout channel either: stdout is the last one (momwire#1488),
        # never a silent exit 0.
        print(f"{ARGUMENT_ERROR_INPUT} {printout_path}")
        print(f"NEC ERROR - {reason}")


def probe_version() -> str:
    """The ``-version`` first line, answered without importing momwire.

    EZNEC never asks: it always passes a deck and a printout path. SimNEC does,
    because a path containing ``nec5`` is its NEC-5 engine class, and its
    configure step runs ``"<path>" -version`` before accepting the engine. It
    takes exit 0 (or 24) with any first line, and records a line matching its
    ``versionNECd`` pattern, ``NEC<digits><non-digit>...``, as the engine's
    version. ``NEC5momwire.<major>.<minor>`` is the NEC-5 twin of the portal's
    ``NEC2momwire.<major>.<minor>``, read through
    :func:`momwire_serve_client.dist_version` like
    ``momwire_nec2c_client.probe_version`` (a frozen bundle's baked version
    first, momwire#1277), so the probe costs no NumPy import and spawns
    nothing.
    """
    major, minor = _mech.dist_version()
    return f"NEC5momwire.{major}.{minor}"


STREAMING_REFUSAL = (
    "momwire serves the streaming mode (decks on stdin, printouts on stdout) as '- -' only; "
    "give it '- -', or two file paths (momwire#1395, momwire#1512)"
)


def _stream(basis: str | None, dialect: str = "nec5") -> int:
    """SimNEC's streaming call (``- -``, momwire#1512): the one-shot shell's
    :func:`momwire.eznec._shell.stream`, in a child that inherits this
    process's stdin and stdout. One interpreter start per stream, which is
    the whole point of the mode; the resident server answers one deck per
    connection, and a stream is one long conversation. ``dialect`` is the
    EZNEC slot: the NEC-4.2 slot streams too (momwire#1592)."""
    import subprocess

    kwargs = "" if basis is None else f", basis={basis!r}"
    if dialect != "nec5":
        kwargs += f", dialect={dialect!r}"
    command = [
        sys.executable,
        "-c",
        "import sys; from momwire.eznec._shell import main; "
        f"sys.exit(main(['-', '-']{kwargs}))",
    ]
    try:
        return subprocess.run(command, check=False).returncode
    except Exception as exc:  # noqa: BLE001 - stderr is this mode's only other channel
        print(
            f"MOMWIRE ENGINE UNAVAILABLE - {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 2


def _report_times(line: bytes | None, started: float, wrote: float) -> None:
    """MOMWIRE_TIMES: the resident engine's ``times:`` line, with this
    launcher's own printout write and its whole wall time, on stderr."""
    done = _time.perf_counter()
    cells: dict = {}
    head = {"threads": 0, "runs": 0}
    counts: dict = {}
    if line is not None:
        served = _mech.parse_times_line(line.decode("latin-1"))
        for key, value in served.items():
            if key.endswith("_ms") and key[:-3] in _mech.TIMES_CELLS:
                cells[key[:-3]] = value
            elif key.startswith("ground_"):
                counts[key[len("ground_") :]] = value
            elif key in head:
                head[key] = value
    cells["write"] = cells.get("write", 0.0) + (done - wrote) * 1e3
    print(
        _mech.times_line(
            "resident",
            cells,
            total_ms=(done - started) * 1e3,
            threads=head["threads"],
            runs=head["runs"],
            counts=counts,
        ),
        file=sys.stderr,
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    """The process entry point. Returns 0, except for a ``-`` that is not
    SimNEC's streaming call (``- -``), which is refused with 2
    (momwire#1395)."""
    args = list(sys.argv[1:] if argv is None else argv)
    prog = sys.argv[0] if argv is None else "momwire-eznec-client"
    # AVX-512 off before any child (server, one-shot, stream) imports numpy:
    # the answer must not depend on whether the CPU has it.
    _mech.pin_cpu_env()

    if args == ["-version"]:
        print(probe_version())
        return 0
    is_nec4, basis = _mech.filename_nec4(prog)
    en_optional = False
    if not args:
        # 4nec2's form (#1488): `ENGINE < nec2d.tmp`, the two names on stdin,
        # for its NEC-5 package and its NEC-4.2 engine alike (#1592).
        from_stdin = _mech.stdin_paths()
        if from_stdin is not None:
            args = from_stdin
            en_optional = True
    if len(args) < 1:
        print(ARGUMENT_ERROR_INPUT)
        return 0
    if len(args) != 2:
        print(ARGUMENT_ERROR_OUTPUT)
        return 0
    deck_path, printout_path = args
    if not is_nec4:
        basis = _mech.filename_basis(prog, _FILENAME_MARKER, consumed=_CLIENT_SEGMENT)
        if basis is None:
            basis = _mech.filename_basis(prog, _NEC5_MARKER)
    dialect = "nec4" if is_nec4 else "nec5"
    if "-" in (deck_path, printout_path):
        if args == ["-", "-"]:
            # SimNEC's streaming call (momwire#1512), in either slot
            # (momwire#1592).
            return _stream(basis, dialect)
        # Any other '-' (momwire#1395): refused on stderr with a failing
        # status rather than a printout written to a file named "-".
        print(STREAMING_REFUSAL, file=sys.stderr)
        return 2

    timed = _mech.times_requested()
    try:
        started = _time.perf_counter()
        with open(deck_path, "rb") as handle:
            deck_bytes = handle.read()
        if en_optional:
            deck_bytes = _mech.EN_OPTIONAL_REQUEST + deck_bytes
        if timed:
            deck_bytes = _mech.TIMES_REQUEST + deck_bytes
        answer = _served_bytes(deck_bytes, basis, DEFAULT_IDLE_TIMEOUT, dialect)
        line, answer = _mech.split_times(answer)
        gmo, printout = _mech.split_answer(answer)
        wrote = _time.perf_counter()
        if gmo is not None:
            with open(_mech.gmo_path(printout_path), "wb") as handle:
                handle.write(gmo)
        with open(printout_path, "wb") as handle:
            handle.write(printout)
        if timed:
            _report_times(line, started, wrote)
        return 0
    except Exception:  # noqa: BLE001 - the served lane is an optimisation; any failure falls through to one-shot
        pass

    try:
        _one_shot(deck_path, printout_path, basis, dialect, en_optional)
    except Exception as exc:  # noqa: BLE001 - the printout is the only channel EZNEC reads; refuse in it, exit 0
        _last_ditch(
            printout_path,
            f"MOMWIRE ENGINE UNAVAILABLE - {type(exc).__name__}: {exc}",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
