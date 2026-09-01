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
exit 0 always, always write the printout, never read stdin.

Like ``momwire_nec2c_client``, this module imports stdlib only — the shared
finding-the-server mechanics live in ``momwire_serve_client`` and the tests
hold both to it.
"""

from __future__ import annotations

import os
import sys

import momwire_serve_client as _mech

# momwire#528/#643: the basis rides on the client's own filename —
# ``momwire-eznec-client-razor-nec5`` (or a renamed copy of the frozen exe)
# selects the twin; the ``client`` segment itself selects nothing.
_FILENAME_MARKER = "eznec-"
_CLIENT_SEGMENT = "client"

DEFAULT_IDLE_TIMEOUT = 900.0

# EZNEC's own fault paths print these and write nothing (there is no output
# path to write to) — captured behavior, same strings as the shell's.
ARGUMENT_ERROR_INPUT = "UNABLE TO OPEN FILE"
ARGUMENT_ERROR_OUTPUT = "UNABLE TO OPEN SECOND FILE"

_RECV_CHUNK = 65536


def config_key(basis: str | None, idle_timeout: float) -> str:
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
    """
    major, minor = _mech.dist_version()
    return _mech.digest(
        [
            f"eznec.{major}.{minor}",
            os.path.realpath(sys.executable),
            f"{idle_timeout!r}",
            "default" if basis is None else f"basis={basis}",
        ]
    )


def _server_command(
    path: str, basis: str | None, idle_timeout: float, log_path: str
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
    return command


def _served_bytes(deck_bytes: bytes, basis: str | None, idle_timeout: float) -> bytes:
    """The printout bytes, answered warm — rungs 1 and 2 of the ladder."""
    path = _mech.socket_path(config_key(basis, idle_timeout))
    log_path = f"{path}.log"
    conn = _mech.obtain(
        path, _server_command(path, basis, idle_timeout, log_path), log_path
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


def _one_shot(deck_path: str, printout_path: str, basis: str | None) -> None:
    """Rung 3: the stock one-shot, correct at today's speed.

    A child process rather than an import because this module's reason to
    exist is never importing momwire — the slow path pays the interpreter
    twice rather than surrender the fast path's whole premise.
    """
    import subprocess

    command = [sys.executable, "-m", "momwire.eznec", deck_path, printout_path]
    if basis is not None:
        # The module entry has no basis flag (EZNEC's argv contract); the
        # frozen twin exes select it by NAME (#643), and this client's
        # subprocess spelling is the -c equivalent of that entry. An empty
        # ``basis`` is a NAMED (empty) basis and must travel verbatim to
        # ``_shell.main``'s own refusal, not round to the default (#733).
        command = [
            sys.executable,
            "-c",
            "import sys; from momwire.eznec._shell import main; "
            f"sys.exit(main(sys.argv[1:], basis={basis!r}))",
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
        pass


def main(argv: list[str] | None = None) -> int:
    """The process entry point. Returns 0. Always returns 0."""
    args = list(sys.argv[1:] if argv is None else argv)
    prog = sys.argv[0] if argv is None else "momwire-eznec-client"

    if len(args) < 1:
        print(ARGUMENT_ERROR_INPUT)
        return 0
    if len(args) != 2:
        print(ARGUMENT_ERROR_OUTPUT)
        return 0
    deck_path, printout_path = args
    basis = _mech.filename_basis(prog, _FILENAME_MARKER, consumed=_CLIENT_SEGMENT)

    try:
        with open(deck_path, "rb") as handle:
            deck_bytes = handle.read()
        answer = _served_bytes(deck_bytes, basis, DEFAULT_IDLE_TIMEOUT)
        with open(printout_path, "wb") as handle:
            handle.write(answer)
        return 0
    except Exception:  # noqa: BLE001 - the served lane is an optimisation; any failure falls through to one-shot
        pass

    try:
        _one_shot(deck_path, printout_path, basis)
    except Exception as exc:  # noqa: BLE001 - the printout is the only channel EZNEC reads; refuse in it, exit 0
        _last_ditch(
            printout_path,
            f"MOMWIRE ENGINE UNAVAILABLE - {type(exc).__name__}: {exc}",
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
