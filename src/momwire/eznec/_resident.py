"""``python -m momwire.eznec --serve`` — the EZNEC seam behind a warm socket.

The #718 arc's whole saving (momwire#532): EZNEC launches its external engine
once per frequency point and the frozen one-shot pays ~1 s of interpreter and
NumPy import every time, against a licensed engine that launches in 18-37 ms.
This process pays the import ONCE and answers deck after deck warm; the thin
client (``momwire_eznec_client`` today, the native exe of the arc's phase 3
eventually) forwards deck bytes and writes the answer to the file EZNEC reads.

The wire contract is the seam's degenerate session, verbatim (#719 U4): one
connection is one deck — the client sends the deck bytes and shuts down its
write side, EOF is the frame, :func:`momwire.serve.run_session` answers once,
the connection closes. No length prefixes, no framing of our own, and the
printout that comes back is byte-for-byte the file to write: the transcoding
the one-shot shell does at :func:`~momwire.eznec._shell.write_printout`
(latin-1, CRLF) happens HERE, in the connection's own text wrapper, so a
client stays a dumb pipe and cannot get it wrong.

``--basis`` is per-server, not per-request: the socket's name already hashes
the engine identity client-side (one warm server per formulation, exactly as
one frozen exe per formulation — momwire#593/#643), so a request naming a
basis would only be a second spelling of the socket it arrived on.
"""

from __future__ import annotations

import io

from ..serve import run_session
from ..serve._server import ConnLog, serve_forever, take_value
from . import _serve
from ._shell import seam

# The one-shot shell's file codec, restated for the socket: the printout
# bytes on the wire ARE the file's bytes (write_printout's contract), and the
# deck arrives however EZNEC wrote it — latin-1, CRLF — read through the same
# universal-newline translation read_deck applies.
_CODEC = "latin-1"


def _connection(conn, number: int, log, solve_lock, *, basis: str) -> None:
    """One deck over one connection: the eznec seam's half of the server."""
    rx = tx = None
    try:
        rx = io.TextIOWrapper(conn.makefile("rb"), encoding=_CODEC, errors="replace")
        tx = io.TextIOWrapper(
            conn.makefile("wb"), encoding=_CODEC, errors="replace", newline="\r\n"
        )
        run_session(
            seam(basis=basis), rx, tx, ConnLog(log, f"[conn {number}] "), solve_lock
        )
    finally:
        for handle in (tx, rx):
            if handle is None:
                continue
            try:
                handle.close()
            except OSError:
                pass


# The two compiled extensions, under the names their files carry, so a log
# line names the thing to go looking for.  ONE line for both because they are
# one fact to a user: either the fast path is live or every solve in this
# process is the pure-Python one.
_EXTENSIONS = ("_accelerators", "_near_interface_accel")

# The affirmative spelling, restated where the gates can import it (smoke.py
# greps a daemon's log for it).  It must NOT be a substring of the fallback
# line, or a half-loaded process would read as a healthy one.
ACCEL_OK = "accelerators: loaded"


def accelerator_status() -> str:
    """The one log line that says whether this daemon's fast path is live.

    momwire#737: the Windows bundle shipped without `libomp140.x86_64.dll`, so
    every deployed solve was pure Python and NOTHING said so — the fallback
    warning goes to stderr, which EZNEC never shows.  "Was the accelerator
    live" is the first question of every slowness report, so the daemon
    answers it once, in the log, before it binds.

    Imported at CALL time rather than at module import: the flags are read
    from the modules that own them (and a test can move them), and this must
    not reorder the daemon's own import graph.
    """
    from .. import _accel, _near_interface

    live = (_accel.LOADED, _near_interface._HAVE_NEAR_INTERFACE_ACCEL)
    missing = [name for name, ok in zip(_EXTENSIONS, live) if not ok]
    if missing:
        return f"accelerators: PURE-PYTHON FALLBACK ({', '.join(missing)} did not load)"
    return f"{ACCEL_OK} ({', '.join(_EXTENSIONS)})"


def _configure(log, argv: list[str]) -> None:
    """The daemon's once-per-process log preamble; None to proceed."""
    if argv:
        log.write(f"unused server arguments: {argv}\n")
    log.write(f"{accelerator_status()}\n")
    return None


def serve_main(argv: list[str]) -> int:
    """``--serve --socket PATH [--basis NAME] [--idle-timeout S] [--log P]``."""
    argv = [a for a in argv if a != "--serve"]
    path, argv = take_value(argv, "--socket")
    log_path, argv = take_value(argv, "--log")
    idle_raw, argv = take_value(argv, "--idle-timeout", "900")
    basis, argv = take_value(argv, "--basis", _serve.BASIS)

    def connection(conn, number, log, solve_lock):
        _connection(conn, number, log, solve_lock, basis=basis)

    return serve_forever(
        path,
        idle_raw,
        log_path,
        connection,
        configure=lambda log: _configure(log, argv),
    )
