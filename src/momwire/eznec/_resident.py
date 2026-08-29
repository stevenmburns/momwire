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


def serve_main(argv: list[str]) -> int:
    """``--serve --socket PATH [--basis NAME] [--idle-timeout S] [--log P]``."""
    argv = [a for a in argv if a != "--serve"]
    path, argv = take_value(argv, "--socket")
    log_path, argv = take_value(argv, "--log")
    idle_raw, argv = take_value(argv, "--idle-timeout", "900")
    basis, argv = take_value(argv, "--basis", _serve.BASIS)

    def connection(conn, number, log, solve_lock):
        _connection(conn, number, log, solve_lock, basis=basis)

    def _configure(log):
        if argv:
            log.write(f"unused server arguments: {argv}\n")
        return None

    return serve_forever(path, idle_raw, log_path, connection, configure=_configure)
