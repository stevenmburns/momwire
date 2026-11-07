"""The resident eznec wire must not need the client's half-close (momwire#1822).

What macOS does, measured with no momwire code at all (the stdlib repro on
the scratch branch ``diag/1822-minimal``): a client sends its bytes and calls
``shutdown(SHUT_WR)``, the call succeeds, every byte arrives, and the
server's blocking ``recv`` never returns the EOF. On a macos-15 runner
(15.8.2) that is about 1 connection in 10^4 with a process per client, 1 in
10^5 with threads, and it happens to a strictly serial client of a warm
server too; Linux has never shown it. A wire whose only frame is that EOF then deadlocks: the
server waits for the end of the deck, and the client waits for the answer.

The kernel's race cannot be forced from user space, but the state it leaves
can: a socket whose ``shutdown`` returns and delivers nothing. ``_EofLost``
below is that socket, handed to the real client through ``_mech.obtain``, so
these gates run the client's own exchange against the server's own framing,
on every platform, with no timing in them. On the EOF-framed wire they fail;
with the deck's length on the wire they pass.

The bound on ``recv`` is the failure detector, not the mechanism: a passing
run answers in milliseconds and never comes near it.
"""

from __future__ import annotations

import io
import shutil
import socket
import tempfile
import threading
from pathlib import Path

import pytest

import momwire_eznec_client as eznec_client
import momwire_serve_client as mech
from momwire.dropin.nec5 import _resident, _shell
from momwire.serve._server import Server

DECKS = Path(__file__).resolve().parent / "fixtures" / "eznec" / "decks"

# Only how long a FAILING run takes to say so.
_DETECT_S = 60.0


class _EofLost:
    """A connected socket on which ``shutdown`` succeeds and does nothing.

    The macOS outcome, made deterministic: the half-close is accepted by the
    kernel and never reaches the peer.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._sock.settimeout(_DETECT_S)
        self.shutdowns = 0

    def shutdown(self, _how) -> None:
        self.shutdowns += 1

    def __getattr__(self, name):
        return getattr(self._sock, name)


@pytest.fixture
def room():
    """A SHORT private directory (test_serve_resident.short_room's rule:
    macOS's ``sun_path`` is 104 bytes and its pytest root is long)."""
    path = Path(tempfile.mkdtemp(prefix="mw-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(params=[mech.UNIX, mech.TCP])
def served(request, room, monkeypatch):
    """An in-process eznec server, and the address the client will be given."""
    if request.param == mech.UNIX and not hasattr(socket, "AF_UNIX"):
        pytest.skip("this build has no AF_UNIX")
    monkeypatch.setenv(mech.TRANSPORT_ENV, request.param)
    path = str(room / f"srv{mech.address_suffix(request.param)}")
    log = io.StringIO()

    def connection(conn, number, log_stream, solve_lock):
        _resident._connection(
            conn, number, log_stream, solve_lock, basis=_shell._serve.BASIS
        )

    server = Server(path, idle_timeout=3600.0, log=log, connection=connection)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert server.listening.wait(10), "the server never started listening"
    yield path, log
    server.stop()
    thread.join(timeout=10)


def _oracle(deck: Path, tmp_path: Path) -> bytes:
    out = tmp_path / "oracle.out"
    _shell.run(deck, out)
    return out.read_bytes()


@pytest.mark.integration
def test_the_client_is_answered_when_its_half_close_is_lost(
    served, monkeypatch, tmp_path
):
    """The red control: the real client, the real server, the EOF dropped."""
    path, log = served
    handed: list[_EofLost] = []

    def obtain(_path, _command, _log_path):
        conn = mech.connect(path)
        assert conn is not None, f"nothing is listening at {path}"
        handed.append(_EofLost(conn))
        return handed[-1]

    monkeypatch.setattr(eznec_client._mech, "obtain", obtain)
    deck = sorted(DECKS.glob("0010*.nec"))[0]
    try:
        answer = eznec_client._served_bytes(deck.read_bytes(), None, 900.0)
    except TimeoutError:
        pytest.fail(
            "deadlock: the server is still waiting for an EOF the client sent "
            f"and the kernel dropped (momwire#1822). Server log:\n{log.getvalue()}"
        )
    assert answer == _oracle(deck, tmp_path)
    assert len(handed) == 1


@pytest.mark.integration
def test_a_framed_deck_is_answered_with_the_write_side_still_open(served, tmp_path):
    """The server's half on its own: the length is the frame, so a deck is
    complete when its bytes are in, whatever the EOF does."""
    path, _log = served
    deck = sorted(DECKS.glob("0010*.nec"))[0]
    conn = mech.connect(path)
    assert conn is not None
    conn.settimeout(_DETECT_S)
    try:
        conn.sendall(mech.deck_frame(deck.read_bytes()))
        chunks = []
        while chunk := conn.recv(65536):
            chunks.append(chunk)
    finally:
        conn.close()
    assert b"".join(chunks) == _oracle(deck, tmp_path)


@pytest.mark.integration
def test_an_unframed_deck_still_ends_at_eof(served, tmp_path):
    """The pre-#1822 wire, as a hand-driven ``nc -U`` or an older client of
    another key speaks it, keeps its old frame: the bytes up to EOF."""
    path, _log = served
    deck = sorted(DECKS.glob("0010*.nec"))[0]
    conn = mech.connect(path)
    assert conn is not None
    conn.settimeout(_DETECT_S)
    try:
        conn.sendall(deck.read_bytes())
        conn.shutdown(socket.SHUT_WR)
        chunks = []
        while chunk := conn.recv(65536):
            chunks.append(chunk)
    finally:
        conn.close()
    assert b"".join(chunks) == _oracle(deck, tmp_path)


@pytest.mark.integration
def test_a_frame_cut_short_is_dropped_without_an_answer(served):
    """A client that dies mid-deck: fewer bytes than the frame promised, then
    EOF. The server must not solve the fragment as if it were the deck."""
    path, log = served
    conn = mech.connect(path)
    assert conn is not None
    conn.settimeout(_DETECT_S)
    try:
        frame = mech.deck_frame(b"CM cut\r\nGW 1 11 0 0 0 0 0 1 0.001\r\nEN\r\n")
        conn.sendall(frame[:-5])
        conn.shutdown(socket.SHUT_WR)
        assert conn.recv(65536) == b""
    finally:
        conn.close()


def test_the_wire_is_in_the_key(monkeypatch):
    """A new client must never reach a server of the same minor version that
    still reads to EOF: it would solve the frame header as deck text."""
    hashed: list[list[str]] = []
    real = mech.digest
    monkeypatch.setattr(
        eznec_client._mech, "digest", lambda parts: hashed.append(parts) or real(parts)
    )
    eznec_client.config_key(None, 900.0)
    assert mech.WIRE in hashed[0]


def test_the_frame_is_the_length_then_the_bytes():
    body = b"CM x\r\nEN\r\n"
    head, sep, rest = mech.deck_frame(body).partition(b"\n")
    assert sep == b"\n"
    assert head == mech.DECK_FRAME_TAG + b" " + str(len(body)).encode("ascii")
    assert rest == body
