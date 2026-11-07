"""The NEC-2 drop-in's stream must not need the client's half-close (momwire#1831).

momwire#1822 found that macOS can lose the EOF of a ``shutdown(SHUT_WR)``:
the call succeeds, every byte arrives, and the peer's blocking ``recv`` never
returns the EOF. #1841 took the NEC-5 wire off that EOF by sending each deck
behind its length. The NEC-2 leg (``momwire-nec2c-shared``, SimNEC's portal
mode) still ended its session with the half-close, so a lost EOF left the
server waiting for the end of input and the client waiting for the server to
finish: the same deadlock, on a stream rather than one deck.

The fix carries the same frame on the stream: :func:`momwire_nec2c_client.pump`
sends each piece of stdin as one ``deck_frame`` and, at stdin's end, the empty
frame ``STREAM_END``, and never half-closes. The NEC-2 server reads the frames
back through ``momwire_serve_client.FrameReader``.

``_EofLost`` is the state the kernel's race leaves (``shutdown`` succeeds and
delivers nothing), handed to the real ``pump`` against the real in-process
NEC-2 server, over both transports. On the EOF-ended stream the red control
fails; on the framed one it passes in milliseconds. The bound is the failure
detector only.
"""

from __future__ import annotations

import importlib.util
import io
import os
import re
import shutil
import socket
import tempfile
import threading
from pathlib import Path

import pytest

import momwire_nec2c_client as client
import momwire_serve_client as mech
from momwire.dropin.nec2c import _shared
from momwire.dropin.nec2c import main as portal_main
from momwire.dropin.nec2c._portal import configure_engine
from momwire.serve._server import Server

REPO = Path(__file__).resolve().parent.parent
FIXTURE_DIR = REPO / "tests" / "fixtures" / "nec_portal"

# Only how long a FAILING run takes to say so.
_DETECT_S = 60.0


def _load_capture_module():
    """The capture script's timing canonicaliser (scripts/ is not a package)."""
    path = REPO / "scripts" / "nec_portal_capture.py"
    spec = importlib.util.spec_from_file_location("nec_portal_capture_1831", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE = _load_capture_module()


class _EofLost:
    """A connected socket on which ``shutdown`` succeeds and does nothing:
    the macOS outcome of momwire#1822, made deterministic."""

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self.shutdowns = 0

    def shutdown(self, _how) -> None:
        self.shutdowns += 1

    def __getattr__(self, name):
        return getattr(self._sock, name)


@pytest.fixture
def room():
    """A SHORT private directory: macOS's ``sun_path`` is 104 bytes."""
    path = Path(tempfile.mkdtemp(prefix="mw-"))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(params=[mech.UNIX, mech.TCP])
def served(request, room, monkeypatch):
    """An in-process NEC-2 server (``_shared._connection``, the real one)."""
    if request.param == mech.UNIX and not hasattr(socket, "AF_UNIX"):
        pytest.skip("this build has no AF_UNIX")
    monkeypatch.setenv(mech.TRANSPORT_ENV, request.param)
    rest, code = configure_engine([], io.StringIO())
    assert (rest, code) == ([], None)
    path = str(room / f"srv{mech.address_suffix(request.param)}")
    log = io.StringIO()
    server = Server(path, idle_timeout=3600.0, log=log, connection=_shared._connection)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert server.listening.wait(10), "the server never started listening"
    yield path, log
    server.stop()
    thread.join(timeout=10)


def _stock(decks: str) -> str:
    out = io.StringIO()
    assert portal_main([], io.StringIO(decks), out, io.StringIO()) == 0
    return out.getvalue()


def _same(left: str, right: str) -> bool:
    return CAPTURE.canonicalize_timings(left) == CAPTURE.canonicalize_timings(right)


def _deck_body(name: str) -> str:
    """The fixture deck without its ``NX``: only the end of input ends it."""
    text = (FIXTURE_DIR / f"{name}.deck").read_text()
    return re.sub(r"(?im)^\s*NX\b.*\n?\Z", "", text)


def _pump(conn, data: bytes, tmp_path: Path) -> tuple[bool, bytes]:
    """The real ``pump`` over ``conn``, stdin a pipe holding ``data``.
    ``(finished, stdout bytes)``; unfinished means it is still waiting."""
    read_fd, write_fd = os.pipe()
    os.write(write_fd, data)
    os.close(write_fd)
    out_path = tmp_path / "stdout"
    out_fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    done: list[int] = []
    thread = threading.Thread(
        target=lambda: done.append(client.pump(conn, read_fd, out_fd)), daemon=True
    )
    thread.start()
    thread.join(_DETECT_S)
    finished = not thread.is_alive()
    if not finished:
        # Unblock the pump so the thread can end: the peer's close.
        try:
            conn._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        thread.join(10)
    os.close(read_fd)
    os.close(out_fd)
    return finished, out_path.read_bytes()


@pytest.mark.integration
@pytest.mark.parametrize("name", ["dipole_free_space"])
def test_the_session_ends_when_the_client_half_close_is_lost(served, tmp_path, name):
    """The red control: the real pump, the real server, the EOF dropped.

    The deck carries no ``NX``, so only the end of input asks the server to
    solve it (NEC's own EOF rule) and then to finish the session."""
    path, log = served
    conn = _EofLost(mech.connect(path))
    decks = _deck_body(name)
    try:
        finished, answer = _pump(conn, decks.encode("utf-8"), tmp_path)
    finally:
        conn.close()
    if not finished:
        pytest.fail(
            "deadlock: the NEC-2 server is still waiting for an EOF the client "
            f"sent and the kernel dropped (momwire#1831). Server log:\n{log.getvalue()}"
        )
    assert _same(answer.decode("utf-8"), _stock(decks))
    assert conn.shutdowns == 0, "the framed stream never half-closes"


@pytest.mark.integration
def test_an_unframed_stream_still_ends_at_eof(served):
    """A hand-driven ``nc -U`` keeps the old wire: bytes, then the EOF."""
    path, _log = served
    decks = _deck_body("dipole_free_space")
    conn = mech.connect(path)
    conn.settimeout(_DETECT_S)
    try:
        conn.sendall(decks.encode("utf-8"))
        conn.shutdown(socket.SHUT_WR)
        chunks = []
        while chunk := conn.recv(65536):
            chunks.append(chunk)
    finally:
        conn.close()
    assert _same(b"".join(chunks).decode("utf-8"), _stock(decks))


def _read_all(raw: bytes) -> bytes:
    return mech.FrameReader(io.BufferedReader(io.BytesIO(raw))).read()


def test_frames_read_back_as_the_stream_and_end_at_the_empty_frame():
    wire = mech.deck_frame(b"CM a\n") + mech.deck_frame(b"EN\n") + mech.STREAM_END
    assert _read_all(wire + b"never read") == b"CM a\nEN\n"


def test_the_stream_end_is_the_empty_frame():
    assert mech.STREAM_END == mech.deck_frame(b"")


def test_an_unframed_stream_reads_to_eof():
    assert _read_all(b"CM a\nGW 1 11\nEN\n") == b"CM a\nGW 1 11\nEN\n"


def test_a_frame_cut_short_ends_the_stream_without_its_fragment():
    wire = mech.deck_frame(b"CM a\n") + mech.deck_frame(b"GW 1 11 0 0 0")[:-4]
    assert _read_all(wire) == b"CM a\n"


def test_a_stream_without_its_end_frame_ends_at_eof():
    assert _read_all(mech.deck_frame(b"CM a\n")) == b"CM a\n"


def test_nothing_is_read_before_the_first_read():
    """The server greets before it reads, and a client may wait for the
    greeting before it sends: building the reader must not block."""

    class _Untouchable(io.RawIOBase):
        def readable(self):
            return True

        def readinto(self, _buffer):
            raise AssertionError("read before the first read")

    mech.FrameReader(io.BufferedReader(_Untouchable()))


def test_the_wire_is_in_the_nec2_key(monkeypatch):
    """A framing client must never reach a server of the same version that
    reads the stream unframed: it would read frame lines as cards."""
    hashed: list[list[str]] = []
    real = mech.digest
    monkeypatch.setattr(
        client._mech, "digest", lambda parts: hashed.append(parts) or real(parts)
    )
    client.config_key([], 900.0)
    assert mech.WIRE in hashed[0]
