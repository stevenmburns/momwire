"""The eznec resident path's gates (#718 phase 2, momwire#532).

The claim under test is the drop-in's whole premise, one layer up from the
residency gate: a deck answered by the WARM server over a socket is
byte-for-byte the FILE the one-shot shell writes — CRLF, latin-1, refusals
and all — and the thin client's fallback ladder bottoms out at today's
behavior, never at a missing printout.
"""

from __future__ import annotations

import io
import socket
import threading
import time
from pathlib import Path

import pytest

import momwire_eznec_client as eznec_client
import momwire_serve_client as mech
from momwire.eznec import _resident, _shell
from momwire.serve._server import Server

DECKS = Path(__file__).resolve().parent / "fixtures" / "eznec" / "decks"


def _deck(prefix: str) -> Path:
    return sorted(DECKS.glob(f"{prefix}*.nec"))[0]


def _expected_file_bytes(deck_path: Path, tmp_path: Path) -> bytes:
    """What the one-shot shell would write for this deck — the oracle."""
    out = tmp_path / "oracle.out"
    _shell.run(deck_path, out)
    return out.read_bytes()


@pytest.fixture
def eznec_server(tmp_path):
    """A live in-process eznec server on a private socket."""
    path = str(tmp_path / "srv.sock")
    log = io.StringIO()

    def connection(conn, number, log_stream, solve_lock):
        _resident._connection(
            conn, number, log_stream, solve_lock, basis=_shell._serve.BASIS
        )

    server = Server(path, idle_timeout=3600.0, log=log, connection=connection)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and server.served == 0:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.connect(path)
        except OSError:
            time.sleep(0.01)
            continue
        finally:
            probe.close()
        break
    yield path, server, log
    server.stop()
    thread.join(timeout=10)


def _ask(path: str, deck_bytes: bytes) -> bytes:
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(path)
    try:
        conn.sendall(deck_bytes)
        conn.shutdown(socket.SHUT_WR)
        chunks = []
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        conn.close()
    return b"".join(chunks)


@pytest.mark.integration
def test_a_served_deck_is_the_one_shot_file_to_the_byte(eznec_server, tmp_path):
    """The wire carries the FILE: latin-1, CRLF, transcoded server-side so
    the client stays a dumb pipe."""
    path, _server, _log = eznec_server
    deck = _deck("0010")
    assert _ask(path, deck.read_bytes()) == _expected_file_bytes(deck, tmp_path)


@pytest.mark.integration
def test_a_refusal_rides_the_wire_in_the_same_frame(eznec_server, tmp_path):
    path, _server, _log = eznec_server
    deck = _deck("0022")
    answer = _ask(path, deck.read_bytes())
    assert answer == _expected_file_bytes(deck, tmp_path)
    assert b"NEC ERROR" in answer


@pytest.mark.integration
def test_the_server_answers_deck_after_deck_warm(eznec_server, tmp_path):
    """One process, many connections — the state the residency gate (#677)
    certified, now over the socket."""
    path, server, _log = eznec_server
    deck = _deck("0010")
    expected = _expected_file_bytes(deck, tmp_path)
    # The fixture's readiness probe is itself a connection, so count deltas.
    start = server.served
    for _ in range(3):
        assert _ask(path, deck.read_bytes()) == expected
    assert server.served - start == 3


def test_an_idle_server_stops_itself_and_unlinks_its_socket(tmp_path):
    path = str(tmp_path / "idle.sock")

    def connection(conn, number, log_stream, solve_lock):  # pragma: no cover
        raise AssertionError("nothing connects in this test")

    server = Server(path, idle_timeout=0.3, log=io.StringIO(), connection=connection)
    code = server.run()
    assert code == 0
    assert not Path(path).exists()


# --------------------------------------------------------------------------
# the thin client
# --------------------------------------------------------------------------


def test_the_client_modules_import_stdlib_only():
    """The same rule the portal's client is held to, at the source, for all
    three thin modules — the saving IS the import not happening."""
    import re

    for module in (eznec_client, mech):
        source = Path(module.__file__).read_text()
        offenders = [
            line.strip()
            for line in source.splitlines()
            if re.match(r"\s*(import|from)\s+(momwire|numpy|scipy)\b", line)
        ]
        assert not offenders, f"{module.__name__}: {offenders}"


def test_the_client_falls_back_to_the_one_shot_when_no_server_can_start(
    monkeypatch, tmp_path
):
    """Rung 3 of the ladder: the resident path failing anywhere must degrade
    to today's behavior — a byte-identical printout at one-shot speed."""

    def refuse(*args, **kwargs):
        raise TimeoutError("no server for you")

    monkeypatch.setattr(mech, "obtain", refuse)
    deck = _deck("0010")
    out = tmp_path / "client.out"
    assert eznec_client.main([str(deck), str(out)]) == 0
    assert out.read_bytes() == _expected_file_bytes(deck, tmp_path)


def test_the_last_ditch_printout_names_the_failure_and_still_exits_zero(
    monkeypatch, tmp_path
):
    def refuse(*args, **kwargs):
        raise TimeoutError("no server")

    def broken(*args, **kwargs):
        raise OSError("no interpreter either")

    monkeypatch.setattr(mech, "obtain", refuse)
    monkeypatch.setattr(eznec_client, "_one_shot", broken)
    deck = _deck("0010")
    out = tmp_path / "client.out"
    assert eznec_client.main([str(deck), str(out)]) == 0
    text = out.read_bytes().decode("latin-1")
    assert "NEC ERROR" in text
    assert "OSError" in text


def test_argument_errors_print_and_write_nothing(capsys, tmp_path):
    assert eznec_client.main([]) == 0
    assert eznec_client.ARGUMENT_ERROR_INPUT in capsys.readouterr().out
    assert eznec_client.main(["a", "b", "c"]) == 0
    assert eznec_client.ARGUMENT_ERROR_OUTPUT in capsys.readouterr().out


@pytest.mark.integration
@pytest.mark.slow
def test_the_client_end_to_end_spawns_one_server_and_reuses_it(tmp_path, monkeypatch):
    """The deployment claim, as a real process tree: two client invocations,
    one spawned server, both printouts byte-identical to the one-shot's."""
    import os
    import shutil
    import signal
    import subprocess
    import sys
    import tempfile

    # NOT under tmp_path: pytest's nested tmp directories can push the socket
    # past sun_path (the client then falls back to the one-shot and the test
    # would pass without testing residency — sockets == [] is that ladder
    # working, not the server). A short private dir mirrors the portal
    # selftest's spelling.
    room = Path(tempfile.mkdtemp(prefix="mw-eznec-e2e-"))
    env = {**os.environ, "MOMWIRE_PORTAL_RUNTIME_DIR": str(room)}
    deck = _deck("0010")
    expected = _expected_file_bytes(deck, tmp_path)

    try:
        outs = []
        for index in range(2):
            out = tmp_path / f"e2e-{index}.out"
            proc = subprocess.run(
                [sys.executable, "-m", "momwire_eznec_client", str(deck), str(out)],
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )
            assert proc.returncode == 0, proc.stderr
            outs.append(out.read_bytes())
            sockets = sorted(p.name for p in room.glob("*.sock"))
            assert len(sockets) == 1, sockets

        assert outs[0] == expected
        assert outs[1] == expected
    finally:
        # Ask the spawned server to go, by pid out of its own log line — the
        # portal selftest's rule.
        for log in room.glob("*.log"):
            for line in log.read_text(errors="replace").splitlines():
                for word in line.split():
                    if word.startswith("pid="):
                        try:
                            os.kill(int(word.split("=", 1)[1]), signal.SIGTERM)
                        except (OSError, ValueError):
                            pass
        shutil.rmtree(room, ignore_errors=True)
