"""The eznec resident path's gates (#718 phase 2, momwire#532).

The claim under test is the drop-in's whole premise, one layer up from the
residency gate: a deck answered by the WARM server over a socket is
byte-for-byte the FILE the one-shot shell writes — CRLF, latin-1, refusals
and all — and the thin client's fallback ladder bottoms out at today's
behavior, never at a missing printout.

Every one of those claims is made TWICE since #718 phase 2 unit 2, once per
transport, and both legs run here on Linux: the ``MOMWIRE_SERVE_TRANSPORT``
override exists so the loopback-TCP path — the one Windows builds without
``AF_UNIX`` take — is exercised on the box the suite actually runs on,
rather than only on a canary nobody watches. What stays Windows-only is the
``msvcrt`` lock spelling and the detached-spawn creation flags.
"""

from __future__ import annotations

import io
import os
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


@pytest.fixture(params=[mech.UNIX, mech.TCP])
def transport(request, monkeypatch):
    """Both transports, selected the way a real deployment selects them.

    Through the override rather than by monkeypatching ``socket``: the thing
    that must hold on a Windows build without ``AF_UNIX`` is the whole TCP
    path — rendezvous file, ephemeral port, bind, connect, unlink — and the
    only way to run it on this box is to ask for it.
    """
    monkeypatch.setenv(mech.TRANSPORT_ENV, request.param)
    return request.param


@pytest.fixture
def eznec_server(tmp_path, transport):
    """A live in-process eznec server on a private address."""
    path = str(tmp_path / f"srv{mech.address_suffix(transport)}")
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
        probe = mech.connect(path)
        if probe is None:
            time.sleep(0.01)
            continue
        probe.close()
        break
    yield path, server, log
    server.stop()
    thread.join(timeout=10)


def _ask(path: str, deck_bytes: bytes) -> bytes:
    conn = mech.connect(path)
    assert conn is not None, f"nothing is listening at {path}"
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


def test_an_idle_server_stops_itself_and_unlinks_its_socket(tmp_path, transport):
    """Both transports leave nothing behind: a socket file and a rendezvous
    file are the same litter to the next client, and the unlink that removes
    them is the same line."""
    path = str(tmp_path / f"idle{mech.address_suffix(transport)}")

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
def test_the_client_end_to_end_spawns_one_server_and_reuses_it(tmp_path, transport):
    """The deployment claim, as a real process tree: two client invocations,
    one spawned server, both printouts byte-identical to the one-shot's.

    Run once per transport, with the override passed in the ENVIRONMENT
    rather than monkeypatched: the client and the server it spawns are two
    separate processes here, and the point of the gate is that they agree
    about the transport without ever exchanging a word about it.
    """
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
    env = {
        **os.environ,
        "MOMWIRE_PORTAL_RUNTIME_DIR": str(room),
        mech.TRANSPORT_ENV: transport,
    }
    suffix = mech.address_suffix(transport)
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
            sockets = sorted(p.name for p in room.glob(f"*{suffix}"))
            assert len(sockets) == 1, sockets

        assert outs[0] == expected
        assert outs[1] == expected
        # ONE server, and the second client REUSED it. Without this the
        # fallback ladder could carry the whole test: a one-shot answer is
        # byte-identical by construction, so the printouts alone prove
        # nothing about residency. A spawned server always logs one
        # `listening pid=` line; two would be a second spawn, none a
        # resident path that never ran.
        listening = [
            line
            for log in room.glob("*.log")
            for line in log.read_text(errors="replace").splitlines()
            if "listening pid=" in line
        ]
        assert len(listening) == 1, listening
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


# --------------------------------------------------------------------------
# the transport itself (#718 phase 2 unit 2)
#
# Everything below runs on Linux and gates the Windows-shaped half of the
# design: which transport gets chosen, what the TCP rendezvous file is worth,
# and which POSIX-only rules stop applying when there is no Unix socket. The
# two genuinely Windows-only leaves — msvcrt's byte lock and the detached
# spawn's creation flags — can only be proven by the windows-latest canary.
# --------------------------------------------------------------------------


def test_the_transport_is_af_unix_where_it_exists_and_tcp_where_it_does_not(
    monkeypatch,
):
    """The autodetect, and the reason it is spelt as that exact question: a
    Unix socket IS the rendezvous, and TCP mode exists only to rebuild one
    out of a file and a port where the build has no AF_UNIX."""
    monkeypatch.delenv(mech.TRANSPORT_ENV, raising=False)
    # Spelt as the rule rather than as "unix", because this file also runs on
    # the windows-latest canary, where the answer depends on the BUILD.
    assert mech.transport() == (mech.UNIX if hasattr(socket, "AF_UNIX") else mech.TCP)

    monkeypatch.delattr(socket, "AF_UNIX", raising=False)
    assert mech.transport() == mech.TCP


def test_the_transport_override_beats_the_autodetect_both_ways(monkeypatch):
    """``tcp`` on a box that HAS AF_UNIX is how this suite runs the whole TCP
    path at all; ``unix`` is how a Windows build that does have AF_UNIX is
    pinned to it. A typo is refused, never silently defaulted."""
    monkeypatch.setenv(mech.TRANSPORT_ENV, "tcp")
    assert mech.transport() == mech.TCP
    monkeypatch.setenv(mech.TRANSPORT_ENV, "  UNIX ")
    assert mech.transport() == mech.UNIX

    monkeypatch.setenv(mech.TRANSPORT_ENV, "sockets-please")
    with pytest.raises(ValueError, match="is not a transport"):
        mech.transport()


def test_an_explicit_unix_transport_never_falls_back_silently(monkeypatch, tmp_path):
    """ "I told you to use a transport this Python does not have" must be an
    error naming the missing attribute, not a quiet switch to TCP — a silent
    fallback would answer a client on a transport its server never bound."""
    monkeypatch.setenv(mech.TRANSPORT_ENV, "unix")
    monkeypatch.delattr(socket, "AF_UNIX", raising=False)
    with pytest.raises(AttributeError, match="AF_UNIX"):
        mech.connect(str(tmp_path / "nothing.sock"))


def test_the_address_suffix_tells_the_two_transports_apart(monkeypatch, tmp_path):
    """A directory can hold leftovers from a run in the other mode. The
    suffixes differ so a stale ``.sock`` can never be read as a rendezvous
    (nor a ``.port`` connected to as a socket)."""
    monkeypatch.setenv("MOMWIRE_PORTAL_RUNTIME_DIR", str(tmp_path))
    monkeypatch.setenv(mech.TRANSPORT_ENV, "unix")
    assert mech.socket_path("k").endswith(".sock")
    monkeypatch.setenv(mech.TRANSPORT_ENV, "tcp")
    assert mech.socket_path("k").endswith(".port")
    assert mech.address_suffix(mech.UNIX) != mech.address_suffix(mech.TCP)
    with pytest.raises(ValueError, match="is not a transport"):
        mech.address_suffix("carrier-pigeon")


def test_the_sun_path_limit_is_a_unix_rule_only(monkeypatch, tmp_path):
    """``sun_path``'s 108 bytes is the kernel's limit on a SOCKET NAME. A
    rendezvous file is an ordinary file, so enforcing it in TCP mode would
    refuse a configuration that works perfectly."""
    deep = tmp_path / ("d" * 90)
    deep.mkdir()
    monkeypatch.setenv("MOMWIRE_PORTAL_RUNTIME_DIR", str(deep))

    monkeypatch.setenv(mech.TRANSPORT_ENV, "unix")
    with pytest.raises(ValueError, match="AF_UNIX limit"):
        mech.socket_path("k")

    monkeypatch.setenv(mech.TRANSPORT_ENV, "tcp")
    assert mech.socket_path("k").endswith(".port")


def test_a_rendezvous_file_is_only_ever_read_whole(tmp_path):
    """The atomic-publish claim, raced.

    A reader is a client, and it arrives whenever it arrives. A partially
    written file would be read as a port that is a PREFIX of the real one —
    a connect to somebody else's service — so every read a hammering reader
    gets must be either "nothing published yet" or one of the ports that was
    actually published, never a fragment of one.
    """
    path = str(tmp_path / "race.port")
    ports = [40000 + n for n in range(200)]
    seen: list = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            seen.append(mech.read_rendezvous(path))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for port in ports:
            mech.publish_rendezvous(path, port)
    finally:
        stop.set()
        thread.join(timeout=10)

    allowed = {None, *((mech.RENDEZVOUS_HOST, port) for port in ports)}
    assert set(seen) <= allowed, sorted(set(seen) - allowed)
    assert mech.read_rendezvous(path) == (mech.RENDEZVOUS_HOST, ports[-1])
    # tmp + os.replace, and the tmp does not survive: a scratch file left in
    # the runtime directory is litter no client knows how to clean.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["race.port"]


def test_unreadable_rendezvous_shapes_all_mean_nobody_listening(tmp_path):
    """No file, an empty one, a truncated one, bytes that are not a port —
    one answer, because the caller does one thing about all of them."""
    path = tmp_path / "shapes.port"
    assert mech.read_rendezvous(str(path)) is None
    for junk in (b"", b"127.0.0.1:", b"127.0.0.1", b":4242\n", b"\xff\xfe\n", b"127."):
        path.write_bytes(junk)
        assert mech.read_rendezvous(str(path)) is None, junk
    path.write_bytes(b"127.0.0.1:4242\n")
    assert mech.read_rendezvous(str(path)) == ("127.0.0.1", 4242)


def _idle_server(path: str) -> Server:
    def connection(conn, number, log_stream, solve_lock):  # pragma: no cover
        raise AssertionError("nothing connects in this test")

    return Server(path, idle_timeout=3600.0, log=io.StringIO(), connection=connection)


def test_a_stale_rendezvous_file_is_claimed_by_the_next_server(monkeypatch, tmp_path):
    """A server that died without unlinking leaves a file naming a port
    nobody answers. That is not a live peer, and the next server must take
    the name over — in one ``os.replace``, so no client ever finds it gone.
    """
    monkeypatch.setenv(mech.TRANSPORT_ENV, "tcp")
    path = str(tmp_path / "stale.port")

    # A port that was real and is not any more: bind, read it, close it.
    dead = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    dead.bind((mech.RENDEZVOUS_HOST, 0))
    dead_port = dead.getsockname()[1]
    dead.close()
    mech.publish_rendezvous(path, dead_port)
    assert mech.connect(path) is None, "a refused connect is nobody listening"

    listener = _idle_server(path)._bind()
    assert listener is not None, "the stale file was mistaken for a live peer"
    try:
        listener.listen(1)
        published = mech.read_rendezvous(path)
        assert published == (mech.RENDEZVOUS_HOST, listener.getsockname()[1])
        assert published != (mech.RENDEZVOUS_HOST, dead_port)
        conn = mech.connect(path)
        assert conn is not None
        conn.close()
    finally:
        listener.close()


def test_a_live_rendezvous_peer_makes_the_second_server_lose_gracefully(
    monkeypatch, tmp_path
):
    """The EADDRINUSE analogue. Unlinking (or replacing) a LIVE peer's name
    would strand every client already talking to it, so the loser exits 0 and
    touches nothing — including the file, which must still name the winner.
    """
    monkeypatch.setenv(mech.TRANSPORT_ENV, "tcp")
    path = str(tmp_path / "peer.port")

    winner = _idle_server(path)._bind()
    assert winner is not None
    try:
        published = mech.read_rendezvous(path)
        assert _idle_server(path)._bind() is None
        assert mech.read_rendezvous(path) == published
    finally:
        winner.close()


def test_the_obtain_lock_excludes_a_second_holder(tmp_path):
    """The serialising half: a crew firing 16 clients at once must start ONE
    server. Both spellings of :class:`_locked` are gated here — ``flock`` on
    POSIX, ``msvcrt.locking`` on the canary — because a lock that does not
    exclude is the defect no amount of retrying covers.
    """
    lock = str(tmp_path / "srv.sock.lock")
    entered = threading.Event()

    def second():
        with mech._locked(lock, timeout=30.0):
            entered.set()

    with mech._locked(lock, timeout=30.0):
        thread = threading.Thread(target=second, daemon=True)
        thread.start()
        assert not entered.wait(0.4), "the lock did not exclude a second holder"
    # msvcrt's LK_LOCK retries about once a second, so "promptly" is seconds.
    assert entered.wait(30), "the lock was not released"
    thread.join(timeout=30)


@pytest.mark.skipif(
    os.name == "nt", reason="the non-blocking probe is fcntl; see the canary"
)
def test_the_obtain_lock_is_released_when_its_holder_dies(tmp_path):
    """The property that made a lock the answer instead of an ``O_EXCL``
    lockfile: the OS takes it back from a corpse. A client killed mid-spawn
    costs one retry, not a wedged directory every future client has to age
    out with a heuristic. Nothing below ever unlocks anything.
    """
    import subprocess
    import sys

    lock = str(tmp_path / "srv.sock.lock")
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time; sys.path.insert(0, sys.argv[1]); "
            "import momwire_serve_client as m; "
            "lock = m._locked(sys.argv[2], timeout=30.0); lock.__enter__(); "
            "print('held', flush=True); time.sleep(60)",
            str(Path(mech.__file__).parent),
            lock,
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(BlockingIOError):
            _try_lock_now(lock)
        holder.kill()
        holder.wait(timeout=10)
        # Nobody unlocked anything; the kernel did.
        with mech._locked(lock, timeout=10.0):
            pass
    finally:
        holder.kill()
        holder.wait(timeout=10)
        holder.stdout.close()


def _try_lock_now(path: str) -> None:
    """Take the lock without waiting — raises if somebody else holds it."""
    import fcntl
    import os as _os

    fd = _os.open(path, _os.O_CREAT | _os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        _os.close(fd)


def test_the_detached_spawn_uses_each_platforms_own_spelling(monkeypatch, tmp_path):
    """One promise — the server outlives the client and never sees its
    console's Ctrl-C — two spellings. POSIX takes a new session; Windows has
    none, and takes ``DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP`` instead.
    Asserted on the ``Popen`` keywords because the nt branch cannot be RUN
    here, and an unasserted branch is a branch that rots.
    """
    import subprocess

    seen: dict = {}

    class _Fake:
        def __init__(self, *args, **kwargs):
            seen.clear()
            seen.update(kwargs)

    monkeypatch.setattr(subprocess, "Popen", _Fake)
    log_path = str(tmp_path / "srv.log")

    mech.spawn_server(["true"], log_path)
    assert seen["start_new_session"] is True
    assert "creationflags" not in seen

    monkeypatch.setattr(mech.os, "name", "nt")
    mech.spawn_server(["true"], log_path)
    assert "start_new_session" not in seen
    assert seen["creationflags"] == (
        getattr(subprocess, "DETACHED_PROCESS", 0)
        | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    )


def test_the_windows_runtime_directory_is_local_appdata(monkeypatch, tmp_path):
    """``%LOCALAPPDATA%`` is that OS's "a directory this user owns", and the
    uid/0700 checks are skipped there because ``os.getuid`` does not exist on
    nt and ``st_uid`` is a constant rather than an owner. The env override
    still beats everything, on every platform.
    """
    local = tmp_path / "AppData" / "Local"
    local.mkdir(parents=True)
    monkeypatch.delenv("MOMWIRE_PORTAL_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setattr(mech.os, "name", "nt")

    assert mech.runtime_dir() == str(local / "momwire-portal")
    assert (local / "momwire-portal").is_dir()

    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setenv("MOMWIRE_PORTAL_RUNTIME_DIR", str(elsewhere))
    assert mech.runtime_dir() == str(elsewhere)


def test_the_windows_runtime_directory_falls_back_to_a_named_tmp(monkeypatch, tmp_path):
    """No ``%LOCALAPPDATA%`` (a service account) — a tmp directory carrying
    the USERNAME, for the same reason the POSIX fallback carries the uid."""
    import getpass
    import tempfile

    monkeypatch.delenv("MOMWIRE_PORTAL_RUNTIME_DIR", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(mech.os, "name", "nt")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    expected = tmp_path / f"momwire-portal-{getpass.getuser()}"
    assert mech.runtime_dir() == str(expected)
    assert expected.is_dir()
