"""The shared engine's server half (#379) — today's daemon behind ``accept()``.

``python -m momwire.portal --serve --socket PATH [engine flags]``. One process
per engine configuration, started on demand by the first
``momwire-nec2c-shared`` client that finds nobody listening (see
:mod:`momwire_nec2c_client`, which owns the socket naming, the spawn lock and
the pumping) and living on afterwards so the next client is answered warm.

What the split buys is not speed at the socket; it is the three things that
died with every one-second engine process:

* **the cross-deck cache survives**. ``--cache`` keys solvers by operator
  identity, and the case it exists for — a value dragged back to something
  already probed, minutes later — needs a process that is still there minutes
  later. :func:`~momwire.portal._portal.configure_engine` is therefore called
  ONCE at start-up, never per connection: it resets the cache exactly as a
  fresh process would, which is right at start-up and would be a lobotomy in
  a connection handler.
* **one cache, not N**. A crew of 16 fragmented the cache 16 ways and paid
  16×90 MB of resident interpreter for it. They funnel into one process here.
* **one thread-pool budget**. Solves are serialised on one global lock, so 16
  concurrent clients cannot oversubscribe the box between them. A pool is
  deliberately NOT built: a warm solve is ~12 ms, the solver already threads
  inside itself, and a second concurrent solve would compete with the first
  for the same cores while doubling peak memory. Serialising is the honest
  version of the budget the crew was pretending to have.

Everything protocol-shaped is :func:`~momwire.portal._portal.resident_loop`,
called once per connection with the socket's own file objects. That is not a
convenience: the printout a client reads must be byte-identical to the stock
engine's, and the only way to guarantee that is for it to be the same loop
rather than a second implementation of the framing.

Failure modes, and where each one stops:

* **a client dies mid-solve** — the write fails, that connection's output is
  discarded and its handler closes. The solve that was running completes and
  warms the cache; every other connection is untouched. A server that died
  with its client would hand the crew a way to take the engine down.
* **stderr** — deck warnings go to the server log, not back down the socket.
  ``NEC2Daemon`` never drains stderr (a full pipe buffer deadlocks the UI),
  and with N clients on one process there is no honest way to route a global
  stream to one of them anyway. The refusal contract is unaffected: ``ERROR:``
  lines live in the PRINTOUT and the ``NX`` sentinel is always answered.
* **nobody comes back** — the server exits after ``--idle-timeout`` seconds
  with no connection open, and unlinks its socket on the way out. A client
  that finds a socket file with nothing behind it removes it under the spawn
  lock and starts a fresh server, so the two halves of that race cannot leave
  a directory a user has to clean by hand.
"""

from __future__ import annotations

import errno
import os
import signal
import socket
import sys
import threading
import time

from ._portal import _SELFTEST_DECKS, configure_engine, resident_loop

# The accept() poll. Nothing waits on it — it only bounds how long after the
# idle deadline the process actually goes away.
_ACCEPT_POLL = 0.25

_LISTEN_BACKLOG = 64


class _ConnLog:
    """A connection's stderr, prefixed and pointed at the server log.

    Per-connection because a log with 16 clients in it is unreadable without
    knowing which said what, and a plain file object because this is a log:
    losing a line to a full disk must never cost a solve.
    """

    def __init__(self, stream, prefix: str) -> None:
        self._stream = stream
        self._prefix = prefix

    def write(self, text: str) -> int:
        try:
            for line in text.splitlines(True):
                self._stream.write(f"{self._prefix}{line}")
        except OSError:
            pass
        return len(text)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except OSError:
            pass


def _take_value(argv: list[str], flag: str, default: str | None = None):
    """Pull ``--flag VALUE`` (or ``--flag=VALUE``) out of ``argv``."""
    rest: list[str] = []
    value = default
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == flag:
            index += 1
            value = argv[index] if index < len(argv) else default
        elif arg.startswith(f"{flag}="):
            value = arg.split("=", 1)[1]
        else:
            rest.append(arg)
        index += 1
    return value, rest


class _Server:
    """The accept loop and the idle clock. One instance is one process."""

    def __init__(self, path: str, idle_timeout: float, log) -> None:
        self.path = path
        self.idle_timeout = idle_timeout
        self.log = log
        # ONE global solve lock — the budget, stated once (see module docstring).
        self.solve_lock = threading.Lock()
        self.state = threading.Lock()
        self.active = 0
        self.idle_since = time.monotonic()
        self.served = 0
        self.stopping = False

    def note(self, text: str) -> None:
        try:
            self.log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {text}\n")
            self.log.flush()
        except OSError:
            pass

    def _bind(self):
        """Bind, or hand the path back to whoever already owns it.

        The client only spawns under its lock and only after unlinking a dead
        socket, so ``EADDRINUSE`` here means a peer server won a race the lock
        was supposed to prevent. Losing gracefully (exit 0, touch nothing) is
        the only safe answer: unlinking a live peer's socket would strand every
        client already talking to it.
        """
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(self.path)
        except OSError as exc:
            listener.close()
            if exc.errno != errno.EADDRINUSE:
                # Anything else (a path over sun_path's 107 bytes, a read-only
                # directory) is a configuration mistake, and the client is
                # waiting on a socket that will never appear — so it must reach
                # the log as itself, not as a bind retried against nothing.
                raise
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(self.path)
            except OSError:
                alive = False
            else:
                alive = True
            finally:
                probe.close()
            if alive:
                return None
            os.unlink(self.path)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(self.path)
        os.chmod(self.path, 0o600)
        listener.listen(_LISTEN_BACKLOG)
        return listener

    def stop(self, *_args) -> None:
        self.stopping = True

    def run(self) -> int:
        listener = self._bind()
        if listener is None:
            self.note(f"another server already owns {self.path}; exiting")
            return 0
        self.note(f"listening pid={os.getpid()} socket={self.path}")
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.stop)

        listener.settimeout(_ACCEPT_POLL)
        try:
            while not self.stopping:
                try:
                    conn, _addr = listener.accept()
                except TimeoutError:
                    if self._idle_expired():
                        self.note(f"idle {self.idle_timeout:g}s; exiting")
                        break
                    continue
                except OSError as exc:
                    self.note(f"accept failed: {exc!r}")
                    break
                with self.state:
                    self.active += 1
                    self.served += 1
                    number = self.served
                threading.Thread(
                    target=self._serve, args=(conn, number), daemon=True
                ).start()
        finally:
            # Unlink before closing: a client that connects in the window
            # between the two gets a live server that is about to go, which its
            # own stale-socket recovery handles; the reverse order can leave the
            # path behind if the process is killed between them.
            try:
                os.unlink(self.path)
            except OSError:
                pass
            listener.close()
        self.note(f"stopped after {self.served} connection(s)")
        return 0

    def _idle_expired(self) -> bool:
        with self.state:
            return (
                self.active == 0
                and time.monotonic() - self.idle_since >= self.idle_timeout
            )

    def _serve(self, conn, number: int) -> None:
        """One connection: banner, decks, EOF. Never raises past here.

        Every exception is this connection's problem alone — a client killed
        mid-solve (SimNEC's ``Process.destroy()``) surfaces as a write failure
        on the socket, and the server's job at that point is to drop the output
        nobody is reading and go back to accepting.
        """
        rx = tx = None
        try:
            rx = conn.makefile("r", encoding="utf-8", errors="replace")
            tx = conn.makefile("w", encoding="utf-8", errors="replace", newline="\n")
            resident_loop(
                rx, tx, _ConnLog(self.log, f"[conn {number}] "), self.solve_lock
            )
        except (BrokenPipeError, ConnectionResetError):
            self.note(f"[conn {number}] client went away; output discarded")
        except OSError as exc:
            self.note(f"[conn {number}] socket error: {exc!r}")
        except BaseException as exc:  # noqa: BLE001 - one bad deck is not the server
            self.note(f"[conn {number}] handler failed: {type(exc).__name__}: {exc}")
        finally:
            for handle in (tx, rx):
                if handle is None:
                    continue
                try:
                    handle.close()
                except OSError:
                    pass
            try:
                conn.close()
            except OSError:
                pass
            with self.state:
                self.active -= 1
                self.idle_since = time.monotonic()


def serve_main(argv: list[str]) -> int:
    """``--serve``: configure the engine once, then answer sockets forever."""
    argv = [a for a in argv if a != "--serve"]
    path, argv = _take_value(argv, "--socket")
    log_path, argv = _take_value(argv, "--log")
    idle_raw, argv = _take_value(argv, "--idle-timeout", "900")

    if not path:
        sys.stderr.write("--serve needs --socket PATH\n")
        return 3
    try:
        idle_timeout = float(idle_raw)
    except (TypeError, ValueError):
        sys.stderr.write(f"--idle-timeout must be a number, not {idle_raw!r}\n")
        return 3

    log = open(log_path or f"{path}.log", "a", encoding="utf-8", errors="replace")
    try:
        # ONCE per process. The engine flags are the server's identity — the
        # client hashed them into the socket name — so a second call could only
        # ever restate them, at the cost of the cache they were chosen for.
        rest, _legacy_probe, code = configure_engine(argv, log)
        if code is not None:
            log.flush()
            return code
        if rest:
            log.write(f"unused server arguments: {rest}\n")
        return _Server(path, idle_timeout, log).run()
    finally:
        log.close()


def shared_selftest(argv: list[str], stdout=None) -> int:
    """``momwire-nec2c-shared --selftest`` — the deployment smoke for #379.

    The stock ``--selftest`` proves the PROCESS; this proves the SPLIT, which
    is the part a bare ``pip install`` can get wrong in ways no unit test on a
    developer box would see. It runs the same embedded decks through two
    SEPARATE client invocations against one private server, and requires the
    property the issue is about: the second invocation is answered by a process
    the first one left behind.
    """
    import shutil
    import subprocess
    import tempfile

    stdout = sys.stdout if stdout is None else stdout
    argv = [a for a in argv if a != "--shared-selftest"]
    room = tempfile.mkdtemp(prefix="momwire-portal-selftest-")
    env = {**os.environ, "MOMWIRE_PORTAL_RUNTIME_DIR": room}
    client = [sys.executable, "-m", "momwire_nec2c_client", *argv]

    def _run(decks: str):
        return subprocess.run(
            client, input=decks, capture_output=True, text=True, timeout=300, env=env
        )

    checks: dict[str, bool] = {}
    try:
        first = _run("".join(_SELFTEST_DECKS[:2]))
        checks["first client exits 0"] = first.returncode == 0
        checks["banner present"] = "VERSION:" in first.stdout
        checks["3 solve groups answered"] = (
            first.stdout.count("ANTENNA INPUT PARAMETERS") == 3
        )
        checks["2 NX sentinels"] = _sentinels(first.stdout) == 2
        checks["client stderr quiet"] = first.stderr.strip() == ""

        sockets = sorted(name for name in os.listdir(room) if name.endswith(".sock"))
        checks["one server socket"] = len(sockets) == 1

        second = _run("".join(_SELFTEST_DECKS[2:]))
        checks["second client exits 0"] = second.returncode == 0
        checks["second client served warm"] = (
            second.returncode == 0
            and _sentinels(second.stdout) == 2
            and second.stdout.count("ANTENNA INPUT PARAMETERS") == 2
        )
        checks["server survived both"] = (
            sorted(name for name in os.listdir(room) if name.endswith(".sock"))
            == sockets
        )
    finally:
        for name in os.listdir(room):
            if name.endswith(".sock"):
                _terminate(os.path.join(room, name))
        shutil.rmtree(room, ignore_errors=True)

    for name, ok in checks.items():
        stdout.write(f"  {'ok  ' if ok else 'FAIL'} {name}\n")
    passed = all(checks.values())
    stdout.write("PASS\n" if passed else "FAIL\n")
    stdout.flush()
    return 0 if passed else 1


def _sentinels(text: str) -> int:
    return sum(
        1
        for line in text.splitlines()
        if line.lstrip().startswith("DATA CARD No:") and " NX " in line
    )


def _terminate(path: str) -> None:
    """Ask the server on ``path`` to go, by pid out of its own log line."""
    try:
        with open(f"{path}.log", encoding="utf-8", errors="replace") as handle:
            pids = [
                int(word.split("=", 1)[1])
                for line in handle
                for word in line.split()
                if word.startswith("pid=")
            ]
    except OSError:
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def shared_main(argv: list[str]) -> int:
    """The ``python -m momwire.portal`` modes that are not the stock daemon."""
    if "--shared-selftest" in argv:
        return shared_selftest(argv)
    return serve_main(argv)
