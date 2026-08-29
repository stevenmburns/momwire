"""The resident server's transport mechanics, seam-agnostic (#718 phase 2).

Moved from ``portal/_shared.py`` (which keeps the portal's spelling of it)
and parameterized the same way #719 U4 parameterized the loop: the accept
loop, the idle clock, the bind-or-lose-gracefully dance, the unlink-on-exit
discipline and the one global solve lock are transport, and identical for
every seam; the ONE thing a dialect owns is what happens on a connection —
how the raw socket is wrapped and which session runs over it. That arrives
as the ``connection`` callable, so the eznec daemon and the SimNEC daemon
are the same server hosting different seams (momwire#532: the resident
process this arc exists to build).

Everything here is POSIX AF_UNIX today; the Windows transport (AF_UNIX
where the build has it, loopback TCP where it does not) is the next unit of
the #718 arc and lands inside this module so neither dialect sees it.
"""

from __future__ import annotations

import errno
import os
import signal
import socket
import sys
import threading
import time

# The accept() poll. Nothing waits on it — it only bounds how long after the
# idle deadline the process actually goes away.
_ACCEPT_POLL = 0.25

_LISTEN_BACKLOG = 64


class ConnLog:
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


def take_value(argv: list[str], flag: str, default: str | None = None):
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


class Server:
    """The accept loop and the idle clock. One instance is one process.

    ``connection(conn, number, log, solve_lock)`` is the seam's half: wrap
    the raw socket however the dialect's bytes demand and run its session
    over it. It may raise freely — socket deaths and bad decks are one
    connection's problem, never the server's, and the bookkeeping wrapper
    here is what guarantees that.
    """

    def __init__(self, path: str, idle_timeout: float, log, connection) -> None:
        self.path = path
        self.idle_timeout = idle_timeout
        self.log = log
        self.connection = connection
        # ONE global solve lock — the budget, stated once: solves from every
        # connection serialise into one thread-pool allocation, which is the
        # honest version of the concurrency a crew of processes pretended to
        # have (portal/_shared.py's module docstring, #379/#385).
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
        # Only the main thread may install handlers; embedded servers (the
        # in-process tests) rely on stop() instead, and a real daemon process
        # always runs this on its main thread.
        if threading.current_thread() is threading.main_thread():
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
        """One connection, bookkept. Never raises past here.

        Every exception is this connection's problem alone — a client killed
        mid-solve (SimNEC's ``Process.destroy()``) surfaces as a write failure
        on the socket, and the server's job at that point is to drop the output
        nobody is reading and go back to accepting.
        """
        try:
            self.connection(conn, number, self.log, self.solve_lock)
        except (BrokenPipeError, ConnectionResetError):
            self.note(f"[conn {number}] client went away; output discarded")
        except OSError as exc:
            self.note(f"[conn {number}] socket error: {exc!r}")
        except BaseException as exc:  # noqa: BLE001 - one bad deck is not the server
            self.note(f"[conn {number}] handler failed: {type(exc).__name__}: {exc}")
        finally:
            try:
                conn.close()
            except OSError:
                pass
            with self.state:
                self.active -= 1
                self.idle_since = time.monotonic()


def serve_forever(
    path: str | None,
    idle_raw: str | None,
    log_path: str | None,
    connection,
    configure=None,
) -> int:
    """The ``--serve`` mode's shared spine: validate, log, run.

    ``configure`` is the seam's once-per-process setup (the portal's
    ``configure_engine``), called with the log before the server binds; it
    returns an exit code to abort with, or None to proceed. A seam whose
    setup is carried entirely by ``connection`` (the eznec daemon: the basis
    rides in the seam closure) passes nothing.
    """
    if not path:
        sys.stderr.write("--serve needs --socket PATH\n")
        return 3
    try:
        idle_timeout = float(idle_raw if idle_raw is not None else 900)
    except (TypeError, ValueError):
        sys.stderr.write(f"--idle-timeout must be a number, not {idle_raw!r}\n")
        return 3

    log = open(log_path or f"{path}.log", "a", encoding="utf-8", errors="replace")
    try:
        if configure is not None:
            code = configure(log)
            if code is not None:
                log.flush()
                return code
        return Server(path, idle_timeout, log, connection).run()
    finally:
        log.close()
