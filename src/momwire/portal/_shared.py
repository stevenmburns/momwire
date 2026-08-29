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

import os
import signal
import sys

from ..serve._server import ConnLog, serve_forever, take_value
from ._portal import _SELFTEST_DECKS, configure_engine, resident_loop


def _connection(conn, number: int, log, solve_lock) -> None:
    """One SimNEC connection: banner, decks, EOF — the portal seam's half of
    :class:`momwire.serve._server.Server` (#718 phase 2 moved the accept
    loop there; what stayed here is exactly what a dialect owns: the byte
    wrapping and the session)."""
    rx = tx = None
    try:
        rx = conn.makefile("r", encoding="utf-8", errors="replace")
        tx = conn.makefile("w", encoding="utf-8", errors="replace", newline="\n")
        resident_loop(rx, tx, ConnLog(log, f"[conn {number}] "), solve_lock)
    finally:
        for handle in (tx, rx):
            if handle is None:
                continue
            try:
                handle.close()
            except OSError:
                pass


def serve_main(argv: list[str]) -> int:
    """``--serve``: configure the engine once, then answer sockets forever."""
    argv = [a for a in argv if a != "--serve"]
    path, argv = take_value(argv, "--socket")
    log_path, argv = take_value(argv, "--log")
    idle_raw, argv = take_value(argv, "--idle-timeout", "900")

    def _configure(log):
        # ONCE per process. The engine flags are the server's identity — the
        # client hashed them into the socket name — so a second call could only
        # ever restate them, at the cost of the cache they were chosen for.
        rest, _legacy_probe, code = configure_engine(argv, log)
        if code is not None:
            return code
        if rest:
            log.write(f"unused server arguments: {rest}\n")
        return None

    return serve_forever(path, idle_raw, log_path, _connection, configure=_configure)


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

    import momwire_serve_client

    # What the client will call its server's address file on THIS box — a
    # ``.sock`` under AF_UNIX, a ``.port`` rendezvous under loopback TCP
    # (#718 phase 2 unit 2). Asked rather than spelt, so the selftest cannot
    # report "no server" on a transport it simply failed to look for.
    suffix = momwire_serve_client.address_suffix()

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

        sockets = sorted(n for n in os.listdir(room) if n.endswith(suffix))
        checks["one server socket"] = len(sockets) == 1

        second = _run("".join(_SELFTEST_DECKS[2:]))
        checks["second client exits 0"] = second.returncode == 0
        checks["second client served warm"] = (
            second.returncode == 0
            and _sentinels(second.stdout) == 2
            and second.stdout.count("ANTENNA INPUT PARAMETERS") == 2
        )
        checks["server survived both"] = (
            sorted(n for n in os.listdir(room) if n.endswith(suffix)) == sockets
        )
    finally:
        for name in os.listdir(room):
            if name.endswith(suffix):
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
