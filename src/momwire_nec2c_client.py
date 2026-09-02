"""``momwire-nec2c-shared`` — the thin client in front of one resident engine.

SimNEC spawns a fresh engine per evaluation burst and discards it: issue #379
measured 51 processes in one crew-16 session, each alive about a second and
each paying the full cold start (Python, NumPy, parse, mesh, first fill —
~680 ms) before it answered anything. Nothing about that is a bad setting; it
is the process model, and the cross-deck cache dies with the process, so its
main case — a value retyped minutes later — can never hit.

This module is the other half of the fix. It is what SimNEC spawns, and it
does no work: it finds (or starts) ONE long-lived server holding one warm
cache and one thread-pool budget, then pumps stdin to it and its answers back
to stdout until SimNEC closes the pipe. The server is
``python -m momwire.portal --serve`` — today's daemon behind an ``accept()``.

**Nothing here may import ``momwire``**, and that is the entire point rather
than a style rule: ``import momwire`` pulls NumPy and SciPy, which is the
~680 ms this exists to delete. Stdlib only, and even the stdlib is imported
inside the functions that need it, so ``-version`` — the probe SimNEC blocks
on at configure time — costs interpreter start-up and a metadata read.
``importlib.metadata`` answers the version off the installed distribution's
metadata WITHOUT importing the package, which is what makes a version probe
possible from here at all.

The stock ``momwire-nec2c`` is untouched and remains the supported default;
this is the opt-in sibling. The name keeps the ``nec2c`` substring SimNEC's
portal dialog gates on and stays clear of the ``out`` substring its
``testCommand`` refuses — see the packaging tests in ``tests/test_portal.py``.

Which server
------------
One server per ENGINE, and the engine is what the command line says it is:
the socket name is a hash of the momwire version, the interpreter path, and
the engine flags (``--basis``, ``--cache``, ``--cache-stats``,
``--legacy-probe``, ``--idle-timeout``). Two portal-dialog entries differing
only in ``--basis`` therefore reach two servers — the same "two entries are
two engines" rule the flags already carried, now with process identity behind
it. The winning client spawns the server with those same flags, so no
preamble protocol is needed and a client and its server cannot disagree about
what they are.

"What the command line says it is" is made literally true by
:func:`resolve_engine`, which runs first: a basis chosen by the executable's
NAME or by ``MOMWIRE_NEC2C_BASIS`` — the other two sources
``configure_engine`` honours — becomes an ordinary ``--basis`` flag before
anything hashes or spawns. Without that step those two routes were invisible
to the hash, so clients that had chosen different engines collided on one
socket and the loser was served the other's physics with a banner naming it
(momwire#628).

POSIX only. The transport is an AF_UNIX socket; Windows CPython gained
``AF_UNIX`` only in 3.12 and the spawn/lock discipline here has never been run
there, so the client refuses on one line rather than half-working.
"""

from __future__ import annotations

import os
import sys

import momwire_serve_client as _mech

# The stock engine's two probe answers, reconstructed rather than imported —
# importing them would cost the NumPy start-up this module exists to avoid.
# Both shapes are pinned against ``_portal``'s own constants by
# ``tests/test_portal_shared.py``; that test is the link this comment claims.
_LEGACY_PROBE_VERSION = "nec2c.ae6ty.9.1"

# Flags that take a value; everything else that survives is a bare flag.
_VALUE_FLAGS = ("--basis", "--cache-stats", "--idle-timeout")

# The basis roster, duplicated rather than imported — importing
# ``momwire.deck.BASES`` would cost the NumPy start-up this module exists to
# avoid, the same trade ``probe_version`` makes for the version string. The
# duplication is a one-way link stated as a FAILURE rather than a comment:
# ``tests/test_portal_shared.py`` asserts this tuple is exactly
# ``sorted(momwire.deck.BASES)``, so a basis added there and not here fails
# the suite instead of reaching a user as "unknown basis" from an engine that
# supports it.
BASIS_NAMES = (
    "arrayblock",
    "bspline",
    "bspline-d1",
    "hmatrix",
    "pulse",
    "razor-2p",
    "razor-nec5",
    "sinusoidal",
    "sinusoidal-galerkin",
)

# Spelt as the stock engine spells it: `', '.join(sorted(_BASES))`.
_CHOICES = ", ".join(BASIS_NAMES)

# The filename marker `_portal._filename_basis` reads for the stock engine,
# and the segment that tells this sibling apart from it.
_FILENAME_MARKER = "nec2c-"
_SHARED_SEGMENT = "shared"

# The server outlives its last client by this long. 900 s covers a coffee
# break between two SimNEC sessions — the retyped-value case #379 is about —
# and bounds the idle cost of a forgotten server at one process.
DEFAULT_IDLE_TIMEOUT = 900.0

# The spawn wait and its connect-poll backoff moved to
# ``momwire_serve_client`` with the obtain dance (#718 phase 2) — the "a cold
# server imports NumPy before it can bind" rationale rides on
# ``SPAWN_TIMEOUT`` there, shared with the eznec client.

_PUMP_CHUNK = 65536


def probe_version(legacy: bool = False) -> str:
    """The ``-version`` first line, answered without importing momwire.

    ``importlib.metadata`` reads the installed distribution's metadata, so
    this is the same number ``momwire.portal.PROBE_VERSION`` computes from the
    same source — and the reason the probe never has to spawn anything. An
    editable install reports the version recorded at ``pip install -e`` time,
    exactly as the stock engine does.
    """
    if legacy:
        return _LEGACY_PROBE_VERSION
    major, minor = _mech.dist_version()
    return f"NEC2momwire.{major}.{minor}"


def split_argv(argv: list[str]) -> tuple[list[str], bool, bool, float]:
    """``(engine_flags, want_version, want_selftest, idle_timeout)``.

    ``--basis=X`` is normalised to ``--basis X`` so that the two spellings of
    one engine are one server. Unknown arguments are kept and forwarded
    verbatim: the client is not the place to have an opinion about the engine's
    command line, and the server parses it with the engine's own code.
    """
    flat: list[str] = []
    for arg in argv:
        head, sep, tail = arg.partition("=")
        if sep and head in _VALUE_FLAGS:
            flat += [head, tail]
        else:
            flat.append(arg)

    want_version = False
    want_selftest = False
    idle_timeout = DEFAULT_IDLE_TIMEOUT
    engine: list[str] = []
    index = 0
    while index < len(flat):
        arg = flat[index]
        bare = arg.lstrip("-").lower()
        if bare == "version":
            want_version = True
        elif bare == "selftest":
            want_selftest = True
        elif arg == "--idle-timeout":
            index += 1
            try:
                idle_timeout = float(flat[index])
            except (IndexError, ValueError):
                idle_timeout = DEFAULT_IDLE_TIMEOUT
        elif arg in _VALUE_FLAGS:
            engine += flat[index : index + 2]
            index += 1
        else:
            engine.append(arg)
        index += 1
    return engine, want_version, want_selftest, idle_timeout


def filename_basis(prog: str) -> str | None:
    """The basis a COPY or SYMLINK of this command selects by its own name.

    momwire#528's idiom, as this sibling has to spell it. The stock engine
    reads everything after ``nec2c-``, but that marker is already inside this
    command's own name — ``momwire-nec2c-shared`` — so a bare copy would
    otherwise resolve the basis ``"shared"``. The ``shared`` segment is
    therefore consumed when it is the whole suffix (selecting nothing, exactly
    as a plain ``momwire-nec2c`` does) and stripped when it leads one:

    =================================  ==========
    ``momwire-nec2c-shared``           ``None``
    ``momwire-nec2c-shared-razor-2p``  ``razor-2p``
    ``momwire-nec2c-razor-2p``         ``razor-2p``
    ``momwire-nec2c``                  ``None``
    =================================  ==========

    The third row is deliberate rather than incidental: it is the name the
    #528 docs teach, so a user who copies THIS command to it has named a
    basis, and answering them under the default instead would be the silent
    wrong answer momwire#628 is about. Validation is the caller's, as it is
    for the stock engine — an unknown suffix must fail fast at the probe.

    Casefolded, and an empty suffix returned as ``""`` rather than ``None``,
    because those are `_solver.basis_from_program_name`'s rules and this copy
    exists only because the client may not import momwire. A COPY is the thing
    that drifts, so what it copies is the whole rule: a Windows rename to
    ``Momwire-Nec2c-Razor-2p.exe`` names razor-2p, and ``momwire-nec2c-``
    names a basis it failed to spell.
    """
    return _mech.filename_basis(prog, _FILENAME_MARKER, consumed=_SHARED_SEGMENT)


def resolve_engine(
    engine: list[str], prog: str | None = None
) -> tuple[list[str], str | None]:
    """``(engine, error)`` — the engine flags with the basis SPELT OUT.

    ``configure_engine`` resolves a basis from three sources, in the order
    explicit ``--basis`` > executable name > ``MOMWIRE_NEC2C_BASIS``. Only the
    first of those is on the command line, and the command line is all
    :func:`config_key` can see — so before momwire#628 two clients that chose
    different engines by the other two routes hashed to ONE socket and the
    loser was answered under an engine it never asked for, banner and all.

    Resolving here fixes that at the root rather than at the hash: the basis
    becomes an ordinary ``--basis`` flag, so it lands in the socket identity
    and in ``_server_command`` together, and the server stops depending on
    inherited environment to know what it is. That is what makes this module's
    "a client and its server cannot disagree about what they are" true.

    The name is checked against :data:`BASIS_NAMES` for the same reason the
    stock engine checks it at configure time: a typo must fail fast and
    nonzero at the ``-version`` probe, where SimNEC surfaces it, rather than
    become a socket nobody can serve and a spawn that times out. This is the
    ONE opinion the client has about the engine's command line, and it has it
    because it already had to parse ``--basis`` to hash it.
    """
    if "--basis" in engine:
        index = engine.index("--basis")
        name = engine[index + 1] if index + 1 < len(engine) else ""
        if name not in BASIS_NAMES:
            return engine, f"unknown --basis {name!r}; choices: {_CHOICES}"
        return engine, None

    suffix = filename_basis(sys.argv[0] if prog is None else prog)
    env_name = os.environ.get("MOMWIRE_NEC2C_BASIS")
    if suffix is not None:
        name, source = suffix, "the executable name"
    elif env_name:
        name, source = env_name, "MOMWIRE_NEC2C_BASIS"
    else:
        return engine, None
    if name not in BASIS_NAMES:
        return engine, f"unknown basis {name!r} from {source}; choices: {_CHOICES}"
    return [*engine, "--basis", name], None


def config_key(engine: list[str], idle_timeout: float) -> str:
    """The server's identity, as a short hex digest.

    Hashed rather than spelt out because the name has to fit in ``sun_path``
    (107 bytes) with a directory in front of it, and because a ``--cache-stats``
    path in a socket name would leak a user's filesystem into a world-listable
    directory. Value flags keep their pairing; bare flags are sorted, so
    ``--cache --legacy-probe`` and ``--legacy-probe --cache`` are one engine.

    The momwire version and the interpreter path are in the key because an
    upgrade or a different venv is a different engine and must not be served
    by whatever is still resident from before.
    """
    pairs: list[tuple[str, str]] = []
    index = 0
    while index < len(engine):
        arg = engine[index]
        if arg in _VALUE_FLAGS and index + 1 < len(engine):
            pairs.append((arg, engine[index + 1]))
            index += 2
        else:
            pairs.append((arg, ""))
            index += 1
    pairs.sort()

    return _mech.digest(
        [
            probe_version(),
            os.path.realpath(sys.executable),
            f"{idle_timeout!r}",
            *(f"{flag}={value}" for flag, value in pairs),
        ]
    )


runtime_dir = _mech.runtime_dir


def socket_path(engine: list[str], idle_timeout: float) -> str:
    return _mech.socket_path(config_key(engine, idle_timeout))


def _connect(path: str):
    return _mech.connect(path)


def _server_command(
    path: str, engine: list[str], idle_timeout: float, log_path: str
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "momwire.portal",
        "--serve",
        "--socket",
        path,
        "--idle-timeout",
        repr(idle_timeout),
        "--log",
        log_path,
        *engine,
    ]


def _spawn_server(path: str, engine: list[str], idle_timeout: float, log_path: str):
    return _mech.spawn_server(
        _server_command(path, engine, idle_timeout, log_path), log_path
    )


def _obtain(path: str, engine: list[str], idle_timeout: float, log_path: str):
    """A connected socket, starting the server if nobody answers — the
    lock-serialised dance, one owner: :func:`momwire_serve_client.obtain`
    (#718 phase 2 moved it; the whole flock-not-O_EXCL rationale lives on
    its docstring)."""
    return _mech.obtain(
        path, _server_command(path, engine, idle_timeout, log_path), log_path
    )


# Stop reading stdin once this much is queued for the socket. Backpressure
# rather than an unbounded buffer: a client that swallowed a whole file into
# memory to look busy would be lying about a queue SimNEC can already see.
_HIGH_WATER = 1 << 20


def _make_selector(stdin_fd: int):
    """A selector that can watch ``stdin_fd``, whatever it turns out to be.

    ``epoll`` — what ``DefaultSelector`` picks on Linux — refuses a REGULAR
    FILE with ``EPERM``, and stdin is a regular file every time a deck arrives
    by shell redirect (``momwire-nec2c-shared < dipole.nec``, the documented
    command-line use). ``select`` accepts one and reports it permanently
    ready, which is the truth about a file. So: try the fast poller, and drop
    to ``select`` for the fds it will not take.
    """
    import selectors

    selector = selectors.DefaultSelector()
    if stdin_fd >= 0:
        try:
            selector.register(stdin_fd, selectors.EVENT_READ)
        except PermissionError:
            selector.close()
            selector = selectors.SelectSelector()
            selector.register(stdin_fd, selectors.EVENT_READ)
    return selector


def pump(conn, stdin_fd: int, stdout_fd: int) -> int:
    """Bytes both ways until one end is done. The client's whole job.

    Bidirectional because the protocol is: SimNEC blocks in ``readLine()``
    waiting for the ``NX`` echo while its next deck may already be on the way.
    Bytes are forwarded verbatim in both directions — the client never parses a
    frame, never fabricates a banner and never invents an error line, so the
    printout SimNEC reads is the server's bytes or nothing.

    Both directions are driven by ONE readiness loop and the socket is
    non-blocking, which is not fussiness: a blocking ``sendall`` of a big deck
    stops us reading the server's output, the server's own writes then fill
    their buffer, and the two halves wedge against each other with the answer
    already computed. Queueing what will not go out and coming back for it
    when the socket says so is what makes that unreachable.

    Closing stdin (SimNEC's ``Process.destroy()``, or a redirect running out)
    half-closes the socket rather than dropping it, so the server sees a clean
    EOF, finishes the deck it is on, and the remaining output is drained
    before we go.
    """
    import selectors
    import socket as _socket

    selector = _make_selector(stdin_fd)
    selector.register(conn, selectors.EVENT_READ)
    conn.setblocking(False)

    pending = bytearray()
    stdin_open = stdin_fd >= 0
    watching_stdin = stdin_open
    half_closed = False

    try:
        while True:
            # The whole state machine, restated every pass rather than patched
            # at each of the places that could change it — the registrations
            # are then idempotent by construction and cannot drift out of step
            # with the buffer they describe.
            selector.modify(
                conn,
                selectors.EVENT_READ | (selectors.EVENT_WRITE if pending else 0),
            )
            if stdin_open and watching_stdin != (len(pending) < _HIGH_WATER):
                watching_stdin = not watching_stdin
                if watching_stdin:
                    selector.register(stdin_fd, selectors.EVENT_READ)
                else:
                    selector.unregister(stdin_fd)
            if not stdin_open and not pending and not half_closed:
                # Nothing left to send and nothing left to send it from: tell
                # the server so, and stay to drain the answer it still owes.
                conn.shutdown(_socket.SHUT_WR)
                half_closed = True

            for key, events in selector.select():
                if key.fileobj is not conn:
                    data = os.read(stdin_fd, _PUMP_CHUNK)
                    if data:
                        pending += data
                    else:
                        selector.unregister(stdin_fd)
                        stdin_open = watching_stdin = False
                    continue
                if events & selectors.EVENT_READ:
                    data = conn.recv(_PUMP_CHUNK)
                    if not data:
                        return 0
                    _write_all(stdout_fd, data)
                if events & selectors.EVENT_WRITE and pending:
                    del pending[: conn.send(pending)]
    except (BrokenPipeError, ConnectionResetError):
        return 0
    finally:
        selector.close()


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte or die trying — a short write would silently truncate
    a printout, and a truncated printout is a SimNEC that blocks forever."""
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    engine, want_version, want_selftest, idle_timeout = split_argv(argv)

    # Before ANY of the three exits below, because all three depend on it: the
    # probe must refuse a typo'd basis the way the stock engine does, the
    # selftest must run under the engine that was asked for, and the socket
    # identity must carry it. Refused on stdout with 3, as `configure_engine`
    # refuses — this is the same configure-time moment, and SimNEC reads it
    # from the same channel.
    engine, error = resolve_engine(engine)
    if error is not None:
        sys.stdout.write(f"{error}\n")
        sys.stdout.flush()
        return 3

    if want_version:
        # Answered here, never by a spawn: SimNEC probes at configure time and
        # at every engine start, and a probe that waited on a cold server would
        # put the ~680 ms back in the one place the user watches for it.
        sys.stdout.write(f"{probe_version('--legacy-probe' in engine)}\n")
        sys.stdout.flush()
        return 0

    import socket

    if not hasattr(socket, "AF_UNIX"):
        sys.stderr.write(
            "momwire-nec2c-shared needs AF_UNIX sockets and this platform has "
            "none; use momwire-nec2c (the stock engine) instead.\n"
        )
        return 2

    if want_selftest:
        # The one place this command is allowed to cost a momwire import: a
        # smoke test is not a live session, and running it through the real
        # module keeps the embedded decks in one file.
        os.execv(
            sys.executable,
            [sys.executable, "-m", "momwire.portal", "--shared-selftest", *engine],
        )

    try:
        path = socket_path(engine, idle_timeout)
        conn = _obtain(path, engine, idle_timeout, f"{path}.log")
    except Exception as exc:  # noqa: BLE001 - one line out, never a traceback
        sys.stderr.write(f"momwire-nec2c-shared: {type(exc).__name__}: {exc}\n")
        return 2

    try:
        stdin_fd = sys.stdin.fileno() if sys.stdin is not None else -1
    except (AttributeError, OSError, ValueError):
        stdin_fd = -1
    try:
        sys.stdout.flush()
        return pump(conn, stdin_fd, sys.stdout.fileno())
    except (BrokenPipeError, ConnectionResetError):
        # SimNEC ends a session by killing the engine, not by closing it; the
        # mirror image (our stdout going away) is an ordinary end of session.
        return 0
    finally:
        conn.close()


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
