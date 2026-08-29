"""The thin clients' shared mechanics — stdlib only, imported by both.

``momwire_nec2c_client`` (the SimNEC leg, #379) and ``momwire_eznec_client``
(the EZNEC leg, momwire#532) delete the same ~680 ms the same way: by not
importing momwire or NumPy, ever. What they share is everything about
FINDING the warm server — the per-user runtime directory, the socket path
and its ``sun_path`` guard, the connect probe, the detached spawn, and the
lock-serialised obtain dance — and none of it is dialect-shaped, so it lives
here once (#718 phase 2). What stays per-client is the wire protocol (the
pipe pump vs the one-shot) and the CLI contract, which ARE the dialects.

Like its importers, this module must stay import-light: the whole point of
the split is that a client's life is milliseconds, and every import here is
paid on every launch. Anything heavier than ``os`` is imported inside the
function that needs it.

POSIX AF_UNIX today. The Windows transport (AF_UNIX where the build has it,
loopback TCP where it does not) and the ``flock`` analogue land HERE in the
arc's next unit, so neither client sees the platform.
"""

from __future__ import annotations

import os

# How long a client waits for a server it spawned before giving up. Generous
# because the wait is the cold NumPy/momwire import — the thing the split
# exists to pay once — on a box that may be busy solving.
SPAWN_TIMEOUT = 120.0

# Connect-poll backoff while the spawned server imports.
_POLL_MIN = 0.005
_POLL_MAX = 0.05


def dist_version() -> tuple[str, str]:
    """``(major, minor)`` of the installed momwire distribution, metadata-only.

    ``importlib.metadata`` reads the installed distribution's metadata, so
    this is the same number the package computes from the same source — and
    the reason no version probe ever has to spawn anything. An editable
    install reports the version recorded at ``pip install -e`` time.
    """
    try:
        from importlib.metadata import version as _pkg_version

        major, minor = _pkg_version("momwire").split(".")[:2]
    except Exception:  # pragma: no cover - no installed metadata (source tree)
        major, minor = "0", "0"
    return major, minor


def digest(parts: list[str]) -> str:
    """A short hex identity for a server configuration.

    Hashed rather than spelt out because the name has to fit in ``sun_path``
    with a directory in front of it, and because a filesystem path in a
    socket name would leak a user's layout into a world-listable directory.
    The caller decides what is identity — the rule from #379: an upgrade, a
    different venv, a different basis is a DIFFERENT engine and must not be
    served by whatever is still resident from before.
    """
    import hashlib

    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


def filename_basis(prog: str, marker: str, consumed: str | None = None) -> str | None:
    """The basis a COPY or SYMLINK of a command selects by its own name.

    momwire#528's idiom, spelt once for every thin client: everything after
    ``marker`` in the program's own basename names the basis. ``consumed`` is
    the client's own trailing segment (``"shared"`` for
    ``momwire-nec2c-shared``) — swallowed when it is the whole suffix and
    stripped when it leads one, so a bare copy of the client selects nothing.

    Casefolded, and an empty suffix returned as ``""`` rather than ``None``,
    because those are ``_solver.basis_from_program_name``'s rules and this
    copy exists only because a client may not import momwire. A COPY is the
    thing that drifts, so what it copies is the whole rule: a Windows rename
    to ``Momwire-Nec2c-Razor.exe`` names razor, and a trailing marker names
    a basis it failed to spell.
    """
    name = prog.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    if name.endswith(".exe"):
        name = name[:-4]
    if marker not in name:
        return None
    suffix = name.split(marker, 1)[1]
    if consumed is not None:
        if suffix == consumed:
            return None
        if suffix.startswith(f"{consumed}-"):
            suffix = suffix[len(consumed) + 1 :]
    return suffix


def runtime_dir() -> str:
    """The per-user directory holding sockets, lockfiles and server logs.

    ``$XDG_RUNTIME_DIR`` first — it is the one directory the OS promises is
    private to this user and cleaned at logout, which is what a socket wants.
    Where it is unset (a bare ssh session, cron, macOS) fall back to
    ``<tmp>/momwire-portal-<uid>``, created 0700 and REFUSED if it already
    exists owned by somebody else: a socket in a world-writable directory is a
    socket anyone can pre-empt.

    ``$MOMWIRE_PORTAL_RUNTIME_DIR`` overrides both, which is how the tests get
    an isolated server and how a user with an odd home can move it. One
    directory for BOTH clients: the keys already separate the engines, and a
    second env override would be a second thing to document and get wrong.
    """
    import tempfile

    base = os.environ.get("MOMWIRE_PORTAL_RUNTIME_DIR")
    if not base:
        xdg = os.environ.get("XDG_RUNTIME_DIR")
        if xdg and os.path.isdir(xdg):
            base = os.path.join(xdg, "momwire-portal")
        else:
            base = os.path.join(tempfile.gettempdir(), f"momwire-portal-{os.getuid()}")
    os.makedirs(base, mode=0o700, exist_ok=True)
    info = os.stat(base)
    if info.st_uid != os.getuid():
        raise PermissionError(f"{base} is not owned by uid {os.getuid()}")
    if info.st_mode & 0o077:
        os.chmod(base, 0o700)
    return base


# ``sun_path`` is 108 bytes including the terminator on Linux (104 on the BSDs
# and macOS); a name that overruns it fails at bind() inside a detached server,
# where the only symptom the user sees is a client waiting for a socket that
# never appears. Checked HERE, against the tighter of the two limits, so the
# refusal names the real cause. The default directories are short by
# construction — the only way to trip this is a long
# ``$MOMWIRE_PORTAL_RUNTIME_DIR``.
SUN_PATH_MAX = 103


def socket_path(key: str) -> str:
    path = os.path.join(runtime_dir(), f"{key}.sock")
    if len(os.fsencode(path)) > SUN_PATH_MAX:
        raise ValueError(
            f"socket path is {len(os.fsencode(path))} bytes, over the "
            f"{SUN_PATH_MAX}-byte AF_UNIX limit: {path}. Set "
            "MOMWIRE_PORTAL_RUNTIME_DIR to a shorter directory."
        )
    return path


def connect(path: str):
    """An open socket, or ``None`` if nothing is listening on ``path``."""
    import socket

    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        conn.connect(path)
    except OSError:
        conn.close()
        return None
    return conn


def spawn_server(command: list[str], log_path: str):
    """Start the server detached: it must outlive the client that started it.

    ``start_new_session`` puts it in its own session and process group, so the
    terminal SIGINT (or the host's ``Process.destroy()`` on the client) does
    not reach it, and stdio goes to the log next to the socket — a server
    writing to an inherited pipe would corrupt some other client's transcript
    the day a deck warns.

    The ``Popen`` handle comes back even though the child is detached: it is
    the only way to tell "still importing NumPy" from "died at start-up", and
    without it a misconfiguration is a client that waits out the whole spawn
    timeout in silence.
    """
    import subprocess

    log = open(log_path, "ab", buffering=0)
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log.close()


def obtain(path: str, server_command: list[str], log_path: str):
    """A connected socket, starting the server if nobody answers.

    The fast path takes no lock at all: a live server answers ``connect`` and
    that is the whole story. Only a failed connect enters the slow path, and
    there the lockfile serialises the decision so a crew firing 16 clients at
    once starts ONE server — the losers block on the lock, and by the time
    they hold it the winner's server is already listening, so their retry is
    the fast path again.

    ``flock`` rather than an ``O_EXCL`` lockfile because the kernel drops a
    flock when the holder dies: a client killed mid-spawn costs one retry, not
    a wedged directory that every future client has to age out with a
    heuristic. A socket file left behind by a server that died without
    unlinking is detected here (connect refuses, the path exists) and removed
    under the lock, which is the only place removing it is safe.
    """
    import fcntl
    import time

    conn = connect(path)
    if conn is not None:
        return conn

    lock_path = f"{path}.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        conn = connect(path)
        if conn is not None:
            return conn
        if os.path.exists(path):
            os.unlink(path)
        server = spawn_server(server_command, log_path)

        deadline = time.monotonic() + SPAWN_TIMEOUT
        delay = _POLL_MIN
        while time.monotonic() < deadline:
            conn = connect(path)
            if conn is not None:
                return conn
            status = server.poll()
            if status is not None:
                # One more try before believing it: a server that bound, was
                # answered and exited between the connect above and this poll
                # is a lost race, not a failure.
                conn = connect(path)
                if conn is not None:
                    return conn
                raise ChildProcessError(
                    f"the momwire server exited {status} without "
                    f"answering {path}; see {log_path}"
                )
            time.sleep(delay)
            delay = min(delay * 1.5, _POLL_MAX)
        raise TimeoutError(
            f"the momwire server did not answer {path} within "
            f"{SPAWN_TIMEOUT:g}s; see {log_path}"
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


__all__ = [
    "SPAWN_TIMEOUT",
    "SUN_PATH_MAX",
    "connect",
    "digest",
    "dist_version",
    "filename_basis",
    "obtain",
    "runtime_dir",
    "socket_path",
    "spawn_server",
]
