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

Two transports, one set of mechanics (#718 phase 2 unit 2). AF_UNIX where
the build has it — every POSIX box, and the newer Windows CPythons — and
loopback TCP with a rendezvous file where it does not. The choice is made
ONCE, by :func:`transport`, and everything downstream (the address file's
suffix, the connect, the bind, the ``sun_path`` guard) reads it from there;
the lock and the detached spawn have per-OS spellings for the same
property, chosen by ``os.name``. Neither client and neither daemon contains
the word Windows.
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

# The two transports, and the suffix each one's ADDRESS FILE carries. The
# suffixes must differ: a directory can hold leftovers from a run in the
# other mode, and a ``.sock`` mistaken for a rendezvous (or the reverse)
# would be a client connecting to nonsense instead of finding nothing.
UNIX = "unix"
TCP = "tcp"
_ADDRESS_SUFFIX = {UNIX: ".sock", TCP: ".port"}

# The transport override. ``tcp`` on a box that HAS AF_UNIX is how the whole
# TCP path is exercised in CI and in the local suite; ``unix`` on a box that
# does not is a loud failure at connect/bind, never a silent fallback.
TRANSPORT_ENV = "MOMWIRE_SERVE_TRANSPORT"

# Loopback, and only loopback: the rendezvous is a substitute for a filesystem
# path, not a network service, and a server reachable off-box would be an
# unauthenticated solver on the LAN.
RENDEZVOUS_HOST = "127.0.0.1"


def transport() -> str:
    """``"unix"`` or ``"tcp"`` — which transport this install serves on.

    One function, one owner: the client's connect, the client's address
    naming and the server's bind all ask HERE, so a client and its server
    cannot disagree about what they are speaking.

    The autodetect is ``socket.AF_UNIX``, because that is exactly the
    question — a Unix socket is a file the two ends rendezvous on, which is
    everything the obtain dance is built out of, and the TCP mode exists
    only to rebuild that rendezvous out of a file plus a port on the builds
    that lack it. :data:`TRANSPORT_ENV` beats the autodetect in both
    directions and never falls back: an explicit ``unix`` where AF_UNIX is
    absent raises at the first socket, which is the honest answer to "I told
    you to use a transport this Python does not have".
    """
    chosen = (os.environ.get(TRANSPORT_ENV) or "").strip().lower()
    if chosen:
        if chosen not in _ADDRESS_SUFFIX:
            raise ValueError(
                f"{TRANSPORT_ENV}={chosen!r} is not a transport; choices: "
                f"{', '.join(sorted(_ADDRESS_SUFFIX))}"
            )
        return chosen

    import socket

    return UNIX if hasattr(socket, "AF_UNIX") else TCP


def address_suffix(mode: str | None = None) -> str:
    """The suffix an address file carries in ``mode`` (default: this box's)."""
    mode = transport() if mode is None else mode
    try:
        return _ADDRESS_SUFFIX[mode]
    except KeyError:
        raise ValueError(
            f"{mode!r} is not a transport; choices: "
            f"{', '.join(sorted(_ADDRESS_SUFFIX))}"
        ) from None


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

    Windows has neither name: ``%LOCALAPPDATA%`` is that OS's spelling of "a
    directory this user owns", with ``<tmp>/momwire-portal-<username>`` behind
    it for the rare service account that has none.

    ``$MOMWIRE_PORTAL_RUNTIME_DIR`` overrides all of them, which is how the
    tests get an isolated server and how a user with an odd home can move it.
    One directory for BOTH clients: the keys already separate the engines, and
    a second env override would be a second thing to document and get wrong.
    """
    import tempfile

    base = os.environ.get("MOMWIRE_PORTAL_RUNTIME_DIR")
    if not base:
        if os.name == "nt":
            local = os.environ.get("LOCALAPPDATA")
            if local and os.path.isdir(local):
                base = os.path.join(local, "momwire-portal")
            else:
                import getpass

                base = os.path.join(
                    tempfile.gettempdir(), f"momwire-portal-{getpass.getuser()}"
                )
        else:
            xdg = os.environ.get("XDG_RUNTIME_DIR")
            if xdg and os.path.isdir(xdg):
                base = os.path.join(xdg, "momwire-portal")
            else:
                base = os.path.join(
                    tempfile.gettempdir(), f"momwire-portal-{os.getuid()}"
                )
    os.makedirs(base, mode=0o700, exist_ok=True)
    if os.name == "nt":
        # No uid check and no 0700: ``os.getuid`` does not exist on nt, and
        # ``st_uid`` is a constant zero there rather than an owner, so the
        # POSIX check below would compare two lies. What replaces it is the
        # DIRECTORY: %LOCALAPPDATA% is per-user by construction (it is inside
        # the user's own profile, which the OS ACLs to that user), and the
        # tmp fallback carries the username in its name for the same reason
        # the POSIX one carries the uid.
        return base
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


def socket_path(key: str, mode: str | None = None) -> str:
    """The ADDRESS FILE for ``key``: the socket itself, or the rendezvous.

    One name for the two transports on purpose — every caller (both clients,
    both daemons' ``--socket`` flag, the lockfile, the log) wants "the file
    this server is known by", and that is what this returns. Only the suffix
    and what lives behind it differ, and ``mode`` exists so a test can ask
    for the other transport's spelling without moving the environment.
    """
    mode = transport() if mode is None else mode
    path = os.path.join(runtime_dir(), f"{key}{address_suffix(mode)}")
    # A TCP rendezvous is an ordinary file, so the kernel's 108-byte name
    # limit for sockets simply does not apply to it; enforcing it there would
    # refuse a configuration that works.
    if mode == UNIX and len(os.fsencode(path)) > SUN_PATH_MAX:
        raise ValueError(
            f"socket path is {len(os.fsencode(path))} bytes, over the "
            f"{SUN_PATH_MAX}-byte AF_UNIX limit: {path}. Set "
            "MOMWIRE_PORTAL_RUNTIME_DIR to a shorter directory."
        )
    return path


def publish_rendezvous(path: str, port: int) -> None:
    """Publish ``127.0.0.1:<port>`` at ``path``, atomically.

    Write-then-``os.replace`` rather than write-in-place because a reader is
    a client that may arrive at any instant: a partially written file would
    be read as a port that is a prefix of the real one, i.e. a connect to
    somebody else's service, which is worse than every failure this whole
    dance exists to avoid. ``os.replace`` is atomic on POSIX and on Windows,
    so a reader sees the old contents or the new ones and nothing between.

    The temporary carries the publisher's pid: two servers racing to claim
    one key would otherwise interleave inside a single scratch file, and the
    survivor of that race would publish a mixture of the two.
    """
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            os.write(fd, f"{RENDEZVOUS_HOST}:{port}\n".encode("ascii"))
        finally:
            os.close(fd)
        # Windows refuses to replace a file a reader currently holds open
        # (their ordinary open lacks FILE_SHARE_DELETE), and a reader is a
        # client mid-``read_rendezvous`` — a 64-byte read, microseconds.
        # Retry briefly rather than fail the bind: the first canary run
        # proved this is not theoretical (WinError 5 in the race gate).
        # POSIX never takes the except.
        import time

        deadline = time.monotonic() + 2.0
        while True:
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.005)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_rendezvous(path: str):
    """``(host, port)`` published at ``path``, or ``None`` if nothing usable is.

    Every unusable shape — no file, an empty one, a truncated one, bytes
    that are not a port — is ONE answer, "nobody listening", because that is
    what the caller does about all of them: take the lock, remove the litter
    and start a server. A rendezvous file is never a reason to raise.
    """
    try:
        with open(path, "rb") as handle:
            text = handle.read(64).decode("ascii")
    except (OSError, UnicodeDecodeError):
        return None
    host, sep, port = text.strip().rpartition(":")
    if not sep or not host or not port.isdigit():
        return None
    return host, int(port)


def connect(path: str):
    """An open socket, or ``None`` if nothing is listening at ``path``.

    ``path`` is the address file in both modes, and both modes answer the
    same two ways: a live server, or ``None``. In TCP mode "no rendezvous
    file" and "a rendezvous nobody answers" are both ``None`` — the second
    is a stale file, and :func:`obtain` is the only place it is safe to
    remove one.
    """
    import socket

    if transport() == TCP:
        address = read_rendezvous(path)
        if address is None:
            return None
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    else:
        # ``socket.AF_UNIX`` missing here is an explicit ``unix`` override on
        # a build without it: an AttributeError naming the attribute, not a
        # quiet fallback to a transport the user did not ask for.
        address = path
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        conn.connect(address)
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

    if os.name == "nt":
        # Windows has no sessions. ``DETACHED_PROCESS`` is the same promise
        # spelt for that OS — the server gets no console, so it does not die
        # with the window the client was launched from — and
        # ``CREATE_NEW_PROCESS_GROUP`` keeps the console's Ctrl-C/Ctrl-Break
        # out of it, which is what ``start_new_session`` buys against SIGINT.
        # ``getattr``: the two flags exist only on nt, and naming them
        # unguarded would be an AttributeError at import on every other box.
        detach = {
            "creationflags": getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        }
    else:
        detach = {"start_new_session": True}

    log = open(log_path, "ab", buffering=0)
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            close_fds=True,
            **detach,
        )
    finally:
        log.close()


class _locked:
    """An exclusive lock on ``path``, held for the ``with`` body.

    Two spellings of ONE property — **the OS releases it when the holder
    dies** — which is the whole reason :func:`obtain` locks rather than
    creating an ``O_EXCL`` lockfile: a client killed mid-spawn must cost one
    retry, not a wedged directory every future client has to age out with a
    heuristic.

    POSIX takes ``fcntl.flock``, which blocks until the lock is free.
    Windows has no flock; ``msvcrt.locking`` with ``LK_LOCK`` takes a byte
    range on the file, is released when the handle closes (including when
    the process dies), and retries internally for about ten seconds before
    raising — so the retry loop here is what turns "ten seconds" into "as
    long as a cold server may take to import NumPy", which is the wait the
    lock exists to cover.
    """

    def __init__(self, path: str, timeout: float = SPAWN_TIMEOUT) -> None:
        self.path = path
        self.timeout = timeout
        self._fd = -1

    def __enter__(self) -> _locked:
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if os.name == "nt":
                self._take_nt()
            else:
                import fcntl

                fcntl.flock(self._fd, fcntl.LOCK_EX)
        except BaseException:
            os.close(self._fd)
            self._fd = -1
            raise
        return self

    def _take_nt(self) -> None:
        import msvcrt
        import time

        deadline = time.monotonic() + self.timeout
        while True:
            try:
                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_LOCK, 1)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise

    def __exit__(self, *_exc) -> None:
        fd, self._fd = self._fd, -1
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            # Closing the descriptor releases the lock either way; a failure
            # to unlock politely must not mask the body's own exception.
            pass
        finally:
            os.close(fd)


def obtain(path: str, server_command: list[str], log_path: str):
    """A connected socket, starting the server if nobody answers.

    The fast path takes no lock at all: a live server answers ``connect`` and
    that is the whole story. Only a failed connect enters the slow path, and
    there the lockfile serialises the decision so a crew firing 16 clients at
    once starts ONE server — the losers block on the lock, and by the time
    they hold it the winner's server is already listening, so their retry is
    the fast path again.

    A lock rather than an ``O_EXCL`` lockfile because the OS drops it when
    the holder dies (:class:`_locked` spells that for both platforms): a
    client killed mid-spawn costs one retry, not a wedged directory that
    every future client has to age out with a heuristic. An address file left
    behind by a server that died without unlinking — a socket with nothing
    behind it, a rendezvous naming a port nobody answers — is detected here
    (connect refuses, the path exists) and removed under the lock, which is
    the only place removing it is safe.
    """
    import time

    conn = connect(path)
    if conn is not None:
        return conn

    with _locked(f"{path}.lock"):
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


__all__ = [
    "RENDEZVOUS_HOST",
    "SPAWN_TIMEOUT",
    "SUN_PATH_MAX",
    "TCP",
    "TRANSPORT_ENV",
    "UNIX",
    "address_suffix",
    "connect",
    "digest",
    "dist_version",
    "filename_basis",
    "obtain",
    "publish_rendezvous",
    "read_rendezvous",
    "runtime_dir",
    "socket_path",
    "spawn_server",
    "transport",
]
