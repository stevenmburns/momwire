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


# The module a FROZEN bundle carries its version in (momwire#1277).
# `scripts/eznec_freeze/build.py` writes it at freeze time, into its own build
# directory and never into this tree, and hands it to PyInstaller; it is a
# top-level name because the readers below must stay import-light and may not
# import the `momwire` package to reach it.
#
# Why the metadata alone is not enough: `importlib.metadata` answers from
# whichever `momwire-*.dist-info` it finds first, and a user who extracts a new
# release zip OVER an old install keeps the old dist-info beside the new one.
# Windows lists directories alphabetically, so 0.66.0 answered for 0.68.0 code.
# The baked constant is a property of the code that was frozen, not of what
# else happens to sit in the folder.
BUILD_VERSION_MODULE = "_momwire_build_version"

_DIST_INFO_PREFIX = "momwire-"
_DIST_INFO_SUFFIX = ".dist-info"


def _frozen() -> bool:
    """True inside a PyInstaller bundle, and ONLY there.

    The baked module is read only when this holds, so a source checkout or a
    pip install can never be answered by a baked file — even one that strayed
    onto ``sys.path`` — and the metadata stays in charge there.
    """
    import sys

    return bool(getattr(sys, "frozen", False))


def _baked_version() -> str | None:
    """The version ``build.py`` froze into this bundle, or None."""
    if not _frozen():
        return None
    try:
        import importlib

        baked = importlib.import_module(BUILD_VERSION_MODULE).VERSION
    except Exception:  # noqa: BLE001 - a version probe must never be the thing that fails
        return None
    return baked if isinstance(baked, str) and baked.strip() else None


# The installed version, read WITHOUT importing `importlib.metadata`
# (momwire#1445). That import drags in `email`, `zipfile`, `pathlib` and
# `inspect` -- ~33 ms on Haswell and ~80 ms on the laptop, most of a thin
# client's whole warm call, paid on every launch because a client's life is
# one call. What it then does for one name is cheap: walk `sys.path` in order,
# take the first `momwire-*.dist-info` (or `.egg-info`) in each entry's listing
# order, and read the `Version` header of its METADATA (else PKG-INFO). The
# scan below is that search -- its own `FastPath`/`Lookup` rule, name test
# included -- spelt with `os.listdir` and a plain read.
#
# It answers only the PLAIN shape and refuses to guess at the rest: a zip or
# an `.egg` on the path, a finder on `sys.meta_path` that answers for
# distributions itself, a missing or unreadable METADATA, a folded or unusual
# header. Every one of those returns "not answered here", and the caller asks
# `importlib.metadata` exactly as it always did, so the worst case is today's
# speed, never a different answer. tests/test_eznec_engine_stamp.py holds the
# two readers equal on planted layouts.
_METADATA_FILES = ("METADATA", "PKG-INFO")
_METADATA_SUFFIXES = (".dist-info", ".egg-info")
# What `importlib.metadata`'s read_text swallows (it then tries the next
# file); anything else it lets escape, so the scan defers on anything else.
_METADATA_MISSING = (
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
    PermissionError,
)


class _NotPlain(Exception):
    """A layout the scan does not answer for; ``importlib.metadata`` does."""


def _metadata_dir(entry: str) -> str | None:
    """The first momwire metadata directory in one ``sys.path`` entry, or None.

    Raises :class:`_NotPlain` for an entry only the real reader can search.
    """
    if os.path.basename(entry).lower().endswith(".egg"):
        raise _NotPlain(entry)
    try:
        names = os.listdir(entry or ".")
    except OSError:
        # A missing directory is searched by nobody; a zip (or any file) on
        # the path is searched by the real reader, through `zipfile`.
        if os.path.isfile(entry):
            raise _NotPlain(entry) from None
        return None
    for name in names:
        low = name.lower()
        if (
            low.endswith(_METADATA_SUFFIXES)
            and low.rpartition(".")[0].partition("-")[0] == "momwire"
        ):
            return os.path.join(entry, name)
    return None


def _header_version(text: str) -> str:
    """The first ``Version`` header of a METADATA text, read the way the
    ``email`` parser's compat32 policy reads an unfolded one."""
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if not line.strip():
            break  # the headers ended
        key, sep, value = line.partition(":")
        if not sep or not key or key != key.strip():
            break  # not a plain header line: the parser's call, not ours
        if key.lower() == "version":
            following = lines[i + 1] if i + 1 < len(lines) else ""
            if following[:1] in (" ", "\t"):
                break  # folded
            return value.lstrip(" \t").rstrip("\r\n")
    raise _NotPlain("Version")


def _scan_metadata_version() -> str:
    """``importlib.metadata.version("momwire")`` on the plain shape.

    Raises :class:`_NotPlain` (or whatever a read raised) when it cannot
    answer with certainty, including when nothing is installed at all.
    """
    import sys

    import _frozen_importlib_external

    for finder in sys.meta_path:
        if finder is _frozen_importlib_external.PathFinder:
            continue
        if getattr(finder, "find_distributions", None) is not None:
            raise _NotPlain(repr(finder))
    for entry in sys.path:
        found = _metadata_dir(entry)
        if found is None:
            continue
        for filename in _METADATA_FILES:
            try:
                with open(os.path.join(found, filename), encoding="utf-8") as handle:
                    text = handle.read()
            except _METADATA_MISSING:
                continue
            if text:
                return _header_version(text)
        raise _NotPlain(found)
    raise _NotPlain("not installed")


def _metadata_version() -> str | None:
    """The installed distribution's version: the scan, else the real reader."""
    try:
        return _scan_metadata_version()
    except Exception:  # noqa: BLE001 - every miss defers to the reader below
        pass
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("momwire")
    except Exception:  # noqa: BLE001 - a version probe must never be the thing that fails
        # Every exception, not just PackageNotFoundError: a broken or
        # half-written dist-info raises from inside the metadata reader.
        return None


def momwire_version() -> str | None:
    """The full momwire version this process is running, or None.

    Resolution order (momwire#1277): the version baked into a frozen bundle
    FIRST, then the installed distribution's metadata.  The metadata is the
    whole answer on a source or pip install, where an editable install reports
    the version recorded at ``pip install -e`` time.  None, never an
    exception: a version probe must never be the thing that fails.

    The metadata is read by :func:`_metadata_version`, which answers the
    common install without importing ``importlib.metadata`` (momwire#1445).
    """
    baked = _baked_version()
    if baked is not None:
        return baked
    return _metadata_version()


def bundle_dist_versions(base: str | None = None) -> list[str]:
    """Every ``momwire-<version>.dist-info`` in a frozen bundle's runtime dir.

    ``base`` defaults to the bundle's ``_internal`` (``sys._MEIPASS``); outside
    a frozen bundle, with no ``base``, the answer is ``[]`` — a site-packages
    is not this function's to audit.  More than one entry is an OVERLAY: a
    release extracted over an older one, which leaves the old dist-info behind
    (momwire#1277).  Sorted, and empty on any filesystem error.
    """
    if base is None:
        if not _frozen():
            return []
        import sys

        base = getattr(sys, "_MEIPASS", None) or os.path.dirname(sys.executable)
    try:
        names = os.listdir(base)
    except OSError:
        return []
    found = []
    for name in names:
        folded = name.casefold()
        if folded.startswith(_DIST_INFO_PREFIX) and folded.endswith(_DIST_INFO_SUFFIX):
            found.append(name[len(_DIST_INFO_PREFIX) : -len(_DIST_INFO_SUFFIX)])
    return sorted(found)


def stale_versions(running: str | None, base: str | None = None) -> list[str]:
    """The bundle's dist-info versions that are NOT the one running.

    Empty on a clean bundle (one dist-info, this version's) and everywhere
    outside a frozen bundle.  Never refuses anything: an overlaid bundle runs
    the code that was extracted last, and this only lets it say so.
    """
    return [v for v in bundle_dist_versions(base) if v != running]


def dist_version() -> tuple[str, str]:
    """``(major, minor)`` of the running momwire, without importing it.

    :func:`momwire_version`'s answer — the frozen bundle's baked version first,
    the installed metadata otherwise — which is the reason no version probe
    ever has to spawn anything.  ``("0", "0")`` when neither answers.
    """
    full = momwire_version()
    parts = full.split(".")[:2] if full else []
    if len(parts) != 2:
        return "0", "0"
    return parts[0], parts[1]


# The interface modes every launcher serves beyond one deck per call:
# ``s`` streaming (``- -``), ``b`` batch (decks chained by ``NX``).
LAUNCHER_MODES = "sb"


def version_line(dialect: str, name: str, modes: str = LAUNCHER_MODES) -> str:
    """The ``-version`` line every launcher answers (momwire#1606).

    Four fields separated by spaces, the format agreed with AE6TY for
    SimNEC: the deck dialect (``NEC5``, ``NEC42`` or ``NEC2``), the
    launcher's own name (always starting ``momwire``), the interface modes in
    square brackets (``[sb]``; ``[]`` would mean one deck per call only), and
    the full momwire version, e.g. ``NEC5 momwire-nec5-razor-2p [sb] 0.82.0``.

    The dialect comes first for the SimNEC builds already deployed: they take
    the engine number from the first digit of the line (``Options.getEngine``,
    ``[a-zA-Z]*([0-9])+?(.*)`` over the whole line), and their NEC-2 and
    NEC-4.2 insulation checks need 2 and 4 there.  The version is the full
    one, read like :func:`dist_version` without importing momwire; ``0.0.0``
    when nothing answers, since a probe must never be the thing that fails.
    """
    return f"{dialect} {name} [{modes}] {momwire_version() or '0.0.0'}"


# momwire#1644: `momwire.deck._solver.JUNCTION_RULE_ENV`, read by the engine
# behind every launcher. A server inherits it from the client that spawned it,
# so it is identity: `junction_rule_identity()` goes into both clients' keys.
JUNCTION_RULE_ENV = "MOMWIRE_JUNCTION_RULE"


def junction_rule_identity() -> list[str]:
    """``["junction_rule=<value>"]`` when ``MOMWIRE_JUNCTION_RULE`` is set,
    else nothing -- so every key without the override is the key it was."""
    value = os.environ.get(JUNCTION_RULE_ENV)
    return [f"junction_rule={value}"] if value else []


def digest(parts: list[str]) -> str:
    """A short hex identity for a server configuration.

    Hashed rather than spelt out because the name has to fit in ``sun_path``
    with a directory in front of it, and because a filesystem path in a
    socket name would leak a user's layout into a world-listable directory.
    The caller decides what is identity — the rule from #379: an upgrade, a
    different venv, a different basis is a DIFFERENT engine and must not be
    served by whatever is still resident from before.

    SHA-256 from CPython's own built-in module where it has one, ahead of
    ``hashlib``: the digest is the same function either way, but ``hashlib``
    loads OpenSSL's ``_hashlib`` first, ~1.1 ms of every warm client launch
    on Haswell (momwire#1445).  ``_sha2`` is the module's name from 3.12,
    ``_sha256`` before; a build with neither gets ``hashlib``.
    """
    try:
        from _sha2 import sha256  # type: ignore[import-not-found]
    except ImportError:
        try:
            from _sha256 import sha256  # type: ignore[import-not-found]
        except ImportError:
            from hashlib import sha256

    return sha256("\0".join(parts).encode("utf-8")).hexdigest()[:16]


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


# `momwire.deck._solver.NEC4_MARKER` and `NEC42_MARKER`, restated: a client
# may not import momwire (momwire#1295, momwire#1443).
NEC4_MARKER = "nec4"
NEC42_MARKER = "nec42"


def filename_nec4(prog: str) -> tuple[bool, str | None]:
    """``(is_nec4, basis)`` for a command NAME — the stdlib copy of
    ``_solver.dialect_from_program_name`` and ``nec4_basis_from_program_name``
    (momwire#1295), whose docstrings are the rule.  ``is_nec4`` is whether the
    name carries ``nec4`` at all; the marker is ``nec42`` when the name
    carries it, else ``nec4`` (momwire#1443); ``basis`` is ``None`` for a name
    ending at the marker, the text after ``<marker>-`` for one that continues
    with a dash, and the whole continuation otherwise, so it reaches the
    refusal by name.
    """
    name = prog.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    if name.endswith(".exe"):
        name = name[:-4]
    if NEC4_MARKER not in name:
        return False, None
    marker = NEC42_MARKER if NEC42_MARKER in name else NEC4_MARKER
    rest = name.split(marker, 1)[1]
    if rest == "":
        return True, None
    return True, rest[1:] if rest.startswith("-") else rest


# --------------------------------------------------------------------------
# the 4nec2 front end (momwire#1488)
# --------------------------------------------------------------------------
#
# 4nec2 runs its engine as `ENGINE.exe < nec2d.tmp`: no arguments, and the
# two file names on stdin, one per line. The deck it names carries no EN card,
# and when its GE card asks (`GE 0 0 1`) the engine writes a `.GMO` geometry
# file beside the printout. Everything below is shared by the engine's own
# shell and the Python thin client, and spelt again in C by the native
# launcher, which cannot import it; the tests hold the copies together.

# The first line of a deck sent to the resident engine by a launcher that
# took its file names from stdin: the deck may end without EN. A deck never
# starts with NUL, so the line cannot be mistaken for a card.
EN_OPTIONAL_REQUEST = b"\x00EN-OPTIONAL\n"

# An answer that carries a `.GMO` file starts with GMO_FRAME_START, then the
# file's bytes, then GMO_FRAME_END, then the printout's bytes. A printout never
# starts with NUL and a `.GMO` file holds digits, signs, points, `E`, spaces
# and CRLF only, so neither marker can occur inside what it delimits. An
# answer without the frame is the printout alone, exactly as before.
GMO_FRAME_START = b"\x00GMO\r\n"
GMO_FRAME_END = b"\x00\r\n"
GMO_SUFFIX = ".GMO"

# MOMWIRE_TIMES=1: per-stage timing, asked for by the user. A launcher that
# has it set sends TIMES_REQUEST ahead of everything else in the request, and
# the engine answers with TIMES_FRAME_START, its `times:` line, TIMES_FRAME_END
# ahead of everything else in the answer (the `.GMO` frame and the printout
# follow as before). A request without the line is answered exactly as before.
TIMES_REQUEST = b"\x00TIMES\n"
TIMES_FRAME_START = b"\x00TIMES\r\n"
TIMES_FRAME_END = b"\x00\r\n"
TIMES_ENV = "MOMWIRE_TIMES"

# The `times:` line's cells, in order: each `<name>_ms`, milliseconds.
TIMES_CELLS = (
    "start",
    "load",
    "seam",
    "ground",
    "fill",
    "factor",
    "rp",
    "near",
    "run",
    "printout",
    "write",
)


def times_requested(environ=None) -> bool:
    """``MOMWIRE_TIMES`` set to anything but ``""`` / 0 / off / false / no."""
    env = os.environ if environ is None else environ
    value = env.get(TIMES_ENV, "").strip().lower()
    return value not in ("", "0", "off", "false", "no")


def times_line(
    route: str,
    cells: dict,
    *,
    total_ms: float,
    threads: int,
    runs: int,
    counts: dict,
) -> str:
    """The ``times:`` line every momwire NEC-5 engine writes on stderr under
    MOMWIRE_TIMES, the standalone executables' (``_mw_times.h``) to the key:
    ``route``, ``threads`` and ``runs``, then every ``<cell>_ms`` of
    :data:`TIMES_CELLS` (0 where a route has no such stage), ``other_ms`` (what
    no stage claimed) and ``total_ms`` -- so the cells and ``other_ms`` add up
    to ``total_ms`` -- and how the ground tables were obtained:
    ``ground_built``, ``ground_disk``, ``ground_blended``."""
    parts = [f"times: route={route} threads={int(threads)} runs={int(runs)}"]
    claimed = 0.0
    for name in TIMES_CELLS:
        value = float(cells.get(name, 0.0))
        claimed += value
        parts.append(f"{name}_ms={value:.3f}")
    parts.append(f"other_ms={max(total_ms - claimed, 0.0):.3f}")
    parts.append(f"total_ms={total_ms:.3f}")
    for name in ("built", "disk", "blended"):
        parts.append(f"ground_{name}={int(counts.get(name, 0))}")
    return " ".join(parts)


def parse_times_line(line: str) -> dict:
    """A ``times:`` line's cells: ``route`` a string, every other value a
    number (``*_ms`` floats, the counts ints)."""
    head, _, rest = line.strip().partition(" ")
    if head != "times:":
        raise ValueError(f"not a times line: {line!r}")
    out: dict = {}
    for item in rest.split():
        key, _, value = item.partition("=")
        if key == "route":
            out[key] = value
        elif key.endswith("_ms"):
            out[key] = float(value)
        else:
            out[key] = int(value)
    return out


def split_times(answer: bytes) -> tuple[bytes | None, bytes]:
    """``(times line, rest)`` from a resident engine's answer bytes; ``None``
    when the answer carries no times frame (as :func:`split_answer`)."""
    if not answer.startswith(TIMES_FRAME_START):
        return None, answer
    end = answer.find(TIMES_FRAME_END, len(TIMES_FRAME_START))
    if end < 0:
        raise ValueError("the resident engine's times frame never closed")
    return answer[len(TIMES_FRAME_START) : end], answer[end + len(TIMES_FRAME_END) :]


def split_answer(answer: bytes) -> tuple[bytes | None, bytes]:
    """``(gmo, printout)`` from a resident engine's answer bytes.

    ``gmo`` is ``None`` when the answer carries no frame. A frame that never
    closes is not an answer this engine writes; it raises ``ValueError`` so
    the launcher's ladder takes the one-shot rung rather than writing it.
    """
    if not answer.startswith(GMO_FRAME_START):
        return None, answer
    end = answer.find(GMO_FRAME_END, len(GMO_FRAME_START))
    if end < 0:
        raise ValueError("the resident engine's .GMO frame never closed")
    gmo = answer[len(GMO_FRAME_START) : end]
    return gmo, answer[end + len(GMO_FRAME_END) :]


def gmo_path(printout_path: str) -> str:
    """``<printout stem>.GMO``: the printout path with its extension replaced.

    The extension is :func:`os.path.splitext`'s: a dot in a directory name is
    not one, a leading dot (``.out``) names a file rather than an extension,
    and a printout with no extension gets ``.GMO`` appended. Separators are
    the OS's own: on Windows both ``\\`` and ``/``, elsewhere ``/`` only.
    """
    stem, _ = os.path.splitext(printout_path)
    return stem + GMO_SUFFIX


def _decode_path(raw: bytes) -> str:
    """A file name as the front end wrote it: the ANSI code page on Windows
    (4nec2 is a Windows program), the filesystem encoding elsewhere."""
    if os.name == "nt":
        return raw.decode("mbcs", errors="replace")
    return os.fsdecode(raw)


def strip_path_line(line: str) -> str:
    """One stdin line as a path: surrounding whitespace (CR and LF included)
    stripped, then ONE pair of surrounding double quotes, as Ward AE6TY's
    port reads them (a ``.tmp`` line ``"..\\out\\x.inp"`` is the path
    ``..\\out\\x.inp``).  Spaces inside the quotes are the path's own.  A
    lone or mismatched quote is left in place, for the OS to refuse as the
    name it is."""
    line = line.strip()
    if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
        line = line[1:-1]
    return line


# The third character of a card's first token when the mnemonic is glued to
# its first field (``GE1``): `momwire.deck._cards._FUSED_FIELD_START`, restated
# because this module may not import momwire.
_FUSED_FIELD_START = "0123456789.+-"


def is_names_line(line: str) -> bool:
    """Whether the first line on a NEC-2 launcher's stdin is 4nec2's first
    file name rather than the first card of a deck (momwire#1611).

    A NEC-2 launcher reads stdin both ways with no arguments: SimNEC writes
    ``NX``-chained decks there, and 4nec2 writes the two file names.  A card
    line is one whose first token is a two-letter mnemonic, alone or glued to
    a number (``GE1``), which is the rule ``momwire.deck.parse_card`` reads a
    mnemonic by; a path is anything else, and so is a line naming a file that
    exists, whatever it looks like.  A blank line is a deck's (a stdin of
    names never starts with one).
    """
    stripped = line.strip()
    if not stripped:
        return False
    head = stripped.replace(",", " ").split()[0]
    card = head[:2].isalpha() and (len(head) == 2 or head[2] in _FUSED_FIELD_START)
    if not card:
        return True
    try:
        return os.path.isfile(strip_path_line(line))
    except (OSError, ValueError):
        return False


def stdin_paths(stream=None) -> list[str] | None:
    """The file names 4nec2 sends on stdin: the first two lines, each read by
    :func:`strip_path_line`.

    ``None`` when there is no stdin to read, or it is a terminal: a person
    who types the bare command gets the argument error at once rather than an
    engine silently waiting on the keyboard. The list is shorter than two
    when stdin ends early, and a line that strips to nothing ends it, so the
    caller's argument errors name which name was missing.
    """
    import sys

    stream = sys.stdin if stream is None else stream
    if stream is None:
        return None
    try:
        if stream.isatty():
            return None
        source = getattr(stream, "buffer", stream)
        paths: list[str] = []
        while len(paths) < 2:
            line = source.readline()
            if isinstance(line, bytes):
                line = _decode_path(line)
            line = strip_path_line(line)
            if not line:
                break
            paths.append(line)
        return paths
    except (OSError, ValueError):
        # Closed, or unreadable (pytest's captured stdin raises OSError):
        # nothing arrived.
        return []


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

    ``tempfile`` is imported only on the fallbacks that read it: it drags in
    ``shutil``, ``random`` and friends, ~5 ms of every warm client launch
    that had a runtime directory all along (momwire#1445).
    """
    base = os.environ.get("MOMWIRE_PORTAL_RUNTIME_DIR")
    if not base:
        if os.name == "nt":
            local = os.environ.get("LOCALAPPDATA")
            if local and os.path.isdir(local):
                base = os.path.join(local, "momwire-portal")
            else:
                import getpass
                import tempfile

                base = os.path.join(
                    tempfile.gettempdir(), f"momwire-portal-{getpass.getuser()}"
                )
        else:
            xdg = os.environ.get("XDG_RUNTIME_DIR")
            if xdg and os.path.isdir(xdg):
                base = os.path.join(xdg, "momwire-portal")
            else:
                import tempfile

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


# --- AVX-512 off ---------------------------------------------------------------
#
# momwire's results must not depend on whether the CPU has AVX-512. Two
# libraries momwire runs on pick AVX-512 code at run time on a CPU that has
# it, and both move last bits:
#
# * numpy's ufuncs: exp, log, tan, the inverse trig and hyperbolic functions,
#   log1p / expm1 / log2 / log10 / exp2 and cbrt differ from libm under
#   AVX-512 (sin, cos, sqrt, hypot and the complex functions do not);
# * OpenBLAS's DYNAMIC_ARCH builds (numpy's and SciPy's bundled libraries)
#   select their SkylakeX kernels, which round an LU differently from the
#   Haswell kernels every AVX2 CPU without AVX-512 runs.
#
# AVX-512 buys momwire nothing today (its own builds stop at avx2, and the
# LU measured no gain), so on an AVX-512 CPU the entry points momwire owns
# turn both off before anything imports numpy: NPY_DISABLE_CPU_FEATURES names
# numpy's AVX-512 dispatch targets, and OPENBLAS_CORETYPE=Haswell holds every
# OpenBLAS in the process to the kernels an AVX2 CPU runs. On a CPU without
# AVX-512 nothing is set, so nothing there can move; and a value the user
# exported is never replaced (an empty NPY_DISABLE_CPU_FEATURES, or a named
# OPENBLAS_CORETYPE, is how to ask for the AVX-512 paths back). Only the
# process's environment is touched: a library user who imports numpy before
# momwire has already chosen, and for them the code-level spellings (libm's
# pow, float_power, ...) remain the guarantee.

NPY_DISABLE_ENV = "NPY_DISABLE_CPU_FEATURES"
NPY_ENABLE_ENV = "NPY_ENABLE_CPU_FEATURES"
OPENBLAS_CORETYPE_ENV = "OPENBLAS_CORETYPE"
OPENBLAS_CORETYPE = "Haswell"

# numpy's AVX-512 dispatch targets, in the two vocabularies its releases have
# used. A name the installed numpy does not dispatch earns an ImportWarning
# (ignored by default, an error under `-W error`), so the list is chosen by
# version: numpy 2.4 regrouped x86 into microarchitecture levels (X86_V4 is
# AVX-512's), 2.3 added AVX512_SPR to the per-feature list, and anything older
# dispatches the per-feature names without it.
_NPY_AVX512_V24 = ("X86_V4", "AVX512_ICL", "AVX512_SPR")
_NPY_AVX512_V23 = (
    "AVX512F",
    "AVX512CD",
    "AVX512_KNL",
    "AVX512_KNM",
    "AVX512_SKX",
    "AVX512_CLX",
    "AVX512_CNL",
    "AVX512_ICL",
    "AVX512_SPR",
)
_NPY_AVX512_OLD = _NPY_AVX512_V23[:-1]


def cpu_has_avx512() -> bool:
    """Whether this CPU (and its OS) runs AVX-512F code, asked without numpy.

    Linux reads the kernel's flags, Windows asks ``IsProcessorFeaturePresent``
    (``PF_AVX512F_INSTRUCTIONS_AVAILABLE``, 41), macOS ``hw.optional.avx512f``.
    Anything else, or any failure to ask, is False: the answer only ever adds
    a pin, so "don't know" must mean "leave the process alone".
    """
    import sys

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="ascii", errors="replace") as handle:
                for line in handle:
                    if line.startswith("flags"):
                        return "avx512f" in line.split()
        except OSError:
            return False
        return False
    if os.name == "nt":
        try:
            import ctypes

            return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(41))
        except (AttributeError, OSError):
            return False
    if sys.platform == "darwin":
        try:
            import ctypes

            value = ctypes.c_int(0)
            size = ctypes.c_size_t(ctypes.sizeof(value))
            libc = ctypes.CDLL(None)
            failed = libc.sysctlbyname(
                b"hw.optional.avx512f", ctypes.byref(value), ctypes.byref(size), None, 0
            )
        except (AttributeError, OSError):
            return False
        return failed == 0 and bool(value.value)
    return False


def _numpy_version() -> str | None:
    """The version of the numpy an ``import numpy`` would load, from the
    ``numpy-<version>.dist-info`` beside it, or None when that cannot be
    told. Neither imported (it would read the very variables this is about
    to set) nor asked of ``importlib.metadata``, whose import alone costs a
    thin client ~25 ms: this is a directory listing, in import order."""
    import sys

    for entry in sys.path:
        try:
            names = os.listdir(entry or ".")
        except OSError:
            continue
        if "numpy" not in names:
            continue
        for name in names:
            if name.startswith("numpy-") and name.endswith(".dist-info"):
                return name[len("numpy-") : -len(".dist-info")]
        return None
    return None


def numpy_avx512_features(numpy_version: str | None) -> str:
    """The NPY_DISABLE_CPU_FEATURES value naming every AVX-512 target the
    given numpy release dispatches; both vocabularies when the version is
    unknown (a frozen bundle without metadata), which costs only the ignored
    ImportWarning for the names that release does not know."""
    parts = []
    for piece in (numpy_version or "").split(".")[:2]:
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    if len(parts) < 2:
        names = _NPY_AVX512_V24 + _NPY_AVX512_V23[:-2]
    elif tuple(parts) >= (2, 4):
        names = _NPY_AVX512_V24
    elif tuple(parts) >= (2, 3):
        names = _NPY_AVX512_V23
    else:
        names = _NPY_AVX512_OLD
    return " ".join(names)


def cpu_env(environ=None, *, avx512: bool | None = None) -> dict[str, str]:
    """The variables :func:`pin_cpu_env` would add to ``environ``: none on a
    CPU without AVX-512, and never one ``environ`` already has.

    numpy refuses to start when both NPY_DISABLE_CPU_FEATURES and
    NPY_ENABLE_CPU_FEATURES are set, so a user's ENABLE list is left to
    speak for itself.
    """
    environ = os.environ if environ is None else environ
    if not (cpu_has_avx512() if avx512 is None else avx512):
        return {}
    added = {}
    if NPY_DISABLE_ENV not in environ and NPY_ENABLE_ENV not in environ:
        added[NPY_DISABLE_ENV] = numpy_avx512_features(_numpy_version())
    if OPENBLAS_CORETYPE_ENV not in environ:
        added[OPENBLAS_CORETYPE_ENV] = OPENBLAS_CORETYPE
    return added


def pin_cpu_env(environ=None) -> dict[str, str]:
    """Turn AVX-512 off for numpy and OpenBLAS (see the block above), in
    ``environ`` (default ``os.environ``, so the process and every child it
    starts read it). Returns what was added. Call it before anything imports
    numpy; afterwards it reaches only child processes."""
    environ = os.environ if environ is None else environ
    added = cpu_env(environ)
    environ.update(added)
    return added


__all__ = [
    "BUILD_VERSION_MODULE",
    "LAUNCHER_MODES",
    "NPY_DISABLE_ENV",
    "NPY_ENABLE_ENV",
    "OPENBLAS_CORETYPE",
    "OPENBLAS_CORETYPE_ENV",
    "RENDEZVOUS_HOST",
    "SPAWN_TIMEOUT",
    "SUN_PATH_MAX",
    "TCP",
    "TRANSPORT_ENV",
    "UNIX",
    "address_suffix",
    "bundle_dist_versions",
    "connect",
    "cpu_env",
    "cpu_has_avx512",
    "digest",
    "dist_version",
    "filename_basis",
    "momwire_version",
    "numpy_avx512_features",
    "obtain",
    "pin_cpu_env",
    "publish_rendezvous",
    "read_rendezvous",
    "runtime_dir",
    "socket_path",
    "spawn_server",
    "stale_versions",
    "transport",
    "version_line",
]
