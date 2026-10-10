"""Smoke-gate the packaged EZNEC drop-in against the unfrozen module.

Usage::

    python scripts/eznec_freeze/smoke.py dist/momwire-eznec/momwire-eznec[.exe]

The argument is a LAUNCHER — the native client EZNEC points at — and every
gate below runs the bundle end to end: launcher, spawned engine, printout.

Ten gates, derived from the seam's own contract (momwire#497 U1):

1. **Byte identity** — on decks that serve, the bundle's printout must
   equal ``python -m momwire.eznec``'s byte for byte (the printout carries no
   wall-clock, so packaging may not change a single byte, CRLF included).
2. **The refusal frame travels** — a deck the seam refuses must still produce
   a printout carrying the ``NEC ERROR`` line, exit 0 (the file is the only
   channel EZNEC reads).
3. **Launch cost, informational** — per-launch wall time is printed but not
   gated; EZNEC launches once per frequency point, so this number is the
   sweep economics (real engine baseline 18–37 ms; the frozen one-shot
   measured ~1.3 s on the sitting-4 box, which is what the launcher's warm
   engine exists to beat).
4. **Every shipped variant is PRESENT and answers, in the basis its NAME
   claims** (momwire#593).  The basis rides on the filename, so this is the
   gate that makes the bundle's shape a fact rather than an intention.
5. **The engine is beside the launcher** — a bundle whose
   ``momwire-eznec-engine`` is missing is a launcher with nothing to spawn,
   and every other gate would still pass through the rung-4 refusal or not
   at all.
6. **Residency** — the launcher really goes resident: two launches, ONE
   spawned daemon, both printouts still the module's.  This is the gate the
   phase-3 flip exists for, and it is Windows's only verdict on the native
   client, because `tests/test_eznec_client_c.py` proves the same shape with
   sh-script engine shims that cannot run there.  Nothing in a printout can
   reveal which rung produced it — rung 3 makes a byte-identical one by
   construction — so this counts the daemon's own ``listening pid=`` lines:
   none means the fallback ladder carried the run silently, which is a
   FAILURE of this gate rather than a pass, and two means a double spawn.
   Since momwire#1418 it also reads the pool's ``worker pid=... ready`` line:
   the daemon solves in worker processes, a frozen worker is the bundle's own
   exe started with ``--serve-worker``, and a pool that cannot start one
   answers in-process with perfect printouts -- so only the log can tell.
7. **Self-containment of the accelerator** (nt only) — the bundle carries
   ``libomp140.x86_64.dll`` AND the daemon loaded the bundle's own copy of it.
   momwire#737 shipped two phases of bundle without that DLL, so every
   deployed solve was pure Python (15.6x at 201 segments, 20.8x at 401).

   The runner-vs-field asymmetry is the whole reason this gate has to name a
   PATH.  windows-latest carries the runtime in ``System32`` (and a Visual
   Studio toolchain carries more copies), and ``System32`` is on the loader's
   search path unconditionally — no environment strips it — so on CI the
   import is rescued and *every* observable short of the source says the
   bundle is healthy: no warning, accelerators loaded, correct answers, full
   speed.  The field box has no such copy, which is why a user met the
   20x-slow bundle CI had signed off eight times.  So this gate reads the
   daemon's ``openmp runtime:`` line and requires the path to resolve INSIDE
   the bundle directory; a path under ``System32`` or a toolchain tree is a
   named failure, not a pass.  The ``PATH`` strip stays as a belt (it does
   remove toolchain directories, and the launcher spawns the engine with the
   inherited environment so the daemon inherits it) — but the source check is
   what makes the gate able to go RED on the runner.

   POSIX skips it and says why: the Linux/macOS wheels link the SYSTEM OpenMP
   runtime on purpose (``_accel.py`` documents the shared-runtime reason), so
   self-containment is not the claim there and PyInstaller collects the
   shared libraries itself.
8. **The printout says which engine answered** — line 2 of every printout
   carries ``momwire <version> <basis> <variant>``, with a real version and
   the basis the launcher's own filename claims.  The version is the field
   this gate exists for: it comes from the package metadata, which a
   PyInstaller bundle carries only because ``build.py`` passes
   ``--copy-metadata momwire``, and a bundle built without it stamps
   ``unknown`` while every other gate here stays green.  Since momwire#1277
   the version is BAKED into the bundle at freeze time and the metadata is
   only the fallback, so the gate requires the build's own version exactly,
   and no overlay marker on a clean bundle.

   **The overlay case** (momwire#1277): a release extracted over an older
   one keeps the older ``momwire-*.dist-info`` in ``_internal``, and the
   metadata lookup answered from it.  So the gate plants a valid
   ``momwire-0.0.1.dist-info`` there — it sorts first, which is the field
   shape — runs the launcher once more in a fresh runtime room, and requires
   the stamp to STILL say the build's version and to end in
   ``(stale: 0.0.1)``.  The planted folder is removed afterwards whatever
   happens.

   It is a gate on what the printout SAYS, never on what the engine DID.
   The stamp is threaded from the filename, so a copy that ignored its own
   name would stamp the name it ignored — gate 4 is the only evidence about
   the solver, and this one can never stand in for it.
9. **The NEC-5 names** (momwire#1239, primary since momwire#1295 phase 4) —
   ``build.py``'s ``NEC5_LAUNCHERS`` are present, answer ``-version`` with
   exit 0 and ``NEC5momwire.<maj>.<min>`` (SimNEC's configure probe), and
   serve in the basis read past ``nec5-``.  One family for both hosts: SimNEC
   picks the deck syntax from ``nec5`` in the path, which the deprecated
   ``momwire-eznec`` names do not carry, and EZNEC reads no name at all.
10. **The NEC-4.2 names** (momwire#1295) — ``build.py``'s ``NEC4_LAUNCHERS``
   are present and serve a deck EZNEC's External NEC-4.2 slot actually wrote
   (``tests/fixtures/eznec_nec42/``): byte-identical to the module run in the
   nec4 dialect and the basis read past ``nec4-`` (but for the three
   wall-clock values a NEC-4.2 printout carries, masked), under the NEC-4.2 banner
   (never NEC-5's), a SOLVE and not a refusal, stamped with that basis, and
   with a feed-point impedance within 5 % of the licensed NEC-4.2's own
   printout of the same deck (the sanity bar
   ``tests/test_eznec_nec42_1295.py`` holds the module to).  The 5 % is a bar
   on "this is an answer about this antenna", not on accuracy: the byte
   identity is the gate on the bundle, the impedance the gate on the deck
   having reached the right dialect at all.

Gate 4 exists because momwire#628 was exactly that bug on the other route:
a copy named for one engine served another, and the printout was internally
CONSISTENT because the banner names whatever actually ran.  Nothing in a
printout can reveal it — line 2's stamp included, which names the basis the
process was TOLD and not the one that solved — so it has to be caught here,
by comparing each exe against the module RUN IN THE BASIS THE NAME ASKS FOR.

That comparison has two blind spots, and gate 4 closes both rather than
trusting it alone:

* **A refusal is byte-equal too.**  Ask a sinusoidal-named copy and both
  sides print the same ``NEC ERROR``; byte identity then proves the filename
  was honoured and NOTHING about serving.  So every variant must additionally
  come back a SOLVE.
* **A build that shipped no variant at all passes vacuously**, because a
  glob loop over zero copies runs zero comparisons.  So the set found beside
  the exe is checked against ``build.py``'s own ``SHIPPED_VARIANTS`` — the
  list that made them — and a missing one is a named failure.

The anti-coincidence check (a variant must differ from the DEFAULT's answer,
so a wrong engine behind a right filename cannot pass by accident) is scoped
to the bases that actually differ on the probe deck.  Several bases render
0010 identically — bspline, hmatrix and arrayblock are one answer here — and
requiring those to differ failed a copy that was serving exactly its name.
"""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "eznec" / "decks"
NEC4_FIXTURES = REPO / "tests" / "fixtures" / "eznec_nec42"

# The one frozen program in the bundle — `build.py`'s ENGINE_NAME and the
# client's ENGINE_NAME, restated here rather than imported because gate 5 is
# about a FILE being there and a name read out of the thing being gated would
# certify itself.
ENGINE_STEM = "momwire-eznec-engine"

# LLVM's OpenMP runtime, which both Windows extensions link because setup.py
# builds them with `/openmp:llvm`.  Restated here for gate 5's reason: read out
# of `build.py`, a build that bundled the wrong file would certify its own
# spelling and this gate would prove nothing.
OPENMP_DLL = "libomp140.x86_64.dll"


def _build_module():
    """``build.py`` itself, read from the file whose copy loop makes the
    launchers.

    Imported by path because ``scripts/`` is not a package, and imported at
    all rather than restated because a second list is what regresses: the
    copy loop and the presence gates have to be the same fact, or the gate
    certifies the shape it was told about instead of the one that shipped.
    """
    spec = importlib.util.spec_from_file_location(
        "eznec_freeze_build", HERE / "build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shipped_variants() -> tuple[str, ...]:
    """``build.py``'s deprecated ``momwire-eznec-<basis>`` copies."""
    return _build_module().SHIPPED_VARIANTS


def _nec5_launchers() -> tuple[str, ...]:
    """``build.py``'s NEC-5 names (EZNEC's NEC-5 slot and SimNEC's)."""
    return tuple(_build_module().NEC5_LAUNCHERS)


def _nec4_launchers() -> tuple[str, ...]:
    """``build.py``'s NEC-4.2 names (EZNEC's External NEC-4.2 slot)."""
    return tuple(_build_module().NEC4_LAUNCHERS)


# Two serving decks spanning the seam's range — a bare-wire rung-1 model and
# a network-heavy feed system — plus the standing refusal (NE over GN 0).
SERVE_IDS = ("0010_dipole-in-free-space", "0000_cardioid-l-network-feed")
REFUSE_ID = "0022_vertical-over-real-ground"

# Gate 4's deck.  0010 is a free-space dipole every basis hosts and on which
# the two SHIPPED bases disagree — bspline answers 85.073+45.369j where
# razor-nec5 answers 79.948+29.919j, the licensed engine's own number.  A
# razor-nec5 exe that ignored its filename and served the default would match
# the wrong column by ~16 ohm and be caught.
#
# Deliberately the same deck as ``SERVE_IDS[0]`` is NOT relied on: gate 4
# renders its own default-basis reference below rather than reading gate 1's
# output file, so moving either list cannot silently disarm the comparison.
BASIS_DECK = "0010_dipole-in-free-space"

# Gate 8's field reader.  Line 2 positionally, never by searching for the
# word: a stamp that went missing and a stamp that landed on another line are
# both failures, and a content search would call the second one a pass.
STAMP_PREFIX = " momwire "
STAMP_LINE = 1
STAMP_FIELDS = 4
NO_VERSION = "unknown"

# The overlay marker the stamp appends in a mixed bundle (momwire#1277), and
# the stale release the overlay case plants.  Restated rather than imported
# from `_printout`, for gate 5's reason: a spelling read out of the thing
# being gated would certify itself.
STALE_MARKER = "(stale: {})"
PLANTED_STALE = "0.0.1"

# What a printout looks like when it is an ANSWER rather than a refusal.
# Both directions are needed: the refusal frame is what the seam prints when
# a basis cannot host the deck, and its absence alone would also be satisfied
# by a truncated file.
SOLVED = "ANTENNA INPUT PARAMETERS"
REFUSED = "NEC ERROR"

# Gate 10's deck: a free-space dipole EZNEC's External NEC-4.2 slot wrote
# (capture 0228), served by every centre-feeding basis, and one on which
# bspline and point-matched sinusoidal answer differently — so a sinusoidal
# launcher that served the default would fail the byte comparison.  Its
# licensed NEC-4.2 printout sits beside it and is the impedance reference.
NEC4_DECK = "0228_dipole-10m-free"
NEC4_BANNER = "NUMERICAL ELECTROMAGNETICS CODE (NEC-4.2)"
NEC5_BANNER = "NUMERICAL ELECTROMAGNETICS CODE (NEC-5)"
NEC4_SANITY_REL = 0.05

# One cell of the ANTENNA INPUT PARAMETERS row.  By number, not by
# whitespace: a negative E12.5 cell fills its field and runs into its
# neighbour (``9.29044E-03-5.22804E-03``).  The same reading
# `tests/test_eznec_nec42_1295.py` makes, restated for gate 5's reason.
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:E[-+]\d+)?")

# The NEC-4.2 printout carries WALL-CLOCK fields, as the licensed engine's
# does — MATRIX TIMING's FILL and FACTOR, and the closing RUN TIME — where the
# NEC-5 one prints none (gate 1's byte identity rests on that).  So gate 10
# compares with those three values masked, and only those: the label, its
# padding and every other byte still have to match.
_WALL_CLOCK = re.compile(rb"(FILL=|FACTOR=|RUN TIME =)( *)\d+\.\d+")


def mask_wall_clock(printout: bytes) -> bytes:
    """``printout`` with its timing VALUES replaced, labels kept."""
    return _WALL_CLOCK.sub(rb"\1\2<t>", printout)


def feed_impedance(printout: str) -> complex | None:
    """The first feed row's impedance under ANTENNA INPUT PARAMETERS, or None.

    The row is the first one after the heading carrying exactly the eleven
    numbers a feed row has (tag, segment, then V, I, Z, Y as re/im pairs and
    the power); Z is the seventh and eighth.
    """
    lines = printout.splitlines()
    start = next((i for i, ln in enumerate(lines) if SOLVED in ln), None)
    if start is None:
        return None
    for line in lines[start + 1 :]:
        cells = _NUMBER.findall(line)
        if len(cells) == 11:
            return complex(float(cells[6]), float(cells[7]))
    return None


def run(cmd: list[str], out: Path, env: dict[str, str] | None = None) -> float:
    started = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, timeout=300, env=env)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise SystemExit(f"exit {result.returncode} from {cmd}: {result.stderr!r}")
    if not out.is_file():
        raise SystemExit(f"no printout written by {cmd}")
    return elapsed


def log_lines(room: Path, marker: str) -> list[str]:
    """Every line carrying ``marker`` that a daemon in this room ever wrote."""
    return [
        line
        for log in sorted(room.glob("*.log"))
        for line in log.read_text(errors="replace").splitlines()
        if marker in line
    ]


def listening(room: Path) -> list[str]:
    """Every ``listening pid=`` line a daemon in this room ever wrote.

    One line per spawned daemon, so this counts SPAWNS across the whole smoke
    run — which is what tells a warm reuse from a second server, and either of
    those from the fallback ladder having quietly done all the work.
    """
    return log_lines(room, "listening pid=")


def stop(room: Path) -> None:
    """Ask every daemon this room started to go, by pid out of its own log.

    The idle timeout would retire them in fifteen minutes anyway; this is so a
    CI runner (or a developer's box) is not left hosting a 117 MB engine for
    that quarter hour after a two-minute smoke.  SIGTERM is TerminateProcess
    on Windows, which is the right blunt instrument for a process whose only
    state is a socket.
    """
    for line in listening(room):
        for word in line.split():
            if word.startswith("pid="):
                try:
                    os.kill(int(word.split("=", 1)[1]), signal.SIGTERM)
                except (OSError, ValueError):
                    pass
    shutil.rmtree(room, ignore_errors=True)


def _dump_room(room: Path) -> None:
    """The runtime room's contents, printed for a red gate 6.

    The room is private and removed in ``main``'s finally, so on CI this
    print is the only surviving evidence.  It tells the two failures apart:
    an EMPTY room means the daemon was never launched at all (the spawn
    itself failed, client-side), while a ``.log`` with a traceback is a
    daemon that launched and died saying why.
    """
    entries = sorted(room.iterdir()) if room.is_dir() else []
    print(f"     runtime room: {[e.name for e in entries] or 'EMPTY'}")
    for log in entries:
        if log.suffix != ".log":
            continue
        text = log.read_text(errors="replace")
        print(f"     -- {log.name} ({len(text)} chars) --")
        for line in text.splitlines()[-40:]:
            print(f"     {line}")


def _system_only_path() -> str:
    """``PATH`` cut back to Windows' own directories.

    Built from ``%SystemRoot%`` rather than spelt ``C:\\Windows``: a runner
    whose system drive is not C: would otherwise get an empty PATH and fail
    this gate for the wrong reason.  Everything else goes — a Visual Studio
    toolchain's copy of the OpenMP runtime is exactly what must not be able to
    rescue the import.
    """
    root = os.environ.get("SystemRoot") or r"C:\Windows"
    return os.pathsep.join([str(Path(root) / "System32"), root])


def _gate_self_contained(exe: Path, work: Path, room: Path) -> int:
    """Gate 7 — the accelerator is live from the BUNDLE'S OWN files (nt only).

    Its own room, so gate 6's spawn count stays exact: this launch starts a
    second daemon (a different environment, and one it must start itself for
    the log to exist at all).
    """
    if os.name != "nt":
        print(
            "skip gate 7: self-containment is an nt claim — the Linux/macOS "
            "wheels link the SYSTEM OpenMP runtime on purpose"
        )
        return 0

    # Imported, not restated: the daemon writes these markers and this gate
    # reads them, and two spellings of one fact is how a gate ends up
    # certifying nothing.  Local because the module import costs a NumPy load
    # that the POSIX skip above has no reason to pay.
    from momwire.eznec._resident import ACCEL_OK, OMP_LINE

    failures = 0
    bundle = exe.parent
    # Both places a `--add-binary <dll>;.` can land in a one-dir build:
    # PyInstaller 6 puts collected binaries in `_internal/`, and asserting on
    # the layout rather than on one guess survives a PyInstaller that moves it.
    found = sorted({*bundle.glob(OPENMP_DLL), *bundle.glob(f"_internal/{OPENMP_DLL}")})
    if not found:
        print(
            f"FAIL {OPENMP_DLL}: not in the bundle — the engine is not self-contained"
        )
        failures += 1
    else:
        where = ", ".join(str(p.relative_to(bundle)) for p in found)
        print(f"ok   {OPENMP_DLL}: in the bundle ({where})")

    deck = FIXTURES / f"{SERVE_IDS[0]}.nec"
    out = work / f"{SERVE_IDS[0]}.stripped-path.out"
    env = {
        **os.environ,
        "MOMWIRE_PORTAL_RUNTIME_DIR": str(room),
        "PATH": _system_only_path(),
    }
    run([str(exe), str(deck), str(out)], out, env=env)

    status = log_lines(room, "accelerators:")
    if not listening(room):
        # No daemon, no log, no verdict: the ladder answered and this gate
        # would otherwise pass by having measured nothing.
        print(f"FAIL {exe.name}: no daemon spawned under a system-only PATH")
        _dump_room(room)
        failures += 1
    elif not any(ACCEL_OK in line for line in status):
        print(
            f"FAIL {exe.name}: the engine fell back to pure Python under a "
            f"system-only PATH — {status or ['no accelerator line at all']}"
        )
        _dump_room(room)
        failures += 1
    else:
        print(f"ok   {exe.name}: accelerator live")
        failures += _openmp_source(bundle, room, OMP_LINE)

    return failures


def _openmp_source(bundle: Path, room: Path, marker: str) -> int:
    """WHICH copy of the runtime the daemon loaded — the gate's real verdict.

    Inside the bundle or it does not count: the daemon exe lives there, so the
    shipped copy resolves under the bundle directory (``_internal/`` included)
    and every rescuing copy — System32, a Visual Studio tree — resolves
    outside it.  ``normcase`` because Windows paths are case-insensitive and
    the loader is free to answer in any casing it likes.
    """
    lines = log_lines(room, marker)
    if not lines:
        print(f"FAIL {marker.strip()} missing from the daemon's log")
        return 1
    where = lines[-1].split(marker, 1)[1].strip()
    if where == "NOT MAPPED":
        # Accelerators loaded and no OpenMP runtime is mapped: a future build
        # that stopped linking one, which makes this gate's subject moot and
        # its answer meaningless rather than green.
        print(f"FAIL {OPENMP_DLL}: accelerators loaded but no runtime is mapped")
        return 1
    root = os.path.normcase(str(bundle.resolve()))
    loaded = os.path.normcase(str(Path(where).resolve()))
    if not loaded.startswith(root + os.sep):
        print(
            f"FAIL {OPENMP_DLL}: the engine loaded {where} — a copy from "
            "OUTSIDE the bundle rescued the import, which is exactly what "
            "hides this bug on a runner and not on a user's box"
        )
        return 1
    print(f"ok   {OPENMP_DLL}: loaded from the bundle ({where})")
    return 0


def _build_version() -> str:
    """The release this bundle was frozen from: the build environment's own.

    The workflow freezes and smokes in one environment (``pip install .``), so
    this is the version ``build.py`` baked.  Read from the metadata, never
    from momwire's resolver, which is part of what gate 8 gates.
    """
    from importlib.metadata import version

    return version("momwire")


def _gate_stamp(
    printout: str, basis: str, name: str, *, stale: str | None = None
) -> int:
    """Gate 8 — line 2 names this engine: version, basis, accelerator variant.

    ``basis`` is what the launcher's NAME asks for, so this reads the one
    field of the stamp a reader of a mailed-in ``NEC5.OUT`` cannot check for
    themselves.  Everything is read positionally off line 2; see the module
    docstring for why this gate is not evidence about the solver.

    ``stale`` is the overlay marker the line must END with — ``None`` for a
    clean bundle, where any trailing text at all is a failure.
    """
    lines = printout.splitlines()
    line = lines[STAMP_LINE] if len(lines) > STAMP_LINE else ""
    if not line.startswith(STAMP_PREFIX):
        print(f"FAIL {name}: line 2 is not a momwire stamp ({line!r})")
        return 1
    head = line
    if stale is not None:
        marker = " " + STALE_MARKER.format(stale)
        if not line.endswith(marker):
            print(
                f"FAIL {name}: the stamp does not end in {marker.strip()!r} ({line!r})"
            )
            return 1
        head = line[: -len(marker)]
    fields = head.split()
    if len(fields) != STAMP_FIELDS:
        print(f"FAIL {name}: the stamp is not {STAMP_FIELDS} fields ({line!r})")
        return 1
    _, version, stamped, variant = fields
    if version == NO_VERSION:
        print(
            f"FAIL {name}: the stamp carries no version — this bundle has "
            "neither the baked build version nor `--copy-metadata momwire`, so "
            "a printout a tester mails back cannot say which release answered "
            f"it ({line!r})"
        )
        return 1
    want = _build_version()
    if version != want:
        print(
            f"FAIL {name}: the stamp says momwire {version}, but this bundle "
            f"was frozen from {want} ({line!r})"
        )
        return 1
    if stamped != basis:
        print(f"FAIL {name}: the stamp names basis {stamped!r}, not {basis!r}")
        return 1
    print(f"ok   {name}: stamped {line.strip()}")
    return 0


def _gate_overlay(exe: Path, work: Path, room: Path, env: dict[str, str]) -> int:
    """Gate 8's overlay case (momwire#1277), in a room of its own.

    Plants a VALID stale dist-info — one ``importlib.metadata`` would happily
    answer from, and one that sorts before the real one, which is the shape
    that stamped 0.66.0 on 0.68.0 code in the field — then requires the stamp
    to keep the build's version and to say the folder is mixed.  The plant is
    removed in ``finally``: every other gate needs a clean bundle.
    """
    from momwire.eznec._serve import BASIS as DEFAULT_BASIS

    internal = exe.parent / "_internal"
    if not internal.is_dir():
        print(f"FAIL overlay: no _internal beside {exe.name} to plant into")
        return 1
    planted = internal / f"momwire-{PLANTED_STALE}.dist-info"
    if planted.exists():
        print(f"FAIL overlay: {planted.name} already exists in the bundle")
        return 1
    try:
        planted.mkdir()
        (planted / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: momwire\nVersion: {PLANTED_STALE}\n",
            encoding="utf-8",
        )
        found = sorted(p.name for p in internal.glob("momwire-*.dist-info"))
        print(f"     overlay: _internal now holds {found}")
        deck = FIXTURES / f"{SERVE_IDS[0]}.nec"
        out = work / f"{SERVE_IDS[0]}.overlay.out"
        run([str(exe), str(deck), str(out)], out, env=env)
        text = out.read_bytes().decode("latin-1")
        return _gate_stamp(
            text, DEFAULT_BASIS, f"{SERVE_IDS[0]} (overlay)", stale=PLANTED_STALE
        )
    finally:
        stop(room)
        shutil.rmtree(planted, ignore_errors=True)


def _gates(exe: Path, work: Path, room: Path, env: dict[str, str]) -> int:
    """Gates 6, 1, 2, 4 and 8 against one bundle, in one runtime directory."""
    failures = 0

    # Gate 8's expected basis for the plain launcher, which selects none.
    # Imported rather than restated, for the reason gate 7 imports its log
    # markers: the default has ONE owner, and a second spelling here is how a
    # gate comes to certify a name nothing uses.  Local because the import
    # costs a NumPy load the gates that only launch subprocesses never pay.
    from momwire.eznec._serve import BASIS as DEFAULT_BASIS

    # gate 6 — the launcher goes RESIDENT.
    #
    # Run FIRST although it is listed last, because the spawn count is exact
    # only while this room has seen a single launcher: gate 4 starts a SECOND
    # daemon in it, the twin's, and this gate could then no longer tell a
    # correct pair of servers from a double spawn.  The order costs nothing —
    # what gates 1-3 then measure is the WARM launch, which is the number the
    # sweep economics are actually made of.
    stem = SERVE_IDS[0]
    deck = FIXTURES / f"{stem}.nec"
    module_out = work / f"{stem}.resident.module.out"
    run([sys.executable, "-m", "momwire.eznec", str(deck), str(module_out)], module_out)
    expected = module_out.read_bytes()
    answers = []
    warm = 0.0
    for index in range(2):
        out = work / f"{stem}.resident.{index}.out"
        warm = run([str(exe), str(deck), str(out)], out, env=env)
        answers.append(out.read_bytes())
    spawns = listening(room)
    if answers[0] != expected or answers[1] != expected:
        print(f"FAIL {exe.name}: a resident launch differs from the module's printout")
        _dump_room(room)
        failures += 1
    elif not spawns:
        # The ladder did the work and the printouts are perfect, which is
        # exactly why this has to be a failure: the bundle would be correct
        # and as slow as the thing phase 3 replaced, and nothing else here
        # would say so.
        print(
            f"FAIL {exe.name}: no daemon was ever spawned — the fallback "
            "ladder carried both launches"
        )
        _dump_room(room)
        failures += 1
    elif len(spawns) > 1:
        print(
            f"FAIL {exe.name}: {len(spawns)} daemons spawned where one "
            "warm server was the whole claim"
        )
        _dump_room(room)
        failures += 1
    else:
        print(f"ok   {stem}: resident, one daemon, warm launch {warm:.3f} s")

    # gate 6b — the daemon solved in a WORKER (momwire#1418).  A frozen worker
    # is the bundle's own exe started with `--serve-worker`, a path no other
    # gate reaches; when it cannot start, the pool answers in-process and every
    # printout above is still perfect, so only the log can say it happened.
    ready = [line for line in log_lines(room, "pool: worker pid=") if " ready " in line]
    failed = log_lines(room, "pool: a worker failed to start")
    if failed or not ready:
        print(
            f"FAIL {exe.name}: the daemon's worker pool never started a worker "
            f"— {failed or 'no worker ready line'}"
        )
        _dump_room(room)
        failures += 1
    else:
        print(f"ok   {exe.name}: answered by a pool worker ({len(ready)} started)")

    for stem in SERVE_IDS:
        deck = FIXTURES / f"{stem}.nec"
        frozen_out = work / f"{stem}.frozen.out"
        module_out = work / f"{stem}.module.out"
        elapsed = run([str(exe), str(deck), str(frozen_out)], frozen_out, env=env)
        run(
            [sys.executable, "-m", "momwire.eznec", str(deck), str(module_out)],
            module_out,
        )
        frozen = frozen_out.read_bytes()
        module = module_out.read_bytes()
        if frozen != module:
            print(f"FAIL {stem}: bundle printout differs from the module's")
            failures += 1
        elif b"\r\n" not in frozen:
            print(f"FAIL {stem}: printout is not CRLF")
            failures += 1
        else:
            print(
                f"ok   {stem}: byte-identical, {len(frozen)} bytes, "
                f"launch {elapsed:.2f} s"
            )
            failures += _gate_stamp(frozen.decode("latin-1"), DEFAULT_BASIS, stem)

    deck = FIXTURES / f"{REFUSE_ID}.nec"
    refuse_out = work / f"{REFUSE_ID}.frozen.out"
    elapsed = run([str(exe), str(deck), str(refuse_out)], refuse_out, env=env)
    text = refuse_out.read_bytes().decode("latin-1")
    if "NEC ERROR" not in text:
        print(f"FAIL {REFUSE_ID}: refusal frame missing from the printout")
        failures += 1
    else:
        print(f"ok   {REFUSE_ID}: refusal reached the printout, launch {elapsed:.2f} s")
        # A refusal is stamped too, and it is the printout most likely to be
        # the one mailed back: the deck that got refused is the reason for
        # writing in.
        failures += _gate_stamp(text, DEFAULT_BASIS, f"{REFUSE_ID} (refusal)")

    # gate 4 — every shipped variant is there, and answers in its own basis
    deck = FIXTURES / f"{BASIS_DECK}.nec"

    # The default's answer on this deck, rendered HERE through the module's
    # own default path rather than borrowed from gate 1's file: it is the
    # reference both halves of gate 4 measure against, and a reference that
    # exists by coincidence is a gate that disarms itself when a list moves.
    default_out = work / f"{BASIS_DECK}.default.module.out"
    run(
        [sys.executable, "-m", "momwire.eznec", str(deck), str(default_out)],
        default_out,
    )
    default = default_out.read_bytes()

    # Keyed off the exe's OWN stem, so the marker's hyphen count is the exe's
    # business and not this loop's: `momwire-eznec` -> `razor-nec5`, and a
    # rename of the bundle does not silently reslice the basis.  Casefolded
    # for the same reason `basis_from_program_name` is — this loop has to read
    # the name the way the exe reads it, or a `Momwire-Eznec-Razor-Nec5.exe`
    # that serves correctly is failed here for a casing the exe ignored.
    #
    # The ENGINE is excluded by name, not by luck: it sits in this folder
    # under a name the glob matches, and `engine` is a segment its own entry
    # point CONSUMES — read as a variant it would be gated as a basis called
    # "engine", which the module run would refuse, failing the smoke over a
    # bundle that is exactly right.
    variants = {
        v.stem[len(exe.stem) + 1 :].casefold(): v
        for v in sorted(exe.parent.glob(f"{exe.stem}-*{exe.suffix}"))
        if v.stem.casefold() != ENGINE_STEM.casefold()
    }
    for basis in _shipped_variants():
        if basis not in variants:
            print(f"FAIL {exe.stem}-{basis}{exe.suffix}: shipped variant is MISSING")
            failures += 1

    # Every copy present is checked, not just the shipped ones: making one is
    # the documented mechanism, so a copy in the folder is a variant to gate.
    for basis, variant in variants.items():
        v_out = work / f"{BASIS_DECK}.{basis}.frozen.out"
        m_out = work / f"{BASIS_DECK}.{basis}.module.out"
        run([str(variant), str(deck), str(v_out)], v_out, env=env)
        run(
            [
                sys.executable,
                "-c",
                "import sys;from momwire.eznec._shell import main;"
                f"sys.exit(main(sys.argv[1:], basis={basis!r}))",
                str(deck),
                str(m_out),
            ],
            m_out,
        )
        frozen, module = v_out.read_bytes(), m_out.read_bytes()
        printout = frozen.decode("latin-1")
        # momwire#628's own shape first, because it is the most specific
        # reading of the same bytes: an exe that matched the default's answer
        # on a deck where the named basis does NOT is an engine that ignored
        # its filename, and saying so beats saying "differs from the module".
        # Guarded by `module != default` because that is what makes the deck
        # able to tell them apart at all.
        if module != default and frozen == default:
            print(f"FAIL {variant.name}: answered as the DEFAULT, not {basis!r}")
            failures += 1
        elif frozen != module:
            print(f"FAIL {variant.name}: does not answer in basis {basis!r}")
            failures += 1
        elif REFUSED in printout or SOLVED not in printout:
            print(f"FAIL {variant.name}: {basis!r} REFUSED this deck, it did not serve")
            failures += 1
        elif module == default:
            # Not a pass by coincidence but a deck that cannot tell these two
            # apart: several bases render 0010 identically.  The name is
            # honoured — byte identity above says so — and the ANSWER is
            # simply not evidence either way.
            print(
                f"ok   {variant.name}: answers in {basis!r} (== default on this deck)"
            )
        else:
            print(f"ok   {variant.name}: answers in {basis!r}, distinct from default")
        failures += _gate_stamp(printout, basis, variant.name)

    return failures


def _gate_nec5(exe: Path, work: Path, env: dict[str, str]) -> int:
    """Gate 9 — the NEC-5 names (momwire#1239; primary since #1295 phase 4).

    SimNEC runs ``"<path>" -version`` before it accepts a nec5 engine, then
    ``<path> <deck> <printout>`` like EZNEC, which runs only the latter.  So
    each shipped name must be present, answer the probe with exit 0 and the
    ``NEC5momwire.<maj>.<min>`` line, and serve 0010 byte-identically to the
    module in the basis its name reads past ``nec5-`` (none for the plain
    name).
    """
    from importlib.metadata import version

    failures = 0
    major, minor = version("momwire").split(".")[:2]
    want = f"NEC5momwire.{major}.{minor}"
    deck = FIXTURES / f"{BASIS_DECK}.nec"
    for stem in _nec5_launchers():
        launcher = exe.with_name(f"{stem}{exe.suffix}")
        if not launcher.is_file():
            print(f"FAIL {launcher.name}: shipped NEC-5 launcher is MISSING")
            failures += 1
            continue
        probe = subprocess.run(
            [str(launcher), "-version"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        first = (probe.stdout.splitlines() or [""])[0]
        if probe.returncode != 0 or first != want:
            print(
                f"FAIL {launcher.name} -version: exit {probe.returncode}, "
                f"{first!r} (want {want!r})"
            )
            failures += 1
            continue
        basis = stem.partition("nec5-")[2] or None
        v_out = work / f"{BASIS_DECK}.{stem}.frozen.out"
        m_out = work / f"{BASIS_DECK}.{stem}.module.out"
        run([str(launcher), str(deck), str(v_out)], v_out, env=env)
        run(
            [
                sys.executable,
                "-c",
                "import sys;from momwire.eznec._shell import main;"
                f"sys.exit(main(sys.argv[1:], basis={basis!r}) if {basis!r} "
                "else main(sys.argv[1:]))",
                str(deck),
                str(m_out),
            ],
            m_out,
        )
        frozen = v_out.read_bytes()
        if frozen != m_out.read_bytes():
            print(f"FAIL {launcher.name}: does not answer in basis {basis!r}")
            failures += 1
        elif SOLVED not in frozen.decode("latin-1"):
            print(f"FAIL {launcher.name}: did not serve {BASIS_DECK}")
            failures += 1
        else:
            print(f"ok   {launcher.name}: {want}, answers in {basis or 'the default'}")
    return failures


def _licensed_nec4_impedance() -> complex:
    """The licensed NEC-4.2's feed-point Z for gate 10's deck, read out of its
    committed printout (``tests/fixtures/eznec_nec42/printouts/``)."""
    (path,) = (NEC4_FIXTURES / "printouts").glob(f"{NEC4_DECK}.out")
    z = feed_impedance(path.read_text(encoding="latin-1"))
    if z is None:
        raise SystemExit(f"no feed row in the licensed printout {path}")
    return z


def _gate_nec4(exe: Path, work: Path, env: dict[str, str]) -> int:
    """Gate 10 — the NEC-4.2 names (momwire#1295).

    Each shipped ``momwire-nec4[-<basis>]`` must be present and serve a deck
    the NEC-4.2 slot wrote, byte-identically to ``_shell.main`` run in the
    nec4 dialect and the basis its name reads past ``nec4-`` — the default
    for the plain name.  Byte identity is the gate on the bundle; the banner,
    the solve, the stamp and the licensed impedance are what make the
    comparison unable to pass on a wrong dialect or a refusal (both sides of
    a refusal are byte-equal too, gate 4's blind spot).
    """
    from momwire.eznec._serve import BASIS as DEFAULT_BASIS

    failures = 0
    deck = NEC4_FIXTURES / f"{NEC4_DECK}.nec"
    licensed = _licensed_nec4_impedance()
    answers: dict[str, bytes] = {}
    for stem in _nec4_launchers():
        launcher = exe.with_name(f"{stem}{exe.suffix}")
        if not launcher.is_file():
            print(f"FAIL {launcher.name}: shipped NEC-4.2 launcher is MISSING")
            failures += 1
            continue
        basis = stem.partition("nec4-")[2] or None
        v_out = work / f"{NEC4_DECK}.{stem}.frozen.out"
        m_out = work / f"{NEC4_DECK}.{stem}.module.out"
        run([str(launcher), str(deck), str(v_out)], v_out, env=env)
        run(
            [
                sys.executable,
                "-c",
                "import sys;from momwire.eznec._shell import main;"
                f"sys.exit(main(sys.argv[1:], basis={basis!r}, dialect='nec4') "
                f"if {basis!r} else main(sys.argv[1:], dialect='nec4'))",
                str(deck),
                str(m_out),
            ],
            m_out,
        )
        frozen = mask_wall_clock(v_out.read_bytes())
        module = mask_wall_clock(m_out.read_bytes())
        printout = frozen.decode("latin-1")
        z = feed_impedance(printout)
        if module.count(b"<t>") != 3:
            # The mask must have found exactly the three fields it exists
            # for; more or fewer means the layout moved and the comparison
            # below is no longer the one this gate was written to make.
            print(
                f"FAIL {launcher.name}: masked {module.count(b'<t>')} wall-clock "
                "fields in the module's printout, expected 3 (FILL, FACTOR, "
                "RUN TIME)"
            )
            failures += 1
        elif frozen != module:
            print(
                f"FAIL {launcher.name}: does not answer as the nec4 module in {basis!r}"
            )
            failures += 1
        elif NEC4_BANNER not in printout or NEC5_BANNER in printout:
            print(f"FAIL {launcher.name}: the printout is not under the NEC-4.2 banner")
            failures += 1
        elif REFUSED in printout or SOLVED not in printout or z is None:
            print(f"FAIL {launcher.name}: REFUSED {NEC4_DECK}, it did not serve")
            failures += 1
        elif abs(z - licensed) / abs(licensed) >= NEC4_SANITY_REL:
            print(
                f"FAIL {launcher.name}: Z = {z:.4f} ohm, not within "
                f"{NEC4_SANITY_REL:.0%} of licensed NEC-4.2's {licensed:.4f}"
            )
            failures += 1
        else:
            answers[stem] = frozen
            print(
                f"ok   {launcher.name}: NEC-4.2 dialect, answers in "
                f"{basis or 'the default'}, Z = {z:.4f} ohm "
                f"(licensed NEC-4.2 {licensed:.4f})"
            )
        failures += _gate_stamp(printout, basis or DEFAULT_BASIS, launcher.name)

    # The names must not all be one engine: on this deck bspline and
    # sinusoidal differ, so two shipped launchers answering byte-identically
    # means one of them ignored its name — which the per-name comparison
    # above catches too, but only while the deck keeps them apart.  This
    # says when it stops doing so, rather than passing on a deck that cannot
    # tell.
    if len(answers) > 1 and len(set(answers.values())) < len(answers):
        print(
            f"FAIL {NEC4_DECK}: two NEC-4.2 launchers answered byte-identically; "
            "the deck no longer tells their bases apart"
        )
        failures += 1
    return failures


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    exe = Path(sys.argv[1]).resolve()
    if not exe.is_file():
        print(f"ERROR: no such executable: {exe}", file=sys.stderr)
        return 2

    work = Path("smoke-out")
    work.mkdir(exist_ok=True)

    # gate 5 — the engine is beside the launcher, and it is checked before
    # anything is launched: a launcher with nothing to spawn still writes a
    # printout on every path (rung 4's named refusal), so without this the
    # bundle's most complete failure would surface as four confusing ones.
    engine = exe.with_name(f"{ENGINE_STEM}{exe.suffix}")
    if not engine.is_file():
        print(f"FAIL {engine.name}: the engine is MISSING from this bundle")
        print("1 smoke failure(s)")
        return 1
    print(f"ok   {engine.name}: the engine is beside the launcher")

    # One private runtime directory for the whole run, and every daemon in it
    # killed at the end.  Private because a smoke that used the real one would
    # be answered by whatever a PREVIOUS run left warm — gate 6 would then
    # count zero spawns and be right to fail, on a bundle that was fine — and
    # because leaving a 117 MB engine resident in a CI runner's %LOCALAPPDATA%
    # for its fifteen idle minutes is not this script's to do.  `mkdtemp`
    # keeps the name short, which is what keeps the socket inside sun_path.
    room = Path(tempfile.mkdtemp(prefix="mw-smoke-"))
    env = {**os.environ, "MOMWIRE_PORTAL_RUNTIME_DIR": str(room)}
    try:
        failures = _gates(exe, work, room, env)
        failures += _gate_nec5(exe, work, env)
        failures += _gate_nec4(exe, work, env)
    finally:
        stop(room)

    # Gate 7 in a room of its own, for the reason gate 6 runs first: its launch
    # is a THIRD daemon, and sharing the room above would make that gate's
    # exact spawn count unreadable.
    room = Path(tempfile.mkdtemp(prefix="mw-smoke-omp-"))
    try:
        failures += _gate_self_contained(exe, work, room)
    finally:
        stop(room)

    # Gate 8's overlay case LAST and in a fresh room: it plants a stale
    # dist-info in the bundle, and a fresh room means a fresh daemon whose
    # whole life is spent in the overlaid folder.
    room = Path(tempfile.mkdtemp(prefix="mw-smoke-overlay-"))
    failures += _gate_overlay(
        exe, work, room, {**os.environ, "MOMWIRE_PORTAL_RUNTIME_DIR": str(room)}
    )

    if failures:
        print(f"{failures} smoke failure(s)")
        return 1
    print("smoke green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
