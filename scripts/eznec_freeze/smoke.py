"""Smoke-gate the packaged EZNEC drop-in against the unfrozen module.

Usage::

    python scripts/eznec_freeze/smoke.py dist/momwire-eznec/momwire-eznec[.exe]

The argument is a LAUNCHER — the native client EZNEC points at — and every
gate below runs the bundle end to end: launcher, spawned engine, printout.

Six gates, derived from the seam's own contract (momwire#497 U1):

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

Gate 4 exists because momwire#628 was exactly that bug on the other route:
a copy named for one engine served another, and the printout was internally
CONSISTENT because the banner names whatever actually ran.  Nothing in a
printout can reveal it, so it has to be caught here, by comparing each exe
against the module RUN IN THE BASIS THE NAME ASKS FOR.

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

# The one frozen program in the bundle — `build.py`'s ENGINE_NAME and the
# client's ENGINE_NAME, restated here rather than imported because gate 5 is
# about a FILE being there and a name read out of the thing being gated would
# certify itself.
ENGINE_STEM = "momwire-eznec-engine"


def _shipped_variants() -> tuple[str, ...]:
    """``build.py``'s list, read from the file whose copy loop makes them.

    Imported by path because ``scripts/`` is not a package, and imported at
    all rather than restated because a second list is what regresses: the
    copy loop and the presence gate have to be the same fact, or the gate
    certifies the shape it was told about instead of the one that shipped.
    """
    spec = importlib.util.spec_from_file_location(
        "eznec_freeze_build", HERE / "build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SHIPPED_VARIANTS


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

# What a printout looks like when it is an ANSWER rather than a refusal.
# Both directions are needed: the refusal frame is what the seam prints when
# a basis cannot host the deck, and its absence alone would also be satisfied
# by a truncated file.
SOLVED = "ANTENNA INPUT PARAMETERS"
REFUSED = "NEC ERROR"


def run(cmd: list[str], out: Path, env: dict[str, str] | None = None) -> float:
    started = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, timeout=300, env=env)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise SystemExit(f"exit {result.returncode} from {cmd}: {result.stderr!r}")
    if not out.is_file():
        raise SystemExit(f"no printout written by {cmd}")
    return elapsed


def listening(room: Path) -> list[str]:
    """Every ``listening pid=`` line a daemon in this room ever wrote.

    One line per spawned daemon, so this counts SPAWNS across the whole smoke
    run — which is what tells a warm reuse from a second server, and either of
    those from the fallback ladder having quietly done all the work.
    """
    return [
        line
        for log in sorted(room.glob("*.log"))
        for line in log.read_text(errors="replace").splitlines()
        if "listening pid=" in line
    ]


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


def _gates(exe: Path, work: Path, room: Path, env: dict[str, str]) -> int:
    """Gates 6, 1, 2 and 4 against one bundle, in one runtime directory."""
    failures = 0

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
        failures += 1
    elif len(spawns) > 1:
        print(
            f"FAIL {exe.name}: {len(spawns)} daemons spawned where one "
            "warm server was the whole claim"
        )
        failures += 1
    else:
        print(f"ok   {stem}: resident, one daemon, warm launch {warm:.3f} s")

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

    deck = FIXTURES / f"{REFUSE_ID}.nec"
    refuse_out = work / f"{REFUSE_ID}.frozen.out"
    elapsed = run([str(exe), str(deck), str(refuse_out)], refuse_out, env=env)
    text = refuse_out.read_bytes().decode("latin-1")
    if "NEC ERROR" not in text:
        print(f"FAIL {REFUSE_ID}: refusal frame missing from the printout")
        failures += 1
    else:
        print(f"ok   {REFUSE_ID}: refusal reached the printout, launch {elapsed:.2f} s")

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
    finally:
        stop(room)

    if failures:
        print(f"{failures} smoke failure(s)")
        return 1
    print("smoke green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
