"""Smoke-gate the frozen EZNEC drop-in against the unfrozen module.

Usage::

    python scripts/eznec_freeze/smoke.py dist/momwire-eznec/momwire-eznec[.exe]

Four gates, derived from the seam's own contract (momwire#497 U1):

1. **Byte identity** — on decks that serve, the frozen exe's printout must
   equal ``python -m momwire.eznec``'s byte for byte (the printout carries no
   wall-clock, so freezing may not change a single byte, CRLF included).
2. **The refusal frame travels** — a deck the seam refuses must still produce
   a printout carrying the ``NEC ERROR`` line, exit 0 (the file is the only
   channel EZNEC reads).
3. **Launch cost, informational** — per-launch wall time is printed but not
   gated; EZNEC launches once per frequency point, so this number is the
   sweep economics (real engine baseline 18–37 ms; one-dir momwire measured
   ~1.3 s on the sitting-4 box).
4. **Every shipped variant is PRESENT and answers, in the basis its NAME
   claims** (momwire#593).  The basis rides on the filename, so this is the
   gate that makes the bundle's shape a fact rather than an intention.

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
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "eznec" / "decks"


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


def run(cmd: list[str], out: Path) -> float:
    started = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, timeout=300)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise SystemExit(f"exit {result.returncode} from {cmd}: {result.stderr!r}")
    if not out.is_file():
        raise SystemExit(f"no printout written by {cmd}")
    return elapsed


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
    failures = 0

    for stem in SERVE_IDS:
        deck = FIXTURES / f"{stem}.nec"
        frozen_out = work / f"{stem}.frozen.out"
        module_out = work / f"{stem}.module.out"
        elapsed = run([str(exe), str(deck), str(frozen_out)], frozen_out)
        run(
            [sys.executable, "-m", "momwire.eznec", str(deck), str(module_out)],
            module_out,
        )
        frozen = frozen_out.read_bytes()
        module = module_out.read_bytes()
        if frozen != module:
            print(f"FAIL {stem}: frozen printout differs from the module's")
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
    elapsed = run([str(exe), str(deck), str(refuse_out)], refuse_out)
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
    variants = {
        v.stem[len(exe.stem) + 1 :].casefold(): v
        for v in sorted(exe.parent.glob(f"{exe.stem}-*{exe.suffix}"))
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
        run([str(variant), str(deck), str(v_out)], v_out)
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

    if failures:
        print(f"{failures} smoke failure(s)")
        return 1
    print("smoke green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
