"""Smoke-gate the frozen EZNEC drop-in against the unfrozen module.

Usage::

    python scripts/eznec_freeze/smoke.py dist/momwire-eznec/momwire-eznec[.exe]

Three gates, derived from the seam's own contract (momwire#497 U1):

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
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures" / "eznec" / "decks"

# Two serving decks spanning the seam's range — a bare-wire rung-1 model and
# a network-heavy feed system — plus the standing refusal (NE over GN 0).
SERVE_IDS = ("0010_dipole-in-free-space", "0000_cardioid-l-network-feed")
REFUSE_ID = "0022_vertical-over-real-ground"


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

    if failures:
        print(f"{failures} smoke failure(s)")
        return 1
    print("smoke green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
