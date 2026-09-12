"""What the AVX2 split actually buys — momwire#1032.

The double build costs a doubled wheel. This is the number that says whether
the baseline half is worth shipping at all, and what a user on a pre-AVX2 CPU
would have lost if we had simply guarded the import and fallen back to numpy.

Three configurations, same box, same interpreter, same decks:

    avx2    today's wheel (`-mavx2 -mfma` / `/arch:AVX2`)
    sse2    the new baseline build
    pure    no extension at all — what a guard-only fix would have delivered

Each runs in its OWN interpreter, selected by `MOMWIRE_FORCE_VARIANT`, and
asserts what it actually loaded before timing anything: a benchmark that
silently measured the same build three times is the failure mode here, and it
looks exactly like "the variants are equally fast".

METHOD. Repeats run back to back to a fixed count and the script reports the
mean of the last fifth alongside the first fifth. On a thermally-limited part
a "min of N short runs" measures turbo headroom rather than the code, so the
drift between those two is printed and is part of the result: if it is large,
the number is a thermal measurement and should be read as one.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPEATS = {"default": 9, "big": 3}
VARIANTS = ("avx2", "sse2", "pure")

# `legacy` names the unsuffixed extension, which a double-build checkout does
# not have — so forcing it is how you get a genuinely pure-Python momwire
# without uninstalling anything. The child asserts it, so a checkout that DOES
# still carry an unsuffixed .so fails loudly instead of mislabelling a row.
_FORCE = {"avx2": "avx2", "sse2": "sse2", "pure": "legacy"}

CHILD = r"""
import json, os, sys, time, warnings
warnings.simplefilter("ignore")
import momwire

want = os.environ["EXPECT_VARIANT"]
got = momwire.accelerator_variant
if want == "pure":
    assert got is None and not momwire.accelerated, f"expected pure Python, got {got!r}"
else:
    assert got == want, f"expected {want!r}, got {got!r}"

sys.path.insert(0, os.environ["AK_SRC"])
from antennaknobs.engines.momwire import MomwireEngine
from momwire import BSplineSolver
from antennaknobs.designs.dipoles import invvee

deck = os.environ["DECK"]
b = invvee.Builder()
if deck == "big":
    b.nominal_nsegs = int(os.environ["BIG_NSEGS"])

# One untimed warm-up: first call pays import-time table building, not fill.
MomwireEngine(b, solver=BSplineSolver, ground="free").impedance()

times = []
for _ in range(int(os.environ["REPEATS"])):
    t0 = time.perf_counter()
    MomwireEngine(b, solver=BSplineSolver, ground="free").impedance()
    times.append(time.perf_counter() - t0)

nsegs = sum(w[2] for w in b.build_wires())
print("RESULT " + json.dumps({"variant": got, "times": times, "nsegs": nsegs}))
"""


def run(variant: str, deck: str, big_nsegs: int) -> dict:  # noqa: D103
    env = {
        **os.environ,
        "MOMWIRE_FORCE_VARIANT": _FORCE[variant],
        "EXPECT_VARIANT": variant,
        "DECK": deck,
        "REPEATS": str(REPEATS[deck]),
        "BIG_NSEGS": str(big_nsegs),
        "AK_SRC": str(Path(__file__).resolve().parents[3] / "src"),
        # Pin the thread count so the three rows differ by instruction set and
        # nothing else; OpenMP's default can vary with load.
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4"),
    }
    out = subprocess.run(
        [sys.executable, "-c", CHILD], env=env, capture_output=True, text=True
    )
    if out.returncode != 0:
        raise SystemExit(f"{variant}/{deck} failed:\n{out.stderr[-3000:]}")
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")][-1]
    return json.loads(line[len("RESULT ") :])


def summarise(times: list[float]) -> tuple[float, float]:
    fifth = max(1, len(times) // 5)
    return statistics.mean(times[-fifth:]), statistics.mean(times[:fifth])


def main() -> int:
    big = int(sys.argv[1]) if len(sys.argv) > 1 else 260
    rows = {}
    for deck in ("default", "big"):
        for variant in VARIANTS:
            t0 = time.time()
            res = run(variant, deck, big)
            steady, first = summarise(res["times"])
            rows[(deck, variant)] = (steady, first, res["nsegs"])
            print(
                f"{deck:8s} {variant:5s} nsegs={res['nsegs']:5d} "
                f"steady={steady * 1e3:9.1f} ms  first={first * 1e3:9.1f} ms  "
                f"drift={100 * (first - steady) / steady:+6.1f}%  "
                f"[{time.time() - t0:.0f}s]"
            )
    print()
    for deck in ("default", "big"):
        base = rows[(deck, "avx2")][0]
        nsegs = rows[(deck, "avx2")][2]
        print(f"--- {deck} deck ({nsegs} segments), relative to avx2 ---")
        for variant in VARIANTS:
            steady = rows[(deck, variant)][0]
            print(f"  {variant:5s} {steady * 1e3:9.1f} ms   {steady / base:6.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
