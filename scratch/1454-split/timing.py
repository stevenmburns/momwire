"""Warm in-process timing of razor-2p on one deck (momwire#1454 step 3).

    PYTHONPATH=<tree>/src python timing.py DECK [--series {on,off}] [--reps N]

Times `momwire.eznec._shell.render` -- the solve path the resident server
runs, not process start -- after two untimed warm-up calls, in three
patterns:

* ``repeat``  the same deck again and again;
* ``sweep5``  the deck at five frequencies in turn (base, ±1.4 %, ±2.7 %),
              cycled, so every call is a frequency the previous call was not;
* ``sweep103`` one pass over 103 frequencies spread ±2 % about the deck's.

and, inside each, the time spent in the exact-kernel correction alone. One
JSON object on stdout: median and minimum ms per call, the cache counters of
each pattern, and how many pairs took each series rule. `interleave.py` runs
this for two trees alternately and tabulates.
"""

import argparse
import json
import re
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

from momwire import RazorSolver, _razor_exact_plan as P  # noqa: E402
from momwire.eznec import _shell  # noqa: E402

try:
    from momwire import _exact_kernel_series as ES
except ImportError:
    ES = None

FR = re.compile(r"^(FR\s+\S+\s+\S+\s+\S+\s+\S+\s+)(\S+)(.*)$", re.M)


def at(deck, scale):
    def sub(m):
        return f"{m.group(1)}{float(m.group(2)) * scale:.6f}{m.group(3)}"

    out, n = FR.subn(sub, deck)
    assert n == 1, n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("deck")
    ap.add_argument("--series", choices=("on", "off"), default="on")
    ap.add_argument("--reps", type=int, default=15)
    ns = ap.parse_args()
    if ES is not None:
        ES.ENABLED = ns.series == "on"
    deck = Path(ns.deck).read_text(encoding="latin-1")

    spent = [0.0]
    orig = RazorSolver._add_exact_kernel_correction

    def timed(self, *a, **k):
        t0 = time.perf_counter()
        try:
            return orig(self, *a, **k)
        finally:
            spent[0] += time.perf_counter() - t0

    RazorSolver._add_exact_kernel_correction = timed

    def run(decks, reps):
        P.clear()
        for d in decks[:2]:
            _shell.render(d, basis="razor-2p")
        if len(decks) == 1:
            _shell.render(decks[0], basis="razor-2p")
        before = dict(P.STATS)
        ser0 = dict(getattr(P, "SERIES_STATS", {}))
        total, corr = [], []
        for _ in range(reps):
            for d in decks:
                spent[0] = 0.0
                t0 = time.perf_counter()
                _shell.render(d, basis="razor-2p")
                total.append(time.perf_counter() - t0)
                corr.append(spent[0])
        stats = {k: P.STATS[k] - before[k] for k in P.STATS}
        stats.update(
            {
                f"series_{k}": v - ser0[k]
                for k, v in getattr(P, "SERIES_STATS", {}).items()
            }
        )
        return {
            "calls": len(total),
            "ms_median": 1e3 * statistics.median(total),
            "ms_min": 1e3 * min(total),
            "corr_ms_median": 1e3 * statistics.median(corr),
            "corr_ms_min": 1e3 * min(corr),
            "stats": stats,
        }

    sweep5 = [at(deck, s) for s in (0.973, 0.986, 1.0, 1.014, 1.027)]
    sweep103 = [at(deck, 0.98 + 0.04 * i / 102) for i in range(103)]
    out = {
        "tree": str(Path(_shell.__file__).resolve().parents[3]),
        "series": ns.series if ES is not None else "absent",
        "deck": Path(ns.deck).name,
        "repeat": run([deck], ns.reps * 5),
        "sweep5": run(sweep5, ns.reps),
        "sweep103": run(sweep103, 2),
    }
    json.dump(out, sys.stdout)
    print()


if __name__ == "__main__":
    main()
