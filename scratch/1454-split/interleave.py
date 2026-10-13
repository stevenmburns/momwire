"""Base versus change, same machine, same session, interleaved (momwire#1454
step 3).

    python interleave.py BASE_TREE CHANGE_TREE DECK [DECK ...] [--rounds N]

Runs `timing.py` for each tree alternately (base, change, base, change, ...),
one fresh process per run, one OpenMP thread, and prints for every deck and
pattern the median over rounds of each run's median ms per call, for the
whole render and for the exact-kernel correction alone.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def one(tree, deck, series):
    env = dict(os.environ, PYTHONPATH=str(Path(tree) / "src"), OMP_NUM_THREADS="1")
    out = subprocess.run(
        [sys.executable, str(HERE / "timing.py"), deck, "--series", series],
        env=env,
        cwd="/",
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("change")
    ap.add_argument("decks", nargs="+")
    ap.add_argument("--rounds", type=int, default=5)
    ns = ap.parse_args()
    arms = (("base", ns.base, "off"), ("change", ns.change, "on"))
    for deck in ns.decks:
        runs = {name: [] for name, _t, _s in arms}
        for _ in range(ns.rounds):
            for name, tree, series in arms:
                runs[name].append(one(tree, deck, series))
        print(f"\n{Path(deck).name}  ({ns.rounds} rounds, interleaved, 1 thread)")
        print(
            f"{'pattern':9} {'what':10} {'base ms':>9} {'change ms':>10} {'delta':>8}"
        )
        for pat in ("repeat", "sweep5", "sweep103"):
            for what, key in (
                ("render", "ms_median"),
                ("correction", "corr_ms_median"),
            ):
                b = statistics.median(r[pat][key] for r in runs["base"])
                c = statistics.median(r[pat][key] for r in runs["change"])
                print(f"{pat:9} {what:10} {b:9.2f} {c:10.2f} {c - b:+8.2f}")
        for name in runs:
            r = runs[name][-1]
            print(f"  {name}: tree {r['tree']}, series {r['series']}")
            for pat in ("repeat", "sweep5", "sweep103"):
                print(f"    {pat}: {r[pat]['calls']} calls, {r[pat]['stats']}")


if __name__ == "__main__":
    main()
