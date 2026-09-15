"""momwire#1064 D5b, a diagnostic of G5: where a deck's cold solve spends its
time, on whichever tree PYTHONPATH selects.

  PYTHONPATH=<momwire src>:<antennaknobs src> python d5_profile.py \\
      --deck dipole1mm --tree T --rep N --out F.jsonl

Profiles one cold solve (G5's own deck spelling, from `t5_fill_cost.py`) with
cProfile. Records the wall time, the calls and cumulative time of the functions
that do the below/below work (the fill, the direct surfaces, the projection
kernel, the band routing), and the top 30 functions by cumulative time.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t5_fill_cost import DECKS  # noqa: E402

FOCUS = (
    "_fill_region",
    "iv_surfaces_direct_below",
    "_six_integrals_below_many",
    "below_six_integrals_batch",
    "remainder_field_proj_below",
    "remainder_field_proj_batch_below",
    "_ensure_for",
    "_interp",
    "compute_impedance",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True, choices=sorted(DECKS))
    ap.add_argument("--tree", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cold, _ = DECKS[args.deck]()
    prof = cProfile.Profile()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        prof.enable()
        cold()
        prof.disable()
        wall = time.perf_counter() - t0
    rows = []
    for (path, line, name), (_cc, ncalls, tottime, cumtime, _callers) in pstats.Stats(
        prof
    ).stats.items():
        rows.append(
            dict(
                func=f"{Path(path).name}:{line}:{name}",
                ncalls=ncalls,
                tottime=tottime,
                cumtime=cumtime,
            )
        )
    rows.sort(key=lambda r: -r["cumtime"])
    focus = {
        key: [r for r in rows if r["func"].endswith(":" + key) or key in r["func"]]
        for key in FOCUS
    }
    rec = dict(
        deck=args.deck,
        tree=args.tree,
        rep=args.rep,
        wall_s=wall,
        focus=focus,
        top=rows[:30],
    )
    with args.out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(args.tree, args.deck, f"{wall:.2f} s")


if __name__ == "__main__":
    main()
