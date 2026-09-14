"""U9 fill-cost timing table (PLAN.md Amendment 5): per deck and tree, the
median cold and warm times over the repeats, the branch/main ratios, max RSS,
whether the low band filled, and the first-vs-last repeat drift per tree.

  python t2_table.py t2_fill_cost.jsonl [--branch-marker momwire-wt-u9src]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--branch-marker", default="momwire-wt-u9src")
    args = ap.parse_args()
    rows = [
        json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()
    ]
    by = defaultdict(list)
    errors = []
    for r in rows:
        tree = "branch" if args.branch_marker in r["momwire"] else "main"
        if "error" in r:
            errors.append((r["deck"], tree, r["rep"], r["error"]))
            continue
        by[(r["deck"], tree)].append(r)
    decks = sorted(
        {d for d, _t in by}, key=lambda d: [r["deck"] for r in rows].index(d)
    )
    print(
        "| deck | main cold | branch cold | cold ratio | main warm | branch warm | warm ratio "
        "| low band (main/branch) | max RSS MB (main/branch) | drift first→last (main, branch) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    out = {}
    for d in decks:
        m, b = by.get((d, "main"), []), by.get((d, "branch"), [])
        if not m or not b:
            print(f"| {d} | missing: main {len(m)}, branch {len(b)} |")
            continue

        def med(rs, k):
            return statistics.median(r[k] for r in rs)

        def drift(rs, k):
            rs = sorted(rs, key=lambda row: row["rep"])
            return rs[-1][k] / rs[0][k] if len(rs) > 1 else float("nan")

        mc, bc, mw, bw = (
            med(m, "cold_s"),
            med(b, "cold_s"),
            med(m, "warm_s"),
            med(b, "warm_s"),
        )
        out[d] = dict(
            main_cold=mc,
            branch_cold=bc,
            cold_ratio=bc / mc,
            main_warm=mw,
            branch_warm=bw,
            warm_ratio=bw / mw,
            n=(len(m), len(b)),
            low_band=(
                any(r["low_band_filled"] for r in m),
                any(r["low_band_filled"] for r in b),
            ),
            maxrss=(max(r["maxrss_mb"] for r in m), max(r["maxrss_mb"] for r in b)),
            drift=(drift(m, "cold_s"), drift(b, "cold_s")),
            z_equal_cold=[r["z_cold"] for r in m][:1] == [r["z_cold"] for r in b][:1],
        )
        o = out[d]
        print(
            f"| {d} | {mc:.2f} s | {bc:.2f} s | **{o['cold_ratio']:.2f}×** | {mw:.2f} s | {bw:.2f} s "
            f"| {o['warm_ratio']:.2f}× | {o['low_band'][0]}/{o['low_band'][1]} "
            f"| {o['maxrss'][0]:.0f}/{o['maxrss'][1]:.0f} | {o['drift'][0]:.2f}, {o['drift'][1]:.2f} |"
        )
    if errors:
        print("\nerrors:")
        for e in errors:
            print("  ", e)
    Path(args.jsonl.with_suffix(".table.json")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
