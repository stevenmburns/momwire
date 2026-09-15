"""momwire#1064 G5 and G1b, read from `t5_fill_cost.jsonl`.

  python t5_table.py

Per deck and tree: the median cold and warm seconds over the repeats, the max
RSS, which grazing bands were filled, and Z on every repeat. Then, as
registered in PLAN.md:

  G5, cold:
    decks in [0.05, 0.1] deg (dipole935, brv_corner): branch/0.55.0 in [0.92, 1.08]
    decks under 0.05 deg (dipole1mm, twonode11):      branch/main   in [0.92, 1.08]
    catalog decks:                  branch/main and branch/0.55.0 in [0.95, 1.05]
  G5, warm: every ratio the deck has in [0.95, 1.05]
  G1b: Z (cold and warm, every repeat) bit-identical, the catalog decks against
       main, and dipole935 and brv_corner against 0.55.0.

The predictions are read beside the bars. An error row, or a (deck, tree) with
no rows, is a failure.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
# `python t5_table.py _joint` reads the re-timing's rows beside the first G5's.
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
CATALOG = ("buried_dipole", "brv_default", "ebc_default", "brv48")
MID = ("dipole935", "brv_corner")
SUB = ("dipole1mm", "twonode11")
TREES_OF = {
    **{d: ("v055", "main", "branch") for d in CATALOG + MID},
    **{d: ("main", "branch") for d in SUB},
}

# Cold predictions: (ratio name, centre, half-width).
PREDICT = {
    **{d: [("branch/main", 1.00, 0.05), ("branch/v055", 1.00, 0.05)] for d in CATALOG},
    "dipole935": [("branch/v055", 1.00, 0.08), ("branch/main", 0.35, 0.06)],
    "brv_corner": [("branch/v055", 1.00, 0.08), ("branch/main", 0.44, 0.08)],
    **{d: [("branch/main", 1.00, 0.08)] for d in SUB},
}


def main():
    rows = [
        json.loads(line)
        for line in (HERE / f"t5_fill_cost{TAG}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    fails = [
        f"{r['deck']}/{r['tree']} rep {r['rep']}: {r['error'][:160]}"
        for r in rows
        if "error" in r
    ]
    by = defaultdict(list)
    for r in rows:
        if "error" not in r:
            by[(r["deck"], r["tree"])].append(r)
    table = {}
    for deck, trees in TREES_OF.items():
        for tree in trees:
            rs = sorted(by.get((deck, tree), []), key=lambda r: r["rep"])
            if not rs:
                fails.append(f"{deck}/{tree}: no rows")
                continue
            table.setdefault(deck, {})[tree] = dict(
                n=len(rs),
                cold_s=statistics.median(r["cold_s"] for r in rs),
                warm_s=statistics.median(r["warm_s"] for r in rs),
                cold_all=[round(r["cold_s"], 3) for r in rs],
                maxrss_mb=max(r["maxrss_mb"] for r in rs),
                bands_filled=rs[0]["bands_filled"],
                z=[(r["z_cold"], r["z_warm"]) for r in rs],
            )

    def ratio(deck, key, a, b):
        return table[deck][a][key] / table[deck][b][key]

    g5, g1b, predictions = {}, {}, {}
    for deck in table:
        t = table[deck]
        checks = {}
        if deck in CATALOG:
            for other in ("main", "v055"):
                if other in t and "branch" in t:
                    checks[f"cold branch/{other}"] = (
                        ratio(deck, "cold_s", "branch", other),
                        0.95,
                        1.05,
                    )
        if deck in MID and "v055" in t and "branch" in t:
            checks["cold branch/v055"] = (
                ratio(deck, "cold_s", "branch", "v055"),
                0.92,
                1.08,
            )
        if deck in SUB and "main" in t and "branch" in t:
            checks["cold branch/main"] = (
                ratio(deck, "cold_s", "branch", "main"),
                0.92,
                1.08,
            )
        for other in ("main", "v055"):
            if other in t and "branch" in t:
                checks[f"warm branch/{other}"] = (
                    ratio(deck, "warm_s", "branch", other),
                    0.95,
                    1.05,
                )
        g5[deck] = {
            k: dict(ratio=v, lo=lo, hi=hi, hit=lo <= v <= hi)
            for k, (v, lo, hi) in checks.items()
        }
        ref = "main" if deck in CATALOG else ("v055" if deck in MID else None)
        if ref and ref in t and "branch" in t:
            zb = {json.dumps(z) for z in t["branch"]["z"]}
            zr = {json.dumps(z) for z in t[ref]["z"]}
            g1b[deck] = dict(
                against=ref,
                bit_identical=zb == zr,
                branch_z=sorted(zb),
                ref_z=sorted(zr),
            )
        preds = {}
        for name, centre, half in PREDICT.get(deck, []):
            a, b = name.split("/")
            if a in t and b in t:
                v = ratio(deck, "cold_s", a, b)
                preds[f"cold {name}"] = dict(
                    ratio=v,
                    band=[centre - half, centre + half],
                    hit=abs(v - centre) <= half,
                )
        predictions[deck] = preds

    g5_verdict = (
        "PASS"
        if not fails and all(c["hit"] for d in g5.values() for c in d.values())
        else "FAIL"
    )
    g1b_verdict = (
        "PASS"
        if not fails and all(v["bit_identical"] for v in g1b.values()) and len(g1b) == 6
        else "FAIL"
    )
    out = dict(
        G5=g5_verdict,
        G1b=g1b_verdict,
        failures=fails,
        table={
            d: {t: {k: v for k, v in r.items() if k != "z"} for t, r in tr.items()}
            for d, tr in table.items()
        },
        g5=g5,
        g1b=g1b,
        predictions=predictions,
    )
    (HERE / f"t5_table{TAG}.json").write_text(json.dumps(out, indent=1))
    for deck in table:
        cells = "  ".join(
            f"{tree} cold {r['cold_s']:.2f}s warm {r['warm_s']:.3f}s rss {r['maxrss_mb']:.0f}"
            for tree, r in table[deck].items()
        )
        print(f"{deck:14s} {cells}")
        for k, c in g5[deck].items():
            print(
                f"    G5 {k:20s} {c['ratio']:.3f}  [{c['lo']}, {c['hi']}]  {'hit' if c['hit'] else 'MISS'}"
            )
        for k, p in predictions[deck].items():
            print(
                f"    prediction {k:20s} {p['ratio']:.3f}  {p['band']}  {'hit' if p['hit'] else 'MISS'}"
            )
        if deck in g1b:
            print(
                f"    G1b vs {g1b[deck]['against']}: {'bit-identical' if g1b[deck]['bit_identical'] else 'DIFFERS'}"
            )
    for f in fails:
        print("FAIL", f)
    print(f"G5: {g5_verdict}   G1b: {g1b_verdict}")


if __name__ == "__main__":
    main()
