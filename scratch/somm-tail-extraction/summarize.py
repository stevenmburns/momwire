"""Tables for REPORT.md from results.jsonl and extra.jsonl.

python summarize.py > tables.md
"""

from __future__ import annotations

import json
from collections import defaultdict

rows = [json.loads(line) for line in open("results.jsonl")]
extra = [json.loads(line) for line in open("extra.jsonl")]

METHODS = ["base", "wa", "ext1", "ext2", "ext3", "ext1+wa", "ext2+wa", "ext3+wa"]
GRAZE = (0.05, 0.1, 0.2, 0.5, 1.0)
STEEP = (10.0, 45.0, 80.0)


def table_per_node(soil):
    print(
        f"\n### {soil}: tail panels / integrand points / worst component vs reference\n"
    )
    print("| R1/λ_m | θ° | " + " | ".join(METHODS) + " |")
    print("|---|---|" + "---|" * len(METHODS))
    by = defaultdict(dict)
    for r in rows:
        if r["soil"] == soil and r["method"] != "REF":
            by[(r["r1_wl"], r["theta_deg"])][r["method"]] = r
    for (r1, th), d in sorted(by.items()):
        cells = []
        for m in METHODS:
            x = d[m]
            cells.append(f"{x['tail_panels']} / {x['points']} / {x['vs_ref'][0]:.0e}")
        print(f"| {r1} | {th} | " + " | ".join(cells) + " |")


def aggregate():
    print("\n### Aggregate over all three soils and R1 ∈ {0.2, 1, 2} λ_m\n")
    print(
        "| method | grazing tail panels (max) | grazing points (mean) | steep points (mean) "
        "| grazing/steep points | worst comp. vs ref | worst vec-norm vs ref | numpy wall, grazing mean |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for m in METHODS:
        g = [r for r in rows if r["method"] == m and r["theta_deg"] in GRAZE]
        s = [r for r in rows if r["method"] == m and r["theta_deg"] in STEEP]
        gp = sum(r["points"] for r in g) / len(g)
        sp = sum(r["points"] for r in s) / len(s)
        allr = g + s
        print(
            f"| {m} | {max(r['tail_panels'] for r in g)} | {gp:.0f} | {sp:.0f} | {gp / sp:.2f} "
            f"| {max(r['vs_ref'][0] for r in allr):.1e} | {max(r['vs_ref'][1] for r in allr):.1e} "
            f"| {1e3 * sum(r['wall'] for r in g) / len(g):.1f} ms |"
        )


def refs():
    rr = [r for r in rows if r["method"] == "REF"]
    w = max(rr, key=lambda r: r["ref_vs_ref2"][1])
    print(
        f"\nReference self-check (base @1e-12 vs ext3 @1e-12, independent tails): worst "
        f"componentwise {max(r['ref_vs_ref2'][0] for r in rr):.1e}, worst vec-norm "
        f"{w['ref_vs_ref2'][1]:.1e} at {w['soil']} R1={w['r1_wl']} θ={w['theta_deg']}; "
        f"over grazing nodes only: componentwise "
        f"{max(r['ref_vs_ref2'][0] for r in rr if r['theta_deg'] in GRAZE):.1e}."
    )


def b_scan():
    b = [r for r in extra if r.get("exp") == "b_scan"]
    print("\n### b scan (ext3, ext3+wa): b = b_fac / a\n")
    print("| soil | R1 | θ° | method | b_fac=0.5 | 1 | 2 | 4 |")
    print("|---|---|---|---|---|---|---|---|")
    key = defaultdict(dict)
    for r in b:
        key[(r["soil"], r["r1_wl"], r["theta_deg"], r["method"])][r["b_fac"]] = r
    for k, d in sorted(key.items()):
        cells = [
            f"{d[f]['tail_panels']} / {d[f]['vs_ref'][0]:.0e}"
            for f in (0.5, 1.0, 2.0, 4.0)
        ]
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | " + " | ".join(cells) + " |")


def under_floor():
    u = [r for r in extra if r.get("exp") == "under_floor"]
    print("\n### Under the floor (current floor 0.016667°), reference = ext3 @1e-12\n")
    print(
        "| soil | R1 | θ° | ref panels | base panels (conv) | base vs ref | wa | ext1+wa | ext3+wa |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for r in u:
        cells = [
            f"{r[m]['tail_panels']} / {r[m]['vs_ref'][0]:.0e}"
            for m in ("wa", "ext1+wa", "ext3+wa")
        ]
        print(
            f"| {r['soil']} | {r['r1_wl']} | {r['theta_deg']} | {r['ref_panels']} "
            f"| {r['base_panels']} ({'yes' if r['base_conv'] else 'NO'}) "
            f"| {r['base_vs_ref'][0]:.0e} | " + " | ".join(cells) + " |"
        )


def sweep():
    s = [r for r in extra if r.get("exp") == "sweep"]
    if not s:
        return
    w = max(s, key=lambda r: r["wa_vs_ref"][0])
    wb = max(s, key=lambda r: r["base_vs_ref"][0])
    print(
        f"\n### Robustness sweep: {len(s)} nodes (3 soils × R1 ∈ {{0.02, 0.1, 0.5, 1.5, 3}} λ_m × "
        f"24 θ from 0.03° to 60°)\n"
    )
    print(
        f"- WA worst componentwise vs ref: {w['wa_vs_ref'][0]:.1e} "
        f"({w['soil']} R1={w['r1_wl']} θ={w['theta_deg']:.3g}); worst vec-norm "
        f"{max(r['wa_vs_ref'][1] for r in s):.1e}."
    )
    print(
        f"- base worst componentwise vs ref: {wb['base_vs_ref'][0]:.1e} "
        f"({wb['soil']} R1={wb['r1_wl']} θ={wb['theta_deg']:.3g}); worst vec-norm "
        f"{max(r['base_vs_ref'][1] for r in s):.1e}."
    )
    over = [r for r in s if r["wa_vs_ref"][0] > 1e-9]
    print(f"- WA nodes over 1e-9 componentwise: {len(over)} of {len(s)}.")
    for r in over[:10]:
        print(
            f"  - {r['soil']} R1={r['r1_wl']} θ={r['theta_deg']:.3g}: wa {r['wa_vs_ref'][0]:.1e}, "
            f"base {r['base_vs_ref'][0]:.1e}, panels {r['wa_panels']} vs {r['base_panels']}"
        )
    print(
        f"- tail panels: base max {max(r['base_panels'] for r in s)}, WA max "
        f"{max(r['wa_panels'] for r in s)}; points: base total "
        f"{sum(r['base_points'] for r in s)}, WA total {sum(r['wa_points'] for r in s)}."
    )


aggregate()
refs()
for soil in ("dan", "salt1.8", "fresh28"):
    table_per_node(soil)
b_scan()
under_floor()
sweep()
