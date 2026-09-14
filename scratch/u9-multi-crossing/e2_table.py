"""U9 Amendment 8 reading: from e1's JSONL, per separation, the checks, each
engine's far-x-3 step, the momwire - NEC-5 difference on Z12 and Z11 at each
rung, the gate, and the non-vacuity guard (the cross-node corner's own effect on
momwire's Z12).

  python e2_table.py e1_two_node.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REL = 0.02  # Amendment 8's formulation allowance, relative to |Z_nec5|
RECIP_NEC5 = 1e-2
RECIP_MW = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    args = ap.parse_args()
    rows = [
        json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()
    ]
    z, recip, errors = {}, {}, []
    for r in rows:
        key = (
            r["d"],
            r["rung"],
            r["engine"],
            r.get("corner", "cross") if r["engine"] == "momwire" else "cross",
        )
        if "error" in r:
            errors.append((key, r["error"][:200]))
            continue
        zz = [complex(*v) for v in r["z"]]
        z[key] = dict(z11=zz[0], z12=zz[1], z21=zz[2], z22=zz[3])
        recip[key] = r["y_reciprocity"]
    out = {"errors": errors, "rows": {}}
    for d in sorted({k[0] for k in z}):
        row = {}
        checks = {}
        for (dd, rung, eng, corner), v in recip.items():
            if dd != d:
                continue
            bar = RECIP_NEC5 if eng == "nec5" else RECIP_MW
            checks[f"{eng}/{rung}/{corner}"] = dict(reciprocity=v, bar=bar, ok=v <= bar)
        row["checks"] = checks
        steps = {}
        for eng in ("momwire", "nec5"):
            a, b = (d, "r1", eng, "cross"), (d, "far3", eng, "cross")
            if a in z and b in z:
                steps[eng] = {q: abs(z[b][q] - z[a][q]) for q in ("z11", "z12")}
        row["steps"] = steps
        gates = {}
        for rung in ("r1", "far3"):
            m, n = (d, rung, "momwire", "cross"), (d, rung, "nec5", "cross")
            if m in z and n in z and "momwire" in steps and "nec5" in steps:
                g = {}
                for q in ("z11", "z12"):
                    diff = z[m][q] - z[n][q]
                    bar = steps["momwire"][q] + steps["nec5"][q] + REL * abs(z[n][q])
                    g[q] = dict(
                        momwire=[z[m][q].real, z[m][q].imag],
                        nec5=[z[n][q].real, z[n][q].imag],
                        diff=[diff.real, diff.imag],
                        abs_diff=abs(diff),
                        bar=bar,
                        hit=abs(diff) <= bar,
                        rel=abs(diff) / abs(z[n][q]),
                    )
                g["polarity_ok"] = (z[m]["z12"].real > 0) == (z[n]["z12"].real > 0)
                gates[rung] = g
        row["gates"] = gates
        c, s = (d, "r1", "momwire", "cross"), (d, "r1", "momwire", "same")
        if c in z and s in z and "r1" in gates:
            corner = abs(z[c]["z12"] - z[s]["z12"])
            row["non_vacuity"] = dict(
                corner_effect_z12=corner,
                gate_bar_z12_r1=gates["r1"]["z12"]["bar"],
                sees_corner=gates["r1"]["z12"]["bar"] < 0.5 * corner,
            )
        out["rows"][str(d)] = row
        print(f"d = {d:g} m:", json.dumps(row, default=str)[:1500])
    for e in errors:
        print("ERROR", e)
    args.jsonl.with_suffix(".table.json").write_text(
        json.dumps(out, indent=1, default=str)
    )


if __name__ == "__main__":
    main()
