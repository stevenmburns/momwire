"""U9 (d) reading (PLAN.md Amendment 3): from d2_lpda_solve.jsonl, each
engine's feed Z per (deck, rung), the change Delta_e(r) = Z_full - Z_one, the
full deck's ladder step s_e, (i)'s gate, and (ii)'s trigger, plus the cost
projection rule for momwire's full far x 3 rung.

  python d2_table.py d2_lpda_solve.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAR3_FULL_SEGS = 8014


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument(
        "--one-sha",
        default=None,
        help="Amendment 6: read only one-node rows written on this deck hash",
    )
    args = ap.parse_args()
    rows = [
        json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()
    ]
    z, cost = {}, {}
    for r in rows:
        key = (r["engine"], r["deck"], r["rung"])
        if args.one_sha and r["deck"] == "one" and r.get("deck_sha256") != args.one_sha:
            continue
        if "error" in r:
            print("ERROR", key, r["error"][:200])
            continue
        z[key] = complex(*r["z"])
        cost[key] = (
            r["seconds"],
            max(r["maxrss_mb"], r.get("maxrss_children_mb", 0.0)),
            r.get("segments"),
        )
    for key in sorted(z):
        s, mb, segs = cost[key]
        print(f"{key}: Z = {z[key]:.4f}  {s:.1f} s  {mb:.0f} MB  {segs} segs")

    rec = dict(z={"/".join(k): [v.real, v.imag] for k, v in z.items()})
    mk = ("momwire", "full", "r1")
    if mk in cost and cost[mk][2]:
        s, mb, segs = cost[mk]
        f = FAR3_FULL_SEGS / segs
        proj = dict(peak_mb=mb * f * f, seconds_upper=s * f**3, factor=f)
        proj["rule1_passes"] = (
            proj["peak_mb"] <= 20_000 and proj["seconds_upper"] <= 6 * 3600
        )
        rec["momwire_full_far3_projection"] = proj
        print("momwire full far3 projection:", proj)

    for e in ("momwire", "nec5"):
        d = {
            r: z[(e, "full", r)] - z[(e, "one", r)]
            for r in ("r1", "far3")
            if (e, "full", r) in z and (e, "one", r) in z
        }
        step = (
            abs(z[(e, "full", "far3")] - z[(e, "full", "r1")])
            if all((e, "full", r) in z for r in ("r1", "far3"))
            else None
        )
        rec[e] = dict(
            delta={r: [v.real, v.imag] for r, v in d.items()},
            abs_delta={r: abs(v) for r, v in d.items()},
            step=step,
            ii_trigger=None
            if step is None or "far3" not in d
            else abs(d["far3"]) <= 2 * step,
        )
        print(e, rec[e])
    if all(
        rec.get(e, {}).get("step") is not None and "far3" in rec[e]["delta"]
        for e in ("momwire", "nec5")
    ):
        dm = complex(*rec["momwire"]["delta"]["far3"])
        dn = complex(*rec["nec5"]["delta"]["far3"])
        lhs = abs(dm - dn)
        rhs = 0.1 * abs(dn) + rec["momwire"]["step"] + rec["nec5"]["step"]
        rec["gate_i"] = dict(lhs=lhs, rhs=rhs, hit=lhs <= rhs)
        print("gate (i):", rec["gate_i"])
    # Amendment 6 (i-1), the WEAKER gate: refine-1 changes, with NEC-5's own
    # full-deck r1 -> far x 3 step as the resolution for BOTH engines.
    if all(
        k in z
        for k in (
            ("momwire", "full", "r1"),
            ("momwire", "one", "r1"),
            ("nec5", "full", "r1"),
            ("nec5", "one", "r1"),
            ("nec5", "full", "far3"),
        )
    ):
        s_n = abs(z[("nec5", "full", "far3")] - z[("nec5", "full", "r1")])
        dm = z[("momwire", "full", "r1")] - z[("momwire", "one", "r1")]
        dn = z[("nec5", "full", "r1")] - z[("nec5", "one", "r1")]
        lhs = abs(dm - dn)
        rhs = 0.1 * abs(dn) + 2.0 * s_n
        rec["gate_i1"] = dict(
            s_nec5=s_n,
            delta_momwire_r1=[dm.real, dm.imag],
            delta_nec5_r1=[dn.real, dn.imag],
            abs_delta_momwire_r1=abs(dm),
            abs_delta_nec5_r1=abs(dn),
            lhs=lhs,
            rhs=rhs,
            hit=lhs <= rhs,
            ii_trigger=abs(dm) <= 2 * s_n or abs(dn) <= 2 * s_n,
        )
        print("gate (i-1):", rec["gate_i1"])
    args.jsonl.with_suffix(".table.json").write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
