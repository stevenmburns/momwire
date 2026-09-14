"""U9 Amendment 8c reading, run only after checks 1-5 have passed. Per
separation: each engine's far-x-3 step, momwire - NEC-5 on Z11 and Z12 at r1 and
far x 3, the gate, the non-vacuity guard, and PE8a-f as Amendment 8 registered
them (PE8c read at far x 3, as 8c says).

  python e6_table.py
"""

from __future__ import annotations

import json

from e5_checks import HERE, SEPARATIONS, cplx, load, one

REL = 0.02  # Amendment 8's formulation allowance, relative to |Z_nec5|
QS = ("z11", "z12")


def pair(v):
    return [v.real, v.imag]


def cz(g, q):
    return complex(*g[q]["diff"])


def main():
    rows = load()
    errors = []
    z = {}
    for d in SEPARATIONS:
        keys = [
            (f"two_node_d{d}_{rung}", eng, "cross", rung)
            for rung in ("r1", "far3")
            for eng in ("momwire", "nec5")
        ]
        keys.append((f"two_node_d{d}_r1", "momwire", "same", "r1"))
        for stem, eng, corner, rung in keys:
            row = one(rows, (stem, eng, corner), errors)
            if row is not None:
                zz = [cplx(v) for v in row["z"]]
                z[(d, rung, eng, corner)] = dict(z11=zz[0], z12=zz[1])
    table = {}
    for d in SEPARATIONS:
        steps = {}
        for eng in ("momwire", "nec5"):
            a, b = z.get((d, "r1", eng, "cross")), z.get((d, "far3", eng, "cross"))
            if a and b:
                steps[eng] = {q: abs(b[q] - a[q]) for q in QS}
        gates = {}
        for rung in ("r1", "far3"):
            m = z.get((d, rung, "momwire", "cross"))
            n = z.get((d, rung, "nec5", "cross"))
            if not (m and n and len(steps) == 2):
                continue
            g = {}
            for q in QS:
                diff = m[q] - n[q]
                bar = steps["momwire"][q] + steps["nec5"][q] + REL * abs(n[q])
                g[q] = dict(
                    momwire=pair(m[q]),
                    nec5=pair(n[q]),
                    diff=pair(diff),
                    abs_diff=abs(diff),
                    bar=bar,
                    hit=abs(diff) <= bar,
                    rel=abs(diff) / abs(n[q]),
                )
            gates[rung] = g
        t = dict(steps=steps, gates=gates)
        c, s = z.get((d, "r1", "momwire", "cross")), z.get((d, "r1", "momwire", "same"))
        if c and s and "r1" in gates:
            effect = abs(c["z12"] - s["z12"])
            bar = gates["r1"]["z12"]["bar"]
            t["guard"] = dict(
                corner_effect_z12=effect,
                gate_bar_z12_r1=bar,
                sees_corner=effect > 2 * bar,
            )
        table[d] = t
        print(f"d = {d} m:", json.dumps(t))
    pe = {}
    if not errors:
        far = [table[d]["gates"]["far3"] for d in SEPARATIONS]
        effects = [table[d]["guard"]["corner_effect_z12"] for d in SEPARATIONS]
        pe = dict(
            PE8a=all(
                0.05 <= g["z12"]["abs_diff"] <= 1.5 and g["z12"]["rel"] <= REL
                for g in far
            ),
            PE8b=all(0.3 <= g["z11"]["abs_diff"] <= 4.0 for g in far),
            PE8c_z11=all(cz(g, "z11").real < 0 and cz(g, "z11").imag > 0 for g in far),
            PE8c_z12_low_confidence=all(cz(g, "z12").real < 0 for g in far),
            PE8d=all(
                table[d]["gates"][rung][q]["hit"]
                for d in SEPARATIONS
                for rung in ("r1", "far3")
                for q in QS
            ),
            PE8e=all(0.2 <= e <= 10.0 for e in effects)
            and all(effects[i] > effects[i + 1] for i in range(len(effects) - 1))
            and all(table[d]["guard"]["sees_corner"] for d in SEPARATIONS),
            PE8f=all(
                table[d]["steps"]["nec5"]["z12"] >= table[d]["steps"]["momwire"]["z12"]
                for d in SEPARATIONS
            ),
        )
    for e in errors:
        print("ERROR", e)
    print("predictions:", json.dumps(pe))
    (HERE / "e6_table.json").write_text(
        json.dumps(dict(errors=errors, table=table, predictions=pe), indent=1)
    )


if __name__ == "__main__":
    main()
