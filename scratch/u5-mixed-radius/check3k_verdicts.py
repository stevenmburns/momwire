"""Read check 3' against its registered predictions (DERIVATION §8.7)."""

import json
from pathlib import Path

D = Path(__file__).resolve().parent
C = complex
c2 = json.loads((D / "check3k_check2.json").read_text())["rows"]
b = json.loads((D / "check3k_b.json").read_text())["rows"]


def zk(r):
    return C(r["vc_keep"]["z"])


def verdict(ok):
    return "HIT" if ok else "MISSED"


print("== C3'.0 (VC_keep = split at equal radii, <= 1e-4 rel)")
eq = [r for r in c2 if r["a_A"] == r["a_B"]]
m2 = max(r["vc_keep_rel_to_split"] for r in eq)
bctl = [r for r in b if r["config"] == "control"]
mb = max(abs(zk(r) - C(r["z"])) / abs(C(r["z"])) for r in bctl)
print(f"check2 rod {m2:.2e}; b control {mb:.2e} -> {verdict(max(m2, mb) <= 1e-4)}")

rows2 = {}
for r in c2:
    if r["a_A"] != r["a_B"]:
        rows2.setdefault((round(r["a_A"] / r["a_B"]), r["refine"]), {})[
            r["spelling"]
        ] = r
rowsb = {}
for r in b:
    rowsb.setdefault(r["config"], {})[r["spelling"]] = r
mixed = [(f"check2 ratio {k[0]} r{k[1]}", v) for k, v in sorted(rows2.items())]
mixed += [(f"b {k}", v) for k, v in rowsb.items() if k != "control"]

print("== P3'.2 (|Z(node_X) - Z(VC_keep(node_Y))| <= 0.1 Ohm) and P3'.4")
p2 = p4 = True
for name, rs in mixed:
    worst = max(
        abs(C(rs[x]["z"]) - zk(rs[y]))
        for x in ("node_A", "node_B")
        for y in ("node_A", "node_B")
    )
    g_keep = abs(zk(rs["node_A"]) - zk(rs["node_B"]))
    g_split = abs(C(rs["node_A"]["z"]) - C(rs["node_B"]["z"]))
    p2 &= worst <= 0.1
    p4 &= g_keep < g_split
    print(
        f"{name:18s} worst {worst:.4f} | gauge keep {g_keep * 1e3:.3f} mOhm "
        f"split {g_split * 1e3:.3f} mOhm | VC_keep(node_B) {zk(rs['node_B']):.6g} "
        f"obs split {C(rs['obs']['z']):.6g} VC_keep(obs) {zk(rs['obs']):.6g}"
    )
print(f"P3'.2 {verdict(p2)}; P3'.4 {verdict(p4)}")

print("== P3'.3 (the jump is the shift, merged form)")


def jump(rs):
    return (zk(rs["obs"]) - zk(rs["node_B"])).real


ok3 = True
for ref in (1, 2):
    q = jump(rows2[(4, ref)]) / jump(rows2[(2, ref)])
    ok3 &= 1.85 <= q <= 2.15
    print(
        f"check2 r{ref}: {jump(rows2[(2, ref)]):.4f} / {jump(rows2[(4, ref)]):.4f} -> {q:.4f}"
    )
qb = jump(rowsb["ratio05"]) / jump(rowsb["ratio2"])
ok3 &= -1.15 <= qb <= -0.85
print(
    f"b: ratio2 {jump(rowsb['ratio2']):.3f} ratio4 {jump(rowsb['ratio4']):.3f} ratio05 {jump(rowsb['ratio05']):.3f} -> {qb:.4f}"
)
print(f"P3'.3 {verdict(ok3)}")

print("== P3'.6 (slope)")
ok6a = ok6b = True
for (ratio, ref), rs in sorted(rows2.items()):
    s_node = rs["node_B"]["vc_keep"]["slope_vs_agard"]
    s_obs = rs["obs"]["vc_keep"]["slope_vs_agard"]
    ok6a &= s_node > 0.05
    ok6b &= s_obs <= 1e-3
    print(
        f"ratio {ratio} r{ref}: VC_keep(node_B) {s_node:.3e} ratio "
        f"{rs['node_B']['vc_keep']['slope_ratio']} | VC_keep(obs) {s_obs:.3e}"
    )
print(f"P3'.6 derived part {verdict(ok6a)}; informed part {verdict(ok6b)}")

print("== P3'.7 (b rod vs NEC-5, <= 0.5 Ohm)")
ctl = rowsb["control"]["obs"]
d_ctl = C(ctl["nec5"]) - C(ctl["z"])
ok7 = True
for cfg in ("ratio2", "ratio4", "ratio05"):
    rs = rowsb[cfg]
    d = C(rs["node_B"]["nec5"]) - zk(rs["node_B"])
    ok7 &= abs(d - d_ctl) <= 0.5
    print(
        f"{cfg}: dZ(VC_keep(node_B)) {d:.4f} minus control {d_ctl:.4f} -> {abs(d - d_ctl):.3f}"
    )
print(f"P3'.7 {verdict(ok7)}")
