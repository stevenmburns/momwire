"""Read check 4 against its registered predictions (DERIVATION §9.5)."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIXED = ("ratio2", "ratio4", "ratio05")
REFINES = (1, 2, 4, 8)
DS = (0.2, 0.1, 0.05, 0.025)

rows = json.loads((HERE / "check4.json").read_text())["rows"]
R = {(row["config"], row["refine"], row["spelling"]): row for row in rows}


def val(cfg, r, key, which="vc_keep"):
    return complex(R[(cfg, r, "node_B")][which][key])


def dev(cfg, r, key, which="vc_keep"):
    return abs(val(cfg, r, key, which) - 1.0)


def verdict(ok):
    return "HIT" if ok else "MISSED"


print("== P4.1 (node slope: mixed deviation rises r=1 -> 8; control <= 1e-3)")
ok = True
for cfg in MIXED:
    d = [dev(cfg, r, "node") for r in REFINES]
    good = d[-1] > d[0]
    ok &= good
    print(f"{cfg:8s} |node*eps-1| r=1,2,4,8: " + " ".join(f"{x:.3e}" for x in d))
d = [dev("control", r, "node") for r in REFINES]
ok &= max(d) <= 1e-3
print("control  |node*eps-1| r=1,2,4,8: " + " ".join(f"{x:.3e}" for x in d))
print(f"P4.1 {verdict(ok)}")

print("== P4.2 (d = 0.1 m: converging, and within 5 % of the control at r = 8)")
ok = True
for key in ("p0.1", "s0.1"):
    x_ctl = val("control", 8, key)
    for cfg in MIXED:
        x2, x4, x8 = (val(cfg, r, key) for r in (2, 4, 8))
        conv = abs(x8 - x4) < abs(x4 - x2)
        near = abs(x8 - x_ctl) <= 0.05 * abs(x_ctl)
        ok &= conv and near
        print(
            f"{key} {cfg:8s} steps {abs(x4 - x2):.3e} -> {abs(x8 - x4):.3e} "
            f"{'conv' if conv else 'NOT conv'}; r=8 {x8:.5f} control {x_ctl:.5f} "
            f"rel {abs(x8 - x_ctl) / abs(x_ctl):.3e}"
        )
print(f"P4.2 {verdict(ok)}")

print("== P4.3 (r = 8: |R_p(d)*eps - 1| falls as d shrinks)")
ok = True
for cfg in ("control", *MIXED):
    d = [dev(cfg, 8, f"p{x:g}") for x in DS]
    good = all(a > b for a, b in zip(d, d[1:]))
    ok &= good
    print(f"{cfg:8s} d=0.2,0.1,0.05,0.025: " + " ".join(f"{x:.3e}" for x in d))
print(f"P4.3 {verdict(ok)}")

print("== P4.4 (transposed solve's d = 0.1 m readouts within 1e-3 relative)")
worst = 0.0
for cfg in ("control", *MIXED):
    for r in REFINES:
        for key in ("p0.1", "s0.1"):
            a, b = val(cfg, r, key), val(cfg, r, key, "transposed")
            worst = max(worst, abs(b - a) / abs(a))
print(f"worst {worst:.3e} -> {verdict(worst <= 1e-3)}")

print("== P4.5 (r = 2: rise thinning raises R more than top thinning; obs reverses)")


def z_keep(cfg):
    return complex(R[(cfg, 2, "node_B")]["vc_keep"]["z"])


def z_obs(cfg):
    return complex(R[(cfg, 2, "obs")]["z"])


rise_k = (z_keep("ratio2") - z_keep("control")).real
top_k = (z_keep("ratio05") - z_keep("control")).real
rise_o = (z_obs("ratio2") - z_obs("control")).real
top_o = (z_obs("ratio05") - z_obs("control")).real
print(f"VC_keep(node_B): rise {rise_k:+.3f} top {top_k:+.3f}")
print(f"obs split:       rise {rise_o:+.3f} top {top_o:+.3f}")
print(f"P4.5 {verdict(rise_k > top_k and rise_o < top_o)}")
