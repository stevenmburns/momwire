"""momwire#1064 G1a, G2a, G2b and G2c, read from `g12_grid.py`'s three captures.

  python g12_compare.py

  G1a  branch against 0.55.0 on the OLD set (theta >= 0.05 deg), numpy surfaces
       and the C++-dispatched kernel: every value bit-identical.
  G2a  branch against main on the FLOOR set (theta < 0.05 deg): max relative
       deviation <= 4.7e-4 (reported, with how many points are bit-identical).
  G2b  the floor band against the direct surfaces (branch --direct): <= 4.7e-4.
  G2c  the low band, the same way: <= 4.7e-4.

Recorded, not gated: branch against main on the OLD set, split at 0.1 deg (U9's
single band had a different th0, so [0.05, 0.1) may differ in low bits).
A point missing from any capture is a failure, never a skip.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAR = 4.7e-4


def load(tree):
    return json.loads((HERE / f"g12_{tree}.json").read_text())


def keyed(points):
    return {(p["r1_lam"], p["theta_deg"]): p for p in points}


def rel(a, b):
    da = complex(*a) - complex(*b)
    return abs(da) / max(abs(complex(*b)), 1e-300)


def worst_rel(pa, pb):
    vals = [rel(pa["surf"][k], pb["surf"][k]) for k in pa["surf"]]
    vals.append(rel(pa["kernel"], pb["kernel"]))
    return max(vals)


def main():
    v, m, b = load("v055"), load("main"), load("branch")
    fails = []
    g1a = dict(points=0, not_bitwise=0, examples=[])
    old_vs_main = {
        "below_0.1": dict(points=0, not_bitwise=0, max_rel=0.0),
        "at_or_above_0.1": dict(points=0, not_bitwise=0, max_rel=0.0),
    }
    g2a = dict(points=0, bitwise=0, max_rel=0.0, where=None)
    g2b = dict(worst=0.0, where=None)
    g2c = dict(worst=0.0, where=None)
    for med, bm in b["media"].items():
        if med not in v["media"] or med not in m["media"]:
            fails.append(f"{med}: missing from a capture")
            continue
        bo, vo, mo = (keyed(t["media"][med]["old"]) for t in (b, v, m))
        if not (set(bo) == set(vo) == set(mo)):
            fails.append(f"{med}: the OLD sets differ between captures")
        for k, pb in bo.items():
            g1a["points"] += 1
            pv = vo.get(k)
            if pv is None or pb["surf"] != pv["surf"] or pb["kernel"] != pv["kernel"]:
                g1a["not_bitwise"] += 1
                if len(g1a["examples"]) < 5:
                    g1a["examples"].append(
                        [med, *k, None if pv is None else worst_rel(pb, pv)]
                    )
            pm = mo.get(k)
            if pm is not None:
                # `g12_grid.py` wrote cell points as `repr(np.float64(...))`;
                # the key strings match across captures, only this parse needs
                # the number out of them.
                th = float(str(k[1]).removeprefix("np.float64(").removesuffix(")"))
                side = "below_0.1" if th < 0.1 else "at_or_above_0.1"
                s = old_vs_main[side]
                s["points"] += 1
                if pb["surf"] != pm["surf"] or pb["kernel"] != pm["kernel"]:
                    s["not_bitwise"] += 1
                    s["max_rel"] = max(s["max_rel"], worst_rel(pb, pm))
        bf = keyed(bm.get("floor", []))
        mf = keyed(m["media"][med].get("floor", []))
        if not bf or set(bf) != set(mf):
            fails.append(f"{med}: the FLOOR sets are missing or differ")
        for k, pb in bf.items():
            pm = mf.get(k)
            if pm is None:
                continue
            g2a["points"] += 1
            r = worst_rel(pb, pm)
            if r == 0.0:
                g2a["bitwise"] += 1
            if r > g2a["max_rel"]:
                g2a["max_rel"], g2a["where"] = r, [med, *k]
        for name, acc in (("direct_floor_band", g2b), ("direct_low_band", g2c)):
            if name not in bm:
                fails.append(f"{med}: {name} missing (branch capture needs --direct)")
                continue
            w, where = bm[name]
            if w > acc["worst"]:
                acc["worst"], acc["where"] = w, [med, where]
    verdicts = dict(
        G1a="PASS" if g1a["points"] and g1a["not_bitwise"] == 0 else "FAIL",
        G2a="PASS" if g2a["points"] and g2a["max_rel"] <= BAR else "FAIL",
        G2b="PASS" if g2b["worst"] <= BAR and g2b["where"] else "FAIL",
        G2c="PASS" if g2c["worst"] <= BAR and g2c["where"] else "FAIL",
    )
    if fails:
        verdicts = {k: "FAIL" for k in verdicts}
    out = dict(
        verdicts=verdicts,
        failures=fails,
        G1a=g1a,
        G2a=g2a,
        G2b=g2b,
        G2c=g2c,
        recorded_branch_vs_main_old_set=old_vs_main,
        trees={t["tree"]: t["momwire"] for t in (v, m, b)},
    )
    (HERE / "g12_compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
