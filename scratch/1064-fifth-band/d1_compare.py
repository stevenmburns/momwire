"""momwire#1064 D1, a diagnostic of G3's failure: is the avx2-vs-sse2 spread the
grid FILL's, and does it predate this change?

  python d1_compare.py

Reads G3's branch captures (`g3_avx2.json`, `g3_sse2.json`) and D1's captures of
the same script on main (`d1_main_g3_avx2.json`, `d1_main_g3_sse2.json`). Per
point it reports the cross-variant spread |C++ avx2 - C++ sse2| / |C++ sse2| on
each tree, their ratio, and the worst same-process C++-vs-numpy difference on
main. As registered:

  * main's cross-variant spread is within 2x of the branch's at every point
    (the fill's contour is unchanged by momwire#1064, and so are the nodes a
    query at these angles reads);
  * main's same-process C++-vs-numpy difference is <= 1e-15.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def spread(avx2, sse2):
    out, same = {}, 0.0
    for med, rows in avx2["media"].items():
        rs = {(r["zone"], r["theta_deg"]): r for r in sse2["media"][med]}
        for r in rows:
            key = (med, r["zone"], r["theta_deg"])
            s = rs[(r["zone"], r["theta_deg"])]
            ca, cs = complex(*r["cpp"]), complex(*s["cpp"])
            na, ns = complex(*r["numpy"]), complex(*s["numpy"])
            out[key] = abs(ca - cs) / abs(cs)
            same = max(same, abs(ca - na) / abs(na), abs(cs - ns) / abs(ns))
    return out, same


def main():
    def load(name):
        return json.loads((HERE / name).read_text())

    br, br_same = spread(load("g3_avx2.json"), load("g3_sse2.json"))
    mn, mn_same = spread(load("d1_main_g3_avx2.json"), load("d1_main_g3_sse2.json"))
    fails, rows, worst_ratio = [], [], 1.0
    for key, b in br.items():
        m = mn.get(key)
        if m is None:
            fails.append(f"{key}: missing from main's capture")
            continue
        lo, hi = sorted((b, m))
        ratio = hi / lo if lo > 0 else (1.0 if hi == 0 else float("inf"))
        worst_ratio = max(worst_ratio, ratio)
        rows.append(dict(point=list(key), branch=b, main=m, ratio=ratio))
    within = worst_ratio <= 2.0
    out = dict(
        diagnostic="D1",
        branch_worst_spread=max(br.values()),
        main_worst_spread=max(mn.values()),
        worst_ratio=worst_ratio,
        spread_within_2x=within,
        main_same_process_worst=mn_same,
        branch_same_process_worst=br_same,
        prediction_hit=within and mn_same <= 1e-15 and not fails,
        failures=fails,
        rows=rows,
    )
    (HERE / "d1_compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in rows:
        print(
            f"{r['point'][0]:7s} {r['point'][1]:5s} "
            f"th={r['point'][2]:<22s} branch {r['branch']:.2e} main {r['main']:.2e} "
            f"x{r['ratio']:.2f}"
        )


if __name__ == "__main__":
    main()
