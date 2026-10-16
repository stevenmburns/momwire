import json
import sys

import numpy as np

S, b, c = sys.argv[1], sys.argv[2], sys.argv[3]


def load(p):
    return {(r["name"], r["basis"]): r for r in map(json.loads, open(f"{S}/{p}.jsonl"))}


B, C = load(b), load(c)
rows = []
moved = 0
nonpin = 0
errs = 0
for k, rb in B.items():
    rc = C[k]
    if rb["err"] != rc["err"]:
        errs += 1
    if rb["sha"] != rc["sha"]:
        moved += 1
        for x, y in zip(rb["text"].splitlines(), rc["text"].splitlines()):
            if x != y and "1.0000E+10" not in x:
                nonpin += 1
                print("  NON-PIN", k, x, "|", y)
    for zb, zc in zip(rb["Z"], rc["Z"]):
        if not zb or not zc:
            continue
        for p, q in zip(zb, zc):
            a = complex(*p)
            d = complex(*q)
            if abs(a) < 1e-6:
                continue
            rows.append((abs(a - d) / abs(a), k, a))
rows.sort(key=lambda t: -t[0])
A = np.load(f"{S}/{b}.npz")
D = np.load(f"{S}/{c}.npz")
ws = []
same = 0
for k in set(A.files) & set(D.files):
    x, y = A[k], D[k]
    sc = np.max(np.abs(x))
    if x.tobytes() == y.tobytes():
        same += 1
    if sc:
        ws.append(np.max(np.abs(x - y)) / sc)
print(
    f"{c} vs {b}: renders {len(B)}, printouts moved {moved} (non-pin lines {nonpin}), err diffs {errs}; Z n={len(rows)} worst {rows[0][0]:.2e} {rows[0][1][0][-30:]} {rows[0][1][1]} |Z|={abs(rows[0][2]):.4g}; >1e-11: {sum(r > 1e-11 for r, *_ in rows)}, >1e-12: {sum(r > 1e-12 for r, *_ in rows)}; tables {len(A.files)}/{len(D.files)} bit-same {same}, worst/scale {max(ws):.2e}"
)
