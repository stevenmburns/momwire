import collections
import json

import numpy as np

A = np.load("g2-base.npz")
D = np.load("g2-wa.npz")
per = collections.defaultdict(list)
for k in sorted(set(A.files) & set(D.files)):
    fam, reg = k.split("-")[0], k.split("-")[1]
    x, y = A[k], D[k]
    # per surface scale, like the anchors' _region_errors
    sc = np.abs(x).reshape(x.shape[0], -1).max(axis=1)
    err = np.abs(x - y).reshape(x.shape[0], -1).max(axis=1)
    rel = float((err / np.where(sc > 0, sc, 1)).max())
    per[(fam, reg)].append(rel)
for (fam, reg), v in sorted(per.items()):
    v = np.array(v)
    print(
        f"{fam:6s} {reg:4s} n={len(v):3d} moved={int((v > 0).sum()):3d} max={v.max():.2e} median={np.median(v):.2e}"
    )
rec = [json.loads(l) for l in open("g2-base.jsonl")]
nb = sum(any(g.startswith("below") for g in r["grids"]) for r in rec)
print("renders building a below table:", nb, "of", len(rec))
