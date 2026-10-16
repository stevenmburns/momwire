"""Per-region comparison of two fullgrid.py outputs.

python fullgrid_cmp.py MAIN.npz BRANCH.npz
"""

import collections
import sys

import numpy as np

BANDS = {0: "floor", 1: "lo", 2: "mid", 3: "graze", 4: "steep"}
A, B = np.load(sys.argv[1]), np.load(sys.argv[2])
agg = collections.defaultdict(list)
for k in A.files:
    soil, sp, i = k.split("|")
    i = int(i)
    x, y = A[k], B[k]
    sc = np.abs(x).reshape(x.shape[0], -1).max(axis=1)
    node = np.abs(x - y) / sc[:, None, None]
    zone = ["inner", "near", "far"][i // 5]
    agg[(sp, zone, BANDS[i % 5])].append(
        (float(node.max()), float(np.median(node)), float((node > 0).mean()), soil)
    )
print("spacing zone  band   worst/scale (soil)   median node   fraction moved")
for k, v in sorted(agg.items()):
    w = max(v, key=lambda t: t[0])
    print(
        f"{k[0]:5s} {k[1]:5s} {k[2]:5s}  {w[0]:.2e} ({w[3]:7s})  "
        f"{np.median([t[1] for t in v]):.1e}  {np.mean([t[2] for t in v]):.2f}"
    )
