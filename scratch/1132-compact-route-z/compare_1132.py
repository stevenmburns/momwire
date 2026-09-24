"""Compare two gate_1132 dumps entry by entry, as uint64 bit patterns."""

import sys

import numpy as np

a = np.load(sys.argv[1])
b = np.load(sys.argv[2])
keys = sorted(set(a.files) & set(b.files))
bad = 0
for k in keys:
    x, y = a[k], b[k]
    if x.shape != y.shape:
        print(f"{k:22s} SHAPE {x.shape} vs {y.shape}")
        bad += 1
        continue
    if np.iscomplexobj(x) or x.dtype == np.float64:
        xb = np.ascontiguousarray(x).view(np.uint64)
        yb = np.ascontiguousarray(y).view(np.uint64)
        nd = int(np.count_nonzero(xb != yb))
        scale = float(np.max(np.abs(x))) or 1.0
        md = float(np.max(np.abs(x - y))) / scale
    else:
        nd = int(np.count_nonzero(x != y))
        md = 0.0
    tag = "bit-identical" if nd == 0 else f"DIFF {nd} words, max rel {md:.3e}"
    if nd:
        bad += 1
    print(f"{k:22s} {str(x.shape):16s} {tag}")
print("only in A:", sorted(set(a.files) - set(b.files)))
print("only in B:", sorted(set(b.files) - set(a.files)))
print("RESULT:", "ALL BIT-IDENTICAL" if bad == 0 else f"{bad} keys differ")
