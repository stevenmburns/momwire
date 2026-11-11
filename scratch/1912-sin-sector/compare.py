"""Max relative difference of two saved Y stacks, per frequency and over
all of them: max|Ya - Yb| / max|Yb| for each k. usage: compare.py A.npy B.npy"""

import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
per_k = [
    float(np.max(np.abs(x - y)) / np.max(np.abs(y))) for x, y in zip(a, b, strict=True)
]
print(
    f"{sys.argv[1]} vs {sys.argv[2]}: max {max(per_k):.3e} per-k {[f'{r:.2e}' for r in per_k]}"
)
