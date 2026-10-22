"""Tables from ref.jsonl + mw.jsonl: Z per rung, per-doubling |dZ|, and the
gap to each reference. Usage: python tables.py ref.jsonl mw.jsonl"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

rows = [json.loads(ln) for p in sys.argv[1:] for ln in open(p)]
Z = defaultdict(dict)
for x in rows:
    key = (x["variant"], x.get("n_rad", 16), x["cfg"], x["feed"], x["engine"])
    Z[key][x["r"]] = complex(x["R"], x["X"]) if "R" in x else x["error"][:70]


def fmt(z):
    return f"{z.real:8.2f}{z.imag:+8.2f}j" if isinstance(z, complex) else z


groups = sorted({k[:4] for k in Z})
for g in groups:
    print(f"\n### {g[0]}  n_rad={g[1]}  radii={g[2]}  feed={g[3]}")
    engines = [k[4] for k in Z if k[:4] == g]
    for e in sorted(engines):
        z = Z[g + (e,)]
        rs = sorted(z)
        line = " | ".join(fmt(z[r]) for r in rs)
        steps = [
            abs(z[b] - z[a])
            for a, b in zip(rs, rs[1:])
            if isinstance(z[a], complex) and isinstance(z[b], complex)
        ]
        print(
            f"{e:20s} r={rs}: {line}   |dZ| per doubling: "
            + ", ".join(f"{s:.3f}" for s in steps)
        )
