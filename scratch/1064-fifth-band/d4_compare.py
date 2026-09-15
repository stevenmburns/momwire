"""momwire#1064 D4a comparison: the direct surfaces across trees, per variant.

  python d4_compare.py

Reads `d4_{v055,main,branch}_{avx2,sse2}.json`. For each variant, and for the
batch and the one-point-at-a-time evaluations, reports whether main and the
branch are bit-identical to 0.55.0, with the worst relative difference; and
whether each capture's batch and single evaluations agree bit for bit.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TREES = ("v055", "main", "branch")
VARIANTS = ("avx2", "sse2")


def worst(a, b):
    w = 0.0
    for k in a:
        for x, y in zip(a[k], b[k], strict=True):
            cx, cy = complex(*x), complex(*y)
            w = max(w, abs(cx - cy) / max(abs(cy), 1e-300))
    return w


def main():
    caps = {
        (t, v): json.loads((HERE / f"d4_{t}_{v}.json").read_text())
        for t in TREES
        for v in VARIANTS
    }
    out = dict(diagnostic="D4a", variants={})
    for v in VARIANTS:
        ref = caps[("v055", v)]
        row = dict(
            accelerators={
                t: caps[(t, v)]["accelerator"].rsplit("/", 1)[-1] for t in TREES
            }
        )
        for t in ("main", "branch"):
            c = caps[(t, v)]
            for mode in ("batch", "single"):
                row[f"{t}_vs_v055_{mode}"] = dict(
                    bit_identical=c[mode] == ref[mode], worst=worst(c[mode], ref[mode])
                )
        row["batch_equals_single"] = {
            t: caps[(t, v)]["batch"] == caps[(t, v)]["single"] for t in TREES
        }
        out["variants"][v] = row
    (HERE / "d4_compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
