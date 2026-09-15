"""momwire#1064 D2, diagnostics of G1a: two `g12_grid.py` captures compared on
the OLD set (theta >= 0.05 deg), by band and by component.

  python d2_compare.py A.json B.json

For each theta class (low [0.05, 0.1), mid [0.1, 1), grazing/steep >= 1 deg):
points, how many surface and kernel values are not bit-identical, the worst
relative difference of each, and which media carry any difference.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def theta(key):
    return float(str(key).removeprefix("np.float64(").removesuffix(")"))


def band(th):
    if th < 0.1:
        return "low [0.05, 0.1)"
    return "mid [0.1, 1)" if th < 1.0 else "grazing/steep >= 1"


def rel(a, b):
    a, b = complex(*a), complex(*b)
    return abs(a - b) / max(abs(b), 1e-300)


def main():
    fa, fb = Path(sys.argv[1]), Path(sys.argv[2])
    ta, tb = (json.loads((HERE / f).read_text()) for f in (fa, fb))
    stats = defaultdict(
        lambda: dict(points=0, surf=0, kernel=0, max_surf=0.0, max_kernel=0.0)
    )
    media = defaultdict(int)
    missing = []
    for med, m in ta["media"].items():
        if med not in tb["media"]:
            missing.append(med)
            continue
        other = {(p["r1_lam"], p["theta_deg"]): p for p in tb["media"][med]["old"]}
        for p in m["old"]:
            k = (p["r1_lam"], p["theta_deg"])
            q = other.get(k)
            if q is None:
                missing.append(f"{med} {k}")
                continue
            s = stats[band(theta(k[1]))]
            s["points"] += 1
            if p["surf"] != q["surf"]:
                s["surf"] += 1
                s["max_surf"] = max(
                    s["max_surf"],
                    max(rel(p["surf"][x], q["surf"][x]) for x in p["surf"]),
                )
                media[med] += 1
            if p["kernel"] != q["kernel"]:
                s["kernel"] += 1
                s["max_kernel"] = max(s["max_kernel"], rel(p["kernel"], q["kernel"]))
                media[med] += 0 if p["surf"] != q["surf"] else 1
    out = dict(
        a=str(fa),
        b=str(fb),
        trees=[ta["tree"], tb["tree"]],
        by_band=dict(stats),
        media_with_differences=dict(media),
        missing=missing,
        bit_identical=not missing
        and all(v["surf"] == 0 and v["kernel"] == 0 for v in stats.values()),
    )
    name = f"d2_{fa.stem}__vs__{fb.stem}.json"
    (HERE / name).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
