"""momwire#1064 G3, read from `g3_dispatch.py`'s two captures.

  python g3_compare.py

Pairwise, at every (medium, zone, theta): C++ avx2, C++ sse2, and numpy (in the
avx2 process and in the sse2 process). The bar is 1e-12 relative on every pair.
A point missing from either capture is a failure.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAR = 1e-12


def main():
    caps = {
        lab: json.loads((HERE / f"g3_{lab}.json").read_text())
        for lab in ("avx2", "sse2")
    }
    fails = []
    worst = {}
    points = 0
    for med in caps["avx2"]["media"]:
        rows = {
            lab: {
                (r["zone"], r["theta_deg"]): r for r in caps[lab]["media"].get(med, [])
            }
            for lab in caps
        }
        if set(rows["avx2"]) != set(rows["sse2"]):
            fails.append(f"{med}: the two captures cover different points")
        for key, ra in rows["avx2"].items():
            rs = rows["sse2"].get(key)
            if rs is None:
                continue
            points += 1
            vals = {
                "cpp_avx2": complex(*ra["cpp"]),
                "cpp_sse2": complex(*rs["cpp"]),
                "numpy@avx2": complex(*ra["numpy"]),
                "numpy@sse2": complex(*rs["numpy"]),
            }
            for (na, a), (nb, bb) in combinations(vals.items(), 2):
                r = abs(a - bb) / max(abs(bb), 1e-300)
                name = f"{na} vs {nb}"
                if r > worst.get(name, (0.0, None))[0]:
                    worst[name] = (r, [med, *key])
    bad = {k: w for k, w in worst.items() if w[0] > BAR}
    verdict = "PASS" if points and not fails and not bad else "FAIL"
    out = dict(
        gate="G3",
        verdict=verdict,
        points=points,
        failures=fails,
        worst=worst,
        accelerators={lab: caps[lab]["accelerator"] for lab in caps},
    )
    (HERE / "g3_compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
