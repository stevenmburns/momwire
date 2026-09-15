"""momwire#1064 D5a, a diagnostic of G5: the direct surfaces' cost at the floor
band's own nodes, on whichever tree PYTHONPATH selects.

  PYTHONPATH=<momwire src> python d5_nodecost.py --tree T --out F.jsonl

Soil A at 7.1 MHz; theta in {0.016667, 0.033333} deg x R1/lam_m in
{0.02, 0.2, 1, 2}, as one batch: a warm-up call, then 5 timed calls. The same
floats on every tree, so the per-node work is identical and only the compiled
contour's speed can differ.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np

from momwire import _ground_refl
from momwire import _sommerfeld_below as below

C0 = 299792458.0
EPS0 = 8.8541878128e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    f = 7.1e6
    k2 = 2.0 * math.pi * f / C0
    om = 2.0 * math.pi * f
    eps_t = _ground_refl.eps_tilde((13.0, 0.005), om, EPS0)
    lam_m = below.lambda_medium(eps_t, k2)
    d = 0.05 / 3.0
    thetas = (0.05 - 2.0 * d, 0.05 - d)
    pts = [(r, t) for r in (0.02, 0.2, 1.0, 2.0) for t in thetas]
    r1 = np.array([r * lam_m for r, _ in pts])
    th = np.radians([t for _, t in pts])
    times = []
    for i in range(6):
        t0 = time.perf_counter()
        below.iv_surfaces_direct_below(eps_t, k2, r1, th, rtol=1e-9, omega=om)
        dt = time.perf_counter() - t0
        if i:
            times.append(dt)
    rec = dict(
        tree=args.tree,
        rep=args.rep,
        nodes=len(pts),
        times=times,
        median=statistics.median(times),
        t_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    with args.out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
