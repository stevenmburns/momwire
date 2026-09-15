"""momwire#1064 D6, a diagnostic of G5: one zone's floor- and low-band nodes
evaluated as ONE batch (U9's single band) or as the TWO batches split B makes,
on one tree.

  PYTHONPATH=<branch src> python d6_batch.py --out d6_batch.json

Soil A at 7.1 MHz, on a grid to the cap. For the inner and near zones, the nodes
are the floor band's own two theta columns and the low band's four, over the
zone's R1 rows: the same six columns U9's single band evaluates. Timed:

  one  a single `iv_surfaces_direct_below` call over all six columns
  two  a call over the low band's four columns, then one over the floor band's two

A warm-up comes first, then 3 repeats of each, alternating which goes first. The
nodes and their per-node work are identical between the two (D4a showed batch
and one-at-a-time evaluation agree bit for bit), so only how the work is batched
differs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np

from momwire import _ground_refl
from momwire import _sommerfeld_below as below

C0 = 299792458.0
EPS0 = 8.8541878128e-12
RI = below.region_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    f = 7.1e6
    k2 = 2.0 * math.pi * f / C0
    om = 2.0 * math.pi * f
    eps_t = _ground_refl.eps_tilde((13.0, 0.005), om, EPS0)
    lam_m = below.lambda_medium(eps_t, k2)
    g = below.SommerfeldGridBelow(
        eps_t, k2, below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m, omega=om
    )
    out = dict(
        cpu_count=os.cpu_count(),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        zones={},
    )
    for zname, zone in (("inner", below._ZONE_INNER), ("near", below._ZONE_NEAR)):
        floor = g._regions[RI(zone, below._BAND_FLOOR)]
        lo = g._regions[RI(zone, below._BAND_LO)]
        r_nodes = floor["r_nodes"]
        th_floor = floor["th_nodes"][:2]
        th_lo = lo["th_nodes"]
        th_all = np.concatenate([th_floor, th_lo])

        def call(thetas, r_nodes=r_nodes):
            rr, tt = np.meshgrid(r_nodes, thetas, indexing="ij")
            t0 = time.perf_counter()
            below.iv_surfaces_direct_below(
                g.eps_t, g.k2, rr, tt, rtol=g._rtol, omega=g.omega, mu=g.mu
            )
            return time.perf_counter() - t0

        call(th_all)  # warm-up
        one, two = [], []
        for rep in range(3):
            order = ("one", "two") if rep % 2 == 0 else ("two", "one")
            for mode in order:
                if mode == "one":
                    one.append(call(th_all))
                else:
                    two.append(call(th_lo) + call(th_floor))
        out["zones"][zname] = dict(
            n_r=int(len(r_nodes)),
            nodes=int(len(r_nodes) * len(th_all)),
            one=one,
            two=two,
            ratio_two_over_one=statistics.median(two) / statistics.median(one),
        )
        print(zname, json.dumps(out["zones"][zname]), flush=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
