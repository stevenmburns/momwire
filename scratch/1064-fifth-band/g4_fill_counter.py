"""momwire#1064 G4: each theta node of the floor and low bands is filled once.

  PYTHONPATH=<branch src> python g4_fill_counter.py --out g4_fill_counter.json

On REAL grids (soil A, 7 MHz, R1 to the cap), per R1 zone and in both fill
orders, a counter wraps `iv_surfaces_direct_below` and records every call the two
bands' fills make: the distinct theta nodes, and the number of points. The gate,
as re-registered for the joint batch: floor band first, ONE call carrying all
six theta columns; low band first, the low band's four and then the floor
band's two; no theta node appears in two calls; the floor band's top two columns are the low band's first two, bit for
bit; and the low band's values are the same bits in both orders.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from momwire import _ground_refl
from momwire import _sommerfeld_below as below

C0 = 299792458.0
EPS0 = 8.8541878128e-12
RI = below.region_index
ZONES = {"inner": below._ZONE_INNER, "near": below._ZONE_NEAR, "far": below._ZONE_FAR}


def fresh_grid():
    f = 7e6
    k2 = 2.0 * math.pi * f / C0
    om = 2.0 * math.pi * f
    eps_t = _ground_refl.eps_tilde((13.0, 0.005), om, EPS0)
    lam_m = below.lambda_medium(eps_t, k2)
    r1_max = below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m
    return below.SommerfeldGridBelow(eps_t, k2, r1_max, omega=om)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    real = below.iv_surfaces_direct_below
    rows, fails = [], []
    low_vals = {}
    for zname, zone in ZONES.items():
        for order in ("floor_first", "low_first"):
            g = fresh_grid()
            calls = []

            def counting(eps_t, k2, R1, theta, _calls=calls, **kw):
                th = np.asarray(theta, dtype=float)
                _calls.append(
                    dict(
                        theta_deg=[repr(math.degrees(t)) for t in np.unique(th)],
                        n_theta=int(np.unique(th).size),
                        points=int(th.size),
                    )
                )
                return real(eps_t, k2, R1, theta, **kw)

            below.iv_surfaces_direct_below = counting
            t0 = time.perf_counter()
            try:
                fi, li = RI(zone, below._BAND_FLOOR), RI(zone, below._BAND_LO)
                if order == "floor_first":
                    g._fill_region(fi)
                    g._fill_region(li)
                else:
                    g._fill_region(li)
                    g._fill_region(fi)
                g._fill_region(fi)
            finally:
                below.iv_surfaces_direct_below = real
            floor, lo = g._regions[fi], g._regions[li]
            n_r = int(lo["n_r"])
            thetas = [t for c in calls for t in c["theta_deg"]]
            shared_bitwise = all(
                np.array_equal(floor["vals"][:, :, f], lo["vals"][:, :, j])
                for f, j in below._SOMM_BELOW_BAND_FLOOR_SHARED
            )
            row = dict(
                zone=zname,
                order=order,
                n_r=n_r,
                calls=calls,
                seconds=round(time.perf_counter() - t0, 2),
                shared_columns_bitwise=shared_bitwise,
            )
            rows.append(row)
            where = f"{zname}/{order}"
            sizes = [c["n_theta"] for c in calls]
            want = [6] if order == "floor_first" else [4, 2]
            if sizes != want:
                fails.append(f"{where}: call theta counts {sizes}, expected {want}")
            if len(thetas) != len(set(thetas)):
                fails.append(f"{where}: a theta node was evaluated in two calls")
            if any(c["points"] != c["n_theta"] * n_r for c in calls):
                fails.append(f"{where}: a call's point count is not n_theta x n_r")
            if not shared_bitwise:
                fails.append(
                    f"{where}: floor-band shared columns are not the low band's"
                )
            low_vals.setdefault(zname, []).append(lo["vals"].copy())
            print(
                where,
                sizes,
                f"{row['seconds']} s",
                "shared ok" if shared_bitwise else "SHARED MISMATCH",
            )
        a, b = low_vals[zname]
        if not np.array_equal(a, b):
            fails.append(f"{zname}: the low band's values differ between fill orders")
    verdict = "FAIL" if fails else "PASS"
    args.out.write_text(
        json.dumps(
            dict(gate="G4", verdict=verdict, failures=fails, rows=rows), indent=1
        )
    )
    for f in fails:
        print("FAIL", f)
    print(f"G4: {verdict} ({len(fails)} failures)")


if __name__ == "__main__":
    main()
