"""momwire#1064 G3 (capture): the below/below projection across the seams, on one
accelerator variant per process, with the numpy dispatch alongside.

  MOMWIRE_FORCE_VARIANT=avx2|sse2 PYTHONPATH=<branch src> \\
      python g3_dispatch.py --label avx2|sse2 --out g3_LABEL.json

For soil A at 7 MHz and soil C at 21 MHz, at R1 in each zone, and at theta in
the registered list: `remainder_field_proj_below` on the variant this process
loaded (C++), then the same call with the accelerator switched off (numpy). The
grid is filled by the C++ call first, so the numpy call reads the same tables.
`g3_compare.py` compares avx2, sse2 and numpy pairwise.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import momwire._accelerators as acc
from momwire import _ground_refl
from momwire import _sommerfeld_below as below

C0 = 299792458.0
EPS0 = 8.8541878128e-12
SOILS = {"A": (13.0, 0.005), "C": (5.0, 0.001)}
MEDIA = [("A", 7e6), ("C", 21e6)]
THETA_DEG = (
    0.05 - 2.0 * (0.05 / 3.0),
    0.02,
    0.0333,
    0.049999999,
    0.05,
    0.050000001,
    0.06,
    0.0666,
    0.07,
    0.099999999,
    0.1,
    0.100000001,
    0.5,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rec = dict(label=args.label, accelerator=acc.__file__, media={})
    for soil, f in MEDIA:
        k2 = 2.0 * math.pi * f / C0
        om = 2.0 * math.pi * f
        eps_t = _ground_refl.eps_tilde(SOILS[soil], om, EPS0)
        lam_m = below.lambda_medium(eps_t, k2)
        k_m = below.k_medium(eps_t, k2)
        g = below.SommerfeldGridBelow(
            eps_t, k2, below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m, omega=om
        )
        zone_r1 = {
            "inner": 0.5 * g.r_break,
            "near": 0.5 * (g.r_break + g.r_near),
            "far": 0.5 * (g.r_near + g.r1_max),
        }
        rows = []
        for zname, r1 in zone_r1.items():
            for thd in THETA_DEG:
                th = math.radians(thd)
                rho, hh = r1 * math.cos(th), r1 * math.sin(th)
                obs = np.array([[0.0, 0.0, -0.5 * hh]])
                src = np.array([[rho, 0.0, -0.5 * hh]])
                t_obs = np.array([[1.0, 0.0, 0.0]])
                t_src = np.array([[0.6, 0.0, 0.8]])
                cpp = below.remainder_field_proj_below(
                    obs, t_obs, src, t_src, 0.0, k2, k_m, g
                )[0, 0]
                real = below._use_below_accel
                below._use_below_accel = lambda: False
                try:
                    npy = below.remainder_field_proj_below(
                        obs, t_obs, src, t_src, 0.0, k2, k_m, g
                    )[0, 0]
                finally:
                    below._use_below_accel = real
                rows.append(
                    dict(
                        zone=zname,
                        theta_deg=repr(thd),
                        cpp=[cpp.real, cpp.imag],
                        numpy=[npy.real, npy.imag],
                    )
                )
        rec["media"][f"{soil}/{f / 1e6:g}MHz"] = rows
        print(args.label, soil, f, "done", flush=True)
    args.out.write_text(json.dumps(rec))


if __name__ == "__main__":
    main()
