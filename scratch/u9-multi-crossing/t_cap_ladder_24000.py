"""U9 GT3: re-measure #838's tail-cap ladder at `_MAX_TAIL_PANELS = 24000`
on that gate's own deck (soil A, 7 MHz, R1 = lambda_m), and time the numpy
dispatch at the refused rung the re-pinned tests will use. No Z.

  PYTHONPATH=<src tree>/src python t_cap_ladder_24000.py --out F
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import momwire
from momwire import _ground_refl
from momwire import _sommerfeld_below as below

C0 = 299792458.0
EPS0 = 8.8541878128e-12
RUNGS = (
    0.12,
    0.10,
    0.09,
    0.08,
    0.06,
    0.05,
    0.04,
    0.03,
    0.023,
    0.02,
    0.05 - 2 * 0.05 / 3,
    0.015,
    0.0125,
    0.01,
)
REFUSED_PROBE = 0.0125


def deck():
    om = 2.0 * math.pi * 7e6
    k2 = om / C0
    eps_t = _ground_refl.eps_tilde((13.0, 0.005), om, EPS0)
    return eps_t, k2, om, below.lambda_medium(eps_t, k2)


def at(th_deg):
    eps_t, k2, om, lam_m = deck()
    h = below.Health()
    t0 = time.time()
    try:
        below.iv_surfaces_direct_below(
            eps_t,
            k2,
            np.array([lam_m]),
            np.radians([th_deg]),
            rtol=1e-9,
            omega=om,
            health=h,
        )
        verdict = "SERVED"
    except ValueError as exc:
        verdict = (
            "REFUSED" if "exhausted its tail budget" in str(exc) else f"OTHER: {exc}"
        )
    return dict(
        theta_deg=th_deg,
        verdict=verdict,
        panels=h.max_tail_panels,
        seconds=round(time.time() - t0, 2),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rec = dict(
        momwire=momwire.__file__,
        budget=below._MAX_TAIL_PANELS,
        floor_deg=below._SOMM_BELOW_TH_MIN_DEG,
    )
    assert below._use_below_accel()
    rec["ladder"] = [at(th) for th in RUNGS]
    below._FORCE_NUMPY = True
    assert not below._use_below_accel()
    rec["numpy_at_refused"] = at(REFUSED_PROBE)
    args.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
