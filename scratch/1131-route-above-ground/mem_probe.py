"""momwire#1131: peak RSS and wall of ONE solve, dense or route, in a fresh
process (run it once per arm; `run_150.sh` does the interleaving).

  PYTHONPATH=<tree>/src:scratch/1131-route-above-ground \\
      python mem_probe.py --radials 48 --arm route --ground sommerfeld
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np

from deck import H_SURFACE, deck

warnings.filterwarnings("ignore")


def vmhwm_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radials", type=int, required=True)
    ap.add_argument("--arm", choices=("dense", "route"), required=True)
    ap.add_argument("--ground", default="sommerfeld")
    ap.add_argument("--height", type=float, default=H_SURFACE)
    ap.add_argument("--coeffs", default=None, help="save the coefficients (.npy)")
    a = ap.parse_args()
    s = deck(a.radials, ground=a.ground, h=a.height, rot=a.arm == "route")
    before = vmhwm_mb()
    t0 = time.perf_counter()
    z, c = s.compute_impedance()
    dt = time.perf_counter() - t0
    z = complex(np.atleast_1d(z)[0])
    if a.coeffs:
        np.save(a.coeffs, c)
    print(
        json.dumps(
            {
                "radials": a.radials,
                "arm": a.arm,
                "ground": a.ground,
                "h": a.height,
                "n_basis": int(c.shape[0]),
                "seconds": dt,
                "vmhwm_mb": vmhwm_mb(),
                "vmhwm_before_solve_mb": before,
                "z": [z.real.hex(), z.imag.hex()],
                "z_float": [z.real, z.imag],
            }
        )
    )


if __name__ == "__main__":
    main()
