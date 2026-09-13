"""momwire#1027 step 2(f), P2f.1 / P2f.2: power balance, each engine against
itself, in lossless free space.

Every watt into a lossless antenna in free space is radiated, so the average
power gain P_rad / P_in is 1 on a self-consistent solve.

* momwire: P_in = `MomwireEngine.input_power()`; P_rad = eta0 k^2 / (32 pi^2)
  * integral |M_perp|^2 dOmega over the full sphere at cell centres, from the
  engine's own segment dipoles (the same `_segment_dipoles` /
  `_evaluate_M_perp` its gain readout uses, so the far field is momwire's own).
* NEC-5: the printed AVERAGE POWER GAIN of an RP run with averaging on
  (`NEC5Engine.average_power_gain`). Its POWER BUDGET is not used: RADIATED
  POWER there is INPUT - WIRE LOSS by construction and cannot fail.

The rod is the step 2(a) rod at refine 16, solved in free space at
f * sqrt(13) = 25.599414 MHz, i.e. the frequency P2a.3' used.

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/1027-rod/step2f_power_balance.py [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import rod_ladder as rl

from antennaknobs.engines.momwire import ETA0, MomwireEngine
from antennaknobs.engines.nec5 import NEC5Engine

EPS_R = 13.0
N_THETA, N_PHI = 180, 8


def free_space_builder(L, refine):
    b = rl.build((EPS_R, 0.0), L, 0.2, 42, refine=refine)
    b.freq = float(b.freq) * math.sqrt(EPS_R)
    return b


def momwire_balance(b):
    eng = MomwireEngine(b, ground="free")
    wavelength = eng._wavelength_for(b.freq)
    k = 2.0 * math.pi / wavelength
    sim, coeffs, z = eng._solved_excited(wavelength)
    mid, dr, i_mid = eng._segment_dipoles(sim, coeffs)
    dth, dph = math.pi / N_THETA, 2.0 * math.pi / N_PHI
    theta = (np.arange(N_THETA) + 0.5) * dth
    phi = np.arange(N_PHI) * dph
    m2 = eng._evaluate_M_perp(mid, dr, i_mid, k, theta, phi, b.freq * 1e6)
    integral = float(np.sum(m2 * np.sin(theta)[:, None]) * dth * dph)
    p_rad = ETA0 * k * k / (32.0 * math.pi**2) * integral
    p_in = float(eng.input_power())
    return dict(z=str(complex(z)), p_in=p_in, p_rad=p_rad, avg_gain=p_rad / p_in)


def nec5_balance(b):
    eng = NEC5Engine(b, ground="free")
    avg, solid = eng.average_power_gain(n_theta=90, n_phi=4)
    return dict(avg_gain=float(avg), solid_angle_over_pi=float(solid) / math.pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    exe = os.environ["NEC5_EXE"]
    meta = dict(
        nec5_exe=exe, nec5_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest()
    )
    print(meta, flush=True)
    out = []
    for L in (0.15, 0.60, 2.40):
        b = free_space_builder(L, 16)
        mw = momwire_balance(b)
        n5 = nec5_balance(free_space_builder(L, 16))
        rec = dict(L=L, freq_mhz=float(b.freq), momwire=mw, nec5=n5)
        out.append(rec)
        print(
            f"L={L:4.2f} f={b.freq:.6f} MHz  momwire avg gain {mw['avg_gain']:.6f} "
            f"(P_in {mw['p_in']:.6e}, P_rad {mw['p_rad']:.6e})  "
            f"NEC-5 avg gain {n5['avg_gain']:.6f} "
            f"(solid angle {n5['solid_angle_over_pi']:.4f} pi)",
            flush=True,
        )
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
