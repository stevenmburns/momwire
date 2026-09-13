"""momwire#1027 step 2(c), C2c.1 / P2c.1: momwire's extended thin-wire kernel
against NEC-5's unchanged deck.

`MomwireEngine(extended_kernel=True)` swaps momwire's reduced kernel for NEC's
O(a^2) tube expansion. Both kernels are solved at every rung, so C2c.1 (the
switch moves R, not bit-identical) is read from the same run as P2c.1.

Media:
  soilA  #1027's rod: buried d = 0.2 m, soil A, 7.1 MHz, L = 0.15 / 0.60 / 2.40 m
  free   the lossless medium as scaled free space (f * sqrt(13), Z / sqrt(13)),
         L = 0.15 / 0.60 m, with R against R_tri = 20 pi^2 (L / lambda)^2

Far mesh refine 8 -> 16 Richardson; stock feed parity.

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/1027-rod/step2c_kernel.py MEDIUM [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import rod_ladder as rl
import step2a_medium as s2
from step2f_power_balance import free_space_builder

from antennaknobs.engines.momwire import MomwireEngine
from antennaknobs.engines.nec5 import NEC5Engine

SOIL_A = ("finite", 13.0, 0.005)
EPS_R = 13.0
C0 = 299792458.0


def row(medium, L, refine):
    if medium == "soilA":
        b = rl.build("A", L, 0.20, 42, refine=refine)
        z_red = complex(MomwireEngine(b, ground=SOIL_A).impedance()[0])
        z_ext = complex(
            MomwireEngine(b, ground=SOIL_A, extended_kernel=True).impedance()[0]
        )
        zn = complex(NEC5Engine(b, ground=SOIL_A).impedance()[0])
    else:
        scale = math.sqrt(EPS_R)
        b = free_space_builder(L, refine)
        z_red = complex(MomwireEngine(b, ground="free").impedance()[0]) / scale
        z_ext = (
            complex(
                MomwireEngine(b, ground="free", extended_kernel=True).impedance()[0]
            )
            / scale
        )
        zn = s2.nec5_z(NEC5Engine(b, ground="free").deck([b.freq])) / scale
    return dict(
        refine=refine,
        momwire_reduced=str(z_red),
        momwire_extended=str(z_ext),
        nec5=str(zn),
        switch_moved=z_red != z_ext,
        dR_reduced=zn.real - z_red.real,
        dR_extended=zn.real - z_ext.real,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("medium", choices=["soilA", "free"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    exe = os.environ["NEC5_EXE"]
    meta = dict(
        nec5_exe=exe, nec5_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest()
    )
    print(meta, flush=True)
    lengths = (0.15, 0.60, 2.40) if args.medium == "soilA" else (0.15, 0.60)
    out = []
    for L in lengths:
        r8, r16 = row(args.medium, L, 8), row(args.medium, L, 16)
        rec = dict(L=L, rows=[r8, r16])
        for k in ("reduced", "extended"):
            m8 = complex(r8[f"momwire_{k}"]).real
            m16 = complex(r16[f"momwire_{k}"]).real
            d_inf = 2.0 * r16[f"dR_{k}"] - r8[f"dR_{k}"]
            rec[f"{k}_R_inf"] = 2.0 * m16 - m8
            rec[f"{k}_frac_pct"] = 100.0 * d_inf / m16
        n_inf = 2.0 * complex(r16["nec5"]).real - complex(r8["nec5"]).real
        rec["nec5_R_inf"] = n_inf
        line = (
            f"L={L:4.2f} [{args.medium}] switch moved R: {r8['switch_moved']}/"
            f"{r16['switch_moved']}  R_inf reduced {rec['reduced_R_inf']:.6g}  "
            f"extended {rec['extended_R_inf']:.6g}  NEC-5 {n_inf:.6g}  "
            f"dR/R reduced {rec['reduced_frac_pct']:+.4f} %  "
            f"extended {rec['extended_frac_pct']:+.4f} %"
        )
        if args.medium == "free":
            f_hz = 7.1e6 * math.sqrt(EPS_R)
            r_tri = 20 * math.pi**2 * (L / (C0 / f_hz)) ** 2 / math.sqrt(EPS_R)
            rec["R_tri"] = r_tri
            line += (
                f"  R/R_tri reduced {rec['reduced_R_inf'] / r_tri:.5f} "
                f"extended {rec['extended_R_inf'] / r_tri:.5f} NEC-5 {n_inf / r_tri:.5f}"
            )
        out.append(rec)
        print(line, flush=True)
    if args.out is not None:
        args.out.write_text(
            json.dumps(dict(meta=meta, medium=args.medium, result=out), indent=1)
        )
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
