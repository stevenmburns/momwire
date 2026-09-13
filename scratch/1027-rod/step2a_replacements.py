"""momwire#1027 step 2(a): the replacements for the withdrawn whole-space rungs.

Neither mode uses `step2a_medium`'s momwire whole-space switch or its NEC-5
infinite-medium card, both of which C2a.5 did not clear for momwire.

  p2a1  P2a.1': deep burial d = 18 m, soil A, L = 0.15 and 0.60 m, both
        engines' ordinary buried paths (e^{-2 alpha d} = 1.8e-4 there)
  p2a3  P2a.3' / P2a.3'': a lossless homogeneous medium spelled as free space
        at f*sqrt(eps_r) with Z / sqrt(eps_r), both engines' ordinary
        free-space paths, refine 8 -> 16 Richardson, L = 0.15 / 0.60 / 2.40 m

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/1027-rod/step2a_replacements.py MODE [--out F]
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

from antennaknobs.engines.momwire import MomwireEngine
from antennaknobs.engines.nec5 import NEC5Engine

EPS_R = 13.0


def gw_lines(b):
    deck = NEC5Engine(b, ground="free").deck([b.freq])
    return [ln for ln in deck.splitlines() if ln.startswith("GW ")]


def scaled_free_row(L, refine):
    """(momwire Z, NEC-5 Z, deck sha, segs) for the lossless medium via free space."""
    scale = math.sqrt(EPS_R)
    b = rl.build((EPS_R, 0.0), L, 0.2, 42, refine=refine)
    mesh = gw_lines(b)
    b.freq = float(b.freq) * scale
    if gw_lines(b) != mesh:
        raise RuntimeError("the scaled-frequency deck changed the mesh")
    deck = NEC5Engine(b, ground="free").deck([b.freq])
    zm = complex(MomwireEngine(b, ground="free").impedance()[0]) / scale
    zn = s2.nec5_z(deck) / scale
    segs = sum(int(ln.split()[2]) for ln in mesh)
    return zm, zn, s2.sha12(deck), segs


def p2a3():
    out = []
    for L in (0.15, 0.60, 2.40):
        rows = {}
        for refine in (8, 16):
            zm, zn, sha, segs = scaled_free_row(L, refine)
            rows[refine] = dict(
                momwire=str(zm),
                nec5=str(zn),
                dR=zn.real - zm.real,
                deck_sha=sha,
                segs=segs,
            )
        d_inf = 2.0 * rows[16]["dR"] - rows[8]["dR"]
        r16 = complex(rows[16]["momwire"]).real
        rec = dict(L=L, rows=rows, dR_inf=d_inf, R16=r16, frac_pct=100 * d_inf / r16)
        out.append(rec)
        print(
            f"  L={L:4.2f} segs {rows[8]['segs']}/{rows[16]['segs']}  "
            f"momwire R8/R16 {complex(rows[8]['momwire']).real:.6g}/{r16:.6g}  "
            f"NEC-5 R16 {complex(rows[16]['nec5']).real:.6g}  "
            f"dR8 {rows[8]['dR']:+.5g} dR16 {rows[16]['dR']:+.5g}  "
            f"dR/R {rec['frac_pct']:+.4f} %",
            flush=True,
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["p2a1", "p2a3"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    exe = os.environ["NEC5_EXE"]
    meta = dict(
        nec5_exe=exe, nec5_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest()
    )
    print(meta, flush=True)
    if args.mode == "p2a1":
        result = [
            s2.ladder(s2.SOIL_A, L, 18.0, whole_space=False) for L in (0.15, 0.60)
        ]
    else:
        result = p2a3()
    if args.out is not None:
        args.out.write_text(
            json.dumps(dict(meta=meta, mode=args.mode, result=result), indent=1)
        )
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
