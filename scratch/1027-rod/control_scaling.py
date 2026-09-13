"""momwire#1027 step 2(a), control C2a.5: both whole-space spellings against a
known answer.

A wire structure in a lossless homogeneous medium (eps_r, mu_r = 1) is the same
structure in free space at the frequency whose free-space wavelength equals the
medium's, with every impedance scaled by eta_m / eta_0 = 1 / sqrt(eps_r):

    Z_medium(f) = Z_free(f * sqrt(eps_r)) / sqrt(eps_r)

The right-hand side goes through each engine's ordinary free-space path, so a
construction error in either whole-space spelling (`step2a_medium`'s momwire
switch, or its NEC-5 card insertion) cannot reproduce it. The mesh is held
fixed and checked: only the solve frequency moves.

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/1027-rod/control_scaling.py [--out F]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import rod_ladder as rl
import step2a_medium as s2

from antennaknobs.engines.momwire import MomwireEngine
from antennaknobs.engines.nec5 import NEC5Engine

EPS_R = 13.0


def gw_lines(b):
    deck = NEC5Engine(b, ground="free").deck([b.freq])
    return [ln for ln in deck.splitlines() if ln.startswith("GW ")]


def rel(a, b):
    return abs(a - b) / abs(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    scale = math.sqrt(EPS_R)
    medium = (EPS_R, 0.0)
    out = []
    for L in (0.15, 0.60):
        b = rl.build(medium, L, 0.2, 42, refine=8)
        f0 = float(b.freq)
        mesh = gw_lines(b)
        with s2.momwire_whole_space():
            zm_ws = complex(MomwireEngine(b, ground=("finite", *medium)).impedance()[0])
        zn_ws = s2.nec5_z(s2.um_deck(b, medium))

        bs = rl.build(medium, L, 0.2, 42, refine=8)
        bs.freq = f0 * scale
        if gw_lines(bs) != mesh:
            raise RuntimeError("the scaled-frequency deck changed the mesh")
        zm_fs = complex(MomwireEngine(bs, ground="free").impedance()[0]) / scale
        zn_fs = s2.nec5_z(NEC5Engine(bs, ground="free").deck([bs.freq])) / scale

        rec = dict(
            L=L,
            f_medium=f0,
            f_free=float(bs.freq),
            segs=sum(int(ln.split()[2]) for ln in mesh),
            momwire_whole_space=str(zm_ws),
            momwire_free_scaled=str(zm_fs),
            momwire_rel_Z=rel(zm_ws, zm_fs),
            momwire_rel_R=rel(zm_ws.real, zm_fs.real),
            nec5_whole_space=str(zn_ws),
            nec5_free_scaled=str(zn_fs),
            nec5_rel_Z=rel(zn_ws, zn_fs),
            nec5_rel_R=rel(zn_ws.real, zn_fs.real),
            dR_over_R_whole_space=(zn_ws.real - zm_ws.real) / zm_ws.real,
            dR_over_R_free_scaled=(zn_fs.real - zm_fs.real) / zm_fs.real,
        )
        out.append(rec)
        print(
            f"L={L:4.2f} segs={rec['segs']} f {f0:.4g} -> {bs.freq:.6g} MHz\n"
            f"  momwire  whole-space {zm_ws:.6g}  free/sqrt(er) {zm_fs:.6g}  "
            f"rel Z {rec['momwire_rel_Z']:.2e}  rel R {rec['momwire_rel_R']:.2e}\n"
            f"  NEC-5    whole-space {zn_ws:.6g}  free/sqrt(er) {zn_fs:.6g}  "
            f"rel Z {rec['nec5_rel_Z']:.2e}  rel R {rec['nec5_rel_R']:.2e}\n"
            f"  dR/R  whole-space {100 * rec['dR_over_R_whole_space']:+.4f} %   "
            f"free-scaled {100 * rec['dR_over_R_free_scaled']:+.4f} %",
            flush=True,
        )
    if args.out is not None:
        args.out.write_text(json.dumps(out, indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
