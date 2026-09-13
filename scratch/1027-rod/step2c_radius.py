"""momwire#1027 step 2(c): the rod's radius at fixed length.

The rod of `step2b_source_refine.SourceRefinedRod`, with the conductor radius a
as a declared builder parameter (never a class attribute: #1027's first ladder
lost a knob that way), carried to both engines through a per-wire `WireSpec`.
The GW radius field and momwire's `_wire_radius` are both checked against it
before a row is read.

Two media:
  soilA  the #1027 quantity: buried at d = 0.2 m, soil A, 7.1 MHz
  free   the lossless medium as scaled free space (f * sqrt(13), Z / sqrt(13)),
         where the short-dipole R_tri = 20 pi^2 (L / lambda)^2 is the answer
         both engines should approach as Omega = 2 ln(L / a) grows

Far mesh refine 8 -> 16 Richardson. `--source-f` sets the source-region
refinement of step 2(b). Stock momwire feed parity (P2b.1 showed the port
split is immaterial on this rod).

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/1027-rod/step2c_radius.py MEDIUM [--source-f F] [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType

import step2a_medium as s2
from step2b_source_refine import SourceRefinedRod

from antennaknobs.engines.momwire import MomwireEngine
from antennaknobs.engines.nec5 import NEC5Engine
from antennaknobs.wire_catalog import WireSpec

EPS_R = 13.0
C0 = 299792458.0
RADII = (0.0005, 0.00015, 0.00005, 0.000015)
LENGTHS = (0.15, 0.60)


class RadiusRod(SourceRefinedRod):
    default_params = MappingProxyType(
        {**SourceRefinedRod.default_params, "radius_m": 0.0005}
    )

    def build_wires(self):
        spec = WireSpec(radius=float(self.radius_m))
        return [w._replace(spec=spec) for w in super().build_wires()]


def build(medium, L, refine, f, a):
    b = RadiusRod()
    b.nominal_nsegs = 42
    b.design_eps_r, b.design_sigma = (13.0, 0.005) if medium == "soilA" else (EPS_R, 0.0)
    b.rod_len, b.rod_depth, b.refine, b.source_f, b.radius_m = L, 0.20, refine, f, a
    if medium == "free":
        b.freq = float(b.freq) * math.sqrt(EPS_R)
    return b


def check_radius(b, g, a):
    deck = NEC5Engine(b, ground=g).deck([b.freq])
    radii = {float(ln.split()[9]) for ln in deck.splitlines() if ln.startswith("GW ")}
    eng = MomwireEngine(b, ground=g)
    mw = eng._wire_radius
    mw_set = {float(mw)} if not hasattr(mw, "__len__") else {float(x) for x in mw}
    if radii != {a} or mw_set != {a}:
        raise RuntimeError(f"radius not carried: NEC-5 {radii}, momwire {mw_set}, asked {a}")
    return eng, deck


def row(medium, L, refine, f, a):
    b = build(medium, L, refine, f, a)
    if medium == "soilA":
        g = ("finite", 13.0, 0.005)
        eng, deck = check_radius(b, g, a)
        zm = complex(eng.impedance()[0])
        zn = complex(NEC5Engine(b, ground=g).impedance()[0])
        scale = 1.0
    else:
        eng, deck = check_radius(b, "free", a)
        scale = math.sqrt(EPS_R)
        zm = complex(eng.impedance()[0]) / scale
        zn = s2.nec5_z(deck) / scale
    segs = sum(int(ln.split()[2]) for ln in deck.splitlines() if ln.startswith("GW "))
    return dict(refine=refine, segs=segs, momwire=str(zm), nec5=str(zn), dR=zn.real - zm.real)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("medium", choices=["soilA", "free"])
    ap.add_argument("--source-f", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    exe = os.environ["NEC5_EXE"]
    meta = dict(
        nec5_exe=exe,
        nec5_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest(),
        source_f=args.source_f,
    )
    print(meta, flush=True)
    out = []
    for L in LENGTHS:
        for a in RADII:
            r8 = row(args.medium, L, 8, args.source_f, a)
            r16 = row(args.medium, L, 16, args.source_f, a)
            mw_inf = 2 * complex(r16["momwire"]).real - complex(r8["momwire"]).real
            n5_inf = 2 * complex(r16["nec5"]).real - complex(r8["nec5"]).real
            rec = dict(
                L=L,
                a=a,
                omega=2 * math.log(L / a),
                rows=[r8, r16],
                momwire_R_inf=mw_inf,
                nec5_R_inf=n5_inf,
                frac_pct=100 * (n5_inf - mw_inf) / complex(r16["momwire"]).real,
            )
            line = (
                f"L={L:4.2f} a={a * 1000:6.3f} mm Omega={rec['omega']:5.2f} "
                f"segs {r8['segs']}/{r16['segs']}  momwire R_inf {mw_inf:.6g}  "
                f"NEC-5 R_inf {n5_inf:.6g}  dR/R {rec['frac_pct']:+.4f} %"
            )
            if args.medium == "free":
                f_hz = 7.1e6 * math.sqrt(EPS_R)
                r_tri = 20 * math.pi**2 * (L / (C0 / f_hz)) ** 2 / math.sqrt(EPS_R)
                rec["R_tri"] = r_tri
                rec["momwire_over_tri"] = mw_inf / r_tri
                rec["nec5_over_tri"] = n5_inf / r_tri
                line += (
                    f"  R/R_tri momwire {rec['momwire_over_tri']:.5f} "
                    f"NEC-5 {rec['nec5_over_tri']:.5f}"
                )
            out.append(rec)
            print(line, flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
