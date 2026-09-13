"""momwire#1027 step 2(b), P2b.3: refine the source region with the ports
matched.

#1027's `refine` never touches the feed wire (two 5 mm segments) or the first
graded panel beside it (h_node = 12.5 mm, two segments). This builder adds
`source_f`: the feed wire gets 2 * source_f segments, keeping the knot source
at its centre, and both graded neighbours use h_node = 12.5 mm / source_f. So
each doubling halves the feed segments and their neighbours. source_f = 1 is
#1027's deck exactly, checked by deck sha against `rod_ladder.build`.

momwire runs with the feed parity patched to even, so it shares NEC-5's knot
source. Segment counts and the knot are asserted equal at every rung.

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/1027-rod/step2b_source_refine.py [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType

import rod_ladder as rl
from step2b_port import SOIL_A, momwire_even_parity, nec5_segs, port_check

from antennaknobs.engines.momwire import MomwireEngine
from antennaknobs.engines.nec5 import NEC5Engine
from antennaknobs.wire_catalog import Wire, graded_wire

H_NODE = 0.0125  # graded_wire's default, the value #1027's deck uses


class SourceRefinedRod(rl.BuriedRodBuilder):
    default_params = MappingProxyType(
        {**rl.BuriedRodBuilder.default_params, "source_f": 1}
    )

    def build_wires(self):
        eps = 0.005
        top = -abs(self.rod_depth)
        bot = top - abs(self.rod_len)
        mid = 0.5 * (top + bot)
        lo, hi = (0.0, 0.0, mid - eps), (0.0, 0.0, mid + eps)
        h = rl.REST_H / max(1, int(self.refine))
        f = int(self.source_f)
        feed_nseg = None if f == 1 else 2 * f
        return [
            graded_wire((0.0, 0.0, bot), lo, toward="p1", h_node=H_NODE / f, rest_h=h),
            Wire(lo, hi, n_seg=feed_nseg, ex=1 + 0j),
            graded_wire(hi, (0.0, 0.0, top), toward="p0", h_node=H_NODE / f, rest_h=h),
        ]


def build(L, refine, f):
    b = SourceRefinedRod()
    b.nominal_nsegs = 42
    b.design_eps_r, b.design_sigma = 13.0, 0.005
    b.rod_len, b.rod_depth, b.refine, b.source_f = L, 0.20, refine, f
    return b


def deck_sha(b):
    deck = NEC5Engine(b, ground=SOIL_A).deck([b.freq])
    return hashlib.sha256(deck.encode()).hexdigest()


def row(L, refine, f):
    b = build(L, refine, f)
    if f == 1 and deck_sha(b) != deck_sha(rl.build("A", L, 0.20, 42, refine=refine)):
        raise RuntimeError("source_f = 1 does not reproduce #1027's deck")
    zn = complex(NEC5Engine(b, ground=SOIL_A).impedance()[0])
    n5 = nec5_segs(b)
    with momwire_even_parity():
        eng = MomwireEngine(b, ground=SOIL_A)
        segs, on_knot, s_f, gap = port_check(eng)
        if segs != n5 or not on_knot:
            raise RuntimeError(
                f"port not matched: momwire {segs} segs vs NEC-5 {n5}, "
                f"feed at {s_f} is {gap:.3g} m from a knot"
            )
        zm = complex(eng.impedance()[0])
    return dict(
        refine=refine,
        source_f=f,
        segs=segs,
        momwire=str(zm),
        nec5=str(zn),
        dR=zn.real - zm.real,
    )


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
    for L in (0.15, 0.60):
        for f in (1, 2, 4):
            r8, r16 = row(L, 8, f), row(L, 16, f)
            d_inf = 2.0 * r16["dR"] - r8["dR"]
            mw8, mw16 = complex(r8["momwire"]).real, complex(r16["momwire"]).real
            n8, n16 = complex(r8["nec5"]).real, complex(r16["nec5"]).real
            rec = dict(
                L=L,
                source_f=f,
                rows=[r8, r16],
                momwire_R_inf=2.0 * mw16 - mw8,
                nec5_R_inf=2.0 * n16 - n8,
                dR_inf=d_inf,
                R16=mw16,
                frac_pct=100.0 * d_inf / mw16,
            )
            out.append(rec)
            print(
                f"L={L:4.2f} F={f} segs {r8['segs']}/{r16['segs']}  "
                f"momwire R_inf {rec['momwire_R_inf']:.6g}  "
                f"NEC-5 R_inf {rec['nec5_R_inf']:.6g}  dR/R {rec['frac_pct']:+.4f} %",
                flush=True,
            )
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
