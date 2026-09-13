"""momwire#1027 step 2(b), P2b.1: the same port on both engines.

antennaknobs hands degree-2 momwire an odd feed-segment count (the rod's 10 mm
feed wire becomes ONE segment, fed mid-segment) and NEC-5 an even one (TWO
5 mm segments, EX at the centre knot). This harness patches the parity rule to
"even" for momwire only, so momwire gets NEC-5's mesh and source: a delta gap
at the knot between the feed wire's two segments. NEC-5's deck is unchanged.
No src/ change: the patch is `antennaknobs.engines.momwire._parity_for_solver`,
restored on exit.

Before a row is read, the harness checks that momwire's segment count equals
the NEC-5 deck's and that momwire's feed arclength lies on a knot.

Modes:
  matched   P2b.1: soil A, d = 0.2 m, refine 8 -> 16, L = 0.15 / 0.60 / 2.40 m
  segment   reading: momwire feed_model="segment" on its native (odd) mesh,
            L = 0.15 / 0.60 m, against the same NEC-5 rows

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/1027-rod/step2b_port.py MODE [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rod_ladder as rl

import antennaknobs.engines.momwire as mwe
from antennaknobs.engines.momwire import MomwireEngine
from antennaknobs.engines.nec5 import NEC5Engine

SOIL_A = ("finite", 13.0, 0.005)


@contextmanager
def momwire_even_parity():
    orig = mwe._parity_for_solver
    mwe._parity_for_solver = lambda solver, solver_kwargs: "even"
    try:
        yield
    finally:
        mwe._parity_for_solver = orig


def port_check(eng):
    """(n_segments, feed on a knot?, feed arclength, nearest knot distance)."""
    (w, s_f, _v) = eng._feeds[0]
    poly = np.asarray(eng._polylines[w], dtype=float)
    knots = mwe._polyline_knots(poly, eng._edge_segments[w])
    arcs = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(knots, axis=0), axis=1))]
    )
    n_segs = sum(sum(e) for e in eng._edge_segments)
    gap = float(np.min(np.abs(arcs - s_f)))
    return n_segs, gap < 1e-12, float(s_f), gap


def nec5_segs(b):
    deck = NEC5Engine(b, ground=SOIL_A).deck([b.freq])
    return sum(int(ln.split()[2]) for ln in deck.splitlines() if ln.startswith("GW "))


def row(L, refine, mode):
    b = rl.build("A", L, 0.20, 42, refine=refine)
    zn = complex(NEC5Engine(b, ground=SOIL_A).impedance()[0])
    n5_segs = nec5_segs(b)
    if mode == "matched":
        with momwire_even_parity():
            eng = MomwireEngine(b, ground=SOIL_A)
            n_segs, on_knot, s_f, gap = port_check(eng)
            if n_segs != n5_segs or not on_knot:
                raise RuntimeError(
                    f"port not matched: momwire {n_segs} segs vs NEC-5 {n5_segs}, "
                    f"feed at {s_f} is {gap:.3g} m from a knot"
                )
            zm = complex(eng.impedance()[0])
    else:
        eng = MomwireEngine(b, ground=SOIL_A, solver_kwargs={"feed_model": "segment"})
        n_segs, on_knot, s_f, gap = port_check(eng)
        zm = complex(eng.impedance()[0])
    return dict(
        refine=refine,
        momwire=str(zm),
        nec5=str(zn),
        dR=zn.real - zm.real,
        momwire_segs=n_segs,
        nec5_segs=n5_segs,
        feed_on_knot=on_knot,
        feed_arclength=s_f,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["matched", "segment"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    exe = os.environ["NEC5_EXE"]
    meta = dict(
        nec5_exe=exe, nec5_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest()
    )
    print(meta, flush=True)
    lengths = (0.15, 0.60, 2.40) if args.mode == "matched" else (0.15, 0.60)
    out = []
    for L in lengths:
        r8, r16 = row(L, 8, args.mode), row(L, 16, args.mode)
        d_inf = 2.0 * r16["dR"] - r8["dR"]
        R16 = complex(r16["momwire"]).real
        rec = dict(
            L=L, rows=[r8, r16], dR_inf=d_inf, R16=R16, frac_pct=100 * d_inf / R16
        )
        out.append(rec)
        print(
            f"L={L:4.2f} [{args.mode}] segs momwire {r8['momwire_segs']}/{r16['momwire_segs']} "
            f"NEC-5 {r8['nec5_segs']}/{r16['nec5_segs']}  feed on knot "
            f"{r8['feed_on_knot']}/{r16['feed_on_knot']}\n"
            f"   momwire R8/R16 {complex(r8['momwire']).real:.6g}/{R16:.6g}  "
            f"NEC-5 R8/R16 {complex(r8['nec5']).real:.6g}/{complex(r16['nec5']).real:.6g}  "
            f"dR8 {r8['dR']:+.5g} dR16 {r16['dR']:+.5g}  dR/R {rec['frac_pct']:+.4f} %",
            flush=True,
        )
    if args.out is not None:
        args.out.write_text(
            json.dumps(dict(meta=meta, mode=args.mode, result=out), indent=1)
        )
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
