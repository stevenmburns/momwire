"""momwire#1027 step 2(b), P2b.2: does the per-engine port split move the
published `buried_radial_vertical` numbers?

The default BRV deck's fed rise reaches momwire as one segment with the delta
gap mid-segment and NEC-5 as two segments with EX at the knot. This solves the
default deck at the shipped `nominal_nsegs` and at 2x on NEC-5, on stock
momwire, and on momwire with the feed parity patched to even (the knot port).
It asserts that the patched mesh equals NEC-5's and that the feed is on a knot
before reading R.

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/1027-rod/step2b_brv.py [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from step2b_port import momwire_even_parity, port_check

from antennaknobs.designs.verticals.buried_radial_vertical import Builder
from antennaknobs.engines.momwire import MomwireEngine
from antennaknobs.engines.nec5 import NEC5Engine


def nec5_segs(b, g):
    deck = NEC5Engine(b, ground=g).deck([b.freq])
    return sum(int(ln.split()[2]) for ln in deck.splitlines() if ln.startswith("GW "))


def one_mesh(nn):
    b = Builder()
    b.nominal_nsegs = nn
    g = ("finite", b.design_eps_r, b.design_sigma)
    zn = complex(NEC5Engine(b, ground=g).impedance()[0])
    n5 = nec5_segs(b, g)
    stock = MomwireEngine(b, ground=g)
    s_segs, s_knot, _, _ = port_check(stock)
    z_odd = complex(stock.impedance()[0])
    with momwire_even_parity():
        eng = MomwireEngine(b, ground=g)
        e_segs, e_knot, s_f, gap = port_check(eng)
        if e_segs != n5 or not e_knot:
            raise RuntimeError(
                f"port not matched: momwire {e_segs} segs vs NEC-5 {n5}, "
                f"feed at {s_f} is {gap:.3g} m from a knot"
            )
        z_even = complex(eng.impedance()[0])
    rec = dict(
        nominal_nsegs=nn,
        nec5=str(zn),
        nec5_segs=n5,
        momwire_stock=str(z_odd),
        momwire_stock_segs=s_segs,
        momwire_stock_feed_on_knot=s_knot,
        momwire_matched=str(z_even),
        momwire_matched_segs=e_segs,
        R_move_rel=abs(z_even.real - z_odd.real) / z_odd.real,
        dR_stock=zn.real - z_odd.real,
        dR_matched=zn.real - z_even.real,
    )
    print(
        f"nominal_nsegs={nn}: NEC-5 {zn:.6g} ({n5} segs)\n"
        f"   momwire stock   {z_odd:.10g} ({s_segs} segs, feed on knot {s_knot})  "
        f"dR {rec['dR_stock']:+.4f}\n"
        f"   momwire matched {z_even:.10g} ({e_segs} segs, feed on knot {e_knot})  "
        f"dR {rec['dR_matched']:+.4f}\n"
        f"   momwire R move on matching the port: {rec['R_move_rel']:.2e} relative",
        flush=True,
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    exe = os.environ["NEC5_EXE"]
    meta = dict(
        nec5_exe=exe, nec5_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest()
    )
    print(meta, flush=True)
    out = [one_mesh(nn) for nn in (21, 42)]
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
