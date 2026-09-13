"""momwire#1027 steps 2(b)/(c), P2b.4: the clean source ladder at two radii.

P2b.3 refined the source region by dividing graded_wire's h_node, which moved
the panel boundaries and re-meshed the middle of the rod non-monotonically. This
ladder holds the boundaries (h_node = 12.5 mm, growth 4) and divides EVERY
segment by F instead: `per_panel` = 2F on both graded wires, the feed wire at 2F
segments with its knot source kept at the centre, and `rest_h` = 3.125 mm / F.
F = 1 is #1027's refine-8 deck exactly, checked by deck sha at a = 0.5 mm.

Two radii separate the thin-wire regime from the refinement: at a = 0.5 mm the
far and source segments fall to Delta/a = 1.6 and 2.5 by F = 4, while at
a = 0.1 mm Delta/a stays >= 7.8 everywhere. The radius reaches both engines
through a per-wire `WireSpec`, checked on the GW card and on momwire's
`_wire_radius`.

Ports matched: momwire's feed parity patched to even, segment counts and the
knot source asserted equal to NEC-5's at every rung.

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/1027-rod/step2b_uniform.py [--out F]
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
from antennaknobs.wire_catalog import Wire, WireSpec, graded_wire

BASE_REST_H = rl.REST_H / 8  # #1027's refine-8 far segment, 3.125 mm


class UniformRod(rl.BuriedRodBuilder):
    default_params = MappingProxyType(
        {**rl.BuriedRodBuilder.default_params, "uniform_f": 1, "radius_m": 0.0005}
    )

    def build_wires(self):
        eps = 0.005
        top = -abs(self.rod_depth)
        bot = top - abs(self.rod_len)
        mid = 0.5 * (top + bot)
        lo, hi = (0.0, 0.0, mid - eps), (0.0, 0.0, mid + eps)
        f = int(self.uniform_f)
        h = BASE_REST_H / f
        spec = WireSpec(radius=float(self.radius_m))
        feed_nseg = None if f == 1 else 2 * f
        return [
            graded_wire(
                (0.0, 0.0, bot), lo, toward="p1", per_panel=2 * f, rest_h=h, spec=spec
            ),
            Wire(lo, hi, n_seg=feed_nseg, ex=1 + 0j, spec=spec),
            graded_wire(
                hi, (0.0, 0.0, top), toward="p0", per_panel=2 * f, rest_h=h, spec=spec
            ),
        ]


def build(L, f, a):
    b = UniformRod()
    b.nominal_nsegs = 42
    b.design_eps_r, b.design_sigma = 13.0, 0.005
    b.rod_len, b.rod_depth, b.uniform_f, b.radius_m = L, 0.20, f, a
    return b


def deck_of(b):
    return NEC5Engine(b, ground=SOIL_A).deck([b.freq])


def row(L, f, a):
    b = build(L, f, a)
    deck = deck_of(b)
    if f == 1 and a == 0.0005:
        ref = deck_of(rl.build("A", L, 0.20, 42, refine=8))
        if (
            hashlib.sha256(deck.encode()).digest()
            != hashlib.sha256(ref.encode()).digest()
        ):
            raise RuntimeError(
                "F = 1, a = 0.5 mm does not reproduce #1027's refine-8 deck"
            )
    radii = {float(ln.split()[9]) for ln in deck.splitlines() if ln.startswith("GW ")}
    if radii != {a}:
        raise RuntimeError(f"NEC-5 radius {radii}, asked {a}")
    zn = complex(NEC5Engine(b, ground=SOIL_A).impedance()[0])
    n5 = nec5_segs(b)
    with momwire_even_parity():
        eng = MomwireEngine(b, ground=SOIL_A)
        if float(eng._wire_radius) != a:
            raise RuntimeError(f"momwire radius {eng._wire_radius}, asked {a}")
        segs, on_knot, s_f, gap = port_check(eng)
        if segs != n5 or not on_knot:
            raise RuntimeError(
                f"port not matched: momwire {segs} segs vs NEC-5 {n5}, "
                f"feed at {s_f} is {gap:.3g} m from a knot"
            )
        zm = complex(eng.impedance()[0])
    frac = 100.0 * (zn.real - zm.real) / zm.real
    rec = dict(L=L, F=f, a=a, segs=segs, momwire=str(zm), nec5=str(zn), frac_pct=frac)
    print(
        f"a={a * 1000:4.2f} mm L={L:4.2f} F={f} segs={segs:4d}  momwire R {zm.real:.6g}  "
        f"NEC-5 R {zn.real:.6g}  dR/R {frac:+.4f} %",
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
    out = [
        row(L, f, a) for a in (0.0005, 0.0001) for L in (0.15, 0.60) for f in (1, 2, 4)
    ]
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
