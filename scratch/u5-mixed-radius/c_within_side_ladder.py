"""U5 step (c): a radius spread WITHIN the above side, against NEC-5 (#1140).

Step (b) (`b_rod_ladder.py`) settled ONE radius per side. This asks the next
question: when the ABOVE side itself carries two radii, what radius does the
crossing node's point test take?

The deck is (b)'s rod with the above side split, which it already is
physically — three wires, and only the middle one touches the node:

    wire 0  rise, hub -> node                 a_rise      BELOW
    wire 1  fed wire, node -> EPS             a_node      ABOVE, AT the node
    wire 2  radiator, EPS -> lambda/4         a_rad       ABOVE, away from it

Step (b) held `a_node == a_rad` and called the pair `a_top`. Letting them
differ is exactly WA7ARK's shape: 8.138E-4 on his jumper at the rod, .002794
on the verticals above it.

THE QUESTION. `crossing_side_radii` resolves `a_above` from EVERY above wire
in the deck and refuses when they differ. But the scope rule already grants a
crossing junction exactly ONE above member, so at the node there is one above
wire with one radius. Hypothesis: `a_above` is wire 1's radius, wire 2 is
ordinary geometry the normal fill already handles at its own radius, and this
half of U5 is a scope narrowing rather than a derivation.

WHY THIS NEEDS NEC-5. The house e~=1 collapse cannot adjudicate it. Measured
2026-09-19 on this geometry: forcing a_above to the node member (0.001) lands
0.0218 ohm from the free-space truth and forcing it to the far wire (0.004)
lands 0.0081 — the deliberately WRONG pick nearer, both inside 0.022 ohm on a
1083 ohm reactance. That is not a tie, it is blindness: at e~=1 the interface
is gone, and these terms exist only because there is an interface. The
question only has an answer at a real one.

So each rung runs BOTH candidate spellings of a_above against the engine:

  node   a_above = wire 1's radius (the hypothesis)
  far    a_above = wire 2's radius (the control that must lose if the
         hypothesis is right; if it wins, or if neither converges, the
         node radius is a derivation and this script says so)

Asserted per rung, as step (b) asserts them: momwire's segment count equals
NEC-5's, the feed lands on a knot, the crossing fill is entered exactly once.
The control config (a_node == a_rad) must reproduce b_rod_ladder's own numbers,
which is what says the deck split changed no physics.

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/u5-mixed-radius/c_within_side_ladder.py CONFIG [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType

from momwire import _below_interface
from b_rod_ladder import EPS, REST_H, SOIL_A, nec5_segs

from antennaknobs.designs.verticals.buried_radial_vertical import Builder
from antennaknobs.engines.momwire import MomwireEngine
from antennaknobs.engines.nec5 import NEC5Engine
from antennaknobs.wire_catalog import Wire, WireSpec, graded_wire

# (a_node, a_rad, a_rise). The control repeats b_rod_ladder's "control" rung
# so the split deck is pinned against the unsplit study.
CONFIGS = {
    "control": (0.00025, 0.00025, 0.00025),
    "above2": (0.00025, 0.000125, 0.00025),
    "above4": (0.00025, 0.0000625, 0.00025),
    "above05": (0.000125, 0.00025, 0.00025),
    # WA7ARK's own ratio, 8.138E-4 : .002794 = 1 : 3.43, on this geometry.
    "wa7ark": (0.00025, 0.000857, 0.00025),
}


class WithinSideRod(Builder):
    """b_rod_ladder's TwoRadiusRod with the above side's two wires given
    their own radii. At `a_node == a_rad` it IS that deck, wire for wire."""

    default_params = MappingProxyType(
        {
            **Builder.default_params,
            "uniform_r": 1,
            "a_node": 0.00025,
            "a_rad": 0.00025,
            "a_rise": 0.00025,
        }
    )

    def build_wires(self):
        r = int(self.uniform_r)
        height = 0.25 * self.design_wavelength * self.length_factor
        node = (0.0, 0.0, 0.0)
        hub = (0.0, 0.0, -self.depth)
        at_node = WireSpec(radius=float(self.a_node))
        radiator = WireSpec(radius=float(self.a_rad))
        rise = WireSpec(radius=float(self.a_rise))
        return [
            graded_wire(
                hub, node, toward="p1", per_panel=2 * r, rest_h=REST_H / r, spec=rise
            ),
            Wire(
                node,
                (0.0, 0.0, EPS),
                n_seg=None if r == 1 else 2 * r,
                ex=1 + 0j,
                spec=at_node,
            ),
            graded_wire(
                (0.0, 0.0, EPS),
                (0.0, 0.0, height),
                toward="p0",
                per_panel=2 * r,
                rest_h=0.25 * self.design_wavelength / self.nominal_nsegs / r,
                spec=radiator,
            ),
        ]


@contextmanager
def a_above_is(a_above, a_below):
    """Force the node's two radii to `(a_above, a_below)`.

    c2's `scope` patch silences the refusal inside `crossing_junctions` by
    handing it uniform radii, but `crossing_side_radii` is reached a SECOND
    way — `BSplineSolver._two_radius_crossing` (`bspline.py:1906`) calls it
    directly, and that is the call whose answer becomes `ctx.a_above` /
    `ctx.a_below` at `bspline.py:2074`. On a within-side spread it raises, so
    the deck never reaches the fill at all.

    Patching it here is not merely a bypass: on a spread deck there IS no
    side-wide above radius to resolve, so naming the candidate explicitly is
    the only way to ask the question this ladder asks.
    """
    orig = _below_interface.crossing_side_radii
    _below_interface.crossing_side_radii = lambda media, radii: (
        float(a_above),
        float(a_below),
    )
    try:
        yield
    finally:
        _below_interface.crossing_side_radii = orig


def build(L, r, a_node, a_rad, a_rise, nn=42):
    b = WithinSideRod()
    b.nominal_nsegs = nn
    b.design_eps_r, b.design_sigma = 13.0, 0.005
    b.depth = L
    b.uniform_r = r
    b.a_node, b.a_rad, b.a_rise = a_node, a_rad, a_rise
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=sorted(CONFIGS))
    ap.add_argument("--lengths", type=float, nargs="+", default=[0.30, 0.60, 1.20])
    ap.add_argument("--refines", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    a_node, a_rad, a_rise = CONFIGS[args.config]
    exe = os.environ["NEC5_EXE"]
    meta = dict(
        config=args.config,
        a_node=a_node,
        a_rad=a_rad,
        a_rise=a_rise,
        nec5_exe=exe,
        nec5_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest(),
    )
    print(meta, flush=True)

    out = []
    for L in args.lengths:
        for r in args.refines:
            b = build(L, r, a_node, a_rad, a_rise)
            n5 = nec5_segs(b)
            zn = complex(NEC5Engine(b, ground=SOIL_A).impedance()[0])
            row = {"L": L, "r": r, "nec5": repr(zn), "nec5_segs": n5, "cand": {}}
            # Both candidate spellings of a_above, same deck, same mesh, same
            # engine reading. `a_rise` is untouched: the below side has one
            # radius here, so nothing about a_below is in question.
            for tag, a_above in (("node", a_node), ("far", a_rad)):
                with a_above_is(a_above, a_rise):
                    zm = complex(MomwireEngine(b, ground=SOIL_A).impedance()[0])
                row["cand"][tag] = {
                    "a_above": a_above,
                    "z": repr(zm),
                    "dZ": repr(zm - zn),
                }
                print(
                    f"L={L:.2f} r={r} segs={n5}  NEC-5 {zn.real:.2f}{zn.imag:+.2f}j"
                    f"  a_above={tag}({a_above})  momwire {zm.real:.3f}{zm.imag:+.3f}j"
                    f"  dZ {abs(zm - zn):.4f} ohm",
                    flush=True,
                )
            n, f = row["cand"]["node"], row["cand"]["far"]
            dn, df = abs(complex(n["dZ"])), abs(complex(f["dZ"]))
            row["verdict"] = "tie" if dn == df else ("node" if dn < df else "far")
            print(
                f"    -> node {dn:.4f} ohm vs far {df:.4f} ohm: {row['verdict']} wins"
                f" by {abs(dn - df):.4f}",
                flush=True,
            )
            out.append(row)
    if args.out:
        args.out.write_text(json.dumps({"meta": meta, "rows": out}, indent=1))
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
