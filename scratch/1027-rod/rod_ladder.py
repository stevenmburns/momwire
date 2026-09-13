"""momwire#956: the wholly buried vertical rod, laddered in length and depth.

The discriminating measurement. #931 showed the residual is a series dZ that
scales with the RISE -- the conductor running from the buried hub up through
the interface to the node. Two mechanisms fit that equally well:

  * ORIENTATION: the conventions load a VERTICAL conductor in the lossy medium
    differently, and the interface is incidental;
  * CROSSING: the difference is the conductor's passage THROUGH z = 0, and
    momwire's crossing-junction treatment is where it lives.

A rod that is wholly below the interface separates them. If dZ still scales
with its length, it is orientation. If it vanishes, it is the crossing.

NOT `scratch/931-study/probe_rod_ladder.py`: that rod deliberately crosses the
interface (it is `buried_radial_vertical` minus the radials, so it keeps the
rise, the node gap and the radiator above ground). This reuses its ladder
DISCIPLINE and none of its geometry.

Carried over from that script, because both were paid for once already:

  * `rest_h` is PINNED, so lengthening the rod adds panels instead of
    stretching them. Unpinned, the #931 ladder varied mesh fineness and length
    together and its dZ column was reading both.
  * every row prints its GW segment count. If the knob turns and the deck does
    not, it is not a ladder.
  * the burial check reads the EMITTED deck's z coordinates, not the builder's
    parameters -- the deck is what gets solved.
"""

import json
import sys
from types import MappingProxyType
import warnings

warnings.filterwarnings("ignore")

from antennaknobs.designs.verticals.buried_radial_vertical import (  # noqa: E402
    Builder,
)
from antennaknobs.engines.momwire import MomwireEngine  # noqa: E402
from antennaknobs.engines.nec5 import NEC5Engine  # noqa: E402
from antennaknobs.wire_catalog import Wire, graded_wire  # noqa: E402

SOILS = {"B": (10.0, 0.002), "A": (13.0, 0.005), "C": (20.0, 0.020)}
REST_H = 0.025  # pinned; see the docstring


class BuriedRodBuilder(Builder):
    """A centre-fed vertical rod with BOTH ends below the interface.

    `rod_len` metres of conductor whose top sits `rod_depth` below z = 0, fed
    at its centre. Subclasses `buried_radial_vertical` so the grading helpers
    and the soil/frequency plumbing are the ones #931 used; only the wires are
    new, and there is no screen, no node at the interface and nothing above it.
    """

    # DECLARED AS PARAMS, NOT CLASS ATTRIBUTES. `Builder.__setattr__` writes
    # every assignment into `self._params`, but a CLASS attribute wins normal
    # lookup, so `b.rod_len = 2.4` would store 2.4 in the params dict and
    # `self.rod_len` would still read the class default -- silently, forever.
    # The first run of this ladder did exactly that: every row came back with
    # segs = 22 and an identical dZ across a 16x span in length, which the
    # printed segment count caught. A ladder whose deck does not move is not a
    # ladder (the lesson `probe_rod_ladder.py` states, earned again).
    default_params = MappingProxyType(
        {
            **Builder.default_params,
            "rod_len": 0.60,
            "rod_depth": 0.20,
            # `rest_h` is pinned against LENGTH so a longer rod adds panels
            # instead of stretching them (#931's lesson). But that alone
            # leaves NO refinement axis -- the first corrected run had segs
            # identical across nn = 21/42/84, so convergence could not be
            # tested at all. `refine` is the separate knob: it divides the
            # panel size without touching the length, so the two axes move
            # independently, which is the whole point.
            "refine": 1,
        }
    )

    def build_wires(self):
        eps = 0.005
        top = -abs(self.rod_depth)
        bot = top - abs(self.rod_len)
        mid = 0.5 * (top + bot)
        lo, hi = (0.0, 0.0, mid - eps), (0.0, 0.0, mid + eps)
        h = REST_H / max(1, int(self.refine))
        return [
            graded_wire((0.0, 0.0, bot), lo, toward="p1", rest_h=h),
            Wire(lo, hi, ex=1 + 0j),
            graded_wire(hi, (0.0, 0.0, top), toward="p0", rest_h=h),
        ]


def build(soil, rod_len, rod_depth, nn, freq=None, refine=1):
    eps_r, sigma = SOILS[soil] if isinstance(soil, str) else soil
    b = BuriedRodBuilder()
    b.nominal_nsegs = nn
    b.design_eps_r = eps_r
    b.design_sigma = sigma
    b.rod_len = rod_len
    b.rod_depth = rod_depth
    b.refine = refine
    if freq is not None:
        b.freq = b.design_freq = freq
    return b


def row(soil, rod_len, rod_depth, nn, freq=None, basis=None, refine=1):
    b = build(soil, rod_len, rod_depth, nn, freq, refine)
    g = ("finite", b.design_eps_r, b.design_sigma)
    text = NEC5Engine(b, ground=g).deck([b.freq])
    gw = [ln for ln in text.splitlines() if ln.startswith("GW ")]
    segs = sum(int(ln.split()[2]) for ln in gw)
    zs = [float(ln.split()[5]) for ln in gw] + [float(ln.split()[8]) for ln in gw]
    kw = {"basis": basis} if basis else {}
    zm = complex(MomwireEngine(b, ground=g, **kw).impedance()[0])
    zn = complex(NEC5Engine(b, ground=g).impedance()[0])
    return {
        "soil": soil if isinstance(soil, str) else list(soil),
        "rod_len": rod_len,
        "rod_depth": rod_depth,
        "nn": nn,
        "refine": refine,
        "freq": b.freq,
        "basis": basis or "bspline-2",
        "segs": segs,
        "z_max": max(zs),
        "buried": max(zs) < 0,
        "momwire_R": zm.real,
        "momwire_X": zm.imag,
        "nec5_R": zn.real,
        "nec5_X": zn.imag,
        "dR": zn.real - zm.real,
        "dX": zn.imag - zm.imag,
        "deck_sha": __import__("hashlib").sha256(text.encode()).hexdigest()[:12],
    }


def main():
    out = []
    print(
        f"{'L':>5} {'d':>5} {'ref':>4} {'segs':>5} {'z_max':>8} "
        f"{'mw R':>9} {'n5 R':>9} {'dR':>8} {'dR/L':>8}"
    )
    for L in (0.15, 0.30, 0.60, 1.20, 2.40):
        for refine in (1, 2, 4):
            r = row("A", L, 0.20, 42, refine=refine)
            out.append(r)
            print(
                f"{L:5.2f} {0.20:5.2f} {refine:4d} {r['segs']:5d} {r['z_max']:8.4f} "
                f"{r['momwire_R']:9.4f} {r['nec5_R']:9.4f} {r['dR']:+8.4f} "
                f"{r['dR'] / L:+8.3f}" + ("" if r["buried"] else "  <-- NOT BURIED")
            )
        print()
    with open(sys.argv[1] if len(sys.argv) > 1 else "rod_rows.jsonl", "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
