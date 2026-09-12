"""momwire#956 re-adjudication: momwire-on-the-branch vs NEC-5, second machine.

Laptop-control read 78.154+46.445j (momwire) vs 78.080+45.974j (NEC-5) on the
`buried_radial_vertical` connected deck, 4 radials, hub 0.15 m, soil A, at
nominal_nsegs 42/84/168 with Richardson on each engine's own order. This
reproduces it on Haswell.

Both engines ladder through the SAME builder knob, so a rung is one geometry
meshed two ways. The segment/GW counts are printed per rung because three arms
in an earlier unit silently did nothing and the check that catches it is
confirming the decks DIFFER before reading their outputs.
"""

import sys
import time
import warnings

warnings.filterwarnings("ignore")

from antennaknobs.designs.verticals.buried_radial_vertical import (  # noqa: E402
    Builder,
)
from antennaknobs.engines.momwire import MomwireEngine  # noqa: E402
from antennaknobs.engines.nec5 import NEC5Engine  # noqa: E402

SOIL_A = (13.0, 0.005)
RUNGS = [int(x) for x in (sys.argv[1:] or ["42", "84", "168"])]


def builder(nn):
    b = Builder()
    b.n_radials = 4
    b.depth = 0.15
    b.nominal_nsegs = nn
    b.design_eps_r, b.design_sigma = SOIL_A
    return b


def richardson(vals, ns, p):
    """Two-rung Richardson on each engine's own convergence order."""
    (n0, z0), (n1, z1) = (ns[-2], vals[-2]), (ns[-1], vals[-1])
    r = (n1 / n0) ** p
    return z1 + (z1 - z0) / (r - 1.0)


def main():
    g = ("finite", *SOIL_A)
    zm, zn, ns = [], [], []
    print(
        f"{'nsegs':>6} {'segs':>6} {'gw':>4} {'momwire':>24} {'NEC-5':>24} "
        f"{'dR':>9} {'dX':>9} {'s':>7}"
    )
    for nn in RUNGS:
        b = builder(nn)
        eng5 = NEC5Engine(b, ground=g)
        text = eng5.deck([b.freq])
        gw = [ln for ln in text.splitlines() if ln.startswith("GW ")]
        segs = sum(int(ln.split()[2]) for ln in gw)
        t0 = time.time()
        a = complex(MomwireEngine(b, ground=g).impedance()[0])
        t1 = time.time()
        c = complex(eng5.impedance()[0])
        zm.append(a)
        zn.append(c)
        ns.append(segs)
        print(
            f"{nn:6d} {segs:6d} {len(gw):4d} "
            f"{a.real:11.4f}{a.imag:+11.4f}j {c.real:11.4f}{c.imag:+11.4f}j "
            f"{c.real - a.real:+9.4f} {c.imag - a.imag:+9.4f} "
            f"{t1 - t0:7.0f}"
        )
        sys.stdout.flush()
    if len(ns) >= 2:
        if len(set(ns)) != len(ns):
            print("\nWARNING: segment counts REPEAT — the ladder is not a ladder")
        for p in (1.0, 2.0):
            rm = richardson(zm, ns, p)
            rn = richardson(zn, ns, p)
            print(
                f"\nRichardson p={p:.0f}:  momwire {rm.real:.4f}{rm.imag:+.4f}j   "
                f"NEC-5 {rn.real:.4f}{rn.imag:+.4f}j   "
                f"dR {rn.real - rm.real:+.4f}  dX {rn.imag - rm.imag:+.4f}"
            )
        print("\nlaptop read: momwire 78.154+46.445j vs NEC-5 78.080+45.974j")


if __name__ == "__main__":
    main()
