"""momwire#1156 probe 5: a MIXED deck refined EVERYWHERE (probe3's
`detached` keeps its near-plane edges fixed under m, and its bare Z sits
12 ohm apart SG vs bspline, flat). A buried centre-fed wire (5 m, depth
0.5) beside a centre-fed wire 1 m above the plane, uniform n = 10m each.
Loading shift per port, SG vs bspline and razor. `jacket-below` jackets
only the buried wire, so it isolates the charge term's route."""

import sys
import time
import warnings

import numpy as np

sys.path.insert(0, "tests")

from momwire._wire_loading import DistributedRLC  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402

WL7 = 299792458.0 / 7.0e6
TRUNKS = (SinusoidalGalerkinSolver, BSplineSolver, RazorSolver)
NAN = float("nan")
LOADS = {
    "L'=1u": dict(distributed_rlc=DistributedRLC("series", l=1e-6)),
    "sigma": dict(wire_conductivity=[3.5e7, 3.5e7]),
    "jacket": dict(insulation_radius=[1.8e-3] * 2, insulation_eps_r=[3.5] * 2),
    "jacket-below": dict(insulation_radius=[1.8e-3, NAN], insulation_eps_r=[3.5, NAN]),
}


def mixed(m=1, eps=(13.0, 0.005), **kw):
    below = np.array([(-2.5, 0.0, -0.5), (2.5, 0.0, -0.5)])
    above = np.array([(-2.5, 0.0, 1.0), (2.5, 0.0, 1.0)])
    d = dict(
        wires=[below, above],
        n_per_edge_per_wire=[[10 * m], [10 * m]],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        ground_z=0.0,
        ground_eps=eps,
        ground_model="sommerfeld",
    )
    d.update(kw)
    return d


def z(cls, d):
    extra = dict(nec5_quadrature=True) if cls is RazorSolver else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.ravel(cls(**d, **extra).compute_impedance()[0])


if __name__ == "__main__":
    ms = [int(x) for x in sys.argv[1:]] or [1, 2, 4]
    gaps = {}
    for m in ms:
        d0 = mixed(m)
        shifts = {}
        for cls in TRUNKS:
            t = time.time()
            z0 = z(cls, d0)
            for lname, kw in LOADS.items():
                shifts[(lname, cls)] = z(cls, dict(d0, **kw)) - z0
            print(
                f"m={m} {cls.__name__:26s} bare {z0[0]:.3f} {z0[1]:.3f} "
                f"({time.time() - t:.1f} s)",
                flush=True,
            )
        for lname in LOADS:
            for p in range(2):
                sg = shifts[(lname, SinusoidalGalerkinSolver)][p]
                row = f"    {lname:12s} port{p} SG {sg:.3f}"
                for cls in TRUNKS[1:]:
                    o = shifts[(lname, cls)][p]
                    g = abs(sg - o)
                    gaps.setdefault((lname, p, cls.__name__[:5]), []).append(g)
                    row += f" | {cls.__name__[:5]} {o:.3f} gap {g:.3f}"
                print(row, flush=True)
    print("gap series:")
    for key, g in gaps.items():
        print(
            f"  {key[0]:12s} port{key[1]} SG-{key[2]}: "
            + " -> ".join(f"{x:.3f}" for x in g)
        )
