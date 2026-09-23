"""momwire#1156 probe 3: SG's loading SHIFT onto bspline's and razor's under
refinement, on a wholly-buried dipole, a detached (mixed) deck, the detached
hub, and a crossing deck. Usage: probe3_convergence.py DECK [M ...]."""

import sys
import time
import warnings

import numpy as np

sys.path.insert(0, "tests")

from momwire._wire_loading import DistributedRLC  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from test_jacket_in_soil_1154 import dipole  # noqa: E402
from test_razor_crossing_loading_1149 import crossing  # noqa: E402
from test_razor_detached_1149 import detached, detached_hub  # noqa: E402

LOADS = {
    "L'=1u": dict(distributed_rlc=DistributedRLC("series", l=1e-6)),
    "sigma": dict(wire_conductivity=3.5e7),
    "jacket": dict(insulation_radius=1.8e-3, insulation_eps_r=3.5),
}
TRUNKS = (SinusoidalGalerkinSolver, BSplineSolver, RazorSolver)


def z(cls, d):
    d = dict(d)
    extra = {}
    if cls is RazorSolver:
        d.pop("junctions", None)
        extra = dict(nec5_quadrature=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return complex(np.ravel(cls(**d, **extra).compute_impedance()[0])[0])


def per_wire(d, kw):
    nw = len(d["wires"])
    return {k: (v if k == "distributed_rlc" else [v] * nw) for k, v in kw.items()}


DECKS = {
    "dipole": lambda m: dipole(n=20 * m),
    "detached": lambda m: detached(m, wire_radius=1e-3),
    "detached_hub": lambda m: dict(detached_hub(m), wire_radius=1e-3),
    "crossing": lambda m: dict(crossing(m), wire_radius=1e-3),
}
if __name__ == "__main__":
    name = sys.argv[1]
    ms = [int(x) for x in sys.argv[2:]] or [1, 2, 4]
    gaps = {(lname, cls.__name__): [] for lname in LOADS for cls in TRUNKS[1:]}
    for m in ms:
        d0 = DECKS[name](m)
        shifts = {}
        for cls in TRUNKS:
            t = time.time()
            z0 = z(cls, d0)
            for lname, kw in LOADS.items():
                shifts[(lname, cls)] = z(cls, dict(d0, **per_wire(d0, kw))) - z0
            print(
                f"{name} m={m} {cls.__name__:26s} bare {z0:.3f} "
                f"({time.time() - t:.1f} s)",
                flush=True,
            )
        for lname in LOADS:
            sg = shifts[(lname, SinusoidalGalerkinSolver)]
            row = f"    {lname:7s} SG {sg:.3f}"
            for cls in TRUNKS[1:]:
                g = abs(sg - shifts[(lname, cls)])
                gaps[(lname, cls.__name__)].append(g)
                row += f" | {cls.__name__[:5]} {shifts[(lname, cls)]:.3f} gap {g:.3f}"
            print(row, flush=True)
    print("gap series:")
    for key, g in gaps.items():
        print(f"  {key[0]:7s} SG-{key[1][:5]}: " + " -> ".join(f"{x:.3f}" for x in g))
