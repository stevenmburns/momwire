"""momwire#1156 probe 2: the issue's repro (1154 probe2's buried dipole) on
all three trunks after the fix, plus a detached and a crossing deck on SG.
Loading SHIFTS (loaded - bare) for R', L', conductivity and a jacket."""

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
from test_razor_detached_1149 import detached_hub  # noqa: E402

LOADS = {
    "R'=20": dict(distributed_rlc=DistributedRLC("series", r=20.0)),
    "L'=1u": dict(distributed_rlc=DistributedRLC("series", l=1e-6)),
    "sigma": dict(wire_conductivity=3.5e7),
    "jacket": dict(insulation_radius=1.8e-3, insulation_eps_r=3.5),
}


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
    out = {}
    for key, v in kw.items():
        out[key] = v if key == "distributed_rlc" else [v] * nw
    return out


DECKS = {
    "dipole": lambda: dipole(n=20),
    "detached_hub": lambda: dict(detached_hub(1), wire_radius=1e-3),
    "crossing": lambda: dict(crossing(1), wire_radius=1e-3),
}
if __name__ == "__main__":
    names = sys.argv[1:] or list(DECKS)
    for name in names:
        d0 = DECKS[name]()
        for cls in (SinusoidalGalerkinSolver, BSplineSolver, RazorSolver):
            t = time.time()
            z0 = z(cls, d0)
            print(f"{name:12s} {cls.__name__:26s} bare {z0:.3f}", flush=True)
            for lname, kw in LOADS.items():
                zl = z(cls, dict(d0, **per_wire(d0, kw)))
                print(f"    {lname:7s} shift {zl - z0:.3f}", flush=True)
            print(f"    ({time.time() - t:.1f} s)", flush=True)
