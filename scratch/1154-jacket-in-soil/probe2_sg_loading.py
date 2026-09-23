"""momwire#1154 probe 2: is SG's fully-buried loading applied at the right w?

SinusoidalGalerkinSolver's fully-buried fill swaps self.k to k_m (complex)
and `_apply_loading` evaluates loading_for(self, k * self.c). Compare the
loading SHIFT (loaded - bare) of each formulation on a buried dipole, for a
pure series resistance (w-independent: any w gives the same Z') and a pure
series inductance (w-dependent).
"""

import warnings

import numpy as np

from momwire._wire_loading import DistributedRLC
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

WL = 299792458.0 / 7.0e6
SOIL = (13.0, 0.005)


def deck(n, **kw):
    length, depth = 5.0, 0.5
    pts = np.array([(-0.5 * length, 0.0, -depth), (0.5 * length, 0.0, -depth)])
    return dict(
        wires=[pts],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 0.5 * length, 1 + 0j)],
        wavelength=WL,
        wire_radius=1e-3,
        ground_z=0.0,
        ground_eps=SOIL,
        ground_model="sommerfeld",
        **kw,
    )


def z(cls, n, **kw):
    extra = dict(nec5_quadrature=True) if cls is RazorSolver else {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = complex(cls(**deck(n, **kw), **extra).compute_impedance()[0])
    msgs = sorted({str(x.category.__name__) + ": " + str(x.message)[:80] for x in w})
    return val, msgs


LOADS = {
    "R' = 20 ohm/m": dict(distributed_rlc=DistributedRLC("series", r=20.0)),
    "L' = 1 uH/m": dict(distributed_rlc=DistributedRLC("series", l=1e-6)),
}

if __name__ == "__main__":
    for n in (20, 40):
        for cls in (BSplineSolver, RazorSolver, SinusoidalGalerkinSolver):
            z0, m0 = z(cls, n)
            print(f"{cls.__name__:26s} n={n} bare {z0:.3f}", flush=True)
            for name, kw in LOADS.items():
                zl, ml = z(cls, n, **kw)
                print(f"    {name:14s} shift {zl - z0:.3f}   {ml}", flush=True)
