"""Control for probe_u1_gates (f): is the catalog counterpoise's razor/bspline
gap the DETACHED route, or razor's end treatment on a feed 25 mm from a free
end? Same radiator, same ladder, three grounds: with the buried screen
(detached route), without it over the same soil (razor's shipped
above-ground Sommerfeld route, no buried code at all), and in free space.

    python scratch/razor-buried-u1/probe_f_control.py
"""

import math
import warnings

import numpy as np

from momwire import BSplineSolver, RazorSolver

warnings.filterwarnings("ignore")
lam = 299792458.0 / 7.1e6
h = 0.25 * lam
radiator = np.array([(0, 0, 0.5), (0, 0, 0.55), (0, 0, 0.5 + h)])
radials = [
    np.array([(0, 0, -0.15), (0.6 * h * math.cos(t), 0.6 * h * math.sin(t), -0.15)])
    for t in (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
]
for r in radials:
    r[np.abs(r) < 1e-9] = 0.0
soil = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")
for label, wires, ground, nrad in (
    ("screen", [radiator] + radials, soil, 4),
    ("no screen, soil", [radiator], soil, 0),
    ("no screen, free", [radiator], {}, 0),
):
    for m in (1, 2, 4, 8):
        kw = dict(
            wires=wires,
            n_per_edge_per_wire=[[2 * m, 20 * m]] + [[6 * m]] * nrad,
            feeds=[(0, 0.025, 1 + 0j)],
            wavelength=lam,
            wire_radius=0.001,
            **ground,
        )
        zr = complex(RazorSolver(**kw, nec5_quadrature=True).compute_impedance()[0])
        zb = complex(BSplineSolver(**kw).compute_impedance()[0])
        print(
            f"{label:16s} x{m} razor {zr:.2f} bspline {zb:.2f} "
            f"|d| {abs(zr - zb):8.2f} rel {abs(zr - zb) / abs(zb):.4f}",
            flush=True,
        )
