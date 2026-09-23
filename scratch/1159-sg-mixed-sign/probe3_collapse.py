"""momwire#1159 probe 3: the eps~ = 1 collapse on the FULL Y matrix and on
every wire's current with every port driven, for a MIXED deck (#980 D2) and
a CROSSING deck (#980 D3). At eps~ = 1 either deck IS free space, so SG must
equal its own free-space solve on the same wires.

Prints which momwire was imported, so a side-by-side against main's source
(PYTHONPATH=<main>/src) says which one ran.
"""

import sys
import warnings

import numpy as np

import momwire
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

sys.path.insert(0, "scratch/1156-sg-buried-loading")
from probe5_mixed_refined import mixed  # noqa: E402

WL7 = 299792458.0 / 7e6
GROUND = ("ground_z", "ground_eps", "ground_model")


def crossing(n=15, eps=(1.0, 0.0)):
    return dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]),
            np.array([(0.0, 0.0, 0.0), (2.0, 0.0, -0.5)]),
        ],
        n_per_edge_per_wire=[[n], [n]],
        feeds=[(0, 1.0, 1 + 0j), (1, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        junctions=[[(0, "start"), (1, "start")]],
        ground_z=0.0,
        ground_eps=eps,
        ground_model="sommerfeld",
    )


def run(d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SinusoidalGalerkinSolver(**d)
        sol = s.compute_port_solution()
        cur = s.currents_at_knots(sol.coeffs.sum(axis=1))
    return np.asarray(sol.y), [np.asarray(c) for c in cur]


print("momwire from", momwire.__file__)
for name, one in (("mixed(2)", mixed(2, eps=(1.0, 0.0))), ("crossing", crossing())):
    free = {k: v for k, v in one.items() if k not in GROUND}
    y1, c1 = run(one)
    yf, cf = run(free)
    rel = np.abs(y1 - yf) / np.abs(yf)
    print(f"{name}: Y rel err per entry\n{np.array2string(rel, precision=3)}")
    for w, (a, b) in enumerate(zip(c1, cf)):
        print(
            f"    wire {w}: |I1 - If|/|If| {np.linalg.norm(a - b) / np.linalg.norm(b):.3e}"
            f"  |I1 + If|/|If| {np.linalg.norm(a + b) / np.linalg.norm(b):.3e}"
        )
