"""momwire#1156 probe 14: SG bit-identity against origin/main.

Run once with this branch's momwire and once with main's source first on
PYTHONPATH (the same compiled accelerators — no C++ differs); the first
line printed is the momwire package imported, so the method records which
one ran. Every deck below that main can solve is dumped as exact float
hex: above-ground and free-space decks, loaded and bare, single and swept,
the metal-loss readout, and the BARE buried decks (loading off takes no new
line). Usage: probe14_ab_main.py OUT.json"""

import json
import sys
import warnings

import numpy as np

import momwire
from momwire._wire_loading import DistributedRLC
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

print(momwire.__file__, flush=True)
sys.path.insert(0, momwire.__file__.rsplit("/src/", 1)[0] + "/tests")
from test_razor_crossing_loading_1149 import crossing  # noqa: E402

WL7 = 299792458.0 / 7.0e6
SOIL_A = (13.0, 0.005)
ABOVE = np.array([(-2.5, 0.0, 3.0), (2.5, 0.0, 3.0)])
BURIED = np.array([(-2.5, 0.0, -0.5), (2.5, 0.0, -0.5)])
SOM = dict(ground_z=0.0, ground_eps=SOIL_A, ground_model="sommerfeld")
LOADS = {
    "bare": {},
    "sigma": dict(wire_conductivity=3.5e7),
    "jacket": dict(insulation_radius=1.8e-3, insulation_eps_r=3.5),
    "rlc": dict(distributed_rlc=DistributedRLC("series", r=5.0, l=2e-7)),
    "all": dict(
        wire_conductivity=3.5e7,
        insulation_radius=1.8e-3,
        insulation_eps_r=3.5,
    ),
}


def base(wires, **kw):
    d = dict(
        wires=wires,
        n_per_edge_per_wire=[[21]] * len(wires),
        feeds=[(0, 2.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
    )
    d.update(kw)
    return d


GEOMS = {
    "free": base([ABOVE]),
    "pec": base([ABOVE], ground_z=0.0),
    "refl-coef": base([ABOVE], ground_z=0.0, ground_eps=SOIL_A),
    "sommerfeld-above": base([ABOVE], **SOM),
    "surface-at-b": base([ABOVE - [0, 0, 3.0 - 1.8e-3]], **SOM),
    "two-wire-free": base([ABOVE, ABOVE + [0, 1.0, 0]]),
}
BARE_BURIED = {
    "buried-dipole": base([BURIED], **SOM),
    "mixed": base([BURIED, ABOVE - [0, 0, 2.0]], **SOM),
    "crossing": dict(crossing(1), wire_radius=1e-3),
}


def hexs(a):
    return [[float(np.real(x)).hex(), float(np.imag(x)).hex()] for x in np.ravel(a)]


out = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for g, d in GEOMS.items():
        for lname, kw in LOADS.items():
            if lname != "bare" and g == "two-wire-free":
                kw = {
                    k: (v if k == "distributed_rlc" else [v, v]) for k, v in kw.items()
                }
            s = SinusoidalGalerkinSolver(**d, **kw)
            z, alpha = s.compute_impedance()
            key = f"{g}/{lname}"
            out[key + "/z"] = hexs(z)
            out[key + "/alpha"] = hexs(alpha)
            out[key + "/loss"] = hexs(s.wire_loss_power(alpha)[1])
            ks = [s.k, s.k * 1.05]
            out[key + "/swept"] = hexs(s.compute_impedance_swept(ks))
            if g in ("free", "pec"):
                out[key + "/y"] = hexs(s.compute_y_matrix())
    for g, d in BARE_BURIED.items():
        s = SinusoidalGalerkinSolver(**d)
        z, alpha = s.compute_impedance()
        out[g + "/bare/z"] = hexs(z)
        out[g + "/bare/alpha"] = hexs(alpha)
        out[g + "/bare/swept"] = hexs(s.compute_impedance_swept([s.k, s.k * 1.05]))
json.dump(out, open(sys.argv[1], "w"), indent=0)
print(len(out), "records", flush=True)
