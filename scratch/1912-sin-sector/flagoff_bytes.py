"""momwire#1912 flag-off byte identity: the DENSE SinusoidalSolver, built
without the new kwarg so the same script runs on a main clone, hashed
array by array. Run it under main and under the branch on one machine and
diff the outputs.

usage: flagoff_bytes.py PATH/TO/tests/test_rotational_sinusoidal_1912.py

(the deck builders are read from the branch's test module by path; nothing
there needs the branch's momwire)."""

import hashlib
import importlib.util
import sys
import warnings

import numpy as np

import momwire
from momwire import SinusoidalSolver

spec = importlib.util.spec_from_file_location("decks", sys.argv[1])
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def h(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()[:16]


def build(wires, npe, feeds, ground="free", **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SinusoidalSolver(
            wires=wires,
            n_per_edge_per_wire=npe,
            wavelength=t.WL,
            wire_radius=0.002,
            feeds=feeds,
            **t.GROUNDS[ground],
            **kw,
        )


cases = {
    "cage4-free": (*t.cage(), [t.FEED, *t.posts()], "free", {}),
    "cage4-somm": (*t.cage(lift=t.LIFT), [t.FEED, *t.posts()], "sommerfeld", {}),
    "cage6-rev-free": (
        *t.reverse(*t.cage(6), [3]),
        [t.FEED],
        "free",
        {},
    ),
    "screen-pec": (*t.screen(droop=0.05), "pec", {}),
    "screen-refl": (*t.screen(droop=0.05), "refl-coef", {}),
    "screen6-loaded-somm": (
        *t.screen(6),
        "sommerfeld",
        dict(wire_conductivity=3.5e7, insulation_radius=0.003, insulation_eps_r=3.0),
    ),
    "tophat-contact-somm": (*t.top_hat(8), "sommerfeld", {}),
    "buried-somm": (*t.buried(), "sommerfeld", {}),
    "parasitic-free": (*t.parasitic(), "free", {}),
}
print("momwire", momwire.__file__)
for name, (w, npe, feeds, ground, kw) in cases.items():
    s = build(w, npe, feeds, ground, **kw)
    geom = s._build_geometry()
    with s._operating_medium(geom) as medium:
        G, _v = s._assemble_Z(geom, s.k, s._medium_eta(medium))
    z, alpha = s.compute_impedance()
    y = s.compute_y_matrix()
    zs = s.compute_impedance_swept(s.k * np.array([0.97, 1.03]))
    print(
        name,
        "G",
        h(G),
        "z",
        h(np.atleast_1d(z)),
        "alpha",
        h(alpha),
        "y",
        h(y),
        "sweep",
        h(zs),
    )
