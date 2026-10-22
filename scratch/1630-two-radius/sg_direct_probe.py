"""SG fill="direct" vs the default mixed-potential fill at a two-radius node:
the U5 rod guard and the 1-radial hub radius responses (resp_probe's CFG).
Run from repo root."""

from __future__ import annotations

import sys
import warnings

sys.path.insert(0, "tests")
sys.path.insert(0, "scratch/1630-two-radius")
import decks
from test_crossing_serve_524 import crossing_deck

from momwire.deck import build_solver, parse
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

warnings.simplefilter("ignore")
A = 0.25e-3
CFG = {
    "c0_all1": (0.001, 0.001),
    "c1_v2": (0.002, 0.001),
    "c3_v19": (0.019, 0.001),
    "c5_b025": (0.001, 0.00025),
}


def zz(z):
    return complex(z[0] if isinstance(z, tuple) else z)


for fill in ("mixed-potential", "direct"):
    rod = {
        k: zz(
            SinusoidalGalerkinSolver(
                **crossing_deck(2, wire_radius=[b, a]), fill=fill
            ).compute_impedance()
        )
        for k, (a, b) in {"eq": (A, A), "rise2": (A, A / 2), "top2": (A / 2, A)}.items()
    }
    print(
        fill,
        "rod eq %s rise2 %+.3f top2 %+.3f"
        % (rod["eq"], (rod["rise2"] - rod["eq"]).real, (rod["top2"] - rod["eq"]).real),
        flush=True,
    )
    hub = {}
    for cfg, (a_v, a_b) in CFG.items():
        built = build_solver(
            parse(
                decks.nec42("hub", 4, "mid", a_v=a_v, a_b=a_b, n_rad=1), dialect="nec4"
            ),
            basis="sinusoidal-galerkin",
        )
        s = built.solver
        s.fill = fill
        hub[cfg] = zz(s.compute_impedance())
    c0 = hub["c0_all1"]
    print(
        fill,
        "hub r=4",
        " ".join(f"{k[:6]} {z - c0:+.2f}" for k, z in hub.items()),
        flush=True,
    )
