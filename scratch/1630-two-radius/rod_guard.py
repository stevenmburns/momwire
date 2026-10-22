"""The U5 physics guard (tests/test_crossing_two_radius_u5.py::test_u5_3) on
the sinusoidal pair: crossing_deck(2) rod, radii [below, above]. Prints R for
the radius configurations, and SG's node inflow residual. Run from repo root:
python scratch/1630-two-radius/rod_guard.py"""

from __future__ import annotations

import sys
import time
import warnings

sys.path.insert(0, "tests")

from test_crossing_serve_524 import crossing_deck

from momwire.bspline import BSplineSolver
from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

A = 0.25e-3
warnings.simplefilter("ignore")
for cls in (SinusoidalSolver, SinusoidalGalerkinSolver, BSplineSolver):
    out = {}
    for name, (aa, ab) in {
        "equal": (A, A),
        "rise2": (A, A / 2),
        "rise4": (A, A / 4),
        "top2": (A / 2, A),
        "near": (A, A * (1 - 1e-7)),
    }.items():
        t0 = time.perf_counter()
        z = cls(**crossing_deck(2, wire_radius=[ab, aa])).compute_impedance()
        z = complex(z[0] if isinstance(z, tuple) else z)
        out[name] = z
        print(
            cls.__name__,
            name,
            f"{z:.6f}",
            f"{time.perf_counter() - t0:.1f}s",
            flush=True,
        )
    r0 = out["equal"].real
    print(
        cls.__name__,
        "rise2 %.3f rise4 %.3f top2 %.3f ratio %.3f near-equal |dZ| %.2e"
        % (
            out["rise2"].real - r0,
            out["rise4"].real - r0,
            out["top2"].real - r0,
            (out["rise4"].real - r0) / (out["rise2"].real - r0),
            abs(out["near"] - out["equal"]),
        ),
        flush=True,
    )
