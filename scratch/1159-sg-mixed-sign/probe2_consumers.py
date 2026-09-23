"""momwire#1159 probe 2: the consumers of the current. SG vs bspline on a
mixed deck in soil, BOTH ports driven (1 V each): Y12, and every wire's
current at the mesh knots through the public `currents_at_knots`.

Usage: probe2_consumers.py [deck] [m ...]   deck in {mixed, detached, hub}
"""

import sys
import warnings

import numpy as np

sys.path.insert(0, "scratch/1156-sg-buried-loading")
sys.path.insert(0, "tests")

from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from probe5_mixed_refined import mixed  # noqa: E402
from test_razor_detached_1149 import detached, detached_hub  # noqa: E402

DECKS = {"mixed": mixed, "detached": detached, "hub": detached_hub}


def solve(cls, d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = cls(**d)
        sol = s.compute_port_solution()
        both = sol.coeffs.sum(axis=1)  # 1 V at every port
        cur = s.currents_at_knots(both)
    return np.asarray(sol.y), cur


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "mixed"
    ms = [int(x) for x in sys.argv[2:]] or [1, 2, 4]
    for m in ms:
        d = DECKS[name](m)
        ysg, csg = solve(SinusoidalGalerkinSolver, d)
        ybs, cbs = solve(BSplineSolver, d)
        print(
            f"{name} m={m}: Y12 SG {ysg[0, 1]:.5e} bs {ybs[0, 1]:.5e} "
            f"rel {abs(ysg[0, 1] - ybs[0, 1]) / abs(ybs[0, 1]):.3e}; "
            f"Y11 rel {abs(ysg[0, 0] - ybs[0, 0]) / abs(ybs[0, 0]):.3e}",
            flush=True,
        )
        for w, (a, b) in enumerate(zip(csg, cbs)):
            a, b = np.asarray(a), np.asarray(b)
            below = bool(np.all(d["wires"][w][:, 2] < 0))
            rel = np.linalg.norm(a - b) / np.linalg.norm(b)
            flip = np.linalg.norm(a + b) / np.linalg.norm(b)
            print(
                f"    wire {w} ({'below' if below else 'above'}): |I_sg - I_bs|/|I_bs| "
                f"{rel:.3e}  |I_sg + I_bs|/|I_bs| {flip:.3e}  max|I_bs| {np.abs(b).max():.3e}"
            )
