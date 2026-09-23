"""momwire#1159 probe 5: the WHOLLY-buried route's port solution and
readouts. Two parallel buried wires (depth 0.5 m and 1.0 m, soil A), one
centre port each. compute_y_matrix / compute_port_solution must run, their
Y must invert to compute_impedance's Z (both ports driven), and SG's Y12,
knot currents and far field must converge on bspline's under refinement.

Prints which momwire was imported (side-by-side against main's source).
"""

import sys
import warnings

import numpy as np

import momwire
from momwire.bspline import BSplineSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

sys.path.insert(0, "scratch/1159-sg-mixed-sign")
from probe4_far_field import gaps, pattern  # noqa: E402

WL7 = 299792458.0 / 7e6


def buried(m=1, eps=(13.0, 0.005)):
    return dict(
        wires=[
            np.array([(-2.5, 0.0, -0.5), (2.5, 0.0, -0.5)]),
            np.array([(-2.5, 1.0, -1.0), (2.5, 1.0, -1.0)]),
        ],
        n_per_edge_per_wire=[[10 * m + 1], [10 * m + 1]],
        feeds=[(0, 2.5, 1 + 0j), (1, 2.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        ground_z=0.0,
        ground_eps=eps,
        ground_model="sommerfeld",
    )


def solve(cls, d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = cls(**d)
        sol = s.compute_port_solution()
        cur = s.currents_at_knots(sol.coeffs.sum(axis=1))
        z = np.asarray(s.compute_impedance()[0])
    return np.asarray(sol.y), [np.asarray(c) for c in cur], z


if __name__ == "__main__":
    print("momwire from", momwire.__file__)
    for m in [int(x) for x in sys.argv[1:]] or [1, 2, 4]:
        d = buried(m)
        try:
            ysg, csg, zsg = solve(SinusoidalGalerkinSolver, d)
        except ValueError as e:
            print(f"m={m}: SG port solution refused: {str(e)[:100]}")
            continue
        ybs, cbs, _ = solve(BSplineSolver, d)
        z_from_y = 1.0 / ysg.sum(axis=1)  # both ports at 1 V
        print(
            f"m={m}: Z(Y)-Z(imp) rel {np.abs(z_from_y - zsg).max() / np.abs(zsg).max():.3e}; "
            f"Y12 SG {ysg[0, 1]:.5e} bs {ybs[0, 1]:.5e} "
            f"rel {abs(ysg[0, 1] - ybs[0, 1]) / abs(ybs[0, 1]):.3e}; "
            f"Y11 rel {abs(ysg[0, 0] - ybs[0, 0]) / abs(ybs[0, 0]):.3e}"
        )
        for w, (a, b) in enumerate(zip(csg, cbs)):
            print(
                f"    wire {w}: |I_sg - I_bs|/|I_bs| {np.linalg.norm(a - b) / np.linalg.norm(b):.3e}"
            )
        fg, pg = gaps(pattern(SinusoidalGalerkinSolver, d), pattern(BSplineSolver, d))
        print(f"    far field: |M| gap {fg:.3e}  power gap {pg:.3e}", flush=True)
