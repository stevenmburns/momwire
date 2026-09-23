"""momwire#1156 probe 6: probe5's port 1 (the ABOVE wire, both ports driven)
disagrees SG vs bspline/razor on the BARE deck. Is it the coupling? The
2x2 short-circuit Y of the bare mixed deck per trunk, and the bare
above-only / below-only single wires for reference."""

import sys
import warnings

import numpy as np

sys.path.insert(0, "scratch/1156-sg-buried-loading")

from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from probe5_mixed_refined import mixed  # noqa: E402

np.set_printoptions(precision=5, linewidth=150)
for m in (1, 4):
    d = mixed(m)
    for cls in (SinusoidalGalerkinSolver, BSplineSolver, RazorSolver):
        extra = dict(nec5_quadrature=True) if cls is RazorSolver else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Y = np.asarray(cls(**d, **extra).compute_y_matrix())
            za = complex(
                np.ravel(
                    cls(
                        **dict(
                            d,
                            wires=[d["wires"][1]],
                            n_per_edge_per_wire=[d["n_per_edge_per_wire"][1]],
                            feeds=[(0, 2.5, 1 + 0j)],
                        ),
                        **extra,
                    ).compute_impedance()[0]
                )[0]
            )
        print(
            f"m={m} {cls.__name__:26s} Y11 {Y[0, 0]:.5e} Y22 {Y[1, 1]:.5e} "
            f"Y12 {Y[0, 1]:.4e} Y21 {Y[1, 0]:.4e} | above alone Z {za:.3f}"
        )
