"""momwire#1156 probe 7: adjudicate probe6's Y12 sign split on the mixed
deck. At eps~ = 1 the mixed deck IS free space, so each trunk's mixed Y12
must equal its own free-space Y12 (same wires, no ground)."""

import sys
import warnings

import numpy as np

sys.path.insert(0, "scratch/1156-sg-buried-loading")

from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from probe5_mixed_refined import mixed  # noqa: E402

for cls in (SinusoidalGalerkinSolver, BSplineSolver, RazorSolver):
    extra = dict(nec5_quadrature=True) if cls is RazorSolver else {}
    one = mixed(2, eps=(1.0, 0.0))
    free = dict(one)
    for key in ("ground_z", "ground_eps", "ground_model"):
        free.pop(key)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_one = np.asarray(cls(**one, **extra).compute_y_matrix())
        y_free = np.asarray(cls(**free, **extra).compute_y_matrix())
    print(
        f"{cls.__name__:26s} eps~=1 Y12 {y_one[0, 1]:.5e}  free Y12 {y_free[0, 1]:.5e}"
        f"  Y11 {y_one[0, 0]:.4e} vs {y_free[0, 0]:.4e}"
    )
