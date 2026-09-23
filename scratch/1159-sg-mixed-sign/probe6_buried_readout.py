"""momwire#1159 probe 6: the wholly-buried READOUT on its own, reached through
`compute_impedance` (which main already served): its alpha read through
`currents_at_knots`, against bspline's on the same deck. Run against main's
source and the branch to size what the air-k readout cost."""

import sys
import warnings

import numpy as np

import momwire
from momwire.bspline import BSplineSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

sys.path.insert(0, "scratch/1159-sg-mixed-sign")
from probe5_wholly_buried import buried  # noqa: E402

print("momwire from", momwire.__file__)
for m in (1, 2):
    d = buried(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cur = []
        for cls in (SinusoidalGalerkinSolver, BSplineSolver):
            s = cls(**d)
            _z, alpha = s.compute_impedance()
            cur.append([np.asarray(c) for c in s.currents_at_knots(alpha)])
    for w, (a, b) in enumerate(zip(*cur)):
        print(
            f"m={m} wire {w}: |I_sg - I_bs|/|I_bs| {np.linalg.norm(a - b) / np.linalg.norm(b):.3e}"
        )
