"""#1408's fat dipole (L = 0.47 lambda, d/lambda 0.0085) on a knot / point
gap, razor-2p and bs2, with and without the exact kernel (momwire#1433).

Records Z and Y = 1/Z per rung: G = Re Y is gap-independent and must
converge; under the exact kernel B = Im Y gains about 4 ln2 w eps0 a
(0.1965 mS here) per halving -- the zero-width gap's log drift, which no
kernel removes.

Usage: python -I run_fat.py VARIANT n0,n1,... >> out.jsonl
"""

import json
import sys
import time
import warnings

import numpy as np

from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

warnings.simplefilter("ignore")
L, A = 0.47, 0.00425
VARS = {
    "razor-2p": (RazorSolver, {"nec5_quadrature": True, "extended_kernel": True}),
    "razor-2p exact": (
        RazorSolver,
        {"nec5_quadrature": True, "extended_kernel": True, "exact_kernel": True},
    ),
    "bs2": (BSplineSolver, {"extended_kernel": True}),
    "bs2 exact": (BSplineSolver, {"exact_kernel": True}),
}
v = sys.argv[1]
cls, kw = VARS[v]
for n in [int(x) for x in sys.argv[2].split(",")]:
    t0 = time.time()
    s = cls(
        wires=[np.array([(0.0, 0.0, -L / 2), (0.0, 0.0, L / 2)])],
        n_per_edge_per_wire=[[n]],
        wavelength=1.0,
        wire_radius=A,
        **kw,
    )
    z = complex(np.atleast_1d(s.compute_impedance()[0])[0])
    print(
        json.dumps(
            dict(
                n=n,
                d_over_a=L / n / A,
                engine=v,
                Z=[z.real, z.imag],
                Y=[(1 / z).real, (1 / z).imag],
                exact=bool(getattr(s, "exact_kernel", False)),
                wall=round(time.time() - t0, 2),
            )
        ),
        flush=True,
    )
