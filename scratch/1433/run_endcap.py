"""#1378's gap-free ladder (antennaknobs scratch/1327-endcap/run_mw.py's deck
and metric) with the exact-kernel lanes (momwire#1433).

5 m dipole on z, free space, 28 MHz, a = 23.8 mm, fed at the centre knot.
S = ∫ I(z) dz / V is the broadside short-circuit current per 1 V/m by
reciprocity: gap-model independent to first order, so the knot gap's log
drift does not enter it.

Usage: python -I run_endcap.py VARIANT N0,N1,... >> out.jsonl
"""

import json
import sys
import time
import warnings

import numpy as np

from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

warnings.simplefilter("ignore")
C = 299792458.0
L = 5.0
A = 0.0238
VARS = {
    "razor-2p": (RazorSolver, {"nec5_quadrature": True, "extended_kernel": True}),
    "razor-2p exact": (
        RazorSolver,
        {"nec5_quadrature": True, "extended_kernel": True, "exact_kernel": True},
    ),
    "bs2 exact": (BSplineSolver, {"exact_kernel": True}),
}
x8, w8 = np.polynomial.legendre.leggauss(8)
v = sys.argv[1]
cls, kw = VARS[v]
for N0 in [int(x) for x in sys.argv[2].split(",")]:
    N = N0 + 1
    t0 = time.time()
    s = cls(
        wires=[np.array([[0, 0, -L / 2], [0, 0, L / 2]])],
        n_per_edge_per_wire=[[N]],
        wavelength=C / 28e6,
        wire_radius=A,
        feeds=[(0, L / 2, 1 + 0j)],
        **kw,
    )
    z, coeffs = s.compute_impedance()
    z = complex(np.atleast_1d(z)[0])
    edges = np.linspace(0, L, N + 1)
    sq = (
        0.5 * (edges[:-1, None] + edges[1:, None])
        + 0.5 * np.diff(edges)[:, None] * x8[None, :]
    ).ravel()
    wq = (0.5 * np.diff(edges)[:, None] * w8[None, :]).ravel()
    cur = np.asarray(s.currents_at_knots(coeffs, s_array=[sq])[0])
    S = complex(np.sum(cur * wq))
    row = dict(
        N=N,
        N0=N0,
        engine=v,
        Z=[z.real, z.imag],
        S=[S.real, S.imag],
        exact=bool(getattr(s, "exact_kernel", False)),
        wall=round(time.time() - t0, 2),
    )
    print(json.dumps(row), flush=True)
