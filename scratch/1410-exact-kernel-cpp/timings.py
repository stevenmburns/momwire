"""Same-machine timings for #1410. usage: timings.py CASE [reps]
CASE: dipole301 | nbs301 | fat1000 | fat3000 | fat3000-fill
Prints wall times: total solve (or fill + correction) and the correction alone
(spied through the production seam). Best of `reps`.
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np

import momwire
from momwire import BSplineSolver

warnings.simplefilter("ignore")
case = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
corr = []
orig = BSplineSolver._add_exact_kernel_correction


def spy(self, *a):
    t = time.perf_counter()
    try:
        return orig(self, *a)
    finally:
        corr.append(time.perf_counter() - t)


BSplineSolver._add_exact_kernel_correction = spy


def nbs():
    wires, radius, ex = [], None, None
    p = Path(__file__).resolve().parent / "nbs688_0.8lambda.nec"
    for line in p.read_text().splitlines():
        f = line.split()
        if f and f[0] == "GW":
            x1, y1, z1, x2, y2, z2, r = map(float, f[3:10])
            wires.append(np.array([(x1, y1, z1), (x2, y2, z2)]))
            radius = r
        elif f and f[0] == "EX":
            ex = int(f[2]) - 1
    return wires, radius, ex


def make(exact=True):
    if case == "dipole301":
        return BSplineSolver(
            wires=[np.array([(0.0, 0.0, -0.235), (0.0, 0.0, 0.235)])],
            n_per_edge_per_wire=[[301]],
            wavelength=1.0,
            wire_radius=0.00425,
            feed_model="segment",
            exact_kernel=exact,
            extended_kernel=not exact,
        )
    if case == "nbs301":
        wires, radius, ex = nbs()
        return BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=[[301]] * len(wires),
            feeds=[
                (ex, float(np.linalg.norm(wires[ex][1] - wires[ex][0])) / 2, 1 + 0j)
            ],
            feed_model="segment",
            wavelength=299.792458 / 400.0,
            wire_radius=radius,
            exact_kernel=exact,
            extended_kernel=True,
        )
    n = int(case[3:7])
    return BSplineSolver(
        wires=[np.array([(0.0, 0.0, 0.0), (0.0, 0.0, n * 0.003)])],
        n_per_edge_per_wire=[[n]],
        wavelength=1.0,
        wire_radius=0.002,
        exact_kernel=exact,
        degree=2,
    )


best_tot, best_corr = np.inf, np.inf
for _ in range(reps):
    corr.clear()
    if case.startswith("fat"):
        s = make(not case.endswith("-fill"))
        geom = s._build_geometry()
        supp_seg, polys, *_ = s._build_basis_polynomials(geom)
        t = time.perf_counter()
        s._compute_Z_operator(geom, supp_seg, polys)
        tot = time.perf_counter() - t
    else:
        s = make()
        t = time.perf_counter()
        s.compute_impedance()
        tot = time.perf_counter() - t
    best_tot = min(best_tot, tot)
    best_corr = min(best_corr, sum(corr) if corr else np.nan)
print(
    f"{case}: total {best_tot:.3f} s  correction {best_corr:.3f} s  ({momwire.__file__})"
)
