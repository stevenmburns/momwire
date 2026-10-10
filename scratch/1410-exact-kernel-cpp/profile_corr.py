"""Profile the exact-kernel correction on a straight fat wire (#1410).

h = 3 mm, a = 2 mm, d = 2 (the #1415 deck). Usage: profile_corr.py N [cprofile]
"""

import sys
import time
import warnings

import numpy as np

from momwire import BSplineSolver

n = int(sys.argv[1])
h, a = 0.003, 0.002
wire = [np.array([(0.0, 0.0, 0.0), (0.0, 0.0, n * h)])]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    s = BSplineSolver(
        wires=wire,
        n_per_edge_per_wire=[[n]],
        wavelength=1.0,
        wire_radius=a,
        exact_kernel=True,
        degree=2,
    )
geom = s._build_geometry()
supp_seg, polys, *_ = s._build_basis_polynomials(geom)
s.exact_kernel = False
t0 = time.perf_counter()
Z = np.array(s._compute_Z_operator(geom, supp_seg, polys))
t1 = time.perf_counter()
s.exact_kernel = True
if len(sys.argv) > 2:
    import cProfile
    import pstats

    pr = cProfile.Profile()
    pr.enable()
    s._add_exact_kernel_correction(Z, geom, supp_seg, polys, s.k)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(25)
else:
    s._add_exact_kernel_correction(Z, geom, supp_seg, polys, s.k)
t2 = time.perf_counter()
print(f"n={n} fill {t1 - t0:.3f} s  correction {t2 - t1:.3f} s")
