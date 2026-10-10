import cProfile
import pstats
import warnings

import numpy as np

from momwire import BSplineSolver

warnings.simplefilter("ignore")
m = 1200
h = np.r_[np.geomspace(0.02, 0.002, m // 2), np.geomspace(0.002, 0.02, m // 2)]
z = np.r_[0.0, np.cumsum(h)]
w = [np.stack([np.zeros_like(z), np.zeros_like(z), z], -1)]
s = BSplineSolver(
    wires=w,
    n_per_edge_per_wire=[[1] * m],
    wavelength=1.0,
    wire_radius=0.002,
    exact_kernel=True,
)
geom = s._build_geometry()
supp_seg, polys, *_ = s._build_basis_polynomials(geom)
s.exact_kernel = False
Z = np.array(s._compute_Z_operator(geom, supp_seg, polys))
s.exact_kernel = True
pr = cProfile.Profile()
pr.enable()
s._add_exact_kernel_correction(Z, geom, supp_seg, polys, s.k)
pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(12)
