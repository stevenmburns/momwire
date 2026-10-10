"""Traced peak inside the correction (tracemalloc, C++ buffers reported),
numpy vs C++ route, uniform 1500 and graded 1200 wires at swept_mem_mb=16."""

import time
import tracemalloc
import warnings

import numpy as np

from momwire import BSplineSolver
from momwire import _exact_kernel as EK

warnings.simplefilter("ignore")


def solver(kind):
    if kind == "uniform1500":
        w, n, a = [np.array([(0.0, 0.0, -2.25), (0.0, 0.0, 2.25)])], [[1500]], 0.00425
    else:
        m = 1200
        h = np.r_[np.geomspace(0.02, 0.002, m // 2), np.geomspace(0.002, 0.02, m // 2)]
        z = np.r_[0.0, np.cumsum(h)]
        w = [np.stack([np.zeros_like(z), np.zeros_like(z), z], -1)]
        n, a = [[1] * m], 0.002
    return BSplineSolver(
        wires=w,
        n_per_edge_per_wire=n,
        wavelength=1.0,
        wire_radius=a,
        exact_kernel=True,
        swept_mem_mb=16,
    )


orig = BSplineSolver._add_exact_kernel_correction
for kind in ("uniform1500", "graded1200"):
    for accel in (False, True):
        EK._USE_ACCEL = accel
        s = solver(kind)
        geom = s._build_geometry()
        supp_seg, polys, *_ = s._build_basis_polynomials(geom)
        rec = {}

        def traced(self, *args, rec=rec):
            tracemalloc.start()
            tracemalloc.reset_peak()
            e = tracemalloc.get_traced_memory()[0]
            t = time.perf_counter()
            try:
                return orig(self, *args)
            finally:
                rec["t"] = time.perf_counter() - t
                rec["mb"] = (tracemalloc.get_traced_memory()[1] - e) / 2**20
                tracemalloc.stop()

        BSplineSolver._add_exact_kernel_correction = traced
        s._compute_Z_operator(geom, supp_seg, polys)
        BSplineSolver._add_exact_kernel_correction = orig
        print(
            f"{kind} {'C++  ' if accel else 'numpy'}: traced peak {rec['mb']:.1f} MB, correction {rec['t']:.2f} s (under tracemalloc)"
        )
