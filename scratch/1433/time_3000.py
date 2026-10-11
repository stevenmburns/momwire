"""Wall time of a 3000-segment fat razor-2p solve, exact kernel off vs on,
paired and alternated (momwire#1433). 10 m dipole at 14 MHz, a = 8 mm
(h/a 0.42). Prints per-run seconds, the correction's own share, and peak RSS.

Usage: python -I time_3000.py [reps]
"""

import json
import resource
import sys
import time
import warnings

import numpy as np

from momwire.razor import RazorSolver

warnings.simplefilter("ignore")
L, N, A, LAM = 10.0, 3000, 0.008, 21.4
T = {}
_orig = RazorSolver._add_exact_kernel_correction


def _timed(self, *a):
    t = time.perf_counter()
    _orig(self, *a)
    T["corr"] = time.perf_counter() - t


RazorSolver._add_exact_kernel_correction = _timed
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for _ in range(reps):
    for exact in (False, True):
        T.clear()
        t0 = time.perf_counter()
        s = RazorSolver(
            wires=[np.array([(0, 0, -L / 2), (0, 0, L / 2)])],
            n_per_edge_per_wire=[[N]],
            wavelength=LAM,
            wire_radius=A,
            nec5_quadrature=True,
            extended_kernel=True,
            exact_kernel=exact,
        )
        z = complex(np.ravel(s.compute_impedance()[0])[0])
        print(
            json.dumps(
                {
                    "exact": exact,
                    "sec": round(time.perf_counter() - t0, 3),
                    "corr_sec": round(T.get("corr", 0.0), 3),
                    "Z": [z.real, z.imag],
                    "maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    // 1024,
                }
            ),
            flush=True,
        )
