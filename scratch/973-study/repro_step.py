"""momwire#973: reproduce the skyloop_lmatch aca_tol STEP, momwire-side only.

The deck is an antennaknobs design, so its solver kwargs were captured once
from `MomwireEngine` and pickled; this script replays them against
`HMatrixSolver` and dense `BSplineSolver` so the whole investigation runs in
the momwire tree with no antennaknobs import.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np

from momwire import BSplineSolver, HMatrixSolver

KW = pickle.loads(Path(sys.argv[1]).read_bytes())
TOLS = (1e-4, 1e-6, 1e-7, 3e-8, 1e-8, 1e-10)


def z_of(solver_cls, **extra):
    kw = dict(KW)
    kw.pop("_args", None)
    kw.update(extra)
    t = time.perf_counter()
    s = solver_cls(**kw)
    z = np.asarray(s.compute_impedance()).ravel()[0]
    return z, time.perf_counter() - t, s


def main():
    zd, td, _ = z_of(BSplineSolver)
    print(f"dense bs2   Z = {zd.real:.6f}{zd.imag:+.6f}j   {td:.3f}s")
    print(f"\n{'aca_tol':>9s}  {'Z':>28s}  {'rel dZ':>10s}  {'s':>6s}")
    for tol in TOLS:
        z, t, _ = z_of(HMatrixSolver, aca_tol=tol)
        rel = abs(z - zd) / abs(zd)
        print(f"{tol:9.0e}  {z.real:14.6f}{z.imag:+13.6f}j  {rel:10.3e}  {t:6.3f}")


if __name__ == "__main__":
    main()
