"""momwire#973: what the ACA stopping rule measures vs the block's TRUE error.

Wraps `aca_partial`, and for every admissible block densifies the block from
the same `get_row` callbacks the algorithm used, so the true relative
Frobenius error ||A - UV||/||A|| can be set beside the quantity the stopping
rule actually tests (||u_r|| ||v_r|| / ||A_approx||).

Geometry comes from the antennaknobs design, so this runs from the AK venv
with PYTHONPATH pointed at this tree.
"""

import sys

import numpy as np

sys.path.insert(0, "/home/smburns/stevenmburns/antennaknobs/scripts")

import bench_catalog as bc  # noqa: E402
import bench_converge as bnc  # noqa: E402

import momwire._aca as A  # noqa: E402
import momwire.hmatrix as HM  # noqa: E402
from antennaknobs.engines.momwire import MomwireEngine  # noqa: E402
from momwire import HMatrixSolver  # noqa: E402

RECORDS = []
_orig = A.aca_partial


def spy(get_row, get_col, m, n, tol=1e-3, max_rank=None):
    U, V = _orig(get_row, get_col, m, n, tol=tol, max_rank=max_rank)
    dense = np.empty((m, n), dtype=np.complex128)
    for i in range(m):
        dense[i, :] = np.asarray(get_row(i), dtype=np.complex128)
    approx = U @ V if U.shape[1] else np.zeros((m, n), dtype=np.complex128)
    nA = float(np.linalg.norm(dense))
    err = float(np.linalg.norm(dense - approx))
    RECORDS.append(
        {"m": m, "n": n, "rank": int(U.shape[1]), "normA": nA,
         "abs_err": err, "rel_err": err / nA if nA > 0 else 0.0,
         "normUV": float(np.linalg.norm(approx))}
    )
    return U, V


def run(tol):
    RECORDS.clear()
    # hmatrix.py does `from ._aca import aca_partial`, so the name is bound in
    # ITS namespace -- patching momwire._aca alone records nothing and silently
    # reports zero blocks while the plateau Z is unchanged.
    HM.aca_partial = spy
    try:
        cls = bnc.load_design("loops.skyloop_lmatch")
        base = bc.default_nseg("loops.skyloop_lmatch")
        b = cls()
        b.nominal_nsegs = base * 2
        eng = MomwireEngine(
            b, solver=HMatrixSolver, solver_kwargs={"degree": 2, "aca_tol": tol},
            ground=("finite", 13.0, 0.005),
        )
        z = eng.impedance()[0]
    finally:
        HM.aca_partial = _orig
    return z, list(RECORDS)


def main():
    for tol in (1e-6, 1e-8):
        z, recs = run(tol)
        rel = np.array([r["rel_err"] for r in recs])
        if rel.size == 0:
            print(f"\n=== aca_tol {tol:.0e}: NO admissible blocks (dense route?)")
            continue
        print(f"\n=== aca_tol {tol:.0e}   Z {z.real:.6f}{z.imag:+.6f}j   "
              f"{len(recs)} admissible blocks")
        print(f"  block rel err: max {rel.max():.3e}  median {np.median(rel):.3e}  "
              f"n>tol {int((rel > tol).sum())}")
        worst = sorted(recs, key=lambda r: -r["rel_err"])[:6]
        print(f"  {'m':>4s} {'n':>4s} {'rank':>4s} {'rel_err':>10s} {'x tol':>8s} {'||A||':>10s}")
        for r in worst:
            print(f"  {r['m']:4d} {r['n']:4d} {r['rank']:4d} {r['rel_err']:10.3e} "
                  f"{r['rel_err'] / tol:8.1f} {r['normA']:10.3e}")


main()
