"""momwire#973: rank-by-rank trace of the global Sommerfeld ACA on skyloop.

Re-implements `aca_partial`'s loop verbatim with instrumentation, so at every
rank we can put the quantity the stopping rule TESTS beside the true residual
it is meant to bound.
"""

import sys

import numpy as np

sys.path.insert(0, "/home/smburns/stevenmburns/antennaknobs/scripts")

import bench_catalog as bc  # noqa: E402
import bench_converge as bnc  # noqa: E402

import momwire.hmatrix as HM  # noqa: E402
from antennaknobs.engines.momwire import MomwireEngine  # noqa: E402
from momwire import HMatrixSolver  # noqa: E402

TRACE = []
_orig = HM.aca_partial


def traced(get_row, get_col, m, n, tol=1e-3, max_rank=None):
    if m != n or m < 100:                      # only the global Sommerfeld block
        return _orig(get_row, get_col, m, n, tol=tol, max_rank=max_rank)
    dense = np.empty((m, n), dtype=np.complex128)
    for i in range(m):
        dense[i, :] = np.asarray(get_row(i), dtype=np.complex128)
    nA = float(np.linalg.norm(dense))

    U, V = [], []
    used_rows = np.zeros(m, dtype=bool)
    used_cols = np.zeros(n, dtype=bool)
    approx_norm2 = 0.0
    i_star, rank = 0, 0
    for _ in range(min(m, n)):
        row = np.asarray(get_row(i_star), dtype=np.complex128).copy()
        for kk in range(rank):
            row -= U[kk][i_star] * V[kk]
        used_rows[i_star] = True
        absrow = np.abs(row)
        absrow[used_cols] = -1.0
        j_star = int(np.argmax(absrow))
        delta = row[j_star]
        if np.abs(delta) < 1e-300 or absrow[j_star] <= 0.0:
            rem = np.flatnonzero(~used_rows)
            if rem.size == 0:
                break
            i_star = int(rem[0])
            continue
        v = row / delta
        col = np.asarray(get_col(j_star), dtype=np.complex128).copy()
        for kk in range(rank):
            col -= V[kk][j_star] * U[kk]
        used_cols[j_star] = True
        u = col
        un = float(np.linalg.norm(u))
        vn = float(np.linalg.norm(v))
        cross = 0.0
        for kk in range(rank):
            cross += np.real(np.vdot(U[kk], u) * np.vdot(V[kk], v))
        approx_norm2 += 2.0 * cross + (un * vn) ** 2
        U.append(u)
        V.append(v)
        rank += 1
        approx = np.array(U).T @ np.array(V)
        true_rel = float(np.linalg.norm(dense - approx)) / nA
        tested = un * vn
        bound = tol * np.sqrt(max(approx_norm2, 0.0))
        TRACE.append((rank, tested, bound, tested <= bound, true_rel))
        if approx_norm2 <= 0.0 or tested <= bound:
            break
        abscol = np.abs(col)
        abscol[used_rows] = -1.0
        if not (~used_rows).any():
            break
        i_star = int(np.argmax(abscol))
    return np.array(U).T.copy(), np.array(V).copy()


def main():
    HM.aca_partial = traced
    cls = bnc.load_design("loops.skyloop_lmatch")
    base = bc.default_nseg("loops.skyloop_lmatch")
    b = cls()
    b.nominal_nsegs = base * 2
    eng = MomwireEngine(b, solver=HMatrixSolver,
                        solver_kwargs={"degree": 2, "aca_tol": float(__import__("os").environ.get("ACA_TOL","1e-6"))},
                        ground=("finite", 13.0, 0.005))
    z = eng.impedance()[0]
    print(f"Z {z.real:.6f}{z.imag:+.6f}j     aca_tol {__import__("os").environ.get("ACA_TOL","1e-6")}\n")
    print(f"{'rank':>5s} {'||u||*||v||':>13s} {'tol*||A~||':>13s} {'stop?':>6s} {'TRUE rel err':>14s}")
    for rank, tested, bound, stop, true_rel in TRACE:
        print(f"{rank:5d} {tested:13.5e} {bound:13.5e} {str(stop):>6s} {true_rel:14.5e}")


main()
