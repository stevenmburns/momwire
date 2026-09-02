"""#813 step 3, probe 3: WHERE the eps~ = 1 residual lives, and is it quadrature?

Probe 2's "bookkeeping control" was ill-posed and this records why, because
it is a structural fact about razor and not a bug in the probe.

`_testing_paths` builds a row's tangents from `wing_sigma`, so zeroing a
wing's sigma switches off BOTH that half of the row's T1 AND that wing of the
column's source tangent. One flag, two effects. So the sigma trick can
express the DIAGONAL half-x-wing pieces (above half x above wing, below half
x below wing) and cannot express the off-diagonal ones -- which is exactly
why the cross blocks have to come from the trunk, where the row axis (a
chopped path) and the column axis (a segment set) are independent.

That makes the eps~ = 1 collapse against razor's free-space Z the gate, with
no cheaper control available underneath it.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _crossing_fill as CF  # noqa: E402
from probe1_blocks import masked_fill, setup  # noqa: E402


def build(f, ctx, n_qp=None):
    rs, geom, jn = f["rs"], f["geom"], f["jn"]
    n = geom["n_basis_total"]
    Z_ab = masked_fill(f, 0, "B")
    Z_be = masked_fill(f, 1, "A")
    rows_a = [(m, "both") for m in f["b_above"]] + [(jn, "B")]
    rows_b = [(m, "both") for m in f["b_below"]] + [(jn, "A")]
    A = CF.path_test_axis(
        n, [r for m, h in rows_a for r in rs._path_test_rows(geom, [m], halves=h)]
    )
    P = CF.path_test_axis(
        n, [r for m, h in rows_b for r in rs._path_test_rows(geom, [m], halves=h)]
    )
    kw = {} if n_qp is None else {"n_qp": n_qp}
    ax_a = CF.axis_data(ctx, f["seg_above"], **kw)
    ax_b = CF.axis_data(ctx, f["seg_below"], **kw)
    R_a = np.concatenate([f["b_above"], [jn]])
    R_b = np.concatenate([f["b_below"], [jn]])
    M = np.zeros((n, n), dtype=complex)
    M[np.ix_(R_a, R_a)] += Z_ab[np.ix_(R_a, R_a)]
    M[np.ix_(R_b, R_b)] += Z_be[np.ix_(R_b, R_b)]
    M[np.ix_(R_a, R_b)] -= CF.cross_complete_block(ctx, A, ax_b, corner=False)[
        np.ix_(R_a, R_b)
    ]
    M[np.ix_(R_b, R_a)] -= CF.cross_complete_block_reversed(ctx, P, ax_a, corner=False)[
        np.ix_(R_b, R_a)
    ]
    return M


def main():
    import inspect

    print("axis_data signature:", inspect.signature(CF.axis_data))
    for nec5 in (False, True):
        f = setup(nec5)
        Z, jn = f["Z"], f["jn"]
        ctx = f["rs"]._crossing_context(f["geom"], ground_eps=(1.0, 0.0))
        M = build(f, ctx)
        d = np.abs(M - Z) / np.abs(Z).max()
        off = d.copy()
        off[jn, :] = 0.0
        off[:, jn] = 0.0
        print(f"\nnec5={nec5}")
        print(f"  whole matrix          rel = {d.max():.3e}")
        print(f"  excluding the tent's row and column = {off.max():.3e}")
        print(f"  (jn, jn) alone        rel = {d[jn, jn]:.3e}")
        r = d[jn, :].copy()
        r[jn] = 0
        c = d[:, jn].copy()
        c[jn] = 0
        print(f"  tent row  (excl jn,jn) = {r.max():.3e}")
        print(f"  tent col  (excl jn,jn) = {c.max():.3e}")
        # Solved impedance, which is what a user sees.
        zf = np.linalg.solve(Z, np.eye(Z.shape[0])[:, [jn]])
        zm = np.linalg.solve(M, np.eye(M.shape[0])[:, [jn]])
        print(
            f"  |dY|/|Y| at the tent port = "
            f"{abs(zm[jn, 0] - zf[jn, 0]) / abs(zf[jn, 0]):.3e}"
        )


if __name__ == "__main__":
    main()
