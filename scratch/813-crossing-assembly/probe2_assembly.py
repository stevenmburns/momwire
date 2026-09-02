"""#813 step 3, probe 2: the four-mask assembly.

    M[R_a, C_a] += above fill        (above half tests above wings)
    M[R_b, C_b] += below fill        (below half tests below wings)
    M[R_a, C_b] -= t_ab              (above half tests below wings, trunk)
    M[R_b, C_a] -= t_ba              (below half tests above wings, reversed)

with R_a = C_a = above bases + the junction tent and R_b = C_b = below bases
+ the junction tent, so the tent is in all four and its (jn, jn) entry is the
sum of its four half-x-wing pieces. Every other entry gets exactly one term.

Run 1 takes all four blocks from razor's own free-space fill: that measures
the BOOKKEEPING alone and should be roundoff. Run 2 swaps the two cross
blocks for the trunk's, which is the real assembly at eps~ = 1.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from momwire import _crossing_fill as CF  # noqa: E402
from probe1_blocks import masked_fill, rel, setup  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def assemble(f, Z_ab, Z_be, t_ab, t_ba):
    n = f["geom"]["n_basis_total"]
    jn = f["jn"]
    R_a = np.concatenate([f["b_above"], [jn]])
    R_b = np.concatenate([f["b_below"], [jn]])
    M = np.zeros((n, n), dtype=complex)
    M[np.ix_(R_a, R_a)] += Z_ab[np.ix_(R_a, R_a)]
    M[np.ix_(R_b, R_b)] += Z_be[np.ix_(R_b, R_b)]
    M[np.ix_(R_a, R_b)] += t_ab[np.ix_(R_a, R_b)]
    M[np.ix_(R_b, R_a)] += t_ba[np.ix_(R_b, R_a)]
    return M


def main():
    for nec5 in (False, True):
        f = setup(nec5)
        Z, jn = f["Z"], f["jn"]
        Z_ab = masked_fill(f, 0, "B")
        Z_be = masked_fill(f, 1, "A")

        # ---- run 1: bookkeeping only, cross blocks from razor's own fill.
        # The 2x2 is (row HALF) x (column WING), and for the junction tent
        # those are four different masked fills: the chop picks the row half,
        # the wing-zeroing picks the column wing. Taking the cross blocks off
        # `Z_ab` / `Z_be` instead double-counts the tent's column, because it
        # is in both column sets -- the first version of this probe did
        # exactly that and read 0.54.
        Z_x1 = masked_fill(f, 1, "B")  # above half tests the BELOW wing
        Z_x2 = masked_fill(f, 0, "A")  # below half tests the ABOVE wing
        M = assemble(f, Z_ab, Z_be, Z_x1, Z_x2)
        print(
            f"nec5={str(nec5):>5}  bookkeeping (razor's own cross blocks): "
            f"rel = {rel(M, Z):.3e}"
        )

        # ---- run 2: the real assembly at eps~ = 1
        rs, geom = f["rs"], f["geom"]
        ctx = rs._crossing_context(geom, ground_eps=(1.0, 0.0))
        n = geom["n_basis_total"]
        R_a = np.concatenate([f["b_above"], [jn]])
        R_b = np.concatenate([f["b_below"], [jn]])
        # above-half paths for the above rows (the tent's above half is "B")
        rows_a = [(m, "both") for m in f["b_above"]] + [(jn, "B")]
        rows_b = [(m, "both") for m in f["b_below"]] + [(jn, "A")]
        A = CF.path_test_axis(
            n,
            [r for m, h in rows_a for r in rs._path_test_rows(geom, [m], halves=h)],
        )
        P = CF.path_test_axis(
            n,
            [r for m, h in rows_b for r in rs._path_test_rows(geom, [m], halves=h)],
        )
        ax_a = CF.axis_data(ctx, f["seg_above"])
        ax_b = CF.axis_data(ctx, f["seg_below"])
        t_ab = -CF.cross_complete_block(ctx, A, ax_b, corner=False)
        t_ba = -CF.cross_complete_block_reversed(ctx, P, ax_a, corner=False)
        M2 = assemble(f, Z_ab, Z_be, t_ab, t_ba)
        print(
            f"nec5={str(nec5):>5}  real assembly at eps~=1:              "
            f"rel = {rel(M2, Z):.3e}"
        )
        # where does the residual live?
        d = np.abs(M2 - Z) / np.abs(Z).max()
        i, j = np.unravel_index(np.argmax(d), d.shape)
        print(
            f"         worst entry ({i},{j}) = {d[i, j]:.3e}   (jn = {jn}); block rels:"
        )
        for nm, r, c in (
            ("a x a", R_a, R_a),
            ("b x b", R_b, R_b),
            ("a x b", R_a, R_b),
            ("b x a", R_b, R_a),
        ):
            ix = np.ix_(r, c)
            print(f"           {nm}: {rel(M2[ix], Z[ix]):.3e}")
        print()


if __name__ == "__main__":
    main()
