"""Every number the #813 step-(1) gates and the PR body quote, in one run."""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from momwire import RazorSolver
from momwire import _crossing_fill as CF
from test_crossing_serve_524 import crossing_deck
from test_buried_serve_553 import SOIL_A


def setup(nec5):
    deck = crossing_deck(1)
    fs = {
        k: v
        for k, v in deck.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    rs = RazorSolver(**fs, nec5_quadrature=nec5, n_qp_path=8)
    geom = rs._build_geometry()
    Z = rs._assemble_Z_from_prepared(geom, rs._assemble_Z_prepare(geom), rs.k, rs.omega)
    so = np.asarray(geom["seg_offsets"])
    bo = np.asarray(geom["basis_offsets"])
    return dict(
        rs=rs,
        geom=geom,
        Z=Z,
        b_idx=np.arange(so[0], so[1]),
        a_idx=np.arange(so[1], so[2]),
        bases_below=np.arange(bo[0], bo[1]),
        bases_above=np.arange(bo[1], bo[2]),
    )


def main():
    print("=" * 72)
    print("(i) eps~ = 1: the reversed block vs razor's free-space Z[below, above]")
    print("=" * 72)
    for nec5 in (False, True):
        f = setup(nec5)
        ctx = f["rs"]._crossing_context(f["geom"], ground_eps=(1.0, 0.0))
        n = f["geom"]["n_basis_total"]
        P = CF.path_test_axis(n, f["rs"]._path_test_rows(f["geom"], f["bases_below"]))
        Q = CF.axis_data(ctx, f["a_idx"])
        ref = f["Z"][np.ix_(f["bases_below"], f["bases_above"])]
        blk = CF.cross_complete_block_reversed(ctx, P, Q, corner=False)
        got = -blk[np.ix_(f["bases_below"], f["bases_above"])]
        rel = np.abs(got - ref).max() / np.abs(ref).max()
        big = np.abs(ref) > 1e-3 * np.abs(ref).max()
        r = np.median(got[big] / ref[big])
        print(
            f"  nec5_quadrature={str(nec5):>5}  dense: rel {rel:.4e}   "
            f"median ratio {r.real:+.8f}{r.imag:+.2e}j"
        )

    print()
    print("=" * 72)
    print("(ii) reciprocity MEASURED: t_ba vs t_ab.T, Galerkin axes both sides")
    print("=" * 72)
    f = setup(False)
    for label, ge in (("eps~ = 1", (1.0, 0.0)), ("soil A ", SOIL_A)):
        ctx = f["rs"]._crossing_context(f["geom"], ground_eps=ge)
        A, B = CF.axis_data(ctx, f["a_idx"]), CF.axis_data(ctx, f["b_idx"])
        fwd = CF.cross_complete_block(ctx, A, B, corner=False).T
        s = np.abs(fwd).max()
        for sw in (CF.SW_BY_PARTS, CF.SW_BY_ROLE):
            rev = CF.cross_complete_block_reversed(ctx, B, A, corner=False, sw_end=sw)
            d = np.abs(rev - fwd).max()
            big = np.abs(fwd) > 1e-3 * s
            r = np.median(rev[big] / fwd[big])
            print(
                f"  {label}  sw_end={sw:>10}: max|t_ba - t_ab.T| {d:.6e}   "
                f"rel {d / s:.4e}   ratio {r.real:+.8f}"
            )
    print()
    print("  and the same with PATH-tested rows (razor paths below, tents above),")
    print("  which is the object step (3) consumes — no transpose exists to compare,")
    print("  so the two sw_end spellings are compared against each other:")
    for label, ge in (("eps~ = 1", (1.0, 0.0)), ("soil A ", SOIL_A)):
        ctx = f["rs"]._crossing_context(f["geom"], ground_eps=ge)
        n = f["geom"]["n_basis_total"]
        P = CF.path_test_axis(n, f["rs"]._path_test_rows(f["geom"], f["bases_below"]))
        Q = CF.axis_data(ctx, f["a_idx"])
        b1 = CF.cross_complete_block_reversed(
            ctx, P, Q, corner=False, sw_end=CF.SW_BY_ROLE
        )
        b2 = CF.cross_complete_block_reversed(
            ctx, P, Q, corner=False, sw_end=CF.SW_BY_PARTS
        )
        sub = np.ix_(f["bases_below"], f["bases_above"])
        d = np.abs(b1[sub] - b2[sub]).max()
        print(
            f"  {label}: max|source - reciprocal| {d:.6e}   "
            f"rel {d / np.abs(b1[sub]).max():.4e}"
        )

    print()
    print("=" * 72)
    print("(iv) trunk lanes: dense vs split (Galerkin axes — the split cannot")
    print("     serve a path-tested axis and now refuses; see below)")
    print("=" * 72)
    for label, ge in (("eps~ = 1", (1.0, 0.0)), ("soil A ", SOIL_A)):
        ctx = f["rs"]._crossing_context(f["geom"], ground_eps=ge)
        A, B = CF.axis_data(ctx, f["a_idx"]), CF.axis_data(ctx, f["b_idx"])
        d1 = CF.cross_complete_block_reversed(ctx, B, A)
        d2 = CF.cross_complete_block_reversed_split(ctx, f["b_idx"], f["a_idx"], B, A)
        print(
            f"  {label}: max|dense - split| {np.abs(d1 - d2).max():.6e}   "
            f"rel {np.abs(d1 - d2).max() / np.abs(d1).max():.4e}"
        )

    print()
    print("  the trap, measured before the guard went in (both directions):")
    print("    forward,  path-tested rows, dense vs split: rel 2.0434e-01")
    print("    reversed, path-tested rows, dense vs split: rel 2.0435e-01")
    print("    forward,  Galerkin rows,    dense vs split: rel 1.7733e-18")
    ctx = f["rs"]._crossing_context(f["geom"], ground_eps=(1.0, 0.0))
    P = CF.path_test_axis(
        f["geom"]["n_basis_total"],
        f["rs"]._path_test_rows(f["geom"], f["bases_below"]),
    )
    Q = CF.axis_data(ctx, f["a_idx"])
    for name, call in (
        (
            "reversed split",
            lambda: CF.cross_complete_block_reversed_split(
                ctx, f["b_idx"], f["a_idx"], P, Q
            ),
        ),
        (
            "forward split ",
            lambda: CF.cross_complete_block_split(ctx, f["a_idx"], f["b_idx"], P, Q),
        ),
    ):
        try:
            call()
            print(f"    {name}: NO REFUSAL (regression)")
        except ValueError as e:
            print(f"    {name}: refused — {str(e)[:58]}...")


main()
