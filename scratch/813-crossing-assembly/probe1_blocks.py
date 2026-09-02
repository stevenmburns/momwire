"""#813 step 3, probe 1: do the four blocks exist at eps~ = 1?

Before any assembly: check each piece separately against razor's free-space
fill on the same geometry, which at eps~ = 1 IS the truth (the interface
vanishes and the crossing deck is one straight wire).

Wing order on the junction tent: side A is the group's first-listed end.
`crossing_deck` lists the BELOW wire first, so wing 0 = below, wing 1 =
above; the ABOVE half of the chopped path is therefore `halves="B"`.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from momwire import RazorSolver  # noqa: E402
from test_crossing_serve_524 import crossing_deck  # noqa: E402

EPS1 = (1.0, 0.0)


def setup(nec5=False):
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
        seg_below=np.arange(so[0], so[1]),
        seg_above=np.arange(so[1], so[2]),
        b_below=np.arange(bo[0], bo[1]),
        b_above=np.arange(bo[1], bo[2]),
        jn=geom["n_basis_total"] - 1,
    )


def wing_check(f):
    """Which wing of the junction tent sits on which wire."""
    geom, jn = f["geom"], f["jn"]
    out = []
    for j in (0, 1):
        s = int(geom["wing_seg"][jn, j])
        out.append("below" if s in f["seg_below"] else "above")
    return out


def masked_fill(f, zero_wing, chop_side):
    """Razor's own fill with one wing of the junction tent switched off and
    its testing path chopped at the node."""
    rs, geom, jn = f["rs"], f["geom"], f["jn"]
    g = dict(geom)
    g["wing_sigma"] = geom["wing_sigma"].copy()
    g["wing_sigma"][jn, zero_wing] = 0.0
    prep = rs._assemble_Z_prepare(g, chop={jn: chop_side})
    return rs._assemble_Z_from_prepared(g, prep, rs.k, rs.omega)


def rel(got, ref):
    return float(np.abs(got - ref).max() / np.abs(ref).max())


def main():
    f = setup()
    print("junction tent wings:", wing_check(f), "(index 0 = side A)")
    print(f"bases: below {f['b_below']}, above {f['b_above']}, junction {f['jn']}")
    Z = f["Z"]

    # The two same-medium blocks, off the masked fills.
    Z_ab = masked_fill(f, 0, "B")  # above fill: below wing off, above half
    Z_be = masked_fill(f, 1, "A")  # below fill: above wing off, below half
    for name, blk, rows, cols in (
        ("above x above", Z_ab, f["b_above"], f["b_above"]),
        ("below x below", Z_be, f["b_below"], f["b_below"]),
        ("above x below (from the above fill)", Z_ab, f["b_above"], f["b_below"]),
        ("below x above (from the below fill)", Z_be, f["b_below"], f["b_above"]),
    ):
        ix = np.ix_(rows, cols)
        print(f"  {name:38s} rel vs free-space Z = {rel(blk[ix], Z[ix]):.3e}")

    print()
    print("the junction tent, by piece:")
    print(
        f"  above-fill column [above rows, jn]  rel = "
        f"{rel(Z_ab[f['b_above'], f['jn']], Z[f['b_above'], f['jn']]):.3e}"
    )
    print(
        f"  below-fill column [below rows, jn]  rel = "
        f"{rel(Z_be[f['b_below'], f['jn']], Z[f['b_below'], f['jn']]):.3e}"
    )
    print("  (each should be WRONG on its own: one wing is switched off)")
    print(
        f"  sum of the two columns on above rows  rel = "
        f"{rel(Z_ab[f['b_above'], f['jn']] + Z_be[f['b_above'], f['jn']], Z[f['b_above'], f['jn']]):.3e}"
    )
    print(
        f"  sum of the two rows at [jn, below cols] rel = "
        f"{rel(Z_ab[f['jn'], f['b_below']] + Z_be[f['jn'], f['b_below']], Z[f['jn'], f['b_below']]):.3e}"
    )


if __name__ == "__main__":
    main()
