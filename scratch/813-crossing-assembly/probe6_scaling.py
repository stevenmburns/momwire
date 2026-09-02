"""#813 step 3, probe 6: what DOES the converged 5e-5 scale with?

Probe 5 established the junction-row residual is order-independent, so the
recorded "quadrature" reason is a misattribution. The two candidates left are
a RADIUS-convention difference (the trunk folds the thin-wire radius as
rho_eff = sqrt(rho^2 + a^2) on its designed tables; razor's kernel uses
R = sqrt(d^2 + a^2)) and a MESH-convergent discretisation difference.

Both are cheap to separate: sweep a, sweep N.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import RazorSolver  # noqa: E402
from momwire import _crossing_fill as CF  # noqa: E402
from probe5_half1_reason import razor_half_row  # noqa: E402
from test_crossing_serve_524 import crossing_deck  # noqa: E402


def one(radius=None, nseg=None):
    deck = crossing_deck(1)
    fs = {
        k: v
        for k, v in deck.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    if radius is not None:
        fs["wire_radius"] = radius
    if nseg is not None:
        # Scale the deck's own per-edge counts rather than flattening them:
        # crossing_deck(1) has 3 edges per wire and a graded node region.
        fs["n_per_edge_per_wire"] = [
            [max(1, int(round(n * nseg))) for n in per]
            for per in fs["n_per_edge_per_wire"]
        ]
    rs = RazorSolver(**fs, nec5_quadrature=False, n_qp_path=8)
    geom = rs._build_geometry()
    Z = rs._assemble_Z_from_prepared(geom, rs._assemble_Z_prepare(geom), rs.k, rs.omega)
    so = np.asarray(geom["seg_offsets"])
    bo = np.asarray(geom["basis_offsets"])
    f = dict(
        rs=rs,
        geom=geom,
        Z=Z,
        seg_below=np.arange(so[0], so[1]),
        seg_above=np.arange(so[1], so[2]),
        b_below=np.arange(bo[0], bo[1]),
        b_above=np.arange(bo[1], bo[2]),
        jn=geom["n_basis_total"] - 1,
    )
    jn = f["jn"]
    ctx = rs._crossing_context(geom, ground_eps=(1.0, 0.0))
    ref = razor_half_row(f, "B")[f["b_below"]]
    A3 = CF.path_test_axis(
        geom["n_basis_total"], rs._path_test_rows(geom, [jn], halves="B")
    )
    B = CF.axis_data(ctx, f["seg_below"])
    got = -CF.cross_complete_block(ctx, A3, B, corner=False)[jn, f["b_below"]]
    return float(np.abs(got - ref).max() / np.abs(ref).max())


def main():
    base = crossing_deck(1)
    print("wire_radius of the deck:", base["wire_radius"])
    print("n_per_edge_per_wire:", base["n_per_edge_per_wire"])
    print("\nradius sweep (mesh fixed):")
    a0 = base["wire_radius"]
    for mult in (0.1, 0.3, 1.0, 3.0, 10.0):
        print(
            f"  a = {a0 * mult:>10.3e} ({mult:>4}x a0)   rel = {one(radius=a0 * mult):.3e}"
        )
    print("\nmesh sweep (radius fixed):")
    for m in (1, 2, 4, 8):
        print(f"  mesh x{m:<2}   rel = {one(nseg=m):.3e}")


if __name__ == "__main__":
    main()
