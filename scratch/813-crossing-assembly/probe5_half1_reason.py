"""#813 step 3, probe 5: half 1's junction-row-half residual, across q.

`tests/test_razor_crossing_axis_813.py` records the junction row's above half
at 5.3e-5 against a 1e-4 bar, with the reason "quadrature: 4-point source
Gauss vs razor's 12 + statics". Probe 4 found the assembly's residual on that
same row does NOT move with the trunk's source order. So ask half 1's own
measurement the same question, on its own construction.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _crossing_fill as CF  # noqa: E402
from probe1_blocks import setup  # noqa: E402


def razor_half_row(f, halves):
    """The reference `test_razor_crossing_axis_813._razor_half_row` builds."""
    rs, geom, jn, Z = f["rs"], f["geom"], f["jn"], f["Z"]
    k, omega = rs.k, rs.omega
    prep = rs._assemble_Z_prepare(geom)
    gA, gB = dict(geom), dict(geom)
    gA["wing_sigma"] = geom["wing_sigma"].copy()
    gA["wing_sigma"][jn, 1] = 0.0
    gB["wing_sigma"] = geom["wing_sigma"].copy()
    gB["wing_sigma"][jn, 0] = 0.0
    Z_A = rs._assemble_Z_from_prepared(gA, rs._assemble_Z_prepare(gA), k, omega)
    Z_B = rs._assemble_Z_from_prepared(gB, rs._assemble_Z_prepare(gB), k, omega)
    T1_half = (Z_B if halves == "B" else Z_A) - (Z_A + Z_B - Z)
    seg_h, seg_t, seg_p0 = geom["seg_h"], geom["seg_t"], geom["seg_p0"]
    cent = seg_p0 + 0.5 * seg_h[:, None] * seg_t
    node = rs._knot_points(geom)[jn]
    s_half = int(geom["wing_seg"][jn, 1 if halves == "B" else 0])
    before, after = (node, cent[s_half]) if halves == "B" else (cent[s_half], node)
    M0 = rs._seg_moments_from_prepared(
        rs._seg_moments_prepare(
            np.array([before, after]), geom, rs._kernel_radius(geom)
        ),
        k,
        2,
        need_m1=False,
    )[0]
    dM0 = M0[1] - M0[0]
    T2h = dM0[prep["s_a"]] * prep["q_a"] + dM0[prep["s_b"]] * prep["q_b"]
    return T1_half[jn] - T2h / (1j * omega * rs.eps)


def main():
    f = setup(False)
    rs, geom, jn = f["rs"], f["geom"], f["jn"]
    bo = np.asarray(geom["basis_offsets"])
    cols_below = np.arange(bo[0], bo[1])
    ctx = rs._crossing_context(geom, ground_eps=(1.0, 0.0))
    ref = razor_half_row(f, "B")[cols_below]
    A3 = CF.path_test_axis(
        geom["n_basis_total"], rs._path_test_rows(geom, [jn], halves="B")
    )
    near0, far0 = CF._NEAR_Q, CF._FAR_Q
    print("half 1's own gate: the junction row's above half vs razor's, across q")
    print(f"{'q':>4}  {'rel':>10}")
    try:
        for q in (4, 6, 8, 12, 16, 24, 32):
            CF._NEAR_Q = CF._FAR_Q = q
            B = CF.axis_data(ctx, f["seg_below"])
            got = -CF.cross_complete_block(ctx, A3, B, corner=False)[jn, cols_below]
            print(f"{q:>4}  {np.abs(got - ref).max() / np.abs(ref).max():>10.3e}")
    finally:
        CF._NEAR_Q, CF._FAR_Q = near0, far0


if __name__ == "__main__":
    main()
