"""#813 step 3, probe 7: the knob probe 4's sweep could not reach.

Probe 4 swept `_NEAR_Q` and found the node row's 5.3e-5 order-independent.
That was the wrong knob. Off the node row the cross blocks pair DIFFERENT
wires, so plain Gauss converges; the node row's chopped half ends AT the node
on the below wire's last segment, whose BT integrand ~ 1/sqrt(a^2 + s^2) from
s = 0 is integrated by `_graded_u`'s a-scale panels — `_NEAR_GROWTH` and
`_NEAR_GX`, which `_NEAR_Q` never touches.

Reproduced here independently before anything is built on it.
"""

import pathlib
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _crossing_fill as CF  # noqa: E402
from probe1_blocks import setup  # noqa: E402
from probe5_half1_reason import razor_half_row  # noqa: E402


def node_row_rel(f, ctx, growth, panel_order, q):
    """Half 1's own junction-row-half comparison under one grading setting."""
    rs, geom, jn = f["rs"], f["geom"], f["jn"]
    saved = (CF._NEAR_GROWTH, CF._NEAR_GX, CF._NEAR_GW, CF._NEAR_Q)
    try:
        CF._NEAR_GROWTH = growth
        CF._NEAR_GX, CF._NEAR_GW = leggauss(panel_order)
        CF._NEAR_Q = q
        ref = razor_half_row(f, "B")[f["b_below"]]
        A3 = CF.path_test_axis(
            geom["n_basis_total"], rs._path_test_rows(geom, [jn], halves="B")
        )
        B = CF.axis_data(ctx, f["seg_below"])
        got = -CF.cross_complete_block(ctx, A3, B, corner=False)[jn, f["b_below"]]
        n_nodes = B["nodes"].shape[0]
        return float(np.abs(got - ref).max() / np.abs(ref).max()), n_nodes
    finally:
        CF._NEAR_GROWTH, CF._NEAR_GX, CF._NEAR_GW, CF._NEAR_Q = saved


def main():
    f = setup(False)
    ctx = f["rs"]._crossing_context(f["geom"], ground_eps=(1.0, 0.0))
    print(f"{'growth':>7} {'panel':>6} {'Q':>3}  {'rel':>10}  {'below-axis nodes':>17}")
    for growth, panel, q in (
        (4.0, 4, 4),  # today's shipped setting
        (4.0, 4, 16),
        (4.0, 16, 4),
        (2.0, 8, 4),
        (2.0, 16, 4),
        (1.5, 16, 4),
        (4.0, 4, 8),
        (4.0, 8, 8),
        (2.0, 8, 8),
        (4.0, 16, 8),
        (2.0, 16, 8),
        (1.5, 16, 8),
        (1.25, 24, 8),
        (1.25, 24, 12),
    ):
        rel, n = node_row_rel(f, ctx, growth, panel, q)
        print(f"{growth:>7} {panel:>6} {q:>3}  {rel:>10.3e}  {n:>17}")


if __name__ == "__main__":
    main()
