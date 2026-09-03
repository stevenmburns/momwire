"""#813 step 3 shape (a), probe 2: what elevation does the finest node reach?

Probe 1 found the binding guard is NOT the below/below remainder's
"strictly below" check but `_assemble_Z_below_plane`'s serve plan: with a
segment endpoint exactly at the interface, that endpoint's own self-pair has
rho = 0 and d + d' = 0, so `atan2(0, 0)` = 0 deg and the 1 deg grazing floor
refuses. At 1e-9 below the plane the same fill runs.

So the question for shape (a) is what elevation the sub-deck's near-plane
geometry actually reaches, on both decks and at both the plan's own sample
set (endpoints + centroids) and the crossing axis's graded panels.
"""

import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from momwire import _crossing_fill as CF  # noqa: E402
from momwire import _medium_spec as MS  # noqa: E402
from momwire import razor as _razor  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import crossing_deck, fan_rise_deck  # noqa: E402

FINE = dict(growth=2.0, panel_order=8, q=12)


def theta_min(pts, gz=0.0):
    """The plan's own scalar: min over pairs of atan2(d + d', rho), degrees."""
    d = gz - pts[:, 2]
    rho = np.hypot(
        pts[:, 0][:, None] - pts[:, 0][None, :],
        pts[:, 1][:, None] - pts[:, 1][None, :],
    )
    hh = d[:, None] + d[None, :]
    return float(np.degrees(np.arctan2(hh, rho)).min())


def report(name, deck):
    b = BSplineSolver(**deck)
    media = b._wire_media()
    rs_deck = {k: v for k, v in deck.items() if k != "junctions"}
    _razor._SERVE_CROSSING = False
    orig = RazorSolver._refuse_buried_geometry
    RazorSolver._refuse_buried_geometry = lambda self: None
    try:
        rs = RazorSolver(**rs_deck, nec5_quadrature=False, n_qp_path=8)
        geom = rs._build_geometry()
    finally:
        RazorSolver._refuse_buried_geometry = orig

    so = np.asarray(geom["seg_offsets"])
    seg_b = np.concatenate(
        [np.arange(so[w], so[w + 1]) for w, m in enumerate(media) if m == MS.BELOW]
    )
    p0, t, h = geom["seg_p0"], geom["seg_t"], geom["seg_h"]
    ends = np.concatenate(
        [
            p0[seg_b],
            p0[seg_b] + h[seg_b, None] * t[seg_b],
            p0[seg_b] + 0.5 * h[seg_b, None] * t[seg_b],
        ]
    )
    ctx = rs._crossing_context(geom, ground_eps=(13.0, 0.005))
    # This branch is off origin/main, which does not carry #836's per-axis
    # kwargs yet, so the fine setting goes through the module constants.
    from numpy.polynomial.legendre import leggauss

    saved = (CF._NEAR_GROWTH, CF._NEAR_GX, CF._NEAR_GW, CF._NEAR_Q)
    try:
        CF._NEAR_GROWTH = FINE["growth"]
        CF._NEAR_GX, CF._NEAR_GW = leggauss(FINE["panel_order"])
        CF._NEAR_Q = FINE["q"]
        ax = CF.axis_data(ctx, seg_b)
    finally:
        CF._NEAR_GROWTH, CF._NEAR_GX, CF._NEAR_GW, CF._NEAR_Q = saved
    nodes = ax["nodes"]
    shallowest = -nodes[:, 2].max()
    print(f"\n{name}")
    print(f"  below segments: {seg_b.size}")
    print(
        f"  plan's sample (endpoints + centroids):  theta_min = {theta_min(ends):8.4f} deg"
    )
    print(f"  crossing axis, graded panels @ {FINE}:")
    print(f"       shallowest node depth = {shallowest:.4e} m")
    print(f"       theta_min over those nodes = {theta_min(nodes):8.4f} deg")
    # the pair the coordinator named: shallowest node against the farthest tip
    i = int(np.argmax(nodes[:, 2]))
    j = int(np.argmax(np.hypot(nodes[:, 0] - nodes[i, 0], nodes[:, 1] - nodes[i, 1])))
    d = -nodes[i, 2] - nodes[j, 2]
    rho = float(np.hypot(nodes[i, 0] - nodes[j, 0], nodes[i, 1] - nodes[j, 1]))
    print(
        f"       (shallowest node, farthest node): rho = {rho:.4g} m, "
        f"theta = {np.degrees(np.arctan2(d, rho)):.4f} deg"
    )


def main():
    warnings.filterwarnings("ignore")
    report("crossing_deck(1)", crossing_deck(1))
    report("fan_rise_deck()", fan_rise_deck())


if __name__ == "__main__":
    main()
