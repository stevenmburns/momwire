"""Where is the 8e-4? Sandwich or ends — measured separately.

With GALERKIN axes on both sides the reversed block must equal the forward
block's transpose. probe2 said it does not, by 8e-4 at soil and by nothing
at eps~ = 1. This splits both blocks into (main sandwich) and (by-parts ends)
and compares each half against the corresponding half of the transpose, so
the disagreement is located instead of fitted.
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from momwire import RazorSolver
from momwire import _crossing_fill as CF
from test_crossing_serve_524 import crossing_deck
from test_buried_serve_553 import SOIL_A


def fwd_parts(ctx, A, B):
    eps_t, _em, k_p, _km, _c2, _am = ctx.medium
    gz = float(ctx.ground_z)
    c1 = CF._c1_moment(ctx.omega, ctx.mu)
    full = CF.cross_complete_block(ctx, A, B, corner=False)
    ends = CF._ends_and_corner(ctx, A, B, eps_t, k_p, c1, gz, corner=False)
    return full - ends, ends


def rev_parts(ctx, P, Q):
    """Reversed: main sandwich as the transpose of the same kernels; ends
    re-assigned by ROLE (BT on the test axis P, SW + SQ on the source Q)."""
    eps_t, _em, k_p, _km, _c2, _am = ctx.medium
    gz = float(ctx.ground_z)
    c1 = CF._c1_moment(ctx.omega, ctx.mu)
    k2sq = k_p * k_p
    dx = Q["nodes"][:, 0][:, None] - P["nodes"][:, 0][None, :]
    dy = Q["nodes"][:, 1][:, None] - P["nodes"][:, 1][None, :]
    rho = np.hypot(dx, dy)
    z = np.broadcast_to((Q["nodes"][:, 2] - gz)[:, None], rho.shape)
    zp = np.broadcast_to((P["nodes"][:, 2] - gz)[None, :], rho.shape)
    tb = CF._tables(ctx, eps_t, k_p, rho, z, zp, CF._CROSS_RTOL)
    U, V, W, dzW = tb["U"].T, tb["V"].T, tb["W"].T, tb["dzW"].T

    txP, tyP, tzP = P["t"].T
    txQ, tyQ, tzQ = Q["t"].T
    FP_w, FQ_w = P["F"] * P["w"], Q["F"] * Q["w"]
    FdP_w, FdQ_w = P["Fd"] * P["w"], Q["Fd"] * Q["w"]
    main = c1 * (
        (FP_w * txP) @ U @ (FQ_w * txQ).T
        + (FP_w * tyP) @ U @ (FQ_w * tyQ).T
        + (FP_w * tzP) @ (k2sq * V - dzW) @ (FQ_w * tzQ).T
        + FdP_w @ W @ (FQ_w * tzQ).T
        + (FP_w * tzP) @ W @ FdQ_w.T
        - FdP_w @ V @ FdQ_w.T
    )
    ends = np.zeros_like(main)
    for pt, sign, fv in P["ends"]:  # BT, test side
        rho_e = np.hypot(Q["nodes"][:, 0] - pt[0], Q["nodes"][:, 1] - pt[1])
        te = CF._tables(
            ctx,
            eps_t,
            k_p,
            rho_e,
            Q["nodes"][:, 2] - gz,
            np.full_like(rho_e, min(pt[2] - gz, 0.0)),
            CF._CROSS_RTOL,
        )
        ends += c1 * sign * np.outer(fv, FdQ_w @ te["V"])
    for pt, sign, fv in Q["ends"]:  # SW + SQ, source side
        rho_e = np.hypot(P["nodes"][:, 0] - pt[0], P["nodes"][:, 1] - pt[1])
        te = CF._tables(
            ctx,
            eps_t,
            k_p,
            rho_e,
            np.full_like(rho_e, max(pt[2] - gz, 0.0)),
            P["nodes"][:, 2] - gz,
            CF._CROSS_RTOL,
        )
        ends += -c1 * sign * np.outer((FP_w * tzP) @ te["W"], fv)
        ends += c1 * sign * np.outer(FdP_w @ te["V"], fv)
    return main, ends


def main():
    deck = crossing_deck(1)
    fs = {
        k: v
        for k, v in deck.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    rs = RazorSolver(**fs, nec5_quadrature=False, n_qp_path=8)
    geom = rs._build_geometry()
    seg_off = np.asarray(geom["seg_offsets"])
    b_idx = np.arange(seg_off[0], seg_off[1])
    a_idx = np.arange(seg_off[1], seg_off[2])

    for label, ge in (("eps~ = 1", (1.0, 0.0)), ("soil A", SOIL_A)):
        ctx = rs._crossing_context(geom, ground_eps=ge)
        A, B = CF.axis_data(ctx, a_idx), CF.axis_data(ctx, b_idx)
        fm, fe = fwd_parts(ctx, A, B)
        rm, re = rev_parts(ctx, B, A)
        print(f"\n=== {label} ===")
        for name, got, want in (
            ("main sandwich", rm, fm.T),
            ("by-parts ends", re, fe.T),
            ("whole block", rm + re, (fm + fe).T),
        ):
            s = np.abs(want).max()
            d = np.abs(got - want).max()
            print(
                f"  {name:>14}: max|want| {s:>12.6e}   max|got-want| {d:>12.6e}"
                f"   rel {d / s:>10.3e}"
            )


main()
