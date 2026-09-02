"""Isolate the W family and read its reversed spelling off the forward block.

probe2 left an 8e-4 residual at soil under every candidate, and flipping the
W sign barely moved it — so the error is not a sign on a term I had, it is a
term I have wrong. This probe stops guessing and MEASURES.

The forward block is the shipped, validated path. Zeroing {W, dzW, dzpW} in
the tables splits it exactly:

    fwd = fwd_noW + fwd_W

and with Galerkin axes on both sides the reversed block must equal fwd.T, so
its W-family contribution must equal fwd_W.T. That is a known matrix. The
probe builds each candidate W-family term of the reversed assembly and reports
which combination reproduces it.
"""

import sys
import pathlib
import contextlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from momwire import RazorSolver
from momwire import _crossing_fill as CF
from test_crossing_serve_524 import crossing_deck
from test_buried_serve_553 import SOIL_A

_ZERO = ("W", "dzW", "dzpW")


@contextlib.contextmanager
def w_family_zeroed():
    real = CF._tables

    def patched(*a, **kw):
        out = dict(real(*a, **kw))
        for kk in _ZERO:
            if kk in out:
                out[kk] = np.zeros_like(out[kk])
        return out

    CF._tables = patched
    try:
        yield
    finally:
        CF._tables = real


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
    ctx = rs._crossing_context(geom, ground_eps=SOIL_A)
    A = CF.axis_data(ctx, a_idx)  # above
    B = CF.axis_data(ctx, b_idx)  # below

    fwd = CF.cross_complete_block(ctx, A, B, corner=False)
    with w_family_zeroed():
        fwd_noW = CF.cross_complete_block(ctx, A, B, corner=False)
    fwd_W = fwd - fwd_noW
    target = fwd_W.T  # what the reversed block's W family must be
    print(
        f"forward W-family contribution: max {np.abs(fwd_W).max():.6e} "
        f"(block max {np.abs(fwd).max():.6e})"
    )

    # The reversed block's raw ingredients, in [P_pt, Q_pt] indexing.
    eps_t, _em, k_p, _km, _c2, _am = ctx.medium
    gz = float(ctx.ground_z)
    c1 = CF._c1_moment(ctx.omega, ctx.mu)
    P, Q = B, A  # P = below (test rows), Q = above (source cols)
    dx = Q["nodes"][:, 0][:, None] - P["nodes"][:, 0][None, :]
    dy = Q["nodes"][:, 1][:, None] - P["nodes"][:, 1][None, :]
    rho = np.hypot(dx, dy)
    z = np.broadcast_to((Q["nodes"][:, 2] - gz)[:, None], rho.shape)
    zp = np.broadcast_to((P["nodes"][:, 2] - gz)[None, :], rho.shape)
    tb = CF._tables(ctx, eps_t, k_p, rho, z, zp, CF._CROSS_RTOL)

    _txP, _tyP, tzP = P["t"].T
    _txQ, _tyQ, tzQ = Q["t"].T
    FP_w, FQ_w = P["F"] * P["w"], Q["F"] * Q["w"]
    FdP_w, FdQ_w = P["Fd"] * P["w"], Q["Fd"] * Q["w"]

    # Candidate slot contributions (without c1), each in [P_basis, Q_basis].
    slots = {}
    for kk in ("W", "dzW", "dzpW"):
        K = tb[kk].T
        slots[f"zz:{kk}"] = -(FP_w * tzP) @ K @ (FQ_w * tzQ).T
        slots[f"tzP-FdQ:{kk}"] = (FP_w * tzP) @ K @ FdQ_w.T
        slots[f"FdP-tzQ:{kk}"] = FdP_w @ K @ (FQ_w * tzQ).T

    # The source-side SW end term (Q's ends, above -> z slot), W only.
    sw = np.zeros_like(target)
    for pt, sign, fv in Q["ends"]:
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
        sw += -sign * np.outer((FP_w * tzP) @ te["W"], fv)
    slots["SW-end(Q):W"] = sw

    print("\nEach candidate slot, scaled by c1, against the target:")
    print(f"  {'slot':>18} {'max|slot|':>14} {'<slot,target>/|slot|^2':>28}")
    tgt = target / c1
    for name, M in slots.items():
        n2 = np.vdot(M, M).real
        coef = np.vdot(M, tgt) / n2 if n2 > 0 else 0.0
        print(
            f"  {name:>18} {np.abs(M * c1).max():>14.6e} "
            f"{coef.real:>+14.6f}{coef.imag:>+13.6f}j"
        )

    # Least squares over the whole candidate set: which combination IS it?
    names = list(slots)
    Mx = np.stack([slots[n].ravel() for n in names], axis=1)
    coef, *_ = np.linalg.lstsq(Mx, tgt.ravel(), rcond=None)
    resid = Mx @ coef - tgt.ravel()
    print(
        f"\nleast-squares over all {len(names)} slots: "
        f"residual {np.abs(resid).max():.6e} vs target {np.abs(tgt).max():.6e}"
    )
    for n, c in zip(names, coef):
        if abs(c) > 1e-6:
            print(f"   {n:>18}  {c.real:>+12.6f}{c.imag:>+12.6f}j")


main()
