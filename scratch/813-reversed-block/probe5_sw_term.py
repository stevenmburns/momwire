"""The whole reciprocity gap is the SW end term — confirmed, not inferred.

probe4: with Galerkin axes both sides the reversed block's MAIN SANDWICH is
the transpose to 3e-16 at eps~ = 1 and at soil (so `dzW` transposed is the
dyad, and no kernel swap is needed), while the BY-PARTS ENDS are bit-equal at
eps~ = 1 and 6.8e-4 apart at soil. W is the only kernel that is 0 at eps~ = 1
and nonzero at soil, and it rides exactly one end term: SW. This drops SW from
both spellings and checks the rest agrees exactly.

That places the reversed block's one real ambiguity on momwire#813's own open
derivation (b) — whether the SW end term the trunk emits belongs to the
source axis's ends (the by-parts derivation's placement) or is carried by
reciprocity from the forward block.
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from momwire import RazorSolver
from momwire import _crossing_fill as CF
from test_crossing_serve_524 import crossing_deck
from test_buried_serve_553 import SOIL_A


def fwd_ends(ctx, A, B, *, with_sw):
    eps_t, _em, k_p, _km, _c2, _am = ctx.medium
    gz, c1 = float(ctx.ground_z), CF._c1_moment(ctx.omega, ctx.mu)
    out = np.zeros((A["n_basis"], B["n_basis"]), dtype=np.complex128)
    _tx, _ty, tzA = A["t"].T
    FA_w, FdA_w = A["F"] * A["w"], A["Fd"] * A["w"]
    FdB_w = B["Fd"] * B["w"]
    for pt, sign, fv in A["ends"]:
        rho_e = np.hypot(pt[0] - B["nodes"][:, 0], pt[1] - B["nodes"][:, 1])
        te = CF._tables(
            ctx,
            eps_t,
            k_p,
            rho_e,
            np.full_like(rho_e, max(pt[2] - gz, 0.0)),
            B["nodes"][:, 2] - gz,
            CF._CROSS_RTOL,
        )
        out += c1 * sign * np.outer(fv, FdB_w @ te["V"])
    for pt, sign, fv in B["ends"]:
        rho_e = np.hypot(A["nodes"][:, 0] - pt[0], A["nodes"][:, 1] - pt[1])
        te = CF._tables(
            ctx,
            eps_t,
            k_p,
            rho_e,
            A["nodes"][:, 2] - gz,
            np.full_like(rho_e, pt[2] - gz),
            CF._CROSS_RTOL,
        )
        if with_sw:
            out += -c1 * sign * np.outer((FA_w * tzA) @ te["W"], fv)
        out += c1 * sign * np.outer(FdA_w @ te["V"], fv)
    return out


def rev_ends(ctx, P, Q, *, with_sw):
    eps_t, _em, k_p, _km, _c2, _am = ctx.medium
    gz, c1 = float(ctx.ground_z), CF._c1_moment(ctx.omega, ctx.mu)
    out = np.zeros((P["n_basis"], Q["n_basis"]), dtype=np.complex128)
    _tx, _ty, tzP = P["t"].T
    FP_w, FdP_w = P["F"] * P["w"], P["Fd"] * P["w"]
    FdQ_w = Q["Fd"] * Q["w"]
    for pt, sign, fv in P["ends"]:  # BT, test side (below -> z' slot)
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
        out += c1 * sign * np.outer(fv, FdQ_w @ te["V"])
    for pt, sign, fv in Q["ends"]:  # SW + SQ, source side (above -> z slot)
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
        if with_sw:
            out += -c1 * sign * np.outer((FP_w * tzP) @ te["W"], fv)
        out += c1 * sign * np.outer(FdP_w @ te["V"], fv)
    return out


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
    b_idx, a_idx = (
        np.arange(seg_off[0], seg_off[1]),
        np.arange(seg_off[1], seg_off[2]),
    )
    for label, ge in (("eps~ = 1", (1.0, 0.0)), ("soil A", SOIL_A)):
        ctx = rs._crossing_context(geom, ground_eps=ge)
        A, B = CF.axis_data(ctx, a_idx), CF.axis_data(ctx, b_idx)
        print(f"\n=== {label} ===")
        for with_sw in (False, True):
            f = fwd_ends(ctx, A, B, with_sw=with_sw).T
            r = rev_ends(ctx, B, A, with_sw=with_sw)
            d = np.abs(r - f).max()
            print(
                f"  ends, SW {'ON ' if with_sw else 'OFF'}: max|want| "
                f"{np.abs(f).max():>12.6e}  max|got-want| {d:>12.6e}"
                f"  rel {d / np.abs(f).max():>10.3e}"
            )
        # The SW term alone, both placements.
        sw_f = (
            fwd_ends(ctx, A, B, with_sw=True) - fwd_ends(ctx, A, B, with_sw=False)
        ).T
        sw_r = rev_ends(ctx, B, A, with_sw=True) - rev_ends(ctx, B, A, with_sw=False)
        print(
            f"  SW alone: |transposed| {np.abs(sw_f).max():.6e}   "
            f"|role-placed| {np.abs(sw_r).max():.6e}   "
            f"max|diff| {np.abs(sw_r - sw_f).max():.6e}"
        )


main()
