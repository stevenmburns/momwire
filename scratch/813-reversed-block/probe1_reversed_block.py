"""Which spelling is the reversed cross block? — momwire#813 step (1).

The trunk's designed tables take an observer ABOVE and a source BELOW
(`six_point` raises unless z >= 0 >= z'), so the block with BELOW rows and
ABOVE columns cannot be had by swapping the two axes: the above point must
stay in the `z` slot whichever ROLE it plays. Two assignments are therefore
independent, and the forward block conflates them because there they agree:

    which table slot   above -> z,  below -> z'
    which by-parts     test  -> BT, source -> SW + SQ

This probe measures the four candidate spellings of the reversed block
against razor's own free-space Z at eps~ = 1, where the interface vanishes
and razor IS the truth (the momwire#651 protocol). The two knobs:

    zz kernel : k^2 V - dzW   or   k^2 V - dzpW
                (dz is the ABOVE coordinate's derivative; the observer is
                 below in the reversed block, so its own derivative is dz')
    W sign    : +W or -W on the two mixed terms and the SW end term
                (in a homogeneous medium dz = -dz', so the sign is exactly
                 what an observer/source swap can flip)
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

from momwire import RazorSolver
from momwire import _crossing_fill as CF
from momwire import _near_interface
from test_crossing_serve_524 import crossing_deck


def build(level=1):
    deck = crossing_deck(level)
    fs = {
        k: v
        for k, v in deck.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    rs = RazorSolver(**fs, nec5_quadrature=False, n_qp_path=8)
    geom = rs._build_geometry()
    prep = rs._assemble_Z_prepare(geom)
    Z = rs._assemble_Z_from_prepared(geom, prep, rs.k, rs.omega)
    ctx = rs._crossing_context(geom, ground_eps=(1.0, 0.0))
    seg_off = np.asarray(geom["seg_offsets"])
    bas_off = np.asarray(geom["basis_offsets"])
    return dict(
        rs=rs,
        geom=geom,
        Z=Z,
        ctx=ctx,
        b_idx=np.arange(seg_off[0], seg_off[1]),
        a_idx=np.arange(seg_off[1], seg_off[2]),
        bases_below=np.arange(bas_off[0], bas_off[1]),
        bases_above=np.arange(bas_off[1], bas_off[2]),
        jn=geom["n_basis_total"] - 1,
    )


def reversed_block(ctx, P_below, Q_above, *, zz_key, w_sign, corner=False):
    """t_ba over (below test rows P) x (above source columns Q).

    The tables are evaluated with Q in the `z` slot and P in the `z'` slot —
    the only order `six_point` accepts — and every kernel matrix is then
    transposed to [P_pt, Q_pt]. The by-parts terms follow the ROLES: BT on
    the test axis P's ends, SW + SQ on the source axis Q's ends.
    """
    eps_t, _em, k_p, _km, _c2, _am = ctx.medium
    gz = float(ctx.ground_z)
    c1 = CF._c1_moment(ctx.omega, ctx.mu)
    k2sq = k_p * k_p

    dx = Q_above["nodes"][:, 0][:, None] - P_below["nodes"][:, 0][None, :]
    dy = Q_above["nodes"][:, 1][:, None] - P_below["nodes"][:, 1][None, :]
    rho = np.hypot(dx, dy)
    z = np.broadcast_to((Q_above["nodes"][:, 2] - gz)[:, None], rho.shape)
    zp = np.broadcast_to((P_below["nodes"][:, 2] - gz)[None, :], rho.shape)
    tb = CF._tables(ctx, eps_t, k_p, rho, z, zp, CF._CROSS_RTOL)
    U = tb["U"].T
    V = tb["V"].T
    W = w_sign * tb["W"].T
    ZZ = k2sq * V - tb[zz_key].T

    txP, tyP, tzP = P_below["t"].T
    txQ, tyQ, tzQ = Q_above["t"].T
    FP_w, FQ_w = P_below["F"] * P_below["w"], Q_above["F"] * Q_above["w"]
    FdP_w, FdQ_w = P_below["Fd"] * P_below["w"], Q_above["Fd"] * Q_above["w"]

    t_ba = c1 * (
        (FP_w * txP) @ U @ (FQ_w * txQ).T
        + (FP_w * tyP) @ U @ (FQ_w * tyQ).T
        + (FP_w * tzP) @ ZZ @ (FQ_w * tzQ).T
        + (FP_w * tzP) @ W @ FdQ_w.T
        + FdP_w @ W @ (FQ_w * tzQ).T
        - FdP_w @ V @ FdQ_w.T
    )

    # BT: the TEST axis's ends (below -> the z' slot, clamped at the plane).
    for pt, sign, fv in P_below["ends"]:
        rho_e = np.hypot(Q_above["nodes"][:, 0] - pt[0], Q_above["nodes"][:, 1] - pt[1])
        te = CF._tables(
            ctx,
            eps_t,
            k_p,
            rho_e,
            Q_above["nodes"][:, 2] - gz,
            np.full_like(rho_e, min(pt[2] - gz, 0.0)),
            CF._CROSS_RTOL,
        )
        t_ba += c1 * sign * np.outer(fv, FdQ_w @ te["V"])
    # SW + SQ: the SOURCE axis's ends (above -> the z slot).
    for pt, sign, fv in Q_above["ends"]:
        rho_e = np.hypot(P_below["nodes"][:, 0] - pt[0], P_below["nodes"][:, 1] - pt[1])
        te = CF._tables(
            ctx,
            eps_t,
            k_p,
            rho_e,
            np.full_like(rho_e, max(pt[2] - gz, 0.0)),
            P_below["nodes"][:, 2] - gz,
            CF._CROSS_RTOL,
        )
        t_ba += -c1 * sign * w_sign * np.outer((FP_w * tzP) @ te["W"], fv)
        t_ba += c1 * sign * np.outer(FdP_w @ te["V"], fv)
    if corner:
        t_ba += (
            CF._ends_and_corner(
                ctx, Q_above, P_below, eps_t, k_p, c1, gz, corner=True
            ).T
            - CF._ends_and_corner(
                ctx, Q_above, P_below, eps_t, k_p, c1, gz, corner=False
            ).T
        )
    return t_ba


def main():
    f = build(1)
    ctx, geom, Z = f["ctx"], f["geom"], f["Z"]
    zb = geom["seg_p0"][f["b_idx"], 2]
    za = geom["seg_p0"][f["a_idx"], 2]
    print(
        f"deck: below-wire z in [{zb.min():.3f}, {zb.max():.3f}], "
        f"above-wire z in [{za.min():.3f}, {za.max():.3f}], node at z=0"
    )
    print(f"KEYS = {_near_interface.KEYS}")

    n_basis = geom["n_basis_total"]
    P = CF.path_test_axis(n_basis, f["rs"]._path_test_rows(geom, f["bases_below"]))
    Q = CF.axis_data(ctx, f["a_idx"])
    ref = Z[np.ix_(f["bases_below"], f["bases_above"])]
    scale = np.abs(ref).max()
    print(
        f"\nreference: razor free-space Z[below rows, above cols], "
        f"shape {ref.shape}, max|ref| {scale:.6e}\n"
    )

    print(
        f"  {'zz kernel':>10} {'W sign':>7} {'max|got-ref|':>14} {'relative':>12} "
        f"{'median ratio':>26}"
    )
    best = None
    for zz_key in ("dzW", "dzpW"):
        for w_sign in (+1.0, -1.0):
            t_ba = reversed_block(ctx, P, Q, zz_key=zz_key, w_sign=w_sign)
            got = -t_ba[np.ix_(f["bases_below"], f["bases_above"])]
            d = np.abs(got - ref).max()
            rel = d / scale
            big = np.abs(ref) > 1e-3 * scale
            r = np.median(got[big] / ref[big])
            print(
                f"  {zz_key:>10} {w_sign:>+7.0f} {d:>14.6e} {rel:>12.4e} "
                f"{r.real:>+12.6f}{r.imag:>+12.6f}j"
            )
            if best is None or rel < best[0]:
                best = (rel, zz_key, w_sign)
    print(f"\n  winner: {best[1]}, W sign {best[2]:+.0f}, relative {best[0]:.4e}")


main()
