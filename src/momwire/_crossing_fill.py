"""The COMPLETE crossing fill — the interface-crossing junction's blocks
(momwire#524 phase 2).

A deck with a crossing junction (a wholly-below wire whose end stands in
the ground plane, junction-joined there to an above wire) is filled with
the one quadrature-convergent node treatment the phase-2 probes measured:
the complete field-form-equivalent mixed-potential spelling, all by-parts
ends and corners, on all four kernel families, one convention —

  cross:  t_ab = M + SW + SQ + BT + CORNER   (designed kernels, graded
          axes, thin-wire radius folded into every pair distance)
  self:   + β_dir·(bnd + corner)(G_dir) − β_img·(bnd + corner)(G_img)
          per medium family, on the same graded axes

Every "drop" spelling is truncation-regularized — at resolved quadrature
the retained ∬f′f′V's ln(a)-class content diverges with its balancing
end/corner terms deleted — so completeness here is a convergence
requirement, not a preference. Continuity of current through the node and
the AGARD slope condition then EMERGE from the fill's own physics: the
node needs no constraint row and no merged dof (split ≡ merged ≡
V-constrained, measured to the digit on the adjudication decks).

The corner term's sign is STRUCTURAL: +c1·V(a) against the value-1 tents,
calibrated once on the soil-A adjudication deck and never re-picked per
medium (the high-σ ladder pinned its medium-independence).

Scope guards this module inherits from the derivation:

  * segments on both axes must be purely horizontal or vertical — the W
    by-parts move uses t̂⊥·∇⊥ = d/dl, exact only there;
  * one wire radius across the deck — the radius rule ρ_eff = √(ρ² + a²)
    is the corner's regularization and a per-pair radius has no pinned
    convention yet.

`bspline.BSplineSolver._compute_Z_operator_buried` is the only caller.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from . import _near_interface
from ._sommerfeld_transmitted import _c1_moment

_TILT_TOL = 1e-12
_CROSS_RTOL = 1e-9
_CORNER_RTOL = 1e-10
_GX8, _GW8 = leggauss(8)


def _graded_u(h, toward_end, a):
    """Quadrature (u, w) on [0, h], log-graded (a-scale, doubling) toward
    u = h (`toward_end='hi'`) or u = 0 (`'lo'`); Gauss-8 per panel. The
    grading is what lets the ln(a)-class end integrals actually converge
    instead of being truncated by segment-scale Gauss."""
    edges = [0.0]
    step = a
    while edges[-1] + step < h:
        edges.append(edges[-1] + step)
        step *= 2.0
    edges.append(h)
    e = np.array(edges)
    if toward_end == "hi":
        e = h - e[::-1]
    mid = 0.5 * (e[:-1] + e[1:])
    half = 0.5 * (e[1:] - e[:-1])
    u = (mid[:, None] + half[:, None] * _GX8[None, :]).ravel()
    w = (half[:, None] * _GW8[None, :]).ravel()
    return u, w


def axis_data(s, geom, seg_idx):
    """Everything one axis of the crossing blocks needs: quadrature nodes,
    per-node tangents and weights, per-basis value/derivative samples, and
    the signed wire-end table for the by-parts terms.

    Segments touching the interface get log-graded panels toward it;
    every other segment keeps the buried fill's own Gauss order. The ends
    table keeps only ends where some basis has nonzero value (value-1
    junction/contact tents); free ends carry no basis and drop out.
    """
    d = s.degree
    q = s._n_qp_buried_field()
    xg, wg = leggauss(q)
    tq = 0.5 * (xg + 1.0)
    gz = float(s.ground_z)
    a_wire = float(s._radius_per_wire[0])
    tol = 1e-12

    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    n_basis = polys.shape[0]

    nodes_l, t_l, w_l, u_l, segpos = [], [], [], [], []
    for g in seg_idx:
        sl, sr = geom["seg_l"][g], geom["seg_r"][g]
        h = geom["h_per_seg"][g]
        tang = geom["tangents"][g]
        if abs(tang[2]) > _TILT_TOL and np.hypot(tang[0], tang[1]) > _TILT_TOL:
            raise NotImplementedError(
                "crossing fill: tilted segments need the tilt tables — the "
                "W by-parts move is exact only for purely horizontal or "
                "vertical segments"
            )
        touch_lo = abs(sl[2] - gz) < tol
        touch_hi = abs(sr[2] - gz) < tol
        if touch_lo or touch_hi:
            u, w = _graded_u(h, "lo" if touch_lo else "hi", a_wire)
        else:
            u = h * tq
            w = 0.5 * h * wg
        nodes_l.append(sl[None, :] + (u / h)[:, None] * (sr - sl)[None, :])
        t_l.append(np.repeat(tang[None, :], len(u), axis=0))
        w_l.append(w)
        u_l.append(u)
        segpos.append(np.full(len(u), g, dtype=np.int64))
    nodes = np.concatenate(nodes_l)
    t_node = np.concatenate(t_l)
    w_node = np.concatenate(w_l)
    u_phys = np.concatenate(u_l)
    segof = np.concatenate(segpos)

    F = np.zeros((n_basis, len(u_phys)))
    Fd = np.zeros((n_basis, len(u_phys)))
    for m in range(n_basis):
        for a_ in range(supp_seg.shape[1]):
            # supp_seg rows are zero-padded; a slot is live only if its
            # polynomial is nonzero (the padding trap).
            if not np.any(polys[m, a_] != 0.0):
                continue
            sel = np.nonzero(segof == supp_seg[m, a_])[0]
            if sel.size == 0:
                continue
            u = u_phys[sel]
            for p in range(d + 1):
                c = polys[m, a_, p]
                if c == 0.0:
                    continue
                F[m, sel] += c * u**p
                if p >= 1:
                    Fd[m, sel] += p * c * u ** (p - 1)

    # Signed wire-end table: (point, sign, per-basis value there). σ = −1
    # at a wire's first segment's u = 0 end, +1 at its last segment's
    # u = h end — the by-parts orientation the derivation pinned.
    seg_off = geom["seg_offsets"]
    ends = []
    on_axis = set(int(g) for g in seg_idx)
    for w in range(len(seg_off) - 1):
        first, last = seg_off[w], seg_off[w + 1] - 1
        if first not in on_axis:
            continue
        for gseg, sign, u_end in ((first, -1.0, 0.0), (last, +1.0, None)):
            hh = geom["h_per_seg"][gseg]
            u = hh if u_end is None else 0.0
            pt = geom["seg_l"][gseg] + (u / hh) * (
                geom["seg_r"][gseg] - geom["seg_l"][gseg]
            )
            fv = np.zeros(n_basis)
            for m in range(n_basis):
                for a_ in range(supp_seg.shape[1]):
                    if supp_seg[m, a_] == gseg and np.any(polys[m, a_] != 0.0):
                        fv[m] += sum(polys[m, a_, p] * u**p for p in range(d + 1))
            if np.any(fv != 0.0):
                ends.append((pt, sign, fv))
    return dict(
        nodes=nodes,
        t=t_node,
        w=w_node,
        F=F,
        Fd=Fd,
        ends=ends,
        n_basis=n_basis,
    )


def _tables(s, eps_t, k_p, rho, z, zp, rtol):
    """Designed tables with the deck's wire radius folded in, z relative
    to the interface."""
    a_wire = float(s._radius_per_wire[0])
    return _near_interface.radius_tables(eps_t, k_p, rho, z, zp, a_wire, rtol=rtol)


def cross_complete_block(s, geom, A, B):
    """t_ab = M + SW + SQ + BT + CORNER over (above axis A × below axis B),
    on designed kernels. Returns the full (n_basis, n_basis) block in the
    subtracting field-block convention (`Z -= t_ab`; t_ba is the transpose
    by reciprocity, measured, not assumed, by the phase-2 probes)."""
    eps_t, _eps_m, k_p, _k_m, _c2, _a_m = s._buried_medium()
    gz = float(s.ground_z)
    c1 = _c1_moment(s.omega, s.mu)
    k2sq = k_p * k_p

    dx = A["nodes"][:, 0][:, None] - B["nodes"][:, 0][None, :]
    dy = A["nodes"][:, 1][:, None] - B["nodes"][:, 1][None, :]
    rho = np.hypot(dx, dy)
    z = np.broadcast_to((A["nodes"][:, 2] - gz)[:, None], rho.shape)
    zp = np.broadcast_to((B["nodes"][:, 2] - gz)[None, :], rho.shape)
    tables = _tables(s, eps_t, k_p, rho, z, zp, _CROSS_RTOL)
    U, V, W, dzW = tables["U"], tables["V"], tables["W"], tables["dzW"]

    wA, wB = A["w"], B["w"]
    txA, tyA, tzA = A["t"].T
    txB, tyB, tzB = B["t"].T
    FA_w, FB_w = A["F"] * wA, B["F"] * wB
    FdA_w, FdB_w = A["Fd"] * wA, B["Fd"] * wB

    s_u = (FA_w * txA) @ U @ (FB_w * txB).T + (FA_w * tyA) @ U @ (FB_w * tyB).T
    s_zz = (FA_w * tzA) @ (k2sq * V - dzW) @ (FB_w * tzB).T
    s_w1 = (FA_w * tzA) @ W @ FdB_w.T
    s_w2 = FdA_w @ W @ (FB_w * tzB).T
    s_phi = -FdA_w @ V @ FdB_w.T
    t_ab = c1 * (s_u + s_zz + s_w1 + s_w2 + s_phi)

    # The by-parts boundary terms — test-side Φ (BT), source-side W and Φ
    # (SW, SQ) — each an end against the other axis's line, radius folded.
    for pt, sign, fv in A["ends"]:
        rho_e = np.hypot(pt[0] - B["nodes"][:, 0], pt[1] - B["nodes"][:, 1])
        te = _tables(
            s,
            eps_t,
            k_p,
            rho_e,
            np.full_like(rho_e, max(pt[2] - gz, 0.0)),
            B["nodes"][:, 2] - gz,
            _CROSS_RTOL,
        )
        t_ab += c1 * sign * np.outer(fv, FdB_w @ te["V"])
    for pt, sign, fv in B["ends"]:
        rho_e = np.hypot(A["nodes"][:, 0] - pt[0], A["nodes"][:, 1] - pt[1])
        te = _tables(
            s,
            eps_t,
            k_p,
            rho_e,
            A["nodes"][:, 2] - gz,
            np.full_like(rho_e, pt[2] - gz),
            _CROSS_RTOL,
        )
        t_ab += -c1 * sign * np.outer((FA_w * tzA) @ te["W"], fv)
        t_ab += c1 * sign * np.outer(FdA_w @ te["V"], fv)

    # The designed corner: node tents against each other through V at
    # R = a exactly. Sign structural (+), never re-picked per medium.
    a_wire = float(s._radius_per_wire[0])
    v_corner = complex(
        _near_interface.six_point(eps_t, k_p, a_wire, 0.0, 0.0, rtol=_CORNER_RTOL)[1]
    )
    for pt_a, _sig_a, fv_a in A["ends"]:
        for pt_b, _sig_b, fv_b in B["ends"]:
            t_ab += (c1 * v_corner) * np.outer(fv_a, fv_b)
    return t_ab


def _g_of_r(k, R):
    return np.exp(-1j * k * R) / R


def _bnd_and_corner(ax, k, a_wire, gz, mirror):
    """The same-medium by-parts boundary shape on one axis (β = 1):
    −test-end rows, −source-end columns, +corner — the derivation's
    −,−,+ sign structure. Closed-form kernel G = e^{−jkR}/R at
    R = √(‖Δr‖² + a²), source positions mirrored through the interface
    for the image family. Returns (bnd, corner)."""
    pts = ax["nodes"]
    Fdw = ax["Fd"] * ax["w"]
    n = ax["n_basis"]
    bnd = np.zeros((n, n), dtype=np.complex128)
    corner = np.zeros((n, n), dtype=np.complex128)

    def _mir(p):
        p = np.array(p, dtype=float, copy=True)
        if mirror:
            p[..., 2] = 2.0 * gz - p[..., 2]
        return p

    src = _mir(pts)
    for ptE, sig, fvT in ax["ends"]:  # test ends (observation, unmirrored)
        dr = src - np.asarray(ptE)[None, :]
        R = np.sqrt(a_wire * a_wire + np.einsum("ij,ij->i", dr, dr))
        bnd += -sig * np.outer(fvT, Fdw @ _g_of_r(k, R))
    for ptEp, sigp, fvS in ax["ends"]:  # source ends (mirrored)
        pe = _mir(np.asarray(ptEp))
        dr = pts - pe[None, :]
        R = np.sqrt(a_wire * a_wire + np.einsum("ij,ij->i", dr, dr))
        bnd += -sigp * np.outer(Fdw @ _g_of_r(k, R), fvS)
    for ptE, sig, fvT in ax["ends"]:
        for ptEp, sigp, fvS in ax["ends"]:
            pe = _mir(np.asarray(ptEp))
            dr = np.asarray(ptE) - pe
            R = np.sqrt(a_wire * a_wire + float(dr @ dr))
            corner += sig * sigp * _g_of_r(k, R) * np.outer(fvT, fvS)
    return bnd, corner


def self_completions(s, geom, ax_b, ax_a):
    """The self families' missing bnd + corner content, both media, on
    graded axes. Returned as the ADDITIVE Z correction (the fill's
    `Z -= image` convention already folded in: β_dir·(bnd+cor)(G_dir)
    − β_img·(bnd+cor)(G_img) per family)."""
    _eps_t, eps_m, k_p, k_m, c2, a_m = s._buried_medium()
    gz = float(s.ground_z)
    a_wire = float(s._radius_per_wire[0])
    omega, eps0 = s.omega, s.eps
    total = np.zeros((ax_b["n_basis"],) * 2, dtype=np.complex128)
    for ax, k, wgt, eps in ((ax_b, k_m, a_m, eps_m), (ax_a, k_p, c2, eps0)):
        bnd_dir, cor_dir = _bnd_and_corner(ax, k, a_wire, gz, mirror=False)
        bnd_img, cor_img = _bnd_and_corner(ax, k, a_wire, gz, mirror=True)
        beta_dir = 1.0 / (1j * omega * eps * 4 * np.pi)
        beta_img = wgt / (1j * omega * eps * 4 * np.pi)
        total += beta_dir * (bnd_dir + cor_dir) - beta_img * (bnd_img + cor_img)
    return total
