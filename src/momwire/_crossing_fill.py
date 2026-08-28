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

The corner term's sign is STRUCTURAL and ORIENTATION-CARRIED:
−σ_test·σ_src·c1·V(a) against the value-1 tents (σ = −1 at a wire's
start, +1 at its end), calibrated once on the soil-A adjudication deck
and never re-picked per medium (the high-σ ladder pinned its
medium-independence). With the fan widening the below axis carries N
node tents: the corner loop emits one σ-carried term per
(above-tent × below-tent) pair, and the self completion's corner emits
the below×below tent pairs at R = a — all at the ONE crossing node the
scope allows, which is what keeps the single V(a) evaluation honest.

Scope guards this module inherits from the derivation:

  * segments on both axes must be purely horizontal or vertical — the W
    by-parts move uses t̂⊥·∇⊥ = d/dl, exact only there;
  * one wire radius across the deck — the radius rule ρ_eff = √(ρ² + a²)
    is the corner's regularization and a per-pair radius has no pinned
    convention yet.

`bspline.BSplineSolver._compute_Z_operator_buried` is the only caller.
"""

from __future__ import annotations

import os

import numpy as np
from numpy.polynomial.legendre import leggauss

from . import _aca, _near_interface
from ._sommerfeld_transmitted import _c1_moment

_TILT_TOL = 1e-12
_CROSS_RTOL = 1e-9
_CORNER_RTOL = 1e-10
_GX8, _GW8 = leggauss(8)

# The #688 admissibility split: far (admissible) segment blocks are evaluated
# on COARSER axes and through the low-rank ACA pass; near / corner-adjacent
# blocks keep the dense graded axes and the designed direct evaluation
# unconditionally. The coarse knobs are the banked density-ladder combo
# (far Gauss 6->4, panels G8->G4, growth x2->x4: <= 3e-4 ohm movement on the
# adjudication decks against gate envelopes 100x wider) — safe HERE because
# the split never lets a near pair see them; a GLOBAL default drop would
# need the deeper-deck ladder the #688 comment calls for. Segments meeting
# at the crossing node have box distance 0 and are inadmissible by
# construction, so every corner-adjacent pair stays dense-direct.
_ADM_ETA = 1.0
_ACA_TOL = 1e-7
# leaf=3 measured as the knee (2026-08-27 ladder): leaf=4 leaves an extra
# dense ring (~7-10% more evaluations); leaf<=2 pushes coarse axes onto
# close non-touching pairs — at leaf=1 that costs three orders of block
# parity (1e-9 -> 1e-6 relative), and at leaf=2 it moved the g1
# adjudication anchor's fourth printed decimal (138.7670 vs the banked
# 138.7671). leaf=3 keeps every banked print to the digit.
_CLUSTER_LEAF_SEGS = 3
_FAR_Q = 4
_FAR_GROWTH = 4.0
# ACA only where sampling undercuts the full coarse product: rank-r
# sampling costs ~(rows+cols)·(m+n) designed points across the four
# kernels (measured ranks 6-8), so a block must have m·n well past
# that to win. Below the guard, direct coarse evaluation is cheaper
# AND memo-dedups against its mirror blocks.
_ACA_COST_GUARD = 48.0
_GX4, _GW4 = leggauss(4)
_CROSS_KEYS = ("U", "V", "W", "dzW")

# The whole-run dense-direct switch (a timing comparison, a bisect); parity
# tests drive both paths in-process by calling the two entries directly.
_FORCE_DENSE = bool(os.environ.get("MOMWIRE_CROSSING_FORCE_DENSE"))


def _graded_u(h, toward_end, a, growth=2.0, gx=_GX8, gw=_GW8):
    """Quadrature (u, w) on [0, h], log-graded (a-scale, `growth`-factored)
    toward u = h (`toward_end='hi'`) or u = 0 (`'lo'`); Gauss-`len(gx)` per
    panel. The grading is what lets the ln(a)-class end integrals actually
    converge instead of being truncated by segment-scale Gauss."""
    edges = [0.0]
    step = a
    while edges[-1] + step < h:
        edges.append(edges[-1] + step)
        step *= growth
    edges.append(h)
    e = np.array(edges)
    if toward_end == "hi":
        e = h - e[::-1]
    mid = 0.5 * (e[:-1] + e[1:])
    half = 0.5 * (e[1:] - e[:-1])
    u = (mid[:, None] + half[:, None] * gx[None, :]).ravel()
    w = (half[:, None] * gw[None, :]).ravel()
    return u, w


def axis_data(s, geom, seg_idx, coarse=False):
    """Everything one axis of the crossing blocks needs: quadrature nodes,
    per-node tangents and weights, per-basis value/derivative samples, and
    the signed wire-end table for the by-parts terms.

    Segments touching the interface get log-graded panels toward it;
    every other segment keeps the buried fill's own Gauss order. The ends
    table keeps only ends where some basis has nonzero value (value-1
    junction/contact tents); free ends carry no basis and drop out.

    `coarse=True` builds the far-block variant of the same axis (the #688
    density knobs); it exists only for admissible blocks and must never be
    fed to the ends/corner terms.
    """
    d = s.degree
    q = _FAR_Q if coarse else s._n_qp_buried_field()
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
            if coarse:
                u, w = _graded_u(
                    h, "lo" if touch_lo else "hi", a_wire, _FAR_GROWTH, _GX4, _GW4
                )
            else:
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
        segof=segof,
    )


def _tables(s, eps_t, k_p, rho, z, zp, rtol, memo=None):
    """Designed tables with the deck's wire radius folded in, z relative
    to the interface. `memo` extends the exact-triple dedup across calls
    (one fill = one memo; ε̃, k₂, rtol fixed for its lifetime)."""
    a_wire = float(s._radius_per_wire[0])
    return _near_interface.radius_tables(
        eps_t, k_p, rho, z, zp, a_wire, rtol=rtol, memo=memo
    )


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
    t_ab += _ends_and_corner(s, A, B, eps_t, k_p, c1, gz)
    return t_ab


def _ends_and_corner(s, A, B, eps_t, k_p, c1, gz, memo=None):
    """The by-parts end terms + the designed corner, on the DENSE axes —
    linear in axis size, so the admissibility split never touches them
    (and the corner must never see coarse axes or a low-rank pass; its
    V(a) rides `six_point` at `_CORNER_RTOL`, outside any memo)."""
    t_ab = np.zeros((A["n_basis"], B["n_basis"]), dtype=np.complex128)
    _txA, _tyA, tzA = A["t"].T
    FA_w = A["F"] * A["w"]
    FdA_w = A["Fd"] * A["w"]
    FdB_w = B["Fd"] * B["w"]

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
            memo=memo,
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
            memo=memo,
        )
        t_ab += -c1 * sign * np.outer((FA_w * tzA) @ te["W"], fv)
        t_ab += c1 * sign * np.outer(FdA_w @ te["V"], fv)

    # The designed corner: node tents against each other through V at
    # R = a exactly. The sign is STRUCTURAL and orientation-carried:
    # −σ_test·σ_src·c1·V(a), which is +c1·V(a) on the deck class the
    # adjudication calibrated it on (above arm STARTING at the node,
    # σ_a σ_b = −1) and flips with the wires' parametrization — an
    # orientation-blind + wrecks a monopole spelled top-down into the
    # node (measured: 10−1007j on the P3 rise deck, the −1000j
    # truncation-class signature). Never re-pick per MEDIUM. It is the
    # INTERFACE corner, so it applies only to end pairs that BOTH stand
    # in the plane — an end elsewhere (the P3 fan's below-hub junction)
    # carries its by-parts terms above but no corner.
    a_wire = float(s._radius_per_wire[0])
    v_corner = None
    for pt_a, sig_a, fv_a in A["ends"]:
        if abs(pt_a[2] - gz) > 1e-12:
            continue
        for pt_b, sig_b, fv_b in B["ends"]:
            if abs(pt_b[2] - gz) > 1e-12:
                continue
            # One V(a) serves every pair only because every in-plane
            # value-1 end stands at the ONE crossing node the scope
            # allows — assert that, don't assume it.
            assert np.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1]) < 1e-9
            if v_corner is None:
                v_corner = complex(
                    _near_interface.six_point(
                        eps_t, k_p, a_wire, 0.0, 0.0, rtol=_CORNER_RTOL
                    )[1]
                )
            t_ab += (-sig_a * sig_b * c1 * v_corner) * np.outer(fv_a, fv_b)
    return t_ab


def _row_weights(ax, ii):
    """The four per-basis row-weight matrices of one axis, restricted to
    the point subset `ii`: (F·w·t̂x, F·w·t̂y, F·w·t̂z, F′·w)."""
    tx, ty, tz = ax["t"][ii].T
    Fw = ax["F"][:, ii] * ax["w"][ii]
    Fdw = ax["Fd"][:, ii] * ax["w"][ii]
    return Fw * tx, Fw * ty, Fw * tz, Fdw


def _sandwich_dense(A, B, iA, iB, K, k2sq):
    """The five-term M+SW+SQ (main) sandwich over dense kernel matrices
    restricted to (iA, iB) — the same term order as the reference fill."""
    P1, P2, P3, P4 = _row_weights(A, iA)
    Q1, Q2, Q3, Q4 = _row_weights(B, iB)
    return (
        P1 @ K["U"] @ Q1.T
        + P2 @ K["U"] @ Q2.T
        + P3 @ (k2sq * K["V"] - K["dzW"]) @ Q3.T
        + P3 @ K["W"] @ Q4.T
        + P4 @ K["W"] @ Q3.T
        - P4 @ K["V"] @ Q4.T
    )


def _axis_segment_tree(geom, seg_idx, leaf):
    """Cluster tree over one axis's SEGMENTS (boxes = segment endpoints).
    Cluster indices are positions into `seg_idx`; returns (tree, seg_idx
    as an array) so callers can map back to global segment ids."""
    idx = np.asarray(seg_idx, dtype=np.int64)
    lo = np.minimum(geom["seg_l"][idx], geom["seg_r"][idx])
    hi = np.maximum(geom["seg_l"][idx], geom["seg_r"][idx])
    tree = _aca.build_cluster_tree(np.arange(idx.size), lo, hi, leaf)
    return tree, idx


def cross_complete_block_split(s, geom, a_idx, b_idx, A, B):
    """`cross_complete_block` through the #688 admissibility split.

    The (above segments × below segments) product is partitioned by the
    standard box rule (`_aca.build_block_tree`, η = 1): inadmissible
    blocks — every pair meeting the crossing node among them, since
    touching boxes have distance 0 — are evaluated dense-direct on the
    dense graded axes, batched into ONE designed-tables call so the
    exact-triple memo and the C++ batch keep their full scope.
    Admissible far blocks are evaluated on the coarse axes and through
    `_aca.aca_partial`, one factorization per kernel {U, V, W, ∂zW},
    sampling designed rows/columns through a shared cache (one designed
    evaluation serves all four kernels at that row/column).

    The by-parts end terms and the corner (−σσ′·c1·V(a)) ride the dense
    axes direct, always — they are linear in axis size and the corner
    routes through neither coarse axes nor the low-rank pass."""
    if _FORCE_DENSE:
        return cross_complete_block(s, geom, A, B)

    eps_t, _eps_m, k_p, _k_m, _c2, _a_m = s._buried_medium()
    gz = float(s.ground_z)
    c1 = _c1_moment(s.omega, s.mu)
    k2sq = k_p * k_p

    tree_a, seg_a = _axis_segment_tree(geom, a_idx, _CLUSTER_LEAF_SEGS)
    tree_b, seg_b = _axis_segment_tree(geom, b_idx, _CLUSTER_LEAF_SEGS)
    far, near = _aca.build_block_tree(tree_a, tree_b, _ADM_ETA)

    t_main = np.zeros((A["n_basis"], B["n_basis"]), dtype=np.complex128)
    memo = {}  # one fill = one memo (eps_t, k_p, _CROSS_RTOL fixed here)

    if far:
        Ac = axis_data(s, geom, a_idx, coarse=True)
        Bc = axis_data(s, geom, b_idx, coarse=True)

    # ---- direct blocks: near pairs on the dense axes + small far blocks
    # on the coarse axes (sampling a small block costs more than its full
    # coarse product), ALL in one batched designed evaluation so the
    # exact-triple memo dedups across every block — a symmetric screen's
    # mirrored blocks are the same triples.
    direct, far_aca = [], []
    for cs, ct in near:
        iA = np.flatnonzero(np.isin(A["segof"], seg_a[cs.indices]))
        iB = np.flatnonzero(np.isin(B["segof"], seg_b[ct.indices]))
        if iA.size and iB.size:
            direct.append((A, B, iA, iB))
    for cs, ct in far:
        iA = np.flatnonzero(np.isin(Ac["segof"], seg_a[cs.indices]))
        iB = np.flatnonzero(np.isin(Bc["segof"], seg_b[ct.indices]))
        if iA.size == 0 or iB.size == 0:
            continue
        if iA.size * iB.size > _ACA_COST_GUARD * (iA.size + iB.size):
            far_aca.append((iA, iB))
        else:
            direct.append((Ac, Bc, iA, iB))

    if direct:
        specs, rho_cat, z_cat, zp_cat = [], [], [], []
        for AX, BX, iA, iB in direct:
            pa, pb = AX["nodes"][iA], BX["nodes"][iB]
            rho = np.hypot(
                pa[:, 0][:, None] - pb[:, 0][None, :],
                pa[:, 1][:, None] - pb[:, 1][None, :],
            )
            specs.append((AX, BX, iA, iB, rho.shape))
            rho_cat.append(rho.ravel())
            z_cat.append(np.repeat(pa[:, 2] - gz, iB.size))
            zp_cat.append(np.tile(pb[:, 2] - gz, iA.size))
        tab = _tables(
            s,
            eps_t,
            k_p,
            np.concatenate(rho_cat),
            np.concatenate(z_cat),
            np.concatenate(zp_cat),
            _CROSS_RTOL,
            memo=memo,
        )
        off = 0
        for AX, BX, iA, iB, shp in specs:
            nel = shp[0] * shp[1]
            K = {kk: tab[kk][off : off + nel].reshape(shp) for kk in _CROSS_KEYS}
            off += nel
            t_main += _sandwich_dense(AX, BX, iA, iB, K, k2sq)

    # ---- large far blocks: coarse axes, low-rank ACA per kernel. The
    # row/column samples ride the SAME memo — identical matrices in
    # mirrored blocks pick identical pivots, so their samples dedup.
    for iA, iB in far_aca:
        pa, pb = Ac["nodes"][iA], Bc["nodes"][iB]
        zA, zB = pa[:, 2] - gz, pb[:, 2] - gz
        rho = np.hypot(
            pa[:, 0][:, None] - pb[:, 0][None, :],
            pa[:, 1][:, None] - pb[:, 1][None, :],
        )
        m, n = iA.size, iB.size
        rows, cols = {}, {}

        def _row6(i, rho=rho, zA=zA, zB=zB, rows=rows, n=n):
            if i not in rows:
                te = _tables(
                    s,
                    eps_t,
                    k_p,
                    rho[i],
                    np.full(n, zA[i]),
                    zB,
                    _CROSS_RTOL,
                    memo=memo,
                )
                rows[i] = np.stack([te[kk] for kk in _CROSS_KEYS])
            return rows[i]

        def _col6(j, rho=rho, zA=zA, zB=zB, cols=cols, m=m):
            if j not in cols:
                te = _tables(
                    s,
                    eps_t,
                    k_p,
                    rho[:, j],
                    zA,
                    np.full(m, zB[j]),
                    _CROSS_RTOL,
                    memo=memo,
                )
                cols[j] = np.stack([te[kk] for kk in _CROSS_KEYS])
            return cols[j]

        Kf = {}
        for ki, kk in enumerate(_CROSS_KEYS):
            Kf[kk] = _aca.aca_partial(
                lambda i, ki=ki: _row6(i)[ki],
                lambda j, ki=ki: _col6(j)[ki],
                m,
                n,
                tol=_ACA_TOL,
            )
        P1, P2, P3, P4 = _row_weights(Ac, iA)
        Q1, Q2, Q3, Q4 = _row_weights(Bc, iB)

        def _lr(P, kk, Q, Kf=Kf):
            Uf, Vf = Kf[kk]
            return (P @ Uf) @ (Vf @ Q.T)

        t_main += (
            _lr(P1, "U", Q1)
            + _lr(P2, "U", Q2)
            + k2sq * _lr(P3, "V", Q3)
            - _lr(P3, "dzW", Q3)
            + _lr(P3, "W", Q4)
            + _lr(P4, "W", Q3)
            - _lr(P4, "V", Q4)
        )

    t_ab = c1 * t_main
    t_ab += _ends_and_corner(s, A, B, eps_t, k_p, c1, gz, memo=memo)
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
