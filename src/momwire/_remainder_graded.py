"""Graded panels for the Sommerfeld remainder's grazing segment pairs.

momwire#1201. The remainder field over a segment pair has a near-image spike
of width ~R_min at a point that is KNOWN in advance: the foot of the
observer's image on the source segment. momwire#631 resolved it with one
Gauss rule of order `ceil(c · len / R_min)` over the whole segment, capped at
192, and #1189 applied that per pair. A single rule converges only
algebraically on a feature of relative width R_min/len, so at the cap a
48-radial surface screen was still 1.1e-2 from its converged impedance.

This module integrates each listed pair on panels that are laid out from the
geometry instead:

* **Inner (source) rule, per observer node.** The source segment is split at
  the foot of that node's image and graded geometrically toward it, the
  smallest panel sized to the node's distance from the source line.
* **Outer (observer) rule, per pair.** The inner integral is a smooth function
  of the observer position except where the foot runs off a source endpoint,
  where the observer's image passes closest to the source line, and where a
  source ENDPOINT crosses one of the interpolant's seams (below). The observer
  segment is split and graded toward each of those points.

Why not subtract the near-image singular part in closed form, as #631 does for
the exact image? Because the remainder is not a closed-form kernel: it is
`e^{-jkR₁}/R₁` times four surfaces TABULATED in (R₁, θ), and across the spike
θ = atan2(z + z', ρ) sweeps from 90° to 0°. The singular part's coefficient is
that angular table, so there is nothing closed-form to subtract; the table
itself is what has to be integrated.

**The interpolant's seams are breakpoints too.** `SommerfeldGrid` is a 4x4
Lagrange interpolant on per-region lattices, so the integrand is not analytic
across (a) the θ = `th_split` band boundary, where the two bands carry
different lattices and the VALUE jumps (2e-4 relative at R₁ = 2h on #631's
grazing wire, 2e-3 at 1 m); (b) the R₁ region breaks; (c) every R₁ lattice
line of the region a point is in, where the stencil shifts and the slope
jumps. A Gauss rule straddling (a) converges like 1/n whatever the grading,
which is also why the brute-force ladder in #1201 creeps rather than
converges. Every such crossing on a segment is found in closed form (each is
a quadratic in the segment parameter) and becomes a panel boundary. The θ
lattice lines inside a band are left alone: splitting at them too moved no
pair moment of #631's grazing wire by more than the q = 4000 Gauss reference
resolves (~1e-9).

With those boundaries the integrand is analytic on every panel, and a fixed
`N_PANEL`-point rule on panels shrinking by `SIGMA` converges exponentially.

The panels are laid out here in numpy; the source-side sum over them runs in
C++ (`remainder_graded_inner`, which generates each panel's nodes itself) when
the accelerator is loaded, and over explicit nodes with
`_sommerfeld.remainder_field_proj_owned` otherwise.

The kernel `t_m · F(r_m, r_n) · t_n` is reciprocal (the projected table is
symmetric to 3e-14 under swapping observer and source), so only i <= j is
integrated and (j, i) is its transpose, which keeps the filled Q symmetric as
the same-order fill always was.
"""

from __future__ import annotations

import math

import numpy as np
from . import _sommerfeld
from ._quadrature import leggauss

# Geometric grading ratio and points per panel. Measured on #631's grazing
# wire (h/lambda = 1.09e-4) against brute force at q = 4000: the self pair's
# moments land within the reference's own error (~7e-9) at (0.25, 8), and the
# whole-deck impedance at (0.25, 8), (0.2, 10) and (0.3, 6) agrees to 2e-9.
SIGMA = 0.25
N_PANEL = 8

# Smallest panel, as a fraction of the segment. Only a genuine CONTACT (a wire
# ending in the plane, where the image distance really reaches zero) ever
# grades this far; 1e-6 bounds such a pair at ~10 panels per side while
# leaving the innermost panel's share of an integrable 1/R corner at the 1e-6
# level. Every non-contact spike in the suite is wider than this.
SCALE_FLOOR = 1e-6

# Pairs per chunk, and observer nodes per inner block. They bound the working
# set (an observer node carries ~50-150 source nodes, so a block is ~2e5
# kernel evaluations and a few tens of MiB), never the answer: every node's
# rule and sum are its own, so the partition is invisible in the bits.
PAIR_CHUNK = 256
OUTER_BLOCK = 2048


def _lattice(grid):
    """The interpolant's seams, as arrays `_crossings` can vectorise over.

    Returns `(th_split, r_breaks, regions)`: `r_breaks` the R₁ region breaks
    below `r1_max`, and `regions` one row per region
    `(r0, dr, n_r, r_lo, r_hi, upper_band)` where `[r_lo, r_hi]` is the R₁
    interval the region SERVES (its lattice may extend past it) and
    `upper_band` is True for the θ > th_split band.
    """
    th_split = math.radians(_sommerfeld._SOMM_TH_SPLIT_DEG)
    r_break, r_near, r1_max = (
        float(grid.r_break),
        float(grid.r_near),
        float(grid.r1_max),
    )
    bounds = [(0.0, r_break), (r_break, r_near), (r_near, r1_max)]
    rows = []
    for idx, reg in enumerate(grid._regions):
        lo, hi = bounds[idx // 2]
        rows.append(
            (float(reg["r0"]), float(reg["dr"]), int(reg["n_r"]), lo, hi, idx % 2 == 1)
        )
    breaks = [r for r in (r_break, r_near) if 0.0 < r < r1_max]
    return th_split, np.array(breaks, dtype=float), rows


def _roots01(a, b, c):
    """Real roots in (0, 1) of a·u² + b·u + c, elementwise.

    Returns `(k, u)`: the input index each root belongs to and the root. The
    stable pairing `q = -(b + sign(b)·√disc)/2`, roots q/a and c/q, avoids the
    cancellation the textbook form suffers when one root is near zero; a
    vanishing `a` is the linear case.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    idx = np.arange(a.size)
    scale = np.abs(a) + np.abs(b) + np.abs(c)
    lin = np.abs(a) <= 1e-14 * scale
    ks, us = [], []
    # Linear.
    ok = lin & (np.abs(b) > 0.0)
    if ok.any():
        ks.append(idx[ok])
        us.append(-c[ok] / b[ok])
    # Quadratic.
    quad = ~lin
    disc = b * b - 4.0 * a * c
    ok = quad & (disc >= 0.0)
    if ok.any():
        sq = np.sqrt(disc[ok])
        bb = b[ok]
        q = -0.5 * (bb + np.where(bb >= 0.0, sq, -sq))
        with np.errstate(divide="ignore", invalid="ignore"):
            r1 = q / a[ok]
            r2 = np.where(q != 0.0, c[ok] / q, np.nan)
        ks += [idx[ok], idx[ok]]
        us += [r1, r2]
    if not ks:
        return np.empty(0, np.int64), np.empty(0)
    k = np.concatenate(ks)
    u = np.concatenate(us)
    keep = np.isfinite(u) & (u > 0.0) & (u < 1.0)
    return k[keep], u[keep]


def _crossings(P, C, D, gz, lattice):
    """Where the pair (P, C + u·D), u in (0, 1), crosses an interpolant seam.

    `P`, `C`, `D` are `(T, 3)`; the geometry the grid is queried with is
    symmetric in its two points (ρ is their horizontal distance, z + z' their
    summed heights), so `P` may be an observer node against a source line or
    a source endpoint against an observer line. Returns `(task, u)`.
    """
    th_split, breaks, regions = lattice
    h0 = (P - C)[:, :2]
    hd = -D[:, :2]
    c0 = (P[:, 2] - gz) + (C[:, 2] - gz)
    c1 = D[:, 2]
    hh0 = np.einsum("ij,ij->i", h0, h0)
    hdd = np.einsum("ij,ij->i", hd, hd)
    h0d = np.einsum("ij,ij->i", h0, hd)
    T = P.shape[0]
    tasks, roots = [], []

    # (a) the band boundary θ = th_split: (z+z')² = tan²θ · ρ².
    t2 = math.tan(th_split) ** 2
    k, u = _roots01(
        c1 * c1 - t2 * hdd, 2.0 * c0 * c1 - 2.0 * t2 * h0d, c0 * c0 - t2 * hh0
    )
    tasks.append(k)
    roots.append(u)

    # R₁(u)² = A u² + B u + C0 along the line; its range over [0, 1].
    A = hdd + c1 * c1
    Bq = 2.0 * h0d + 2.0 * c0 * c1
    C0 = hh0 + c0 * c0
    with np.errstate(divide="ignore", invalid="ignore"):
        ustar = np.clip(np.where(A > 0.0, -Bq / (2.0 * A), 0.0), 0.0, 1.0)
    r2_at = lambda uu: A * uu * uu + Bq * uu + C0  # noqa: E731
    r_lo = np.sqrt(np.maximum(r2_at(ustar), 0.0))
    r_hi = np.sqrt(np.maximum(np.maximum(r2_at(0.0), r2_at(1.0)), 0.0))

    # (b) the R₁ region breaks.
    for r in breaks:
        k, u = _roots01(A, Bq, C0 - r * r)
        tasks.append(k)
        roots.append(u)

    # (c) the lattice lines of the region a crossing lands in. Candidate lines
    # per task are the ones inside [r_lo, r_hi] and inside the region's served
    # interval; a root counts only if the point is in that region's band.
    for r0, dr, n_r, lo, hi, upper in regions:
        a_lo = np.maximum(r_lo, lo)
        a_hi = np.minimum(r_hi, hi)
        k0 = np.maximum(np.ceil((a_lo - r0) / dr), 1.0)
        k1 = np.minimum(np.floor((a_hi - r0) / dr), float(n_r - 1))
        n = np.maximum(k1 - k0 + 1.0, 0.0).astype(np.int64)
        tot = int(n.sum())
        if tot == 0:
            continue
        own = np.repeat(np.arange(T), n)
        start = np.repeat(np.cumsum(n) - n, n)
        kk = k0[own] + (np.arange(tot) - start)
        rr = r0 + kk * dr
        k, u = _roots01(A[own], Bq[own], C0[own] - rr * rr)
        own = own[k]
        rho = np.hypot(h0[own, 0] + u * hd[own, 0], h0[own, 1] + u * hd[own, 1])
        th = np.arctan2(c0[own] + u * c1[own], rho)
        in_band = (th > th_split) if upper else (th <= th_split)
        tasks.append(own[in_band])
        roots.append(u[in_band])

    return np.concatenate(tasks), np.concatenate(roots)


def _panels(task, bp, scale, sigma):
    """Graded panels on [0, 1] per task, as `(owner, x0, x1)` sorted by owner.

    `task`, `bp`, `scale` list every breakpoint of every task, INCLUDING 0 and
    1. Each gap between consecutive breakpoints is one panel when both its
    ends' scales exceed it; otherwise it is halved and each half is graded
    geometrically (ratio `sigma`) toward its end, down to a panel no longer
    than that end's scale.
    """
    order = np.lexsort((bp, task))
    task, bp, scale = task[order], bp[order], scale[order]
    # A breakpoint listed twice (a seam crossing at the foot, say) takes the
    # tighter of its scales on both sides.
    first = np.r_[True, (task[1:] != task[:-1]) | (bp[1:] != bp[:-1])]
    starts = np.flatnonzero(first)
    tight = np.minimum.reduceat(scale, starts)
    task, bp, scale = task[starts], bp[starts], tight
    same = task[1:] == task[:-1]
    a, b = bp[:-1][same], bp[1:][same]
    da, db = scale[:-1][same], scale[1:][same]
    t = task[:-1][same]
    L = b - a
    live = L > 0.0
    a, b, da, db, t, L = a[live], b[live], da[live], db[live], t[live], L[live]
    single = (da >= L) & (db >= L)
    split = ~single
    n1, n2 = int(single.sum()), int(split.sum())
    anchor = np.concatenate([a[single], a[split], b[split]])
    direc = np.concatenate([np.ones(n1 + n2), -np.ones(n2)])
    length = np.concatenate([L[single], 0.5 * L[split], 0.5 * L[split]])
    delta = np.concatenate([np.full(n1, np.inf), da[split], db[split]])
    owner = np.concatenate([t[single], t[split], t[split]])

    with np.errstate(divide="ignore"):
        K = np.where(
            delta >= length,
            0,
            np.ceil(np.log(length / delta) / math.log(1.0 / sigma)),
        ).astype(np.int64)
    npan = K + 1
    rep = np.repeat(np.arange(anchor.size), npan)
    k = np.arange(rep.size) - np.repeat(np.cumsum(npan) - npan, npan)
    far = length[rep] * sigma ** k.astype(float)
    near = np.where(k == K[rep], 0.0, far * sigma)
    lo = np.where(direc[rep] > 0.0, near, -far)
    hi = np.where(direc[rep] > 0.0, far, -near)
    x0 = anchor[rep] + lo
    x1 = anchor[rep] + hi
    own = owner[rep]
    srt = np.argsort(own, kind="stable")
    return own[srt], x0[srt], x1[srt]


def _nodes(owner, x0, x1, xg, wg):
    """Gauss nodes of the panels `(owner, x0, x1)`: `(owner, x, w)`."""
    half = 0.5 * (x1 - x0)
    x = x0[:, None] + half[:, None] * (xg[None, :] + 1.0)
    w = half[:, None] * wg[None, :]
    return np.repeat(owner, xg.size), x.ravel(), w.ravel()


def _mirror(p, gz):
    m = p.copy()
    m[..., 2] = 2.0 * gz - p[..., 2]
    return m


def _seg_dist(P, A, D):
    """Distance from points P to segments A + t·D (t in [0, 1]), and t raw."""
    dd = np.einsum("ij,ij->i", D, D)
    with np.errstate(divide="ignore", invalid="ignore"):
        traw = np.where(dd > 0.0, np.einsum("ij,ij->i", P - A, D) / dd, 0.0)
    t = np.clip(traw, 0.0, 1.0)
    return np.linalg.norm(P - (A + t[:, None] * D), axis=1), traw


def pair_moments(
    seg_l, seg_r, tang, h, I, J, gz, k, grid, d1, *, cancel_flag=0, checkpoint=None
):
    """Segment-pair remainder moments `Jf[n, p, P]` for pairs (I[n], J[n]).

    `Jf[n, p, P] = ∫∫ u^p · t_i·F(r_i(u), r_j(u'))·t_j · u'^P du du'` over the
    two segments in physical arc length — the quantity
    `BSplineSolver._remainder_pair_moments` returns at a fixed Gauss order,
    here integrated on the graded panels of this module's docstring.
    """
    I = np.asarray(I, dtype=np.int64)
    J = np.asarray(J, dtype=np.int64)
    out = np.zeros((I.size, d1, d1), dtype=np.complex128)
    if I.size == 0:
        return out
    # Integrate each unordered pair once; the transpose serves its mirror.
    lo_ij = np.minimum(I, J)
    hi_ij = np.maximum(I, J)
    key = lo_ij * (int(max(I.max(), J.max())) + 1) + hi_ij
    ukey, first, inv = np.unique(key, return_index=True, return_inverse=True)
    Iu, Ju = lo_ij[first], hi_ij[first]
    Jf_u = np.zeros((ukey.size, d1, d1), dtype=np.complex128)
    lattice = _lattice(grid)
    xg, wg = leggauss(N_PANEL)
    powers = np.arange(d1)
    for c0 in range(0, Iu.size, PAIR_CHUNK):
        if checkpoint is not None:
            checkpoint()
        c1 = min(c0 + PAIR_CHUNK, Iu.size)
        Jf_u[c0:c1] = _chunk_moments(
            seg_l, seg_r, tang, h, Iu[c0:c1], Ju[c0:c1], gz, k, grid, lattice,
            xg, wg, powers, cancel_flag, checkpoint,
        )  # fmt: skip
    fwd = I <= J
    out[fwd] = Jf_u[inv[fwd]]
    out[~fwd] = np.swapaxes(Jf_u[inv[~fwd]], 1, 2)
    return out


def _chunk_moments(
    seg_l, seg_r, tang, h, I, J, gz, k, grid, lattice, xg, wg, powers, cancel_flag,
    checkpoint,
):  # fmt: skip
    npair = I.size
    oA, oD = seg_l[I], seg_r[I] - seg_l[I]
    sA, sD = seg_l[J], seg_r[J] - seg_l[J]
    Li, Lj = h[I], h[J]
    mA, mD = _mirror(oA, gz), oD.copy()
    mD[:, 2] = -oD[:, 2]

    # ---- outer breakpoints, per pair (observer parameter s) ----
    pid = [np.arange(npair), np.arange(npair)]
    sv = [np.zeros(npair), np.ones(npair)]
    kind_seg = [np.ones(npair, bool), np.ones(npair, bool)]  # scale = dist to seg
    anchor = [np.zeros((npair, 3)), np.zeros((npair, 3))]
    # The foot of the observer's image, t*(s) = t0 + alpha·s, leaving [0, 1].
    sdd = np.einsum("ij,ij->i", sD, sD)
    t0 = np.einsum("ij,ij->i", mA - sA, sD) / sdd
    alpha = np.einsum("ij,ij->i", mD, sD) / sdd
    moving = np.abs(alpha) > 1e-12
    for edge in (0.0, 1.0):
        with np.errstate(divide="ignore", invalid="ignore"):
            s_e = np.where(moving, (edge - t0) / np.where(moving, alpha, 1.0), -1.0)
        ok = (s_e > 0.0) & (s_e < 1.0)
        pid.append(np.flatnonzero(ok))
        sv.append(s_e[ok])
        kind_seg.append(np.ones(int(ok.sum()), bool))
        anchor.append(np.zeros((int(ok.sum()), 3)))
    # Closest approach of the observer's image line to the source line.
    w0 = mA - sA
    aa = np.einsum("ij,ij->i", mD, mD)
    bb = np.einsum("ij,ij->i", mD, sD)
    den = aa * sdd - bb * bb
    skew = den > 1e-12 * aa * sdd
    with np.errstate(divide="ignore", invalid="ignore"):
        s_c = np.where(
            skew,
            (bb * np.einsum("ij,ij->i", sD, w0) - sdd * np.einsum("ij,ij->i", mD, w0))
            / np.where(skew, den, 1.0),
            -1.0,
        )
    ok = (s_c > 0.0) & (s_c < 1.0)
    pid.append(np.flatnonzero(ok))
    sv.append(s_c[ok])
    kind_seg.append(np.ones(int(ok.sum()), bool))
    anchor.append(np.zeros((int(ok.sum()), 3)))
    # Where a SOURCE ENDPOINT crosses an interpolant seam as the observer moves.
    for E in (sA, sA + sD):
        kk, uu = _crossings(E, oA, oD, gz, lattice)
        pid.append(kk)
        sv.append(uu)
        kind_seg.append(np.zeros(kk.size, bool))
        anchor.append(E[kk])
    pid = np.concatenate(pid)
    sv = np.concatenate(sv)
    kind_seg = np.concatenate(kind_seg)
    anchor = np.concatenate(anchor)
    mpt = mA[pid] + sv[:, None] * mD[pid]
    d_seg, _ = _seg_dist(mpt, sA[pid], sD[pid])
    d_pt = np.linalg.norm(mpt - anchor, axis=1)
    scale = np.maximum(np.where(kind_seg, d_seg, d_pt) / Li[pid], SCALE_FLOOR)
    o_pair, s_o, w_o = _nodes(*_panels(pid, sv, scale, SIGMA), xg, wg)

    # ---- inner rule and kernel, a block of outer nodes at a time ----
    n_o = o_pair.size
    obs = oA[o_pair] + s_o[:, None] * oD[o_pair]
    t_obs = tang[I][o_pair]
    t_src = tang[J][o_pair]
    A_o, D_o, Lj_o = sA[o_pair], sD[o_pair], Lj[o_pair]
    v = np.empty((n_o, powers.size), dtype=np.complex128)
    for b0 in range(0, n_o, OUTER_BLOCK):
        if checkpoint is not None:
            checkpoint()
        b1 = min(b0 + OUTER_BLOCK, n_o)
        v[b0:b1] = _inner_moments(
            obs[b0:b1], t_obs[b0:b1], A_o[b0:b1], D_o[b0:b1], t_src[b0:b1],
            Lj_o[b0:b1], gz, k, grid, lattice, xg, wg, powers, cancel_flag,
        )  # fmt: skip

    # ---- outer contraction ----
    li_n = Li[o_pair]
    u_o = li_n * s_o
    wo = (li_n * w_o)[:, None] * u_o[:, None] ** powers[None, :]  # (n_o, d1): p
    starts_o = np.flatnonzero(np.r_[True, o_pair[1:] != o_pair[:-1]])
    return np.add.reduceat(wo[:, :, None] * v[:, None, :], starts_o, axis=0)


def _inner_moments(
    obs, t_obs, A_o, D_o, t_src, Lj_o, gz, k, grid, lattice, xg, wg, powers, cancel_flag
):
    """`v[n, P] = ∫ t·F(obs[n], A + t·D)·t' · u'^P du'` per outer node n,
    on the source segment split at the node's foot and at every interpolant
    seam it crosses, graded toward each."""
    n_o = obs.shape[0]
    mobs = _mirror(obs, gz)
    _, traw = _seg_dist(mobs, A_o, D_o)
    ar = np.arange(n_o)
    foot = (traw > 0.0) & (traw < 1.0)
    kx, ux = _crossings(obs, A_o, D_o, gz, lattice)
    tid = np.concatenate([ar, ar, ar[foot], kx])
    tv = np.concatenate([np.zeros(n_o), np.ones(n_o), traw[foot], ux])
    spt = A_o[tid] + tv[:, None] * D_o[tid]
    r1 = np.linalg.norm(mobs[tid] - spt, axis=1)
    scale_i = np.maximum(r1 / Lj_o[tid], SCALE_FLOOR)
    p_own, p0, p1 = _panels(tid, tv, scale_i, SIGMA)
    if _sommerfeld._acc is not None and hasattr(
        _sommerfeld._acc, "remainder_graded_inner"
    ):
        ptr = np.searchsorted(p_own, np.arange(n_o + 1))
        return _sommerfeld._acc.remainder_graded_inner(
            np.ascontiguousarray(obs),
            np.ascontiguousarray(t_obs),
            np.ascontiguousarray(A_o),
            np.ascontiguousarray(D_o),
            np.ascontiguousarray(t_src),
            np.ascontiguousarray(Lj_o, dtype=np.float64),
            ptr.astype(np.int64),
            p0,
            p1,
            np.ascontiguousarray(xg),
            np.ascontiguousarray(wg),
            int(powers.size),
            float(gz),
            float(k),
            *_sommerfeld.grid_cpp_args(grid),
            int(cancel_flag),
        )
    i_own, t_i, w_i = _nodes(p_own, p0, p1, xg, wg)

    src = A_o[i_own] + t_i[:, None] * D_o[i_own]
    F = _sommerfeld.remainder_field_proj_owned(
        obs, t_obs, src, t_src[i_own], i_own, gz, k, grid, cancel_flag
    )
    lj_n = Lj_o[i_own]
    u_i = lj_n * t_i
    wu = (lj_n * w_i)[:, None] * u_i[:, None] ** powers[None, :]
    starts_i = np.flatnonzero(np.r_[True, i_own[1:] != i_own[:-1]])
    return np.add.reduceat(F[:, None] * wu, starts_i, axis=0)
