"""The sinusoidal-Galerkin fill in MIXED-POTENTIAL form, on the B-spline
solver's pair-moment machinery (momwire#1354).

`SinusoidalGalerkinSolver`'s shipped fill is the direct-field form,
Z_mn = <f_m, E[f_n]>: NEC's closed-form field of each segment's three current
shapes, evaluated at every test segment's Gauss nodes. Every pair costs the
same 8 observers x (8 source nodes + 3 end phases) whatever its distance,
which is why that fill sits at ~7x NEC-4.2 where `BSplineSolver` sits at ~3x.

This module writes the SAME operator the way the B-spline trunk writes it,

    Z_mn = jk eta (t_m . t_n) <f_m, G f_n> + (eta / jk) <f'_m, G f'_n>,

i.e. as pair MOMENTS of the Green's function against the basis shapes, with the
shapes evaluated at the quadrature nodes. The two forms are one integration by
parts apart, and the boundary terms telescope over a basis's support because
the NEC basis is continuous, vanishes at free ends and satisfies KCL at
junctions — so block by block (free space, image, remainder) the mixed form
equals the direct form exactly, and only the quadrature differs. That is what
lets the B-spline fill's distance-tiered pair orders serve this basis unchanged:
a far pair costs a 4x4 rule instead of 88 closed-form evaluations.

The basis hook is the shape set. On a segment of length h, with xi the arc from
the segment CENTRE along its natural tangent, the three shapes are the folded
set `{1, sin k xi, cos k xi - 1}` (#203/#606: `cos k xi - 1` is spelled
`-2 sin^2(k xi / 2)`, so nothing here is larger than the answer), a basis's
restriction to the segment is `c0 + c1 sin k xi + c2 (cos k xi - 1)` with
`(c0, c1, c2) = (sigma AC, B, sigma C)` straight from `_basis_coefs`, and its
derivative is the same set with the coefficients rotated,
`f' = k [c1 . 1 - c2 . sin k xi + c1 . (cos k xi - 1)]` (`dshape_coefs`). The
B-spline trunk's monomial shapes are the same structure with `u^p` for the
shapes and `p u^(p-1)` for the derivative; the moment kernels differ in nothing
but the shape evaluated at each node.

Near pairs are the one place the sinusoidal and B-spline trunks part: the
B-spline same-edge block has analytic static moments for monomials, which
sinusoids do not have. A parallel pair (every collinear pair, every self pair,
a horizontal wire against its own image) reduces instead to ONE integral over
the separation delta = x - x' along the common line, `int G(delta) Lambda(delta)
d delta`, with Lambda the shapes' correlation over the overlap (smooth,
quadratured) and the `1/R` of G removed exactly by delta = rho sinh t
(`parallel_pair_moments`). A touching non-parallel pair (a bend, a junction)
takes a composite rule graded toward the shared node on both segments.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse

from . import _accel
from ._quadrature import leggauss

_acc = _accel.acc
_HAVE_SIN_TIERED = _acc is not None and hasattr(
    _acc, "seg_seg_full_moments_sinusoidal_tiered"
)
_HAVE_SIN_ASSEMBLE = _acc is not None and hasattr(
    _acc, "assemble_Z_sinusoidal_windowed"
)

N_SHAPES = 3

# Parallel pairs closer than this many (longer) segment lengths take the exact
# one-dimensional reduction; non-parallel pairs closer than it take the graded
# or high-order product rule. Beyond it every pair is a ladder pair. 2.0 is the
# buried ladder's first threshold (`bspline.BURIED_PAIR_ORDER_LADDER`): a pair
# two segment lengths apart has R >= h over its whole domain and the base
# product rule already converges there.
NEAR_RATIO = 2.0
# Nodes per piece of the sinh-substituted delta integral, and nodes of the
# inner (kernel-free) shape correlation. See `parallel_pair_moments`.
PARALLEL_N_T = 24
PARALLEL_N_XI = 8
# Nodes per panel of the graded rule on a touching non-parallel pair, and the
# product order on a near non-parallel pair that does not touch.
GRADED_N_PER_PANEL = 8
NEAR_N_QP = 32


def shape_values(k, xi):
    """The folded shape set at arcs `xi` from the segment centre: an array
    (3, *xi.shape) of `1`, `sin k xi`, `-2 sin^2(k xi / 2)`."""
    xi = np.asarray(xi)
    arg = k * xi
    half = np.sin(0.5 * arg)
    out = np.empty((N_SHAPES,) + xi.shape, dtype=np.result_type(k, xi, np.complex128))
    out[0] = 1.0
    out[1] = np.sin(arg)
    out[2] = -2.0 * half * half
    return out


def dshape_coefs(k, coefs):
    """Derivative coefficients on the same shape set: `f = c0 + c1 sin + c2
    (cos - 1)` has `f' = k [c1 + (-c2) sin + c1 (cos - 1)]`. `coefs` is
    (..., 3); `k` is a scalar or broadcasts over the leading axes."""
    c = np.asarray(coefs)
    k = np.asarray(k)  # scalar, or one per leading index of `coefs`
    out = np.empty_like(c, dtype=np.complex128)
    out[..., 0] = k * c[..., 1]
    out[..., 1] = -k * c[..., 2]
    out[..., 2] = k * c[..., 1]
    return out


def segment_ends(centers, tangents, h):
    """(seg_l, seg_r) from centre / tangent / length: the arc u in [0, h] of
    the pair kernels runs from `seg_l` along the natural tangent, so xi = u -
    h/2 is exactly the shape set's arc."""
    half = 0.5 * np.asarray(h, dtype=float)[:, None]
    t = np.asarray(tangents, dtype=float)
    c = np.asarray(centers, dtype=float)
    return c - half * t, c + half * t


def _gl01(n):
    x, w = leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def pair_moments_product(seg_l_i, seg_r_i, seg_l_j, seg_r_j, a2, k, t01, w01):
    """Numpy reference of the product-rule pair moments on ONE rule:
    J[p, q, i, j] = sum_ab w_a w_b S_p(xi_a) S_q(xi'_b) G(R_ab), with
    G = exp(-jkR) / (4 pi R), R^2 = |r_a - r'_b|^2 + a2, on every (i, j).
    `a2` is a scalar or per-row (N_i,)."""
    seg_l_i, seg_r_i = np.asarray(seg_l_i, float), np.asarray(seg_r_i, float)
    seg_l_j, seg_r_j = np.asarray(seg_l_j, float), np.asarray(seg_r_j, float)
    len_i = np.linalg.norm(seg_r_i - seg_l_i, axis=1)
    len_j = np.linalg.norm(seg_r_j - seg_l_j, axis=1)
    t = t01[None, :, None]
    pos_i = (1 - t) * seg_l_i[:, None, :] + t * seg_r_i[:, None, :]
    pos_j = (1 - t) * seg_l_j[:, None, :] + t * seg_r_j[:, None, :]
    xi_i = (t01[None, :] - 0.5) * len_i[:, None]
    xi_j = (t01[None, :] - 0.5) * len_j[:, None]
    ws_i = (
        w01[None, None, :] * len_i[None, :, None] * shape_values(k, xi_i)
    )  # (3, Ni, q)
    ws_j = w01[None, None, :] * len_j[None, :, None] * shape_values(k, xi_j)
    diff = pos_i[:, :, None, None, :] - pos_j[None, None, :, :, :]
    a2b = a2 if np.ndim(a2) == 0 else np.asarray(a2)[:, None, None, None]
    R = np.sqrt((diff * diff).sum(-1) + a2b)
    G = np.exp(-1j * k * R) / (4 * np.pi * R)
    return np.einsum("piq,iqjr,Pjr->pPij", ws_i, G, ws_j)


def pair_moments_tiered(seg_l_i, seg_r_i, seg_l_j, seg_r_j, a2, k, n_qp, ladder):
    """The ladder pair moments: order `n_qp` below the first threshold, each
    tier's order at and above its ratio (`bspline.pair_order_ladder`'s
    selector, centre distance over the longer segment). C++ when built, the
    numpy product rule per tier otherwise. `a2` scalar (one radius per call;
    the caller splits mixed-radius rows into runs)."""
    seg_l_i, seg_r_i = np.asarray(seg_l_i, float), np.asarray(seg_r_i, float)
    seg_l_j, seg_r_j = np.asarray(seg_l_j, float), np.asarray(seg_r_j, float)
    ladder = tuple(ladder or ())
    if _HAVE_SIN_TIERED:
        from ._bspline_kernels import _ladder_arrays

        tier_t, tier_w, tier_n, tier_r = _ladder_arrays(n_qp, ladder)
        return _acc.seg_seg_full_moments_sinusoidal_tiered(
            np.ascontiguousarray(seg_l_i),
            np.ascontiguousarray(seg_r_i),
            np.ascontiguousarray(seg_l_j),
            np.ascontiguousarray(seg_r_j),
            float(a2),
            complex(k),
            tier_t,
            tier_w,
            tier_n,
            tier_r,
        )
    return _pair_moments_tiered_numpy(
        seg_l_i, seg_r_i, seg_l_j, seg_r_j, a2, k, n_qp, ladder
    )


def _pair_moments_tiered_numpy(seg_l_i, seg_r_i, seg_l_j, seg_r_j, a2, k, n_qp, ladder):
    block = pair_moments_product(
        seg_l_i, seg_r_i, seg_l_j, seg_r_j, a2, k, *_gl01(n_qp)
    )
    if not ladder:
        return block
    from ._bspline_kernels import _pair_ratio

    ratio = _pair_ratio(seg_l_i, seg_r_i, seg_l_j, seg_r_j)
    for r, n in ladder:
        tier = pair_moments_product(
            seg_l_i, seg_r_i, seg_l_j, seg_r_j, a2, k, *_gl01(n)
        )
        block = np.where(ratio[None, None] >= r, tier, block)
    return block


# ----------------------------------------------------------------------
# Near pairs
# ----------------------------------------------------------------------


def classify_pairs(c_i, t_i, h_i, c_j, t_j, h_j, *, near_ratio=NEAR_RATIO, tol=1e-9):
    """Which (i, j) pairs are near, and of which kind.

    Returns `(ii, jj, parallel, touching, shared_end_i, shared_end_j)` over
    the pairs with centre distance below `near_ratio` x the longer segment:
    `parallel` where |t_i . t_j| = 1, `touching` where an end of i coincides
    with an end of j (non-parallel only — a parallel touching pair is served by
    the one-dimensional reduction), `shared_end_*` = 0 for the segment's
    start (u = 0) and 1 for its end, or -1.
    """
    c_i, c_j = np.asarray(c_i, float), np.asarray(c_j, float)
    t_i, t_j = np.asarray(t_i, float), np.asarray(t_j, float)
    h_i, h_j = np.asarray(h_i, float), np.asarray(h_j, float)
    # Candidates within near_ratio x the longest segment of either side by
    # tree query, then the exact per-pair ratio.
    from scipy.spatial import cKDTree

    reach = near_ratio * max(float(h_i.max()), float(h_j.max()))
    hits = cKDTree(c_i).query_ball_tree(cKDTree(c_j), reach)
    ii = np.repeat(np.arange(c_i.shape[0]), [len(x) for x in hits])
    jj = np.fromiter((j for x in hits for j in x), dtype=np.int64, count=ii.size)
    if ii.size:
        dist = np.linalg.norm(c_i[ii] - c_j[jj], axis=1)
        keep = dist < near_ratio * np.maximum(h_i[ii], h_j[jj])
        ii, jj = ii[keep], jj[keep]
        order = np.lexsort((jj, ii))
        ii, jj = ii[order], jj[order]
    dot = np.einsum("ij,ij->i", t_i[ii], t_j[jj])
    parallel = np.abs(np.abs(dot) - 1.0) <= tol
    ends_i = np.stack(segment_ends(c_i, t_i, h_i), axis=1)  # (Ni, 2, 3)
    ends_j = np.stack(segment_ends(c_j, t_j, h_j), axis=1)
    scale = np.maximum(h_i[ii], h_j[jj])
    d = np.linalg.norm(ends_i[ii][:, :, None, :] - ends_j[jj][:, None, :, :], axis=3)
    close = d <= tol * scale[:, None, None]
    touching = close.any(axis=(1, 2)) & ~parallel
    shared_i = np.where(touching, np.argmax(close.any(axis=2), axis=1), -1)
    shared_j = np.where(touching, np.argmax(close.any(axis=1), axis=1), -1)
    return ii, jj, parallel, touching, shared_i, shared_j


def parallel_pair_moments(
    c_i, t_i, h_i, c_j, t_j, h_j, a2, k, *, n_t=PARALLEL_N_T, n_xi=PARALLEL_N_XI
):
    """Exact (to quadrature) moments of PARALLEL pairs, (3, 3, n_pairs).

    With x the line coordinate along t_j (origin at c_j), the source arc is
    xi' = x' and the test arc xi runs x = d_par + s xi, s = t_i . t_j = +-1,
    d_par = (c_i - c_j) . t_j. The kernel depends on the pair only through
    delta = x - x' and the perpendicular offset, R^2 = delta^2 + rho^2 with
    rho^2 = d_perp^2 + a2, so

        J[p, q] = int d delta  G(delta) Lambda_pq(delta),
        Lambda_pq(delta) = int_{Omega(delta)} S_p(xi) S_q(d_par + s xi - delta) d xi,

    Omega the xi-range where both arcs are inside their segments. Lambda is
    piecewise analytic in delta with breakpoints where Omega's ends switch
    (d_par +- (h_i - h_j)/2) and vanishes outside d_par +- (h_i + h_j)/2;
    each piece is integrated in t with delta = rho sinh t, which turns `d
    delta / R` into `dt` exactly, so the 1/R spike of width rho at delta = 0
    (the self pair's whole difficulty) is gone before any rule is applied,
    and Gauss-Legendre in t sees exp(-jkR) Lambda, smooth on each piece.
    Lambda itself is a kernel-free integral of trig shapes over an interval
    shorter than a segment: `n_xi` Gauss nodes are exact to rounding for
    k h below ~1.
    """
    c_i, c_j = np.asarray(c_i, float), np.asarray(c_j, float)
    t_i, t_j = np.asarray(t_i, float), np.asarray(t_j, float)
    h_i, h_j = np.asarray(h_i, float), np.asarray(h_j, float)
    n = c_i.shape[0]
    s = np.sign(np.einsum("ij,ij->i", t_i, t_j))
    dc = c_i - c_j
    d_par = np.einsum("ij,ij->i", dc, t_j)
    d_perp2 = np.maximum(np.einsum("ij,ij->i", dc, dc) - d_par * d_par, 0.0)
    rho = np.sqrt(d_perp2 + a2)

    # Breakpoints of Lambda in delta: the four corner values, sorted per pair.
    br = np.stack(
        [
            d_par - 0.5 * (h_i + h_j),
            d_par - 0.5 * (h_i - h_j),
            d_par + 0.5 * (h_i - h_j),
            d_par + 0.5 * (h_i + h_j),
        ],
        axis=1,
    )
    br.sort(axis=1)
    gt, gw = leggauss(n_t)
    gx, gwx = leggauss(n_xi)
    out = np.zeros((N_SHAPES, N_SHAPES, n), dtype=np.complex128)
    for piece in range(3):
        lo_t = np.arcsinh(br[:, piece] / rho)
        hi_t = np.arcsinh(br[:, piece + 1] / rho)
        mid, half = 0.5 * (hi_t + lo_t), 0.5 * (hi_t - lo_t)
        t = mid[:, None] + half[:, None] * gt[None, :]  # (n, n_t)
        w_t = half[:, None] * gw[None, :]
        delta = rho[:, None] * np.sinh(t)
        R = rho[:, None] * np.cosh(t)
        # G d delta = exp(-jkR) / (4 pi R) . R dt
        Gdt = np.exp(-1j * k * R) / (4 * np.pi) * w_t
        beta = d_par[:, None] - delta  # xi' = s xi + beta
        # Omega(delta) = [lo, hi] in xi: xi' = s xi + beta inside [-h_j/2, h_j/2].
        lo = np.where(
            s[:, None] > 0,
            np.maximum(-0.5 * h_i[:, None], -beta - 0.5 * h_j[:, None]),
            np.maximum(-0.5 * h_i[:, None], beta - 0.5 * h_j[:, None]),
        )
        hi = np.where(
            s[:, None] > 0,
            np.minimum(0.5 * h_i[:, None], -beta + 0.5 * h_j[:, None]),
            np.minimum(0.5 * h_i[:, None], beta + 0.5 * h_j[:, None]),
        )
        width = np.maximum(hi - lo, 0.0)
        xm, xh = 0.5 * (hi + lo), 0.5 * width
        xi = xm[:, :, None] + xh[:, :, None] * gx[None, None, :]  # (n, n_t, n_xi)
        w_xi = xh[:, :, None] * gwx[None, None, :]
        xi_src = s[:, None, None] * xi + beta[:, :, None]
        S_i = shape_values(k, xi)  # (3, n, n_t, n_xi)
        S_j = shape_values(k, xi_src)
        lam = np.einsum("pntx,qntx,ntx->pqnt", S_i, S_j, w_xi)
        out += np.einsum("pqnt,nt->pqn", lam, Gdt)
    return out


def graded_endpoint_rule01(eps, n_per_panel, end):
    """Composite Gauss rule on u in [0, 1] with panels doubling away from
    `end` (0 or 1): widths eps, 2 eps, ... up to the far end."""
    eps = float(np.clip(eps, 1e-9, 0.5))
    d = eps * 2.0 ** np.arange(64)
    d = d[d < 1.0]
    edges = np.concatenate(([0.0], d, [1.0]))
    gx, gw = leggauss(n_per_panel)
    lo, hi = edges[:-1], edges[1:]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    x = (mid[:, None] + half[:, None] * gx[None, :]).ravel()
    w = (half[:, None] * gw[None, :]).ravel()
    if end == 1:
        x = 1.0 - x
    return x, w


def near_pair_moments(
    c_i, t_i, h_i, c_j, t_j, h_j, a2, k, *, pairs=None, near_ratio=NEAR_RATIO
):
    """Moments of every near pair of (i-segments, j-segments) by the rule its
    kind needs: `(ii, jj, J)` with J (3, 3, n_near). `a2` is per row (N_i,)
    or scalar. A touching non-parallel pair takes the graded composite rule
    toward the shared node on both segments; a near non-parallel pair that
    does not touch takes the `NEAR_N_QP` product rule; a parallel pair the
    one-dimensional reduction."""
    c_i, c_j = np.asarray(c_i, float), np.asarray(c_j, float)
    t_i, t_j = np.asarray(t_i, float), np.asarray(t_j, float)
    h_i, h_j = np.asarray(h_i, float), np.asarray(h_j, float)
    if pairs is None:
        pairs = classify_pairs(c_i, t_i, h_i, c_j, t_j, h_j, near_ratio=near_ratio)
    ii, jj, parallel, touching, end_i, end_j = pairs
    a2_row = np.broadcast_to(np.asarray(a2, float), (c_i.shape[0],))
    J = np.zeros((N_SHAPES, N_SHAPES, ii.shape[0]), dtype=np.complex128)
    if parallel.any():
        p = parallel
        J[:, :, p] = parallel_pair_moments(
            c_i[ii[p]],
            t_i[ii[p]],
            h_i[ii[p]],
            c_j[jj[p]],
            t_j[jj[p]],
            h_j[jj[p]],
            a2_row[ii[p]][None, :].T[:, 0] if False else a2_row[ii[p]],
            k,
        )
    rest = ~parallel
    if not rest.any():
        return ii, jj, J
    sl_i, sr_i = segment_ends(c_i, t_i, h_i)
    sl_j, sr_j = segment_ends(c_j, t_j, h_j)
    # Touching pairs share a rule whenever their (shared end, a/h) agree, so
    # they are integrated in batches; the rest take one product rule.
    t01, w01 = _gl01(NEAR_N_QP)
    rest_idx = np.nonzero(rest)[0]
    eps_i = np.sqrt(a2_row[ii[rest_idx]]) / h_i[ii[rest_idx]]
    eps_j = np.sqrt(a2_row[ii[rest_idx]]) / h_j[jj[rest_idx]]
    keys = {}
    for n, idx in enumerate(rest_idx):
        key = (
            (
                int(end_i[idx]),
                int(end_j[idx]),
                round(float(eps_i[n]), 12),
                round(float(eps_j[n]), 12),
            )
            if touching[idx]
            else None
        )
        keys.setdefault(key, []).append(idx)
    for key, group in keys.items():
        group = np.asarray(group)
        if key is None:
            ti, wi = tj, wj = t01, w01
        else:
            ti, wi = graded_endpoint_rule01(key[2], GRADED_N_PER_PANEL, key[0])
            tj, wj = graded_endpoint_rule01(key[3], GRADED_N_PER_PANEL, key[1])
        gi, gj = ii[group], jj[group]
        J[:, :, group] = _pair_moments_two_rules_batch(
            sl_i[gi], sr_i[gi], sl_j[gj], sr_j[gj], a2_row[gi], k, ti, wi, tj, wj
        )
    return ii, jj, J


def _pair_moments_two_rules_batch(l_i, r_i, l_j, r_j, a2, k, ti, wi, tj, wj):
    """Product-rule moments of a batch of pairs sharing the two rules:
    (3, 3, n_pairs)."""
    len_i = np.linalg.norm(r_i - l_i, axis=1)
    len_j = np.linalg.norm(r_j - l_j, axis=1)
    pos_i = (1 - ti)[None, :, None] * l_i[:, None, :] + ti[None, :, None] * r_i[
        :, None, :
    ]
    pos_j = (1 - tj)[None, :, None] * l_j[:, None, :] + tj[None, :, None] * r_j[
        :, None, :
    ]
    ws_i = (wi[None, :] * len_i[:, None])[None] * shape_values(
        k, (ti[None, :] - 0.5) * len_i[:, None]
    )  # (3, n, qi)
    ws_j = (wj[None, :] * len_j[:, None])[None] * shape_values(
        k, (tj[None, :] - 0.5) * len_j[:, None]
    )
    diff = pos_i[:, :, None, :] - pos_j[:, None, :, :]
    R = np.sqrt((diff * diff).sum(-1) + np.asarray(a2)[:, None, None])
    G = np.exp(-1j * k * R) / (4 * np.pi * R)  # (n, qi, qj)
    return np.einsum("pnq,nqr,Pnr->pPn", ws_i, G, ws_j)


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------


def basis_csr(seg_view, k):
    """The CSR-by-segment coefficient tables the assembler eats, from
    `_basis_coefs`' view: `(starts, jbasis, coef, dcoef)` with `coef` /
    `dcoef` (nnz, 3) complex on the folded shape set. `k` is the scalar
    wavenumber, or per entry through `seg_view["k_entry"]`."""
    starts = np.asarray(seg_view["starts"], dtype=np.int64)
    jbasis = np.asarray(seg_view["jbasis"], dtype=np.int64)
    sig = np.asarray(seg_view["sigma"]).astype(np.complex128)
    coef = np.stack(
        [
            sig * seg_view["AC"],
            np.asarray(seg_view["B"], np.complex128),
            sig * seg_view["C"],
        ],
        axis=1,
    )
    k_entry = seg_view.get("k_entry")
    dcoef = dshape_coefs(k if k_entry is None else np.asarray(k_entry), coef)
    return starts, jbasis, np.ascontiguousarray(coef), np.ascontiguousarray(dcoef)


def csr_subset(starts, jbasis, coef, dcoef, idx):
    """The CSR tables restricted to the segments `idx` (in that order): the
    sub-fill's own `(starts, jbasis, coef, dcoef)`, bases unrenumbered."""
    starts = np.asarray(starts)
    idx = np.asarray(idx, dtype=np.int64)
    cnt = np.diff(starts)[idx]
    sub_starts = np.concatenate(([0], np.cumsum(cnt))).astype(np.int64)
    ent = np.repeat(starts[idx] - sub_starts[:-1], cnt) + np.arange(
        sub_starts[-1], dtype=np.int64
    )
    return sub_starts, np.asarray(jbasis)[ent], coef[ent], dcoef[ent]


def assemble_window(
    Z,
    J,
    rows_seg,
    cols_seg,
    starts,
    jbasis,
    coef,
    dcoef,
    tangents_rows,
    tangents_cols,
    c_a,
    c_phi,
    *,
    w_a=None,
    w_phi=None,
    scale=1.0,
):
    """Accumulate one moment window into Z (in place):

        Z[i, j] += scale . sum_{m in rows, n in cols} sum_{e on m, e' on n}
                   [ c_a . w_a[m,n] . (t_m . t_n) . coef_e^T J[:, :, m, n] coef_e'
                   + c_phi . w_phi[m,n] . dcoef_e^T J[:, :, m, n] dcoef_e' ]

    over the CSR entries e of segment `rows_seg[m]` (basis `jbasis[e]`) and e'
    of `cols_seg[n]`. `J` is (3, 3, n_rows, n_cols); `tangents_*` (n, 3) are
    the window's own tangents (an image window hands the MIRRORED source
    tangents, which is where the image's sign lives); `w_a` / `w_phi` are
    optional per-pair complex weights (the Fresnel or C2 tables), default 1.
    """
    if _HAVE_SIN_ASSEMBLE:
        _acc.assemble_Z_sinusoidal_windowed(
            np.ascontiguousarray(J, dtype=np.complex128),
            np.ascontiguousarray(rows_seg, dtype=np.int64),
            np.ascontiguousarray(cols_seg, dtype=np.int64),
            np.ascontiguousarray(starts, dtype=np.int64),
            np.ascontiguousarray(jbasis, dtype=np.int64),
            np.ascontiguousarray(coef, dtype=np.complex128),
            np.ascontiguousarray(dcoef, dtype=np.complex128),
            np.ascontiguousarray(tangents_rows, dtype=np.float64),
            np.ascontiguousarray(tangents_cols, dtype=np.float64),
            complex(c_a * scale),
            complex(c_phi * scale),
            None if w_a is None else np.ascontiguousarray(w_a, dtype=np.complex128),
            None if w_phi is None else np.ascontiguousarray(w_phi, dtype=np.complex128),
            Z,
        )
        return
    _assemble_window_numpy(
        Z,
        J,
        rows_seg,
        cols_seg,
        starts,
        jbasis,
        coef,
        dcoef,
        tangents_rows,
        tangents_cols,
        c_a,
        c_phi,
        w_a=w_a,
        w_phi=w_phi,
        scale=scale,
    )


def _assemble_window_numpy(
    Z,
    J,
    rows_seg,
    cols_seg,
    starts,
    jbasis,
    coef,
    dcoef,
    tangents_rows,
    tangents_cols,
    c_a,
    c_phi,
    *,
    w_a=None,
    w_phi=None,
    scale=1.0,
):
    rows_seg = np.asarray(rows_seg)
    cols_seg = np.asarray(cols_seg)
    td = np.asarray(tangents_rows) @ np.asarray(tangents_cols).T  # (n_rows, n_cols)
    wa = td if w_a is None else td * w_a
    wp = np.ones_like(td) if w_phi is None else np.asarray(w_phi)
    # Entries per window segment, as ragged (entry, local segment) lists.
    e_r = np.concatenate([np.arange(starts[s], starts[s + 1]) for s in rows_seg])
    m_r = np.repeat(np.arange(rows_seg.size), np.diff(starts)[rows_seg])
    e_c = np.concatenate([np.arange(starts[s], starts[s + 1]) for s in cols_seg])
    n_c = np.repeat(np.arange(cols_seg.size), np.diff(starts)[cols_seg])
    Jw = J[:, :, m_r[:, None], n_c[None, :]]  # (3, 3, E_r, E_c)
    A = (
        np.einsum("ep,pqef,fq->ef", coef[e_r], Jw, coef[e_c])
        * wa[m_r[:, None], n_c[None, :]]
    )
    P = (
        np.einsum("ep,pqef,fq->ef", dcoef[e_r], Jw, dcoef[e_c])
        * wp[m_r[:, None], n_c[None, :]]
    )
    val = scale * (c_a * A + c_phi * P)
    np.add.at(Z, (jbasis[e_r][:, None], jbasis[e_c][None, :]), val)


# The direct form tests E = -jwA - grad Phi, so its G is MINUS the potential
# form's jk eta <f, G f> + (eta / jk) <f', G f'>.
DIRECT_SIGN = -1.0
# The remainder enters G as `+rem` (`_fold_ground_block`: free - (c2 img - rem)).
REMAINDER_SIGN = 1.0
# bspline's `self_completions` are an additive correction to its Z = -G.
COMPLETION_SIGN = -1.0
# `_crossing_fill.axis_data`'s density on the completion axes (None = the
# module's defaults, bspline's). Swept on the D3 fan deck in `probe_fan_axes`.
COMPLETION_Q = None
COMPLETION_PANEL_ORDER = None
COMPLETION_GROWTH = None

# Bytes of pair-moment window held at once: (3, 3, rows, N) complex128.
WINDOW_BYTES = 256 << 20


class WindowFill:
    """One block of the mixed-potential fill into G, an observer window at a
    time: tiered pair moments for the window's rows against every source,
    the near pairs overwritten by their own rules, then the assembler.

    `starts / jbasis / coef / dcoef` are `basis_csr`'s tables for the deck's
    segments (the source side AND the observer side index them: an image
    block's sources are the same segments mirrored); `(k, eta)` is the
    block's operating point; `n_qp` and `ladder` the B-spline fill's base
    order and pair-order ladder. `scale` multiplies the block (the image's
    `-coefficient`); `weights(i0, i1) -> (w_a, w_phi)` supplies optional
    per-pair tables for the window's rows.
    """

    def __init__(
        self, G, starts, jbasis, coef, dcoef, k, eta, *, n_qp, ladder, checkpoint=None
    ):
        self.G = G
        self.starts = np.asarray(starts, dtype=np.int64)
        self.jbasis = np.asarray(jbasis, dtype=np.int64)
        self.coef = np.ascontiguousarray(coef, dtype=np.complex128)
        self.dcoef = np.ascontiguousarray(dcoef, dtype=np.complex128)
        self.k = k
        self.c_a = 1j * k * eta
        self.c_phi = eta / (1j * k)
        self.n_qp = int(n_qp)
        self.ladder = tuple(ladder or ())
        self.checkpoint = checkpoint

    def accumulate(
        self,
        obs_c,
        obs_t,
        obs_h,
        a_row,
        src_c,
        src_t,
        src_h,
        *,
        scale=1.0,
        weights=None,
        obs_idx=None,
        src_idx=None,
    ):
        """`obs_idx` / `src_idx` are the GLOBAL segment ids of the observer
        and source lists (a class block of a mixed deck); default identity."""
        obs_c, obs_t = np.asarray(obs_c, float), np.asarray(obs_t, float)
        src_c, src_t = np.asarray(src_c, float), np.asarray(src_t, float)
        obs_h, src_h = np.asarray(obs_h, float), np.asarray(src_h, float)
        a_row = np.broadcast_to(np.asarray(a_row, float), (obs_c.shape[0],))
        n_obs, n_src = obs_c.shape[0], src_c.shape[0]
        sl_o, sr_o = segment_ends(obs_c, obs_t, obs_h)
        sl_s, sr_s = segment_ends(src_c, src_t, src_h)
        rows = max(1, WINDOW_BYTES // (N_SHAPES * N_SHAPES * 16 * max(n_src, 1)))
        obs_idx = (
            np.arange(n_obs, dtype=np.int64)
            if obs_idx is None
            else np.asarray(obs_idx, dtype=np.int64)
        )
        all_cols = (
            np.arange(n_src, dtype=np.int64)
            if src_idx is None
            else np.asarray(src_idx, dtype=np.int64)
        )
        for i0 in range(0, n_obs, rows):
            if self.checkpoint is not None:
                self.checkpoint()
            i1 = min(i0 + rows, n_obs)
            J = self._window_moments(sl_o, sr_o, sl_s, sr_s, a_row, i0, i1)
            ii, jj, Jn = near_pair_moments(
                obs_c[i0:i1],
                obs_t[i0:i1],
                obs_h[i0:i1],
                src_c,
                src_t,
                src_h,
                a_row[i0:i1] ** 2,
                self.k,
            )
            if ii.size:
                J[:, :, ii, jj] = Jn
            w_a = w_phi = None
            if weights is not None:
                w_a, w_phi = weights(i0, i1)
            assemble_window(
                self.G,
                J,
                obs_idx[i0:i1],
                all_cols,
                self.starts,
                self.jbasis,
                self.coef,
                self.dcoef,
                obs_t[i0:i1],
                src_t,
                self.c_a,
                self.c_phi,
                w_a=w_a,
                w_phi=w_phi,
                scale=scale,
            )
            del J

    def _window_moments(self, sl_o, sr_o, sl_s, sr_s, a_row, i0, i1):
        """The window's tiered moments, one call per run of equal observer
        radius (the kernel takes one a^2 per call)."""
        a = a_row[i0:i1]
        bounds = np.flatnonzero(np.diff(a)) + 1
        if bounds.size == 0:
            return pair_moments_tiered(
                sl_o[i0:i1],
                sr_o[i0:i1],
                sl_s,
                sr_s,
                float(a[0]) ** 2,
                self.k,
                self.n_qp,
                self.ladder,
            )
        edges = np.concatenate(([0], bounds, [i1 - i0]))
        return np.concatenate(
            [
                pair_moments_tiered(
                    sl_o[i0 + s : i0 + e],
                    sr_o[i0 + s : i0 + e],
                    sl_s,
                    sr_s,
                    float(a[s]) ** 2,
                    self.k,
                    self.n_qp,
                    self.ladder,
                )
                for s, e in zip(edges[:-1], edges[1:])
            ],
            axis=2,
        )


# ----------------------------------------------------------------------
# The Sommerfeld remainder (above the plane) on the B-spline fused kernel
# ----------------------------------------------------------------------

# Taylor degree that writes the shape set as polynomials in u on one segment:
# (k h / 2)^(D+1) / (D+1)! is 3e-18 at k h = 0.5 and 2e-13 at k h = 1.2, below
# the graded remainder's own 1e-9 (`_remainder_graded`). Only the listed
# grazing pairs take this route; the base fill evaluates the shapes directly.
REMAINDER_TAYLOR_DEGREE = 12


def remainder_shape_weights(k, h, t01, w01):
    """`W[p, seg, q] = (h w_q) S_p(xi_q)` for the fused remainder kernel: the
    Gauss weight on the physical arc times the shape at the node, the shape
    hook of `BSplineSolver._Z_sommerfeld_remainder`'s `W` (there `u^p`).
    Real at a real k, which the kernel requires."""
    h = np.asarray(h, float)
    xi = (t01[None, :] - 0.5) * h[:, None]  # (N, q)
    S = shape_values(k, xi)  # (3, N, q)
    if np.iscomplexobj(S) and np.abs(S.imag).max() != 0.0:
        raise ValueError("remainder_shape_weights needs a real k")
    return np.ascontiguousarray((h[:, None] * w01[None, :])[None] * S.real)


def basis_wings(starts, jbasis, coef, width=N_SHAPES):
    """The CSR basis as fixed-width wing tables: `(rows_basis, loc, pl)`,
    a basis with more than `width` support entries split over several rows
    (`rows_basis` names each row's basis), unused slots on segment 0 with a
    zero polynomial — the B-spline `supp_seg` / `polys` convention, which is
    how the fused remainder kernel can assemble this basis unchanged. The
    rows' Q folds back onto the bases by `rows_basis`."""
    starts = np.asarray(starts)
    jbasis = np.asarray(jbasis)
    seg_of_entry = np.repeat(np.arange(starts.size - 1), np.diff(starts))
    order = np.argsort(jbasis, kind="stable")
    counts = np.bincount(jbasis[order])
    n_basis = counts.size
    n_rows_per = np.maximum((counts + width - 1) // width, 1)
    rows_basis = np.repeat(np.arange(n_basis), n_rows_per)
    R = rows_basis.size
    loc = np.zeros((R, width), dtype=np.int64)
    pl = np.zeros((R, width, N_SHAPES), dtype=np.complex128)
    row_start = np.concatenate(([0], np.cumsum(n_rows_per)))
    pos = np.arange(order.size) - np.repeat(np.cumsum(counts) - counts, counts)
    row = row_start[jbasis[order]] + pos // width
    slot = pos % width
    loc[row, slot] = seg_of_entry[order]
    pl[row, slot] = coef[order]
    return rows_basis, loc, pl


def taylor_shape_coefs(k, h, degree=REMAINDER_TAYLOR_DEGREE):
    """`tau[p, m]` with `S_p(u - h/2) = sum_m tau[p, m] u^m` on `u in [0, h]`,
    per segment: (3, degree + 1, n). The Taylor series of sin / cos about
    the centre, re-expanded in u by the binomial shift."""
    h = np.asarray(h, float)
    D = degree
    m = np.arange(D + 1)
    # Series in xi: sin k xi = sum_odd (-1)^((m-1)/2) k^m xi^m / m!,
    # cos k xi - 1 = sum_even>=2 (-1)^(m/2) k^m xi^m / m!.
    import math

    fact = np.array([math.factorial(i) for i in m], dtype=float)
    k_pow = k**m / fact
    s_xi = np.where(m % 2 == 1, np.where((m // 2) % 2 == 0, 1.0, -1.0), 0.0) * k_pow
    c_xi = (
        np.where((m % 2 == 0) & (m > 0), np.where((m // 2) % 2 == 0, 1.0, -1.0), 0.0)
        * k_pow
    )
    one = np.zeros(D + 1)
    one[0] = 1.0
    # xi^m = (u - h/2)^m = sum_j C(m, j) u^j (-h/2)^(m-j)
    binom = np.array(
        [[math.comb(mm, j) if j <= mm else 0 for j in m] for mm in m], dtype=float
    )
    shift = (-0.5 * h)[None, None, :] ** np.maximum(m[:, None] - m[None, :], 0)[
        :, :, None
    ]
    T = binom[:, :, None] * shift  # (m, j, n): xi^m -> sum_j T[m, j] u^j
    out = np.empty((N_SHAPES, D + 1, h.size), dtype=np.result_type(k, 1.0))
    for p, series in enumerate((one, s_xi, c_xi)):
        out[p] = np.einsum("m,mjn->jn", series, T)
    return out


def scatter_pair_moments(G, starts, jbasis, coef, I, J, dJ, scale):
    """`G[i, j] += scale . c_i^T dJ[n] c_j` over every (entry on I[n], entry
    on J[n]) of the listed segment pairs, one `add.at` for the whole list."""
    starts = np.asarray(starts)
    jbasis = np.asarray(jbasis)
    cnt = np.diff(starts)
    nI, nJ = cnt[I], cnt[J]
    reps = nI * nJ
    pair = np.repeat(np.arange(I.size), reps)
    local = np.arange(pair.size) - np.repeat(np.cumsum(reps) - reps, reps)
    eI = starts[I][pair] + local // nJ[pair]
    eJ = starts[J][pair] + local % nJ[pair]
    val = np.einsum("ep,epP,eP->e", coef[eI], dJ[pair], coef[eJ])
    np.add.at(G, (jbasis[eI], jbasis[eJ]), scale * val)


def remainder_raised_pairs(nodes, seg_l, seg_r, gz, q, n_seg):
    """The segment pairs whose field-form remainder order the geometry raises
    above `q`, as bspline lists them (`_remainder_qp_pairs`: the pair rule,
    symmetrised, dilated to touching neighbours): `(I, J, Q)`."""
    from . import _quadrature, bspline

    I, J, Qp = _quadrature.remainder_qp_pairs(
        nodes, seg_l, seg_r, gz, q, bspline._REMAINDER_QP_CAP, bspline._REMAINDER_QP_C
    )
    if I.size == 0:
        return I, J, Qp
    I, J = np.concatenate([I, J]), np.concatenate([J, I])
    Qp = np.concatenate([Qp, Qp])
    ptr, idx = bspline._segment_touch_lists(seg_l, seg_r)
    k1, ii = bspline._csr_expand(ptr, idx, I)
    I, J, Qp = bspline._pair_max(ii, J[k1], Qp[k1], n_seg)
    k2, jj = bspline._csr_expand(ptr, idx, J)
    return bspline._pair_max(I[k2], jj, Qp[k2], n_seg)


def remainder_Q_above(
    G,
    starts,
    jbasis,
    coef,
    seg_l,
    seg_r,
    tang,
    h,
    gz,
    k,
    grid,
    *,
    base_q,
    scale,
    cancel_flag=0,
    checkpoint=None,
):
    """Accumulate `scale` times the field-form remainder block
    Q[i, j] = int int f_i f_j t_i . F(r, r') . t_j into G, the B-spline fill's
    route on this basis: every pair at the base order through the fused C++
    kernel (`sommerfeld_remainder_bspline_Q`, symmetric route), then the
    grazing pairs `_quadrature.remainder_qp_pairs` lists re-integrated on
    `_remainder_graded`'s panels — their monomial moments to
    `REMAINDER_TAYLOR_DEGREE`, contracted with the shapes' Taylor
    coefficients — minus the base-order moments they replace."""
    from . import _quadrature, _remainder_graded, _sommerfeld, bspline

    seg_l, seg_r = np.asarray(seg_l, float), np.asarray(seg_r, float)
    tang, h = np.asarray(tang, float), np.asarray(h, float)
    n_seg = seg_l.shape[0]
    q = int(base_q)
    t01, w01 = _gl01(q)
    nodes = np.ascontiguousarray(
        seg_l[:, None, :] + t01[None, :, None] * (seg_r - seg_l)[:, None, :]
    )
    W = remainder_shape_weights(k, h, t01, w01)
    coef = np.asarray(coef)
    if np.abs(coef.imag).max() != 0.0:
        raise ValueError("remainder_Q_above needs real basis coefficients (a real k)")
    rows_basis, loc, pl = basis_wings(starts, jbasis, coef)
    pl = np.ascontiguousarray(pl.real)
    tang_c = np.ascontiguousarray(tang)
    Qr = _acc.sommerfeld_remainder_bspline_Q(
        nodes,
        tang_c,
        W,
        nodes,
        tang_c,
        W,
        loc,
        pl,
        loc,
        pl,
        float(gz),
        float(k),
        *_sommerfeld.grid_cpp_args(grid),
        int(cancel_flag),
    )
    n_basis = G.shape[0]
    S = scipy.sparse.csr_matrix(
        (np.ones(rows_basis.size), (rows_basis, np.arange(rows_basis.size))),
        shape=(n_basis, rows_basis.size),
    )
    G += scale * np.asarray(S @ np.asarray(S @ Qr).T).T
    del Qr

    # The grazing pairs, on graded panels (momwire#1189 / #1201 as bspline
    # lists and dilates them).
    I, J, _Qp = _quadrature.remainder_qp_pairs(
        nodes,
        seg_l,
        seg_r,
        gz,
        q,
        bspline._REMAINDER_QP_CAP,
        bspline._REMAINDER_GRADED_C,
        edge_tol=bspline._REMAINDER_GRADED_EDGE_TOL,
    )
    if I.size == 0:
        return
    I, J = np.concatenate([I, J]), np.concatenate([J, I])
    Qp = np.concatenate([_Qp, _Qp])
    ptr, idx = bspline._segment_touch_lists(seg_l, seg_r)
    k1, ii = bspline._csr_expand(ptr, idx, I)
    I, J, Qp = bspline._pair_max(ii, J[k1], Qp[k1], n_seg)
    k2, jj = bspline._csr_expand(ptr, idx, J)
    I, J, Qp = bspline._pair_max(I[k2], jj, Qp[k2], n_seg)
    D1 = REMAINDER_TAYLOR_DEGREE + 1
    dJ_mono = _remainder_graded.pair_moments(
        seg_l,
        seg_r,
        tang,
        h,
        I,
        J,
        gz,
        k,
        grid,
        D1,
        cancel_flag=cancel_flag,
        checkpoint=checkpoint,
    )  # (n, D1, D1): int int u^m F u'^M
    tau = taylor_shape_coefs(k, h)  # (3, D1, n_seg)
    dJ = np.einsum("pmn,nmM,PMn->npP", tau[:, :, I], dJ_mono, tau[:, :, J])
    # The base-order moments these pairs were filled with.
    obs = np.repeat(nodes[I], q, axis=1).reshape(-1, 3)  # (n, q, q) -> node pairs
    src = np.tile(nodes[J], (1, q, 1)).reshape(-1, 3)
    owner = np.repeat(np.arange(I.size), q * q)
    F = _sommerfeld.remainder_field_proj_owned(
        obs,
        np.repeat(tang[I], q * q, axis=0),
        src,
        np.repeat(tang[J], q * q, axis=0),
        owner,
        gz,
        k,
        grid,
        cancel_flag,
    ).reshape(I.size, q, q)
    lo = np.einsum("pnq,nqr,Pnr->npP", W[:, I, :], F, W[:, J, :])
    dJ -= lo
    scatter_pair_moments(G, starts, jbasis, coef, I, J, dJ, scale)


# ----------------------------------------------------------------------
# The field-form remainder blocks (below the plane) on `assemble_field_galerkin`
# ----------------------------------------------------------------------

_HAVE_FIELD_MOMENTS = _acc is not None and hasattr(
    _acc, "field_pair_moments_sinusoidal"
)


def field_pair_moments(F, W_obs, W_src):
    """`Jf[p, P, i, j] = sum_{q, r} W_obs[p, i, q] F[i q, j r] W_src[P, j, r]`,
    the field-form pair moments of a projected table (3, 3, n_obs, n_src);
    C++ when built, einsum otherwise."""
    if _HAVE_FIELD_MOMENTS:
        return _acc.field_pair_moments_sinusoidal(
            np.ascontiguousarray(F, dtype=np.complex128),
            np.ascontiguousarray(W_obs, dtype=np.complex128),
            np.ascontiguousarray(W_src, dtype=np.complex128),
        )
    n_obs, q = W_obs.shape[1], W_obs.shape[2]
    n_src = W_src.shape[1]
    fq = np.asarray(F).reshape(n_obs, q, n_src, q)
    return np.einsum("piq,iqjr,Pjr->pPij", W_obs, fq, W_src, optimize=True)


def remainder_Q_field(
    G,
    starts,
    jbasis,
    coef,
    seg_l,
    seg_r,
    tang,
    h,
    segs,
    k,
    q,
    proj_fn,
    *,
    gz,
    scale,
    cancel_flag=0,
    checkpoint=None,
):
    """Accumulate `scale` times the field-form block
    Q[i, j] = int int f_i f_j t_i . F(r, r') . t_j over the segments `segs`
    (global ids; `seg_l` .. `h` are the class's own arrays in that order)
    into G: bspline's below/below remainder route — `proj_fn(obs, t_obs,
    src, t_src)` its projected field table at `q` Gauss nodes a segment, one
    observer chunk at a time — with the pair moments taken against the
    (complex) shape weights and assembled through the fill's own windowed
    assembler, the tangent dot already inside F."""
    from . import _below_interface

    segs = np.asarray(segs, dtype=np.int64)
    nodes, t_nodes, u_phys, w_node = _below_interface.field_nodes(
        np.asarray(seg_l, float),
        np.asarray(seg_r, float),
        np.asarray(tang, float),
        np.asarray(h, float),
        int(q),
    )
    S = shape_values(k, u_phys - 0.5 * np.asarray(h, float)[:, None])  # (3, n, q)
    W = np.ascontiguousarray(w_node[None] * S, dtype=np.complex128)
    unit = np.zeros((segs.size, 3))
    unit[:, 0] = 1.0  # F carries the tangent projection: td = 1
    zero_d = np.zeros_like(np.asarray(coef))
    n_src = segs.size
    chunk = max(1, (1 << 19) // max(n_src * q * q, 1))
    for i0 in range(0, segs.size, chunk):
        if checkpoint is not None:
            checkpoint()
        i1 = min(i0 + chunk, segs.size)
        proj = proj_fn(nodes[i0 * q : i1 * q], t_nodes[i0 * q : i1 * q], nodes, t_nodes)
        J = field_pair_moments(proj, W[:, i0:i1], W)
        assemble_window(
            G,
            J,
            segs[i0:i1],
            segs,
            starts,
            jbasis,
            coef,
            zero_d,
            unit[i0:i1],
            unit,
            scale,
            0.0,
        )
    # The pairs the geometry raises above the base order (a segment near the
    # plane against another: the kernel's spike sits at the image foot, and q
    # nodes cannot see it — momwire#1189's rule, which bspline applies above
    # the plane and this fill applies in the medium too), re-integrated at
    # their own order minus what the base fill gave them.
    seg_l, seg_r, tang, h = (np.asarray(x, float) for x in (seg_l, seg_r, tang, h))
    I, J, Qp = remainder_raised_pairs(
        nodes.reshape(segs.size, q, 3), seg_l, seg_r, gz, q, np.asarray(starts).size - 1
    )
    if I.size == 0:
        return
    # Listed pairs index the class list; `I` / `J` are positions in `segs`.
    loc = np.full(np.asarray(starts).size - 1, -1, dtype=np.int64)
    loc[segs] = np.arange(segs.size)
    I_loc, J_loc = loc[I], loc[J]
    keep = (I_loc >= 0) & (J_loc >= 0)
    I_loc, J_loc, Qp = I_loc[keep], J_loc[keep], Qp[keep]
    dJ = np.zeros((I_loc.size, N_SHAPES, N_SHAPES), dtype=np.complex128)

    def _moments(obs_segs, src_segs, qq):
        nd, tn, up, wn = _below_interface.field_nodes(
            seg_l[obs_segs], seg_r[obs_segs], tang[obs_segs], h[obs_segs], int(qq)
        )
        nd_s, tn_s, up_s, wn_s = _below_interface.field_nodes(
            seg_l[src_segs], seg_r[src_segs], tang[src_segs], h[src_segs], int(qq)
        )
        Wo = wn[None] * shape_values(k, up - 0.5 * h[obs_segs][:, None])
        Wsrc = wn_s[None] * shape_values(k, up_s - 0.5 * h[src_segs][:, None])
        F = proj_fn(nd, tn, nd_s, tn_s)
        return field_pair_moments(F, Wo, Wsrc)  # (3, 3, n_obs, n_src)

    for qq in np.unique(Qp):
        sel = np.flatnonzero(Qp == qq)
        uI, iI = np.unique(I_loc[sel], return_inverse=True)
        uJ, iJ = np.unique(J_loc[sel], return_inverse=True)
        if checkpoint is not None:
            checkpoint()
        hi = _moments(uI, uJ, qq)
        lo = _moments(uI, uJ, q)
        dJ[sel] = np.moveaxis((hi - lo)[:, :, iI, iJ], -1, 0)
    scatter_pair_moments(G, starts, jbasis, coef, segs[I_loc], segs[J_loc], dJ, scale)
