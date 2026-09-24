"""Cached Gauss-Legendre quadrature nodes.

`numpy.polynomial.legendre.leggauss(n)` computes its nodes/weights via an
eigendecomposition of the Jacobi companion matrix — not free. The MoM
kernels call it once per same-edge block per wavenumber across a swept-k
solve (hundreds of times for a 41-point sweep), always with the same handful
of `n` values. Memoize on `n`.

The cached arrays are marked read-only so a shared entry can never be mutated
by a caller; every kernel here only ever reads them (e.g. ``0.5 * (xi + 1)``),
which allocates fresh arrays.
"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _leggauss_cached(n: int):
    xi, w = np.polynomial.legendre.leggauss(n)
    xi.setflags(write=False)
    w.setflags(write=False)
    return xi, w


def leggauss(n):
    """Memoized `numpy.polynomial.legendre.leggauss`. Returns read-only
    `(nodes, weights)` arrays — do not mutate; derive new arrays instead."""
    return _leggauss_cached(int(n))


def remainder_qp(obs_pts, src_l, src_r, ground_z, base, cap, c):
    """The Sommerfeld remainder's source order, keyed to grazing height.

    momwire#510 for razor, momwire#631 for bspline — one rule, because it is
    a statement about GEOMETRY and not about either formulation. Both trunks
    lay a single Gauss rule of one order over every source segment, and both
    constructors default it low on the premise that the remainder field is
    "smooth on the scale of a segment". That premise is true wherever either
    unit was gated and false at grazing: when an observer sits almost
    directly over a source segment's IMAGE, the projected remainder carries a
    spike of width ~R_min in a segment of length `len`, and a handful of
    points cannot see a feature of relative width R_min/len.

    So the order is keyed to the geometry the same way momwire#443 keyed the
    interpolation grid to its boundary layer: `ceil(c · len / R_min)`, with
    `R_min` the nearest any observer comes to the segment's mirror, clipped
    below by `base` and above by `cap`.

    `cap` and `c` are REQUIRED rather than defaulted from module constants:
    each trunk owns its own, a default argument would bind whichever module
    this lives in at import time, and the gate that pins pre-#510 behaviour
    patches its trunk's constant and must still be able to move it.

    Two properties matter as much as the rule.

    **A deck with nothing grazing is bit-identical.** Every ratio comes out
    below 1 and the clip returns `base` exactly, so the order is the number
    it always was and no shipped gate moves. That is why this is a
    max-with-base rather than a replacement.

    **The cap is a real limit, not a formality.** The order is one scalar for
    the whole fill, so a single grazing pair raises it for every source
    segment; the cap is what stops one wire in a large model multiplying the
    remainder's cost without bound. A deck grazing enough to need more than
    `cap` is served MORE accurately than before but not to the binary.
    `remainder_qp_pairs` below is the same rule evaluated per observer/source
    segment pair (momwire#1189), which removes that coupling for any trunk
    that can fill pairs at different orders; this scalar is its maximum.
    """
    src_l = np.asarray(src_l, dtype=float)
    src_r = np.asarray(src_r, dtype=float)
    obs = np.asarray(obs_pts, dtype=float)
    if src_l.size == 0 or obs.size == 0:
        return int(base)

    # The mirror of each source segment, which is what an observer's distance
    # to the remainder's singular ridge is measured against.
    mir_l, mir_r = src_l.copy(), src_r.copy()
    mir_l[:, 2] = 2.0 * ground_z - src_l[:, 2]
    mir_r[:, 2] = 2.0 * ground_z - src_r[:, 2]

    d = mir_r - mir_l
    dd = np.einsum("ij,ij->i", d, d)
    lengths = np.linalg.norm(src_r - src_l, axis=1)

    # O(N) short-circuit before the O(N) numpy passes below, so a deck with
    # nothing near the plane pays nothing for a rule that cannot fire on it.
    # An observer sits at z_o and a mirror point at 2·ground_z - z_s, so their
    # separation is at least (z_o - ground_z) + (z_s - ground_z) >= 2·h_min,
    # giving worst <= max(len) / (2·h_min). When even that bound asks for no
    # more than `base` the clip would return `base` exactly, so this returns
    # the same integer the loop would — it is a speed path, not a policy.
    h_min = min(
        float(np.min(src_l[:, 2] - ground_z)), float(np.min(src_r[:, 2] - ground_z))
    )
    h_min = min(h_min, float(np.min(obs[:, 2] - ground_z)))
    if h_min > 0.0:
        bound = float(np.max(lengths)) / (2.0 * h_min)
        if int(np.ceil(c * bound)) <= int(base):
            return int(base)

    worst = 0.0
    # Per source segment rather than one (n_obs, n_src, 3) array: the observer
    # axis is a quadrature-point set, so the dense form is n_obs·n_src·3 and
    # would be hundreds of MB on a large deck for a number that is only used
    # to pick an integer.
    for j in range(src_l.shape[0]):
        if lengths[j] <= 0.0:
            continue
        ap = obs - mir_l[j]
        if dd[j] > 0.0:
            t_raw = ap @ d[j] / dd[j]
            t = np.clip(t_raw, 0.0, 1.0)
            closest = mir_l[j] + t[:, None] * d[j]
            # BROADSIDE approaches only. The spike this rule exists for is an
            # observer sitting ACROSS a source segment's image at a small
            # perpendicular offset — a feature of width R_min in a segment of
            # length `len`. When the perpendicular foot falls off the end of
            # the mirror the observer is approaching END-ON instead, along the
            # segment's own axis, and that is ordinary integrable 1/R
            # behaviour with no narrow feature to resolve.
            #
            # The distinction is not academic: a vertical wire ENDING in the
            # plane has a collinear mirror, so its lowest quadrature node sits
            # a hair from the mirror's endpoint and the raw ratio reads ~30.
            # Raising the order there is worse than useless — it walks the
            # nodes toward the plane, where the remainder is singular and the
            # interpolation grid is at the edge of its band — and it moved
            # four shipped contact gates when this guard was missing. It is
            # also why momwire#510's own account of contact ("the image is
            # collinear and there is no spike to miss") is exactly right.
            interior = (t_raw > 0.0) & (t_raw < 1.0)
            if not interior.any():
                continue
            r_min = float(
                np.min(np.linalg.norm(obs[interior] - closest[interior], axis=1))
            )
        else:
            continue  # a degenerate mirror has no broadside to approach
        if r_min <= 0.0:
            return int(cap)
        worst = max(worst, float(lengths[j]) / r_min)

    need = int(np.ceil(c * worst)) if worst > 0.0 else 0
    return int(min(max(int(base), need), int(cap)))


def remainder_qp_pairs(obs_nodes, src_l, src_r, ground_z, base, cap, c):
    """`remainder_qp` evaluated per (observer segment, source segment) pair.

    momwire#1189. `remainder_qp` returns ONE order for the whole fill, keyed
    to the worst pair in it, so one wire near the plane raised every pair to
    that order: on 4nec2's GndScreen (16 radials at 1.8 mm) every one of the
    ~200k segment pairs ran at q = 192 for the sake of the ~400 radial self
    pairs, and the remainder took 529 of a 547 s solve.

    This is the same geometry, the same broadside-only guard and the same
    `ceil(c · len / R_min)` clipped to `[base, cap]` — only the reduction
    changes: `R_min` is taken over the observer nodes of ONE observer segment
    against the mirror of ONE source segment, instead of over every observer
    node in the deck. So by construction the scalar rule is the maximum of
    this one over all pairs (and `base` when nothing is returned).

    `obs_nodes` is `(n_obs_seg, n_node, 3)`: the observer points grouped by
    the segment they belong to. Returns `(I, J, Q)`, int64 arrays listing
    ONLY the pairs whose order is above `base` (sorted by `I` then `J`);
    every pair not listed takes `base`. A deck with nothing grazing returns
    three empty arrays, through the same O(N) short-circuit as the scalar
    rule, and that is what keeps its fill bit-identical.
    """
    empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int64))
    src_l = np.asarray(src_l, dtype=float)
    src_r = np.asarray(src_r, dtype=float)
    obs3 = np.asarray(obs_nodes, dtype=float)
    if src_l.size == 0 or obs3.size == 0:
        return empty
    n_obs_seg, n_node = obs3.shape[0], obs3.shape[1]
    obs = obs3.reshape(-1, 3)
    base, cap = int(base), int(cap)

    mir_l, mir_r = src_l.copy(), src_r.copy()
    mir_l[:, 2] = 2.0 * ground_z - src_l[:, 2]
    mir_r[:, 2] = 2.0 * ground_z - src_r[:, 2]
    d = mir_r - mir_l
    dd = np.einsum("ij,ij->i", d, d)
    lengths = np.linalg.norm(src_r - src_l, axis=1)

    # The scalar rule's short-circuit, unchanged: if even the deck-wide bound
    # asks for no more than `base`, no pair can.
    h_min = min(
        float(np.min(src_l[:, 2] - ground_z)), float(np.min(src_r[:, 2] - ground_z))
    )
    h_min = min(h_min, float(np.min(obs[:, 2] - ground_z)))
    if h_min > 0.0:
        bound = float(np.max(lengths)) / (2.0 * h_min)
        if int(np.ceil(c * bound)) <= base:
            return empty

    I_out, J_out, Q_out = [], [], []
    for j in range(src_l.shape[0]):
        if lengths[j] <= 0.0 or not dd[j] > 0.0:
            continue  # the scalar rule skips both the same way
        ap = obs - mir_l[j]
        t_raw = ap @ d[j] / dd[j]
        # The same broadside-only guard as `remainder_qp` (see there for why
        # an end-on approach is not the spike this rule resolves).
        interior = (t_raw > 0.0) & (t_raw < 1.0)
        if not interior.any():
            continue
        t = np.clip(t_raw, 0.0, 1.0)
        closest = mir_l[j] + t[:, None] * d[j]
        dist = np.where(interior, np.linalg.norm(obs - closest, axis=1), np.inf)
        r_min = dist.reshape(n_obs_seg, n_node).min(axis=1)
        hit = np.isfinite(r_min)
        with np.errstate(divide="ignore"):
            ratio = np.where(hit & (r_min > 0.0), lengths[j] / r_min, 0.0)
        need = np.where(ratio > 0.0, np.ceil(c * ratio), 0.0)
        q = np.minimum(np.maximum(need, base), cap).astype(np.int64)
        # R_min == 0 is the scalar rule's "return cap" case.
        q[hit & (r_min <= 0.0)] = cap
        rows = np.flatnonzero(q > base)
        if rows.size:
            I_out.append(rows.astype(np.int64))
            J_out.append(np.full(rows.size, j, dtype=np.int64))
            Q_out.append(q[rows])
    if not I_out:
        return empty
    I = np.concatenate(I_out)
    J = np.concatenate(J_out)
    Q = np.concatenate(Q_out)
    order = np.lexsort((J, I))
    return I[order], J[order], Q[order]
