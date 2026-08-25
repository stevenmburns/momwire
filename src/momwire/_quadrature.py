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
    Per-segment orders would remove that coupling and are the follow-up.
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
