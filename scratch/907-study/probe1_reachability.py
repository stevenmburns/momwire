"""#907 probe 1: is the free-space ladder even REACHABLE?

Free space base order is 8 and the proposed only tier is (16.0, 4). The
phase guard drops every tier below `_LADDER_PHASE_LIMITED_BELOW` = 8 when the
block's longest segment passes kL = 0.5. So the proposed free-space ladder is
ENTIRELY dropped on any block with kL > 0.5 -- there is no order-8 tier left
to fall back on, because 8 IS the base.

This probe measures, per deck: the kL of each off-edge block, whether the
guard keeps or drops the tier, and the share of pairs at ratio >= 16.
"""

import numpy as np

import momwire.bspline as _bs
from momwire._bspline_kernels import (
    _LADDER_PHASE_KL_CEILING,
    _LADDER_PHASE_LIMITED_BELOW,
    _ladder_for_block,
    _pair_ratio,
)
from momwire.bspline import BSplineSolver

LADDER = ((16.0, 4),)


def bent_deck(n_per=100, f=None):
    """A bent multi-edge free-space deck: 4 edges, n_per segments each."""
    w = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    return dict(
        wires=[w],
        n_per_edge_per_wire=[[n_per] * 4],
        feeds=[(0, 0.5, 1 + 0j)],
        **({} if f is None else dict(f=f)),
    )


def probe(name, deck):
    s = BSplineSolver(**deck)
    seen = []
    real = _bs._seg_seg_full_moments_offedge

    def spy(seg_l_i, seg_r_i, seg_l_j, seg_r_j, a, k, max_d, n_qp, **kw):
        len_max = max(
            float(
                np.linalg.norm(np.asarray(seg_r_i) - np.asarray(seg_l_i), axis=1).max()
            ),
            float(
                np.linalg.norm(np.asarray(seg_r_j) - np.asarray(seg_l_j), axis=1).max()
            ),
        )
        kept = _ladder_for_block(LADDER, k, seg_l_i, seg_r_i, seg_l_j, seg_r_j)
        r = _pair_ratio(seg_l_i, seg_r_i, seg_l_j, seg_r_j)
        seen.append((abs(k) * len_max, bool(kept), r.size, int((r >= 16.0).sum())))
        return real(seg_l_i, seg_r_i, seg_l_j, seg_r_j, a, k, max_d, n_qp, **kw)

    _bs._seg_seg_full_moments_offedge = spy
    try:
        z, _ = s.compute_impedance()
    finally:
        _bs._seg_seg_full_moments_offedge = real

    if not seen:
        print(f"{name}: NO off-edge blocks at all")
        return
    kl = np.array([x[0] for x in seen])
    kept = np.array([x[1] for x in seen])
    tot = sum(x[2] for x in seen)
    far = sum(x[3] for x in seen)
    far_kept = sum(x[3] for x in seen if x[1])
    print(f"{name}:  Z={z:.6f}")
    print(
        f"   off-edge blocks {len(seen)}   kL min/med/max "
        f"{kl.min():.3f}/{np.median(kl):.3f}/{kl.max():.3f}   ceiling {_LADDER_PHASE_KL_CEILING}"
    )
    print(
        f"   blocks KEEPING the tier: {kept.sum()}/{len(seen)}  "
        f"({100 * kept.mean():.1f}%)"
    )
    print(
        f"   pairs total {tot:,}   at ratio>=16 {far:,} ({100 * far / tot:.1f}%)   "
        f"of those, in kept blocks {far_kept:,} ({100 * far_kept / max(tot, 1):.1f}% of all pairs)"
    )


print(
    f"phase guard: tiers with n_qp < {_LADDER_PHASE_LIMITED_BELOW} dropped when "
    f"kL > {_LADDER_PHASE_KL_CEILING}"
)
print(
    f"free-space base is 8, proposed tier is {LADDER} -> the ONLY tier is "
    f"phase-limited\n"
)

# lambda at f: seg length L = 1.0/n_per metres per edge.
for n_per in (10, 20, 40, 100):
    probe(f"bent 4-edge, {n_per}/edge ({4 * n_per} seg)", bent_deck(n_per))
