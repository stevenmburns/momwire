"""#907 probe 3: does ONE coarse segment disable the ladder deck-wide?

The phase guard is per BLOCK and keys on the block's LONGEST segment; probe 2
showed the off-edge fill takes the whole deck as a single block. So a deck
that is finely meshed everywhere except one coarse wire should lose the tier
for ALL of its pairs -- the realistic shape being a fine radiator plus a
coarsely modelled support or boom.
"""

import numpy as np

import momwire.bspline as _bs
from momwire._bspline_kernels import _ladder_for_block, _pair_ratio
from momwire.bspline import BSplineSolver

LADDER = ((16.0, 4),)


def loop_plus_strut(n_per_edge=100, side=0.25, strut_segs=1, strut_len=0.3):
    """A finely meshed square loop plus one coarsely meshed straight wire."""
    w = np.array(
        [
            [0.0, 0.0, 0.0],
            [side, 0.0, 0.0],
            [side, side, 0.0],
            [0.0, side, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    strut = np.array([[0.0, 0.0, -0.4], [0.0, strut_len, -0.4]])
    return dict(
        wires=[w, strut],
        n_per_edge_per_wire=[[n_per_edge] * 4, [strut_segs]],
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=1.0,
        wire_radius=1e-4,
    )


def census(name, deck):
    s = BSplineSolver(**deck)
    seen = []
    real = _bs._seg_seg_full_moments_offedge

    def spy(sli, sri, slj, srj, a, k, max_d, n_qp, **kw):
        lm = max(
            float(np.linalg.norm(np.asarray(sri) - np.asarray(sli), axis=1).max()),
            float(np.linalg.norm(np.asarray(srj) - np.asarray(slj), axis=1).max()),
        )
        kept = _ladder_for_block(LADDER, k, sli, sri, slj, srj)
        r = _pair_ratio(sli, sri, slj, srj)
        seen.append((abs(k) * lm, bool(kept), r.size, int((r >= 16.0).sum())))
        return real(sli, sri, slj, srj, a, k, max_d, n_qp, **kw)

    _bs._seg_seg_full_moments_offedge = spy
    try:
        s.compute_impedance()
    finally:
        _bs._seg_seg_full_moments_offedge = real

    tot = sum(x[2] for x in seen)
    far = sum(x[3] for x in seen)
    far_kept = sum(x[3] for x in seen if x[1])
    kl = max(x[0] for x in seen)
    print(
        f"{name}\n   blocks {len(seen)}   kL max {kl:.4f}   "
        f"pairs {tot:,}  ratio>=16 {far:,} ({100 * far / tot:.1f}%)  "
        f"SERVED at 4 {far_kept:,} ({100 * far_kept / tot:.1f}%)"
    )


# strut_len 0.3 lambda in ONE segment -> kL = 1.88, far over the 0.5 ceiling.
census(
    "loop(400 fine) + strut as 1 seg of 0.3 lam  [kL=1.88]",
    loop_plus_strut(strut_segs=1, strut_len=0.3),
)
# the same strut meshed at lambda/20 -> kL = 0.094, under the ceiling.
census(
    "loop(400 fine) + SAME strut meshed at lam/20 (6 seg)",
    loop_plus_strut(strut_segs=6, strut_len=0.3),
)
