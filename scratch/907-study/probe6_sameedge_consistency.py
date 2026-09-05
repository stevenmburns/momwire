"""#907 probe 6: the same-edge correction must use the SWEEP's ladder.

The chunked fills add a full off-edge block for every pair, then SUBTRACT it
back on same-edge blocks (`corr = (A_st + A_reg) - J_edge`). That subtraction
only cancels if J_edge is filled with exactly the arithmetic the sweep used.

But the phase guard resolves per block on the block's LONGEST segment:
  * the sweep window is seg[i0:i1] x seg[ALL]  -> sees the GLOBAL max
  * the correction window is seg[sl] x seg[sl] -> sees ONE EDGE's max
so a deck whose global max trips kL = 0.5 while one edge's max does not gets
an order-8 term added and an order-4 term subtracted on that edge's far pairs.

This probe reports the effective ladder per window on such a deck.
"""

import traceback

import numpy as np

import momwire.bspline as _bs
from momwire._bspline_kernels import _ladder_for_block, _normalize_ladder, _pair_ratio
from momwire.bspline import BSplineSolver


def mixed_mesh_deck(**over):
    """A finely meshed long wire (40 seg over 0.5 lam -> ratios past 16 inside
    the edge) plus a second wire carried as ONE coarse 0.3-lam segment."""
    fine = np.array([[0.0, -0.25, 0.0], [0.0, 0.25, 0.0]])
    coarse = np.array([[0.35, 0.0, 0.0], [0.35, 0.3, 0.0]])
    d = dict(
        wires=[fine, coarse],
        n_per_edge_per_wire=[[200], [1]],
        feeds=[(0, 0.25, 1 + 0j)],
        wavelength=1.0,
        wire_radius=1e-4,
    )
    d.update(over)
    return d


def windows(name, deck, ladder):
    s = BSplineSolver(**deck)
    lad = _normalize_ladder(ladder, s.n_qp_pair)
    print(f"\n{name}\n   base n_qp={s.n_qp_pair}  ladder={lad}")
    rows = []
    real = _bs._seg_seg_full_moments_offedge

    def spy(sli, sri, slj, srj, a, k, max_d, n_qp, **kw):
        eff = _ladder_for_block(lad, k, sli, sri, slj, srj)
        lm = max(
            float(np.linalg.norm(np.asarray(sri) - np.asarray(sli), axis=1).max()),
            float(np.linalg.norm(np.asarray(srj) - np.asarray(slj), axis=1).max()),
        )
        r = _pair_ratio(sli, sri, slj, srj)
        st = traceback.extract_stack()[-2]
        rows.append(
            (
                f"{st.name}:{st.lineno}",
                len(sli),
                len(slj),
                abs(k) * lm,
                eff,
                int((r >= 16.0).sum()),
            )
        )
        return real(sli, sri, slj, srj, a, k, max_d, n_qp, **kw)

    _bs._seg_seg_full_moments_offedge = spy
    try:
        s.compute_impedance()
    finally:
        _bs._seg_seg_full_moments_offedge = real

    for kind, ni, nj, kl, eff, far in rows:
        print(
            f"   {kind:34s} {ni:4d}x{nj:4d}  kL={kl:.3f}  far_pairs={far:6d}  "
            f"effective ladder={eff}"
        )
    effs = {r[4] for r in rows if r[5] > 0}
    print(f"   -> distinct effective ladders over windows WITH far pairs: {effs}")
    if len(effs) > 1:
        print("   *** MISMATCH: sweep and correction disagree on those pairs ***")


# free space, base 8, the ladder this issue proposes
windows("free space, mixed mesh, DENSE", mixed_mesh_deck(), ((16.0, 4),))
windows(
    "free space, mixed mesh, CHUNKED (swept_mem_mb=1)",
    mixed_mesh_deck(swept_mem_mb=1),
    ((16.0, 4),),
)
