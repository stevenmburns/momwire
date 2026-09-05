"""#907 probe 7: is the same-edge/sweep ladder mismatch LIVE on main?

Free space does not pass the ladder yet, so there the mismatch is latent. The
BURIED subset path (momwire#906) passes `ladder=self.pair_order_ladder` to
both its sweep (bspline.py:5164) and its same-edge correction (:5201). If a
buried deck's windows straddle the kL = 0.5 guard, the bug is live on main.

Buried base is 32 and the ladder is ((2,8),(16,4)); only the order-4 tier is
phase-limited, so the mismatch needs far pairs (ratio >= 16) inside a
same-edge block plus a straddle across windows.
"""

import traceback

import numpy as np

import momwire.bspline as _bs
from momwire._bspline_kernels import _ladder_for_block, _normalize_ladder, _pair_ratio
from momwire.bspline import BSplineSolver


def buried_mixed(**over):
    """A finely meshed buried radial (ratios past 16 inside its own edge)
    plus a coarsely meshed buried wire that trips the guard for the sweep."""
    fine = np.array([[0.0, 0.0, -0.05], [0.5, 0.0, -0.05]])
    coarse = np.array([[0.0, 0.3, -0.05], [0.3, 0.3, -0.05]])
    d = dict(
        wires=[fine, coarse],
        n_per_edge_per_wire=[[60], [1]],
        feeds=[(0, 0.25, 1 + 0j)],
        wavelength=1.0,
        wire_radius=1e-4,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )
    d.update(over)
    return d


def windows(name, deck):
    s = BSplineSolver(**deck)
    lad = _normalize_ladder(s.pair_order_ladder, s.n_qp_pair)
    print(f"\n{name}\n   base n_qp={s.n_qp_pair}   resolved ladder={lad}")
    rows = []
    real = _bs._seg_seg_full_moments_offedge

    def spy(sli, sri, slj, srj, a, k, max_d, n_qp, **kw):
        passed = kw.get("ladder", None)
        eff = _ladder_for_block(_normalize_ladder(passed, n_qp), k, sli, sri, slj, srj)
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
        z, _ = s.compute_impedance()
    finally:
        _bs._seg_seg_full_moments_offedge = real

    seen = {}
    for kind, ni, nj, kl, eff, far in rows:
        key = (kind, ni, nj, round(kl, 4), eff, far > 0)
        seen[key] = seen.get(key, 0) + 1
    for (kind, ni, nj, kl, eff, hasfar), n in seen.items():
        print(
            f"   {kind:36s} {ni:4d}x{nj:4d} x{n:<3d} kL={kl:.3f} "
            f"far={'Y' if hasfar else 'n'}  eff={eff}"
        )
    effs = {r[4] for r in rows if r[5] > 0}
    print(f"   Z={z:.6f}")
    print(f"   -> distinct effective ladders over windows WITH far pairs: {effs}")
    print("   *** LIVE MISMATCH ***" if len(effs) > 1 else "   consistent")


windows("buried mixed mesh (ladder IS wired here on main)", buried_mixed())
