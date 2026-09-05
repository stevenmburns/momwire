"""#907 probe 2: realistic free-space decks -- reachability and pair census.

Decks are sized ELECTRICALLY (wavelength=1.0, so a length is in wavelengths)
because the phase guard is a kL test: it drops the order-4 tier -- which in
free space is the ONLY tier, since the base is already 8 -- whenever a block's
longest segment exceeds kL = 0.5, i.e. L > lambda/12.566.
"""

import numpy as np

import momwire.bspline as _bs
from momwire._bspline_kernels import _ladder_for_block, _pair_ratio
from momwire.bspline import BSplineSolver

LADDER = ((16.0, 4),)
LAM = 1.0


def square_loop(n_per_edge, side=0.25):
    """Bent multi-edge deck: a closed square loop, perimeter 4*side lambda."""
    w = np.array(
        [
            [0.0, 0.0, 0.0],
            [side, 0.0, 0.0],
            [side, side, 0.0],
            [0.0, side, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    return dict(
        wires=[w],
        n_per_edge_per_wire=[[n_per_edge] * 4],
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=LAM,
        wire_radius=1e-4,
    )


def vee(n_per_edge, arm=0.25):
    """Bent 2-edge deck (a vee dipole), arms of `arm` lambda."""
    w = np.array(
        [
            [-arm * 0.707, 0.0, arm * 0.707],
            [0.0, 0.0, 0.0],
            [arm * 0.707, 0.0, arm * 0.707],
        ]
    )
    return dict(
        wires=[w],
        n_per_edge_per_wire=[[n_per_edge] * 2],
        feeds=[(0, arm, 1 + 0j)],
        wavelength=LAM,
        wire_radius=1e-4,
    )


def yagi(n_elem=5, n_seg=40, spacing=0.2, half=0.24):
    """Multi-wire deck: n_elem parallel half-wave dipoles."""
    (
        wires,
        nppw,
    ) = [], []
    for i in range(n_elem):
        x = i * spacing
        wires.append(np.array([[x, -half, 0.0], [x, half, 0.0]]))
        nppw.append([n_seg])
    return dict(
        wires=wires,
        n_per_edge_per_wire=nppw,
        feeds=[(0, half, 1 + 0j)],
        wavelength=LAM,
        wire_radius=1e-4,
    )


def census(name, deck, ladder=LADDER):
    s = BSplineSolver(**deck)
    seen = []
    real = _bs._seg_seg_full_moments_offedge

    def spy(sli, sri, slj, srj, a, k, max_d, n_qp, **kw):
        lm = max(
            float(np.linalg.norm(np.asarray(sri) - np.asarray(sli), axis=1).max()),
            float(np.linalg.norm(np.asarray(srj) - np.asarray(slj), axis=1).max()),
        )
        kept = _ladder_for_block(ladder, k, sli, sri, slj, srj)
        r = _pair_ratio(sli, sri, slj, srj)
        seen.append(
            (abs(k) * lm, bool(kept), r.size, int((r >= 16.0).sum()), int(n_qp))
        )
        return real(sli, sri, slj, srj, a, k, max_d, n_qp, **kw)

    _bs._seg_seg_full_moments_offedge = spy
    try:
        z, _ = s.compute_impedance()
    finally:
        _bs._seg_seg_full_moments_offedge = real

    if not seen:
        print(f"{name}: NO off-edge blocks")
        return
    kl = np.array([x[0] for x in seen])
    kept = np.array([x[1] for x in seen])
    tot = sum(x[2] for x in seen)
    far = sum(x[3] for x in seen)
    far_kept = sum(x[3] for x in seen if x[1])
    print(f"{name}")
    print(f"   Z={z:.6f}   base n_qp={seen[0][4]}   off-edge blocks={len(seen)}")
    print(
        f"   kL max {kl.max():.4f}  (ceiling 0.5)   blocks keeping the tier "
        f"{kept.sum()}/{len(seen)}"
    )
    print(
        f"   pairs {tot:,}   ratio>=16 {far:,} ({100 * far / tot:.1f}%)   "
        f"SERVED at order 4 {far_kept:,} ({100 * far_kept / tot:.1f}%)"
    )


if __name__ == "__main__":
    print("=== realistic meshes (kL well under the 0.5 ceiling) ===")
    census("square loop 1.0 lambda, 100/edge (400 seg)", square_loop(100))
    census("square loop 1.0 lambda, 25/edge (100 seg)", square_loop(25))
    census("vee dipole, 100/arm (200 seg)", vee(100))
    census("yagi 5 elem x 40 seg (200 seg)", yagi())

    print("\n=== coarse: where the guard actually fires (L > lambda/12.57) ===")
    census("square loop 1.0 lambda, 3/edge (12 seg, kL=0.52)", square_loop(3))
    census("vee dipole, 4/arm (8 seg)", vee(4))
