"""Where D2's near-plane collapse error lives — momwire#1004, phase 1.

#1004 measures a mixed deck (one wire above the plane, one below, NO junction)
at ε̃ = 1, where the interface is not there and the answer must reproduce the
same deck in free space. The relative miss grows from 2.3e-10 at a 0.5 m gap to
1.9e-07 at 5 mm, and #980 D3's zero-gap gate continues it to 1.4e-06 — flat
under refinement, so a floor rather than discretisation.

The issue's own words: "The suspect is the transmitted block's near-plane
behaviour (the grid's theta floor is at its finest exactly here), but that is a
guess — the measurement above is the evidence."

So this does not argue about it. `_assemble_Z` hands back the operator, and the
basis indices partition by wire, so the ε̃=1-minus-free-space DIFFERENCE splits
into the three pair classes the fill computes separately:

    above x above   at k_p          (should be untouched by the interface)
    below x below   at k_m -> k_p   (D1's serve, collapses by construction)
    cross           the transmitted block, both directions

Whichever block carries the growth as the gap closes is the answer to (a), and
it is read off rather than reasoned about. The two diagonal classes are the
control: if they are flat while the cross grows, the transmitted block is
implicated by measurement; if a diagonal grows too, the guess is wrong and the
plan says so.

Scratch only — nothing in `src/momwire` is touched. Measured at the AK pointer
commit 12584f2; `origin/main` differs from it by 131 added lines in
`hmatrix.py` and nothing else under `src/momwire`, with no C++ delta, so this
is main's behaviour for this fill.
"""

from __future__ import annotations

import sys
import time
import warnings

import numpy as np

warnings.simplefilter("ignore")

from momwire import BSplineSolver, SinusoidalGalerkinSolver  # noqa: E402

C0 = 299792458.0
WL7 = C0 / 7e6
SEGS = 15
GAPS = (0.5, 0.1, 0.02, 0.005)


def deck(gap: float):
    """#1004's deck: each wire's near end lifted `gap` clear of the plane."""
    above = np.array([(0.0, 0.0, gap), (0.0, 0.0, gap + 1.0)])
    below = np.array([(0.0, 0.0, -gap - 1.0), (0.0, 0.0, -gap)])
    return above, below


def mk(cls, wires, *, free: bool, eps=(1.0, 0.0)):
    ground = (
        {} if free else dict(ground_z=0.0, ground_eps=eps, ground_model="sommerfeld")
    )
    return cls(
        wires=list(wires),
        n_per_edge_per_wire=[[SEGS]] * len(wires),
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        **ground,
    )


def z_of(s) -> complex:
    return complex(s.compute_impedance()[0])


def G_of(s) -> np.ndarray:
    geom = s._build_geometry()
    if getattr(s, "_is_mixed", None) is not None and s._is_mixed(geom):
        return s._assemble_Z(geom, s.k)[0]
    with s._operating_medium(geom):
        return s._assemble_Z(geom, s.k)[0]


def n_above(cls, above) -> int:
    """Basis count for the above wire alone — the block split point."""
    return G_of(mk(cls, [above], free=True)).shape[0]


def main() -> int:
    cls = SinusoidalGalerkinSolver
    label = "SinusoidalGalerkinSolver (D2)"
    if len(sys.argv) > 1 and sys.argv[1] == "bspline":
        cls, label = BSplineSolver, "BSplineSolver (the arbiter)"

    print("=" * 78)
    print(f"REPRODUCTION — {label}, {SEGS} seg/wire, 7 MHz, eps-tilde = 1")
    print("=" * 78)
    print(f"{'gap d':>8} {'rel vs free space':>20}   (#1004's table for D2)")
    ref = {0.5: "2.3e-10", 0.1: "5.8e-09", 0.02: "1.6e-07", 0.005: "1.9e-07"}
    zrel = {}
    for g in GAPS:
        above, below = deck(g)
        t0 = time.time()
        zf = z_of(mk(cls, [above, below], free=True))
        zc = z_of(mk(cls, [above, below], free=False))
        rel = abs(zc - zf) / abs(zf)
        zrel[g] = rel
        print(f"{g:8.3f} {rel:20.3e}   issue: {ref[g]:>9}  [{time.time() - t0:.0f}s]")

    print()
    print("=" * 78)
    print("WHERE IT LIVES — the eps-tilde=1 minus free-space operator, by class")
    print("=" * 78)
    print(
        f"{'gap d':>8} {'above x above':>14} {'below x below':>14} "
        f"{'cross a<-b':>12} {'cross b<-a':>12}"
    )
    blocks = {}
    for g in GAPS:
        above, below = deck(g)
        na = n_above(cls, above)
        Gf = G_of(mk(cls, [above, below], free=True))
        Gc = G_of(mk(cls, [above, below], free=False))
        assert Gf.shape == Gc.shape, (Gf.shape, Gc.shape)
        D = Gc - Gf
        scale = np.abs(Gf).max()
        aa = np.abs(D[:na, :na]).max() / scale
        bb = np.abs(D[na:, na:]).max() / scale
        ab = np.abs(D[:na, na:]).max() / scale
        ba = np.abs(D[na:, :na]).max() / scale
        blocks[g] = (aa, bb, ab, ba)
        print(f"{g:8.3f} {aa:14.3e} {bb:14.3e} {ab:12.3e} {ba:12.3e}")

    print()
    print("=" * 78)
    print("GROWTH 0.5 m -> 5 mm (the thing that has to be explained)")
    print("=" * 78)
    names = ("above x above", "below x below", "cross a<-b", "cross b<-a")
    lo, hi = blocks[GAPS[0]], blocks[GAPS[-1]]
    for i, nm in enumerate(names):
        if lo[i] > 0:
            print(f"  {nm:14s} {lo[i]:9.3e} -> {hi[i]:9.3e}   {hi[i] / lo[i]:10.1f}x")
        else:
            print(f"  {nm:14s} {lo[i]:9.3e} -> {hi[i]:9.3e}   (from zero)")
    print(
        f"  {'|Z| relative':14s} {zrel[GAPS[0]]:9.3e} -> {zrel[GAPS[-1]]:9.3e}   "
        f"{zrel[GAPS[-1]] / zrel[GAPS[0]]:10.1f}x"
    )
    print()
    print("  A block that is flat while another grows is the control, not noise:")
    print("  it says the interface term is collapsing correctly for that class.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
