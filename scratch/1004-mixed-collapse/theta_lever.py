"""Is the near-plane collapse error a theta-interpolation floor? — momwire#1004 (b).

#1004 guesses "the transmitted block's near-plane behaviour (the grid's theta
floor is at its finest exactly here)". `split_the_collapse_error.py` confirmed
the CROSS block carries all of it. This asks whether THETA RESOLUTION is the
lever, by moving `_DTH_DEG` (the grid's theta step, 3.0 degrees shipped) and
watching the cross-block error.

REGISTERED BEFORE RUNNING — two of my predictions were burned tonight and the
answer to that is to keep registering them, not to stop:

  * If theta interpolation carries it, halving `_DTH_DEG` cuts the cross-block
    error at d = 5 mm by **>= 4x** (the stencil is 4-point Lagrange, so a
    smooth integrand would give ~16x; 4x is the falsifiable bar).
  * If it falls by **< 2x**, theta resolution is NOT the lever and the issue's
    guess is dead: the error would then be in the surfaces themselves near
    grazing, or in the R axis, not in how finely theta is sampled.

GATE FIRST: `_DTH_DEG` must actually change `n_th`, or the knob is inert and
every row below would read as "refining theta does not help" for the wrong
reason. Scratch only; nothing in `src/momwire` is edited.
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.simplefilter("ignore")

from momwire import SinusoidalGalerkinSolver as SG  # noqa: E402
from momwire import _sommerfeld_transmitted as T  # noqa: E402

C0 = 299792458.0
WL7 = C0 / 7e6
SEGS = 15
GAP = 0.005

_ORIG_DTH = T._DTH_DEG


def deck(gap):
    return (
        np.array([(0.0, 0.0, gap), (0.0, 0.0, gap + 1.0)]),
        np.array([(0.0, 0.0, -gap - 1.0), (0.0, 0.0, -gap)]),
    )


def mk(wires, *, free):
    ground = (
        {}
        if free
        else dict(ground_z=0.0, ground_eps=(1.0, 0.0), ground_model="sommerfeld")
    )
    return SG(
        wires=list(wires),
        n_per_edge_per_wire=[[SEGS]] * len(wires),
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        **ground,
    )


def G_of(s):
    geom = s._build_geometry()
    if s._is_mixed(geom):
        return s._assemble_Z(geom, s.k)[0]
    with s._operating_medium(geom):
        return s._assemble_Z(geom, s.k)[0]


def _set_theta(dth):
    """Move the step AND drop the cached grids.

    `get_grid_below_above` memoises into `_GRID_CACHE` on a key that does not
    include `_DTH_DEG`, so without this the second setting silently reuses the
    first grid and the knob reads as inert — the failure the gate is for.
    """
    T._DTH_DEG = dth
    T._GRID_CACHE.clear()


def cross_err(dth):
    _set_theta(dth)
    above, below = deck(GAP)
    na = G_of(mk([above], free=True)).shape[0]
    Gf = G_of(mk([above, below], free=True))
    Gc = G_of(mk([above, below], free=False))
    D = Gc - Gf
    scale = np.abs(Gf).max()
    return max(np.abs(D[:na, na:]).max(), np.abs(D[na:, :na]).max()) / scale


def n_th_for(dth):
    _set_theta(dth)
    above, below = deck(GAP)
    G_of(mk([above, below], free=False))
    for obj in T._GRID_CACHE.values():
        if hasattr(obj, "n_th"):
            return obj.n_th
    return None


def main():
    print("=" * 74)
    print("GATE — the knob must move the grid, or nothing below means anything")
    print("=" * 74)
    seen = {}
    for dth in (_ORIG_DTH, _ORIG_DTH / 2):
        seen[dth] = n_th_for(dth)
        print(f"  _DTH_DEG={dth:5.2f} -> n_th={seen[dth]}")
    _set_theta(_ORIG_DTH)
    vals = [v for v in seen.values() if v is not None]
    assert len(vals) == 2 and vals[0] != vals[1], (
        f"halving _DTH_DEG did not change n_th ({seen}); the knob is inert"
    )
    print("  PASS — theta resolution is plumbed\n")

    print("=" * 74)
    print(f"CROSS-BLOCK ERROR at d = {GAP} m, vs theta step")
    print("=" * 74)
    base = None
    for dth in (_ORIG_DTH, _ORIG_DTH / 2, _ORIG_DTH / 4):
        try:
            e = cross_err(dth)
        finally:
            _set_theta(_ORIG_DTH)
        base = base if base is not None else e
        print(
            f"  _DTH_DEG={dth:5.3f}  cross-block rel = {e:.4e}   "
            f"{base / e:6.2f}x better than shipped"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
