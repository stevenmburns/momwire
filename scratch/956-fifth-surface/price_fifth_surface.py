"""Price the fifth transmitted surface's contribution to the rise's Z — momwire#956.

#956's verdict is that the ~2 Ω contact-class residual against NEC-5 belongs to
the conductor's PASSAGE through the air/soil interface. The plan comment argues
it is not the crossing node — the residual is distributed along the rise and has
no intercept, while every candidate inside `_crossing_fill` is lumped at the
node — and points instead at the below→above transmitted coupling, whose FIFTH
surface (`TzH` = E_z^H = −C₁cosφ ∂²V_T/∂ρ∂z′) phase 0 named the highest-risk
quantity in the kit and measured the licensed engine departing from by O(1).

This asks the one question that needs no oracle: **how many ohms is that term
worth on this deck?** If removing it moves R by ≫2 Ω, the residual sits inside
its error budget and the question becomes whose fifth surface is right — which
phase 0 already answered against the engine, with empymod. If it moves R by
≪2 Ω, the term cannot carry the residual and #1004 is next.

REGISTERED PREDICTION, from the plan comment, before any row was taken:
the fifth surface's own contribution to R is ≥ 2 Ω at d = 0.15 m and scales as
~d^1.1 across the hub-depth ladder — the residual's own exponent.

INSTRUMENTATION. Scratch only: nothing in `src/momwire` is edited. The term is
scaled by monkeypatching the two producers that return the surface dict, so all
three `_combine_transmitted*` consumers see the scaled value without this script
re-deriving their algebra. `MOMWIRE_TRANSMITTED_FORCE_NUMPY=1` is set first —
the family has C++ twins (`transmitted_six_integrals_batch`,
`transmitted_field_proj_batch`) that would compute the field internally and make
a Python-side patch a silent no-op. A knob that changes nothing looks exactly
like a null result, so G1–G3 below check the plumbing before any row is read.
"""

from __future__ import annotations

import os
import sys
import warnings

os.environ["MOMWIRE_TRANSMITTED_FORCE_NUMPY"] = "1"

import numpy as np  # noqa: E402

warnings.simplefilter("ignore")

AK_SRC = "/home/smburns/stevenmburns/antennaknobs/src"
if AK_SRC not in sys.path:
    sys.path.insert(0, AK_SRC)

from momwire import BSplineSolver, _sommerfeld_transmitted as trans  # noqa: E402

from antennaknobs.designs.verticals import buried_radial_vertical as brv  # noqa: E402
from antennaknobs.engines.momwire import MomwireEngine  # noqa: E402

SOIL_A = ("finite", 13.0, 0.005)

_ORIG_DIRECT = trans.t_surfaces_direct
_ORIG_EVAL = trans.TransmittedGrid.eval


def _scale_fifth(factor: float) -> None:
    """Multiply every returned `TzH` by `factor`, at both producers."""

    def direct(*a, **kw):
        out = _ORIG_DIRECT(*a, **kw)
        return {**out, "TzH": out["TzH"] * factor}

    def eval_(self, *a, **kw):
        out = _ORIG_EVAL(self, *a, **kw)
        return {**out, "TzH": out["TzH"] * factor}

    trans.t_surfaces_direct = direct
    trans.TransmittedGrid.eval = eval_


def _restore() -> None:
    trans.t_surfaces_direct = _ORIG_DIRECT
    trans.TransmittedGrid.eval = _ORIG_EVAL


def solve(depth: float, nsegs: int | None = None) -> complex:
    b = brv.Builder()
    b.depth = depth
    if nsegs is not None:
        b.nominal_nsegs = nsegs
    return MomwireEngine(b, solver=BSplineSolver, ground=SOIL_A).impedance()[0]


def deck_crosses_the_interface(depth: float) -> tuple[float, float]:
    """(min z, max z) of the built geometry — the rise must span z = 0."""
    b = brv.Builder()
    b.depth = depth
    zs = [p[2] for w in b.build_wires() for p in (w[0], w[1])]
    return min(zs), max(zs)


def main() -> int:
    depths = [float(x) for x in (sys.argv[1:] or ["0.15", "0.60", "1.20"])]

    print("=" * 78)
    print("GATES — each can fail, and what it means if it does")
    print("=" * 78)

    # G1: the C++ twins are off, or the patch below is a no-op.
    assert not trans._use_transmitted_accel(), (
        "the transmitted family is still on the C++ path; the scaling patch "
        "would be silently inert and every row below would read as 'no effect'"
    )
    print("G1  transmitted accelerator OFF (numpy path)            PASS")

    # G2: the deck genuinely crosses the interface.
    lo, hi = deck_crosses_the_interface(depths[0])
    assert lo < 0.0 < hi, f"deck does not span z=0: z in [{lo}, {hi}]"
    print(f"G2  deck spans the interface: z in [{lo:.3f}, {hi:.3f}]   PASS")

    # G3: the patch is INERT at factor 1 and LIVE at factor 0.
    base = solve(depths[0])
    _scale_fifth(1.0)
    at_one = solve(depths[0])
    _restore()
    assert abs(at_one - base) < 1e-9 * abs(base), (
        f"factor=1.0 changed the answer ({base} -> {at_one}); the patch is not "
        f"a clean identity and nothing below can be attributed to the term"
    )
    print(f"G3a factor=1.0 is an identity: {base.real:.4f} ohm            PASS")

    _scale_fifth(0.0)
    at_zero = solve(depths[0])
    _restore()
    moved = abs(at_zero - base)
    assert moved > 1e-6 * abs(base), (
        f"factor=0.0 did not move Z ({base} -> {at_zero}); the fifth surface "
        f"is NOT plumbed through this deck's fill and this experiment cannot "
        f"answer the question — a flag flip that changes nothing is unplumbed"
    )
    print(f"G3b factor=0.0 moves Z by {moved:.4f} ohm                  PASS")
    print()

    print("=" * 78)
    print("THE LADDER — R(factor) at each hub depth, soil A, 7.1 MHz")
    print("=" * 78)
    print(
        f"{'depth':>7} {'R(1.0)':>10} {'R(0.5)':>10} {'R(0.0)':>10} "
        f"{'contribution':>13} {'half-check':>11}"
    )
    rows = {}
    for d in depths:
        vals = {}
        for f in (1.0, 0.5, 0.0):
            if f == 1.0:
                vals[f] = solve(d)
            else:
                _scale_fifth(f)
                vals[f] = solve(d)
                _restore()
        contrib = vals[1.0].real - vals[0.0].real
        # If the term entered linearly, R(0.5) would sit exactly halfway.
        halfway = vals[1.0].real - 0.5 * contrib
        rows[d] = (vals, contrib)
        print(
            f"{d:7.2f} {vals[1.0].real:10.4f} {vals[0.5].real:10.4f} "
            f"{vals[0.0].real:10.4f} {contrib:13.4f} "
            f"{vals[0.5].real - halfway:+11.4f}"
        )

    print()
    print("=" * 78)
    print("AGAINST THE REGISTERED PREDICTION")
    print("=" * 78)
    c0 = rows[depths[0]][1]
    print(
        f"  contribution at d={depths[0]:.2f} m : {c0:+.4f} ohm   "
        f"(predicted >= 2 ohm: {'YES' if abs(c0) >= 2.0 else 'NO'})"
    )
    if len(depths) > 1:
        d_lo, d_hi = depths[0], depths[-1]
        c_lo, c_hi = rows[d_lo][1], rows[d_hi][1]
        if c_lo != 0:
            expo = np.log(abs(c_hi / c_lo)) / np.log(d_hi / d_lo)
            print(
                f"  exponent over {d_lo:.2f} -> {d_hi:.2f} m       : d^{expo:.2f}   "
                f"(predicted ~d^1.1: {'YES' if 0.8 <= expo <= 1.4 else 'NO'})"
            )
    print()
    print("  The residual this is measured against is +2.1 ohm (#956).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
