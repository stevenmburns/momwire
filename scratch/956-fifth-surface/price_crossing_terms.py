"""Price the crossing fill's boundary terms on the rise — momwire#956.

The companion script `price_fifth_surface.py` asked whether the below→above
TRANSMITTED family's fifth surface carries the ~2 Ω residual. It does not, and
not because the term is small: **it is not in the impedance path at all.**
Zeroing `TzH` moves Z by exactly nothing, and the reason is structural —
`_crossing_fill` imports only `_c1_moment` from `_sommerfeld_transmitted`, and
nothing outside that module consumes `t_surfaces_direct`, `TransmittedGrid` or
`_combine_transmitted*`. The five surfaces are a near-field quantity (the NE/NH
and RP consumers #570 tracks as pending), not a Z one. That falsifies the plan
comment's hypothesis (b) before any building, which is what the gate was for.

So this prices the terms that ARE in the rise's path: the by-parts ends and the
corner, which is the other half of what #956's verdict named
("ρ_eff regularisation at the corner, the by-parts boundary term").

`_ends_and_corner` already carries a `corner` switch, so the corner alone is
separable from the ends. Scaling is done by monkeypatch from scratch — nothing
in `src/momwire` is edited.

Read the numbers against **+2.1 Ω**, the residual #956 measured against NEC-5
on this deck class. A term whose whole contribution is far below that cannot
carry it however wrong it is; a term far above it could carry it with a small
relative error.
"""

from __future__ import annotations

import sys
import warnings

warnings.simplefilter("ignore")

AK_SRC = "/home/smburns/stevenmburns/antennaknobs/src"
if AK_SRC not in sys.path:
    sys.path.insert(0, AK_SRC)

from momwire import BSplineSolver, _crossing_fill as cf  # noqa: E402

from antennaknobs.designs.verticals import buried_radial_vertical as brv  # noqa: E402
from antennaknobs.engines.momwire import MomwireEngine  # noqa: E402

SOIL_A = ("finite", 13.0, 0.005)
RESIDUAL = 2.1  # ohm, #956's measured ΔR against NEC-5 on this class

_ORIG = cf._ends_and_corner


def _patch(*, factor: float = 1.0, force_corner: bool | None = None) -> None:
    """Scale the ends+corner block, and/or override its `corner` argument.

    `force_corner=False` drops the corner term and keeps the ends, which is the
    separation the fill already supports; `factor=0.0` drops both.
    """

    def wrapped(ctx, A, B, eps_t, k_p, c1, gz, memo=None, *, corner=True):
        use = corner if force_corner is None else force_corner
        out = _ORIG(ctx, A, B, eps_t, k_p, c1, gz, memo=memo, corner=use)
        return out * factor

    cf._ends_and_corner = wrapped


def _restore() -> None:
    cf._ends_and_corner = _ORIG


def solve(depth: float) -> complex:
    b = brv.Builder()
    b.depth = depth
    return MomwireEngine(b, solver=BSplineSolver, ground=SOIL_A).impedance()[0]


def main() -> int:
    depths = [float(x) for x in (sys.argv[1:] or ["0.15", "0.60", "1.20"])]

    print("=" * 78)
    print("GATES")
    print("=" * 78)
    base = solve(depths[0])

    _patch(factor=1.0)
    at_one = solve(depths[0])
    _restore()
    assert abs(at_one - base) < 1e-9 * abs(base), (
        f"the wrapper is not an identity at factor=1 ({base} -> {at_one})"
    )
    print(f"G1  wrapper is an identity at factor=1: {base.real:.4f} ohm   PASS")

    _patch(factor=0.0)
    at_zero = solve(depths[0])
    _restore()
    moved = abs(at_zero - base)
    assert moved > 1e-6 * abs(base), (
        f"zeroing ends+corner did not move Z ({base} -> {at_zero}) — the knob "
        f"is unplumbed and nothing below means anything"
    )
    print(f"G2  factor=0 moves Z by {moved:.4f} ohm                 PASS")
    print()

    print("=" * 78)
    print(f"ENDS+CORNER, priced against the {RESIDUAL:+.1f} ohm residual")
    print("=" * 78)
    print(
        f"{'depth':>7} {'R(full)':>10} {'R(no e+c)':>10} {'ends+corner':>12} "
        f"{'R(no corner)':>13} {'corner only':>12}"
    )
    rows = {}
    for d in depths:
        full = solve(d)

        _patch(factor=0.0)
        none_ = solve(d)
        _restore()

        _patch(force_corner=False)
        nocorner = solve(d)
        _restore()

        ec = full.real - none_.real
        corner_only = full.real - nocorner.real
        rows[d] = (ec, corner_only)
        print(
            f"{d:7.2f} {full.real:10.4f} {none_.real:10.4f} {ec:12.4f} "
            f"{nocorner.real:13.4f} {corner_only:12.4f}"
        )

    print()
    print("=" * 78)
    print("READ")
    print("=" * 78)
    for d, (ec, co) in rows.items():
        print(
            f"  d={d:.2f} m: ends+corner is {abs(ec) / RESIDUAL:7.1f}x the residual; "
            f"corner alone {abs(co) / RESIDUAL:7.1f}x"
        )
    print()
    print("  A term worth many times the residual could carry it with a small")
    print("  relative error; one worth a fraction of it cannot, however wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
