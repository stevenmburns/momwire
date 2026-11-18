"""The buried pair quadrature, per solver: one declared record.

bspline and SinusoidalGalerkinSolver (SG) share the B-spline fill's tiered
pair moments (`_bspline_kernels`, through `_sinusoidal_mp.WindowFill` on SG's
side) but do not run them the same way on a deck with wires below the
interface. bspline runs its near-contact pairs at a raised base order
(momwire#1752). SG runs the same machinery at older orders, and then
OVERWRITES every pair closer than its near ratio with rules of its own
(`_sinusoidal_mp.near_pair_moments`), so for SG it is those near rules, not
the base order, that set the near-pair quadrature error.

That divergence is written down here. Each solver reads its entry instead of
keeping literals of its own, and `tests/test_buried_quadrature_policy.py`
pins bspline's C++ copies to its entry and holds the claim that SG's base
order and far ladder serve none of SG's near pairs.

A ladder is the momwire#906 shape, a tuple of `(ratio, n_qp)`: a pair whose
centre distance over the longer segment meets a rung's ratio takes that
rung's order, the rest the base order. Ratios strictly ascend and orders
strictly descend below the base, the rule `_bspline_kernels._normalize_ladder`
enforces at fill time and `BuriedQuadrature` checks at import.

Nothing here is a switch: the record carries the shipped numbers, and
changing one changes a solver's answers and has to be measured as such.
"""

from __future__ import annotations

from dataclasses import dataclass

Ladder = tuple[tuple[float, int], ...]


def _check_ladder(ladder: Ladder, base: int, what: str) -> None:
    prev_r, prev_n = 0.0, int(base)
    for r, n in ladder:
        if not (float(r) > prev_r and 1 <= int(n) < prev_n):
            raise ValueError(
                f"{what}: a ladder's ratios must strictly ascend past 0 and its "
                f"orders strictly descend below the base {base}, got {ladder}"
            )
        prev_r, prev_n = float(r), int(n)


@dataclass(frozen=True)
class NearPairRules:
    """SG's near-pair rules (`_sinusoidal_mp.near_pair_moments`). They apply
    on every SG mixed-potential fill, buried or not; they are recorded here
    because on a buried deck they are what bspline's raised base order is to
    bspline.

    near_ratio : pairs whose centre distance over the longer segment is
        below this are overwritten by the rules below (`classify_pairs`).
    graded_n_per_panel : Gauss nodes per panel of the graded rule
        (`graded_endpoint_rule01`, panels doubling away from the shared
        node) on a TOUCHING non-parallel pair, on both segments.
    near_n_qp : the product order on a near non-parallel pair that does not
        touch.
    graded_acute_wedge_deg, graded_acute_n_per_panel : a touching pair whose
        wedge (the angle between its two segments at the shared node, 180 for
        a straight continuation) is under the first takes the second in place
        of `graded_n_per_panel` (`_sinusoidal_mp.touching_wedge_deg`).
    parallel_n_t, parallel_n_xi : the parallel pair's one-dimensional
        reduction, nodes per piece of the sinh-substituted delta integral
        and of the inner shape correlation.
    """

    near_ratio: float
    graded_n_per_panel: int
    near_n_qp: int
    parallel_n_t: int
    parallel_n_xi: int
    graded_acute_wedge_deg: float
    graded_acute_n_per_panel: int


@dataclass(frozen=True)
class BuriedQuadrature:
    """One solver's buried pair quadrature.

    base_order, ladder : the tiered fill's pair order and ladder on a deck's
        BELOW-interface pairs when the deck also has wires above it (a
        crossing deck).
    wholly_buried_order, wholly_buried_ladder : the same for a deck whose
        every wire is below the interface.
    near : the rules that overwrite the near pairs after the tiered fill, or
        None when the solver has no such overwrite (bspline: its base order
        serves its near pairs itself).
    reason : why these values, with the measurement behind them.
    """

    base_order: int
    ladder: Ladder
    wholly_buried_order: int
    wholly_buried_ladder: Ladder
    near: NearPairRules | None
    reason: str

    def __post_init__(self) -> None:
        _check_ladder(self.ladder, self.base_order, "ladder")
        _check_ladder(
            self.wholly_buried_ladder, self.wholly_buried_order, "wholly_buried_ladder"
        )


# bspline: one rule for every buried deck, crossing or wholly buried
# (`BSplineSolver.n_qp_pair` / `pair_order_ladder` resolve it whenever a wire
# is below the interface). Its C++ twins, `_nec5_bspline_printout_core.cpp`
# and `_accel_razor_solve.cpp`'s buried-stages hook, carry the same literals.
BSPLINE = BuriedQuadrature(
    base_order=128,
    ladder=((0.1, 32), (1.5, 16), (2.0, 8), (4.0, 4)),
    wholly_buried_order=128,
    wholly_buried_ladder=((0.1, 32), (1.5, 16), (2.0, 8), (4.0, 4)),
    near=None,
    reason=(
        "momwire#1752. At the old base 32 (momwire#760) the near-contact pairs "
        "(a buried wire's own image where it climbs to a crossing node at the "
        "surface) were not converged: fx_slant_base_source read 86.905-17.784j "
        "at 32, 86.653-18.783j at 64, 86.622-18.923j at 128 and "
        "86.6215-18.9278j at 256/512, a 1.18 ohm quadrature error against a "
        "0.52 ohm mesh error; 128 lands within 0.004 ohm. S2 puts 128 only "
        "under the 0.1-length rung and keeps 32 to 1.5 lengths (16 from 1.0 "
        "moved the dense-hub screens 0.01 ohm off q = 128); H2 starts the "
        "order-4 rung at 4 lengths instead of 16 (Z_in moves <= 2e-10 of "
        "max|Z| on the 15-deck set)."
    ),
)

# SG: bspline's buried order and ladder from before #1752 on the crossing
# route (`_assemble_Z_mp_mixed`'s below class), bspline's FREE-SPACE order and
# ladder on the wholly-buried route (`_assemble_Z_mp` at k_m), and the near
# rules that overwrite every pair under `near_ratio` on both.
SINUSOIDAL_GALERKIN = BuriedQuadrature(
    base_order=32,
    ladder=((2.0, 8), (16.0, 4)),
    wholly_buried_order=8,
    wholly_buried_ladder=((16.0, 4),),
    near=NearPairRules(
        near_ratio=2.0,
        graded_n_per_panel=8,
        near_n_qp=32,
        parallel_n_t=24,
        parallel_n_xi=8,
        graded_acute_wedge_deg=15.0,
        graded_acute_n_per_panel=24,
    ),
    reason=(
        "Not bspline's #1752 values, and not because SG converged at them. "
        "SG's base order serves NO pair on the crossing route: the ladder's "
        "first rung (2.0) is the near ratio, and every pair under it is "
        "overwritten by the near rules. On the wholly-buried route the base "
        "order 8 serves the pairs from 2 to 16 lengths. So raising the base "
        "order or the ladder cannot reach a near-contact error, which is why "
        "PD2-measure (11-17, scratch branch measure-sg-buried-orders-1117) "
        "saw raised orders move Z only at drift level: that measurement "
        "never exercised the near pairs. A trace the same day found that SG "
        "DOES carry a near-contact quadrature error on fx_slant_base_source "
        "at m = 1: 0.425 ohm, from the radial's node segment against its "
        "own image (touching at the crossing node, ratio 0.02) on the graded "
        "rule at 8 nodes per panel, against a reference with every near rule "
        "raised: 3.0e-2 ohm at 16, 3.5e-3 ohm at 24, 8.7e-4 at 48. That pair's "
        "WEDGE (the angle between the two segments at their shared node) is "
        "2.3 degrees, and the graded rule's error is the wedge's: over 372 "
        "decks with touching pairs (11-17: the repo's fixtures, the "
        "antennaknobs catalog at free / PEC / Sommerfeld, synthetic bends, "
        "the buried set) every 8-to-24 move above 1e-8 ohm sits on a deck "
        "whose smallest wedge is under 21 degrees, falling from 4.9e-2 ohm "
        "on a wire folded back at 5 degrees to 4.6e-4 at 10, 8.6e-5 on a "
        "32-radial hub (11.25), 4.1e-7 on the catalog discone (15.3) and "
        "1.1e-7 at 20; shallow bends and right angles move <= 1e-8. So a "
        "touching pair whose wedge is under 15 degrees takes 24 nodes a "
        "panel and the rest keep 8 (Steve, 11-17): on every deck it matches "
        "24-everywhere within 4.1e-7 ohm (the discone, left at 8), and the "
        "cost of 24 lands only where the wedge is acute: on the 21 decks over "
        "0.5 s with touching pairs, solve time 1.02x median and 1.12x max "
        "(the 32-radial hub) against 8, where 24 everywhere cost 1.74x and "
        "4.2x (Skylake, paired medians)."
    ),
)

POLICIES: dict[str, BuriedQuadrature] = {
    "bspline": BSPLINE,
    "sinusoidal-galerkin": SINUSOIDAL_GALERKIN,
}


def sinusoidal_galerkin() -> BuriedQuadrature:
    """SG's entry, read at fill time (not bound at import) so a test can
    swap it through `POLICIES` and reach the production route."""
    return POLICIES["sinusoidal-galerkin"]
