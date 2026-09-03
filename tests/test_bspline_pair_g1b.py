"""G1-B — the bspline degree-1 / degree-2 PAIR is the underground consistency
check (AK PLAN-MIDTERM goal 1, decided 2026-09-03).

Underground the reference is measurement (Brown-Lewis-Epstein 1937, gated in
`test_ble_1937_838`), not NEC-5, and the engine is bspline degree 2. What
replaces the razor/NEC-5 pair as the second reading is the SAME trunk on a
different basis: degree 1 (the tent basis, Galerkin-tested). This file gates
that pair on every buried anchor the suite serves with a feed ABOVE ground.

The class, measured 2026-09-03 (laptop, momwire main 53a15a6, scripts in AK
`scratch/g1b-bs1-bs2/`): at each anchor's own mesh and quadrature the two
degrees sit 0.08–0.96 Ω apart, and they walk to ONE limit under refinement —

    anchor          |bs1-bs2| x1   x3      x5      x9     (whole-mesh scale)
    crossing_g1        0.9619   0.6141  0.2422  0.2842
    hub_deck(4)        0.4753   0.1537
    fan n2 (far-only)  0.4046   0.5941  0.0949          (node grading fixed)
    ble45 N=2          0.0751   0.0555
    ble45 N=15         0.2414   0.093
    ble45 N=30         0.2388
    AK catalog BRV     0.5509 (hub) / 0.5511 (bundle), n_qp_pair 32

Degree 1 does not walk monotonically (the tent basis oscillates: crossing
139.58 / 138.19 / 138.56 / 138.15 in R), so the gate is on the SEPARATION at
the anchor mesh, not on a shrink. The bar is 1.5 Ω over the 0.96 worst case.

Two things this file deliberately does NOT gate:

* **Decks fed IN the soil** (`test_buried_serve_553`'s bvd1 / bhd10 dipoles and
  its elevated-over-detached-radial `served_deck`). Both degrees move by ohms
  per refinement rung there and have not settled at any reachable mesh —
  bvd1 still moves 3.8 Ω (bs2) per rung at ×27, Δ/a = 3.4 — while their
  separation shrinks (37 → 5.2 Ω). The anchor mesh is not an impedance anchor
  on that class; the 553 gates are shape gates (monotone, shrinking) and stay
  so. Whatever a buried-fed number may claim is a G1-C question.
* **Agreement with razor or NEC-5.** By decision (momwire#813/#814) those are
  above-ground twins; never re-gate a buried number against either.

A finding recorded here because it is where it was measured: `CROSSING_G1`'s
comment quotes a mesh envelope of 0.021 Ω (g1↔g2, which refines the NODE).
Refining the FAR mesh moves degree 2 by 0.36 Ω at ×3 and 0.61 Ω at ×9. The
banked number is the q-converged print at the g1 far mesh, and its 0.05 gate
is a regression gate at that mesh — sound, but not "far-mesh converged".
"""

import pytest

from momwire.bspline import BSplineSolver

from test_ble_1937_838 import ble_deck
from test_crossing_serve_524 import (
    CROSSING_G1,
    CROSSING_G1_QP,
    FAN_SOIL_A_N2,
    FAN_SOIL_A_N2_QP,
    crossing_deck,
    fan_rise_deck_graded,
    hub_deck,
)

# The stated class: the two degrees agree within this at the anchor mesh.
# 0.96 Ω is the measured worst case (crossing_g1); the AK catalog deck sits at
# 0.55 and the BLE screens under 0.25.
PAIR_CLASS_OHM = 1.5

# Below this the two solves did not run different bases — the mutation the
# gate must catch is `degree` being ignored (then bs1 ≡ bs2 to the bit).
DISTINCT_BASES_OHM = 1e-6

# name: (build, n_qp_pair or None for the deck's own, banked degree-2 anchor
#        or None where none is banked (record only), measured separation 2026-09-03)
_ROWS = {
    "crossing-g1": (lambda: crossing_deck(1), CROSSING_G1_QP, CROSSING_G1, 0.9619),
    "hub-4": (lambda: hub_deck(4), 32, None, 0.4753),
    "fan-n2": (
        lambda: fan_rise_deck_graded("n2"),
        FAN_SOIL_A_N2_QP,
        FAN_SOIL_A_N2,
        0.4046,
    ),
    "ble45-n2": (lambda: ble_deck(2), None, None, 0.0751),
    "ble45-n15": (lambda: ble_deck(15), None, None, 0.2414),
}
# Every row rides the crossgate lane. The fan and BLE rows because their
# degree-2 anchors do; the crossing and hub rows because solving BOTH degrees
# at q = 64 / 32 costs 21-26 s on the g++ 11.4 box (momwire#839's machine),
# past the default lane's 20 s time-budget ceiling, so they were a latent flake
# there and on any loaded runner. This is a certification gate, and the
# push-only lane is where those run.
_CROSSGATE = set(_ROWS)


def _solve(build, degree, nqp):
    kw = dict(build)
    if nqp is not None:
        kw["n_qp_pair"] = nqp
    z, _ = BSplineSolver(**kw, degree=degree).compute_impedance()
    return z


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(r, marks=[pytest.mark.crossgate] if r in _CROSSGATE else [])
        for r in sorted(_ROWS)
    ],
)
def test_g1b_the_two_degrees_agree_on_every_above_fed_buried_anchor(
    row, record_property
):
    build, nqp, banked, measured = _ROWS[row]
    z1 = _solve(build(), 1, nqp)
    # Degree 2 is SOLVED even where a banked anchor exists: the pair must be
    # two readings of one tree, so that ignoring `degree` collapses them to
    # the bit (the mutation below catches). The banked anchor is recorded
    # only; its own gate lives with the anchor.
    z2 = _solve(build(), 2, nqp)
    d = abs(z1 - z2)
    record_property("bs1_Z", f"{z1:.4f}")
    record_property("bs2_Z", f"{z2:.4f}")
    if banked is not None:
        record_property("bs2_banked_Z", f"{banked:.4f}")
        record_property("bs2_vs_banked_ohm", float(abs(z2 - banked)))
    record_property("pair_sep_ohm", float(d))
    record_property("pair_sep_measured_2026_09_03", measured)
    assert d > DISTINCT_BASES_OHM, (
        f"{row}: degree 1 and degree 2 answered the same number {z1:.6f} — the "
        "degree kwarg is not reaching the basis, so this is one reading, not a pair"
    )
    assert d <= PAIR_CLASS_OHM, (
        f"{row}: bspline degree 1 answers {z1:.4f} where degree 2 answers "
        f"{z2:.4f} — {d:.4f} ohm apart, outside the {PAIR_CLASS_OHM:g} ohm pair "
        f"class (measured {measured:.4f} on 2026-09-03). The two degrees share "
        "one trunk; a split this size is a fill or basis regression, not a "
        "physics disagreement. Never resolve it by re-gating against razor or "
        "NEC-5 underground."
    )
