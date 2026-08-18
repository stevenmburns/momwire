"""The production (Gauss-Legendre) lane's bar: convergence, gated as decay.

momwire#398 unit 6, the maintainer's decision of 2026-08-17. The two
quadrature lanes carry two different claims, and each is gated on its own
terms:

* the `nec5_quadrature=True` lane claims the FORMULATION TWIN — a fixed,
  N-independent offset from the binary's printed answers — and is gated at
  the sharp bar (`test_razor_pec_ground.py`, `test_razor_ground_contact.py`);
* the default Gauss-Legendre lane claims CONVERGENCE TO THE CONTINUUM.
  Converged GL is deliberately not NEC-5's rule, so its residual against the
  binary is a quadrature difference that must VANISH as the mesh refines —
  and measured, it does, on every gated geometry: dipole 1.372 → 0.116 Ω,
  fat dipole 0.957 → 0.130, inverted-V 1.314 → 0.548 over N = 24…96 on the
  clearance ladders, monopole 0.195 → 0.029 and inverted-L 0.520 → 0.140 on
  the contact ladders (against the mirror-deck oracle unit 3's decision
  fixed). So the production gate is DECAY pins: the residual strictly
  shrinking down each ladder, with the finest rung pinned at its measured
  level (+25 % headroom).

The LOOP is the recorded exception: its GL residual is non-monotone
(1.778 → 4.544 → 3.106 over the ladder — four junctions, and the slowest
mesh in the set) and carries an ENVELOPE pin only, so a regression is still
caught while the non-decay stays an honest record rather than a bar it
cannot meet.

The finite grounds take no NEC-5 bar at all (Michalski limit offset — the
refusal prose in `razor.py` explains); their production bars are the
cross-formulation difference-of-columns gates units 4 and 5 landed
(`test_razor_refl_coef_ground.py`, `test_razor_sommerfeld_ground.py`,
0.25 Ω at N = 192 against three in-house rows), ratified as final by the
same decision.
"""

import numpy as np
import pytest

from golden_razor_contact_nec5 import CONTACT_LADDERS
from golden_razor_pec_nec5 import PEC_LADDERS
from test_razor_ground_contact import INVL_A, MONO_LEN, _z
from test_razor_pec_ground import _razor_pec

# Certification ladders: ~12 s of GL solves, on the scrub's slow side of
# the line (momwire#400) — the fast loop keeps its budget, CI still runs
# these on every PR via the test-slow lane.
pytestmark = pytest.mark.slow

# Finest-rung pins: the measured N=96 residual + 25 % headroom, per
# geometry. A tightening mesh rate is welcome; a loosening one fails here.
_TOP_RUNG_BAR = {
    "dipole": 0.145,
    "fat-dipole": 0.163,
    "invvee": 0.685,
    "monopole": 0.037,
    "invl": 0.175,
}
_LOOP_ENVELOPE = 5.0

_CONTACT_GEOMS = {
    "monopole": (
        [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_LEN]])],
        lambda n: [[n]],
    ),
    "invl": (
        [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, INVL_A], [INVL_A, 0.0, INVL_A]])],
        lambda n: [[n // 2, n // 2]],
    ),
}


@pytest.fixture(scope="module")
def gl_lane():
    """Every gated point of the production lane, solved once: the four
    clearance ladders plus the two contact ladders on the default
    Gauss-Legendre path quadrature."""
    lane = {
        name: {n: _razor_pec(name, n) for n, *_ in rows}
        for name, rows in PEC_LADDERS.items()
    }
    for name, (wires, split) in _CONTACT_GEOMS.items():
        lane[name] = {
            n: complex(_z(wires, split(n), feed=0.0, ground=0.0)[0])
            for n, *_ in CONTACT_LADDERS[name]
        }
    return lane


def _residuals(name, lane):
    """(N, |Z_GL − oracle|) down the ladder, oracle per the unit-3/6
    decisions: the binary's printed answer for a clearance deck, its mirror
    deck halved for a contact one."""
    if name in PEC_LADDERS:
        return [(n, abs(lane[name][n] - z_nec5)) for n, z_nec5, *_ in PEC_LADDERS[name]]
    return [
        (n, abs(lane[name][n] - z_mirror))
        for n, _zc, z_mirror, *_ in CONTACT_LADDERS[name]
    ]


def test_the_gl_columns_are_what_razor_computes(gl_lane):
    """The golden modules' GL columns must still be live razor answers —
    the same anti-stale pin the sharp lanes carry, so no decay claim below
    can quietly gate a fossil."""
    for name, rows in PEC_LADDERS.items():
        for n, *_rest, z_gl in rows:
            assert abs(gl_lane[name][n] - z_gl) < 5e-6, f"{name} N={n}"
    for name, rows in CONTACT_LADDERS.items():
        for n, *_rest, z_gl in rows:
            assert abs(gl_lane[name][n] - z_gl) < 5e-6, f"{name} N={n}"


@pytest.mark.parametrize("name", sorted(_TOP_RUNG_BAR))
def test_production_residual_decays_down_the_ladder(name, gl_lane):
    """|Z_GL − oracle| strictly shrinks with N, finishing under the pinned
    top rung — the convergence claim, gated as a SHAPE rather than as a
    fixed offset (which is the other lane's claim, not this one's)."""
    res = _residuals(name, gl_lane)
    for (n_a, r_a), (n_b, r_b) in zip(res, res[1:]):
        assert r_b < r_a, f"{name}: residual grew {r_a:.3f} → {r_b:.3f} at N={n_b}"
    n_top, r_top = res[-1]
    assert r_top <= _TOP_RUNG_BAR[name], (
        f"{name}: finest rung (N={n_top}) residual {r_top:.3f} > "
        f"{_TOP_RUNG_BAR[name]} Ω pin"
    )


def test_the_loop_is_enveloped_not_gated(gl_lane):
    """The loop's GL residual is non-monotone (recorded: 1.78 → 4.54 → 3.11
    over the ladder) and the decision keeps that an honest record. The
    envelope still catches a regression an order above today's level."""
    res = _residuals("loop", gl_lane)
    for n, r in res:
        assert r <= _LOOP_ENVELOPE, f"loop N={n}: {r:.3f} > envelope"
    # ...and the non-monotonicity is real, so a future refactor that makes
    # it decay should retire this test for a decay pin, knowingly.
    values = [r for _n, r in res]
    assert any(b > a for a, b in zip(values, values[1:]))
