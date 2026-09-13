"""A TWO-RADIUS crossing node (antennaknobs plan U5).

A crossing deck whose above wires share one radius and whose below wires share
another is served by `BSplineSolver`. The rule and its evidence are in the
scratch record (`scratch/u5-mixed-radius/`, momwire#1049):

* every line test takes its observer wire's radius, and every point test AT
  the crossing node — both families' node rows — takes the buried radius. The
  observer-side reading instead evaluates the node's point tests on two
  surfaces, which leaves the thin-wire potential's jump across the node
  uncounted. That puts it 10–97 Ω from NEC-5 on a two-radius rod ladder, and
  the one-potential rule inside the equal-radius band everywhere;
* the crossing junction keeps its KCL row. At two radii the split fill's own
  continuity does not converge under refinement, so the multiplier closes it.

The deck is check 2's momwire-native rod, `crossing_deck(2)`: 2 m below, 10 m
above, fed at 4.33 m, soil A with Sommerfeld ground, graded toward the node.
Radii are per wire, `[below, above]`.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _crossing_fill
from momwire import razor as _razor
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

from test_crossing_serve_524 import crossing_deck, fan_rise_deck

A = 0.25e-3


def _rod(a_above, a_below):
    return BSplineSolver(**crossing_deck(2, wire_radius=[a_below, a_above]))


def _kcl_A(s):
    return s._build_basis_polynomials(s._build_geometry())[2]


@pytest.fixture(scope="module")
def two_radius_rod():
    s = _rod(A, A / 2)
    z, coeffs = s.compute_impedance()
    return s, complex(z), coeffs


def test_u5_1_equal_radii_take_the_shipped_path(monkeypatch):
    """One radius: the two-radius fill is never entered and the crossing
    junction carries no KCL row, exactly as before U5."""

    def entered(*_a, **_k):
        raise AssertionError("the two-radius fill was entered at equal radii")

    monkeypatch.setattr(_crossing_fill, "cross_complete_blocks_two_radius", entered)
    monkeypatch.setattr(_crossing_fill, "self_completions_two_radius", entered)
    s = _rod(A, A)
    assert s._two_radius_crossing() is None
    assert _kcl_A(s).shape[0] == 0
    z, _ = s.compute_impedance()
    assert np.isfinite(complex(z))


def test_u5_2_a_two_radius_node_is_served_with_a_kcl_row(two_radius_rod):
    s, _z, coeffs = two_radius_rod
    assert s._two_radius_crossing() == (A, A / 2)
    kcl = _kcl_A(s)
    assert kcl.shape[0] == 1
    nz = np.flatnonzero(kcl[0])
    assert sorted(kcl[0, nz].tolist()) == [-1.0, 1.0]
    below_len = 2.0
    cur = s.currents_at_knots(coeffs, s_array=[np.array([below_len]), np.array([0.0])])
    i_m, i_p = complex(cur[0][0]), complex(cur[1][0])
    assert abs(i_p - i_m) / abs(i_p) <= 1e-9


def test_u5_3_the_z_guard_reads_the_physics():
    """Thinning the BURIED member raises R by several ohms; thinning the ABOVE
    member by the same factor barely moves it; and the rise's response is
    logarithmic in the radius ratio.

    That is the physics of a short-circuit through lossy ground: the buried
    member's own log term sets the grounding resistance. The two-surface
    reading swaps the pattern — on this deck it answers −0.13 Ω for the
    thinned rise and +8.59 Ω for the thinned top, where the served rule
    answers +7.83 and +0.51 (the scratch record's check 4, r = 2). The bars
    are ordering and ratios with tolerances, not literals, so a legitimate
    re-pin of the fill does not break them.
    """

    def r(a_above, a_below):
        return complex(_rod(a_above, a_below).compute_impedance()[0]).real

    r0 = r(A, A)
    rise2 = r(A, A / 2) - r0
    rise4 = r(A, A / 4) - r0
    top2 = r(A / 2, A) - r0
    assert rise2 >= 3.0, rise2
    assert abs(top2) <= 0.25 * rise2, (top2, rise2)
    assert 1.7 <= rise4 / rise2 <= 2.3, (rise4, rise2)


def test_u5_4_a_fans_kcl_row_covers_every_member():
    """A node fan (two buried radials at one radius, the monopole at another):
    the crossing junction's row carries every member's directional basis,
    each with its outflow sign (every member meets the node at its END)."""
    s = BSplineSolver(**fan_rise_deck(n_radials=2, wire_radius=[A / 2, A / 2, A]))
    assert s._two_radius_crossing() == (A, A / 2)
    kcl = _kcl_A(s)
    assert kcl.shape[0] == 1
    nz = np.flatnonzero(kcl[0])
    assert kcl[0, nz].tolist() == [-1.0, -1.0, -1.0]


def test_u5_5a_a_radius_spread_within_a_side_is_refused_by_name():
    s = BSplineSolver(**fan_rise_deck(n_radials=2, wire_radius=[A / 2, A / 4, A]))
    with pytest.raises(NotImplementedError, match="differ within the below wires"):
        s._crossing_junctions()


def test_u5_5c_razor_still_refuses_a_mixed_radius_crossing(monkeypatch):
    """Only BSpline opts into the two-radius node: razor's crossing fill was
    not measured under the rule, so the shared scope check keeps its
    one-radius refusal for it."""
    monkeypatch.setattr(_razor, "_SERVE_CROSSING", True)
    deck = crossing_deck(2, wire_radius=[A / 2, A])
    with pytest.raises(NotImplementedError, match="per-wire radii"):
        RazorSolver(**deck, n_qp_path=8)._crossing_junctions()


def test_u5_6_the_multiplier_reaches_the_y_matrix(two_radius_rod):
    s, z, _coeffs = two_radius_rod
    y = complex(np.asarray(s.compute_y_matrix()).reshape(-1)[0])
    assert abs(1.0 / y - z) <= 1e-12 * abs(z)
