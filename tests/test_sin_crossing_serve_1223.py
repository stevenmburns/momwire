"""The point-matched lane serves the crossing node — momwire#1223 U5.

The two-k basis (U2) and the point-observer cross rows (U4), wired into the
solve. The node condition is imposed (I continuous, I'_above =
I'_below/eps~); the class blocks are stage 1's; a crossing deck takes no
transmitted grid.

Gates:
  * eps~ = 1: the crossing deck's impedance and basis coefficients are the
    free-space ordinary junction's (measured 1e-8..1e-7 on Z, well inside
    SG's own 1.4e-6 floor on its collapse);
  * soil: the current is continuous at the node in the SOLUTION, and the
    lane closes on SG under refinement (the convergence ladders are U6);
  * scope (decision 6): one two-member node per deck. A multi-node deck and
    a buried hub are refused by name, and so is a dense block past its size
    (decision 5).
"""

from __future__ import annotations

import numpy as np
import pytest
from test_sin_crossing_recipe_1223 import DECKS, WL7, _eps_tilde

from momwire import sinusoidal as sin_mod
from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

SOIL_A = _eps_tilde(13.0, 0.005)


def _make(cls, deck, eps, n, ground=True):
    wires, junctions = DECKS[deck]
    g = dict(ground_z=0.0, ground_eps=eps, ground_model="sommerfeld") if ground else {}
    return cls(
        wires=list(wires),
        n_per_edge_per_wire=[[n], [n]],
        feeds=[(0, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        junctions=junctions,
        **g,
    )


@pytest.mark.parametrize("n", [9, 15])
@pytest.mark.parametrize("deck", sorted(DECKS))
def test_at_eps_one_the_crossing_deck_is_the_ordinary_junction(deck, n):
    crossing = _make(SinusoidalSolver, deck, (1.0, 0.0), n).compute_port_solution()
    free = _make(SinusoidalSolver, deck, None, n, ground=False).compute_port_solution()
    z_x, z_f = complex(np.ravel(crossing.y)[0]), complex(np.ravel(free.y)[0])
    assert abs(z_x - z_f) <= 1e-6 * abs(z_f)
    a_x, a_f = np.asarray(crossing.coeffs).ravel(), np.asarray(free.coeffs).ravel()
    assert np.max(np.abs(a_x - a_f)) <= 1e-6 * np.max(np.abs(a_f))


@pytest.mark.parametrize("deck", sorted(DECKS))
def test_the_solved_current_is_continuous_at_the_node(deck):
    """Continuity is imposed in the basis (U2), so it holds in any solution:
    the current flowing into the node on the above wire leaves it on the
    buried one."""
    s = _make(SinusoidalSolver, deck, SOIL_A, 9)
    alpha = np.asarray(s.compute_port_solution().coeffs).ravel()
    per_wire = s.currents_at_knots(alpha)
    (w_a, e_a), (w_b, e_b) = DECKS[deck][1][0]
    i_a = per_wire[w_a][0 if e_a == "start" else -1]
    i_b = per_wire[w_b][0 if e_b == "start" else -1]
    # Along each wire's own direction: a member STARTING at the node carries
    # its current away from it, one ENDING there carries it in.
    out_a = i_a if e_a == "start" else -i_a
    out_b = i_b if e_b == "start" else -i_b
    assert abs(out_a + out_b) <= 1e-10 * abs(i_a)


@pytest.mark.parametrize("deck", sorted(DECKS))
def test_the_lane_closes_on_sg_under_refinement(deck):
    """Collocation and Galerkin agree in the continuum, not at a fixed mesh:
    the gap shrinks on every rung (measured 1.4 % → 0.9 % → 0.6 % on the bent
    deck at n = 9 / 15 / 25). U6 owns the ladders and the bars; this pins the
    trend and that the two stay in the same class."""
    gaps = []
    for n in (9, 15, 25):
        zs = complex(_make(SinusoidalSolver, deck, SOIL_A, n).compute_impedance()[0])
        zg = complex(
            _make(SinusoidalGalerkinSolver, deck, SOIL_A, n).compute_impedance()[0]
        )
        gaps.append(abs(zs - zg) / abs(zg))
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 0.01


def _ground():
    return dict(
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
        wavelength=WL7,
        wire_radius=0.001,
    )


def test_a_multi_node_deck_is_refused_by_name():
    s = SinusoidalSolver(
        wires=[
            [[0, 0, 2.0], [0, 0, 0.0]],
            [[0, 0, 0.0], [0, 0, -1.0]],
            [[5, 0, 2.0], [5, 0, 0.0]],
            [[5, 0, 0.0], [5, 0, -1.0]],
        ],
        n_per_edge_per_wire=[[7]] * 4,
        feeds=[(0, 1.0, 1 + 0j)],
        junctions=[[(0, "end"), (1, "start")], [(2, "end"), (3, "start")]],
        **_ground(),
    )
    with pytest.raises(NotImplementedError) as err:
        s.compute_impedance()
    assert str(err.value).endswith(sin_mod._CROSSING_MULTI_NODE_REFUSAL)
    assert (
        SinusoidalSolver.capabilities.refusal("buried", "crossing_multi_node")
        == sin_mod._CROSSING_MULTI_NODE_REFUSAL
    )


def test_a_buried_hub_is_refused_by_name():
    s = SinusoidalSolver(
        wires=[
            [[0, 0, 2.0], [0, 0, 0.0]],
            [[0, 0, 0.0], [2, 0, -0.3]],
            [[0, 0, 0.0], [-2, 0, -0.3]],
        ],
        n_per_edge_per_wire=[[7]] * 3,
        feeds=[(0, 1.0, 1 + 0j)],
        junctions=[[(0, "end"), (1, "start"), (2, "start")]],
        **_ground(),
    )
    with pytest.raises(NotImplementedError) as err:
        s.compute_impedance()
    assert str(err.value).endswith(sin_mod._CROSSING_HUB_REFUSAL)
    assert (
        SinusoidalSolver.capabilities.refusal("buried", "crossing_hub")
        == sin_mod._CROSSING_HUB_REFUSAL
    )


def test_a_dense_block_past_its_size_is_refused_by_name(monkeypatch):
    monkeypatch.setattr(sin_mod, "_CROSSING_POINT_PAIRS_MAX", 10)
    with pytest.raises(NotImplementedError, match="momwire#1224"):
        _make(SinusoidalSolver, "bent", SOIL_A, 9).compute_impedance()


def test_sg_keeps_serving_what_this_lane_refuses():
    """The scope limits are the point-matched lane's, not SG's."""
    s = SinusoidalGalerkinSolver(
        wires=[
            [[0, 0, 2.0], [0, 0, 0.0]],
            [[0, 0, 0.0], [0, 0, -1.0]],
            [[5, 0, 2.0], [5, 0, 0.0]],
            [[5, 0, 0.0], [5, 0, -1.0]],
        ],
        n_per_edge_per_wire=[[7]] * 4,
        feeds=[(0, 1.0, 1 + 0j)],
        junctions=[[(0, "end"), (1, "start")], [(2, "end"), (3, "start")]],
        **_ground(),
    )
    assert np.isfinite(complex(s.compute_impedance()[0]))
