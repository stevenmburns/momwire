"""The point-matched crossing node's basis — momwire#1223 U2, the two-k recipe.

Decided node condition (the "correct" one, #1223): the current is
continuous and I'_above = I'_below/eps~, with eps~ = k_m²/k_p². SG imposes
nothing and lets it emerge from its Galerkin wings; collocation has no wings,
so the condition lives in the basis. A junction edge that crosses the
interface takes the neighbour constant w = a_i·k_j/k_i in place of NEC's
a_j, in its extension and in its P atom, and everything else runs at each
segment's own k.

These gates read the basis alone, through the sampler the crossing fill
uses. The solve is stage 2's later units: the point-matched lane still
refuses a crossing deck, so the geometry is reached through a subclass that
declares the crossing served.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _crossing_fill
from momwire.sinusoidal import SinusoidalBasisSampler, SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

C0 = 299792458.0
WL7 = C0 / 7e6

# SG's crossing deck (test_sg_crossing_980_d3): a vertical wire standing on a
# slanted buried one, both STARTING at the node.
BENT = (
    [[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
    [[0.0, 0.0, 0.0], [2.0, 0.0, -0.5]],
)
BENT_J = [[(0, "start"), (1, "start")]]
# A straight rod through the plane: the above wire ENDS at the node, the
# buried one starts there, so the other end/start pairing is covered too.
ROD = (
    [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]],
    [[0.0, 0.0, 0.0], [0.0, 0.0, -1.2]],
)
ROD_J = [[(0, "end"), (1, "start")]]

DECKS = {"bent": (BENT, BENT_J), "rod": (ROD, ROD_J)}
SOILS = [(4.0, 1e-3), (13.0, 0.005), (30.0, 0.05), (80.0, 5.0)]


class _Crossing(SinusoidalSolver):
    """The point-matched lane with its crossing refusal lifted, to reach the
    geometry and the basis; stage 2 lifts it on the real class."""

    def _serves_crossing(self):
        return True


def _eps_tilde(eps_r, sigma):
    return complex(eps_r, -sigma / (2 * np.pi * 7e6 * 8.8541878128e-12))


def _solver(deck, ground_eps, cls=_Crossing, n=9):
    wires, junctions = DECKS[deck]
    return cls(
        wires=list(wires),
        n_per_edge_per_wire=[[n], [n]],
        feeds=[(0, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        junctions=junctions,
        ground_z=0.0,
        ground_eps=ground_eps,
        ground_model="sommerfeld",
    )


def _two_k_view(s):
    geom = s._build_geometry()
    below = np.asarray(geom["seg_centers"])[:, 2] < s.ground_z
    med = _crossing_fill.buried_medium(s.ground_eps, s.omega, s.eps, s.k)
    k_seg = np.where(below, med.k_m, med.k_p)
    return geom, below, med, s._basis_coefs(geom, k_seg)


def _node_sides(geom, below):
    """(segment, arc u, outward tangent) of the node end on each side."""
    seg_l, seg_r = np.asarray(geom["seg_l"]), np.asarray(geom["seg_r"])
    h, tang = np.asarray(geom["seg_h"]), np.asarray(geom["seg_tangents"])
    out = {}
    for g in range(int(geom["n_segs"])):
        for u, pt, sign in ((0.0, seg_l[g], 1.0), (h[g], seg_r[g], -1.0)):
            if np.linalg.norm(pt) < 1e-9:
                out[bool(below[g])] = (g, u, sign * tang[g])
    return out[False], out[True]


def _at(sampler, geom, g, u):
    n = sampler.n_basis
    h = np.asarray(geom["seg_h"])
    sl, rows = sampler._entries(g)
    f, fd = sampler._value_and_slope(sl, np.array([u - 0.5 * h[g]]))
    v, d = np.zeros(n, complex), np.zeros(n, complex)
    v[rows], d[rows] = f[:, 0], fd[:, 0]
    return v, d


def _node_values(s):
    """Every basis's current and slope at the node, on each side, as the
    OUTWARD current (away from the node along that side's wire) and its
    derivative in the outward arc."""
    geom, below, med, view = _two_k_view(s)
    sampler = SinusoidalBasisSampler(view, med.k_p, geom["seg_h"], int(geom["n_segs"]))
    sides = []
    for g, u, out in _node_sides(geom, below):
        v, d = _at(sampler, geom, g, u)
        # The segment's own arc runs along its tangent; `out` is +tangent
        # when the node is at seg_l and -tangent at seg_r.
        sign = float(np.dot(out, np.asarray(geom["seg_tangents"])[g]))
        sides.append((sign * v, d))
    return med, sides


# --------------------------------------------------------------------------
# the node condition
# --------------------------------------------------------------------------


@pytest.mark.parametrize("deck", sorted(DECKS))
@pytest.mark.parametrize("soil", SOILS, ids=[f"er{e:g}-s{s:g}" for e, s in SOILS])
def test_the_current_is_continuous_and_the_slope_ratio_is_one_over_eps(deck, soil):
    """For every basis touching the node: the current flowing IN from the
    above side equals the current flowing OUT on the buried side, and the
    slope ratio along the path is 1/eps~, to rounding."""
    med, ((i_a, d_a), (i_b, d_b)) = _node_values(_solver(deck, _eps_tilde(*soil)))
    touching = np.nonzero((np.abs(i_a) > 0) | (np.abs(i_b) > 0))[0]
    assert touching.size >= 2
    scale = np.max(np.abs(i_a[touching]))
    # Outward on each side: inflow from above is -i_a, outflow below is i_b.
    np.testing.assert_allclose(
        -i_a[touching], i_b[touching], rtol=0, atol=1e-12 * scale
    )
    # Along the path (through the node from above to below), each side's
    # derivative is its own segment's arc derivative whichever end the node
    # is at: a node at seg_l reverses both the value and the arc on the
    # above side, a node at seg_r reverses both on the buried side.
    both = np.nonzero((np.abs(d_a) > 0) & (np.abs(d_b) > 0))[0]
    assert both.size >= 2  # the two node bases span the interface
    ratio = d_a[both] / d_b[both]
    target = med.k_p**2 / med.k_m**2
    np.testing.assert_allclose(ratio, target, rtol=1e-12)


@pytest.mark.parametrize("deck", sorted(DECKS))
def test_at_eps_one_the_recipe_is_the_ordinary_junction(deck):
    """k_m = k_p: every edge is within one medium, w = a_j, and the
    per-segment view is the scalar view entry for entry."""
    s = _solver(deck, (1.0, 0.0))
    geom, _below, med, view = _two_k_view(s)
    assert med.k_m == pytest.approx(med.k_p)
    ref = s._basis_coefs(geom, med.k_p)
    for key in ("starts", "jbasis", "sigma"):
        np.testing.assert_array_equal(view[key], ref[key])
    # The per-segment path runs in complex arithmetic where the scalar one is
    # real, so a sum or two lands an ulp or so apart (1.2e-14 measured).
    for key in ("A", "B", "C", "AC"):
        np.testing.assert_allclose(view[key], ref[key], rtol=1e-13, atol=1e-300)
    _med, ((i_a, d_a), (i_b, d_b)) = _node_values(s)
    t = np.nonzero((np.abs(d_a) > 0) & (np.abs(d_b) > 0))[0]
    assert t.size >= 2
    np.testing.assert_allclose(d_a[t] / d_b[t], 1.0, rtol=1e-12)


# --------------------------------------------------------------------------
# what did not move
# --------------------------------------------------------------------------


def test_a_detached_deck_is_the_stitched_view():
    """No edge crosses the interface on a detached deck, so the two-k view is
    D2's per-segment stitch."""
    s = SinusoidalSolver(
        wires=[
            [[0.0, 0.0, 1.0], [0.0, 0.0, 6.0]],
            [[-2.0, 0.0, -0.3], [2.0, 0.0, -0.3]],
        ],
        n_per_edge_per_wire=[[9], [9]],
        feeds=[(0, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=_eps_tilde(13.0, 0.005),
        ground_model="sommerfeld",
    )
    geom, below, med, view = _two_k_view(s)
    ref = s._stitch_basis_coefs(geom, below, med.k_p, med.k_m)
    for key in ("starts", "jbasis", "sigma"):
        np.testing.assert_array_equal(view[key], ref[key])
    for key in ("A", "B", "C", "AC", "k_entry"):
        np.testing.assert_allclose(view[key], ref[key], rtol=1e-14, atol=1e-300)


def test_the_node_is_ordinary_on_this_lane_and_c0_on_sg():
    """The family hook: the point-matched lane gives the crossing node
    junction edges and no C0 flags; SG keeps its C0 node with no edges."""
    eps = _eps_tilde(13.0, 0.005)
    ss = _solver("bent", eps)._build_geometry()
    sg = _solver("bent", eps, cls=SinusoidalGalerkinSolver)._build_geometry()
    assert not ss["crossing_minus"].any() and not ss["crossing_plus"].any()
    assert int(np.sum(ss["nm_count"]) + np.sum(ss["np_count"])) > int(
        np.sum(sg["nm_count"]) + np.sum(sg["np_count"])
    )
    assert sg["crossing_minus"].any() or sg["crossing_plus"].any()


def test_the_real_class_still_refuses_a_crossing_deck_until_stage_2():
    """U2 is the basis only: the solve is not served yet, and the node is not
    touched on the real class (its crossing set is still empty)."""
    s = _solver("bent", _eps_tilde(13.0, 0.005), cls=SinusoidalSolver)
    assert s._crossing_junction_indices() == frozenset()
    with pytest.raises(NotImplementedError, match="crossing junction"):
        s.compute_impedance()
