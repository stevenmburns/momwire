"""razor-2p on decks with SEVERAL crossing nodes — momwire#1149 U2b-3.

The shared crossing scope has served any number of crossing junctions from
`MIN_CROSSING_NODE_SEPARATION_M` apart since antennaknobs plan U9, and razor's
scan and assembly never assumed one: `_crossing_tents` lists every tent that
spans the plane and `_crossing_node_charges` evaluates every row endpoint
against every tent's node, so a second node is a second column of point
charges and the cross-node pairs are the same formula at a larger separation.
What U2b-3 adds is the gates, razor's exact buried pre-flight
(`buried_serve_refusal`, for antennaknobs#1464) and razor's `CoarseCrossingNode`
advisory.

Measured 2026-09-22 (`scratch/razor-buried-u2b/`, probe 3):

  * two tents counted, one node-term call carrying both; the own-node block of
    a two-node deck is the one-rod deck's to rounding, and the two cross-node
    blocks equal each other (the deck's translation symmetry);
  * eps~ = 1 collapse 6.8e-13 on both lanes;
  * 2-port non-reciprocity with ASYMMETRIC ports (above on rod 1, buried on rod
    2; like ports would make Y12 = Y21 by symmetry): 9.4e-4 -> 2.3e-4 ->
    5.9e-5 (4.05 / 3.95), and 2.8e-3 -> 7.2e-4 -> 1.8e-4 (3.87 / 3.90) on a
    hub screen and a three-leg fan 6 m apart. With the node term zeroed both
    are flat (4.3e-2, 0.34);
  * convergence onto bspline's U9 route, every entry of Z: ratios 0.48-0.58
    per doubling on both decks;
  * **the grazing floor is not at parity with bspline's, and is not meant to
    be.** Each fill's floor is a property of the points IT evaluates. bspline's
    crossing axes are graded on the a-scale into the plane, so its below/below
    pairs reach the floor at 12 / 6 / 3 m (x1 / x2 / x4) on `two_node_deck`;
    razor's remainder pairs (testing-path points x Gauss nodes) stay deeper and
    reach it at 128 / 64 / 48 m. Razor's pre-flight asks exactly those pairs,
    and matches its own fill everywhere probed.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _crossing_fill  # noqa: E402
from momwire import razor as _razor  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import (  # noqa: E402
    A_WIRE,
    SOIL_A,
    WL7,
    crossing_deck,
    hub_deck,
    two_node_deck,
)
from test_razor_detached_1149 import detached  # noqa: E402


def refined(deck, m):
    deck["n_per_edge_per_wire"] = [
        [n * m for n in e] for e in deck["n_per_edge_per_wire"]
    ]
    return deck


def two(m=1, sep=2.0, **kw):
    """`two_node_deck`, ports above on rod 1 (4.5) and BURIED on rod 2 (1.0)."""
    d = two_node_deck(sep, **kw)
    d["feeds"] = [(1, 4.5, 1 + 0j), (2, 1.0, 1 + 0j)]
    return refined(d, m)


def hub_fan(m=1, sep=6.0):
    """Node 1: `hub_deck(2)` (two radials at 0.15 m into a hub, one rise, the
    10 m mast). Node 2, `sep` m along x: a 10 m mast over three 2 m legs
    leaning 45 deg down and out, mast first, so all three of its tents cross.
    Ports: mast 1 at 4.0 and leg 0 of the fan at 1.0. 6 m is where bspline's
    below grazing floor still serves the x4 rung (8 m does not)."""
    h = hub_deck(n_radials=2)
    wires = list(h["wires"])
    npe = [list(e) for e in h["n_per_edge_per_wire"]]
    junctions = [list(j) for j in h["junctions"]]
    mast1 = len(wires) - 1
    shift = np.array([sep, 0.0, 0.0])
    mast2 = len(wires)
    wires.append(np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)]) + shift)
    npe.append([20])
    s = 2.0 / np.sqrt(2.0)
    legs = []
    for a in (0.3, 0.3 + 2 * np.pi / 3, 0.3 + 4 * np.pi / 3):
        legs.append(len(wires))
        wires.append(
            np.array([(s * np.cos(a), s * np.sin(a), -s), (0.0, 0.0, 0.0)]) + shift
        )
        npe.append([8])
    junctions.append([(i, "end") for i in legs] + [(mast2, "start")])
    d = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=junctions,
        feeds=[(mast1, 4.0, 1 + 0j), (legs[0], 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    return refined(d, m)


def razor(d, **kw):
    kw.setdefault("nec5_quadrature", True)
    return RazorSolver(**{k: v for k, v in d.items() if k != "junctions"}, **kw)


def nonrec(d):
    Y = np.asarray(razor(d).compute_y_matrix())
    return abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])


def ratios(xs):
    return [a / b for a, b in zip(xs, xs[1:])]


def node_term(s):
    geom = s._build_geometry()
    tents = s._crossing_tents(geom)
    ctx = s._crossing_context(geom, k=s.k, omega=s.omega)
    A = s._crossing_path_axis(geom, tents, "above")
    P = s._crossing_path_axis(geom, tents, "below")
    return geom, tents, s._crossing_node_charges(geom, tents, ctx, A, P, s.omega)


@pytest.fixture
def without_the_node_term(monkeypatch):
    monkeypatch.setattr(
        RazorSolver,
        "_crossing_node_charges",
        lambda self, g, t, *a, **k: np.zeros(
            (g["n_basis_total"], len(t)), dtype=np.complex128
        ),
    )


# ----------------------------------------------------------------------
# the route, counted, and the node term per node
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, deck, n_tents",
    [("two", two(), 2), pytest.param("hub/fan", hub_fan(), 4, marks=pytest.mark.slow)],
)
def test_every_node_is_a_column_of_the_one_node_term(monkeypatch, name, deck, n_tents):
    """Through the REAL constructor: both nodes labelled, every crossing tent
    counted, and ONE node-term call carrying all of them."""
    calls = []
    orig = RazorSolver._crossing_node_charges

    def counted(self, geom, tents, *a, **k):
        calls.append(len(tents))
        return orig(self, geom, tents, *a, **k)

    monkeypatch.setattr(RazorSolver, "_crossing_node_charges", counted)
    s = razor(deck)
    assert s._crossing and len(s._crossing_junctions()) == 2
    assert len(s._crossing_tents(s._build_geometry())) == n_tents
    z = np.asarray(s.compute_impedance()[0])
    assert np.all(np.isfinite(z))
    assert calls == [n_tents]


def test_the_own_node_block_is_the_one_rod_decks():
    """The term is a function of (row endpoint, node) alone, so rod 1's rows
    against rod 1's tent are the one-rod deck's entries, and the second node
    only ADDS its own column — the cross-node pairs."""
    one = crossing_deck(1)
    s1 = razor(one)
    g1, t1, n1 = node_term(s1)
    s2 = razor(two_node_deck(2.0))
    g2, t2, n2 = node_term(s2)
    rows1 = np.arange(np.asarray(g1["basis_offsets"])[2])  # rod 1's own tents
    rows2 = np.arange(np.asarray(g2["basis_offsets"])[2])
    assert rows1.size == rows2.size
    own = n2[np.ix_(rows2, [0])]
    ref = n1[np.ix_(rows1, [0])]
    assert np.max(np.abs(own - ref)) <= 1e-12 * np.max(np.abs(ref))
    # the crossing tents' own rows too (they sit after the wire bases)
    assert abs(n2[t2[0][0], 0] - n1[t1[0][0], 0]) <= 1e-12 * abs(n1[t1[0][0], 0])


def test_the_cross_node_blocks_are_symmetric_and_fall_with_separation():
    """Rod 2's rows seeing rod 1's node charge, and rod 1's rows seeing rod
    2's, are the same numbers by the deck's translation symmetry, and they
    fall with separation. They are NOT small next to the own-node block:
    the term is the remainder's potential, whose 1/R parts cancel at every
    distance, so the own node (R = a) and a node 2 m away read the same
    order (peak 2.38 vs 1.77)."""
    peaks = []
    for sep in (2.0, 8.0):
        s = razor(two_node_deck(sep))
        g, tents, n = node_term(s)
        off = np.asarray(g["basis_offsets"])
        r1, r2 = np.arange(off[0], off[2]), np.arange(off[2], off[4])
        c12, c21 = n[np.ix_(r2, [0])], n[np.ix_(r1, [1])]
        assert np.max(np.abs(c12 - c21)) <= 1e-12 * np.max(np.abs(c12))
        assert np.all(np.abs(c12) > 0)
        peaks.append(float(np.max(np.abs(c12))))
    assert peaks[1] < peaks[0], peaks


# ----------------------------------------------------------------------
# reference-free gates
# ----------------------------------------------------------------------


@pytest.mark.parametrize("lane", [True, False], ids=["2pt", "gl"])
def test_eps_one_collapses_to_razors_free_space_fill(lane):
    """Measured 6.8e-13 on both lanes."""
    d1 = two(1, ground_eps=(1.0, 0.0))
    df = {
        k: v
        for k, v in d1.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    s1, sf = razor(d1, nec5_quadrature=lane), razor(df, nec5_quadrature=lane)
    g1, gf = s1._build_geometry(), sf._build_geometry()
    assert s1._crossing and np.array_equal(g1["wing_seg"], gf["wing_seg"])
    Z1, Zf = s1._assemble_Z(g1, s1.k), sf._assemble_Z(gf, sf.k)
    assert float(np.max(np.abs(Z1 - Zf)) / np.max(np.abs(Zf))) < 1e-9


def test_reciprocity_decays_with_asymmetric_ports():
    """Above on rod 1, buried on rod 2: 4.05 / 3.95 measured."""
    xs = [nonrec(two(m)) for m in (1, 2, 4)]
    assert all(r >= 3.0 for r in ratios(xs)), (xs, ratios(xs))


def test_without_the_node_term_it_is_flat(without_the_node_term):
    """The red control: 4.26e-2 -> 4.18e-2."""
    xs = [nonrec(two(m)) for m in (1, 2)]
    assert all(x > 1e-2 for x in xs) and ratios(xs)[0] < 1.2, xs


@pytest.mark.slow
def test_reciprocity_decays_on_the_hub_fan_pair():
    """3.87 / 3.90 measured (0.34, flat, with the node term zeroed)."""
    xs = [nonrec(hub_fan(m)) for m in (1, 2, 4)]
    assert all(r >= 3.0 for r in ratios(xs)), (xs, ratios(xs))


@pytest.mark.slow
def test_reciprocity_decays_where_bspline_refuses():
    """12 m apart: bspline's grazing floor refuses every rung from x1, razor's
    serves them, and they are healthy by the reference-free gate."""
    assert BSplineSolver(**two(1, 12.0)).buried_serve_refusal() is not None
    xs = [nonrec(two(m, 12.0)) for m in (1, 2, 4)]
    assert all(r >= 3.0 for r in ratios(xs)), (xs, ratios(xs))


@pytest.mark.slow
@pytest.mark.parametrize("name, mk", [("two", two), ("hub/fan", hub_fan)])
def test_z_converges_onto_bsplines_u9_route(name, mk):
    """Every entry of the 2-port Z, gap shrinking per doubling: measured
    0.48-0.58."""
    gaps = []
    for m in (1, 2, 4):
        Yr = np.asarray(razor(mk(m)).compute_y_matrix())
        Yb = np.asarray(BSplineSolver(**mk(m)).compute_y_matrix())
        gaps.append(np.abs(np.linalg.inv(Yr) - np.linalg.inv(Yb)))
    for i, j in ((0, 0), (1, 1), (0, 1)):
        g = [x[i, j] for x in gaps]
        assert all(r <= 0.65 for r in [b / a for a, b in zip(g, g[1:])]), (i, j, g)


# ----------------------------------------------------------------------
# the pre-flight
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "deck",
    [two(), hub_fan(), crossing_deck(1), detached(), None],
    ids=["two", "hub/fan", "crossing", "detached", "free"],
)
def test_the_preflight_serves_what_the_fill_serves(deck):
    if deck is None:
        deck = {
            k: v
            for k, v in detached().items()
            if k not in ("ground_z", "ground_eps", "ground_model")
        }
    assert razor(deck).buried_serve_refusal() is None


@pytest.mark.parametrize("m, sep", [(1, 128.0), (2, 64.0), (4, 48.0)])
def test_the_preflight_refuses_with_the_fills_own_sentence(m, sep):
    """At the first separation razor refuses on each rung — and before
    U2b-3 the first two of these passed the endpoint plan and died inside
    the grid instead, in the grid's words, where a pre-flight asking the plan
    would have said "served" (probe 3g)."""
    s = razor(two(m, sep))
    pre = s.buried_serve_refusal()
    assert pre is not None and "grazing floor" in pre
    with pytest.raises(ValueError) as exc:
        s._assemble_Z(s._build_geometry(), s.k)
    assert str(exc.value) == pre


def test_the_endpoint_plan_alone_passed_the_deck_the_grid_refuses(monkeypatch):
    """The red control for the gate above: with the evaluated-pairs half
    removed, the pre-flight says "served" at 64 m x2 and the fill then
    refuses in the GRID's words."""
    monkeypatch.setattr(
        RazorSolver, "_below_remainder_th_min", lambda self, geom: (np.inf, 0.0)
    )
    s = razor(two(2, 64.0))
    assert s.buried_serve_refusal() is None
    with pytest.raises(ValueError, match="below/below grid tabulated from theta"):
        s._assemble_Z(s._build_geometry(), s.k)


def test_the_preflight_names_a_buried_jacket_and_serves_bare_loading():
    """Bare-metal loading on a crossing deck is served since momwire#1149
    U3, so the pre-flight says None for it; a jacket on the buried wire is
    what it still names, in the fill's own words."""
    assert (
        razor(crossing_deck(1), wire_conductivity=3.5e7).buried_serve_refusal() is None
    )
    s = razor(
        crossing_deck(1),
        wire_conductivity=3.5e7,
        insulation_radius=[0.002, np.nan],
        insulation_eps_r=[3.0, np.nan],
    )
    assert s.buried_serve_refusal() == _razor._CROSSING_BURIED_JACKET_REFUSAL
    with pytest.raises(NotImplementedError) as exc:
        s.compute_impedance()
    assert str(exc.value) == _razor._CROSSING_BURIED_JACKET_REFUSAL


def test_grazing_floor_parity_is_not_claimed():
    """The measured fact the module docstring states: on `two_node_deck` at
    x1, 12 m is refused by bspline and served by razor."""
    d = two(1, 12.0)
    assert BSplineSolver(**d).buried_serve_refusal() is not None
    assert razor(d).buried_serve_refusal() is None


# ----------------------------------------------------------------------
# what stays refused, and the advisory
# ----------------------------------------------------------------------


def test_two_radii_with_several_nodes_stay_refused_by_name():
    radii = [2.0 * A_WIRE, A_WIRE, 2.0 * A_WIRE, A_WIRE]
    with pytest.raises(NotImplementedError, match="several nodes has no measured"):
        razor(two_node_deck(wire_radius=radii))


def test_a_coarse_node_draws_razors_advisory():
    """crossing_deck(1)'s 50 mm node is above the shared 25 mm bar, so razor
    raises `CoarseCrossingNode` where bspline does — with razor's sentences
    (what the node costs a path-tested fill, and no n_qp_pair lever)."""
    with pytest.warns(_crossing_fill.CoarseCrossingNode) as rec:
        razor(crossing_deck(1)).compute_impedance()
    node = [r for r in rec if issubclass(r.category, _crossing_fill.CoarseCrossingNode)]
    assert len(node) == 1
    msg = str(node[0].message)
    assert _razor._RAZOR_NODE_WORTH in msg and _razor._RAZOR_NODE_LEVERS in msg
    assert "n_qp_pair" not in msg


@pytest.mark.parametrize(
    "deck", [refined(crossing_deck(1), 4), detached()], ids=["graded", "detached"]
)
def test_a_resolved_node_or_no_node_draws_none(deck):
    with warnings.catch_warnings():
        warnings.simplefilter("error", _crossing_fill.CoarseCrossingNode)
        razor(deck).compute_impedance()
