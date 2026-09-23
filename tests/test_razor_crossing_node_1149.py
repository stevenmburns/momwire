"""razor-2p's CROSSING node — momwire#1149 U2.

A crossing deck carries a junction IN the interface: a buried member and an
above member meeting at a node on the plane (a bonded ground screen, a
ground rod). Razor fills it as the four-term crossing assembly
(`_assemble_Z_crossing`), splitting every tent that spans the plane into two
half tents, one per medium. Until U2 that assembly converged to a
NON-reciprocal limit carrying a spurious node resistance (+9.9 ohm against
bspline on `crossing_deck`, +20.7 on the four-radial hub).

**The mechanism.** A half tent carries its current up to the node and stops,
so it implies a point charge there; the two halves' charges are equal and
opposite, so the whole tent's is zero and every term of the assembly may
carry both or drop both. All of them dropped it — the families' direct and
image terms spell a half tent's charge as its wing's doublet, and the trunk's
cross blocks do too on a path axis — except the families' Sommerfeld
remainder, which is the field of the half tent's CURRENT and so carries the
node charge with the rest. `RazorSolver._crossing_node_charges` takes it back
out: per row, the node charge's direct+image potential minus its exact
(transmitted V) potential at the row's T2 endpoints, which is minus the
remainder's own share. It is identically zero at eps~ = 1, which is why the
collapse gates could never see the defect.

The gates are REFERENCE-FREE or convergence ones (the 2026-09-22 decision on
momwire#1149). Every ladder here refines EVERY edge, node edges included: a
ladder that holds the node-adjacent edges fixed (the scoping's probe 2) holds
the node's own discretisation error fixed, and its non-reciprocity floors at
~7e-6 for that reason alone (`scratch/razor-buried-u2/`, probes 3-8). Measured
2026-09-22:

  * **2-port non-reciprocity decays ~4x per doubling** on `crossing_deck`
    (graded; ports at above 4.5 / buried 1.0): soil A 6.45e-4 -> 1.56e-4 ->
    3.67e-5 -> 8.65e-6 -> 2.06e-6 (x1..x16), lossless eps_r 13 at 3.5-4.3x,
    eps_r 80 at 4.0x, lossy eps_r 13 at 4.0-4.4x. Without the term it is flat
    at 3.2e-2 / 2.6e-2 / 0.11. The same on a 30-degree bent deck (4.2-4.6x),
    at a node carrying three crossing tents (4.0x) and on a two-node deck
    (4.0x).
  * **the driving point converges onto bspline** (feed on a knot at every
    rung): `crossing_deck` |razor - bspline| 6.78 -> 3.19 -> 1.68 -> 0.95 ->
    0.57 ohm (x1..x16), hub_deck(4) 7.47 -> 3.96 -> 2.23 -> 1.30 (x1..x8), the
    catalog buried_radial_vertical 1.86 -> 0.81 -> 0.42 (x1..x4). Without the
    term the gap is flat at +9.9 / +20.7 ohm of resistance.
  * **antennaknobs' power balance** on buried_radial_vertical, N = 1/2/4/8:
    eta <= 1 and monotone, eta*R within 0.36-0.39 % of bspline (was
    0.67-0.85 %), R_in -0.24..-0.32 ohm off bspline (was +3.0..+7.3).

Instrument, not a gate (momwire#1149 decision 2): at equal mesh against the
licensed reference (verified against our licensed materials), razor's
driving point on `crossing_deck` is within 0.012-0.036 ohm at x4..x16, and
its dR across the contrast sweep (eps_r 1.1 to 80, lossy and lossless) is
<= 0.14 ohm where it was +6.7 / +13.1 / +31.7 at eps_r 13 / 30 / 80.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

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

LOSSLESS_13 = (13.0, 0.0)
LOSSLESS_80 = (80.0, 0.0)


def refined(deck, m):
    """Every edge count times `m`, node edges included."""
    deck["n_per_edge_per_wire"] = [
        [n * m for n in e] for e in deck["n_per_edge_per_wire"]
    ]
    return deck


def two_port_crossing(m, eps=SOIL_A, lean=0.0):
    """`crossing_deck(1)` refined x`m`, ports on knots at every rung: above
    arclength 4.5 and buried arclength 1.0. `lean` tilts the above wire in x
    and the buried wire the other way in y, arclength-preserving."""
    d = crossing_deck(1, ground_eps=eps)
    if lean:
        t = np.radians(lean)
        below, above = d["wires"]
        d["wires"] = [
            np.array([(0.0, -np.sin(t) * z, np.cos(t) * z) for z in below[:, 2]]),
            np.array([(np.sin(t) * z, 0.0, np.cos(t) * z) for z in above[:, 2]]),
        ]
    d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
    return refined(d, m)


def tripod(m, monopole_first):
    """A 10 m monopole over three buried legs leaning 45 deg down and out.
    Razor pairs every member at a node against the FIRST wire there, so with
    the monopole first all three tents cross the plane; with a leg first one
    crosses and two are below-family tents. Ports: monopole 4.0, leg 1.0."""
    s = 2.0 / np.sqrt(2.0)
    legs = [
        np.array([(s * np.cos(a), s * np.sin(a), -s), (0.0, 0.0, 0.0)])
        for a in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)
    ]
    mono = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)])
    if monopole_first:
        wires, im, il = [mono, *legs], 0, [1, 2, 3]
    else:
        wires, im, il = [*legs, mono], 3, [0, 1, 2]
    d = dict(
        wires=wires,
        n_per_edge_per_wire=[[20] if i == im else [8] for i in range(4)],
        junctions=[[(i, "end") for i in il] + [(im, "start")]],
        feeds=[(im, 4.0, 1 + 0j), (il[0], 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    return refined(d, m)


def razor(deck, **kw):
    d = {k: v for k, v in deck.items() if k != "junctions"}
    return RazorSolver(**d, nec5_quadrature=True, **kw)


def nonrec(deck):
    Y = np.asarray(razor(deck).compute_y_matrix())
    return abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])


def ratios(xs):
    return [a / b for a, b in zip(xs, xs[1:])]


@pytest.fixture
def without_the_node_term(monkeypatch):
    """The pre-U2 fill: the node term zeroed, everything else untouched."""

    def zero(self, geom, tents, *a, **kw):
        return np.zeros((geom["n_basis_total"], len(tents)), dtype=np.complex128)

    monkeypatch.setattr(RazorSolver, "_crossing_node_charges", zero)


# ----------------------------------------------------------------------
# the route, counted
# ----------------------------------------------------------------------


def test_a_crossing_deck_is_served_and_takes_the_node_term(monkeypatch):
    """Through the REAL constructor: the crossing route, the node term called
    once per fill with the deck's one crossing tent, and a finite answer."""
    calls = []
    orig = RazorSolver._crossing_node_charges

    def counted(self, geom, tents, *a, **kw):
        calls.append(len(tents))
        return orig(self, geom, tents, *a, **kw)

    monkeypatch.setattr(RazorSolver, "_crossing_node_charges", counted)
    assert _razor._SERVE_CROSSING is True
    assert RazorSolver.capabilities.refusal("buried", "crossing_junction") is None
    s = razor(crossing_deck(1))
    assert s._crossing and not s._detached and not s._below_plane
    z = complex(s.compute_impedance()[0])
    assert calls == [1]
    assert np.isfinite(z) and z.real > 0


def test_the_node_term_is_confined_to_the_crossing_tents_columns(monkeypatch):
    """Its (n, T) block lands on the crossing tents' columns only, and every
    row of the deck receives it (a node charge is seen from everywhere)."""
    s = razor(crossing_deck(1))
    geom = s._build_geometry()
    tents = s._crossing_tents(geom)
    assert len(tents) == 1
    Z = s._assemble_Z_crossing(geom, s.k, s.omega)
    monkeypatch.setattr(
        RazorSolver,
        "_crossing_node_charges",
        lambda self, g, t, *a, **k: np.zeros(
            (g["n_basis_total"], len(t)), dtype=np.complex128
        ),
    )
    dZ = Z - s._assemble_Z_crossing(geom, s.k, s.omega)
    col = tents[0][0]
    others = np.delete(dZ, col, axis=1)
    assert np.array_equal(others, np.zeros_like(others))
    assert np.all(np.abs(dZ[:, col]) > 0)


def test_the_node_term_vanishes_at_eps1():
    """At eps~ = 1 the exact potential of a node charge IS the direct one and
    C2 = A_m = 0, so the term is zero to round-off — which is why the eps~ = 1
    collapse gates could never see the defect, and why they still pass."""
    s = razor(crossing_deck(1, ground_eps=(1.0, 0.0)))
    geom = s._build_geometry()
    Z = s._assemble_Z_crossing(geom, s.k, s.omega)
    tents = s._crossing_tents(geom)
    ctx = s._crossing_context(geom, k=s.k, omega=s.omega)
    A = s._crossing_path_axis(geom, tents, "above")
    P = s._crossing_path_axis(geom, tents, "below")
    dZ = s._crossing_node_charges(geom, tents, ctx, A, P, s.omega)
    assert np.max(np.abs(dZ)) <= 1e-12 * np.max(np.abs(Z)), np.max(np.abs(dZ))


def test_the_node_charges_of_a_crossing_tent_cancel():
    """The algebra the term rests on: the two half tents' node charges are
    equal and opposite, on every crossing tent of every deck shape here
    (the method raises if not; this pins that it is not raising)."""
    for deck in (crossing_deck(1), hub_deck(), tripod(1, True), two_node_deck(2.0)):
        s = razor(deck)
        geom = s._build_geometry()
        tents = s._crossing_tents(geom)
        wr, wg = geom["wing_rise"], geom["wing_sigma"]
        for col, jb in tents:
            ja = 1 - jb
            qa = -wg[col, ja] * (1.0 if wr[col, ja] else -1.0)
            qb = -wg[col, jb] * (1.0 if wr[col, jb] else -1.0)
            assert qa == -qb != 0.0


# ----------------------------------------------------------------------
# (1) reciprocity decays -- reference-free
# ----------------------------------------------------------------------


def test_reciprocity_decays_on_the_crossing_deck():
    """x1 -> x2 -> x4, soil A: 6.45e-4 -> 1.56e-4 -> 3.67e-5 measured."""
    xs = [nonrec(two_port_crossing(m)) for m in (1, 2, 4)]
    assert all(r >= 3.0 for r in ratios(xs)), (xs, ratios(xs))
    assert xs[-1] < 1e-4


def test_without_the_node_term_reciprocity_is_flat(without_the_node_term):
    """The red control for the gate above: the pre-U2 fill on the SAME
    ladder sits at 3.2e-2 and does not move. A reciprocity gate that passed
    here would be measuring nothing."""
    xs = [nonrec(two_port_crossing(m)) for m in (1, 2)]
    assert all(x > 2e-2 for x in xs), xs
    assert all(r < 1.2 for r in ratios(xs)), xs


@pytest.mark.slow
@pytest.mark.parametrize(
    "eps, bar",
    [(LOSSLESS_13, 3.0), (LOSSLESS_80, 3.0), ((13.0, 0.05), 3.0)],
)
def test_reciprocity_decays_at_every_contrast(eps, bar):
    """The defect grew with contrast (0.114 at eps_r 80); the fixed fill
    decays at every one: 3.5-4.3x / 4.0x / 4.0-4.4x measured to x16."""
    xs = [nonrec(two_port_crossing(m, eps)) for m in (1, 2, 4, 8)]
    assert all(r >= bar for r in ratios(xs)), (xs, ratios(xs))


@pytest.mark.slow
def test_reciprocity_decays_on_a_bent_deck():
    """30 deg leans put horizontal separations into the node term and the
    cross blocks' horizontal dyads to work: 4.16 / 4.46 measured."""
    xs = [nonrec(two_port_crossing(m, lean=30.0)) for m in (1, 2, 4)]
    assert all(r >= 3.0 for r in ratios(xs)), (xs, ratios(xs))


@pytest.mark.slow
def test_reciprocity_decays_with_three_crossing_tents_at_one_node():
    xs = [nonrec(tripod(m, True)) for m in (1, 2, 4)]
    assert all(r >= 3.0 for r in ratios(xs)), (xs, ratios(xs))


@pytest.mark.slow
def test_reciprocity_decays_on_a_two_node_deck():
    """`two_node_deck` at 2 m (ports at 4.5 and 2.5, so Y12 = Y21 is not a
    symmetry of the deck): 4.05 / 4.01 measured. Wider spacings put the
    fine rungs past the below grazing floor."""

    def deck(m):
        d = two_node_deck(2.0)
        d["feeds"] = [(1, 4.5, 1 + 0j), (3, 2.5, 1 + 0j)]
        return refined(d, m)

    xs = [nonrec(deck(m)) for m in (1, 2, 4)]
    assert all(r >= 3.0 for r in ratios(xs)), (xs, ratios(xs))


def test_the_node_term_is_linear_in_the_tent_pairing():
    """Three crossing tents at the node (monopole first) and one (a leg
    first) span one current space and, path by path, one test space, so the
    two fills are one system in two bases. The node term must not break
    that: it is charged per crossing tent, and a below-family tent is the
    difference of two crossing ones."""
    ya = np.asarray(razor(tripod(1, True)).compute_y_matrix())
    yb = np.asarray(razor(tripod(1, False)).compute_y_matrix())
    # port order differs with the wire order only through the feed list,
    # which both decks spell monopole-then-leg
    assert np.max(np.abs(ya - yb)) <= 1e-9 * np.max(np.abs(ya))


# ----------------------------------------------------------------------
# (2) convergence onto bspline
# ----------------------------------------------------------------------


def _gap(deck):
    zb = complex(BSplineSolver(**deck).compute_impedance()[0])
    zr = complex(razor(deck).compute_impedance()[0])
    return zr - zb


def _crossing_single(m):
    d = crossing_deck(1)
    d["feeds"] = [(1, 4.5, 1 + 0j)]
    return refined(d, m)


def test_the_crossing_deck_converges_onto_bspline():
    """|razor - bspline| 6.78 -> 3.19 -> 1.68 ohm (x1..x4); razor is first
    order in the far mesh (#845), so the bar is the gap SHRINKING."""
    g = [abs(_gap(_crossing_single(m))) for m in (1, 2, 4)]
    assert all(r >= 1.7 for r in ratios(g)), g
    assert g[-1] < 2.0


def test_without_the_node_term_the_gap_is_a_resistance(without_the_node_term):
    """The red control: +7.6 -> +9.0 ohm of R, growing, not shrinking."""
    g = [_gap(_crossing_single(m)) for m in (1, 2)]
    assert all(x.real > 7.0 for x in g), g
    assert abs(g[1]) > abs(g[0]) * 0.9, g


@pytest.mark.slow
def test_the_hub_deck_converges_onto_bspline():
    """hub_deck(4), feed on a knot (monopole 4.0): 7.47 -> 3.96 -> 2.23 ohm.
    Without the term +19.4 -> +20.5 ohm of R."""

    def deck(m):
        d = hub_deck()
        d["feeds"] = [(len(d["wires"]) - 1, 4.0, 1 + 0j)]
        return refined(d, m)

    g = [abs(_gap(deck(m))) for m in (1, 2, 4)]
    assert all(r >= 1.7 for r in ratios(g)), g


@pytest.mark.slow
@pytest.mark.parametrize("n_radials", [1, 2, 4])
def test_the_node_resistance_no_longer_grows_with_the_screen(n_radials):
    """The scoping's signature was a spurious node resistance growing with
    the number of buried members. On hub_deck(N), feed on a knot, measured
    (`scratch/razor-buried-u2/probe16_hub_radials.py`): without the term the
    R gap to bspline is +9.9 / +17.4 / +20.5 / +21.7 ohm at N = 1/2/4/8 (x4);
    with it -1.13 / -0.51 / -0.47 / -0.41, shrinking with mesh at every N
    (|gap| 9.93 -> 5.90 at N = 1, 7.31 -> 4.10 at N = 2, 7.47 -> 3.96 at
    N = 4, x1 -> x2)."""

    def deck(m):
        d = hub_deck(n_radials=n_radials)
        d["feeds"] = [(len(d["wires"]) - 1, 4.0, 1 + 0j)]
        return refined(d, m)

    g1, g2 = _gap(deck(1)), _gap(deck(2))
    assert abs(g1) / abs(g2) >= 1.5, (g1, g2)
    assert -2.5 < g2.real < 0.0, g2
