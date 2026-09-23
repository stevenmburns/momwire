"""razor-2p's TWO-RADIUS crossing node — momwire#1149 U2b-2.

A ground rod at one radius under a mast at another (the v0.61.0 shape, bspline
since antennaknobs plan U5) is served by razor as the crossing deck it is,
with razor's own radius convention: every term at its SOURCE wire's radius,
as razor's reduced kernel takes it everywhere (`_seg_moments_prepare`).

  * the forward cross block (above rows, buried sources) at the buried
    radius, the reversed one at the above side's — partitioned by radius when
    the above side carries several, which is momwire#1140's deck (a far above
    wire at a third radius);
  * the node term (`_crossing_node_charges`): each half tent's charge at its
    own wire's radius. The quantity it removes is radius-free, so the radius
    only regularises the one endpoint at the node; measured, every node radius
    moved to the other side's value moves Z by at most 1.1e-3 ohm and a -> a/100
    by at most 1.4e-3 — orders below any gate here.

bspline's rule differs in spelling (line tests at the observer's radius, every
point test at the node at the buried one) and closes continuity with a KCL
multiplier; razor's tent basis carries continuity itself and its source rule
gives one potential at the node as a function of position, so the jump
bspline's rule exists to avoid does not arise. The two are held together by
the soil radius-response differential below, which the eps~ = 1 collapse
cannot see.

Deck: `crossing_deck(2)`, the U5 rod (2 m below, 10 m above), radii
`[below, above]` about A = 0.25 mm, fed at above arclength 4.5 so the feed is
a knot at every rung. Every edge refined by `m`. Measured 2026-09-22
(`scratch/razor-buried-u2b/`, probe 2):

  * **eps~ = 1 collapse** to razor's free-space fill of the same deck: whole
    matrix 1.8e-12 (rise/2) / 1.7e-12 (top/4); the observer, min, max and
    wire-0 rules 4.3e-2 / 7.3e-2 (the node rows carry it);
  * **2-port non-reciprocity** decays 4.1-4.25x per doubling to x8; the
    observer rule is flat at 8.3e-2 / 1.4e-1;
  * **the radius response against bspline** — R(mixed) - R(equal), razor
    minus bspline — at x1/x2/x4/x8: rise/2 -0.19/-0.079/-0.037/-0.020,
    rise/4 -0.39/-0.16/-0.076/-0.039, top/2 +0.10/+0.056/+0.033/+0.021,
    top/4 +0.19/+0.10/+0.060/+0.038 ohm (halving per doubling), on responses
    of 8.0 / 15.9 / 0.62 / 1.13 ohm. The observer cross-block rule misses by
    8-16 ohm;
  * **convergence onto bspline**: |dZ| 6.85 -> 3.03 -> 1.50 -> 0.81 (rise/2),
    6.55 -> 2.84 -> 1.38 -> 0.73 (top/4), 7.19 -> 3.59 -> 2.00 -> 1.20 ohm
    (the #1140 deck).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire import _crossing_fill  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import above_side_deck, crossing_deck  # noqa: E402

A = 0.25e-3


def rod(m=1, a_above=A, a_below=A, **kw):
    d = crossing_deck(2, wire_radius=[a_below, a_above], **kw)
    d["feeds"] = [(1, 4.5, 1 + 0j)]
    d["n_per_edge_per_wire"] = [[n * m for n in e] for e in d["n_per_edge_per_wire"]]
    return d


def deck_1140(m=1):
    """momwire#1140's deck: the node's above member at 1 mm, the far above
    wire at 4 mm, the rod at 2 mm. Fed at absolute above arclength 4.5."""
    d = above_side_deck(wire_radius=[0.002, 0.001, 0.004])
    d["feeds"] = [(2, 4.0, 1 + 0j)]
    d["n_per_edge_per_wire"] = [[n * m for n in e] for e in d["n_per_edge_per_wire"]]
    return d


def razor(d, **kw):
    return RazorSolver(
        **{k: v for k, v in d.items() if k != "junctions"}, nec5_quadrature=True, **kw
    )


def zr(d):
    return complex(razor(d).compute_impedance()[0])


def zb(d):
    return complex(BSplineSolver(**d).compute_impedance()[0])


def nonrec(d):
    Y = np.asarray(razor(d).compute_y_matrix())
    return abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])


def ratios(xs):
    return [a / b for a, b in zip(xs, xs[1:])]


@pytest.fixture
def observer_rule(monkeypatch):
    """The red control: each cross block at its ROW side's radius instead of
    its source's. Installed per deck with `observer_rule(a_above, a_below)`."""
    fwd = _crossing_fill.cross_complete_block
    rev = _crossing_fill.cross_complete_block_reversed

    def install(a_above, a_below):
        monkeypatch.setattr(
            _crossing_fill,
            "cross_complete_block",
            lambda ctx, *a, **k: fwd(ctx._replace(a_wire=a_above), *a, **k),
        )
        monkeypatch.setattr(
            _crossing_fill,
            "cross_complete_block_reversed",
            lambda ctx, *a, **k: rev(ctx._replace(a_wire=a_below), *a, **k),
        )

    return install


# ----------------------------------------------------------------------
# the route, counted
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, deck, expect",
    [
        ("rise/2", rod(1, A, A / 2), [("fwd", A / 2), ("rev", A)]),
        ("top/2", rod(1, A / 2, A), [("fwd", A), ("rev", A / 2)]),
        ("1140", deck_1140(), [("fwd", 0.002), ("rev", 0.001), ("rev", 0.004)]),
    ],
)
def test_a_two_radius_crossing_is_served_at_source_radii(
    monkeypatch, name, deck, expect
):
    """Through the REAL constructor: the shared scope passes (razor opts into
    `two_radius`), each cross block is filled at its source radius, one call
    per radius, and the node term runs once with the deck's one tent."""
    calls, tents = [], []
    fwd = _crossing_fill.cross_complete_block
    rev = _crossing_fill.cross_complete_block_reversed
    node = RazorSolver._crossing_node_charges

    def f(ctx, *a, **k):
        calls.append(("fwd", float(ctx.a_wire)))
        return fwd(ctx, *a, **k)

    def r(ctx, *a, **k):
        calls.append(("rev", float(ctx.a_wire)))
        return rev(ctx, *a, **k)

    def n(self, geom, t, *a, **k):
        tents.append(len(t))
        return node(self, geom, t, *a, **k)

    monkeypatch.setattr(_crossing_fill, "cross_complete_block", f)
    monkeypatch.setattr(_crossing_fill, "cross_complete_block_reversed", r)
    monkeypatch.setattr(RazorSolver, "_crossing_node_charges", n)
    s = razor(deck)
    assert s._crossing and s._crossing_junctions() == (0,)
    z = complex(s.compute_impedance()[0])
    assert np.isfinite(z) and z.real > 0
    assert calls == expect
    assert tents == [1]


def test_the_node_term_takes_each_half_tents_own_radius(monkeypatch):
    """The per-wing rule, read off the method: the above rows' table call is
    at the above wing's radius and the below rows' at the below wing's."""
    seen = []
    tables = _crossing_fill._tables

    def spy(ctx, *a, **k):
        seen.append(float(ctx.a_wire))
        return tables(ctx, *a, **k)

    s = razor(rod(1, A, A / 4))
    geom = s._build_geometry()
    tents = s._crossing_tents(geom)
    ctx = s._crossing_context(geom, k=s.k, omega=s.omega)
    Ax = s._crossing_path_axis(geom, tents, "above")
    Px = s._crossing_path_axis(geom, tents, "below")
    monkeypatch.setattr(_crossing_fill, "_tables", spy)
    s._crossing_node_charges(geom, tents, ctx, Ax, Px, s.omega)
    assert seen == [A, A / 4]


def test_the_node_radius_only_regularises():
    """The node term removes a radius-free quantity, so its radius is a
    regulariser and not a knob: every node radius divided by 100 moves the
    driving point by ~1e-3 ohm (measured 3.1e-4 / 1.4e-3 at x4), against
    radius responses of 1-16 ohm."""
    for aa, ab in ((A, A / 4), (A / 4, A)):
        d = rod(2, aa, ab)
        s0 = razor(d)
        z0 = complex(s0.compute_impedance()[0])
        s1 = razor(d)
        real = s1._seg_radius
        node = s1._crossing_node_charges

        def shrunk(geom, *a, _real=real, _node=node, _s=s1, **k):
            _s._seg_radius = lambda g: _real(g) / 100.0
            try:
                return _node(geom, *a, **k)
            finally:
                _s._seg_radius = _real

        s1._crossing_node_charges = shrunk
        z1 = complex(s1.compute_impedance()[0])
        assert 0.0 < abs(z1 - z0) < 5e-3, (aa, ab, z0, z1)


# ----------------------------------------------------------------------
# the eps~ = 1 collapse, and what it discriminates
# ----------------------------------------------------------------------


def razor_lane(d, lane):
    return RazorSolver(
        **{k: v for k, v in d.items() if k != "junctions"}, nec5_quadrature=lane
    )


@pytest.mark.parametrize(
    "lane", [True, pytest.param(False, marks=pytest.mark.slow)], ids=["2pt", "gl"]
)
@pytest.mark.parametrize("radii", [(A, A / 2), (A / 4, A)], ids=["rise/2", "top/4"])
def test_eps_one_collapses_to_razors_free_space_fill(radii, lane):
    """Whole matrix, every block: measured 1.8e-12 / 1.7e-12 on both lanes."""
    d1 = rod(1, *radii, ground_eps=(1.0, 0.0))
    df = {
        k: v
        for k, v in d1.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    s1, sf = razor_lane(d1, lane), razor_lane(df, lane)
    g1, gf = s1._build_geometry(), sf._build_geometry()
    assert s1._crossing and np.array_equal(g1["wing_seg"], gf["wing_seg"])
    Z1, Zf = s1._assemble_Z(g1, s1.k), sf._assemble_Z(gf, sf.k)
    rel = float(np.max(np.abs(Z1 - Zf)) / np.max(np.abs(Zf)))
    assert rel < 1e-7, rel


@pytest.mark.parametrize("radii", [(A, A / 2), (A / 4, A)], ids=["rise/2", "top/4"])
def test_the_collapse_rejects_the_observer_rule(observer_rule, radii):
    """The red control: with each cross block at its row side's radius the
    same collapse misses by 4.3e-2 / 7.3e-2."""
    observer_rule(*radii)
    d1 = rod(1, *radii, ground_eps=(1.0, 0.0))
    df = {
        k: v
        for k, v in d1.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    s1, sf = razor_lane(d1, True), razor_lane(df, True)
    g1, gf = s1._build_geometry(), sf._build_geometry()
    Z1, Zf = s1._assemble_Z(g1, s1.k), sf._assemble_Z(gf, sf.k)
    assert float(np.max(np.abs(Z1 - Zf)) / np.max(np.abs(Zf))) > 1e-2


# ----------------------------------------------------------------------
# reciprocity
# ----------------------------------------------------------------------


def _two_port(m, radii):
    d = rod(m, *radii)
    d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
    return d


@pytest.mark.parametrize("radii", [(A, A / 2), (A / 4, A)], ids=["rise/2", "top/4"])
def test_reciprocity_decays_on_the_two_radius_rod(radii):
    """Measured 4.11 / 4.17 (rise/2) and 4.12 / 4.22 (top/4)."""
    xs = [nonrec(_two_port(m, radii)) for m in (1, 2, 4)]
    assert all(r >= 3.0 for r in ratios(xs)), (xs, ratios(xs))


def test_under_the_observer_rule_reciprocity_is_flat(observer_rule):
    """The red control: 8.27e-2 -> 8.26e-2."""
    observer_rule(A, A / 2)
    xs = [nonrec(_two_port(m, (A, A / 2))) for m in (1, 2)]
    assert all(x > 1e-2 for x in xs) and ratios(xs)[0] < 1.2, xs


# ----------------------------------------------------------------------
# the soil radius response against bspline -- what eps~ = 1 cannot see
# ----------------------------------------------------------------------


def _response(m, radii, solve):
    return solve(rod(m, *radii)).real - solve(rod(m)).real


@pytest.mark.parametrize("radii", [(A, A / 2), (A / 2, A)], ids=["rise/2", "top/2"])
def test_the_radius_response_matches_bsplines(radii):
    """R(mixed) - R(equal) on razor against bspline at x2, the edit loop's
    rung: measured -0.079 on 8.0 ohm (rise/2) and +0.056 on 0.62 ohm (top/2).
    The x4 bar over all four cases is the slow lane's."""
    dr, db = _response(2, radii, zr), _response(2, radii, zb)
    assert abs(dr - db) < 0.1, (dr, db)


def test_the_response_rejects_the_observer_rule(observer_rule):
    """The red control: the observer rule's response misses bspline's by
    ~8 ohm on rise/2."""
    db = _response(2, (A, A / 2), zb)
    observer_rule(A, A / 2)
    dr = _response(2, (A, A / 2), zr)
    assert abs(dr - db) > 4.0, (dr, db)


@pytest.mark.slow
@pytest.mark.parametrize(
    "radii",
    [(A, A / 2), (A, A / 4), (A / 2, A), (A / 4, A)],
    ids=["rise/2", "rise/4", "top/2", "top/4"],
)
def test_the_radius_response_matches_bsplines_at_x4(radii):
    """The task's bar: within 0.1 ohm at x4 on every case. Measured -0.037,
    -0.076, +0.033, +0.060 on 8.0 / 15.9 / 0.62 / 1.13 ohm."""
    dr, db = _response(4, radii, zr), _response(4, radii, zb)
    assert abs(dr - db) < 0.1, (dr, db)


@pytest.mark.slow
@pytest.mark.parametrize(
    "radii",
    [(A, A / 2), (A, A / 4), (A / 2, A), (A / 4, A)],
    ids=["rise/2", "rise/4", "top/2", "top/4"],
)
def test_the_radius_response_converges_onto_bsplines(radii):
    """The differential halves per doubling (x2 -> x4 -> x8) and is within
    0.1 ohm from x4."""
    gaps = [abs(_response(m, radii, zr) - _response(m, radii, zb)) for m in (2, 4, 8)]
    assert all(r >= 1.5 for r in ratios(gaps)), gaps
    assert gaps[1] < 0.1 and gaps[2] < 0.1, gaps


# ----------------------------------------------------------------------
# convergence onto bspline
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "name, mk",
    [
        ("rise/2", lambda m: rod(m, A, A / 2)),
        ("top/4", lambda m: rod(m, A / 4, A)),
        ("1140", deck_1140),
    ],
)
def test_the_driving_point_converges_onto_bspline(name, mk):
    """|razor - bspline| shrinking per doubling (razor first order, #845):
    measured ratios 0.43-0.60."""
    g = [abs(zr(mk(m)) - zb(mk(m))) for m in (1, 2, 4)]
    assert all(r <= 0.65 for r in [b / a for a, b in zip(g, g[1:])]), g


# ----------------------------------------------------------------------
# what stays refused: bspline's scope, bspline's sentences
# ----------------------------------------------------------------------


def test_a_spread_among_the_buried_wires_is_refused_as_bspline_refuses_it():
    from test_crossing_serve_524 import hub_deck

    radii = [0.001, 0.001, 0.001, 0.0005, 0.001, 0.002]
    with pytest.raises(NotImplementedError, match="differ within the below wires"):
        razor(hub_deck(wire_radius=radii))
