"""`BSplineSolver.port_sites` — point ports that move along one wire (#1915).

The oracle throughout is a FRESH solver built with an ordinary extra point
feed at x, read through `compute_y_matrix()`: it never touches the site path,
so agreement is the claim "a cached-column combination is the solve with the
port there", not the site path agreeing with itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import BSplineSolver, PortSites, terminate_ports
from momwire.networks import Driven, Load, Network, NetworkReducer, PortOnWire

WL = 20.0
A = 0.001
REL_TOL = 1e-10


def _dipole(**kw):
    """A 9.5 m z-directed wire, 15 cells, centre-fed."""
    return dict(
        wires=[np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 9.5)])],
        n_per_edge_per_wire=[[15]],
        wavelength=WL,
        wire_radius=A,
        **kw,
    )


def _over_ground(**kw):
    """A 9.5 m horizontal wire 2 m over lossy Sommerfeld ground."""
    return dict(
        wires=[np.array([(0.0, 0.0, 2.0), (9.5, 0.0, 2.0)])],
        n_per_edge_per_wire=[[15]],
        wavelength=WL,
        wire_radius=A,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
        **kw,
    )


def _bent(**kw):
    """Feed on wire 0; the knob wire 1 meets it at a junction (KCL active)."""
    return dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 5.0)]),
            np.array([(0.0, 0.0, 5.0), (4.0, 0.0, 7.0)]),
        ],
        n_per_edge_per_wire=[[10], [10]],
        junctions=[[(0, "end"), (1, "start")]],
        wavelength=WL,
        wire_radius=A,
        **kw,
    )


def _tee(**kw):
    """Three wires at one node, singular enrichment on (raw variant)."""
    return dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 4.0)]),
            np.array([(0.0, 0.0, 4.0), (3.0, 0.0, 4.0)]),
            np.array([(0.0, 0.0, 4.0), (-2.5, 0.0, 5.5)]),
        ],
        n_per_edge_per_wire=[[8], [8], [8]],
        junctions=[[(0, "end"), (1, "start"), (2, "start")]],
        use_singular_enrichment=True,
        enrichment_variant="raw",
        wavelength=WL,
        wire_radius=A,
        **kw,
    )


def _wire_arc(deck, w):
    pl = deck["wires"][w]
    return float(np.sum(np.linalg.norm(np.diff(pl, axis=0), axis=1)))


def _feed(deck):
    return [(0, _wire_arc(deck, 0) / 2.0, 1.0 + 0j)]


def _fresh_y(deck, degree, wire, xs):
    """compute_y_matrix of an ordinary solver with a 0 V point feed at each x."""
    feeds = _feed(deck) + [(wire, float(x), 0j) for x in xs]
    return BSplineSolver(degree=degree, feeds=feeds, **deck).compute_y_matrix()


def _sites(deck, degree, wire, span=None):
    s = BSplineSolver(degree=degree, feeds=_feed(deck), **deck)
    return s, s.port_sites(wire, span)


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _positions(solver, wire):
    """On a knot, mid-cell, near both ends (inside), and an arbitrary point."""
    arc = solver._build_geometry()["per_wire"][wire]["arc_at_knot"]
    h = float(arc[1] - arc[0])
    L = float(arc[-1])
    return [
        float(arc[4]),  # exactly on an interior knot
        0.5 * (float(arc[6]) + float(arc[7])),  # mid-cell
        0.07 * h,  # near the start, inside the first cell
        L - 0.11 * h,  # near the end
        0.6180339 * L,  # between knots
    ]


CASES = [
    pytest.param(_dipole, 0, id="free"),
    pytest.param(_over_ground, 0, id="sommerfeld"),
    pytest.param(_bent, 1, id="junction"),
]


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("deck_fn, wire", CASES)
def test_y_equals_a_fresh_solve_with_the_port_there(deck_fn, wire, degree):
    deck = deck_fn()
    solver, ps = _sites(deck, degree, wire)
    assert isinstance(ps, PortSites) and ps.n_ports == 1
    worst = 0.0
    for x in _positions(solver, wire):
        if wire == 0 and abs(x - _feed(deck)[0][1]) < 1e-9:
            continue
        worst = max(worst, _rel(ps.y([x]), _fresh_y(deck, degree, wire, [x])))
    assert worst <= REL_TOL, worst


@pytest.mark.parametrize("degree", [1, 2])
def test_two_moving_ports_at_once(degree):
    """Two positions give the 3-port; the cross term x1-x2 is a site-site
    readout, the one the single-x gate never reads."""
    deck = _bent()
    solver, ps = _sites(deck, degree, 1)
    xs = _positions(solver, 1)[:2]
    assert _rel(ps.y(xs), _fresh_y(deck, degree, 1, xs)) <= REL_TOL


def test_enrichment_rides_on_the_site_columns():
    deck = _tee()
    solver, ps = _sites(deck, 2, 1)
    for x in _positions(solver, 1):
        assert _rel(ps.y([x]), _fresh_y(deck, 2, 1, [x])) <= REL_TOL


def test_skipping_the_constraint_machinery_would_fail():
    """The adversary: Bᵀ Z⁻¹ B over the same columns, solved WITHOUT the
    junction KCL rows. If the site path skipped them it would equal this,
    and this is far from the fresh solve."""
    deck = _bent()
    solver = BSplineSolver(degree=2, feeds=_feed(deck), **deck)
    x = 0.3  # inside the junction-end cell, where the directional basis lives
    geom = solver._build_geometry()
    supp, polys, kcl_A, wk, wbg = solver._build_basis_polynomials(geom)
    assert kcl_A.shape[0] > 0
    n = supp.shape[0]
    Z = solver._compute_Z_operator(geom, supp, polys)
    B = np.stack(
        [
            solver._build_source_vector(geom, wk, wbg, n, wi=0, s_f=_feed(deck)[0][1]),
            solver._build_source_vector(geom, wk, wbg, n, wi=1, s_f=x),
        ],
        axis=1,
    )
    naive = B.T @ np.linalg.solve(Z, B)
    fresh = _fresh_y(deck, 2, 1, [x])
    assert _rel(naive, fresh) > 1e-3
    assert _rel(solver.port_sites(1).y([x]), fresh) <= REL_TOL


def test_y_moves_with_x():
    deck = _dipole()
    _s, ps = _sites(deck, 2, 0)
    a, b = ps.y([2.0]), ps.y([2.0 + 1e-3])
    assert _rel(a, b) > 1e-6
    # The configured block does not move: shorted sites are the plain wire.
    assert np.array_equal(a[0, 0], b[0, 0])


def test_shorted_site_is_the_bare_wire():
    """Y_ff over [feed, site] is the bare solver's 1/Z_in, and a zero load
    terminates back to it."""
    deck = _dipole()
    solver, ps = _sites(deck, 2, 0)
    z_bare, _ = BSplineSolver(degree=2, feeds=_feed(deck), **deck).compute_impedance()
    y2 = ps.y([2.0])
    assert abs(1 / y2[0, 0] - z_bare) <= REL_TOL * abs(z_bare)
    assert terminate_ports(y2, {1: 0.0})[0, 0] == y2[0, 0]


# ----------------------------------------------------------------------
# dY/dx
# ----------------------------------------------------------------------


def _fd(deck, degree, wire, x, h, kind="central"):
    if kind == "central":
        return (
            _fresh_y(deck, degree, wire, [x + h])
            - _fresh_y(deck, degree, wire, [x - h])
        ) / (2 * h)
    if kind == "forward":
        return (
            _fresh_y(deck, degree, wire, [x + h]) - _fresh_y(deck, degree, wire, [x])
        ) / h
    return (
        _fresh_y(deck, degree, wire, [x]) - _fresh_y(deck, degree, wire, [x - h])
    ) / h


FD_STEP = 1e-5  # metres; cells are 0.633 m (free) and 0.5 m (junction wire)
FD_TOL = 1e-6


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("deck_fn, wire", CASES)
def test_dy_dx_matches_a_central_difference_mid_cell(deck_fn, wire, degree):
    deck = deck_fn()
    solver, ps = _sites(deck, degree, wire)
    arc = solver._build_geometry()["per_wire"][wire]["arc_at_knot"]
    for c in (2, 7):
        x = float(arc[c]) + 0.37 * float(arc[c + 1] - arc[c])
        got = ps.dy_dx([x])[0]
        assert _rel(got, _fd(deck, degree, wire, x, FD_STEP)) <= FD_TOL


def test_dy_dx_is_continuous_at_a_knot_for_d2():
    deck = _dipole()
    solver, ps = _sites(deck, 2, 0)
    knot = float(solver._build_geometry()["per_wire"][0]["arc_at_knot"][5])
    got = ps.dy_dx([knot])[0]
    assert _rel(got, _fd(deck, 2, 0, knot, 1e-6)) <= 1e-5


def test_dy_dx_at_a_knot_for_d1_is_the_right_hand_derivative():
    """d = 1 is C0 at a knot: the two one-sided derivatives differ, and the
    value returned exactly on the knot is the forward (right-hand) one."""
    deck = _dipole()
    solver, ps = _sites(deck, 1, 0)
    knot = float(solver._build_geometry()["per_wire"][0]["arc_at_knot"][5])
    got = ps.dy_dx([knot])[0]
    fwd = _fd(deck, 1, 0, knot, 1e-6, "forward")
    bwd = _fd(deck, 1, 0, knot, 1e-6, "backward")
    assert _rel(fwd, bwd) > 1e-2  # the jump is real
    assert _rel(got, fwd) <= 1e-5
    assert _rel(got, bwd) > 1e-2


@pytest.mark.parametrize("degree", [1, 2])
def test_dy_dx_two_positions_against_its_own_y(degree):
    """∂Y/∂x_j moves only row/column j; checked by differencing `y` itself."""
    deck = _bent()
    _s, ps = _sites(deck, degree, 1)
    xs = np.array([1.13, 3.71])
    d = ps.dy_dx(xs)
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        fd = (ps.y(xs + e) - ps.y(xs - e)) / (2 * h)
        assert _rel(d[j], fd) <= 1e-6


# ----------------------------------------------------------------------
# A load at the moving port
# ----------------------------------------------------------------------


@pytest.mark.parametrize("z_load", [50.0 - 300.0j, 5.0 + 800.0j])
def test_a_load_at_x_matches_the_network_reducer_on_a_fresh_solve(z_load):
    """BSplineSolver has no lumped point load of its own: a load is a 0 V gap
    port terminated by the port algebra (`momwire.deck`). So the oracle is the
    fresh solver's 2-port through `NetworkReducer` with a `Load` branch."""
    deck = _dipole()
    _s, ps = _sites(deck, 2, 0)
    x = 2.77
    z_ours = 1.0 / terminate_ports(ps.y([x]), {1: z_load})[0, 0]
    net = Network(
        ports={"feed": PortOnWire("feed"), "site": PortOnWire("site")},
        branches=[Load(port="site", z=z_load)],
        sources=[Driven(port="feed")],
    )
    red = NetworkReducer(net, {"feed": 0, "site": 1}, 2)
    z_ref = np.ravel(red.driven_impedance(_fresh_y(deck, 2, 0, [x]), WL))[0]
    assert abs(z_ours - z_ref) <= 1e-9 * abs(z_ref)


# ----------------------------------------------------------------------
# Window, memory, refusals
# ----------------------------------------------------------------------


def test_span_bounds_the_cache_and_still_agrees():
    deck = _dipole()
    _s, whole = _sites(deck, 2, 0)
    _s, part = _sites(deck, 2, 0, span=(1.0, 3.0))
    assert part.n_sites < whole.n_sites
    assert part.nbytes < whole.nbytes
    assert part.nbytes >= part._columns.nbytes > 0
    assert _rel(part.y([1.0, 2.2, 3.0]), whole.y([1.0, 2.2, 3.0])) <= REL_TOL
    with pytest.raises(ValueError, match="outside the cached window"):
        part.y([3.5])


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(feed_model="segment"), "feed_model='point'"),
        (dict(feed_smoothing_factor=0.5), "feed_model='point'"),
        (
            dict(use_singular_enrichment=True, enrichment_variant="auto"),
            "enrichment_variant='auto'",
        ),
    ],
)
def test_refusals(kw, match):
    deck = _dipole(**kw)
    s = BSplineSolver(degree=2, feeds=_feed(deck), **deck)
    with pytest.raises(ValueError, match=match):
        s.port_sites(0)


def test_off_the_wire_is_refused():
    deck = _dipole()
    _s, ps = _sites(deck, 2, 0)
    with pytest.raises(ValueError):
        ps.y([9.6])
    with pytest.raises(ValueError, match="out of range"):
        _s.port_sites(3)
