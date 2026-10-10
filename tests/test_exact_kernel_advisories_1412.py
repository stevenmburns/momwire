"""The two mesh advisories follow the exact ring kernel — momwire#1412.

`exact_kernel="auto"` (momwire#1408) turns the exact ring kernel on by itself
on fat wires, so the advisories must read the RESOLVED kernel:

* `ShortSegments` is silent wherever the exact kernel runs. Every segment is
  in a coaxial group, so its self and same-line pairs are exact; short
  segments at a bend, a T junction, a radius step, beside an offset wire and
  above a ground image were measured to converge first order too (the
  `ShortSegments` docstring), so none of them re-arms it. It still fires on
  every other family and where the exact kernel did not run (requested off,
  "auto" on a route it does not serve, True on a route the fill refuses).
* `GapMeshFloor` fires on the same meshes, but under the exact kernel its
  text is the point-gap drift (no limiting reactance) and names the end-port
  feed; the end-port junction ports themselves are not gaps there.

Construction-only: both advisories are raised before any fill.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import BSplineSolver, HMatrixSolver
from momwire._wire_spec import (
    GapMeshFloor,
    ShortSegments,
    advise_short_segments,
    solver_gaps,
)
from momwire.razor import RazorSolver

A = 0.0125  # lambda = 1 m: d/lambda 0.025, a fat wire
W = 2 * A  # the end-port gap
H = 0.23


def _caught(cls=BSplineSolver, **kw):
    kw.setdefault("wavelength", 1.0)
    kw.setdefault("wire_radius", A)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = cls(**kw)
    by = {}
    for w in caught:
        by.setdefault(w.category, []).append(str(w.message))
    return s, by


def _dipole(ratio, **kw):
    """A centre-fed dipole meshed at delta/a = `ratio` everywhere."""
    n = max(1, round(2 * H / (ratio * A)))
    return dict(
        wires=[np.array([(0.0, 0.0, -H), (0.0, 0.0, H)])],
        n_per_edge_per_wire=[[n]],
        **kw,
    )


def _endport(ratio, **kw):
    """The end-port feed: two facing wire ends across a 2 a gap, each a
    one-member junction carrying a port, no gap feed."""
    n = max(1, round((H - W / 2) / (ratio * A)))
    return dict(
        wires=[
            np.array([(0.0, 0.0, -H), (0.0, 0.0, -W / 2)]),
            np.array([(0.0, 0.0, W / 2), (0.0, 0.0, H)]),
        ],
        n_per_edge_per_wire=[[n], [n]],
        feeds=[],
        junctions=[[(0, "end")], [(1, "start")]],
        junction_ports=[(0, 1 + 0j), (1, -1 + 0j)],
        **kw,
    )


def _bent(ratio, **kw):
    """The bend family's shape: a straight run that turns 45 degrees, plus a
    T junction (a stub off the bend's node) and a parallel wire 4 a away."""

    def n(length):
        return max(1, round(length / (ratio * A)))

    up = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.12)])
    bend = np.array([(0.0, 0.0, 0.12), (0.1, 0.0, 0.22)])
    stub = np.array([(0.0, 0.0, 0.12), (-0.08, 0.0, 0.12)])
    par = np.array([(4 * A, 0.0, 0.0), (4 * A, 0.0, 0.10)])
    return dict(
        wires=[up, bend, stub, par],
        n_per_edge_per_wire=[[n(0.12)], [n(0.1414)], [n(0.08)], [n(0.10)]],
        junctions=[[(0, "end"), (1, "start"), (2, "start")]],
        feeds=[(0, 0.06, 1 + 0j)],
        **kw,
    )


def _screen(**kw):
    """A fat buried wire under a fed mast: a route the exact kernel does not
    serve (buried wires)."""
    return dict(
        wires=[
            np.array([(0.0, 0.0, -0.05), (0.5, 0.0, -0.05)]),
            np.array([(0.0, 0.0, -0.05), (0.0, 0.0, 0.0)]),
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.25)]),
        ],
        n_per_edge_per_wire=[[40], [4], [20]],
        feeds=[(2, 0.125, 1 + 0j)],
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
        **kw,
    )


# --------------------------------------------------------------------------
# ShortSegments
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ek", [True, "auto"])
@pytest.mark.parametrize("deck", [_dipole, _endport, _bent])
def test_short_segments_silent_where_the_exact_kernel_runs(deck, ek):
    s, by = _caught(**deck(0.5, exact_kernel=ek))
    assert s.exact_kernel is True
    assert ShortSegments not in by


@pytest.mark.parametrize("deck", [_dipole, _endport, _bent])
def test_short_segments_fires_where_it_does_not(deck):
    s, by = _caught(**deck(0.5, exact_kernel=False))
    assert s.exact_kernel is False
    assert len(by[ShortSegments]) == 1
    assert "momwire#1378" in by[ShortSegments][0]


@pytest.mark.parametrize("ek", [True, "auto"])
def test_short_segments_fires_on_a_route_the_exact_kernel_does_not_serve(ek):
    """Buried: "auto" resolves off, and True would be refused by the fill;
    neither runs the exact kernel, so the advisory stands."""
    _s, by = _caught(**_screen(exact_kernel=ek))
    assert len(by[ShortSegments]) == 1


def test_short_segments_fires_on_a_subclass_that_refuses_it():
    s, by = _caught(HMatrixSolver, **_dipole(0.5, exact_kernel="auto"))
    assert s.exact_kernel is False
    assert len(by[ShortSegments]) == 1


def test_short_segments_unchanged_on_other_families():
    _s, by = _caught(RazorSolver, **_dipole(0.5))
    assert len(by[ShortSegments]) == 1


def test_advise_short_segments_flag():
    pl = [np.array([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        advise_short_segments("X", pl, [[10]], [0.1], exact_kernel=True)
        assert caught == []
        advise_short_segments("X", pl, [[10]], [0.1])
    assert [w.category for w in caught] == [ShortSegments]


# --------------------------------------------------------------------------
# GapMeshFloor
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ek", [True, "auto"])
def test_gap_floor_under_the_exact_kernel_states_the_drift(ek):
    s, by = _caught(**_dipole(0.25, exact_kernel=ek))
    assert s.exact_kernel is True
    (msg,) = by[GapMeshFloor]
    assert "source 0" in msg
    assert "no limiting reactance" in msg
    assert "4 ln2 w eps0 a" in msg
    assert "end-port feed" in msg and "junction_ports" in msg
    assert "momwire#1408" in msg and "momwire#959" in msg
    # the thin-wire wording, and the thin-wire fixes, are gone
    assert "toward 0 ohm" not in msg
    assert "momwire#1330" not in msg
    assert "extended_kernel=True" not in msg


def test_gap_floor_off_names_the_exact_kernel():
    """bspline with the kernel available and not asked for: the thin-wire
    text, and the fix this solver has."""
    _s, by = _caught(**_dipole(0.25, exact_kernel=False))
    (msg,) = by[GapMeshFloor]
    assert "toward 0 ohm" in msg
    assert "exact_kernel=True with an end-port feed" in msg
    assert "momwire#1408" in msg
    assert "momwire#1330" not in msg


@pytest.mark.parametrize(
    "cls, deck",
    [(RazorSolver, _dipole(0.25)), (BSplineSolver, _screen(exact_kernel="auto"))],
    ids=["razor", "bspline-buried"],
)
def test_gap_floor_without_an_exact_kernel_points_at_1330(cls, deck):
    # the buried deck's mast feed sits on a fine run: mesh it so
    deck = dict(deck)
    if cls is BSplineSolver:
        deck["n_per_edge_per_wire"] = [[40], [4], [100]]
    _s, by = _caught(cls, **deck)
    (msg,) = by[GapMeshFloor]
    assert "toward 0 ohm" in msg and "momwire#1330" in msg
    assert "exact_kernel=True" not in msg


@pytest.mark.parametrize("ek", [True, "auto"])
def test_end_ports_are_not_gaps_under_the_exact_kernel(ek):
    """The end-port feed's gap is the distance between the two ends, fixed
    under refinement; under the exact kernel it converges (momwire#1408's
    end-port ladder), so a fine mesh beside it raises nothing."""
    s, by = _caught(**_endport(0.25, exact_kernel=ek))
    assert s.exact_kernel is True
    assert GapMeshFloor not in by
    assert ShortSegments not in by


def test_end_ports_are_gaps_under_the_thin_wire_kernel():
    _s, by = _caught(**_endport(0.25, exact_kernel=False))
    (msg,) = by[GapMeshFloor]
    assert "junction port 0" in msg and "junction port 1" in msg


def test_a_gap_feed_beside_end_ports_still_counts():
    """Only the one-member junction ports drop out: a delta-gap feed in the
    same deck is a zero-width gap and is still named."""
    deck = _endport(0.25, exact_kernel=True)
    deck["feeds"] = [(0, 0.05, 1 + 0j)]
    _s, by = _caught(**deck)
    (msg,) = by[GapMeshFloor]
    assert "source 0" in msg and "junction port" not in msg


def test_solver_gaps_keeps_a_port_on_a_shared_node():
    """`end_ports_are_gaps=False` drops one-member junction ports only; a
    port on a node two wires share is a zero-width gap and stays."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = BSplineSolver(
            wires=[
                np.array([(0.0, 0.0, -H), (0.0, 0.0, 0.0)]),
                np.array([(0.0, 0.0, 0.0), (0.0, 0.0, H)]),
                np.array([(1.0, 0.0, -H), (1.0, 0.0, -W / 2)]),
                np.array([(1.0, 0.0, W / 2), (1.0, 0.0, H)]),
            ],
            n_per_edge_per_wire=[[9]] * 4,
            feeds=[],
            junctions=[[(0, "end"), (1, "start")], [(2, "end")], [(3, "start")]],
            junction_ports=[(0, 1 + 0j), (1, 1 + 0j), (2, -1 + 0j)],
            wavelength=1.0,
            wire_radius=A,
        )
    labels = [g[0] for g in solver_gaps(s, end_ports_are_gaps=False)]
    assert labels == ["junction port 0 (junction 0)"]
    assert len(solver_gaps(s)) == 3
