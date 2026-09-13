"""momwire#1059: `PortOnWire` takes a wire and a position along it, and every
solver reports where each feed landed.

The position is antennaknobs' way to feed a wire somewhere other than its
middle without cutting it (antennaknobs#1469): a fraction along the wire,
which the caller turns into a site the chosen mesh can carry. The snapping
families move a request onto their own grid (#623), so a caller that promises
no feed moves silently needs each family to say where the feed went.

The gates hold the report to the solve rather than to a second reading of the
rules: a grid-locked family solved at the point it reports must give the
answer it gave at the request, bit for bit.
"""

import copy
import math
import pickle

import numpy as np
import pytest

from momwire import FeedPlacement, _feed_snap
from momwire.bspline import BSplineSolver
from momwire.harrington import HarringtonSolver
from momwire.networks import Driven, Load, Network, PortOnWire, PortOnWireFloating
from momwire.pulse import PulseSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

# ----------------------------------------------------------------------
# The spec
# ----------------------------------------------------------------------


def test_a_port_built_as_before_is_the_port_it_was():
    p = PortOnWire("feed")
    assert (p.name, p.distributed, p.wire, p.at) == ("feed", False, None, None)
    assert p.wire_name == "feed"
    assert PortOnWire("feed", True) == PortOnWire("feed", distributed=True)


def test_a_port_names_its_wire_and_a_position_on_it():
    p = PortOnWire("load2", wire="w3", at=np.float64(0.25))
    assert p.wire_name == "w3"
    assert p.at == 0.25 and type(p.at) is float


@pytest.mark.parametrize(
    "at", [0, 0.0, 1, 1.0, -0.1, 1.5, math.nan, math.inf, True, "0.5"]
)
def test_at_is_a_fraction_strictly_inside_the_wire(at):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        PortOnWire("feed", at=at)


@pytest.mark.parametrize("wire", ["", 3])
def test_wire_is_a_non_empty_name(wire):
    with pytest.raises(ValueError, match="non-empty wire name"):
        PortOnWire("feed", wire=wire)


def test_a_distributed_port_takes_a_wire_but_no_position():
    """A finite gap spans its whole wire, so a position would be ignored."""
    assert PortOnWire("feed", distributed=True, wire="w1").wire_name == "w1"
    with pytest.raises(ValueError, match="takes no `at`"):
        PortOnWire("feed", distributed=True, at=0.3)


def test_a_floating_port_carries_the_position_too():
    p = PortOnWireFloating("bal", wire="w1", at=0.4)
    assert (p.wire_name, p.at) == ("w1", 0.4)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        PortOnWireFloating("bal", at=2.0)


def test_one_wire_carries_a_feed_and_a_load():
    """The limit this lifts: a port's name no longer has to be its wire's."""
    net = Network(
        ports={
            "feed": PortOnWire("feed", wire="w1", at=0.5),
            "load1": PortOnWire("load1", wire="w1", at=0.2),
        },
        branches=[Load(port="load1", r=50.0)],
        sources=[Driven(port="feed")],
    )
    assert {p.wire_name for p in net.ports.values()} == {"w1"}


@pytest.mark.parametrize(
    "clone",
    [copy.copy, copy.deepcopy, lambda p: pickle.loads(pickle.dumps(p))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_the_position_survives_copy_and_pickle(clone):
    p = PortOnWire("load1", wire="w1", at=0.2)
    q = clone(p)
    assert q == p and hash(q) == hash(p)


# ----------------------------------------------------------------------
# The placement reports, on #623's deck: a straight 10 m wire on a 20 m
# wavelength, N = 20, so h = 0.5 m and every grid point is exact.
# ----------------------------------------------------------------------
WIRE = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
N = 20
WL = 20.0
OFF_GRID = 2.9  # inside the cell [2.5, 3.0]: its centre is 2.75, its right knot 3.0
MIDPOINT = 5.0  # a knot, so it ties the centre-snappers between 4.75 and 5.25


def _solver(cls, arc, **kw):
    return cls(
        wires=[WIRE],
        n_per_edge_per_wire=[[N]],
        wavelength=WL,
        feed_arclength=arc,
        **kw,
    )


def _z(solver):
    return np.atleast_1d(solver.compute_impedance()[0])[0]


# (id, class, kwargs, where OFF_GRID lands, where a None request lands)
FAMILIES = [
    ("bspline-d2-point", BSplineSolver, {"degree": 2}, OFF_GRID, MIDPOINT),
    ("bspline-d1-point", BSplineSolver, {"degree": 1}, OFF_GRID, MIDPOINT),
    # A segment gap on a knot takes the cell to its right, so the midpoint
    # request spans [5.0, 5.5].
    (
        "bspline-segment",
        BSplineSolver,
        {"degree": 2, "feed_model": "segment"},
        2.75,
        5.25,
    ),
    (
        "galerkin-point",
        SinusoidalGalerkinSolver,
        {"feed_model": "point"},
        OFF_GRID,
        MIDPOINT,
    ),
    # The centre-snappers tie at the midpoint knot, and the smaller arclength
    # wins (#623): the report states that half-cell move.
    (
        "galerkin-segment",
        SinusoidalGalerkinSolver,
        {"feed_model": "segment"},
        2.75,
        4.75,
    ),
    ("sinusoidal-pm", SinusoidalSolver, {}, 2.75, 4.75),
    ("pulse", PulseSolver, {}, 2.75, 4.75),
    ("harrington", HarringtonSolver, {}, 2.75, 4.75),
    ("razor", RazorSolver, {}, 3.0, MIDPOINT),
]
GRID_LOCKED = [row for row in FAMILIES if row[3] != OFF_GRID]


def _rows(rows):
    return {
        "argvalues": [row[1:] for row in rows],
        "ids": [row[0] for row in rows],
    }


@pytest.mark.parametrize("cls,kw,off_grid,midpoint", **_rows(FAMILIES))
def test_each_family_reports_where_its_feed_lands(cls, kw, off_grid, midpoint):
    (p,) = _solver(cls, OFF_GRID, **kw).feed_placements()
    assert isinstance(p, FeedPlacement)
    assert p.wire == 0 and p.requested == OFF_GRID
    assert p.placed == pytest.approx(off_grid, abs=1e-12)
    assert p.offset == pytest.approx(off_grid - OFF_GRID, abs=1e-12)

    (m,) = _solver(cls, None, **kw).feed_placements()
    assert m.requested == pytest.approx(MIDPOINT, abs=1e-12)
    assert m.placed == pytest.approx(midpoint, abs=1e-12)


@pytest.mark.parametrize("cls,kw,off_grid,_midpoint", **_rows(GRID_LOCKED))
def test_a_grid_locked_family_solves_at_the_point_it_reports(
    cls, kw, off_grid, _midpoint
):
    """The report is the solve's own pick, not a second opinion. Driving the
    reported point gives exactly the answer the request gave, because a
    grid-locked family sees a request only through that pick."""
    asked = _solver(cls, OFF_GRID, **kw)
    (p,) = asked.feed_placements()
    assert _z(_solver(cls, p.placed, **kw)) == _z(asked)


def test_razor_reports_a_lumped_load_on_the_knot_a_feed_would_take():
    s = RazorSolver(
        wires=[WIRE],
        n_per_edge_per_wire=[[N]],
        wavelength=WL,
        lumped_loads=[(0, OFF_GRID, 50.0), (0, None, 10.0)],
    )
    near, middle = s.load_placements()
    assert (near.wire, near.requested) == (0, OFF_GRID)
    assert near.placed == pytest.approx(3.0, abs=1e-12)
    assert middle.placed == pytest.approx(MIDPOINT, abs=1e-12)
    assert s.feed_placements()[0].placed == pytest.approx(MIDPOINT, abs=1e-12)


@pytest.mark.parametrize("cls", [PulseSolver, RazorSolver], ids=["pulse", "razor"])
def test_asking_where_feeds_landed_leaves_the_623_tally_alone(
    cls, monkeypatch, tmp_path
):
    """`MOMWIRE_623_TALLY` counts the snaps a solve makes. A placement report
    re-asks a question the solve already asked, and must not count twice."""
    tally = []
    monkeypatch.setenv(_feed_snap._TAP, str(tmp_path / "tally"))
    monkeypatch.setattr(_feed_snap, "_TALLY", tally)
    s = _solver(cls, OFF_GRID)
    geom = s._build_geometry()
    s._feed_basis_indices(geom)
    assert len(tally) == 1
    s.feed_placements()
    assert len(tally) == 1
