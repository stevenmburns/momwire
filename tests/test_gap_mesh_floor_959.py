"""The mesh floor AT A GAP warns — momwire#959, a stopgap for momwire#1330
(bspline's exact-kernel text: tests/test_exact_kernel_advisories_1412.py).

Measured (the `GapMeshFloor` docstring carries the numbers): the delta-gap
model collapses toward 0 ohm when the segments at a gap are shorter than the
wire radius, and how many such segments it takes depends on the gap model.
Free runs of short segments, and two-wire junctions, are harmless and stay
silent. Every check here is construction-only: the advisory is raised before
any fill.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire._wire_spec import (
    GAP_FLOOR_EXTENDED,
    GAP_FLOOR_REDUCED,
    GapMeshFloor,
    gap_past_floor,
)
from momwire.bspline import BSplineSolver
from momwire.harrington import HarringtonSolver
from momwire.pulse import PulseSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

WL = 299792458.0 / 600e6
A = 3.175e-3
L = 0.24
D0 = 2.0 * A


def _advice(cls, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls(wavelength=WL, wire_radius=A, **kw)
    return [str(w.message) for w in caught if issubclass(w.category, GapMeshFloor)]


def _fed(n, d, **kw):
    """A dipole on a 2 a mesh with `n` segments of `d` a centred on the feed."""
    span = n * d * A
    wire = np.array(
        [(0, 0, -L / 2), (0, 0, -span / 2), (0, 0, span / 2), (0, 0, L / 2)]
    )
    side = int(round((L / 2 - span / 2) / D0))
    return dict(
        wires=[wire],
        n_per_edge_per_wire=[[side, n, side]],
        feeds=[(0, L / 2, 1 + 0j)],
        **kw,
    )


# (family, kwargs, the stencil that fails as measured, one that does not)
CASES = [
    ("bspline", BSplineSolver, {}, (3, 0.1), (1, 0.1)),
    ("bspline-segment", BSplineSolver, {"feed_model": "segment"}, (3, 0.3), (1, 0.1)),
    ("sin-galerkin", SinusoidalGalerkinSolver, {}, (3, 0.3), (1, 0.1)),
    ("sinusoidal", SinusoidalSolver, {}, (1, 0.3), (1, 2.0)),
    ("pulse", PulseSolver, {}, (1, 0.3), (1, 2.0)),
    ("harrington", HarringtonSolver, {}, (1, 0.3), (1, 2.0)),
    ("razor", RazorSolver, {}, (2, 0.3), (2, 1.0)),
]


@pytest.mark.parametrize("_id, cls, kw, bad, ok", CASES, ids=[c[0] for c in CASES])
def test_the_measured_failing_stencil_warns_and_its_neighbour_does_not(
    _id, cls, kw, bad, ok
):
    found = _advice(cls, **_fed(*bad, **kw))
    assert len(found) == 1
    msg = found[0]
    assert "source 0 (wire 0 at 0.12 m)" in msg
    # bspline can take the exact kernel and names it (momwire#1412); the
    # other families point at momwire#1330
    fix = "momwire#1408" if cls is BSplineSolver else "momwire#1330"
    assert fix in msg and "momwire#959" in msg
    assert "coarsen" not in msg and "fewer segments" not in msg
    assert _advice(cls, **_fed(*ok, **kw)) == []


@pytest.mark.parametrize("cls", [BSplineSolver, SinusoidalSolver, RazorSolver])
def test_a_free_run_away_from_every_gap_is_silent(cls):
    """Ten radii of 0.2 a segments at x = -0.06, the feed on a 2 a mesh:
    measured <= 1e-4 on every family."""
    r = 10 * A
    wire = np.array([(0, 0, -L / 2), (0, 0, -0.06), (0, 0, -0.06 + r), (0, 0, L / 2)])
    npe = [
        int(round((L / 2 - 0.06) / D0)),
        50,
        int(round((L / 2 + 0.06 - r) / D0)),
    ]
    centre = L / 2
    assert (
        _advice(
            cls, wires=[wire], n_per_edge_per_wire=[npe], feeds=[(0, centre, 1 + 0j)]
        )
        == []
    )


def _loaded(n):
    """A fine span of `n` 0.2 a segments around x = -0.06, on a 2 a mesh."""
    r = n * 0.2 * A
    x = -0.06
    wire = np.array(
        [(0, 0, -L / 2), (0, 0, x - r / 2), (0, 0, x + r / 2), (0, 0, L / 2)]
    )
    npe = [
        int(round((x - r / 2 + L / 2) / D0)),
        n,
        int(round((L / 2 - x - r / 2) / D0)),
    ]
    return wire, npe, x + L / 2


def test_a_load_is_a_gap():
    """Loads fail like feeds (a 50 ohm load vanished from the answer, 28 %),
    so razor's lumped load and a second port both warn."""
    wire, npe, at = _loaded(10)
    msg = _advice(
        RazorSolver,
        wires=[wire],
        n_per_edge_per_wire=[npe],
        feeds=[(0, L / 2, 1 + 0j)],
        lumped_loads=[(0, at, 50.0)],
    )
    assert len(msg) == 1 and "load 0 (wire 0 at" in msg[0]
    wire, npe, at = _loaded(11)
    msg = _advice(
        BSplineSolver,
        wires=[wire],
        n_per_edge_per_wire=[npe],
        feeds=[(0, L / 2, 1 + 0j), (0, at, 0j)],
    )
    assert len(msg) == 1 and "source 1" in msg[0] and "source 0" not in msg[0]


def _tee(**kw):
    x, r = -0.06, 2 * A
    k = 5
    return dict(
        wires=[
            np.array([(0, 0, -L / 2), (0, 0, x - r / 2), (0, 0, x)]),
            np.array([(0, 0, x), (0, 0, x + r / 2), (0, 0, L / 2)]),
            np.array([(0, 0, x), (r / 2, 0, x), (0.04, 0, x)]),
        ],
        n_per_edge_per_wire=[
            [int(round((x - r / 2 + L / 2) / D0)), k],
            [k, int(round((L / 2 - x - r / 2) / D0))],
            [k, int(round((0.04 - r / 2) / D0))],
        ],
        feeds=[(1, -x, 1 + 0j)],
        **kw,
    )


def test_a_tee_counts_only_where_it_was_measured_to_fail():
    """A fine T node: harmless on the reduced kernel and on bspline / razor
    with the extended one; the sinusoidal families fail there under the
    extended kernel (0.27-0.69)."""
    for cls in (SinusoidalSolver, SinusoidalGalerkinSolver, BSplineSolver, RazorSolver):
        assert _advice(cls, **_tee()) == []
    for cls in (BSplineSolver, RazorSolver):
        assert _advice(cls, **_tee(extended_kernel=True)) == []
    for cls in (SinusoidalSolver, SinusoidalGalerkinSolver):
        msg = _advice(cls, **_tee(extended_kernel=True))
        assert len(msg) == 1 and "junction 0 (3 wires)" in msg[0]


def test_the_extended_kernel_is_named_only_where_it_helps():
    between = 0.7  # under the reduced floor, over the extended one
    assert GAP_FLOOR_EXTENDED < between < GAP_FLOOR_REDUCED
    (msg,) = _advice(SinusoidalSolver, **_fed(1, between))
    assert "extended_kernel=True holds to delta/a ~ 0.6" in msg
    assert _advice(SinusoidalSolver, **_fed(1, between, extended_kernel=True)) == []
    (msg,) = _advice(SinusoidalSolver, **_fed(1, 0.3))
    assert "extended_kernel" not in msg


def test_one_advisory_names_every_gap():
    deck = _fed(5, 0.1)
    deck["feeds"] = [(0, L / 2, 1 + 0j), (0, L / 2 - 0.1 * A, 1 + 0j)]
    found = _advice(BSplineSolver, **deck)
    assert len(found) == 1
    assert "source 0" in found[0] and "source 1" in found[0]


def test_dans_sliver_at_the_crossing_stays_silent():
    """thr_n204's split: a 9.46 um (0.009 a) piece above z = 0 between a
    25.4 mm below half and 25.4 mm segments, fed at the cut. One short
    segment between long ones is not past any family's floor."""
    a = 1.02616e-3
    wire = np.array([(0, 0, -0.0254), (0, 0, 0.0), (0, 0, 9.46e-6), (0, 0, 5.15813)])
    for cls in (BSplineSolver, SinusoidalGalerkinSolver, RazorSolver):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cls(
                wires=[wire],
                n_per_edge_per_wire=[[1, 1, 203]],
                wire_radius=a,
                wavelength=299792458.0 / 14e6,
                feeds=[(0, 0.0254, 1 + 0j)],
            )
        assert not [w for w in caught if issubclass(w.category, GapMeshFloor)]


def test_the_stencil():
    """`gap_past_floor` on a hand-made mesh: 10 segments, 3..6 short."""
    pl = np.array([(0.0, 0, 0), (1.0, 0, 0), (1.4, 0, 0), (2.4, 0, 0)])
    npe = [1, 4, 1]  # lengths 1, 0.1 x4, 1
    a, floor = 1.0, 0.85
    # mid-segment in the run: 1 own + 1 beyond on the left, 2 on the right
    assert gap_past_floor(pl, npe, a, 1.15, floor, 1)[0]
    assert not gap_past_floor(pl, npe, a, 1.05, floor, 1)[0]  # 0 beyond, left
    assert gap_past_floor(pl, npe, a, 1.05, floor, 0)[0]
    assert not gap_past_floor(pl, npe, a, 0.5, floor, 0)[0]  # a long gap
    fires, ratio, span = gap_past_floor(pl, npe, a, 1.2, floor, 0)  # a knot
    assert fires and ratio == pytest.approx(0.1) and span == pytest.approx(0.4)
