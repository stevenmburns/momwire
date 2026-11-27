"""razor2p's reused windows assembled from the class values (momwire#2144,
`reuse::assemble_rows`).

With `pair_reuse` on, each window's entries come straight from the class
values, tile by tile, with no (n_need, n_seg) moment planes and no T1. It is
the same arithmetic as the route it replaces -- `gather_direct`, then
`assemble_t1` (or its weighted twin's constant-mirror window), then the fused
`t2_rows` -- so the answer must agree to the BIT, not to a tolerance. That
route is still in the tree: `verify` takes it, because it checks the planes
it gathers. So `verify` against on is the gate, on both routes, and each gate
first proves which branch ran (the window counters), since two fills that
silently took the same branch agree to the bit and measure nothing.

The decks cover what the tile reads: straight runs (the plain case), a
junction (wings on different wires, rising and falling), a monopole on the
plane (a grounded row drops its A centroid from T2), the PEC image and the
composing ground's weighted image.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _features, razor
from momwire._accel import acc
from momwire.razor import RazorSolver

pytestmark = [
    pytest.mark.reach(8),
    pytest.mark.skipif(
        acc is None or not getattr(acc, "razor_reuse_rows_2144", False),
        reason="needs the razor reuse row assembler",
    ),
    pytest.mark.skipif(
        acc is not None and not _features.certified("pair_reuse", solver="razor"),
        reason="razor2p's pair reuse is not certified on this platform",
    ),
    pytest.mark.filterwarnings("ignore"),
]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_RAZOR_PAIR_REUSE", raising=False)
    monkeypatch.setattr(razor, "_RAZOR_CXX", False)
    acc.razor_reuse_probe(0)
    acc.razor_reuse_plan_allowance(0)
    acc.razor_reuse_cache_clear()
    # cheap decks, below the small-deck thresholds on purpose
    acc.razor_reuse_set_small_deck_segs(0, 0)
    razor._PAIR_REUSE_ROUTE.clear()
    yield
    acc.razor_reuse_set_small_deck_segs(-1, -1)
    acc.razor_reuse_cache_clear()


SOMM = {"ground_z": 0.0, "ground_eps": (13.0, 0.005), "ground_model": "sommerfeld"}
GROUNDS = {"free": {}, "pec": {"ground_z": 0.0}, "sommerfeld": SOMM}


def _dipoles():
    return [np.array([[x, -5.0, 8.0], [x, 5.0, 8.0]]) for x in (0.0, 3.0)]


def _junction():
    """A vee meeting a dipole's run at a through knot, beside a reversed
    dipole: wings on different wires, rising and falling, and tiles whose
    columns reach across wires."""
    return [
        np.array([[0.0, 0.0, 6.0], [0.0, 3.0, 9.0]]),
        np.array([[0.0, 0.0, 6.0], [0.0, -3.0, 9.0]]),
        np.array([[2.0, 5.0, 8.0], [2.0, -5.0, 8.0]]),
    ]


def _monopole():
    """A vertical on the plane beside a parasitic one: its first basis row
    is grounded."""
    return [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        np.array([[1.5, 0.0, 0.0], [1.5, 0.0, 5.0]]),
    ]


DECKS = {
    ("dipoles", "free"): _dipoles,
    ("dipoles", "pec"): _dipoles,
    ("dipoles", "sommerfeld"): _dipoles,
    ("junction", "free"): _junction,
    ("junction", "sommerfeld"): _junction,
    ("monopole", "pec"): _monopole,
    ("monopole", "sommerfeld"): _monopole,
}


def _solve(wires, mode, ground, nsegs=37):
    s = RazorSolver(
        wires=wires,
        nsegs=nsegs,
        wire_radius=0.001,
        wavelength=20.0,
        nec5_quadrature=True,
        extended_kernel=True,
        razor_pair_reuse=mode,
        **GROUNDS[ground],
    )
    return s.compute_port_solution().y


def _rows_windows():
    return razor._REUSE_ROWS_COUNT[0]


@pytest.mark.parametrize(("deck", "ground"), sorted(DECKS))
def test_on_is_the_three_kernel_route_to_the_bit(deck, ground):
    wires = DECKS[deck, ground]()
    r0 = _rows_windows()
    checked = _solve(wires, "verify", ground)
    # verify fills through the planes it checks: no window took the rows
    assert _rows_windows() == r0
    on = _solve(wires, True, ground)
    assert razor._PAIR_REUSE_ROUTE.get("reuse", 0) == 2
    # one window per block at least: the real one, and the image over a plane
    assert _rows_windows() >= r0 + (1 if ground == "free" else 2)
    np.testing.assert_array_equal(on, checked)


def test_a_grounded_row_is_on_the_rows_route():
    """The monopole's grounded row is what the tile's T2 special-cases: its
    A centroid is dropped. Its plan serves (the reuse ran), and the deck's
    first row really is grounded, so the bit gate above covers it."""
    s = RazorSolver(
        wires=_monopole(),
        nsegs=37,
        wire_radius=0.001,
        wavelength=20.0,
        nec5_quadrature=True,
        extended_kernel=True,
        razor_pair_reuse=True,
        **GROUNDS["pec"],
    )
    geom = s._build_geometry()
    prepared = s._assemble_Z_prepare(geom)
    assert prepared["grounded"].size > 0


@pytest.mark.parametrize("ground", ["free", "pec", "sommerfeld"])
def test_the_cxx_route_takes_the_rows_and_agrees_to_the_bit(ground, monkeypatch):
    """The solve core (the executables' fill) assembles its reused windows
    by the same function: on and `verify` agree to the bit, on takes the
    rows (`reuse_row_windows`) and `verify` the planes, and both are the
    Python route's answer bit for bit."""
    wires = _junction()
    py = _solve(wires, True, ground)
    monkeypatch.setattr(razor, "_RAZOR_CXX", True)
    ctx = razor._cxx_context()
    c0 = dict(ctx.counters)
    on = _solve(wires, True, ground)
    c1 = dict(ctx.counters)
    checked = _solve(wires, "verify", ground)
    c2 = dict(ctx.counters)
    assert razor._CXX_FRONT_ROUTE.get("front", 0) > 0
    assert c1["reuse_row_windows"] > c0["reuse_row_windows"]
    assert c1["reuse_row_windows"] - c0["reuse_row_windows"] == (
        c1["reuse_windows"] - c0["reuse_windows"]
    )
    assert c2["reuse_row_windows"] == c1["reuse_row_windows"]
    assert c2["reuse_windows"] > c1["reuse_windows"]
    np.testing.assert_array_equal(on, checked)
    np.testing.assert_array_equal(on, py)
