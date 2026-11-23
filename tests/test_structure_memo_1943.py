"""momwire#1943 lever 1: a solver per FR step shares the per-structure
constructor work, by IDENTITY of the prepared mesh's frozen polylines.

The nec2c launchers build a solver per frequency from one `PreparedMesh`, and
each construction re-ran the duplicate-wire search, the junction coincidence
check, the short-segment census and the sector map's digest over the same
arrays. `prepare_mesh` now freezes its polylines and those four recognise a
frozen array by who it is (`momwire._identity_memo`).

The gates: each check computed once across a four-frequency FR run through
the portal's own seam, the port answers BIT-identical to a run whose memos
hold nothing, and a writable array (any caller but a prepared mesh) never
served from a memo.
"""

from __future__ import annotations

import numpy as np
import pytest

import momwire._rotational_symmetry as RS
import momwire._wire_spec as WS
import momwire.sinusoidal as S
from momwire import _identity_memo as IM
from momwire.deck import parse
from momwire.deck._solver import prepare_mesh
from momwire.dropin.nec2c import _portal

from test_sector_map_once_1929 import DECK

# `_JUNCTIONS_MEMO` holds the whole normalised junction answer, so the
# coincidence check inside it runs once (a miss) and is never asked again: its
# three later steps are served one level up. `_GEOMETRY_MEMO` is the solver's
# geometry tables.
_MEMOS = (
    WS._DUPLICATES_MEMO,
    WS._JUNCTIONS_MEMO,
    WS._SHORT_SEGMENTS_MEMO,
    S._GEOMETRY_MEMO,
)


@pytest.fixture(params=[False, True], ids=["dense", "sector"])
def route(request, monkeypatch):
    """Both routes: dense, and `rotational_symmetry=True` on every
    SinusoidalSolver the portal builds (the ladder harness's `mw_sector.py`)."""
    if request.param:
        real = S.SinusoidalSolver.__init__

        def init(self, *a, **k):
            k["rotational_symmetry"] = True
            real(self, *a, **k)

        monkeypatch.setattr(S.SinusoidalSolver, "__init__", init)
    for memo in _MEMOS:
        memo.clear()
    RS._SECTOR_MAP_CACHE.clear()
    yield request.param
    for memo in _MEMOS:
        memo.clear()
    RS._SECTOR_MAP_CACHE.clear()


def _sweep():
    deck = _portal.parse_deck(DECK)
    solver = _portal.DeckSolver(deck, basis="sinusoidal")
    (group,) = deck.groups
    out = []
    for f in group.freqs_mhz:
        r = solver.solve_group(group, f)
        out.append((solver.at(f)["Y"], r.v_gap, r.i_port, r.segment_currents, r.coeffs))
    return out


def test_each_check_runs_once_across_the_fr_loop(route):
    before = [(m.hits, m.misses) for m in _MEMOS]
    coincidence = (WS._COINCIDENCE_MEMO.hits, WS._COINCIDENCE_MEMO.misses)
    memoised = _sweep()
    assert len(memoised) == 4
    for memo, (hits, misses) in zip(_MEMOS, before, strict=True):
        # One computation for the structure, and the other three steps served.
        assert memo.misses - misses == 1
        assert memo.hits - hits == 3
    assert WS._COINCIDENCE_MEMO.misses - coincidence[1] == 1
    assert WS._COINCIDENCE_MEMO.hits - coincidence[0] == 0

    # The same sweep with memos that hold nothing: the same bits.
    for memo in _MEMOS:
        memo.clear()
    RS._SECTOR_MAP_CACHE.clear()
    memos = (*_MEMOS, WS._COINCIDENCE_MEMO)
    sizes = [m._size for m in memos]
    try:
        for memo in memos:
            memo._size = 0
        fresh = _sweep()
    finally:
        for memo, size in zip(memos, sizes, strict=True):
            memo._size = size
    for a, b in zip(memoised, fresh, strict=True):
        for x, y in zip(a, b, strict=True):
            x, y = np.asarray(x), np.asarray(y)
            assert x.dtype == y.dtype and x.shape == y.shape
            assert x.tobytes() == y.tobytes()


def test_the_prepared_mesh_is_frozen():
    mesh = prepare_mesh(parse(DECK))
    assert all(IM.frozen(p) for p in mesh.mesh.polylines)
    with pytest.raises(ValueError):
        mesh.mesh.polylines[0][0, 0] = 1.0


def test_only_a_frozen_array_is_recognised():
    base = np.zeros((3, 3))
    view = base[:2]
    view.setflags(write=False)
    # A read-only view of a writable buffer can still change under it.
    assert not IM.frozen(view)
    assert not IM.frozen(base)
    assert IM.frozen_tuple([base]) is None
    owned = np.ones((2, 3))
    IM.freeze([owned])
    assert IM.frozen(owned) and IM.frozen(owned[1:])

    memo = IM.IdentityMemo()
    calls = []

    def compute():
        calls.append(1)
        return len(calls)

    # Writable: never stored, never served.
    assert memo.get(IM.frozen_tuple([base]), None, compute) == 1
    assert memo.get(IM.frozen_tuple([base]), None, compute) == 2
    # Frozen: served for the same array and an equal extra...
    assert memo.get((owned,), ("x",), compute) == 3
    assert memo.get((owned,), ("x",), compute) == 3
    # ...and not for an equal COPY, nor for another extra.
    twin = owned.copy()
    IM.freeze([twin])
    assert memo.get((twin,), ("x",), compute) == 4
    assert memo.get((owned,), ("y",), compute) == 5


def test_a_writable_polyline_is_recomputed_every_time():
    wires = [np.array([[0.0, 0, 0], [0, 0, 1.0]]), np.array([[0.0, 0, 0], [0, 0, 1.0]])]
    misses = WS._DUPLICATES_MEMO.misses
    hits = WS._DUPLICATES_MEMO.hits
    assert WS.find_duplicated_wires(wires) == [(1, 0, False)]
    wires[1][1, 2] = 2.0  # writable, so it may change -- and it did
    assert WS.find_duplicated_wires(wires) == []
    assert WS._DUPLICATES_MEMO.hits == hits
    assert WS._DUPLICATES_MEMO.misses == misses


def test_a_refusal_is_never_stored():
    a = np.array([[0.0, 0, 0], [0, 0, 1.0]])
    b = np.array([[0.0, 0, 0.5], [0, 1.0, 0.5]])
    IM.freeze([a, b])
    bad = [[(0, "end"), (1, "start")]]  # 0.5 m apart: not one point
    for _ in range(2):
        with pytest.raises(ValueError, match="do not coincide"):
            WS.check_junction_coincidence([a, b], [[3], [3]], bad)
