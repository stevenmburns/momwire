"""The crossing fill's product rows evaluated in tiles — momwire#1173 design C
phase 1.

Design B builds the main sandwich's distinct (ρ_eff, z, z′) rows from O(N)
factors but evaluates them in one call, so their (rows, 6) value block — the
largest resident term of razor hub_deck(16) x16, ~150 MB — is whole.
`_ProductTiles` evaluates them a tile at a time, each tile a union of WHOLE
exact-ρ columns, serves each below node's table column when its tile
completes it, and keeps only V and W (the end loops' tables) in a row-ordered
store. `_streamed_sandwich` contracts the column sets as they come.

Bit-identical by construction (the argument is `_ProductTiles`' docstring), so
the gate is the bits: Z of a razor fill through its real constructor
(`_assemble_Z`) at a tile budget small enough to cut every product into many
tiles, against the grid route with every design-B switch off — the code
design B was gated against to the bit — at `np.array_equal`. The counters say
the tiles ran (`tiles` > the product count), every row was evaluated exactly
once (`tile_rows` == `product_rows`; a second evaluation raises), and whether
the held path ran: never on a product grouped above (a below node's rows
share its key), always on one grouped below (a node's rows span the line).

The negative controls make the tiles wrong on purpose and require the same
comparison to FAIL: "split" evaluates each tile as two calls, cutting its
columns — a column's second part then runs under the wrong rule witness
(s_min); "held" serves each held row's U and dz′W from its neighbour.

Measured 2026-09-24 by script against origin/main 4f08471, Z to the bit (see
the PR): razor hub_deck(16) x2 / x4 / x8 (x16 on Skylake), crossing_deck(1),
detached, detached_hub, fan_rise, the sloped radials, WA7ARK's ground-rod
EFHW, and lean20 / two_node (grid route, unchanged), at the shipped budget and
at a tiny one; both lanes to hub x4.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _crossing_fill as cf
from momwire import _near_interface as ni

import test_product_grid_1173 as pg
from test_crossing_serve_524 import crossing_deck, fan_rise_deck, two_node_deck
from test_razor_detached_1149 import detached, detached_hub
from test_tilted_crossing_936 import _crossing_deck as tilted_deck
from test_triple_memo_1168 import _hub, _razor


# Every gate here runs at the shipped cost rule and with every plane taken
# (conftest's `sheet_modes`, momwire#1173 Design E phase 2).
pytestmark = pytest.mark.usefixtures("sheet_modes")


_TINY = {"cf._TILE_ROWS": 40}
# WA7ARK's product is 74 k rows; 40-row tiles would be ~860 kernel calls.
# 4,000 still cuts it into tens of tiles and still holds rows.
_TINY_BY_DECK = {"wa7ark": {"cf._TILE_ROWS": 4_000}}


def _rod_under_lean():
    deck = tilted_deck(20.0, n_mast=6)
    deck["wires"][0] = np.array([(0.0, 0.0, -2.0), (0.0, 0.0, 0.0)])
    deck["n_per_edge_per_wire"][0] = [8]
    return _razor(deck)


# deck -> (maker, the product's shape: "z" grouped above, "zp" grouped
# below, "column" a product that is ONE exact-ρ column — a vertical below
# member straight under the mast puts every pair at ρ = a — and so one tile
# whatever the budget: a column is never cut)
DECKS = {
    "crossing1": (lambda: _razor(crossing_deck(1)), "column"),
    "detached": (lambda: _razor(detached()), "column"),
    "detached_hub": (lambda: _razor(detached_hub()), "z"),
    "fan_rise": (lambda: _razor(fan_rise_deck()), "z"),
    "sloped": (lambda: _razor(pg.sloped_radials_deck()), "z"),
    "rod_under_lean": (_rod_under_lean, "zp"),
    "wa7ark": (lambda: pg._wa7ark(True), "zp"),
}
_SLOW = {"detached_hub", "sloped", "rod_under_lean"}


def _grid(make):
    """The reference: the grid route and the lookup path (design B off)."""
    return pg._fill(make, **{"cf._PRODUCT_TABLES": False, "cf._PRODUCT_ENDS": False})


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, marks=pytest.mark.slow) if c in _SLOW else c for c in DECKS],
)
def test_tiles_are_the_grid_route_to_the_bit(case):
    make, slot = DECKS[case]
    ref, r_off = _grid(make)
    got, r = pg._fill(make, **_TINY_BY_DECK.get(case, _TINY))
    assert r_off["cf.main_product"] == 0 and r_off["cf.tiles"] == 0
    n_products = r["cf.main_product"]
    assert n_products >= 2 and r["cf.main_generic"] == 0, r
    # Many tiles per product (one per product when it is one column), every
    # row exactly once.
    if slot == "column":
        assert r["cf.tiles"] == n_products, r
    else:
        assert r["cf.tiles"] > 2 * n_products, r
        assert r["cf.stream_chunks"] > 0, r
    assert r["cf.tile_rows"] == r["cf.product_rows"] > 0, r
    if slot == "z":
        assert r["cf.tile_held_rows"] == 0, r  # a node's rows share its key
    elif slot == "zp":
        assert r["cf.tile_held_rows"] > 0, r  # a node's rows span the line
    assert np.array_equal(pg._bits(got), pg._bits(ref))


def test_the_shipped_budget_is_one_tile_on_a_small_deck():
    """At 2^17 rows a small deck is one tile, served as design B served it."""
    make = DECKS["crossing1"][0]
    ref, _r = _grid(make)
    got, r = pg._fill(make)
    assert r["cf.tiles"] == r["cf.main_product"] == 2, r
    assert r["cf.tile_rows"] == r["cf.product_rows"], r
    assert np.array_equal(pg._bits(got), pg._bits(ref))


@pytest.mark.slow
def test_two_groups_tile_to_the_bit():
    """The multi-group merge (two masts, taken by default): rows shared by
    the two groups are one row under one key, evaluated once."""
    make = lambda: _razor(two_node_deck(separation=12.0))  # noqa: E731
    ref, _r = _grid(make)
    got, r = pg._fill(make, **_TINY)
    assert r["cf.main_product_groups"] == 2, r
    assert r["cf.tiles"] > r["cf.main_product"], r
    assert r["cf.tile_rows"] == r["cf.product_rows"], r
    assert np.array_equal(pg._bits(got), pg._bits(ref))


@pytest.mark.slow
@pytest.mark.parametrize("x", [2, 4])
def test_hub16_tiles_to_the_bit(x):
    """The headline deck, at a budget that cuts each block into tens of
    tiles; nothing is ever held (the mast is one above group)."""
    make = lambda: _razor(_hub(x))  # noqa: E731
    ref, _r = _grid(make)
    got, r = pg._fill(make, **{"cf._TILE_ROWS": 2_000})
    assert r["cf.main_product"] == 2 and r["cf.tiles"] > 20, r
    assert r["cf.tile_rows"] == r["cf.product_rows"], r
    assert r["cf.tile_held_rows"] == 0, r
    assert np.array_equal(pg._bits(got), pg._bits(ref))


# ----------------------------------------------------------------------
# negative controls: the same gate FAILS on wrong tiles
# ----------------------------------------------------------------------


@pytest.mark.parametrize("control, case", [("split", "fan_rise"), ("held", "wa7ark")])
def test_negative_controls_move_z(control, case):
    """ "split" cuts every tile's columns (the rule witness s_min is then
    wrong for a column's second part); "held" serves the held rows'
    U and dz′W from a neighbour. Each must move Z at the tiny budget."""
    make, _slot = DECKS[case]
    ref, _r = _grid(make)
    tiny = _TINY_BY_DECK.get(case, _TINY)
    got, r = pg._fill(make, **{"cf._PRODUCT_NEG_CONTROL": control, **tiny})
    assert r["cf.tiles"] > 2 * r["cf.main_product"], r
    if control == "held":
        assert r["cf.tile_held_rows"] > 0, r
    moved = int(np.count_nonzero(got != ref))
    assert moved > 0, f"{control}: the gate is blind to it"


# ----------------------------------------------------------------------
# unit rows: the tiles are the one call's columns, whole
# ----------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["one_group", "below_group", "two_groups"])
def test_tiles_are_whole_columns_of_the_one_call(shape, monkeypatch):
    """At a budget of a few rows, on node sets with repeated z, signed zeros
    and triples shared between groups: each call is handed whole exact-ρ
    columns (no ρ in two calls), each call's rows ascend in row order, and
    the calls together hand over exactly the one-call rows."""
    rng = np.random.default_rng(1173)
    zs_a = rng.choice([0.5, 1.0, 1.5, 2.0], 12)
    zs_b = -rng.choice([0.0, 0.1, 0.2], 9)
    zs_b[0] = -0.0
    xy_b = rng.choice([0.0, 1.0, 2.0], (9, 2))
    if shape == "one_group":
        A, B = pg._axes(np.zeros((12, 2)), zs_a, xy_b, zs_b)
    elif shape == "below_group":
        A, B = pg._axes(
            rng.choice([0.0, 3.0, 4.0], (12, 2)), zs_a, np.zeros((9, 2)), zs_b
        )
    else:
        xy_a = np.zeros((12, 2))
        xy_a[6:] = (2.0, 0.0)
        A, B = pg._axes(xy_a, zs_a, xy_b, zs_b)
    flags = {
        "_PRODUCT_MAX_CAND_FRAC": 10.0,
        "_PRODUCT_MAX_GROUP_FRAC": 1.0,
        "_TILE_ROWS": 5,
    }
    (product, _fast, chunk_idx), seen = pg._plan_rows(A, B, monkeypatch, **flags)
    want = pg._grid_rows(pg._Ctx(), A, B, 0.0)
    assert len(seen) > 2
    rho_calls = [set(s[:, 0].tolist()) for s in seen]
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert not rho_calls[i] & rho_calls[j], "a column was cut across tiles"
    got = np.concatenate(seen)
    order = np.argsort(product.value_rows(got))
    assert np.array_equal(got[order], want)
    assert np.array_equal(np.signbit(got[order]), np.signbit(want))
    for s in seen:
        v = product.value_rows(s)
        assert np.all(v[1:] > v[:-1]), "a tile's rows are not in row order"
    # The store holds each row's V and W; the others were never kept.
    nB = B["nodes"].shape[0]
    block = product.value_block(chunk_idx(np.arange(nB)).ravel())
    assert np.isnan(block[:, ni.KEYS.index("U")]).all()
    assert not np.isnan(block[:, ni.KEYS.index("V")]).any()


def test_the_product_refuses_a_lookup_before_its_tiles_finish(monkeypatch):
    """A HIT on a product still being evaluated refuses; a MISS is answered
    (the factors decide it, so it is what the finished product would say —
    design C phase 2's slow end loops run before the tiles on exactly
    that)."""
    A, B = pg._axes(
        np.zeros((3, 2)), [0.5, 1.0, 1.5], [(1.0, 0.0), (2.0, 0.0)], [-0.1, -0.2]
    )
    with monkeypatch.context() as mp:
        mp.setattr(cf, "_TILE_ROWS", 2)
        plan = cf._product_plan(pg._Ctx(), 1.0, 1.0, A, B, 0.0)
        tiles = cf._ProductTiles(plan, 1.0, 1.0, 1)
        tiles.keep_values()
        memo = ni.ProductMemo()
        memo.set_product(tiles.product)
        held_row = plan.rows(np.array([0]))
        hit, _block = memo.lookup(np.zeros((1, 3)))
        assert not hit.any()
        gen = tiles.chunks(1)
        next(gen)
        with pytest.raises(RuntimeError, match="still being evaluated"):
            memo.lookup(held_row)
        list(gen)
    assert tiles.product.complete
    hit, _block = memo.lookup(np.zeros((1, 3)))
    assert not hit.any()
    hit, _block = memo.lookup(held_row)
    assert hit.all()
