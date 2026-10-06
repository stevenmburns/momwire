"""The crossing fill with its end loops inside the product's tiles, each
finished cross-block column streamed into Z, and its below block folded into
Z a window at a time — momwire#1173 design C phase 2.

Phase 1 kept a (rows, 2) V/W store for the end loops, which ran after the
main sandwich; each cross block and its end accumulators were whole beside
Z; and the crossing assembly held the whole below block `Z_b` beside Z.
`_FusedEnds` runs the loops inside the tile pass (the slow ends' spans before
it) and finishes the block a unit — a below basis function, a column of the
forward block and a row of the reversed one — at a time, folding each into Z
as it finishes; and `_assemble_Z_below_plane(into=)` adds each row window of
the below block into Z as it is finished.

Both are bit-identical by construction (the arguments are `_FusedEnds`' and
`_assemble_Z_below_plane`'s docstrings), so the gate is the bits: Z of a razor
fill through its real constructor (`_assemble_Z`), against the phase-1 route
in the same process (`_FUSED_ENDS = False`, `_BELOW_FOLD_INTO = False`), at
`np.array_equal` on the uint64 view. The counters say the new routes ran:
every product block fused (`fused_blocks`), in which mode ("stream", or
"post" when every end is slow: the default lane, loops after the tiles), no
V/W store was kept (`tile_stores`), every unit of every streamed block was
finished and folded (`stream_units`), in several batches with never more
than a fraction of them waiting (`stream_finish_calls`,
`stream_held_units`), rows were held across tiles where a unit straddles
them (`fused_held_rows`), and the below block went in by windows.

The negative controls make each route wrong on purpose and require the same
comparison to FAIL: "fused_order" replays a unit's vector-loop ends last
first, "shift" pairs each below window's direct rows with its image rows one
row out of step.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from momwire import _crossing_fill as cf
from momwire import razor as rz

import test_product_grid_1173 as pg
from test_crossing_serve_524 import crossing_deck, fan_rise_deck, invl_deck
from test_razor_detached_1149 import detached_hub
from momwire.razor import RazorSolver
from test_triple_memo_1168 import _hub, _razor


def _razor_default(deck):
    """The deck as razor on the DEFAULT quadrature lane (Gauss paths)."""
    return RazorSolver(**{k: v for k, v in deck.items() if k != "junctions"})


_PHASE1 = {"cf._FUSED_ENDS": False, "rz._BELOW_FOLD_INTO": False}


def _fill(make, **flags):
    """`pg._fill` with the below-fold counter reset and reported too."""
    rz._BELOW_FOLD_ROUTES["windows"] = 0
    Z, routes = pg._fill(make, **flags)
    routes["rz.below_windows"] = rz._BELOW_FOLD_ROUTES["windows"]
    return Z, routes


def _same(a, b):
    return a.shape == b.shape and np.array_equal(pg._bits(a), pg._bits(b))


# deck -> (maker, flags, the mode its blocks must take)
DECKS = {
    "crossing1": (lambda: _razor(crossing_deck(1)), {}, "stream"),
    "crossing1_default": (lambda: _razor_default(crossing_deck(1)), {}, "post"),
    "fan_rise": (lambda: _razor(fan_rise_deck()), {}, "stream"),
    "fan_rise_tiny": (lambda: _razor(fan_rise_deck()), {"cf._TILE_ROWS": 40}, "stream"),
    "wa7ark_tiny": (lambda: pg._wa7ark(True), {"cf._TILE_ROWS": 4_000}, "stream"),
    "detached_hub": (lambda: _razor(detached_hub()), {}, "stream"),
    "hub16_x2": (lambda: _razor(_hub(2)), {"cf._TILE_ROWS": 2_000}, "stream"),
    # Several groups (momwire#1335): the inverted L's above side is the mast
    # plus one group per top-wire node, in both blocks.
    "invl4": (lambda: _razor(invl_deck(n_radials=4)), {}, "stream"),
    "invl4_tiny": (
        lambda: _razor(invl_deck(n_radials=4)),
        {"cf._TILE_ROWS": 500},
        "stream",
    ),
}
_SLOW = {
    "crossing1_default",
    "fan_rise_tiny",
    "wa7ark_tiny",
    "detached_hub",
    "hub16_x2",
}


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, marks=pytest.mark.slow) if c in _SLOW else c for c in DECKS],
)
def test_fused_fill_is_the_phase1_route_to_the_bit(case):
    make, flags, mode = DECKS[case]
    ref, r0 = _fill(make, **{**flags, **_PHASE1})
    got, r = _fill(make, **flags)
    # The reference is phase 1: a store per product, the whole below block.
    assert r0["cf.fused_blocks"] == 0 and r0["rz.below_windows"] == 0, r0
    assert r0["cf.tile_stores"] == r0["cf.main_product"] >= 2, r0
    # The gated fill fused every product block and kept no store...
    n = r["cf.main_product"]
    assert n >= 2 and r["cf.fused_blocks"] == n and r["cf.tile_stores"] == 0, r
    assert r["cf.fused_declined"] == 0, r
    assert r[f"cf.fused_mode_{mode}"] == n, r
    if mode == "stream":
        # Every unit finished and folded, straight into Z; the ends read
        # inside the tiles.
        assert r["cf.stream_units"] > 0 and r["cf.fused_row_ends"] > 0, r
    else:
        assert r["cf.stream_units"] == 0, r
    # ...the same ends took the same fast / slow paths...
    for key in ("ends_fast", "ends_fast_grouped", "ends_fast_line", "ends_slow"):
        assert r[f"cf.{key}"] == r0[f"cf.{key}"], key
    # ...and the below block went into Z by windows.
    assert r["rz.below_windows"] > 0, r
    assert _same(got, ref)


@pytest.mark.slow
def test_tiny_tiles_hold_the_rows_a_straddling_unit_reads():
    """At 40-row tiles a unit's matvec row spans several tiles, so the rows
    it reads early are held — and Z still does not move."""
    make, flags, _mode = DECKS["fan_rise_tiny"]
    ref, _r = _fill(make, **{**flags, **_PHASE1})
    got, r = _fill(make, **flags)
    assert r["cf.fused_held_rows"] > 0 and r["cf.fused_unit_tiles"] > 2, r
    # The units finish in many batches, never all waiting at once.
    assert r["cf.stream_finish_calls"] > 4, r
    assert 0 < r["cf.stream_held_units"] < r["cf.stream_units"] // 4, r
    assert _same(got, ref)


def test_a_slow_end_asking_a_product_row_declines_the_fusion(monkeypatch):
    """With no end recognised as fast, the ends that WERE product rows ask
    the product through the memo — which a store-less product cannot answer
    — so the block declines and keeps phase 1's store; Z is phase 1's."""
    make = DECKS["crossing1"][0]
    with monkeypatch.context() as mp:
        mp.setattr(cf, "_fast_end_desc", lambda *a, **k: None)
        # ...which only the one-end classifier calls; the batched one
        # (momwire#1224) is gated to agree with it, not routed through it.
        mp.setattr(cf, "_END_CLASSIFY_BATCHED", False)
        ref, _r = _fill(make, **_PHASE1)
        got, r = _fill(make)
    assert r["cf.fused_declined_hit"] == r["cf.main_product"] >= 2, r
    assert r["cf.fused_blocks"] == 0 and r["cf.tile_stores"] >= 2, r
    assert _same(got, ref)


def test_a_product_of_several_groups_fuses_and_keeps_no_store():
    """momwire#1335: a product of several groups (the inverted L's) used to
    decline the fusion and keep the V/W store. Now both its blocks stream,
    with no store; at 500-row tiles a line end's keys and a unit's grouped
    rows span tiles, so rows are held across them — and Z is the declining
    route's (`_FUSED_MULTI_GROUP = False`) to the bit."""
    make, flags, _mode = DECKS["invl4_tiny"]
    ref, r0 = _fill(make, **{**flags, "cf._FUSED_MULTI_GROUP": False})
    got, r = _fill(make, **flags)
    n = r0["cf.main_product"]
    assert n >= 2 and r0["cf.main_product_groups"] > 1, r0
    assert r0["cf.fused_declined_groups"] == n == r0["cf.tile_stores"], r0
    assert r["cf.fused_mode_stream"] == n and r["cf.tile_stores"] == 0, r
    # Both loops' kinds ran inside the tiles: grouped vector ends over
    # several tiles' units, line ends formed from held rows.
    assert r["cf.fused_unit_tiles"] > 1 and r["cf.fused_col_te"] > 0, r
    assert r["cf.fused_held_rows"] > 0, r
    assert _same(got, ref)


def test_recycled_held_slots_are_the_one_slot_per_row_store_to_the_bit():
    """momwire#1335: a many-group product's held store reuses a slot once
    its row's last reader has run, so it holds the most rows live at once.
    The slots only hold copies, so Z is the one-slot-per-row store's
    (`_RECYCLE_HELD = False`) to the bit, with fewer slots."""
    make, flags, _mode = DECKS["invl4_tiny"]
    ref, r0 = _fill(make, **{**flags, "cf._RECYCLE_HELD": False})
    got, r = _fill(make, **flags)
    assert r["cf.fused_mode_stream"] == r["cf.main_product"] >= 2, r
    assert 0 < r["cf.held_slots"] < r0["cf.held_slots"], (r, r0)
    assert _same(got, ref)


@pytest.mark.parametrize("rows", [7, 1 << 20])
def test_batched_hit_check_walks_as_the_one_end_loop(rows, monkeypatch):
    """`_classify`'s slow-end hit check a span of rows at a time decides,
    counts and stops as the end-by-end check: on the declining case (every
    end slow, so the ends that are product rows hit) and on a fusing one."""
    make = DECKS["crossing1"][0]
    with monkeypatch.context() as mp:
        mp.setattr(cf, "_fast_end_desc", lambda *a, **k: None)
        mp.setattr(cf, "_END_CLASSIFY_BATCHED", False)
        ref, r0 = _fill(make, **{"cf._CLASSIFY_HIT_BATCHED": False})
        got, r = _fill(make, **{"cf._CLASSIFY_HIT_ROWS": rows})
    assert r["cf.fused_declined_hit"] == r0["cf.fused_declined_hit"] >= 2, r
    assert r["cf.ends_line_on_node"] == r0["cf.ends_line_on_node"], (r, r0)
    assert _same(got, ref)
    make, flags, _mode = DECKS["invl4"]
    ref, r0 = _fill(make, **{**flags, "cf._CLASSIFY_HIT_BATCHED": False})
    got, r = _fill(make, **{**flags, "cf._CLASSIFY_HIT_ROWS": rows})
    assert r["cf.fused_mode_stream"] == r0["cf.fused_mode_stream"] >= 2, r
    assert r["cf.ends_line_on_node"] == r0["cf.ends_line_on_node"], (r, r0)
    assert _same(got, ref)


@pytest.mark.parametrize(
    "flag",
    [
        "cf._PREPASS_MISSES_PROVEN",
        "cf._LINE_END_AT_COLUMN",
        "cf._TABLE_MATVECS",
        "cf._TILE_MATVECS_ROWS",
    ],
)
def test_each_1335_fused_switch_keeps_z(flag):
    """momwire#1335: each wall-time switch of the multi-group fused route,
    off, is the gated route's Z to the bit (invl, 500-row tiles: rows held
    across tiles, line ends and grouped ends spanning them)."""
    make, flags, _mode = DECKS["invl4_tiny"]
    ref, r0 = _fill(make, **{**flags, flag: False})
    got, r = _fill(make, **flags)
    assert r["cf.fused_mode_stream"] == r["cf.main_product"] >= 2, r
    assert r0["cf.fused_mode_stream"] == r0["cf.main_product"], r0
    assert _same(got, ref)


def test_batched_rank1_adds_are_the_per_term_loop_to_the_bit():
    """`_rank1_adds` against `_rank1_add` term by term, with rows several
    terms write (so the order of the additions is what is tested)."""
    rng = np.random.default_rng(13354)
    t0 = rng.standard_normal((9, 31)) + 1j * rng.standard_normal((9, 31))
    terms = []
    for _ in range(40):
        k = int(rng.integers(1, 4))
        nz = np.sort(rng.choice(9, k, replace=False))
        a = rng.standard_normal(k)
        b = rng.standard_normal(31) + 1j * rng.standard_normal(31)
        s = complex(rng.standard_normal(), rng.standard_normal()) * (-1) ** k
        terms.append((nz, a, b, s))
    want, got = t0.copy(), t0.copy()
    buf = cf._Rank1Buffer()
    for nz, a, b, s in terms:
        cf._rank1_add(want, nz, a, b, s, buf)
    cf._rank1_adds(got, terms)
    assert np.array_equal(got.view(np.uint64), want.view(np.uint64))


def test_batched_vector_adds_keep_z():
    make, flags, _mode = DECKS["invl4_tiny"]
    ref, _r0 = _fill(make, **{**flags, "cf._BATCHED_VECTOR_ADDS": False})
    got, r = _fill(make, **flags)
    assert r["cf.fused_mode_stream"] == r["cf.main_product"] >= 2, r
    assert _same(got, ref)


def test_held_slots_never_share_a_slot_between_live_rows():
    """`_held_slots` against brute force: two rows live in one tile (from
    their own tile through their last reading tile) never share a slot, and
    the slot count is the most rows live in any tile. A row read at its own
    tile or before is refused."""
    rng = np.random.default_rng(1335)
    n_tiles, n_rows = 30, 6000
    tile = np.sort(rng.integers(0, n_tiles, n_rows))
    by_tile = [np.flatnonzero(tile == t) for t in range(n_tiles)]
    last = np.full(n_rows, -1, dtype=np.int16)
    late = rng.random(n_rows) < 0.6
    late &= tile < n_tiles - 1
    span = 1 + rng.integers(0, 5, n_rows)
    last[late] = np.minimum(tile[late] + span[late], n_tiles - 1)
    hpos = np.full(n_rows, -1, dtype=np.int64)
    n_slots = cf._held_slots(n_tiles, lambda t: by_tile[t], last, hpos)
    assert np.array_equal(hpos >= 0, last >= 0)
    live_max = 0
    for t in range(n_tiles):
        live = (last >= 0) & (tile <= t) & (t <= last)
        live_max = max(live_max, int(live.sum()))
        s_t = hpos[live]
        assert np.unique(s_t).size == s_t.size, t
    assert n_slots == live_max and hpos.max() < n_slots
    assert cf._held_live_peak(n_tiles, lambda t: by_tile[t], last) == n_slots
    bad = last.copy()
    bad[np.flatnonzero(late)[0]] = tile[np.flatnonzero(late)[0]]
    with pytest.raises(AssertionError, match="not read after its own tile"):
        cf._held_slots(n_tiles, lambda t: by_tile[t], bad)


@pytest.mark.parametrize(
    "control, case, flag",
    [
        ("fused_order", "fan_rise_tiny", "cf._PRODUCT_NEG_CONTROL"),
        ("group0", "invl4", "cf._PRODUCT_NEG_CONTROL"),
        ("recycle", "invl4_tiny", "cf._PRODUCT_NEG_CONTROL"),
        ("shift", "crossing1", "rz._BELOW_FOLD_CONTROL"),
    ],
)
def test_negative_controls_move_z(control, case, flag):
    make, flags, _mode = DECKS[case]
    ref, _r = _fill(make, **{**flags, **_PHASE1})
    got, r = _fill(make, **{**flags, flag: control})
    if control in ("fused_order", "group0", "recycle"):
        assert r["cf.fused_mode_stream"] >= 2, r
    else:
        assert r["rz.below_windows"] > 0, r
    moved = int(np.count_nonzero(pg._bits(got) != pg._bits(ref)))
    assert moved > 0, f"{control}: the gate is blind to it"


def test_a_csr_on_its_stored_columns_is_the_whole_matvec_row_for_row():
    """`_csr_rows` + `_csr_on_cols` against `v[need]` give the whole
    matvec's rows to the bit (stored order kept, unsorted indices and an
    explicit zero included)."""
    rng = np.random.default_rng(1173)
    M = sp.random_array((60, 400), density=0.02, rng=rng, format="csr")
    M.data[3] = 0.0
    perm = rng.permutation(M.indices.size)
    for i in range(M.shape[0]):  # scramble each row's stored order
        a, b = M.indptr[i], M.indptr[i + 1]
        o = a + np.argsort(perm[a:b])
        M.indices[a:b], M.data[a:b] = M.indices[o].copy(), M.data[o].copy()
    M.has_sorted_indices = False
    v = rng.standard_normal(400) + 1j * rng.standard_normal(400)
    w = rng.standard_normal(400)
    want = cf._real_matvec_c(M, w * v)
    rows = np.sort(rng.choice(60, 17, replace=False))
    Mr, need = cf._csr_on_cols(cf._csr_rows(M, rows))
    got = cf._real_matvec_c(Mr, w[need] * v[need])
    assert np.array_equal(got.view(np.uint64), want[rows].view(np.uint64))
