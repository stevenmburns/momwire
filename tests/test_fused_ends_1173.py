"""The crossing fill with its end loops inside the product's tiles and its
below block folded into Z a window at a time — momwire#1173 design C phase 2.

Phase 1 kept a (rows, 2) V/W store for the end loops, which ran after the
main sandwich, and the crossing assembly held the whole below block `Z_b`
beside Z. `_FusedEnds` runs the loops inside the tile pass (the slow ends'
spans before it, the column loop after it), so no store is kept; and
`_assemble_Z_below_plane(into=)` adds each row window of the below block into
Z as it is finished.

Both are bit-identical by construction (the arguments are `_FusedEnds`' and
`_assemble_Z_below_plane`'s docstrings), so the gate is the bits: Z of a razor
fill through its real constructor (`_assemble_Z`), against the phase-1 route
in the same process (`_FUSED_ENDS = False`, `_BELOW_FOLD_INTO = False`), at
`np.array_equal` on the uint64 view. The counters say the new routes ran:
every product block fused (`fused_blocks`), which of the row loop's three
modes ran ("grouped": per-unit completion across tiles; "line": one tile per
end, per-row chains; "post": no fast row ends, the loop after the tiles), no
V/W store was kept (`tile_stores`), rows were held across tiles where a unit
straddles them (`fused_held_rows`), and the below block went in by windows.

The negative controls make each route wrong on purpose and require the same
comparison to FAIL: "fused_order" applies a tile's row ends in reverse (and
ignores the chains), "shift" pairs each below window's direct rows with its
image rows one row out of step.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from momwire import _crossing_fill as cf
from momwire import razor as rz

import test_product_grid_1173 as pg
from test_crossing_serve_524 import crossing_deck, fan_rise_deck
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


# deck -> (maker, lane flags, the row-loop modes its blocks must take)
DECKS = {
    "crossing1": (lambda: _razor(crossing_deck(1)), {}, {"line"}),
    "crossing1_default": (
        lambda: _razor_default(crossing_deck(1)),
        {},
        {"post"},
    ),
    "fan_rise": (lambda: _razor(fan_rise_deck()), {}, {"grouped", "line"}),
    "fan_rise_tiny": (
        lambda: _razor(fan_rise_deck()),
        {"cf._TILE_ROWS": 40},
        {"grouped", "line"},
    ),
    "wa7ark_tiny": (
        lambda: pg._wa7ark(True),
        {"cf._TILE_ROWS": 4_000},
        {"grouped", "line"},
    ),
    "detached_hub": (lambda: _razor(detached_hub()), {}, {"grouped", "line"}),
    "hub16_x2": (
        lambda: _razor(_hub(2)),
        {"cf._TILE_ROWS": 2_000},
        {"grouped", "line"},
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
    make, flags, modes = DECKS[case]
    ref, r0 = _fill(make, **{**flags, **_PHASE1})
    got, r = _fill(make, **flags)
    # The reference is phase 1: a store per product, the whole below block.
    assert r0["cf.fused_blocks"] == 0 and r0["rz.below_windows"] == 0, r0
    assert r0["cf.tile_stores"] == r0["cf.main_product"] >= 2, r0
    # The gated fill fused every product block and kept no store...
    n = r["cf.main_product"]
    assert n >= 2 and r["cf.fused_blocks"] == n and r["cf.tile_stores"] == 0, r
    assert r["cf.fused_declined"] == 0, r
    ran = {m for m in ("grouped", "line", "post") if r[f"cf.fused_mode_{m}"]}
    assert ran == modes, (ran, r)
    if modes - {"post"}:
        assert r["cf.fused_row_ends"] > 0, r
    if "grouped" in modes:
        assert r["cf.fused_unit_tiles"] >= 1, r
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
    make, flags, _modes = DECKS["fan_rise_tiny"]
    ref, _r = _fill(make, **{**flags, **_PHASE1})
    got, r = _fill(make, **flags)
    assert r["cf.fused_held_rows"] > 0, r
    assert _same(got, ref)


def test_a_slow_end_asking_a_product_row_declines_the_fusion(monkeypatch):
    """With no end recognised as fast, the ends that WERE product rows ask
    the product through the memo — which a store-less product cannot answer
    — so the block declines and keeps phase 1's store; Z is phase 1's."""
    make = DECKS["crossing1"][0]
    with monkeypatch.context() as mp:
        mp.setattr(cf, "_fast_end_desc", lambda *a, **k: None)
        ref, _r = _fill(make, **_PHASE1)
        got, r = _fill(make)
    assert r["cf.fused_declined_hit"] == r["cf.main_product"] >= 2, r
    assert r["cf.fused_blocks"] == 0 and r["cf.tile_stores"] >= 2, r
    assert _same(got, ref)


@pytest.mark.parametrize(
    "control, case, flag",
    [
        ("fused_order", "fan_rise_tiny", "cf._PRODUCT_NEG_CONTROL"),
        ("shift", "crossing1", "rz._BELOW_FOLD_CONTROL"),
    ],
)
def test_negative_controls_move_z(control, case, flag):
    make, flags, _modes = DECKS[case]
    ref, _r = _fill(make, **{**flags, **_PHASE1})
    got, r = _fill(make, **{**flags, flag: control})
    if control == "fused_order":
        assert r["cf.fused_blocks"] >= 2, r
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
