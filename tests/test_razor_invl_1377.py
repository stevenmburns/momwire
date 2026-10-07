"""Razor's inverted L with less bookkeeping memory and time — momwire#1377.

Every change here is integer bookkeeping in the crossing fill's product
route: the equality classes, ranks and orders it computes are pure functions
of exact-`==` classes and walk order, so each lever is gated twice — its
kernel against the reference spelling on adversarial inputs (-0.0 with 0.0,
NaN, repeats), and the whole inverted-L fill with the lever off against it
on, Z compared on its uint64 view, with a spy proving the lever ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _accel
from momwire import _crossing_fill as cf

import test_product_grid_1173 as pg
from test_crossing_serve_524 import invl_deck
from test_triple_memo_1168 import _razor

acc = _accel.acc
needs_classes = pytest.mark.skipif(
    acc is None or not getattr(acc, "factorize_float_classes_1377", False),
    reason="accelerator without factorize_float_classes",
)


def _invl():
    return _razor(invl_deck(n_radials=4))


# ---------------------------------------------------------------- M1
@needs_classes
@pytest.mark.parametrize("n", [0, 1, 2, 7, 1000, 50_000])
def test_float_classes_are_factorize_rows(n):
    """The group count and inverse of `factorize_rows` on one column."""
    rng = np.random.default_rng(1377 + n)
    pool = np.array([0.0, -0.0, 1.5, 2.25, np.nan, 3.0, -1.0, np.inf, -np.inf])
    col = rng.choice(pool, size=n)
    if n >= 1000:  # many distinct values beside the repeats
        col[::3] = rng.integers(0, n, size=col[::3].size) * 0.25
    first, inv = acc.factorize_rows([col])
    n_got, inv_got = acc.factorize_float_classes(col)
    assert inv_got.dtype == np.int32
    assert n_got == np.asarray(first).size
    assert np.array_equal(inv_got.astype(np.int64), np.asarray(inv))


@needs_classes
def test_float_classes_all_distinct_fill_the_table():
    """Every row its own group: the table's highest load (3/4)."""
    col = np.arange(3 * (1 << 14), dtype=float)
    n_got, inv = acc.factorize_float_classes(col)
    assert n_got == col.size
    assert np.array_equal(inv, np.arange(col.size, dtype=np.int32))


@needs_classes
def test_lean_key_classes_do_not_move_the_inverted_l():
    """The tiles' exact-ρ classes by the lean kernel are the reference
    route's (`_LEAN_KEY_CLASSES = False`, `_near_interface._factorize`) to
    the bit, and the many-group products took the lean kernel."""
    calls = []
    real = acc.factorize_float_classes

    def spy(*a, **k):
        calls.append(a[0].size)
        return real(*a, **k)

    ref, _r = pg._fill(_invl, **{"cf._LEAN_KEY_CLASSES": False})
    with pytest.MonkeyPatch.context() as m:
        m.setattr(acc, "factorize_float_classes", spy, raising=False)
        got, r = pg._fill(_invl)
    assert len(calls) == r["cf.tile_column_products"] >= 1, (calls, r)
    assert np.array_equal(pg._bits(got), pg._bits(ref))


@pytest.mark.parametrize("span", [1, 5, 1 << 20])
def test_class_spans_do_not_move_the_inverted_l(span, monkeypatch):
    """The chunked class sums are exact in any grouping."""
    ref, _r = pg._fill(_invl)
    monkeypatch.setattr(cf, "_CLASS_SPAN", span)
    got, _r = pg._fill(_invl)
    assert np.array_equal(pg._bits(got), pg._bits(ref))


@needs_classes
def test_class_sums_are_bincount():
    rng = np.random.default_rng(13771)
    n = 5_000
    idx = rng.integers(0, n, size=40_000).astype(np.int32)
    w = rng.integers(0, 9_000, size=idx.size).astype(np.int32)
    got = acc.class_sums(idx, w, n)
    want = np.bincount(idx, weights=w, minlength=n)
    assert got.dtype == np.float64
    assert np.array_equal(got, want)
    with pytest.raises(RuntimeError, match="out of range"):
        acc.class_sums(np.array([n], dtype=np.int32), np.ones(1, dtype=np.int32), n)


# ---------------------------------------------------------------- tiles
needs_tiles = pytest.mark.skipif(
    acc is None or not getattr(acc, "product_tiles_1377", False),
    reason="accelerator without the product tile kernels",
)


@needs_tiles
def test_tile_kernels_do_not_move_the_inverted_l():
    """`product_rows` and `tile_block` against numpy's gathers and scatters
    (`_TILE_KERNELS = False`): Z to the bit, and the fused tiles took the
    kernel."""
    ref, r0 = pg._fill(_invl, **{"cf._TILE_KERNELS": False})
    got, r = pg._fill(_invl)
    assert r0["cf.tile_blocks_fast"] == 0
    assert (
        r["cf.tile_blocks_fast"] >= 1 and r["cf.tiles"] >= r["cf.tile_blocks_fast"]
    ), r
    assert np.array_equal(pg._bits(got), pg._bits(ref))


@needs_tiles
@pytest.mark.parametrize("kp_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("slot_z", [True, False])
def test_product_rows_are_the_numpy_gathers(kp_dtype, slot_z):
    rng = np.random.default_rng(13772)
    nA, nB, nG, n_key = 40, 30, 5, 200
    zA, zB = rng.normal(size=nA), rng.normal(size=nB)
    key_r = rng.random(n_key) + 0.5
    n_line = nB if slot_z else nA
    kid = rng.integers(0, n_key, size=(nG, n_line)).astype(np.int32)
    grank = rng.integers(0, nG, size=nA if slot_z else nB).astype(np.int64)
    kept_pos = rng.integers(0, nA * nB, size=500).astype(kp_dtype)
    ids = rng.integers(0, kept_pos.size, size=300).astype(np.int32)
    a, b = np.divmod(kept_pos[ids].astype(np.int64), nB)
    want = np.empty((ids.size, 3))
    want[:, 0] = key_r[kid[grank[a], b]] if slot_z else key_r[kid[grank[b], a]]
    want[:, 1], want[:, 2] = zA[a], zB[b]
    got = acc.product_rows(ids, kept_pos, nB, slot_z, grank, kid, key_r, zA, zB)
    assert np.array_equal(got.view(np.uint64), want.view(np.uint64))


def _u64(a):
    return np.ascontiguousarray(a).view(np.uint64)


@needs_tiles
@pytest.mark.parametrize("with_pos", [False, True])
def test_tile_block_is_the_numpy_copies(with_pos):
    rng = np.random.default_rng(13773)
    n_rows, m, n_held = 1000, 120, 50
    vals = rng.normal(size=(m, 6)) + 1j * rng.normal(size=(m, 6))
    pos = rng.permutation(m) if with_pos else None
    ids = np.sort(rng.choice(n_rows, size=m, replace=False)).astype(np.int32)
    hpos = np.full(n_rows, -1, dtype=np.int32)
    hpos[ids[::3]] = rng.permutation(n_held)[: ids[::3].size]
    kcols = [0, 1, 2, 5]
    done = np.zeros(n_rows, dtype=bool)
    loc = np.full(n_rows, -1, dtype=np.int32)
    held = np.zeros((n_held, 4), dtype=np.complex128, order="F")
    v = vals if pos is None else vals[pos]
    want_tb = v[:, kcols]
    want_held = held.copy(order="F")
    keep = hpos[ids] >= 0
    want_held[hpos[ids][keep]] = want_tb[keep]
    tb, twice = acc.tile_block(
        vals,
        np.zeros(0, np.int64) if pos is None else pos,
        kcols,
        ids,
        done,
        loc,
        hpos,
        held,
    )
    assert not twice and tb.flags.f_contiguous
    assert np.array_equal(_u64(tb), _u64(want_tb))
    assert np.array_equal(_u64(held), _u64(want_held))
    assert done[ids].all() and done.sum() == m
    assert np.array_equal(loc[ids], np.arange(m))
    _tb, twice = acc.tile_block(
        vals, np.zeros(0, np.int64), kcols, ids, done, loc, hpos, held
    )
    assert twice and _tb is None
    acc.fill_rows_i32(ids, loc, -1)
    assert (loc == -1).all()
