"""The crossing fill's dedup bookkeeping without sorts it did not need —
momwire#1224.

Each helper here is pinned to the spelling it replaced, kept inline as the
reference, on inputs chosen to reach the awkward cases (signed zeros, rows
differing in one column, repeated keys), with a red control that a wrong
order or numbering cannot pass:

* `_first_appearance` — the groups numbered by their first index through a
  mask and its running count, against `argsort(first, kind="stable")`;
  `_unique_tri`, `_first_groups` and `_first_ints` all go through it.
* `_ProductTiles._tile_rows` — a several-group tile's distinct rows from a
  row mask, against `np.unique(np.concatenate(parts))`.
* `KeyIndex` — each key's class numbers from `np.unique(...,
  return_inverse=True)`, against the searched construction; `ids` must
  answer the same global key ids on hits and misses.

The integrated bits (Z through the real fill) are
`test_product_fast_ends_1224` and `test_product_grid_1173`.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _crossing_fill as cf
from momwire import _near_interface as ni


def _by_argsort(first):
    """The pre-#1224 spelling."""
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return rank, first[order]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_first_appearance_is_the_argsort(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 5000))
    first = rng.permutation(n)[: int(rng.integers(1, n + 1))].astype(np.intp)
    rank, srt = ni._first_appearance(first, n)
    want_rank, want_srt = _by_argsort(first)
    assert rank.dtype == want_rank.dtype and np.array_equal(rank, want_rank)
    assert srt.dtype == want_srt.dtype and np.array_equal(srt, want_srt)
    # Red control: a numbering by descending first index is not it.
    assert first.size == 1 or not np.array_equal(rank, first.size - 1 - want_rank)


def _unique_tri_before_1224(tri):
    n = tri.shape[0]
    idx = np.lexsort((tri[:, 2], tri[:, 1], tri[:, 0]))
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = np.any(tri[idx][1:] != tri[idx][:-1], axis=1)
    gid = np.cumsum(new) - 1
    first = idx[new]
    inv = np.empty(n, dtype=np.intp)
    inv[idx] = gid
    rank, srt = _by_argsort(first)
    return tri[srt], rank[inv]


def test_unique_tri_first_groups_first_ints_are_the_argsort_route():
    rng = np.random.default_rng(1224)
    tri = rng.integers(0, 4, size=(3000, 3)).astype(float) * 0.25
    tri[rng.random(3000) < 0.2, 1] = -0.0
    tri[rng.random(3000) < 0.02, 2] = np.nan
    tri = np.ascontiguousarray(tri)
    rows, inv = ni._unique_tri(tri)
    want_rows, want_inv = _unique_tri_before_1224(tri)
    assert np.array_equal(rows, want_rows, equal_nan=True)
    assert np.array_equal(np.signbit(rows), np.signbit(want_rows))
    assert inv.dtype == want_inv.dtype and np.array_equal(inv, want_inv)
    # `_first_groups` on two columns: the same first-appearance numbering.
    f, r = cf._first_groups(tri[:, 0], tri[:, 1])
    w_rows, w_inv = _unique_tri_before_1224(
        np.ascontiguousarray(np.stack([tri[:, 0], tri[:, 1], np.zeros(3000)], 1))
    )
    assert np.array_equal(r, w_inv) and np.array_equal(tri[f, :2], w_rows[:, :2])
    ids = rng.integers(0, 50, size=2000)
    f, r = cf._first_ints(ids)
    _u, first, inv0 = np.unique(ids, return_index=True, return_inverse=True)
    rank, srt = _by_argsort(first)
    assert np.array_equal(f, srt) and np.array_equal(r, rank[inv0.ravel()])


class _Plan:
    def __init__(self, rowtab, n_rows):
        self.rowtab, self.n_rows = rowtab, n_rows


def test_tile_rows_mask_is_np_unique():
    """Three groups whose row tables share rows, cut into tiles by key; each
    tile's ids against `np.unique` of the same parts, and the mask is left
    clear between tiles."""
    rng = np.random.default_rng(7)
    n_rows = 500
    rowtab = [rng.integers(0, n_rows, size=(4, 30)) for _ in range(3)]
    tiles = cf._ProductTiles.__new__(cf._ProductTiles)
    tiles.plan = _Plan(rowtab, n_rows)
    tiles._mark = None
    tiles._gkeys = []
    n_tiles = 3
    for tab in rowtab:
        o = rng.permutation(tab.shape[1])
        tiles._gkeys.append((o, np.array([0, 10, 20, 30])))
    for t in range(n_tiles):
        parts = [
            tab[:, o[b[t] : b[t + 1]]].ravel()
            for tab, (o, b) in zip(rowtab, tiles._gkeys)
        ]
        got = tiles._tile_rows(t)
        want = np.unique(np.concatenate(parts))
        assert got.dtype == want.dtype and np.array_equal(got, want)
        assert not tiles._mark.any()


def _key_ids_searched(key_r, key_zl, r, zl):
    """`ProductSet.value_rows`' key step before momwire#1224: class numbers
    searched back from the unique arrays."""
    r_ids, zl_ids = ni._SortedIds(np.unique(key_r)), ni._SortedIds(np.unique(key_zl))
    n_zl = np.unique(key_zl).size
    codes = ni._SortedCodes(
        r_ids.ids(key_r).astype(np.int64) * n_zl + zl_ids.ids(key_zl)
    )
    ri, li = r_ids.ids(r), zl_ids.ids(zl)
    ok = (ri >= 0) & (li >= 0)
    out = np.full(r.shape, -1, dtype=np.intp)
    out[ok] = codes.ids(ri[ok].astype(np.int64) * n_zl + li[ok])
    return out


def test_key_index_answers_the_searched_key_ids():
    rng = np.random.default_rng(11)
    r = rng.integers(1, 30, size=400) * 0.01 + 0.001
    zl = rng.integers(-5, 1, size=400) * 0.1
    zl[zl == 0] = np.where(rng.random(int((zl == 0).sum())) < 0.5, -0.0, 0.0)
    pairs = np.unique(np.stack([r, zl + 0.0], 1), axis=0)
    key_r, key_zl = pairs[:, 0].copy(), pairs[:, 1].copy()
    flip = (key_zl == 0) & (rng.random(key_zl.size) < 0.5)
    key_zl[flip] = -0.0  # stored keys carry either signed zero
    keys = ni.KeyIndex(key_r, key_zl)
    q_r = np.concatenate([key_r, key_r + 1e-3, r])
    q_zl = np.concatenate([np.where(key_zl == 0, 0.0, key_zl), key_zl, zl])
    got = keys.ids(q_r, q_zl)
    want = _key_ids_searched(key_r, key_zl, q_r, q_zl)
    assert np.array_equal(got, want)
    assert (got[: key_r.size] == np.arange(key_r.size)).all()
    assert (got[key_r.size : 2 * key_r.size] == -1).all()
    # The tiles' exact-rho classes are np.unique's.
    u, inv = np.unique(key_r, return_inverse=True)
    for _ in range(2):  # handed over once, formed again after
        got_u, got_inv = keys.take_r_classes()
        assert np.array_equal(got_u, u) and np.array_equal(got_inv, inv.ravel())


class _Ctx:
    a_wire = 0.001


@pytest.mark.parametrize("seed", [0, 1])
def test_a_raw_row_formed_again_is_the_plans_row(seed):
    """Past `_PRODUCT_FAST_RAW_GROUPS` groups the fast ends re-form a group's
    raw ρ row when an end asks (`_raw_row`); it must be the plan's row to the
    bit, on both slots (the grouped side above, and below). Red control: the
    row of the next group is not it."""
    rng = np.random.default_rng(seed)
    xy_a = np.round(rng.random((40, 2)) * 7, 3)[rng.integers(0, 40, 60)]
    xy_b = rng.random((90, 2)) * 5 - 2
    A = {"nodes": np.column_stack([xy_a, rng.random(60) * 3])}
    B = {"nodes": np.column_stack([xy_b, -rng.random(90)])}
    for a, b in ((A, B), (B, A)):
        aa = {"nodes": a["nodes"].copy()}
        bb = {"nodes": b["nodes"].copy()}
        if a is B:  # the repeats on the BELOW side: it is the grouped one
            aa["nodes"][:, 2] *= -1
            bb["nodes"][:, 2] *= -1
        plan = cf._product_plan(_Ctx(), 1.0, 1.0, aa, bb, 0.0)
        fast = plan.fast
        assert fast.grouped_slot == ("z" if a is A else "zp")
        assert fast.raw is not None and fast.raw.shape[0] > 1
        bare = fast._replace(raw=None)
        for g in range(fast.raw.shape[0]):
            assert np.array_equal(cf._raw_row(bare, g), fast.raw[g])
        assert not np.array_equal(cf._raw_row(bare, 1), fast.raw[0])
