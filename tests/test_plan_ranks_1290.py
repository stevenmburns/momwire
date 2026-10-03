"""A many-group product plan's bookkeeping in one pass (momwire#1290).

`group_first_ranks` (`_accel_factorize.cpp`) ranks every group's line keys
by first appearance -- `_first_ints(kid[g])` row by row -- with a stamp per
global key instead of a hash table per row, and a many-group product's
tiles take their rows from one radix sort of the rows by tile instead of a
gather from every group at every tile. Both are integers only, so the
gates are `==`: the same ranks, positions and ids against the per-row
reference, and Z to the bit with each switch off, on the inverted-L (the
deck with many groups), with the new paths counted so the gates are not
vacuous.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _accel
from momwire import _crossing_fill as cf
from momwire.razor import RazorSolver

from test_crossing_serve_524 import invl_deck
from test_factorize_1224 import _solve

pytestmark = pytest.mark.skipif(
    not cf._HAVE_GROUP_RANKS_ACCEL, reason="the accelerator lacks group_first_ranks"
)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_ranks_are_first_ints_per_row(seed):
    rng = np.random.default_rng(seed)
    n_id = int(rng.integers(5, 400))
    kid = rng.integers(0, n_id, (int(rng.integers(1, 30)), int(rng.integers(1, 90))))
    kid = kid.astype(np.int32)
    rank, n, first, ids = _accel.acc.group_first_ranks(kid, n_id)
    off = np.concatenate(([0], np.cumsum(n)))
    for g in range(kid.shape[0]):
        f, r = cf._first_ints(kid[g].astype(np.int64))
        assert np.array_equal(rank[g], r)
        assert n[g] == f.size
        assert np.array_equal(first[off[g] : off[g + 1]], f)
        assert np.array_equal(ids[off[g] : off[g + 1]], kid[g, f])


def test_the_merge_numbers_rows_as_the_hash_does():
    """`merge_rows_by_z` against `_first_ints` of the walk's codes, on
    random blocks that share z ids and keys across groups."""
    rng = np.random.default_rng(9)
    n_key, n_z, ng = 40, 6, 9
    kids = [
        rng.choice(n_key, int(rng.integers(1, 15)), replace=False).astype(np.int32)
        for _ in range(ng)
    ]
    off = np.concatenate(([0], np.cumsum([k.size for k in kids]))).astype(np.int64)
    grp = rng.integers(0, ng, 30)
    z = rng.integers(0, n_z, 30)
    lens = np.array([kids[g].size for g in grp])
    start = np.concatenate(([0], np.cumsum(lens)[:-1])).astype(np.int64)
    row, blk, j = _accel.acc.merge_rows_by_z(
        z, grp, start, np.concatenate(kids), off, n_key, n_z
    )
    codes = np.concatenate(
        [z[b] * n_key + kids[grp[b]].astype(np.int64) for b in range(30)]
    )
    first, inv = cf._first_ints(codes)
    assert np.array_equal(row, inv)
    assert np.array_equal(start[blk] + j, first)


def test_an_id_off_the_table_is_refused():
    with pytest.raises(RuntimeError, match="id out of range"):
        _accel.acc.group_first_ranks(np.array([[0, 3]], dtype=np.int32), 3)


@pytest.mark.parametrize("switch", ["_GROUP_RANKS", "_TILE_ROW_ORDER", "_MERGE_BY_Z"])
def test_razor_z_does_not_move(switch, monkeypatch):
    def make():
        return RazorSolver(**invl_deck(n_radials=4), nec5_quadrature=True)

    monkeypatch.setattr(cf, switch, False)
    z_ref, i_ref = _solve(make, ("_assemble_Z",))
    monkeypatch.setattr(cf, switch, True)
    seen = []
    if switch in ("_GROUP_RANKS", "_MERGE_BY_Z"):
        name = "group_first_ranks" if switch == "_GROUP_RANKS" else "merge_rows_by_z"
        real = getattr(_accel.acc, name)

        def counted(*a):
            seen.append(1)
            return real(*a)

        monkeypatch.setattr(_accel.acc, name, counted)
    else:
        real_rows = cf._ProductTiles._tile_rows

        def counted_rows(self, t):
            seen.append(self._by_tile is not None)
            return real_rows(self, t)

        monkeypatch.setattr(cf._ProductTiles, "_tile_rows", counted_rows)
    z_new, i_new = _solve(make, ("_assemble_Z",))
    assert any(seen), seen
    assert np.array_equal(z_new, z_ref)
    assert np.array_equal(i_new, i_ref)


def test_a_few_key_lookups_by_the_queries_are_the_key_table_s():
    """`KeyIndex`'s reverse lookup (the queries indexed, every key passed
    through them) answers what the key table answers, signed zeros, misses,
    repeats and all, and the table is built once the calls run out."""
    from momwire import _near_interface as ni

    rng = np.random.default_rng(1290)
    pairs = np.unique(
        np.stack(
            [rng.integers(1, 3000, 4000) * 0.01, rng.integers(-3, 1, 4000) * 0.5], 1
        ),
        axis=0,
    )
    key_r, key_zl = pairs[:, 0].copy(), pairs[:, 1].copy()
    key_zl[key_zl == 0] = -0.0
    pick = rng.integers(0, key_r.size, 60)
    q_r = np.concatenate([key_r[pick], key_r[:5] + 1e-3, key_r[pick[:7]]])
    q_zl = np.concatenate(
        [np.where(key_zl[pick] == 0, 0.0, key_zl[pick]), key_zl[:5], key_zl[pick[:7]]]
    )
    keys = ni.KeyIndex(key_r, key_zl)
    if keys._index is not ni._LAZY_CODES:
        pytest.skip("the hash index is not built")
    got = keys.ids(q_r, q_zl)
    assert keys._index is ni._LAZY_CODES, "the reverse path did not serve"
    full = ni.KeyIndex(key_r, key_zl)
    full._reverse_left = 0
    want = full.ids(q_r, q_zl)
    assert full._index is not ni._LAZY_CODES
    assert np.array_equal(got, want)
    assert (got[:60] == pick).all() and (got[60:65] == -1).all()
