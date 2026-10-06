"""The crossing plan's integer bookkeeping with less transient memory —
momwire#1335.

At razor's inverted L x32 the reversed cross block's plan factorized a
31.5 M-row key table (`factorize_rows` over the raveled line and a broadcast
COPY of its z, a hash table sized at twice the rows, an int64 inverse and
then its int32 copy: ~1.1 GB beside the line), and `group_first_ranks` /
`merge_rows_by_z` grew their outputs in vectors and copied them out. Each is
a pure function of equality classes and walk order, so the gate is the
integers: `factorize_line_keys` against `factorize_rows` of the very columns
`_product_plan` used to hand it, and the two rewritten kernels against their
references, on adversarial floats (-0.0 with 0.0, NaN rows, repeats) and on
the plan of a real inverted-L fill, whose Z must not move either.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _accel
from momwire import _crossing_fill as cf
from momwire import _near_interface as ni

import test_product_grid_1173 as pg
from test_crossing_serve_524 import invl_deck
from test_triple_memo_1168 import _razor

acc = _accel.acc
pytestmark = pytest.mark.skipif(
    acc is None or not getattr(acc, "factorize_line_keys_1335", False),
    reason="accelerator without factorize_line_keys",
)


def _reference(line, lz):
    nG, nL = line.shape
    first, inv = acc.factorize_rows(
        [line.ravel(), np.broadcast_to(lz, line.shape).ravel()]
    )
    return np.asarray(first), np.asarray(inv).reshape(nG, nL)


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (7, 1), (13, 29), (64, 300)])
def test_line_keys_are_factorize_rows_of_the_broadcast_columns(shape):
    rng = np.random.default_rng(1335 + shape[0] * 1000 + shape[1])
    nG, nL = shape
    # Few distinct values, so keys repeat within and across rows.
    pool = np.array([0.0, -0.0, 1.5, 2.25, np.nan, 3.0, -1.0])
    line = rng.choice(pool, size=(nG, nL))
    lz = rng.choice(pool, size=nL)
    want_f, want_k = _reference(line, lz)
    got_f, got_k = acc.factorize_line_keys(line, lz)
    assert got_k.dtype == np.int32 and got_k.shape == (nG, nL)
    assert np.array_equal(np.asarray(got_f), want_f)
    assert np.array_equal(got_k.astype(np.int64), want_k)


def test_line_keys_rehash_keeps_the_numbering():
    """Many distinct keys force the table through several rehashes."""
    rng = np.random.default_rng(13351)
    line = rng.integers(0, 40_000, size=(37, 4_001)).astype(float) * 0.5
    lz = rng.integers(0, 3, size=4_001).astype(float)
    want_f, want_k = _reference(line, lz)
    got_f, got_k = acc.factorize_line_keys(line, lz)
    assert np.array_equal(np.asarray(got_f), want_f)
    assert np.array_equal(got_k.astype(np.int64), want_k)


def test_line_keys_refuse_a_wrong_lz():
    with pytest.raises(RuntimeError, match="one lz per line node"):
        acc.factorize_line_keys(np.zeros((2, 3)), np.zeros(4))


def test_group_first_ranks_is_first_ints_per_row():
    rng = np.random.default_rng(13352)
    kid = rng.integers(0, 50, size=(40, 300)).astype(np.int32)
    rank, n, first, ids = acc.group_first_ranks(kid, 50)
    off = 0
    for g in range(kid.shape[0]):
        f, r = cf._first_ints(kid[g].astype(np.int64))
        assert np.array_equal(rank[g], r)
        assert n[g] == f.size
        assert np.array_equal(first[off : off + f.size], f)
        assert np.array_equal(ids[off : off + f.size], kid[g, f])
        off += f.size
    assert off == first.size == ids.size


def test_the_inverted_l_plan_and_z_do_not_move():
    """The whole fill with each switch off is the gated route's Z to the
    bit, and the many-group plans took the new kernel."""
    make = lambda: _razor(invl_deck(n_radials=4))  # noqa: E731
    calls = []
    real = acc.factorize_line_keys

    def spy(*a, **k):
        calls.append(a[0].shape)
        return real(*a, **k)

    ref, _r0 = pg._fill(make, **{"cf._LINE_KEYS_ACCEL": False})
    with pytest.MonkeyPatch.context() as m:
        m.setattr(acc, "factorize_line_keys", spy, raising=False)
        got, r = pg._fill(make)
    assert len(calls) == r["cf.main_product"] >= 2, (calls, r)
    assert any(s[0] > 1 for s in calls), calls
    assert np.array_equal(pg._bits(got), pg._bits(ref))


def test_switch_off_is_the_hash_kernel_route():
    """`_LINE_KEYS_ACCEL = False` and `_FACTORIZE = False` both leave the
    plan's integers alone (the sorted spelling is the reference of both)."""
    make = lambda: _razor(invl_deck(n_radials=4))  # noqa: E731
    ref, _r = pg._fill(make, **{"cf._LINE_KEYS_ACCEL": False})
    with pytest.MonkeyPatch.context() as m:
        m.setattr(ni, "_FACTORIZE", False)
        got, _r = pg._fill(make)
    assert np.array_equal(pg._bits(got), pg._bits(ref))


@pytest.mark.parametrize("chunk", [1, 7, 1000, 1 << 20])
def test_stable_tile_order_is_the_stable_argsort(chunk, monkeypatch):
    """The tiles' row order by counting, a chunk at a time, against
    `np.argsort(kind="stable")` and its `searchsorted` bounds."""
    monkeypatch.setattr(cf, "_TILE_ORDER_CHUNK", chunk)
    rng = np.random.default_rng(13353)
    n_tiles = 37
    t_row = rng.integers(0, n_tiles, 20_011).astype(np.int16)
    t_row[t_row == 5] = 6  # an empty tile
    o, b = cf._stable_tile_order(t_row, n_tiles, np.int32)
    want = np.argsort(t_row, kind="stable")
    assert o.dtype == np.int32
    assert np.array_equal(o, want)
    assert np.array_equal(b, np.searchsorted(t_row[want], np.arange(n_tiles + 1)))
