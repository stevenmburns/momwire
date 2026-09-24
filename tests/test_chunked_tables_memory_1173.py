"""The chunked main sandwich's tables, held in less memory — momwire#1173.

At razor hub_deck(16) x16 the buried fill's process peak was the reversed
block's `_chunked_tables` pass-1 merge: the chunks' unique rows concatenated,
gathered into first-appearance order, then re-stacked by `_unique_rows` and
copied once more as its sorted array — four (M, 3) float copies of one list,
beside an int64 inverse per grid pair. The one evaluation after it went
through `designed_tables`, whose dedup of an already-distinct list is the
identity and whose scatter was a (6, m) copy of the block.

The rework moves integers and copies floats only (int32 indices, a scatter in
place of concatenate-then-gather, `_unique_tri` on the merged array,
`designed_rows` for the distinct list), so the gate is the bits: every chunk's
four kernel tables equal, at `np.array_equal`, those of ONE `_tables` call
over the whole grid, and that call's memo holds the same rows and values in
the same order. The spies prove the reworked path is the one that ran.

The negative control evaluates each chunk with its own `_tables` call on the
shared memo — the naive chunking `_chunked_tables` exists to avoid, which
regroups the column route's fresh triples — and the same comparison must
FAIL, which is what says it can see a regrouped evaluation. On crossing_deck(1)
it moves 1,723 to 2,144 entries of each kernel table per block (of 5,580 and
4,108), at most 1.1e-13 of that table's max.

Measured 2026-09-23 by script against origin/main 7d58486, Z to the bit
through `compute_impedance`: razor hub_deck(16) x2, x4 and x8 and
crossing_deck(1) and the detached pair, at the shipped budget and at a tiny
one (20 kB; 1 MiB at x8), twin and numpy lanes on crossing1 and x2; bspline
and SG hub x2 (which reach `_unique_tri` through `designed_tables` only).
Process peak (ru_maxrss, fresh process): x8 568 -> 495 MB, x16 1981 -> 1744 MB.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _crossing_fill as cf
from momwire import _near_interface as ni

from test_crossing_serve_524 import crossing_deck, hub_deck
from test_razor_detached_1149 import detached
from test_triple_memo_1168 import _razor

_TINY = 20_000


def _captured_grids(deck, monkeypatch):
    """The `_chunked_tables` argument tuples of one razor fill (forward and
    reversed block), taken through the real constructor at a tiny budget."""
    real = cf._chunked_tables
    calls = []

    def spy(*a):
        calls.append(a)
        yield from real(*a)

    with monkeypatch.context() as mp:
        mp.setattr(cf, "_MAIN_CHUNK_BYTES", _TINY)
        mp.setattr(cf, "_chunked_tables", spy)
        _razor(deck).compute_impedance()
    assert len(calls) == 2, len(calls)
    return calls


def _one_call(ctx, eps_t, k_p, rho, zA, zB):
    memo = ni.TripleMemo()
    z = np.broadcast_to(zA[:, None], rho.shape)
    zp = np.broadcast_to(zB[None, :], rho.shape)
    t = cf._tables(ctx, eps_t, k_p, rho, z, zp, cf._CROSS_RTOL, memo=memo)
    return {key: t[key] for key in cf._CROSS_KEYS}, memo


def _assembled(chunks, rho):
    out = {key: np.empty(rho.shape, dtype=np.complex128) for key in cf._CROSS_KEYS}
    for sl, K in chunks:
        for key in cf._CROSS_KEYS:
            out[key][:, sl] = K[key]
    return out


def _same_memo(a, b):
    return a.keys() == b.keys() and all(
        np.array_equal(x, y) for x, y in zip(a.values(), b.values())
    )


DECKS = {"crossing1": lambda: crossing_deck(1), "detached": detached}


class _Evaluations:
    """Counts the one-evaluation entry points while installed."""

    def __init__(self, mp):
        self.counts = {"rows": 0, "tables": 0}
        rows_real, tables_real = ni.designed_rows, ni.designed_tables

        def rows_spy(*a, **k):
            self.counts["rows"] += 1
            return rows_real(*a, **k)

        def tables_spy(*a, **k):
            self.counts["tables"] += 1
            return tables_real(*a, **k)

        mp.setattr(ni, "designed_rows", rows_spy)
        mp.setattr(ni, "designed_tables", tables_spy)


@pytest.mark.parametrize("case", sorted(DECKS))
def test_chunked_tables_are_the_one_call_bit_for_bit(case, monkeypatch):
    """Every chunk's tables and the memo against one whole-grid call, with
    the reworked path proved to have run: `designed_rows` once per grid and
    `designed_tables` never, and the stored inverses int32."""
    for ctx, eps_t, k_p, rho, zA, zB, cols, _memo in _captured_grids(
        DECKS[case](), monkeypatch
    ):
        assert len(cols) > 10, len(cols)
        ref, ref_memo = _one_call(ctx, eps_t, k_p, rho, zA, zB)
        memo = ni.TripleMemo()
        with monkeypatch.context() as mp:
            spies = _Evaluations(mp)
            gen = cf._chunked_tables(ctx, eps_t, k_p, rho, zA, zB, cols, memo)
            first = next(gen)
            dtypes = {inv.dtype for inv, _m in gen.gi_frame.f_locals["inverses"]}
            chunks = [first, *gen]
        assert spies.counts == {"rows": 1, "tables": 0}, spies.counts
        assert dtypes == {np.dtype(np.int32)}, dtypes
        assert [sl for sl, _K in chunks] == list(cols)
        got = _assembled(chunks, rho)
        for key in cf._CROSS_KEYS:
            assert np.array_equal(got[key], ref[key]), key
        assert _same_memo(memo, ref_memo)


def test_negative_control_per_chunk_calls_move_the_tables(monkeypatch):
    """The gate above can fail: one `_tables` call per chunk on the shared
    memo regroups the fresh triples of a column the chunk boundary cuts
    (crossing_deck(1) puts every pair at one ρ), and the tables move — at
    rounding, since the grouping changes a rule, not the quantity."""
    moved = 0
    for ctx, eps_t, k_p, rho, zA, zB, cols, _memo in _captured_grids(
        crossing_deck(1), monkeypatch
    ):
        ref, _ref_memo = _one_call(ctx, eps_t, k_p, rho, zA, zB)
        memo = ni.TripleMemo()
        naive = []
        for sl in cols:
            z = np.broadcast_to(zA[:, None], (rho.shape[0], sl.stop - sl.start))
            zp = np.broadcast_to(zB[None, sl], z.shape)
            t = cf._tables(ctx, eps_t, k_p, rho[:, sl], z, zp, cf._CROSS_RTOL, memo)
            naive.append((sl, t))
        bad = _assembled(naive, rho)
        for key in cf._CROSS_KEYS:
            moved += int(np.count_nonzero(bad[key] != ref[key]))
            scale = np.abs(ref[key]).max()
            assert np.abs(bad[key] - ref[key]).max() <= 1e-12 * scale, key
    assert moved > 0, "per-chunk grouping moved nothing: the gate is blind"


def test_designed_rows_is_designed_tables_on_distinct_rows():
    """`designed_rows` against `designed_tables` on a distinct list (with a
    −0.0 z′ that the dedup folds with nothing), memo hits included: the same
    block, column j being `KEYS[j]`, and the same memo afterwards."""
    rng = np.random.default_rng(1173)
    rho = rng.uniform(0.05, 3.0, 40)
    rows = np.stack([rho, rng.uniform(0.0, 2.0, 40), -rng.uniform(0.0, 2.0, 40)], 1)
    rows[3, 2] = -0.0
    rows[7:12, 0] = rows[7, 0]  # a shared ρ column
    args = (4.0 - 0.5j, 0.9)
    m_ref, m_new = ni.TripleMemo(), ni.TripleMemo()
    warm = rows[::3]
    ni.designed_tables(*args, warm[:, 0], warm[:, 1], warm[:, 2], memo=m_ref)
    ni.designed_rows(*args, warm, memo=m_new)
    ref = ni.designed_tables(*args, rows[:, 0], rows[:, 1], rows[:, 2], memo=m_ref)
    got = ni.designed_rows(*args, rows, memo=m_new)
    for j, key in enumerate(ni.KEYS):
        assert np.array_equal(got[:, j], ref[key]), key
    assert _same_memo(m_new, m_ref)
    with pytest.raises(TypeError, match="_designed_tables_reference"):
        ni.designed_rows(*args, rows, memo={})


def _unique_rows_before_1173(tri):
    """`_unique_rows`' body before momwire#1173, verbatim: the boundary
    compare on the sorted (n, 3) copy."""
    n = tri.shape[0]
    idx = np.lexsort((tri[:, 2], tri[:, 1], tri[:, 0]))
    srt = tri[idx]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    np.any(srt[1:] != srt[:-1], axis=1, out=new_group[1:])
    gid = np.cumsum(new_group) - 1
    first = idx[new_group]
    inverse = np.empty(n, dtype=np.intp)
    inverse[idx] = gid
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return tri[first[order]], rank[inverse]


def test_unique_tri_is_the_row_compare_on_awkward_values():
    """The per-column boundary compare against the whole-row one, on rows
    that repeat, differ in one column only, fold −0.0 with 0.0, and carry
    NaN (never equal, so every NaN row is its own group)."""
    rng = np.random.default_rng(904)
    base = rng.integers(0, 3, size=(500, 3)).astype(float)
    base[rng.random(500) < 0.1, 1] = -0.0
    base[rng.random(500) < 0.05, 2] = np.nan
    tri = np.ascontiguousarray(base)
    want_rows, want_inv = _unique_rows_before_1173(tri)
    got_rows, got_inv = ni._unique_tri(tri)
    assert np.array_equal(got_rows, want_rows, equal_nan=True)
    assert np.array_equal(np.signbit(got_rows), np.signbit(want_rows))
    assert got_inv.dtype == want_inv.dtype and np.array_equal(got_inv, want_inv)
    r, inv = ni._unique_tri(np.empty((0, 3)))
    assert r.shape == (0, 3) and inv.size == 0


@pytest.mark.slow
def test_chunked_hub16_x2_is_the_one_call_z(monkeypatch):
    """razor hub_deck(16) x2: Z at the shipped budget (one `_tables` call per
    block, `_chunked_tables` never entered) against the tiny budget (thousands
    of chunks through the reworked path), to the bit."""
    d = hub_deck(n_radials=16)
    npe = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [[m * 2 for m in e] for e in npe[:16]] + [
        npe[16],
        [npe[17][0] * 2],
    ]
    real = cf._chunked_tables
    served = []

    def spy(*a):
        for item in real(*a):
            served.append(item[0])
            yield item

    monkeypatch.setattr(cf, "_chunked_tables", spy)
    s = _razor(d)
    # Z read where it is filled: the solve factors it in place.
    whole = s._assemble_Z(s._build_geometry(), s.k)
    assert served == []
    monkeypatch.setattr(cf, "_MAIN_CHUNK_BYTES", _TINY)
    s = _razor(d)
    Z = s._assemble_Z(s._build_geometry(), s.k)
    assert len(served) > 1000, len(served)
    assert np.array_equal(Z, whole)
