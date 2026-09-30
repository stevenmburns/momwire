"""Bit-identity gates for the point rows' memory cut -- momwire#1267.

`_chunked_point_tables` evaluates the point grid's unique rows in batches of
whole ρ_eff columns (`_near_interface.column_batches`, `_POINT_EVAL_ROWS`)
instead of one `designed_rows` and one `point_designed_rows` call, and keeps
in the block's memo only the rows its ends loop can ask (`designed_rows`'
`keep`). The sinusoidal mixed assembly writes the class blocks and the
crossing rows straight into its Φ and G. None of it may move a bit.

Gates:
  * on real point grids (the bent crossing deck and the buried inverted-L,
    both directions, sheets planned as the block plans them), the batched
    evaluation equals the one call byte for byte, six and point values, at
    budgets that force many batches -- counted, so the gate cannot pass on
    one batch;
  * the grids are ones a naive row cut splits (a ρ_eff column in two
    batches), which is the cut the batches must not take: swapped in for
    `column_batches`, a naive cut fails these same equalities (checked by
    hand, momwire#1267; the #1168 audit's naive batching moved Z the same
    way);
  * `keep`: the memo retains exactly the kept fresh rows, with the one
    call's values, and the block's own values do not depend on it;
  * the whole block, chunked at small evaluation budgets, equals the dense
    reference route bit for bit;
  * `_crossing_point_rows(into=G)` is `G + rows` and the mixed assembly's
    in-place class blocks are the returned blocks scattered, to the bit.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_crossing_serve_524 import invl_deck
from test_point_rows_chunked_1224 import _blocks, _run

from momwire import _crossing_fill as cf
from momwire import _near_interface as ni
from momwire.sinusoidal import SinusoidalSolver


def _bytes_equal(a, b):
    """Byte equality of two complex arrays (NaN payloads included)."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def _invl_blocks():
    """`point_observer_block`'s two directions' arguments on the 8-radial
    buried inverted-L (the deck #1267 measures, at x1)."""
    s = SinusoidalSolver(**invl_deck(n_radials=8, x=1))
    geom = s._build_geometry()
    below = np.asarray(geom["seg_centers"])[:, 2] < s.ground_z
    med = cf.buried_medium(s.ground_eps, s.omega, s.eps, s.k)
    view = s._basis_coefs(geom, np.where(below, med.k_m, med.k_p))
    ctx = s._crossing_context(geom, view, med)
    a_idx, b_idx = np.nonzero(~below)[0], np.nonzero(below)[0]
    c = np.asarray(geom["seg_centers"])
    t = np.asarray(geom["seg_tangents"])
    ax_a, ax_b = cf.axis_data(ctx, a_idx), cf.axis_data(ctx, b_idx)
    return [
        (ctx, c[a_idx], t[a_idx], ax_b, True),
        (ctx, c[b_idx], t[b_idx], ax_a, False),
    ]


@pytest.fixture(scope="module")
def grids():
    """(name, eps_t, k_p, plan, unique folded rows) of each real point grid:
    the rows `_chunked_point_tables` evaluates (the grid's distinct (ρ_eff,
    z, z′) in first-appearance order) and the sheet plan its block makes."""
    out = []
    for name, blocks in (("bent", _blocks("bent", 9)), ("invl", _invl_blocks())):
        for ctx, obs, _t, src, above in blocks:
            eps_t, k_p, gz, _c1, memo = cf._block_preamble(ctx)
            cf._plan_point_sheets(ctx, eps_t, k_p, gz, memo, obs, src["nodes"], above)
            P = np.asarray(obs, dtype=float)
            nodes = np.asarray(src["nodes"], dtype=float)
            _dx, _dy, rho, z, zp = cf._point_pair_grid(P, nodes, gz, above)
            rows, _inv = ni._unique_rows(ni.radius_fold(rho, ctx.a_wire), z, zp)
            tag = f"{name}-{'above' if above else 'below'}"
            out.append((tag, eps_t, k_p, memo.sheet_plan, rows))
    return out


def _one_call(eps_t, k_p, rows, plan):
    memo = ni.ProductMemo()
    memo.sheet_plan = plan
    six = ni.designed_rows(eps_t, k_p, rows, rtol=cf._CROSS_RTOL, memo=memo)
    pt = ni.point_designed_rows(eps_t, k_p, rows, plan=plan)
    return six, pt, memo


def _batched(eps_t, k_p, rows, plan, max_rows, keep=None):
    memo = ni.ProductMemo()
    memo.sheet_plan = plan
    six = np.empty((rows.shape[0], ni.N_KEYS), dtype=np.complex128)
    pt = {k: np.empty(rows.shape[0], dtype=np.complex128) for k in ni.POINT_KEYS}
    batches = ni.column_batches(rows[:, 0], max_rows)
    for sel in batches:
        six[sel] = ni.designed_rows(
            eps_t, k_p, rows[sel], rtol=cf._CROSS_RTOL, memo=memo, keep=keep
        )
        for k, v in ni.point_designed_rows(eps_t, k_p, rows[sel], plan=plan).items():
            pt[k][sel] = v
    return six, pt, memo, len(batches)


def _naive_splits_a_column(rho, max_rows):
    """True when cutting the list every `max_rows` rows puts one ρ value in
    two batches."""
    ids = np.arange(rho.size) // max_rows
    _u, inv = np.unique(rho, return_inverse=True)
    lo = np.full(_u.size, np.iinfo(np.intp).max)
    hi = np.full(_u.size, -1)
    np.minimum.at(lo, inv, ids)
    np.maximum.at(hi, inv, ids)
    return bool(np.any(lo != hi))


@pytest.mark.parametrize("max_rows", [7, 64])
def test_batched_evaluation_is_the_one_call_bit_for_bit(grids, max_rows):
    for tag, eps_t, k_p, plan, rows in grids:
        six1, pt1, _m = _one_call(eps_t, k_p, rows, plan)
        six_b, pt_b, _m, n_batches = _batched(eps_t, k_p, rows, plan, max_rows)
        assert n_batches > 1, tag  # the cut was exercised
        assert _naive_splits_a_column(rows[:, 0], max_rows), tag  # non-vacuous
        assert _bytes_equal(six1, six_b), tag
        for k in ni.POINT_KEYS:
            assert _bytes_equal(pt1[k], pt_b[k]), (tag, k)


def test_column_batches_partition_whole_columns():
    rng = np.random.default_rng(1267)
    rho = rng.choice(np.array([0.5, 1.0, 1.5, 2.0, 2.5, -0.0, 0.0]), size=500)
    for max_rows in (1, 3, 50, 499, 500, 10_000):
        batches = ni.column_batches(rho, max_rows)
        flat = np.concatenate(batches)
        assert np.array_equal(np.sort(flat), np.arange(rho.size))
        for sel in batches:
            assert np.all(np.diff(sel) > 0)  # ascending: first-seen order kept
        owner = {}
        for b, sel in enumerate(batches):
            for r in np.unique(rho[sel]):
                assert owner.setdefault(float(r), b) == b  # -0.0 == 0.0: one column
        _u, counts = np.unique(rho, return_counts=True)
        assert max(s.size for s in batches) <= max(max_rows, 1) + counts.max() - 1
    assert ni.column_batches(np.empty(0), 4) == []


def test_keep_retains_exactly_the_kept_rows(grids):
    """`keep` changes only what the memo retains: the block is the same
    bits, the memo holds exactly the kept rows, with the one call's values."""
    tag, eps_t, k_p, plan, rows = grids[-1]
    six1, _pt1, memo1 = _one_call(eps_t, k_p, rows, plan)
    kept = rows[:: max(1, rows.shape[0] // 37)]
    keep = ni.TripleMemo.key_set(kept)
    six_b, _pt_b, memo_b, n_batches = _batched(eps_t, k_p, rows, plan, 64, keep=keep)
    assert n_batches > 1, tag
    assert _bytes_equal(six1, six_b)
    assert len(memo_b) == kept.shape[0]
    assert memo_b.contains(kept).all()
    held = memo_b.contains(rows)
    assert np.array_equal(held, keep.contains(rows))
    hit_b, val_b = memo_b.lookup(kept)
    hit_1, val_1 = memo1.lookup(kept)
    assert hit_b.all() and hit_1.all()
    assert _bytes_equal(val_b, val_1)


def test_contains_is_lookups_hit():
    rows = np.array([[1.0, 2.0, -1.0], [0.5, 0.0, -0.0], [np.nan, 1.0, -1.0]])
    memo = ni.TripleMemo.key_set(rows)
    ask = np.array(
        [[1.0, 2.0, -1.0], [0.5, -0.0, 0.0], [np.nan, 1.0, -1.0], [1.0, 2.0, -2.0]]
    )
    hit, _block = memo.lookup(ask)
    assert np.array_equal(memo.contains(ask), hit)
    assert hit.tolist() == [True, True, False, False]
    assert not ni.TripleMemo().contains(ask).any()


@pytest.fixture
def _restore_point_state():
    saved = (cf._POINT_CHUNKED, cf._POINT_CHUNK_PAIRS, cf._POINT_EVAL_ROWS)
    yield
    cf._POINT_CHUNKED, cf._POINT_CHUNK_PAIRS, cf._POINT_EVAL_ROWS = saved


@pytest.mark.usefixtures("_restore_point_state")
@pytest.mark.parametrize("eval_rows", [1, 17])
def test_block_at_small_eval_batches_equals_dense(eval_rows):
    # The bent deck: the rod's grids are one ρ_eff column per direction,
    # which no budget can cut.
    args_list = _blocks("bent", 9)
    cf._POINT_CHUNKED = False
    dense = _run(args_list)
    cf._POINT_CHUNKED = True
    cf._POINT_EVAL_ROWS = eval_rows
    before = cf._ROUTES["point_eval_batches"]
    chunked = _run(args_list)
    assert cf._ROUTES["point_eval_batches"] - before > len(args_list)
    for d, c in zip(dense, chunked):
        assert _bytes_equal(d, c)


def test_crossing_rows_into_is_the_sum():
    s = SinusoidalSolver(**invl_deck(n_radials=8, x=1))
    geom = s._build_geometry()
    below = np.asarray(geom["seg_centers"])[:, 2] < s.ground_z
    med = cf.buried_medium(s.ground_eps, s.omega, s.eps, s.k)
    view = s._basis_coefs(geom, np.where(below, med.k_m, med.k_p))
    rows = s._crossing_point_rows(geom, view, med, below)
    n = rows.shape[0]
    G = np.random.default_rng(7).standard_normal((n, n)) * (1 + 1j)
    want = G + rows
    got = s._crossing_point_rows(geom, view, med, below, into=G)
    assert got is G
    assert _bytes_equal(want, G)


def test_class_blocks_in_place_are_the_scattered_blocks(monkeypatch):
    """The mixed assembly's G with `_class_block(into=Φ)` against the
    returned-block route scattered as the assembly used to scatter it."""
    deck = invl_deck(n_radials=8, x=1)
    s = SinusoidalSolver(**deck)
    geom = s._build_geometry()
    G_new, _view = s._assemble_Z_mixed(geom, s._fill_eta(s.k, None))

    orig = SinusoidalSolver._class_block

    def returned(self, geom, keep, idx, *a, into=None):
        block = orig(self, geom, keep, idx, *a)
        for P, b in zip(into, block):
            P[np.ix_(idx, idx)] = b

    monkeypatch.setattr(SinusoidalSolver, "_class_block", returned)
    s2 = SinusoidalSolver(**deck)
    G_old, _view = s2._assemble_Z_mixed(s2._build_geometry(), s2._fill_eta(s2.k, None))
    assert _bytes_equal(G_new, G_old)
