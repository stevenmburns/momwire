"""Bit-identity gate for the chunked point-observer block — momwire#1224
stage 3 units 3 and 4.

`_chunked_point_tables` replaces `point_observer_block`'s dense whole-grid
tables with one dedup, a designed-tables/point-family evaluation, and a
gather + contraction per OBSERVER-ROW chunk (`_POINT_CHUNKED = True`, the
default). This must be BIT-IDENTICAL to `_point_observer_block_dense`
(`_POINT_CHUNKED = False`), the stage-2 route kept as the reference, at
EVERY chunk size — including one observer per chunk, the narrowest cut
`_chunked_point_tables`'s pass-1/pass-2 merge can take.

Unit 3 gates (the GATHER):
  * the chunked route equals the dense reference bit-for-bit
    (`np.array_equal` on the raw complex block) at several forced
    `_POINT_CHUNK_PAIRS` budgets, one of them narrow enough to force one
    observer per chunk;
  * `_crossing_fill._ROUTES["point_chunked"]` / `["point_chunks"]` prove the
    chunked route actually ran, and how many chunks it took, so the test
    cannot pass with `_POINT_CHUNKED` silently False;
  * a TEST-ONLY negative control (`_POINT_NEG_CONTROL`, an off-by-one
    observer row in the gather) is shown to FAIL that same equality, so the
    gate is not vacuously green (`_PRODUCT_NEG_CONTROL`'s pattern).

Unit 4 gates (the EVALUATION, below): `_chunked_point_tables`'s merged
unique-triple list is itself evaluated through `designed_rows` /
`point_designed_rows` in calls of WHOLE ρ_eff columns
(`_crossing_fill._column_chunked_point_eval`, `_POINT_EVAL_CHUNKED = True`
the default) instead of one call over the whole list
(`_POINT_EVAL_CHUNKED = False`, kept as this gate's reference) — the fill's
last measured memory transient after unit 3. Same shape of gates:
bit-identity at several `_POINT_EVAL_CHUNK_ROWS` budgets including one row
per chunk (forcing one ρ_eff column per call, the narrowest cut), a
`_ROUTES["point_eval_chunked"]` / `["point_eval_chunks"]` proof the route
ran, and a TEST-ONLY negative control (`_POINT_EVAL_NEG_CONTROL`, which
deliberately splits one ρ_eff column's members across two evaluation calls)
shown to FAIL the same equality.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_point_observer_block_1223 import _crossing
from test_sin_crossing_recipe_1223 import DECKS, _eps_tilde

from momwire import _crossing_fill as cf

SOIL = (13.0, 0.005)


def _blocks(deck, n):
    """The two directions' `point_observer_block` arguments for `deck`, at
    axis density `n` — everything geometry-dependent, shared by the dense
    and chunked calls below (neither route touches `axis_data`)."""
    geom, below, ctx = _crossing(deck, _eps_tilde(*SOIL), n)
    a_idx, b_idx = np.nonzero(~below)[0], np.nonzero(below)[0]
    centres = np.asarray(geom["seg_centers"])
    tangents = np.asarray(geom["seg_tangents"])
    ax_a = cf.axis_data(ctx, a_idx)
    ax_b = cf.axis_data(ctx, b_idx)
    return [
        (ctx, centres[a_idx], tangents[a_idx], ax_b, True),
        (ctx, centres[b_idx], tangents[b_idx], ax_a, False),
    ]


def _run(args_list):
    return [
        cf.point_observer_block(a[0], a[1], a[2], a[3], observers_above=a[4])
        for a in args_list
    ]


def _run_chunked(args_list, pairs):
    cf._POINT_CHUNKED = True
    cf._POINT_CHUNK_PAIRS = pairs
    before_calls = cf._ROUTES["point_chunked"]
    before_chunks = cf._ROUTES["point_chunks"]
    out = _run(args_list)
    calls = cf._ROUTES["point_chunked"] - before_calls
    chunks = cf._ROUTES["point_chunks"] - before_chunks
    return out, calls, chunks


@pytest.fixture(autouse=True)
def _restore_module_state():
    chunked0 = cf._POINT_CHUNKED
    pairs0 = cf._POINT_CHUNK_PAIRS
    neg0 = cf._POINT_NEG_CONTROL
    eval_chunked0 = cf._POINT_EVAL_CHUNKED
    eval_rows0 = cf._POINT_EVAL_CHUNK_ROWS
    eval_neg0 = cf._POINT_EVAL_NEG_CONTROL
    yield
    cf._POINT_CHUNKED = chunked0
    cf._POINT_CHUNK_PAIRS = pairs0
    cf._POINT_NEG_CONTROL = neg0
    cf._POINT_EVAL_CHUNKED = eval_chunked0
    cf._POINT_EVAL_CHUNK_ROWS = eval_rows0
    cf._POINT_EVAL_NEG_CONTROL = eval_neg0


@pytest.mark.parametrize("pairs", [1, 5, 1 << 18])
@pytest.mark.parametrize("deck", sorted(DECKS))
def test_chunked_equals_dense_bit_for_bit(deck, pairs):
    args_list = _blocks(deck, 9)

    cf._POINT_CHUNKED = False
    dense = _run(args_list)

    chunked, calls, chunks = _run_chunked(args_list, pairs)

    assert calls == len(args_list)  # the chunked route ran, once per direction
    assert chunks >= len(args_list)  # and took at least one chunk each
    for d, c in zip(dense, chunked):
        assert np.array_equal(d, c)


def test_a_pair_budget_of_one_forces_one_observer_per_chunk():
    """`_POINT_CHUNK_PAIRS = 1` makes `step = max(1, 1 // nB) == 1` on every
    direction (`_point_observer_block_chunked`), so `_chunked_point_tables`
    is walked one observer row at a time — the narrowest cut its pass-1/
    pass-2 merge can take, and still bit-identical."""
    args_list = _blocks("rod", 9)
    n_obs_total = sum(a[1].shape[0] for a in args_list)

    cf._POINT_CHUNKED = False
    dense = _run(args_list)

    chunked, calls, chunks = _run_chunked(args_list, 1)

    assert calls == len(args_list)
    assert chunks == n_obs_total
    for d, c in zip(dense, chunked):
        assert np.array_equal(d, c)


def test_negative_control_fails_the_gate():
    """`_POINT_NEG_CONTROL` breaks the chunked gather by one observer row on
    purpose (`_chunked_point_tables`'s `np.roll`), so the same equality this
    module gates on must FAIL — proof the gate can fail, not just pass."""
    args_list = _blocks("rod", 9)

    cf._POINT_CHUNKED = False
    dense = _run(args_list)

    cf._POINT_CHUNKED = True
    # A wide-enough budget that a direction's whole axis is one chunk (many
    # rows), so `np.roll(idx, 1, axis=0)` actually moves an observer's row
    # to a DIFFERENT observer instead of rolling a one-row chunk onto itself.
    cf._POINT_CHUNK_PAIRS = 1 << 18
    cf._POINT_NEG_CONTROL = True
    broken = _run(args_list)

    assert any(not np.array_equal(d, c) for d, c in zip(dense, broken))


# --- momwire#1224 stage 3 unit 4: the EVALUATION route ---------------------


def _run_eval_chunked(args_list, rows):
    cf._POINT_EVAL_CHUNKED = True
    cf._POINT_EVAL_CHUNK_ROWS = rows
    before_calls = cf._ROUTES["point_eval_chunked"]
    before_chunks = cf._ROUTES["point_eval_chunks"]
    out = _run(args_list)
    calls = cf._ROUTES["point_eval_chunked"] - before_calls
    chunks = cf._ROUTES["point_eval_chunks"] - before_chunks
    return out, calls, chunks


@pytest.mark.parametrize("rows", [1, 3, 50, 1 << 17])
@pytest.mark.parametrize("deck", sorted(DECKS))
def test_eval_chunked_equals_reference_bit_for_bit(deck, rows):
    """`_column_chunked_point_eval` (default, `_POINT_EVAL_CHUNKED = True`)
    against the ONE-SHOT `designed_rows` / `point_designed_rows` call
    (`_POINT_EVAL_CHUNKED = False`, this gate's reference), at several
    column-chunk row budgets including one row — the narrowest cut
    `_column_ranges` can take (one ρ_eff column per call)."""
    args_list = _blocks(deck, 9)

    cf._POINT_EVAL_CHUNKED = False
    reference = _run(args_list)

    chunked, calls, chunks = _run_eval_chunked(args_list, rows)

    assert calls == len(args_list)  # the column-chunked route ran per direction
    assert chunks >= len(args_list)  # and took at least one chunk each
    for r, c in zip(reference, chunked):
        assert np.array_equal(r, c)


def test_a_row_budget_of_one_forces_one_column_per_chunk():
    """`_POINT_EVAL_CHUNK_ROWS = 1` forces every chunk to hold exactly one
    ρ_eff column (`_column_ranges`: any column after the first always pushes
    `chunk_rows` past a budget of 1), so the number of evaluation chunks a
    direction takes equals its number of DISTINCT ρ_eff values among the
    merged unique triples — never the number of triples, when any column
    holds more than one member. Still bit-identical."""
    args_list = _blocks("rod", 9)

    cf._POINT_EVAL_CHUNKED = False
    reference = _run(args_list)

    chunked, calls, chunks = _run_eval_chunked(args_list, 1)

    assert calls == len(args_list)
    assert chunks >= len(args_list)
    for r, c in zip(reference, chunked):
        assert np.array_equal(r, c)


def test_negative_control_eval_split_fails_the_gate():
    """`_POINT_EVAL_NEG_CONTROL` deliberately splits one ρ_eff column's
    members across two `designed_rows` / `point_designed_rows` calls instead
    of one (`_column_chunked_point_eval`'s tail-column split), so the same
    equality this module gates on must FAIL — proof the gate can fail, not
    just pass, on BOTH fixture decks (measured: on the smaller `rod` deck at
    axis density 9 the split column happens to carry enough members that the
    difference is visible; both decks are checked rather than assumed)."""
    for deck in sorted(DECKS):
        args_list = _blocks(deck, 9)

        cf._POINT_EVAL_CHUNKED = False
        reference = _run(args_list)

        cf._POINT_EVAL_CHUNKED = True
        # A wide-enough budget that a direction's whole grid is one chunk
        # (many rows, many columns), so the split column carries several
        # members and the tail-split has real member sets on both sides.
        cf._POINT_EVAL_CHUNK_ROWS = 1 << 17
        cf._POINT_EVAL_NEG_CONTROL = True
        broken = _run(args_list)
        cf._POINT_EVAL_NEG_CONTROL = False

        assert any(not np.array_equal(r, b) for r, b in zip(reference, broken)), (
            f"deck {deck!r}: the deliberately split column did not change a bit"
        )


def test_column_ranges_never_splits_a_column():
    """`_column_ranges` on synthetic ρ_eff values: every returned chunk's
    rows share no ρ_eff value with any other chunk's rows (no column split),
    and every row appears in exactly one chunk, at every budget tried
    (including budgets bigger than the whole input, one column, and a
    budget of one row)."""
    rng = np.random.default_rng(1224)
    rho = rng.choice([1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 5.0, 8.0, 13.0], size=237)
    rng.shuffle(rho)

    for budget in (1, 2, 5, 25, 1000):
        chunks = cf._column_ranges(rho, budget)
        seen = np.zeros(rho.size, dtype=bool)
        chunk_rhos = []
        for idx in chunks:
            assert not seen[idx].any()  # no row visited twice
            seen[idx] = True
            vals = set(rho[idx].tolist())
            chunk_rhos.append(vals)
        assert seen.all()  # every row is in some chunk
        # No two chunks share a ρ_eff value (a column is whole in one chunk).
        for i, a in enumerate(chunk_rhos):
            for b in chunk_rhos[i + 1 :]:
                assert a.isdisjoint(b)


def test_column_ranges_oversized_column_stands_alone():
    """A single column bigger than the budget is its own (over-budget)
    chunk rather than being cut to fit: the design constraint `_column_
    ranges`' docstring states directly, isolated from the "no split" check
    above so the budget can be picked smaller than the oversized column on
    purpose."""
    rng = np.random.default_rng(1224)
    small = rng.choice([1.0, 2.0, 3.0, 5.0, 8.0], size=40)
    # rho == 13.0 carries 37 rows, bigger than every budget tried below.
    rho = np.concatenate([small, np.full(37, 13.0)])
    rng.shuffle(rho)

    for budget in (1, 5, 30):
        chunks = cf._column_ranges(rho, budget)
        big = [idx for idx in chunks if 13.0 in set(rho[idx].tolist())]
        assert len(big) == 1
        assert big[0].size == 37
        assert np.all(rho[big[0]] == 13.0)


@pytest.mark.parametrize("budget", [1, 4])
def test_column_ranges_empty_input(budget):
    assert cf._column_ranges(np.array([], dtype=float), budget) == []
