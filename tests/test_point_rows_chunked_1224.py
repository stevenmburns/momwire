"""Bit-identity gate for the chunked point-observer block — momwire#1224
stage 3 unit 3.

`_chunked_point_tables` replaces `point_observer_block`'s dense whole-grid
tables with one dedup, one designed-tables/point-family evaluation, and a
gather + contraction per OBSERVER-ROW chunk (`_POINT_CHUNKED = True`, the
default). This must be BIT-IDENTICAL to `_point_observer_block_dense`
(`_POINT_CHUNKED = False`), the stage-2 route kept as the reference, at
EVERY chunk size — including one observer per chunk, the narrowest cut
`_chunked_point_tables`'s pass-1/pass-2 merge can take.

Gates:
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
    yield
    cf._POINT_CHUNKED = chunked0
    cf._POINT_CHUNK_PAIRS = pairs0
    cf._POINT_NEG_CONTROL = neg0


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
