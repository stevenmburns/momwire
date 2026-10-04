"""Bit-identity gates for the point rows' evaluation tiles -- momwire#1224
perf item 5.

`_chunked_point_tables` used to evaluate every unique row of a direction's
(observers x source nodes) grid up front and hold all of their values while
the observer chunks were gathered; that whole-grid hold is why the
point-matched lane refused past 4 M pairs per direction. It now groups the
observer chunks into tiles of `_POINT_TILE_PAIRS` pairs, evaluates each
ρ_eff column whole at the first tile that reads it and frees it after the
last (`_PointTileSchedule`), and the cap is gone. None of it may move a bit.

Gates:
  * the block, tiled at budgets that force many tiles, equals the one-tile
    block byte for byte on real grids (the bent crossing deck, the buried
    inverted-L) -- counted, so the gate cannot pass on one tile, and with a
    pool smaller than the grid's unique rows, so the tiles really held less;
  * the grids are ones whose ρ_eff columns cross a tile boundary, which is
    the case the column widening exists for: the TEST-ONLY negative control
    (`_POINT_TILE_NEG_CONTROL`: each row evaluated at its own first tile,
    so a column is split across calls) fails the same equality;
  * through the production seam, `SinusoidalSolver.compute_impedance` on a
    crossing deck at a tile budget that forces several tiles per direction
    returns the same Z and currents bytes as at one tile, and the counters
    prove the tiled route ran inside that solve;
    -- the shape of a deck past the old 4 M-pair refusal (invl x32, 15 M
    pairs a direction), shrunk so a small deck takes several tiles in the
    default lane.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_point_rows_chunked_1224 import _blocks, _run
from test_point_rows_memory_1267 import _bytes_equal, _invl_blocks
from test_sin_crossing_serve_1223 import SOIL_A, _make

from momwire import _crossing_fill as cf
from momwire.sinusoidal import SinusoidalSolver


@pytest.fixture(autouse=True)
def _restore_point_state():
    saved = (
        cf._POINT_CHUNKED,
        cf._POINT_CHUNK_PAIRS,
        cf._POINT_EVAL_ROWS,
        cf._POINT_TILE_PAIRS,
        cf._POINT_TILE_MIN_PAIRS,
        cf._POINT_TILE_NEG_CONTROL,
    )
    # Tiling starts past the old 4 M-pair cap; these grids are small, so
    # every test here tiles from the first pair.
    cf._POINT_TILE_MIN_PAIRS = 0
    yield
    (
        cf._POINT_CHUNKED,
        cf._POINT_CHUNK_PAIRS,
        cf._POINT_EVAL_ROWS,
        cf._POINT_TILE_PAIRS,
        cf._POINT_TILE_MIN_PAIRS,
        cf._POINT_TILE_NEG_CONTROL,
    ) = saved


def _grids():
    return {"bent": _blocks("bent", 9), "invl": _invl_blocks()}


def _tiled(args_list, chunk_pairs, tile_pairs):
    cf._POINT_CHUNKED = True
    cf._POINT_CHUNK_PAIRS = chunk_pairs
    cf._POINT_TILE_PAIRS = tile_pairs
    before = dict(cf._ROUTES)
    out = _run(args_list)
    delta = {k: cf._ROUTES[k] - before[k] for k in cf._ROUTES}
    return out, delta


def _unique_rows_total(args_list):
    """The grids' unique folded rows, summed over the directions."""
    from momwire import _near_interface as ni

    total = 0
    for ctx, obs, _t, src, above in args_list:
        _e, _k, gz, _c1, _memo = cf._block_preamble(ctx)
        P = np.asarray(obs, dtype=float)
        nodes = np.asarray(src["nodes"], dtype=float)
        _dx, _dy, rho, z, zp = cf._point_pair_grid(P, nodes, gz, above)
        rows, _inv = ni._unique_rows(ni.radius_fold(rho, ctx.a_wire), z, zp)
        total += rows.shape[0]
    return total


def _column_crosses_a_tile(args_list, chunk_pairs, tile_pairs):
    """True when some direction's ρ_eff column has members read in two
    tiles -- the split the schedule's column widening must undo."""
    from momwire import _near_interface as ni

    for ctx, obs, _t, src, above in args_list:
        _e, _k, gz, _c1, _memo = cf._block_preamble(ctx)
        P = np.asarray(obs, dtype=float)
        nodes = np.asarray(src["nodes"], dtype=float)
        nA, nB = P.shape[0], nodes.shape[0]
        step = max(1, chunk_pairs // nB)
        per = max(1, tile_pairs // (step * nB))
        _dx, _dy, rho, _z, _zp = cf._point_pair_grid(P, nodes, gz, above)
        r = ni.radius_fold(rho, ctx.a_wire)
        tile = (np.arange(nA) // step // per)[:, None] * np.ones((1, nB), int)
        _u, inv = np.unique(r.ravel(), return_inverse=True)
        lo = np.full(_u.size, np.iinfo(np.intp).max)
        hi = np.full(_u.size, -1)
        np.minimum.at(lo, inv, tile.ravel())
        np.maximum.at(hi, inv, tile.ravel())
        if np.any(lo != hi):
            return True
    return False


# (observer-chunk pairs, tile pairs): one observer row per chunk and per
# tile (the narrowest cut), a few rows per tile, and tiles of several
# chunks each.
BUDGETS = [(1, 1), (1, 40), (40, 200)]


@pytest.mark.parametrize("budget", BUDGETS)
@pytest.mark.parametrize("grid", ["bent", "invl"])
def test_tiled_block_equals_one_tile_bit_for_bit(grid, budget):
    args_list = _grids()[grid]
    chunk_pairs, tile_pairs = budget
    one, d1 = _tiled(args_list, chunk_pairs, 1 << 40)
    assert d1["point_tiles"] == len(args_list)  # one tile per direction
    tiled, dt = _tiled(args_list, chunk_pairs, tile_pairs)
    assert dt["point_chunked"] == len(args_list)
    assert dt["point_tiles"] > 2 * len(args_list)  # many tiles ran
    # Columns cross tile boundaries at this budget (non-vacuous) ...
    assert _column_crosses_a_tile(args_list, chunk_pairs, tile_pairs)
    # ... and the pool really held fewer rows than the grid's unique set.
    assert dt["point_tile_pool_max_rows"] < _unique_rows_total(args_list)
    for a, b in zip(one, tiled):
        assert _bytes_equal(a, b)


def test_keep_mask_is_keep():
    """`designed_rows(keep_mask=keep.contains(rows))` -- the lookup the tile
    schedule makes once, up front -- is `designed_rows(keep=keep)`: the same
    block bytes and the same rows retained in the memo, with their values."""
    from momwire import _near_interface as ni

    ctx, obs, _t, src, above = _invl_blocks()[1]
    eps_t, k_p, gz, _c1, _memo = cf._block_preamble(ctx)
    P = np.asarray(obs, dtype=float)
    nodes = np.asarray(src["nodes"], dtype=float)
    _dx, _dy, rho, z, zp = cf._point_pair_grid(P, nodes, gz, above)
    rows, _inv = ni._unique_rows(ni.radius_fold(rho, ctx.a_wire), z, zp)
    keep = ni.TripleMemo.key_set(rows[:: max(1, rows.shape[0] // 41)])
    out = []
    for kw in ({"keep": keep}, {"keep_mask": keep.contains(rows)}):
        memo = ni.ProductMemo()
        blk = ni.designed_rows(eps_t, k_p, rows, rtol=cf._CROSS_RTOL, memo=memo, **kw)
        out.append((blk, memo))
    (b1, m1), (b2, m2) = out
    assert _bytes_equal(b1, b2)
    assert len(m1) == len(m2) == len(keep)
    held = m1.contains(rows)
    assert np.array_equal(held, m2.contains(rows))
    h1, v1 = m1.lookup(rows[held])
    h2, v2 = m2.lookup(rows[held])
    assert h1.all() and h2.all() and _bytes_equal(v1, v2)


def test_tiled_block_equals_the_dense_reference():
    """And the tiled block is the stage-2 dense route's bits, the reference
    the chunked route has always been gated against."""
    args_list = _grids()["invl"]
    cf._POINT_CHUNKED = False
    dense = _run(args_list)
    tiled, dt = _tiled(args_list, 1, 40)
    assert dt["point_tiles"] > 2 * len(args_list)
    for a, b in zip(dense, tiled):
        assert _bytes_equal(a, b)


def test_negative_control_splitting_columns_fails_the_gate():
    """Evaluating each row at its OWN first tile (a column cut across calls)
    moves bits on these grids -- the gate above can fail, and the column
    widening is what it holds."""
    moved = False
    for args_list in _grids().values():
        one, _d = _tiled(args_list, 1, 1 << 40)
        cf._POINT_TILE_NEG_CONTROL = True
        broken, _d = _tiled(args_list, 1, 1)
        cf._POINT_TILE_NEG_CONTROL = False
        moved |= any(not _bytes_equal(a, b) for a, b in zip(one, broken))
    assert moved


def _solve(tile_pairs):
    """The lean crossing deck (`test_sin_crossing_serve_1223`), one observer
    per chunk, at `tile_pairs` pairs a tile."""
    cf._POINT_CHUNK_PAIRS = 1
    cf._POINT_TILE_PAIRS = tile_pairs
    before = dict(cf._ROUTES)
    s = _make(SinusoidalSolver, "lean", SOIL_A, 9)
    z, cur = s.compute_impedance()
    delta = {k: cf._ROUTES[k] - before[k] for k in cf._ROUTES}
    return np.atleast_1d(np.asarray(z)), np.asarray(cur), delta


def test_solve_through_tiles_is_the_one_tile_solve_bit_for_bit():
    """The production seam: the solver's own crossing assembly, at a tile
    budget that cuts the directions into several tiles, returns the one-
    tile solve's Z and currents to the byte. The counters are read around
    `compute_impedance` itself, so they prove the tiles ran in the solve.
    This is the old refusal's shape -- a direction past the block's budget,
    which stage 2 refused by name -- now solving."""
    z1, c1, d1 = _solve(1 << 40)
    assert d1["point_chunked"] == 2 and d1["point_tiles"] == 2
    zt, ct, dt = _solve(1)
    assert dt["point_chunked"] == 2
    assert dt["point_tiles"] > 4  # both directions took several tiles
    assert _bytes_equal(z1, zt)
    assert _bytes_equal(c1, ct)
    assert np.all(np.isfinite(zt)) and np.all(np.isfinite(ct))


def test_a_grid_under_the_minimum_is_one_tile():
    """`_POINT_TILE_MIN_PAIRS`: a direction of at most that many pairs is
    evaluated untiled whatever the tile budget -- the schedule's sort is not
    paid where its pool saving is small, and every deck the old cap served
    takes exactly its old evaluation."""
    args_list = _grids()["invl"]
    pairs = [a[1].shape[0] * a[3]["nodes"].shape[0] for a in args_list]
    cf._POINT_TILE_MIN_PAIRS = max(pairs)
    out, d = _tiled(args_list, 1, 1)
    assert d["point_tiles"] == len(args_list)
    cf._POINT_TILE_MIN_PAIRS = max(pairs) - 1
    _out, d = _tiled(args_list, 1, 1)
    assert d["point_tiles"] > len(args_list)
