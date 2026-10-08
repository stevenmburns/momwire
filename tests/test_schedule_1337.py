"""The fill-schedule layer (momwire#1337): `momwire._schedule`.

What is gated:

* **each walk is the loop it replaced** -- the same blocks in the same
  order, the checkpoint once per chunk (or block, or k-chunk, or replayed
  window), and the budget rules' arithmetic spelled out;
* **the solvers' fills come through it** -- a B-spline chunked fill (dense,
  image, buried subset, batched sweep) and a razor fill (prepare, replay,
  fold) each move `_schedule.STATS`, so a green bit gate elsewhere is a gate
  on THIS code and not on a path that no longer runs;
* **a restricted walk keeps the dense chunk boundaries** -- the property
  `ObserverRows.windows` documents and the momwire#1131 bit gates rely on.

The bit-identity of the migrated fills against the code before the move is
held by the suites that already pinned those fills (the §3.1 gates in
`docs/design/solver-architecture.md` among them), not restated here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from momwire import _schedule
from momwire import bspline as _bs
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_crossing_serve_524 import hub_deck  # noqa: E402

HALF = 0.962 * 22 / 4


def _dipole(cls, n=60, **kw):
    return cls(
        wires=[np.array([[0.0, 0.0, -HALF], [0.0, 0.0, HALF]])],
        n_per_edge_per_wire=[[n]],
        wavelength=22.0,
        **kw,
    )


@pytest.fixture
def stats():
    _schedule.reset_stats()
    yield _schedule.STATS
    _schedule.reset_stats()


# ----------------------------------------------------------------------
# The walks, against the loops they replaced
# ----------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 3, 7, 50])
def test_sweep_is_the_nested_loop(chunk):
    spans, sources = [(0, 11), (20, 24)], [(0, 5), (9, 13)]
    want, got = [], []
    for r0, r1 in spans:
        for i0 in range(r0, r1, chunk):
            want.append("tick")
            for j0, j1 in sources:
                want.append((i0, min(i0 + chunk, r1), j0, j1))
    n = _schedule.sweep(
        spans,
        chunk,
        lambda *b: got.append(b),
        sources=sources,
        checkpoint=lambda: got.append("tick"),
    )
    assert got == want
    assert n == sum(1 for g in got if g != "tick")


def test_a_restricted_sweep_keeps_the_dense_chunk_boundaries():
    class Runs(_schedule.ObserverRows):
        __slots__ = ()

    rows = np.r_[2:5, 9:14]
    restrict = Runs(rows, 16, 16, lambda r: r, compact=True)
    got = []
    _schedule.sweep(
        [(0, 16)], 4, lambda *b: got.append(b[:2]), sources=[(0, 16)], restrict=restrict
    )
    # dense chunks [0,4) [4,8) [8,12) [12,16), each cut to the requested runs
    assert got == [(2, 4), (4, 5), (9, 12), (12, 14)]
    assert restrict.kwargs()["row_of"][2] == 0
    with pytest.raises(ValueError, match="part of the edge"):
        restrict.covers(slice(3, 7))


def test_walk_k_chunks_prepare_and_replay():
    ticks = []
    seen = []
    _schedule.walk(
        [(1, "a"), (2, "b"), (3, "c")],
        lambda i, s: seen.append(s),
        checkpoint=lambda: ticks.append(1),
        keep=lambda b: b[0] != 2,
    )
    assert seen == ["a", "c"] and len(ticks) == 2
    assert list(_schedule.k_chunks(7, 3)) == [(0, 3), (3, 6), (6, 7)]
    windows = _schedule.prepare(10, 4, lambda lo, hi: (hi - lo, f"{lo}:{hi}"))
    assert windows == [(0, 4, 4, "0:4"), (4, 8, 4, "4:8"), (8, 10, 2, "8:10")]
    ticks.clear()
    assert list(_schedule.replay(windows, lambda: ticks.append(1))) == windows
    assert len(ticks) == 3


def test_the_budget_rules_are_the_trunks_own():
    # bspline: bytes against swept_mem_mb, never fewer than one row
    assert _schedule.mb_rows(1, 9 * 100 * 16) == (1 << 20) // (9 * 100 * 16)
    assert _schedule.mb_rows(0, 123) == 1
    # razor: elements against _CHUNK_ELEMS
    assert _schedule.elem_rows(2_000_000, 0) == 2_000_000
    assert _schedule.elem_rows(10, 3) == 3
    # the sweep's k-chunk: the budget less the reserve, clipped to [1, n_k]
    assert _schedule.k_chunk(9, 1, 1 << 18) == 4
    assert _schedule.k_chunk(9, 1, 1 << 18, reserve_bytes=1 << 19) == 2
    assert _schedule.k_chunk(9, 1, 1, reserve_bytes=1 << 21) == 1
    assert _schedule.k_chunk(3, 256, 1) == 3
    assert _schedule.fits_mb(1 << 20, 1) and not _schedule.fits_mb((1 << 20) + 1, 1)


def test_fold_rows_is_the_whole_block_subtraction():
    rng = np.random.default_rng(1337)
    out = rng.normal(size=(9, 5)) + 1j * rng.normal(size=(9, 5))
    blk = rng.normal(size=(9, 5)) + 1j * rng.normal(size=(9, 5))
    want = out - blk
    _schedule.fold_rows(
        out, ((lo, min(lo + 4, 9), blk[lo : lo + 4]) for lo in (0, 4, 8))
    )
    assert np.array_equal(out.view(np.uint64), want.view(np.uint64))


# ----------------------------------------------------------------------
# The solvers' fills come through the layer
# ----------------------------------------------------------------------


def _bspline_parts(s):
    geom = s._build_geometry()
    supp, polys, *_ = s._build_basis_polynomials(geom)
    return geom, supp, polys


@pytest.mark.skipif(
    not _bs._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL, reason="no windowed assembler"
)
@pytest.mark.parametrize("ground", [{}, {"ground_z": -2.0 * HALF}])
def test_bspline_chunked_fill_walks_the_layer(stats, ground):
    # (9, 300, 300) complex moments are 13 MB: the chunked fill, not the tensor
    s = _dipole(BSplineSolver, n=300, swept_mem_mb=1, **ground)
    geom, supp, polys = _bspline_parts(s)
    assert not s._dense_tensor_fits_budget(geom["n_segs_total"])
    s._compute_Z_operator(geom, supp, polys)
    assert stats["sweeps"] >= 2 and stats["chunks"] > 2
    assert stats["walked"] >= 1  # the same-edge correction


def test_bspline_buried_subset_fill_walks_the_layer(stats):
    if not BSplineSolver(**hub_deck())._buried_chunked_serves:
        pytest.skip("the chunked buried fill is not built")
    s = BSplineSolver(**hub_deck())
    s.swept_mem_mb = 0
    geom, supp, polys = _bspline_parts(s)
    s._compute_Z_operator(geom, supp, polys)
    assert stats["blocks"] > stats["sweeps"] >= 4


def test_bspline_batched_sweep_walks_the_k_axis(stats):
    s = _dipole(BSplineSolver, n=40, swept_mem_mb=1)
    if not s._swept_batched_available():
        pytest.skip("the batched sweep is not built")
    ks = 2 * np.pi / 22.0 * np.array([0.95, 1.0, 1.05])
    s.compute_impedance_swept(ks)
    assert stats["k_chunks"] >= 2


def test_razor_prepare_replay_and_fold_walk_the_layer(stats, monkeypatch):
    from momwire import razor as _razor

    monkeypatch.setattr(_razor, "_CHUNK_ELEMS", 400)
    s = _dipole(RazorSolver, n=40, ground_z=-2.0 * HALF)
    geom = s._build_geometry()
    prepared = s._assemble_Z_prepare(geom)
    assert stats["prepares"] == 2  # the real and the mirrored source sets
    n_windows = len(prepared["t1_row_chunks"])
    assert n_windows > 1
    s._assemble_Z_from_prepared(geom, prepared, s.k, s.c * s.k)
    assert stats["replays"] >= 2 * n_windows
    assert stats["folds"] == n_windows  # the image block, a window at a time
