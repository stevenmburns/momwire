"""The in-medium same-edge kernel in pieces on a thread pool (momwire#1290).

`_seg_seg_reg_moments_from_geometry` at a complex k has no C++ twin; its
kernel table and einsum now run in fixed-size pieces on threads, numpy's own
arithmetic on every element. The pieces may not move a bit, so the gates are
uint64 against the whole-array calls (`_REG_CPLX_PIECES = False`):

- G-1290-P1  straight edges whose table is and is not a whole number of
             pieces, whose segment count leaves a one-row last einsum piece,
             and a windowed (rows=) geometry.
- G-1290-P2  the thread count does not reach the answer: a one-worker pool
             gives the same bits.
- G-1290-P3  end to end, the buried Z with and without the pieces, the
             pieces counted as run.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire import _bspline_kernels as bk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_field_galerkin_914 import screen_deck  # noqa: E402

K_M = complex(0.98, -0.21)


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _moments(n_seg, n_qp, monkeypatch, pieces, rows=None):
    monkeypatch.setattr(bk, "_REG_CPLX_PIECES", pieces)
    ends = np.cumsum(np.r_[0.0, 0.05 + 0.01 * np.sin(np.arange(n_seg))])
    geo = bk._seg_seg_reg_geometry(ends, 1e-3, 2, n_qp, rows=rows)
    return bk._seg_seg_reg_moments_from_geometry(geo, K_M)


@pytest.mark.parametrize(
    "n_seg, n_qp, rows",
    [
        (64, 8, None),  # 262144 table entries: exactly four pieces
        (101, 8, None),  # not a whole number of pieces
        (49, 6, None),  # 49 = 3 * 16 + 1: a one-row last einsum piece
        (200, 4, slice(30, 113)),  # windowed rows
    ],
)
def test_g1290_p1_pieces_are_the_whole_calls(n_seg, n_qp, rows, monkeypatch):
    got = _moments(n_seg, n_qp, monkeypatch, True, rows)
    ref = _moments(n_seg, n_qp, monkeypatch, False, rows)
    assert got.shape == ref.shape
    assert np.array_equal(_bits(got), _bits(ref))


def test_g1290_p2_the_thread_count_does_not_reach_the_answer(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    many = _moments(101, 8, monkeypatch, True)
    one = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(bk, "_reg_cplx_pool", lambda: one)
    try:
        single = _moments(101, 8, monkeypatch, True)
    finally:
        one.shutdown()
    assert np.array_equal(_bits(many), _bits(single))


@pytest.mark.slow
def test_g1290_p3_the_buried_z_with_and_without_the_pieces(monkeypatch):
    def fill(pieces):
        monkeypatch.setattr(bk, "_REG_CPLX_PIECES", pieces)
        ran = {"pieces": 0}
        real = bk._reg_cplx_pool

        class Count:
            def map(self, fn, it):
                ran["pieces"] += 1
                return real().map(fn, it)

        monkeypatch.setattr(bk, "_reg_cplx_pool", lambda: Count())
        s = BSplineSolver(**screen_deck(12, n_per_radial=60))
        geom = s._build_geometry()
        supp_seg, polys, *_ = s._build_basis_polynomials(geom)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Z = s._compute_Z_operator_buried(geom, supp_seg, polys)
        monkeypatch.undo()
        return Z, ran["pieces"]

    Zp, n_p = fill(True)
    Zw, n_w = fill(False)
    assert n_p > 0 and n_w == 0, (n_p, n_w)
    assert np.array_equal(_bits(Zp), _bits(Zw))
