"""SG's band scatter without `np.add.at` — momwire#1290.

`_scatter_band` and the extended-kernel bracket accumulated T rows with
`np.add.at`, whose element-at-a-time loop was 1.28 s of a 6.9 s free-space
solve at N = 2816. `_ordered_row_scatter` performs the same additions in the
same per-row order, rank by rank, so it must be `np.add.at` to the bit.

Gates:

  * bit-identity to `np.add.at` on 2-D rows and on a 1-D column view, with
    repeated destinations in shuffled order, a non-zero starting `dest` (the
    carried partial rows), signed zeros, and junction-sized multiplicities;
  * the gate can fail: the same data summed in another order does differ, so
    agreement is evidence about the order and not an accident of the data;
  * chunking within a rank changes nothing (a tiny chunk budget forces many).

End to end, the fused fill's own bit-identity to the whole-triple product is
`tests/test_sg_fused_banded_fill_1224.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import sinusoidal_galerkin as sg


def _case(seed, n_dest, n_entries, width, max_mult=None):
    rng = np.random.default_rng(seed)
    if max_mult is None:
        idx = rng.integers(0, n_dest, size=n_entries)
    else:
        idx = np.repeat(np.arange(n_dest), rng.integers(1, max_mult + 1, n_dest))
        rng.shuffle(idx)
    shape = (idx.size,) if width is None else (idx.size, width)
    dshape = (n_dest,) if width is None else (n_dest, width)
    # Magnitudes over 16 decades make every reassociation visible.
    scale = 10.0 ** rng.uniform(-8, 8, size=shape)
    src = scale * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    src.flat[:: max(1, src.size // 7)] = -0.0
    dest = rng.standard_normal(dshape) + 1j * rng.standard_normal(dshape)
    return dest, idx.astype(np.int64), src


@pytest.mark.parametrize(
    "seed,n_dest,n_entries,width,max_mult",
    [
        (0, 50, 400, 33, None),
        (1, 200, 200, 64, 2),
        (2, 10, 300, 17, 40),
        (3, 64, 500, None, None),
        (4, 1, 30, 5, None),
    ],
)
def test_ordered_row_scatter_is_add_at_to_the_bit(
    seed, n_dest, n_entries, width, max_mult
):
    dest, idx, src = _case(seed, n_dest, n_entries, width, max_mult)
    want = dest.copy()
    np.add.at(want, idx, src)
    got = dest.copy()
    sg._ordered_row_scatter(got, idx, src)
    assert np.array_equal(got.view(np.float64), want.view(np.float64))
    assert np.array_equal(np.signbit(got.real), np.signbit(want.real))


def test_ordered_row_scatter_into_a_column_view():
    dest, idx, src = _case(5, 40, 160, None, None)
    D_want = np.zeros((40, 3), dtype=np.complex128)
    D_got = D_want.copy()
    D_want[:, 1] = dest
    D_got[:, 1] = dest
    np.add.at(D_want[:, 1], idx, src)
    sg._ordered_row_scatter(D_got[:, 1], idx, src)
    assert np.array_equal(D_got, D_want)


def test_the_gate_sees_order():
    dest, idx, src = _case(6, 20, 400, 9, None)
    want = dest.copy()
    np.add.at(want, idx, src)
    rev = dest.copy()
    np.add.at(rev, idx[::-1], src[::-1])
    assert not np.array_equal(rev, want)


def test_chunking_moves_nothing(monkeypatch):
    dest, idx, src = _case(7, 30, 600, 11, None)
    want = dest.copy()
    np.add.at(want, idx, src)
    monkeypatch.setattr(sg, "_ROW_SCATTER_CHUNK_BYTES", 1)
    got = dest.copy()
    sg._ordered_row_scatter(got, idx, src)
    assert np.array_equal(got, want)


def test_empty_index_is_a_no_op():
    dest = np.ones((4, 2), dtype=np.complex128)
    sg._ordered_row_scatter(
        dest, np.empty(0, np.int64), np.empty((0, 2), np.complex128)
    )
    assert np.array_equal(dest, np.ones((4, 2)))
