"""The crossing end loops' matvecs read straight from the product's stores
(momwire#1224, `end_matvecs` in `_accel_left_gather.cpp`).

Each fast end's two vectors, `_real_matvec_c(M, w * V)` and its W twin, used
to be formed from a V/W table gathered out of the tile block, the held store
or the row-ordered V/W store, one end at a time. `end_matvecs` forms them
for a batch of ends reading the stored floats in place, summing as scipy's
one-vector CSR product sums, so on a build where neither side contracts it
is the numpy route to the bit (measured on Skylake; against numpy these rows
hold it to the summation-order bound on every platform, since scipy's build
decides whether ITS sum contracts). `_real_matvec_c` itself goes through the
kernel, so every route's end vectors are one loop's and agree to the bit by
construction. These rows pin that on random operands from
both sources, and on razor's crossing decks (the fused route on the hub,
the unfused one on the inverted-L) with a count proving the kernel served.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _accel
from momwire import _crossing_fill as cf
from momwire.razor import RazorSolver

from test_crossing_serve_524 import hub_deck, invl_deck
from test_factorize_1224 import _solve
from test_left_gather_1224 import _complex, _csr, within_sum_bound, z_within_rounding

pytestmark = pytest.mark.skipif(
    not cf._HAVE_END_MATVECS_ACCEL,
    reason="the accelerator does not carry end_matvecs",
)


def _operands(seed, held):
    rng = np.random.default_rng(seed)
    n, nb, E = 97, 31, 13
    M = _csr(rng, nb, n, 0.08, unsorted=bool(seed % 2))
    w = rng.standard_normal(n) * 10.0 ** rng.uniform(-2, 2, n)
    n_tb = 400
    tb = _complex(rng, (n_tb, 4))
    R = rng.integers(0, 1000, (E, n))
    loc = np.full(1000, -1, dtype=np.int32)
    loc[rng.choice(1000, n_tb, replace=False)] = np.arange(n_tb, dtype=np.int32)
    hpos = None
    held_arr = np.empty((0, 4), dtype=np.complex128, order="F")
    if held:
        hpos = np.full(1000, -1, dtype=np.int32)
        gone = np.flatnonzero(loc < 0)
        hpos[gone] = np.arange(gone.size, dtype=np.int32)
        held_arr = _complex(rng, (gone.size, 4))
    else:
        R = np.flatnonzero(loc >= 0)[rng.integers(0, n_tb, (E, n))]
    return M, w, R, loc, tb, held_arr, hpos


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("k", [1, 2])
def test_tile_matvecs_are_the_numpy_vectors(held, k, monkeypatch):
    M, w, R, loc, tb, held_arr, hpos = _operands(10 * k + held, held)
    got = cf._tile_matvecs(M, w, R, k, loc, tb, held_arr, hpos)
    monkeypatch.setattr(cf, "_END_MATVECS", False)
    want = cf._tile_matvecs(M, w, R, k, loc, tb, held_arr, hpos)
    # and the reference itself is `_vw`'s gather and `_real_matvec_c`, end by end
    for e in range(R.shape[0]):
        V, W = cf._FusedEnds._vw(R[e], loc, tb, held_arr, hpos)
        x = w * (V if k == 1 else W)
        assert np.array_equal(want[e], cf._real_matvec_c(M, x))  # numpy, both
        assert within_sum_bound(got[e], want[e], M, x)
    monkeypatch.setattr(cf, "_END_MATVECS", True)
    for e in range(R.shape[0]):
        # The one-vector route is the kernel too: to the bit.
        V, W = cf._FusedEnds._vw(R[e], loc, tb, held_arr, hpos)
        assert np.array_equal(got[e], cf._real_matvec_c(M, w * (V if k == 1 else W)))


@pytest.mark.parametrize("col", [0, 1])
def test_store_matvecs_are_the_numpy_vectors(col, monkeypatch):
    rng = np.random.default_rng(7 + col)
    M = _csr(rng, 23, 61, 0.1, unsorted=True)
    w = rng.standard_normal(61)
    store = _complex(rng, (500, 2))
    R = rng.integers(0, 500, (9, 61))
    got = cf._store_matvecs(M, w, R, store, col)
    for e in range(R.shape[0]):
        assert np.array_equal(got[e], cf._real_matvec_c(M, w * store[R[e], col]))
    monkeypatch.setattr(cf, "_END_MATVECS", False)
    want = cf._store_matvecs(M, w, R, store, col)
    for e in range(R.shape[0]):
        assert within_sum_bound(got[e], want[e], M, w * store[R[e], col])


def test_a_fused_end_reading_a_row_not_in_hand_is_refused():
    M, w, R, loc, tb, held_arr, _hpos = _operands(3, True)
    with pytest.raises(AssertionError, match="not in hand"):
        cf._tile_matvecs(M, w, R, 1, loc, tb, held_arr, None)


DECKS = {
    "buried": lambda: hub_deck(n_radials=4),
    "invl": lambda: invl_deck(n_radials=4),
}


@pytest.mark.parametrize("deck", sorted(DECKS))
def test_razor_z_does_not_move(deck, monkeypatch):
    def make():
        return RazorSolver(**DECKS[deck](), nec5_quadrature=True)

    monkeypatch.setattr(cf, "_END_MATVECS", False)
    z_ref, i_ref = _solve(make, ("_assemble_Z",))
    monkeypatch.setattr(cf, "_END_MATVECS", True)
    calls = []
    acc = _accel.acc

    class _Counted:
        def __getattr__(self, name):
            return getattr(acc, name)

        def end_matvecs(self, *a):
            calls.append(1)
            return acc.end_matvecs(*a)

    monkeypatch.setattr(_accel, "acc", _Counted())
    z_new, i_new = _solve(make, ("_assemble_Z",))
    assert calls, "end_matvecs never served"
    z_within_rounding(z_new, z_ref, i_new, i_ref)


def test_a_complex_basis_takes_the_numpy_vectors(monkeypatch):
    """Complex samples (the sinusoidal sampler's) are not the kernel's: the
    numpy form serves them, imaginary parts and all."""
    M, w, R, loc, tb, held_arr, hpos = _operands(5, True)
    Mc = M.astype(np.complex128)
    Mc.data = Mc.data * (1 + 0.5j)
    served = []
    acc = _accel.acc

    class _Spy:
        def __getattr__(self, name):
            return getattr(acc, name)

        def end_matvecs(self, *a):
            served.append(1)
            return acc.end_matvecs(*a)

    monkeypatch.setattr(_accel, "acc", _Spy())
    got = cf._tile_matvecs(Mc, w, R, 1, loc, tb, held_arr, hpos)
    assert not served
    for e in range(R.shape[0]):
        V, _W = cf._FusedEnds._vw(R[e], loc, tb, held_arr, hpos)
        assert np.array_equal(got[e], cf._real_matvec_c(Mc, w * V))
