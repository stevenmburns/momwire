"""The main sandwich's five-term combine in C++ (momwire#1290,
`combine_rows` in `_accel_left_gather.cpp`).

`_crossing_fill._combine` formed the block as six scipy dense @ sparse.T
products summed by numpy, and the streamed sandwich first copied each
chunk's left products into the contraction's column order and sliced the
right weights to `Q[J][:, need]`. `combine_rows` reads the products where
they are held (the chunk's or the held set's, through a column map) and
the weights' full rows, with scipy's CSR product order from zero and
numpy's combine order. Every route takes it, so the routes agree to the bit
by construction on any build; against scipy and numpy it is the same bits
where neither contracts (x86-64) and within the summation-order bound
anywhere (`test_left_gather_1224.within_sum_bound`), since a build that
fuses a*x + acc rounds once where this rounds twice.
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
    not cf._HAVE_COMBINE_ACCEL,
    reason="the accelerator does not carry combine_rows",
)


def _numpy_combine(Ls, Qz):
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cf, "_COMBINE_ACCEL", False)
        return cf._combine(Ls, Qz)


def _operands(rng, r_a, n_cols, n_rows, *, unsorted=False):
    Q4 = [
        _csr(rng, n_rows, n_cols, d, unsorted=unsorted and k == 2)
        for k, d in enumerate((0.08, 0.08, 0.06, 0.05))
    ]
    Ls = [_complex(rng, (r_a, n_cols), "C") for _ in range(6)]
    return Ls, (Q4[0], Q4[1], Q4[2], Q4[3], Q4[2], Q4[3])


@pytest.mark.parametrize("unsorted", [False, True])
def test_the_kernel_is_the_numpy_combine(unsorted):
    rng = np.random.default_rng(1290 + unsorted)
    Ls, Qs = _operands(rng, 23, 71, 40, unsorted=unsorted)
    got = cf._combine(Ls, Qs)
    want = _numpy_combine(Ls, Qs)
    assert got.shape == want.shape == (23, 40) and got.dtype == np.complex128
    # Term by term (the others zero, so the combine adds exact zeros): each
    # is `Ls[i] @ Qs[i].T`, the transpose of the CSR sum `Qs[i] @ Ls[i].T`
    # the bound is written for.
    for i, q in enumerate(Qs):
        alone = [np.zeros_like(L) for L in Ls]
        alone[i] = Ls[i]
        sign = -1 if i == 5 else 1
        assert within_sum_bound(
            sign * cf._combine(alone, Qs).T,
            sign * _numpy_combine(alone, Qs).T,
            q,
            Ls[i].T,
        )


def test_the_column_map_reads_what_the_gathered_products_held():
    """`colmap` over a chunk and a held set: the block the gathered
    products and the sliced rows give `_combine`, to the bit -- both are
    the same kernel loop over the same terms in the same order."""
    rng = np.random.default_rng(7)
    r_a, n_cols, n_rows = 17, 60, 30
    Ls, Qs = _operands(rng, r_a, n_cols, n_rows, unsorted=True)
    held_cols = np.sort(rng.choice(n_cols, 25, replace=False))
    new_cols = np.setdiff1d(np.arange(n_cols), held_cols)
    held = [np.ascontiguousarray(L[:, held_cols]) for L in Ls]
    new = [np.ascontiguousarray(L[:, new_cols]) for L in Ls]
    colmap = np.full(n_cols, cf._NOT_IN_HAND, dtype=np.int64)
    colmap[held_cols] = -1 - np.arange(held_cols.size)
    colmap[new_cols] = np.arange(new_cols.size)
    J = np.sort(rng.choice(n_rows, 12, replace=False))
    got = _accel.acc.combine_rows(new, held, colmap, J, *cf._csr_args(Qs), 3)
    need = np.unique(np.concatenate([q[J].indices for q in Qs[:4]]))
    want = cf._combine([L[:, need] for L in Ls], [q[J][:, need] for q in Qs])
    assert np.array_equal(got, want)
    # A column the map does not name is refused, not read as column 0.
    colmap[held_cols[0]] = cf._NOT_IN_HAND
    rows = [j for j in range(n_rows) if held_cols[0] in Qs[0][[j]].indices]
    if rows:
        with pytest.raises(RuntimeError, match="not in hand"):
            _accel.acc.combine_rows(
                new, held, colmap, np.array(rows), *cf._csr_args(Qs), 1
            )


DECKS = {
    "buried": lambda: hub_deck(n_radials=4),
    "invl": lambda: invl_deck(n_radials=4),
}


@pytest.mark.parametrize("deck", sorted(DECKS))
def test_razor_z_does_not_move(deck, monkeypatch):
    def make():
        return RazorSolver(**DECKS[deck](), nec5_quadrature=True)

    monkeypatch.setattr(cf, "_COMBINE_ACCEL", False)
    z_ref, i_ref = _solve(make, ("_assemble_Z",))
    monkeypatch.setattr(cf, "_COMBINE_ACCEL", True)
    calls = []
    real = _accel.acc.combine_rows

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(_accel.acc, "combine_rows", counted)
    z_new, i_new = _solve(make, ("_assemble_Z",))
    assert calls, "the kernel never ran"
    z_within_rounding(z_new, z_ref, i_new, i_ref)
