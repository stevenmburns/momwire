"""The main sandwich's left products straight from the product tiles
(momwire#1224, `_accel_left_gather.cpp`).

`left_products_gathered` forms `_crossing_fill._left_products`' six sparse
@ dense products from the tiles' stores in place, instead of from the
(nA, |cols|) tables `_ProductTiles._gather` used to copy out first. Its
arithmetic is scipy's CSR product order from zero, so on a build where
neither side contracts (this repo's GCC/clang flags; scipy's x86-64 wheels)
it is the numpy route to the bit -- measured so on Skylake, but not pinned:
scipy's own build decides whether ITS loop contracts a*x + acc into an FMA
(Apple Silicon's clang build does), and then the two differ in the last
bits. So against numpy these rows hold the kernel to that difference's
bound (`within_sum_bound`), on every platform. What IS pinned to the bit is
every route against every other: the tiles, the whole-table dict and the
grid route all form their products through this kernel, by construction.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from momwire import _accel
from momwire import _crossing_fill as cf
from momwire.razor import RazorSolver

from test_crossing_serve_524 import hub_deck, invl_deck
from test_factorize_1224 import _solve

pytestmark = pytest.mark.skipif(
    not cf._HAVE_LEFT_GATHER_ACCEL,
    reason="the accelerator does not carry left_products_gathered",
)


EPS = np.finfo(float).eps


def within_sum_bound(got, want, M, X, *, slack=4.0):
    """`got` and `want` are the same CSR sums `M @ X` up to their rounding:
    per output element, the two orders of a*x + acc (fused or not) differ by
    at most one rounding of each term and of each partial sum, so by
    `nnz_row * eps * sum_row |a| |x|` (real and imaginary parts apart);
    `slack` covers a numpy-side multiply-add folded into X (k2 V + dz'W,
    w * V) rounded the other way. Zero where the bound is zero."""
    A = abs(M)
    nnz = np.diff(M.indptr).max(initial=0)
    for part in (np.real, np.imag):
        bound = slack * max(nnz, 1) * EPS * (A @ np.abs(part(X)))
        if not np.all(np.abs(part(got) - part(want)) <= bound):
            return False
    return True


def z_within_rounding(z_new, z_ref, i_new, i_ref):
    """Z from a route whose sums differ from the reference's only in their
    rounding (`within_sum_bound`): every entry is a short sum of such
    products and stays within 1e-12 of max|Z| (N * eps = 3e-14 at the
    N ~ 150 of these decks, with room); the currents move by at most the
    condition number times Z's relative change."""
    scale = np.max(np.abs(z_ref))
    dz = np.max(np.abs(z_new - z_ref)) / scale
    assert dz <= 1e-12, dz
    cond = np.linalg.cond(z_ref)
    di = np.max(np.abs(i_new - i_ref)) / np.max(np.abs(i_ref))
    assert di <= 10.0 * cond * max(dz, EPS), (di, cond, dz)


def _csr(rng, n_out, n_a, density, *, unsorted=False):
    m = sp.random(n_out, n_a, density=density, format="csr", random_state=rng)
    m.data = rng.standard_normal(m.data.size) * 10.0 ** rng.uniform(-3, 3, m.data.size)
    if unsorted:
        # A row's stored order is the sum's order: reverse it, as a column
        # slice can leave it unsorted.
        for i in range(n_out):
            a, b = m.indptr[i], m.indptr[i + 1]
            m.indices[a:b] = m.indices[a:b][::-1].copy()
            m.data[a:b] = m.data[a:b][::-1].copy()
    return m


def _complex(rng, shape, order="F"):
    v = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    v *= 10.0 ** rng.uniform(-6, 6, shape)
    v[rng.random(shape) < 0.03] = 0.0
    return np.asarray(v, order=order)


@pytest.mark.parametrize("store", [False, True])
@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("order", ["F", "C"])
def test_the_kernel_is_the_numpy_products(store, held, order):
    rng = np.random.default_rng(1224 + 4 * store + 2 * held + (order == "C"))
    n_a, n_c, n_out = 137, 41, 59
    Ps = tuple(
        _csr(rng, n_out, n_a, d, unsorted=k == 1)
        for k, d in enumerate((0.05, 0.05, 0.05, 0.02))
    )
    n_tb, n_rows = 500, 900
    nk = 2 if store else 4
    tb = _complex(rng, (n_tb, nk), order)
    li = rng.integers(0, n_tb, (n_a, n_c)).astype(np.int32)
    miss, hp, held_arr = None, None, np.empty((0, nk), dtype=np.complex128, order="F")
    if held:
        held_arr = _complex(rng, (300, nk), order)
        miss = rng.random((n_a, n_c)) < 0.3
        li[miss] = -1
        hp = rng.integers(0, 300, int(miss.sum()))
    idx = rng.integers(0, n_rows, (n_a, n_c))
    vw = _complex(rng, (n_rows, 2), order) if store else None
    K = cf._TileTables(idx, li, miss, hp, tb, held_arr, vw)
    k2sq = 0.9187**2
    got = K.left_products(Ps, k2sq)
    assert got is not None, "the kernel did not serve"
    # Every route through the kernel: the whole-table dict to the bit.
    dense = cf._left_products(Ps, K.as_dict(), k2sq)
    for g, d in zip(got, dense):
        assert g.shape == d.shape and g.dtype == d.dtype
        assert np.array_equal(g, d)
    # Against numpy: within the summation-order bound (the bit on x86-64).
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cf, "_LEFT_GATHER", False)
        T = K.as_dict()
        want = cf._left_products(Ps, T, k2sq)
    X = (T["U"], T["U"], k2sq * T["V"] + T["dzpW"], T["W"], T["W"], T["V"])
    for g, w, P, x in zip(got, want, (Ps[0], Ps[1], Ps[2], Ps[2], Ps[3], Ps[3]), X):
        assert within_sum_bound(g, w, P, x)


def test_a_miss_with_no_held_slot_is_refused():
    rng = np.random.default_rng(5)
    Ps = tuple(_csr(rng, 4, 6, 0.5) for _ in range(4))
    li = np.full((6, 3), -1, dtype=np.int32)
    with pytest.raises(RuntimeError, match="not in hand"):
        _accel.acc.left_products_gathered(
            [P.indptr for P in Ps],
            [P.indices for P in Ps],
            [P.data for P in Ps],
            4,
            1.0,
            li,
            np.zeros((0, 0), dtype=np.int32),
            _complex(rng, (5, 4)),
            np.empty((0, 4), dtype=np.complex128),
            0,
            1,
            2,
            3,
            np.zeros((0, 0), dtype=np.int64),
            np.zeros((0, 0), dtype=np.complex128),
            1,
        )


DECKS = {
    "buried": lambda: hub_deck(n_radials=4),
    "invl": lambda: invl_deck(n_radials=4),
}


@pytest.mark.parametrize("deck", sorted(DECKS))
def test_razor_z_does_not_move(deck, monkeypatch):
    def make():
        return RazorSolver(**DECKS[deck](), nec5_quadrature=True)

    monkeypatch.setattr(cf, "_LEFT_GATHER", False)
    z_ref, i_ref = _solve(make, ("_assemble_Z",))
    monkeypatch.setattr(cf, "_LEFT_GATHER", True)
    served = []
    real = cf._TileTables.left_products

    def counted(self, Ps, k2sq):
        out = real(self, Ps, k2sq)
        served.append(out is not None)
        return out

    monkeypatch.setattr(cf._TileTables, "left_products", counted)
    z_new, i_new = _solve(make, ("_assemble_Z",))
    assert served and all(served), served
    z_within_rounding(z_new, z_ref, i_new, i_ref)


def test_a_complex_basis_takes_the_numpy_products():
    """The sinusoidal sampler's samples are complex; the kernel's matrices
    are real, so it declines them (and refuses them outright if handed
    them), rather than cast the imaginary parts away."""
    rng = np.random.default_rng(31)
    Ps = tuple(_csr(rng, 5, 12, 0.4).astype(np.complex128) for _ in range(4))
    tb = _complex(rng, (40, 4))
    li = rng.integers(0, 40, (12, 3)).astype(np.int32)
    K = cf._TileTables(rng.integers(0, 40, (12, 3)), li, None, None, tb, tb[:0], None)
    assert K.left_products(Ps, 1.0) is None
    with pytest.raises(TypeError):
        _accel.acc.left_products_gathered(
            [P.indptr for P in Ps],
            [P.indices for P in Ps],
            [P.data for P in Ps],
            5,
            1.0,
            li,
            np.zeros((0, 0), dtype=np.int32),
            tb,
            tb[:0],
            0,
            1,
            2,
            3,
            np.zeros((0, 0), dtype=np.int64),
            np.zeros((0, 0), dtype=np.complex128),
            1,
        )
