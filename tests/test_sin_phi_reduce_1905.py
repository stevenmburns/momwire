"""The point-matched fill's Φ → Z reduction in C++ — momwire#1905 step 1.

`_reduce_phi_band` hands its sparse regime to the accelerator's
`reduce_phi_band` (`_accel_phi_reduce.cpp`), which performs scipy's
`csr_matvecs` arithmetic element by element — a zero plus the column's
stored entries in stored order, one unfused complex multiply-add each — then
`(A + B) + C`, threaded over observer rows. The scipy tiles it replaces ran
on one thread.

Gates:

  * on the real Φ and M of the #1290 decks (the free-space 8-dipole array
    and the inverted-L over buried radials, the mixed fill's tables) the
    kernel equals the scipy tiles BYTE FOR BYTE, at band heights with
    ragged and one-row tails, in place (`out` is Φ₀) and not;
  * on synthetic CSC operands that scipy's constructor would never build —
    unsorted and repeated rows in a column, empty columns, int64 indices,
    products that round to -0 — so the claim rests on the stored order, not
    on a canonical M;
  * the thread count moves nothing, and the gate sees order (the same
    entries reversed within each column do not reproduce it);
  * what the kernel cannot take goes to the scipy tiles;
  * end to end, `_assemble_Z` and the mixed fill with the kernel on equal
    the scipy route (`_PHI_REDUCE_ACCEL = False`) byte for byte, and a spy
    counts that each route actually ran.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse
from test_crossing_serve_524 import invl_deck
from phi_reduce_drift import EXACT, TOL_Z, PhiReduceDrift, matches
from test_sin_reduce_tiles_1290 import _bytes_equal, _free_array_deck, tables  # noqa: F401

from momwire import sinusoidal as sn
from momwire.sinusoidal import SinusoidalSolver

pytestmark = pytest.mark.skipif(
    not sn._HAVE_PHI_REDUCE_ACCEL,
    reason="the accelerator does not carry reduce_phi_band",
)


def _bits_equal(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def _scipy_route(Phi, Ms, out, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(sn, "_PHI_REDUCE_ACCEL", False)
        return sn._reduce_phi_band(Phi, Ms, out)


def _kernel(Phi, Ms, out, n_threads=0):
    args = []
    for M in Ms:
        args += [M.indptr, M.indices, M.data]
    sn._acc.reduce_phi_band(out, Phi[0], Phi[1], Phi[2], *args, n_threads=n_threads)
    return out


@pytest.fixture
def counts(monkeypatch):
    """Calls to the kernel, and sparse products scipy ran."""
    seen = {"accel": 0, "scipy": 0}
    acc = sn._acc

    class _Acc:
        def __getattr__(self, name):
            return getattr(acc, name)

        def reduce_phi_band(self, *a, **kw):
            seen["accel"] += 1
            return acc.reduce_phi_band(*a, **kw)

    cls = scipy.sparse.csr_matrix
    vec, multi = cls._matmul_vector, cls._matmul_multivector

    def spy_vec(self, other):
        seen["scipy"] += 1
        return vec(self, other)

    def spy_multi(self, other):
        seen["scipy"] += 1
        return multi(self, other)

    monkeypatch.setattr(sn, "_acc", _Acc())
    monkeypatch.setattr(cls, "_matmul_vector", spy_vec)
    monkeypatch.setattr(cls, "_matmul_multivector", spy_multi)
    return seen


@pytest.mark.parametrize("deck", ["free", "invl"])
@pytest.mark.parametrize("rows", [1, 2, 17, 33, 100, "all"])
def test_kernel_is_the_scipy_tiles(tables, deck, rows, monkeypatch):  # noqa: F811
    Phi, Ms = tables[deck]
    N = Phi[0].shape[0]
    assert N >= sn._DENSE_ASSEMBLY_THRESHOLD
    n = N if rows == "all" else rows
    r0 = (N - n) // 2
    band = [P[r0 : r0 + n] for P in Phi]
    ref = _scipy_route(band, Ms, np.full((n, N), np.nan + 0j), monkeypatch)
    out = np.full((n, N), np.nan + 0j)
    got = sn._reduce_phi_band(band, Ms, out)
    assert got is out
    assert matches(out, ref, f"tiles {deck} rows={rows}")
    # In place, as `_reduce_phi_rows_in_place` calls it: out IS Φ₀.
    inplace = [P.copy() for P in band]
    sn._reduce_phi_band(inplace, Ms, inplace[0])
    assert matches(inplace[0], ref, f"tiles in place {deck} rows={rows}")


def test_rows_in_place_through_bands(tables, monkeypatch):  # noqa: F811
    Phi, Ms = tables["invl"]
    with monkeypatch.context() as m:
        m.setattr(sn, "_PHI_REDUCE_ACCEL", False)
        ref = sn._reduce_phi_rows_in_place([P.copy() for P in Phi], Ms, 100)
    for band in (7, 100, Phi[0].shape[0]):
        work = [P.copy() for P in Phi]
        got = sn._reduce_phi_rows_in_place(work, Ms, band)
        assert got is work[0]
        assert matches(got, ref, f"rows in place band={band}")


def _synthetic(seed, n_rows, n_src, n_cols, per_col, empty_frac=0.2):
    """Three CSC Ms built from raw arrays — rows unsorted within a column,
    some repeated, some columns empty — and three Φ with signed zeros."""
    rng = np.random.default_rng(seed)
    Ms = []
    for _ in range(3):
        counts = rng.integers(1, per_col + 1, n_cols)
        counts[rng.random(n_cols) < empty_frac] = 0
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        nnz = int(indptr[-1])
        indices = rng.integers(0, n_src, nnz).astype(np.int64)
        data = rng.standard_normal(nnz) + 1j * rng.standard_normal(nnz)
        data[rng.random(nnz) < 0.1] = -0.0 + 0.0j
        Ms.append(
            scipy.sparse.csc_matrix((data, indices, indptr), shape=(n_src, n_cols))
        )
    Phi = []
    for _ in range(3):
        P = rng.standard_normal((n_rows, n_src)) + 1j * rng.standard_normal(
            (n_rows, n_src)
        )
        P[rng.random(P.shape) < 0.1] = -0.0 - 0.0j
        Phi.append(P)
    return Phi, Ms


@pytest.mark.parametrize(
    "seed,n_rows,n_src,n_cols,per_col",
    [(0, 40, 50, 60, 4), (1, 1, 9, 9, 3), (2, 65, 120, 80, 1), (3, 7, 3, 200, 9)],
)
def test_kernel_is_scipy_on_raw_csc(seed, n_rows, n_src, n_cols, per_col, monkeypatch):
    Phi, Ms = _synthetic(seed, n_rows, n_src, n_cols, per_col)
    if per_col > 1:  # one entry per column is sorted by construction
        assert not all(M.has_sorted_indices for M in Ms)
    ref = _scipy_route(Phi, Ms, np.empty((n_rows, n_cols), complex), monkeypatch)
    got = _kernel(Phi, Ms, np.full((n_rows, n_cols), np.nan + 0j))
    assert matches(got, ref, f"raw csc seed={seed}")
    # An empty column is +0, not -0 and not left as garbage.
    empty = np.flatnonzero(
        (np.diff(Ms[0].indptr) == 0)
        & (np.diff(Ms[1].indptr) == 0)
        & (np.diff(Ms[2].indptr) == 0)
    )
    if empty.size:
        assert not np.signbit(got[:, empty].view(np.float64)).any()


def test_int32_indices_are_taken(monkeypatch):
    Phi, Ms = _synthetic(5, 20, 30, 30, 3)
    Ms32 = [
        scipy.sparse.csc_matrix(
            (M.data, M.indices.astype(np.int32), M.indptr.astype(np.int32)),
            shape=M.shape,
        )
        for M in Ms
    ]
    ref = _scipy_route(Phi, Ms, np.empty((20, 30), complex), monkeypatch)
    assert matches(
        _kernel(Phi, Ms32, np.empty((20, 30), complex)), ref, "int32 indices"
    )


def test_thread_count_moves_nothing():
    Phi, Ms = _synthetic(6, 97, 80, 80, 5)
    want = _kernel(Phi, Ms, np.empty((97, 80), complex), n_threads=1)
    for nt in (2, 3, 0):
        assert _bits_equal(_kernel(Phi, Ms, np.empty((97, 80), complex), nt), want)


def test_the_gate_sees_order():
    """Agreement is evidence about the order: the same entries reversed
    within each column do not reproduce the kernel."""
    Phi, Ms = _synthetic(7, 30, 40, 40, 8, empty_frac=0.0)
    got = _kernel(Phi, Ms, np.empty((30, 40), complex))
    rev = []
    for M in Ms:
        idx = np.concatenate(
            [np.arange(a, b)[::-1] for a, b in zip(M.indptr[:-1], M.indptr[1:])]
        )
        rev.append(
            scipy.sparse.csc_matrix(
                (M.data[idx], M.indices[idx], M.indptr.copy()), shape=M.shape
            )
        )
    assert not _bits_equal(_kernel(Phi, rev, np.empty((30, 40), complex)), got)


def test_what_the_kernel_cannot_take_goes_to_scipy(counts, monkeypatch):
    Phi, Ms = _synthetic(8, 10, 12, 12, 3)
    real_Ms = [
        scipy.sparse.csc_matrix((M.data.real, M.indices, M.indptr), shape=M.shape)
        for M in Ms
    ]
    out = np.empty((10, 12), complex)
    sn._reduce_phi_band(Phi, real_Ms, out)
    assert counts["accel"] == 0 and counts["scipy"] > 0
    counts["scipy"] = 0
    strided = [np.repeat(P, 2, axis=1)[:, ::2] for P in Phi]
    sn._reduce_phi_band(strided, Ms, out)
    assert counts["accel"] == 0 and counts["scipy"] > 0
    counts["scipy"] = 0
    sn._reduce_phi_band(Phi, Ms, out)
    assert counts == {"accel": 1, "scipy": 0}


@pytest.mark.parametrize(
    "deck",
    [
        pytest.param(_free_array_deck, id="free-array"),
        pytest.param(lambda: invl_deck(x=4), id="invl-buried-mixed"),
    ],
)
def test_the_fill_is_the_scipy_route(deck, counts, monkeypatch):
    def fill():
        s = SinusoidalSolver(**deck())
        geom = s._build_geometry()
        G, _view = s._assemble_Z(geom, s.k)
        return np.array(G, copy=True)

    with monkeypatch.context() as m:
        m.setattr(sn, "_PHI_REDUCE_ACCEL", False)
        ref = fill()
    assert counts["accel"] == 0 and counts["scipy"] > 0
    counts["scipy"] = 0
    got = fill()
    assert counts["accel"] > 0 and counts["scipy"] == 0
    assert matches(
        got, ref, f"_assemble_Z {deck.__name__ if hasattr(deck, '__name__') else deck}"
    )


@pytest.mark.parametrize(
    "deck",
    [
        pytest.param(_free_array_deck, id="free-array"),
        pytest.param(lambda: invl_deck(x=4), id="invl-buried-mixed"),
    ],
)
def test_the_impedance_is_the_scipy_routes(deck, monkeypatch):
    """End to end: the solved impedance with the kernel against the scipy
    route. Bitwise on x86-64; elsewhere the drift is reported."""
    import warnings

    def zin():
        z = SinusoidalSolver(**deck()).compute_impedance()
        return np.atleast_1d(np.asarray(z[0] if isinstance(z, tuple) else z))

    with monkeypatch.context() as m:
        m.setattr(sn, "_PHI_REDUCE_ACCEL", False)
        ref = zin()
    got = zin()
    if EXACT:
        assert got.tobytes() == ref.tobytes()
        return
    d = float(np.max(np.abs(got - ref) / np.abs(ref)))
    warnings.warn(f"Zin relative drift = {d:.3e}", PhiReduceDrift, stacklevel=1)
    assert d <= TOL_Z
