"""momwire#1132: the sector route fills a ROW-COMPACT Z.

The route (momwire#1029) reads only sector 0's rows and the axial rows of Z,
but its `rows=` fill wrote them into a square (n, n) column-major destination
that transparent huge pages made fully resident — 1047 MB at 150 radials to
hold 11 MB. `compact=True` allocates `(len(R), n)` and hands every writer a
`row_of` map (basis row -> compact row, -1 off R). Each writer moves only the
ADDRESS of its adds, so the gates here are bit equality, not tolerance:

* the compact Z's SHAPE, order and row set (R is exactly what the solve reads);
* the compact rows equal the square `rows=` fill's rows, as uint64, with and
  without loading (the loading's COO remap) and on the numpy fallback of the
  field-form block;
* the route's Z_in and currents equal the square route's, bit for bit;
* NEGATIVE CONTROLS: a wrong `row_of` moves the answer, and a map that sends
  a written row nowhere is refused by the kernels rather than dropped;
* the sliced fallback (no `row_of` kernels) is the same bits;
* the dense path is untouched: no `compact=`, no tuple, the square F-order Z.

The cross-BUILD half of the gate (this tree's .so against main's, including
decks that never pass the new argument) is a scratch record, not a test: a
test runs against one build.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from test_rotational_symmetry_1029 import solver

from momwire import _rotational_symmetry as RS
from momwire import bspline as B


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _fill(s, compact):
    geom = s._build_geometry()
    supp_seg, polys, _kcl, _wk, wbg = s._build_basis_polynomials(geom)
    rows = RS.observer_rows(s, geom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = s._compute_Z_operator_buried(
            geom, supp_seg, polys, rows=rows, compact=compact
        )
    sectors, axial = RS.dof_groups(s, wbg, supp_seg.shape[0])
    return out, sectors, axial, supp_seg.shape[0]


def _square_rows(s):
    Z, sectors, axial, _n = _fill(s, compact=False)
    R = np.union1d(sectors[0], axial)
    return Z[R], R


def _route(s, compact, monkeypatch):
    monkeypatch.setattr(RS, "COMPACT_Z", compact)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z, c = s.compute_impedance()
    return np.atleast_1d(np.asarray(z)), np.asarray(c)


@pytest.fixture(scope="module")
def four():
    """The 4-radial screen: the compact fill, its row set and the groups."""
    s = solver(4)
    (Zc, R), sectors, axial, n = _fill(s, compact=True)
    return s, Zc, R, sectors, axial, n


def test_the_compact_z_holds_exactly_the_rows_the_solve_reads(four):
    s, Zc, R, sectors, axial, n = four
    assert B._HAVE_WINDOWED_ROW_OF and B._HAVE_FIELD_GALERKIN_ROW_OF, (
        "the .so predates momwire#1132: rebuild (`make build`)"
    )
    assert Zc.shape == (R.size, n)
    assert R.size < n
    assert Zc.dtype == np.complex128 and Zc.flags.c_contiguous
    assert np.array_equal(R, np.union1d(sectors[0], axial))
    # every held row was written: none is still the zeros it was born as
    assert np.all(np.any(Zc != 0, axis=1))


def test_the_compact_rows_are_the_square_rows_bit_for_bit(four):
    s, Zc, R, *_ = four
    Zq, Rq = _square_rows(s)
    assert np.array_equal(R, Rq)
    assert np.array_equal(_bits(Zc), _bits(Zq))


def test_the_loading_remap_keeps_the_rows_bit_for_bit():
    """Loading is `np.add.at` over COO entries in EVERY row; the compact fill
    keeps the held rows' entries in their own order. Off on the reference
    deck, which is exactly why it is gated on a deck where it is on."""
    s = solver(4, wire_conductivity=1e6)
    assert s._loading_active
    (Zc, R), *_ = _fill(s, compact=True)
    Zq, Rq = _square_rows(solver(4, wire_conductivity=1e6))
    assert np.array_equal(R, Rq)
    assert np.array_equal(_bits(Zc), _bits(Zq))
    # and the loading did land in the held rows
    (Zc0, _), *_ = _fill(solver(4), compact=True)
    assert not np.array_equal(_bits(Zc), _bits(Zc0))


def test_the_numpy_field_block_remaps_rows_bit_for_bit(monkeypatch):
    """The field-form block's numpy reference (a .so without the #914 twin)
    indexes the compact rows through `row_of` and drops padded slots rather
    than letting -1 index the last row."""
    monkeypatch.setattr(B, "_HAVE_FIELD_GALERKIN_ACCEL", False)
    (Zc, R), *_ = _fill(solver(4), compact=True)
    Zq, _ = _square_rows(solver(4))
    assert np.array_equal(_bits(Zc), _bits(Zq))


def test_the_route_answer_is_the_square_routes_bit_for_bit(monkeypatch):
    zc, cc = _route(solver(4), True, monkeypatch)
    zq, cq = _route(solver(4), False, monkeypatch)
    assert np.array_equal(_bits(zc), _bits(zq))
    assert np.array_equal(_bits(cc), _bits(cq))


def test_negative_control_a_wrong_row_map_moves_the_answer(four):
    """The bit gates above could pass vacuously if the map were never read.
    Swap two sector-0 rows in the map the solve reads the compact Z through,
    and the answer must move."""
    s, Zc, R, sectors, axial, n = four
    geom = s._build_geometry()
    supp_seg, polys, kcl_A, wk, wbg = s._build_basis_polynomials(geom)
    v, _pv, _vpf, _av, kcl = s._feed_drive_and_readout(
        geom, wk, wbg, supp_seg.shape[0], kcl_A
    )
    row_of = np.full(n, -1, dtype=np.int64)
    row_of[R] = np.arange(R.size)
    good = RS.solve(s, Zc, v, kcl, sectors, axial, row_of=row_of)
    bad_map = row_of.copy()
    a, b = sectors[0][0], sectors[0][1]
    bad_map[a], bad_map[b] = row_of[b], row_of[a]
    bad = RS.solve(s, Zc, v, kcl, sectors, axial, row_of=bad_map)
    rel = np.max(np.abs(bad - good)) / np.max(np.abs(good))
    assert rel > 1e-3, rel


def test_negative_control_the_kernels_refuse_a_row_sent_nowhere(four, monkeypatch):
    """A map that drops a LIVE row the window writes is a caller bug, and the
    windowed assemblers' Python side refuses it by name instead of quietly
    leaving that row out."""
    s, Zc, R, sectors, axial, n = four
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    rows = RS.observer_rows(s, geom)
    row_of = np.full(n, -1, dtype=np.int64)
    short = R[1:]  # sector-0/axial row R[0] has nowhere to go
    row_of[short] = np.arange(short.size)
    Z = np.zeros((short.size, n), dtype=np.complex128)
    with pytest.raises(ValueError, match="no row in the compact Z"):
        s._accumulate_Z_subset_chunked(
            Z,
            geom,
            s.k,
            np.arange(geom["n_segs_total"], dtype=np.int64),
            supp_seg,
            polys,
            mirror_sources=False,
            eps=float(s.eps),
            scale=1.0,
            obs_idx=rows,
            row_of=row_of,
        )


def test_the_cxx_contract_refuses_a_map_into_a_missing_row():
    """The C++ check behind the Python one: a row_of entry past the compact Z
    is refused with the GIL held, never written under a released one."""
    acc = B._acc
    n = 3
    supp = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int64)
    polys = np.ones((n, 3, 3))
    tang = np.tile([1.0, 0.0, 0.0], (3, 1))
    J = np.zeros((3, 3, 3, 3), dtype=np.complex128)
    Z = np.zeros((1, n), dtype=np.complex128)
    m_idx = np.arange(n, dtype=np.int64)
    with pytest.raises(RuntimeError, match="outside the compact Z"):
        acc.assemble_Z_bspline_windowed(
            J,
            supp,
            polys,
            tang,
            m_idx,
            m_idx,
            0,
            3,
            0,
            3,
            1.0,
            1.0,
            1.0,
            Z,
            row_of=np.array([0, 1, 2], dtype=np.int64),
        )
    with pytest.raises(RuntimeError, match="two written bases to one Z row"):
        acc.assemble_Z_bspline_windowed(
            J,
            supp,
            polys,
            tang,
            m_idx[:2],
            m_idx,
            0,
            3,
            0,
            3,
            1.0,
            1.0,
            1.0,
            np.zeros((2, n), dtype=np.complex128),
            row_of=np.array([0, 0, 1], dtype=np.int64),
        )


def test_the_sliced_fallback_is_the_same_bits(four, monkeypatch):
    """A .so without the `row_of` kernels fills the square Z and slices it."""
    s, Zc, R, *_ = four
    monkeypatch.setattr(B, "_HAVE_WINDOWED_ROW_OF", False)
    (Zf, Rf), *_ = _fill(solver(4), compact=True)
    assert np.array_equal(R, Rf)
    assert np.array_equal(_bits(Zc), _bits(Zf))


def test_the_dense_path_is_untouched(four):
    s, *_ = four
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = s._compute_Z_operator_buried(geom, supp_seg, polys)
    assert isinstance(Z, np.ndarray)
    assert Z.shape == (supp_seg.shape[0],) * 2 and Z.flags.f_contiguous
    with pytest.raises(ValueError, match="pass rows= as well"):
        s._compute_Z_operator_buried(geom, supp_seg, polys, compact=True)
