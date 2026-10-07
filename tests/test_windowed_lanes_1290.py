"""The windowed assembler's AVX2 lane kernel -- momwire#1290.

`assemble_Z_bspline_windowed` (and its in-medium twin) fill Z from the chunked
moment tensor on every free-space deck too big for the dense tensor, and on
the buried fill. The AVX2 build now runs a lane kernel: eight entries at a
time over cache tiles, each entry's arithmetic operation for operation the
per-entry loop's. `reference=True` still runs that loop, so the gates here
are bit equality against it, not tolerance:

* kernel level, on synthetic windows built to reach every lane case: J read
  as one contiguous run and staged lane by lane, lanes dead because their
  wing lies outside the window (blended away) or all dead (skipped), a tail
  group shorter than eight, offset windows, an empty column window, d = 1, 2
  and 3, the complex-eps twin and the row-compact target (momwire#1132);
* NEGATIVE CONTROL: one ulp on the moments moves the rows the window wrote,
  so agreement is evidence about the arithmetic, not an accident of the data;
* end to end, a free-space deck forced through the chunked fill and a buried
  deck: Z_in and currents equal the reference's bit for bit, with the calls
  COUNTED, so neither gate can pass without the assembler having run;
* the lane kernel is compiled into the x86-64 AVX2 variant (elsewhere both
  spellings are the reference loop, and the bit gates hold trivially).

On Windows (MSVC, /fp:fast) the lanes run too since momwire#1371, and every
lanes-vs-walk comparison here takes the derived win32 tolerance instead of bit
equality (`assert_lanes_match`, tests/_lane_gate.py). Linux and macOS keep the
bit gates.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

from momwire import _accel
from momwire import bspline as B
from _lane_gate import assert_lanes_match

acc = B._acc

pytestmark = pytest.mark.skipif(
    not B._HAVE_BSPLINE_WINDOWED_ASSEMBLE_ACCEL,
    reason="needs the C++ windowed assembler",
)

OMEGA = 2 * np.pi * 7e6
EPS0 = 8.854187817e-12
MU0 = 4e-7 * np.pi


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def test_the_lane_kernel_is_compiled_into_the_avx2_variant():
    assert hasattr(acc, "windowed_lanes_1290"), (
        "the .so predates momwire#1290: rebuild (`make build`)"
    )
    if _accel.VARIANT == "avx2":
        assert acc.windowed_lanes_1290


def _case(seed, d, n_basis=45, n_segs=60, window=(5, 50, 3, 55)):
    """A synthetic window. Most bases sit on consecutive segments, as on a
    wire, so their lane groups read J as one run; a few are moved to
    scattered segments and a few given a wing at -1 (inactive), so other
    groups stage lane by lane and carry dead lanes."""
    rng = np.random.default_rng(seed)
    nm = d + 1
    i0, i1, j0, j1 = window
    base = np.arange(n_basis)[:, None] + np.arange(nm)[None, :]
    supp = base % n_segs
    scatter = rng.choice(n_basis, size=6, replace=False)
    supp[scatter] = rng.integers(0, n_segs, size=(6, nm))
    supp[rng.choice(n_basis, size=4, replace=False), 0] = -1
    supp = supp.astype(np.int64)
    # Coefficients over many decades make any reassociation visible.
    polys = rng.standard_normal((n_basis, nm, nm)) * 10.0 ** rng.uniform(
        -3, 3, size=(n_basis, nm, nm)
    )
    tang = rng.standard_normal((n_segs, 3))
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    shape = (nm, nm, i1 - i0, j1 - j0)
    J = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * 10.0 ** (
        rng.uniform(-6, 0, size=shape)
    )
    m_idx = np.nonzero(((supp >= i0) & (supp < i1)).any(axis=1))[0]
    # Columns: a sorted run (contiguous groups), then a shuffled tail whose
    # length leaves the last group short of eight.
    n_idx = np.arange(n_basis)
    tail = rng.permutation(n_idx[29:])
    n_idx = np.concatenate([n_idx[:29], tail]).astype(np.int64)
    Z0 = rng.standard_normal((n_basis, n_basis)) + 1j * rng.standard_normal(
        (n_basis, n_basis)
    )
    return dict(
        J=np.ascontiguousarray(J),
        supp=supp,
        polys=np.ascontiguousarray(polys),
        tang=np.ascontiguousarray(tang),
        m_idx=m_idx.astype(np.int64),
        n_idx=n_idx,
        window=window,
        Z0=np.asfortranarray(Z0),
    )


def _run(c, reference, *, cplx=False, row_of=None):
    i0, i1, j0, j1 = c["window"]
    if row_of is None:
        Z = c["Z0"].copy(order="F")
        kw = {}
    else:
        Z = np.asfortranarray(c["Z0"][: row_of.max() + 1])
        kw = {"row_of": row_of}
    args = (c["J"], c["supp"], c["polys"], c["tang"], c["m_idx"], c["n_idx"])
    win = (i0, i1, j0, j1)
    if cplx:
        eps = EPS0 * (13.0 - 2.5j)
        acc.assemble_Z_bspline_windowed_cplx_eps(
            *args, *win, OMEGA, eps, MU0, Z, reference=reference, **kw
        )
    else:
        acc.assemble_Z_bspline_windowed(
            *args, *win, OMEGA, EPS0, MU0, Z, reference=reference, **kw
        )
    return Z


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("cplx", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_lanes_are_the_reference_to_the_bit(d, cplx, seed):
    c = _case(seed, d)
    want = _run(c, True, cplx=cplx)
    got = _run(c, False, cplx=cplx)
    assert not np.array_equal(want, c["Z0"])  # the window added something
    assert_lanes_match(got, want)


@pytest.mark.parametrize("cplx", [False, True])
def test_lanes_into_a_row_compact_z(cplx):
    c = _case(2, 2)
    n_basis = c["supp"].shape[0]
    row_of = np.full(n_basis, -1, dtype=np.int64)
    row_of[c["m_idx"]] = np.arange(c["m_idx"].size)[::-1]
    want = _run(c, True, cplx=cplx, row_of=row_of)
    got = _run(c, False, cplx=cplx, row_of=row_of)
    assert_lanes_match(got, want)


@pytest.mark.parametrize(
    "window",
    [
        (0, 60, 0, 60),  # the whole mesh: every group a contiguous run
        (40, 41, 7, 9),  # one row, a two-column window: mostly dead lanes
        (5, 50, 20, 20),  # no columns: every group dead, nothing to stage
    ],
)
def test_lanes_on_edge_windows(window):
    c = _case(3, 2, window=window)
    want = _run(c, True)
    got = _run(c, False)
    assert_lanes_match(got, want)


def test_the_gate_sees_one_ulp():
    c = _case(4, 2)
    want = _run(c, True)
    # One ulp up on every moment: the entries the window wrote must move.
    J = np.nextafter(c["J"].real, np.inf) + 1j * c["J"].imag
    got = _run(dict(c, J=J), False)
    moved = _bits(got) != _bits(want)
    assert moved.any(axis=1)[c["m_idx"]].mean() > 0.9


def _counting(monkeypatch, reference):
    """Route both windowed assemblers through `reference` and count calls."""
    calls = {"n": 0}
    for name in ("assemble_Z_bspline_windowed", "assemble_Z_bspline_windowed_cplx_eps"):
        f = functools.partial(getattr(acc, name), reference=reference)

        def counted(*a, _f=f, **k):
            calls["n"] += 1
            return _f(*a, **k)

        monkeypatch.setattr(acc, name, counted)
    return calls


def _solve(make, monkeypatch, reference):
    with monkeypatch.context() as mp:
        calls = _counting(mp, reference)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z, cur = make().compute_impedance()
    return np.atleast_1d(np.asarray(z)), np.asarray(cur), calls["n"]


def _free_array():
    """Three parallel dipoles and a bent wire, with `swept_mem_mb=1` so the
    (3, 3, N, N) tensor overflows the budget and the chunked fill runs."""
    wires = [np.array([(-5.0, 4.0 * j, 10.0), (5.0, 4.0 * j, 10.0)]) for j in range(3)]
    wires.append(np.array([(-5.0, -4.0, 10.0), (0.0, -4.0, 12.0), (5.0, -4.0, 10.0)]))
    return B.BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[41], [41], [41], [20, 21]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=299792458.0 / 14e6,
        wire_radius=0.001,
        swept_mem_mb=1,
    )


def test_a_free_space_solve_is_the_reference_to_the_bit(monkeypatch):
    z_ref, c_ref, n_ref = _solve(_free_array, monkeypatch, True)
    z, c, n = _solve(_free_array, monkeypatch, False)
    assert n_ref == n > 1  # the chunked fill ran, window by window
    assert_lanes_match(z, z_ref)
    assert_lanes_match(c, c_ref)


def test_a_buried_solve_is_the_reference_to_the_bit(monkeypatch):
    from test_rotational_symmetry_1029 import solver

    def make():
        return solver(4, rotational_symmetry=False)

    z_ref, c_ref, n_ref = _solve(make, monkeypatch, True)
    z, c, n = _solve(make, monkeypatch, False)
    assert n_ref == n > 0
    assert_lanes_match(z, z_ref)
    assert_lanes_match(c, c_ref)
