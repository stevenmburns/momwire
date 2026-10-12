"""The same-edge correction's glue in parallel C++ (momwire#1441 step 0a).

At 8 threads half of a bs2 free-space solve ran on one thread, mostly in
three elementwise numpy steps of the windowed same-edge correction. They are
now C++ with OpenMP, numpy's own arithmetic in numpy's order, so the gates
are uint64 against the numpy spellings (`_SAME_EDGE_GLUE_ACCEL = False`):

- G-1441-0a-1  the Toeplitz gather: full, windowed, single-row and empty.
- G-1441-0a-2  the regular part's distance table, square and windowed, on a
               non-uniform edge.
- G-1441-0a-3  the combine, signed zeros included (numpy's `x + 0j`
               promotion makes `0.0 + -0.0`, which is `+0.0`).
- G-1441-0a-4  end to end, a bs2 dipole's Z with and without the glue, each
               kernel counted as run, so an agreement cannot be two numpy runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire import _bspline_kernels as bk

pytestmark = pytest.mark.skipif(
    not bk._HAVE_SAME_EDGE_GLUE_ACCEL, reason="accelerator without the #1441 kernels"
)


def _bits(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint64) if a.dtype == np.float64 else a.view(np.uint64)


def _both(monkeypatch, fn):
    monkeypatch.setattr(bk, "_SAME_EDGE_GLUE_ACCEL", True)
    got = fn()
    monkeypatch.setattr(bk, "_SAME_EDGE_GLUE_ACCEL", False)
    ref = fn()
    return got, ref


@pytest.mark.parametrize(
    ("N", "r0", "r1"), [(7, 0, 7), (40, 9, 31), (40, 39, 40), (40, 12, 12), (1, 0, 1)]
)
def test_g1441_0a_1_gather_is_numpys(N, r0, r1, monkeypatch):
    rng = np.random.default_rng(N + r0)
    table = rng.standard_normal((3, 3, 2 * N - 1))
    got, ref = _both(monkeypatch, lambda: bk._toeplitz_gather(table, N, r0, r1))
    assert got.shape == ref.shape == (3, 3, r1 - r0, N)
    assert np.array_equal(_bits(got), _bits(ref))


@pytest.mark.parametrize("rows", [None, slice(5, 23), slice(0, 1)])
def test_g1441_0a_2_distance_table_is_numpys(rows, monkeypatch):
    n_seg = 37
    ends = np.cumsum(np.r_[0.0, 0.05 + 0.01 * np.sin(np.arange(n_seg))])
    got, ref = _both(
        monkeypatch, lambda: bk._seg_seg_reg_geometry(ends, 1.3e-3, 2, 6, rows=rows)
    )
    assert np.array_equal(_bits(got["R"]), _bits(ref["R"]))
    assert np.array_equal(_bits(got["wu_pow"]), _bits(ref["wu_pow"]))


def test_g1441_0a_3_combine_is_numpys_signed_zeros_included(monkeypatch):
    rng = np.random.default_rng(1441)
    shape = (3, 3, 11, 29)
    A_st = rng.standard_normal(shape)
    A_reg = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    J = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    # Exact zeros of both signs where the promotion's `0.0 +` decides the sign.
    A_reg.imag[0, 0, :3, :3] = -0.0
    J.imag[0, 0, :3, :3] = 0.0
    A_st[1, 1, :2, :2] = -0.0
    A_reg.real[1, 1, :2, :2] = 0.0
    got, ref = _both(monkeypatch, lambda: bk._same_edge_corr(A_st, A_reg, J))
    assert got.dtype == ref.dtype == np.complex128
    assert np.array_equal(_bits(got), _bits(ref))


class _Counting:
    """The accelerator module, forwarding everything, counting three calls."""

    NAMES = ("toeplitz_gather_rows", "reg_distance_table", "same_edge_corr_combine")

    def __init__(self, acc):
        self._acc = acc
        self.calls = dict.fromkeys(self.NAMES, 0)

    def __getattr__(self, name):
        fn = getattr(self._acc, name)
        if name not in self.NAMES:
            return fn

        def counted(*a, **kw):
            self.calls[name] += 1
            return fn(*a, **kw)

        return counted


def _dipole_z(n_seg, swept_mem_mb):
    wl = 299792458.0 / 14.2e6
    half = 0.48 * wl / 2
    solver = BSplineSolver(
        wires=[np.array([(-half, 0.0, 10.0), (half, 0.0, 10.0)])],
        n_per_edge_per_wire=[[n_seg]],
        feeds=[(0, half, 1 + 0j)],
        wavelength=wl,
        wire_radius=1e-3,
        # A small budget windows the same-edge correction; at the default a
        # 401-segment edge is one whole block and the windowed route (where
        # the gather and the combine live) never runs.
        swept_mem_mb=swept_mem_mb,
    )
    z = solver.compute_impedance()
    return np.asarray(z[0] if isinstance(z, tuple) else z)


@pytest.mark.parametrize("swept_mem_mb", [1, 4])
def test_g1441_0a_4_a_dipoles_z_is_bit_identical_and_the_kernels_ran(
    swept_mem_mb, monkeypatch
):
    counting = _Counting(bk._acc)
    monkeypatch.setattr(bk, "_acc", counting)
    got, ref = _both(monkeypatch, lambda: _dipole_z(401, swept_mem_mb))
    assert np.array_equal(_bits(got), _bits(ref))
    # The C++ ran on the first solve: what agrees is C++ against numpy.
    assert all(n > 0 for n in counting.calls.values()), counting.calls
