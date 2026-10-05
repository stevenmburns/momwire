"""SG's band scatter in C++ — momwire#1290 (`_accel_row_scatter.cpp`).

`_row_scatter` hands `_scatter_band`'s and the extended-kernel bracket's
`np.add.at` to the accelerator's `ordered_row_scatter`, which adds each
destination row's entries one at a time in ascending entry order, reading
the source rows in place, and takes a `fresh` row as +0 without reading it.
Plain elementwise adds in `np.add.at`'s order, so the claim is the bit on
every platform, by construction rather than by matching a library.

Gates:

  * the kernel equals `np.add.at` AND the numpy reference
    (`_ordered_row_scatter`) to the bit, signed zeros included: repeated
    destinations in shuffled order, carried (non-zero) rows, junction-sized
    multiplicities, and `fresh` rows that hold NaN garbage beforehand;
  * rows that are views into a wider array (row stride != width);
  * what it cannot take (1-D, column-strided rows, real dtype) goes to the
    reference, which still honours `fresh`;
  * end to end, SG's Z and currents through the production seam with the
    kernel on equal the reference route (`_ROW_SCATTER_ACCEL = False`)
    byte for byte, on bands thin enough to carry bases across boundaries,
    and spies count that each route actually ran.

The fused fill against the whole-triple product, through this kernel, is
`tests/test_sg_fused_banded_fill_1224.py`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import sinusoidal_galerkin as sg
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

from test_sg_fused_banded_fill_1224 import _bands_every, dipole, ell
from test_sg_ordered_scatter_1290 import _case


def _direct_sg(*args, **kw):
    """This module's subject is the DIRECT-field fill's own machinery, so
    every solver here names it (`SinusoidalGalerkinSolver`'s default fill is
    the mixed-potential one since momwire#1354, gated in
    `test_sinusoidal_mp_1354.py`)."""
    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver as _cls

    kw.setdefault("fill", "direct")
    return _cls(*args, **kw)


pytestmark = pytest.mark.skipif(
    not sg._HAVE_ROW_SCATTER_ACCEL,
    reason="the accelerator does not carry ordered_row_scatter",
)


def _bits_equal(a, b):
    return np.array_equal(a.view(np.float64), b.view(np.float64)) and np.array_equal(
        np.signbit(a.view(np.float64)), np.signbit(b.view(np.float64))
    )


@pytest.mark.parametrize(
    "seed,n_dest,n_entries,width,max_mult",
    [
        (0, 50, 400, 33, None),
        (1, 200, 200, 64, 2),
        (2, 10, 300, 17, 40),
        (4, 1, 30, 5, None),
        (8, 65, 191, 2816, 3),
    ],
)
def test_kernel_is_add_at_to_the_bit(seed, n_dest, n_entries, width, max_mult):
    dest, idx, src = _case(seed, n_dest, n_entries, width, max_mult)
    want = dest.copy()
    np.add.at(want, idx, src)
    ref = dest.copy()
    sg._ordered_row_scatter(ref, idx, src)
    got = dest.copy()
    sg._acc.ordered_row_scatter(got, idx, src, np.zeros(0, dtype=bool))
    assert _bits_equal(got, want)
    assert _bits_equal(got, ref)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fresh_rows_start_from_positive_zero(seed):
    """A fresh row is `np.zeros` then `+=`: +0 + -0 is +0, so a row whose
    only entry is -0 must come out +0 — and whatever dest held there (NaN
    here) is never read."""
    dest, idx, src = _case(seed, 40, 120, 9, 3)
    rng = np.random.default_rng(seed)
    fresh = rng.random(dest.shape[0]) < 0.6
    # A fresh row receiving only -0.0: the signed zero must not survive.
    lone = int(np.flatnonzero(fresh)[0])
    src[idx == lone] = -0.0 - 0.0j
    want = dest.copy()
    want[fresh] = 0
    np.add.at(want, idx, src)
    got = dest.copy()
    got[fresh] = np.nan
    sg._acc.ordered_row_scatter(got, idx, src, fresh)
    assert _bits_equal(got, want)
    assert not np.signbit(got[lone].real).any()
    # A fresh row no entry reaches is +0, not left as garbage.
    idx_gap = idx[idx != lone]
    src_gap = src[idx != lone]
    got = dest.copy()
    got[fresh] = np.nan
    sg._acc.ordered_row_scatter(got, idx_gap, src_gap, fresh)
    assert np.array_equal(got[lone], np.zeros(dest.shape[1], dtype=np.complex128))
    assert not np.signbit(got[lone].real).any()


def test_rows_inside_a_wider_array():
    dest, idx, src = _case(9, 30, 200, 7, None)
    wide = np.zeros((30, 12), dtype=np.complex128)
    wide[:, 3:10] = dest
    want = dest.copy()
    np.add.at(want, idx, src)
    big = np.zeros((src.shape[0], 20), dtype=np.complex128)
    big[:, 5:12] = src
    sg._row_scatter(wide[:, 3:10], idx, big[:, 5:12])
    assert _bits_equal(wide[:, 3:10], want)
    assert not wide[:, :3].any() and not wide[:, 10:].any()


def test_the_gate_sees_order():
    """Agreement is evidence about the order: the same adds in reverse
    entry order do not reproduce the kernel."""
    dest, idx, src = _case(6, 20, 400, 9, None)
    got = dest.copy()
    sg._acc.ordered_row_scatter(got, idx, src, np.zeros(0, dtype=bool))
    rev = dest.copy()
    np.add.at(rev, idx[::-1], src[::-1])
    assert not np.array_equal(rev, got)


def test_out_of_range_index_is_refused():
    dest = np.zeros((3, 2), dtype=np.complex128)
    src = np.ones((1, 2), dtype=np.complex128)
    with pytest.raises(RuntimeError, match="out of range"):
        sg._acc.ordered_row_scatter(dest, np.array([3]), src, np.zeros(0, bool))


def _counting(monkeypatch):
    """Count the kernel's and the reference's calls through `sg`'s names."""
    counts = {"accel": 0, "ref": 0}
    acc, real_ref = sg._acc, sg._ordered_row_scatter

    class _Acc:
        def __getattr__(self, name):
            return getattr(acc, name)

        def ordered_row_scatter(self, *a, **kw):
            counts["accel"] += 1
            return acc.ordered_row_scatter(*a, **kw)

    def ref(*a, **kw):
        counts["ref"] += 1
        return real_ref(*a, **kw)

    monkeypatch.setattr(sg, "_acc", _Acc())
    monkeypatch.setattr(sg, "_ordered_row_scatter", ref)
    return counts


@pytest.mark.parametrize(
    "dest_shape,src_view",
    [
        ((10,), lambda s: s[:, 0].copy()),  # 1-D
        ((10, 4), lambda s: s[:, ::2]),  # column-strided rows
    ],
)
def test_what_the_kernel_cannot_take_goes_to_the_reference(
    dest_shape, src_view, monkeypatch
):
    rng = np.random.default_rng(3)
    idx = rng.integers(0, 10, 50)
    src = src_view(rng.standard_normal((50, 8)) + 1j * rng.standard_normal((50, 8)))
    dest = np.full(dest_shape, np.nan, dtype=np.complex128)
    fresh = np.ones(10, dtype=bool)
    want = np.zeros(dest_shape, dtype=np.complex128)
    np.add.at(want, idx, src)
    counts = _counting(monkeypatch)
    sg._row_scatter(dest, idx, src, fresh)
    assert counts == {"accel": 0, "ref": 1}
    assert _bits_equal(dest, want)


DECKS = {
    "free-dipole": lambda: _direct_sg(**dipole()),
    "ek-ell": lambda: _direct_sg(**ell(extended_kernel=True)),
}


def _solve(name, monkeypatch, accel, step):
    with monkeypatch.context() as m:
        m.setattr(sg, "_BAND_MIN_SEGMENTS", 1)
        m.setattr(SinusoidalGalerkinSolver, "_segment_bands", _bands_every(step))
        m.setattr(sg, "_ROW_SCATTER_ACCEL", accel)
        counts = _counting(m)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z, cur = DECKS[name]().compute_impedance()
    return np.asarray(z), np.concatenate([np.ravel(c) for c in cur]), counts


@pytest.mark.parametrize("name", list(DECKS))
@pytest.mark.parametrize("step", [2, 7])
def test_solve_is_the_reference_route_to_the_bit(name, step, monkeypatch):
    z_ref, i_ref, c_ref = _solve(name, monkeypatch, False, step)
    z_new, i_new, c_new = _solve(name, monkeypatch, True, step)
    assert c_ref["accel"] == 0 and c_ref["ref"] > 0
    assert c_new["ref"] == 0 and c_new["accel"] == c_ref["ref"]
    assert np.array_equal(z_new, z_ref)
    assert np.array_equal(i_new, i_ref)
