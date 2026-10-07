"""The WEIGHTED windowed assembler's AVX2 lane kernel.

`assemble_Z_bspline_weighted_windowed` (and its in-medium twin) subtract the
ground image from Z, window by window, on every grounded deck the chunked fill
serves -- the above-ground Sommerfeld deck's exact-image sweep among them. On
the AVX2 build it now runs the unweighted twin's lane kernel (momwire#1290)
carrying the per-pair complex weights: eight entries a step over cache tiles,
each entry's arithmetic operation for operation the per-entry loop's.
`reference=True` still runs that loop, so the gates here are bit equality
against it, as in `test_windowed_lanes_1290.py`:

* kernel level, on synthetic windows that reach every lane case (contiguous
  and staged groups, dead and all-dead lanes, a short tail group, edge
  windows, d = 1, 2 and 3, the complex-eps twin, the row-compact target) with
  weights and a scale whose parts span many decades;
* NEGATIVE CONTROLS: one ulp on the moments, and separately on the weights,
  moves the rows the window wrote;
* end to end, grounded decks forced through the chunked fill (Sommerfeld,
  PEC, refl-coef): Z_in and currents equal the reference's bit for bit, with
  the calls COUNTED so the gate cannot pass without the assembler running.

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

from momwire import bspline as B
from _lane_gate import assert_lanes_match

acc = B._acc

pytestmark = pytest.mark.skipif(
    not B._HAVE_BSPLINE_W_WINDOWED_ASSEMBLE_ACCEL,
    reason="needs the C++ weighted windowed assembler",
)

OMEGA = 2 * np.pi * 7e6
EPS0 = 8.854187817e-12
MU0 = 4e-7 * np.pi
NAMES = (
    "assemble_Z_bspline_weighted_windowed",
    "assemble_Z_bspline_weighted_windowed_cplx_eps",
)


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _cplx(rng, shape, lo, hi):
    mag = 10.0 ** rng.uniform(lo, hi, size=shape)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * mag


def _case(seed, d, n_basis=45, n_segs=60, window=(5, 50, 3, 55)):
    """A synthetic window, as the unweighted twin's gate builds it, plus
    the two complex weight windows."""
    rng = np.random.default_rng(seed)
    nm = d + 1
    i0, i1, j0, j1 = window
    base = np.arange(n_basis)[:, None] + np.arange(nm)[None, :]
    supp = base % n_segs
    scatter = rng.choice(n_basis, size=6, replace=False)
    supp[scatter] = rng.integers(0, n_segs, size=(6, nm))
    supp[rng.choice(n_basis, size=4, replace=False), 0] = -1
    supp = supp.astype(np.int64)
    polys = rng.standard_normal((n_basis, nm, nm)) * 10.0 ** rng.uniform(
        -3, 3, size=(n_basis, nm, nm)
    )
    J = _cplx(rng, (nm, nm, i1 - i0, j1 - j0), -6, 0)
    wA = _cplx(rng, (i1 - i0, j1 - j0), -2, 2)
    wPhi = _cplx(rng, (i1 - i0, j1 - j0), -2, 2)
    m_idx = np.nonzero(((supp >= i0) & (supp < i1)).any(axis=1))[0]
    n_idx = np.arange(n_basis)
    tail = rng.permutation(n_idx[29:])
    n_idx = np.concatenate([n_idx[:29], tail]).astype(np.int64)
    Z0 = _cplx(rng, (n_basis, n_basis), 0, 0)
    return dict(
        J=np.ascontiguousarray(J),
        supp=supp,
        polys=np.ascontiguousarray(polys),
        wA=np.ascontiguousarray(wA),
        wPhi=np.ascontiguousarray(wPhi),
        m_idx=m_idx.astype(np.int64),
        n_idx=n_idx,
        window=window,
        scale=complex(-0.75, 0.3125),
        Z0=np.asfortranarray(Z0),
    )


def _run(c, reference, *, cplx=False, row_of=None):
    if row_of is None:
        Z = c["Z0"].copy(order="F")
        kw = {}
    else:
        Z = np.asfortranarray(c["Z0"][: row_of.max() + 1])
        kw = {"row_of": row_of}
    args = (c["J"], c["supp"], c["polys"], c["wA"], c["wPhi"], c["m_idx"], c["n_idx"])
    if cplx:
        eps = EPS0 * (13.0 - 2.5j)
        acc.assemble_Z_bspline_weighted_windowed_cplx_eps(
            *args, *c["window"], OMEGA, eps, MU0, c["scale"], Z,
            reference=reference, **kw,
        )  # fmt: skip
    else:
        acc.assemble_Z_bspline_weighted_windowed(
            *args, *c["window"], OMEGA, EPS0, MU0, c["scale"], Z,
            reference=reference, **kw,
        )  # fmt: skip
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


@pytest.mark.parametrize("which", ["J", "wA", "wPhi"])
def test_the_gate_sees_one_ulp(which):
    c = _case(4, 2)
    want = _run(c, True)
    x = c[which]
    bumped = np.nextafter(x.real, np.inf) + 1j * x.imag
    got = _run(dict(c, **{which: np.ascontiguousarray(bumped)}), False)
    moved = _bits(got) != _bits(want)
    assert moved.any(axis=1)[c["m_idx"]].mean() > 0.9


def _counting(monkeypatch, reference):
    calls = {"n": 0}
    for name in NAMES:
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


def _grounded_array(**ground):
    """Two parallel horizontal dipoles and a bent wire over ground, with
    `swept_mem_mb=1` so the chunked fill (and its image sweep) runs."""
    wires = [np.array([(-5.0, 4.0 * j, 6.0), (5.0, 4.0 * j, 6.0)]) for j in range(2)]
    wires.append(np.array([(-5.0, -4.0, 6.0), (0.0, -4.0, 8.0), (5.0, -4.0, 6.0)]))

    def make():
        return B.BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=[[41], [41], [20, 21]],
            feeds=[(0, 5.0, 1 + 0j)],
            wavelength=299792458.0 / 14e6,
            wire_radius=0.001,
            swept_mem_mb=1,
            ground_z=0.0,
            **ground,
        )

    return make


@pytest.mark.parametrize(
    "ground",
    [
        {"ground_eps": (13.0, 0.005), "ground_model": "sommerfeld"},
        {},  # PEC
        {"ground_eps": (13.0, 0.005), "ground_model": "refl-coef"},
    ],
    ids=["sommerfeld", "pec", "refl-coef"],
)
def test_a_grounded_solve_is_the_reference_to_the_bit(ground, monkeypatch):
    make = _grounded_array(**ground)
    z_ref, c_ref, n_ref = _solve(make, monkeypatch, True)
    z, c, n = _solve(make, monkeypatch, False)
    assert n_ref == n > 1  # the image sweep ran, window by window
    assert_lanes_match(z, z_ref)
    assert_lanes_match(c, c_ref)
