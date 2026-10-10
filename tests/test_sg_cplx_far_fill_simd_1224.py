"""The complex-k Galerkin far fill on a vector exp/sin sweep — momwire#1224.

`galerkin_far_fill_cplx_impl` evaluated complex `std::exp`/`std::sin` per
(observer, source) pair and per source node, scalar. It is now staged like the
real fill: a flat table of real distances d, one `omp simd` sweep onto
libmvec's exp and sin (`sg_cplx_phase_sweep`, its own TU), and every complex
transcendental spelled from exp(Im k d), expm1(Im k d), sin(Re k d) and
sin(Re k d / 2). That is a numerics change, not a refactor, so it is pinned
here three ways:

  * the sweep against scalar references, per entry (as
    test_razor_cplx_brackets pins Design D4's);
  * G and Z against a STORED reference made by the scalar spelling this
    replaced (`tests/data/sg_cplx_far_fill_ref_1224.npz`, written by a0026e4
    on Haswell, GCC 11.4, AVX2), on a wholly buried dipole, the smallest
    crossing deck and a four-radial hub. Measured on that box, as G's worst
    row ||dG_i||/||G_i|| and |dZ|/|Z|:

        build              buried-dipole       crossing            hub4
        a0026e4 sse2       2.8e-18 / 0         1.0e-14 / 6.1e-14   1.5e-15 / 2.5e-13
        this    avx2       8.6e-17 / 0         5.1e-16 / 2.5e-15   7.8e-16 / 2.7e-13
        this    sse2       4.3e-17 / 7.1e-17   1.0e-14 / 6.1e-14   1.5e-15 / 2.2e-13

    so the change sits at or under the scalar spelling's own AVX2-vs-SSE2
    spread. The bars, G_ROW_BAR = 1e-12 and Z_BAR = 1e-11, are ~100x and ~40x
    over the largest of those, because the reference is replayed on every
    toolchain CI builds (clang and Apple's libm on macOS among them) and
    per-toolchain bit equality is not claimed. A wrong term in any spelling
    misses them by orders of magnitude. G is measured by rows, not elements:
    a far entry near a zero of the kernel carries a large ELEMENT-wise
    relative difference (up to 1e-11 here) that is an absolute ~1e-16;
  * column-independence: the sweep pads its table to the vector width, so
    no entry takes the loop's scalar tail and the complex fill needs none of
    `_SIMD_TAIL_PERIOD`'s source padding. Every padding gives the whole
    fill's columns to the bit — and the red control shows that without the
    sweep's own padding the tail DOES move bits, on the build where the
    vector stage fires (linux, AVX2), so the padding is load-bearing rather
    than vacuous.
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from test_crossing_serve_524 import hub_deck
from test_sg_mixed_class_fill_1224 import _kernel_columns

import momwire._accel as _accel
from momwire import sinusoidal_galerkin as sg


def _direct_sg(*args, **kw):
    """This module's subject is the DIRECT-field fill's own machinery, so
    every solver here names it (`SinusoidalGalerkinSolver`'s default fill is
    the mixed-potential one since momwire#1354, gated in
    `test_sinusoidal_mp_1354.py`)."""
    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver as _cls

    kw.setdefault("fill", "direct")
    return _cls(*args, **kw)


pytestmark = pytest.mark.skipif(
    not sg._HAVE_GALERKIN_FAR_FILL_CPLX
    or not hasattr(_accel.acc, "sg_cplx_phase_sweep"),
    reason="no compiled complex far fill with the vector sweep in this build",
)

_REF = Path(__file__).with_name("data") / "sg_cplx_far_fill_ref_1224.npz"
G_ROW_BAR = 1e-12
Z_BAR = 1e-11
SWEEP_BAR = 1e-13

C0 = 299792458.0
WL7 = C0 / 7e6
SOMM = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")

# Where the vector stage exists to be observed: glibc libmvec at AVX2.
_VECTOR_BUILD = sys.platform == "linux" and _accel.VARIANT == "avx2"


# ---------------------------------------------------------------------------
# The decks and capture the stored reference was written with. Kept verbatim
# (scratch/1224-cplx-simd/refdecks.py in antennaknobs made the file): an edit
# here invalidates the reference rather than the code under test.


def _buried_dipole():
    return dict(
        wires=[np.array([(-10.0, 0.0, -0.5), (10.0, 0.0, -0.5)])],
        n_per_edge_per_wire=[[41]],
        feeds=[(0, 10.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        **SOMM,
    )


def _crossing():
    """test_sg_crossing_980_d3's deck: one wire rising from the interface,
    one leaving it below, joined in the plane."""
    return dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 2.0)]),
            np.array([(0.0, 0.0, 0.0), (2.0, 0.0, -0.5)]),
        ],
        n_per_edge_per_wire=[[15], [15]],
        feeds=[(0, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        junctions=[[(0, "start"), (1, "start")]],
        **SOMM,
    )


def _hub4():
    """Four buried radials to a hub, a rise through the interface and a
    monopole: `hub_deck(n_radials=4)` of test_crossing_serve_524."""
    dirs = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
    wires = [
        np.array([(5.0 * dx, 5.0 * dy, -0.15), (0.0, 0.0, -0.15)]) for dx, dy in dirs
    ]
    npe = [[10] for _ in dirs]
    wires.append(np.array([(0.0, 0.0, -0.15), (0.0, 0.0, 0.0)]))
    npe.append([2])
    wires.append(np.array([(0.0, 0.0, 10.0), (0.0, 0.0, 0.0)]))
    npe.append([15])
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[
            [(i, "end") for i in range(4)] + [(4, "start")],
            [(4, "end"), (5, "end")],
        ],
        feeds=[(5, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=1e-3,
        **SOMM,
    )


DECKS = {"buried-dipole": _buried_dipole, "crossing": _crossing, "hub4": _hub4}


def _capture(make):
    """(G, Z, complex-fill calls) through the production seam: G is copied at
    `_solve_in_place`, before the in-place solve overwrites it."""
    seen = {}
    solve = sg._solve_in_place
    fill = _accel.acc.sinusoidal_galerkin_far_fill_cplx
    calls = [0]

    def spy_solve(G, rhs):
        seen.setdefault("G", np.array(G, copy=True))
        return solve(G, rhs)

    def spy_fill(*a, **k):
        calls[0] += 1
        return fill(*a, **k)

    sg._solve_in_place = spy_solve
    _accel.acc.sinusoidal_galerkin_far_fill_cplx = spy_fill
    try:
        Z, _ = _direct_sg(**make()).compute_impedance()
    finally:
        sg._solve_in_place = solve
        _accel.acc.sinusoidal_galerkin_far_fill_cplx = fill
    return seen["G"], complex(np.atleast_1d(Z)[0]), calls[0]


# ---------------------------------------------------------------------------
# The sweep, entry by entry.


def _sweep_grid():
    """(d, k) pairs putting a = Im(k) d across the expm1 switch at |a| = 0.35
    from both sides and both signs, down to a -> 0, out to deep decay, with
    y = Re(k) d from ~0 to many turns."""
    near = 0.35 + np.linspace(-0.02, 0.02, 401)
    ulps = np.array([np.nextafter(0.35, 0.0), 0.35, np.nextafter(0.35, 1.0)])
    small = np.logspace(-12, np.log10(0.35), 300)
    large = np.logspace(np.log10(0.35), np.log10(700.0), 300)
    a = np.concatenate([near, ulps, small, large, [0.0]])
    return np.concatenate([a, -a])


@pytest.mark.parametrize("k_re", [1e-6, 0.3, 0.58, 2.1, 40.0])
def test_the_sweep_matches_scalar_libm(k_re):
    k_im = -0.238
    a = _sweep_grid()
    d = a / k_im
    ea, em, s, h = _accel.acc.sg_cplx_phase_sweep(d, complex(k_re, k_im))
    a_exact = k_im * d
    y = k_re * d
    for got, ref, what in (
        (ea, np.exp(a_exact), "exp"),
        (em, np.expm1(a_exact), "expm1"),
        (s, np.array([math.sin(v) for v in y]), "sin"),
        (h, np.array([math.sin(0.5 * v) for v in y]), "sin(y/2)"),
    ):
        scale = np.where(ref != 0.0, np.abs(ref), 1.0)
        rel = np.abs(got - ref) / scale
        worst = int(np.argmax(rel))
        assert rel[worst] < SWEEP_BAR, (
            f"{what}: {rel[worst]:.3e} at a={a_exact[worst]!r}, y={y[worst]!r}"
        )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 9, 64, 65])
def test_the_sweep_is_position_independent(n):
    """With the sweep's own padding, an entry's bits depend on its distance
    alone: the same distances at every offset in a longer table give the same
    values. This is what makes the complex fill column-independent."""
    rng = np.random.default_rng(1224 + n)
    d = rng.uniform(-30.0, 30.0, n)
    k = complex(0.58, -0.238)
    alone = _accel.acc.sg_cplx_phase_sweep(d, k)
    for off in range(1, 8):
        padded = np.concatenate([rng.uniform(0.1, 9.0, off), d])
        shifted = _accel.acc.sg_cplx_phase_sweep(padded, k)
        for x, y in zip(alone, shifted):
            assert np.array_equal(x, y[off:]), f"offset {off} moved bits"


def test_the_sweeps_own_padding_is_load_bearing():
    """The red control. Without the padding (`pad=False`) the last n mod 4
    entries take the loop's scalar tail — scalar libm, whose last bits are
    not libmvec's — so the same distances read differently in the tail than
    in the vector body. Asserted where the vector stage is known to exist
    (linux, AVX2, libmvec). Elsewhere nothing is asserted: the sweep may be
    scalar throughout (the SSE2 baseline) or the platform's own vectoriser's
    (MSVC), and neither is this file's claim."""
    rng = np.random.default_rng(1224)
    k = complex(0.58, -0.238)
    moved = 0
    for _ in range(64):
        d = rng.uniform(-30.0, 30.0, 3)
        body = _accel.acc.sg_cplx_phase_sweep(np.concatenate([d, d[:1]]), k)
        tail = _accel.acc.sg_cplx_phase_sweep(d, k, pad=False)
        moved += any(not np.array_equal(b[:3], t) for b, t in zip(body, tail))
    if _VECTOR_BUILD:
        assert moved > 0, "the scalar tail never differed: is the sweep vectorised?"


# ---------------------------------------------------------------------------
# The fill.


@pytest.mark.parametrize(
    "name",
    [
        # The buried dipole is the slow deck (median 5.5 s on Linux CI); the
        # crossing and the hub keep the complex fill on every PR.
        pytest.param(n, marks=[pytest.mark.slow] if n == "buried-dipole" else [])
        for n in sorted(DECKS)
    ],
)
def test_g_and_z_match_the_scalar_spelling(name):
    ref = np.load(_REF)
    G, Z, calls = _capture(DECKS[name])
    assert calls > 0, "the complex far fill never ran"
    G_ref, Z_ref = ref[f"{name}:G"], complex(ref[f"{name}:Z"][0])
    assert G.shape == G_ref.shape
    row = np.linalg.norm(G - G_ref, axis=1) / np.linalg.norm(G_ref, axis=1)
    assert row.max() < G_ROW_BAR, f"{name}: G worst row {row.max():.3e}"
    z_rel = abs(Z - Z_ref) / abs(Z_ref)
    assert z_rel < Z_BAR, f"{name}: Z {z_rel:.3e}"


def test_the_complex_kernel_needs_no_padding():
    """`test_the_real_kernel_needs_the_padding`'s question asked of the
    below class: hub16's geometry at k_m, random column subsets with and
    without the deck's last segment, under EVERY padding 0-3. The real
    kernel is bit-equal only under `_class_view`'s rule; the complex one is
    bit-equal under all four, because its sweep pads itself."""
    s = _direct_sg(**hub_deck(n_radials=16))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        geom = s._build_geometry()
        below = s._below_segments(geom)
        medium = s._fill_medium(geom)
        view = s._stitch_basis_coefs(geom, below, medium.k_p, medium.k_m)
        view = s._crossing_wing_view(geom, view, below, medium)
        ctx = s._test_context(geom, view, medium.k_m)
    k, eta = medium.k_m, s._medium_eta(medium)
    assert np.iscomplexobj(k) and k.imag < 0.0
    n = int(geom["n_segs"])
    fill = _accel.acc.sinusoidal_galerkin_far_fill_cplx
    calls = [0]

    def spy(*a, **kw):
        calls[0] += 1
        return fill(*a, **kw)

    _accel.acc.sinusoidal_galerkin_far_fill_cplx = spy
    try:
        whole = s._far_fill_accel(
            k, ctx, geom["seg_centers"], geom["seg_tangents"], eta=eta
        )
        rng = np.random.default_rng(1224)
        for trial in range(8):
            cols = np.sort(rng.choice(n, int(rng.integers(3, n)), replace=False))
            if trial % 2:
                cols = np.unique(np.append(cols, n - 1))
            for pad in range(4):
                got = _kernel_columns(s, ctx, geom, k, eta, cols, pad)
                for w, g in zip(whole, got):
                    assert np.array_equal(w[:, cols], g), (
                        f"trial {trial}, pad {pad}: "
                        f"{int((w[:, cols] != g).sum())} cells differ"
                    )
    finally:
        _accel.acc.sinusoidal_galerkin_far_fill_cplx = fill
    assert calls[0] == 1 + 8 * 4, f"the complex entry point ran {calls[0]} times"
