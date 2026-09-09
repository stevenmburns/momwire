"""momwire#980 step E: the complex-wavenumber far fill.

Step A left an in-medium k on the numpy path because the C++ fill's `k` is a
`double`. This adds `sinusoidal_galerkin_far_fill_cplx`, a SEPARATE body from
`sinusoidal_galerkin_far_fill` (momwire#991 records why, and the gate that
would retire the copy), and routes complex k to it.

The oracle is step A's: the numpy far fill at the same k_m. What that oracle
can show is agreement to reassociation, so the gate is stated against the
floor the REAL fill has against ITS OWN numpy path on the same decks — a twin
that agreed to 1e-16 would be as suspicious as one that agreed to 1e-6.
Measured 2026-09-09 on gcc 13 / AVX2:

    real  C++ vs numpy, real k:  worst 1.01e-14  (vertical-161)
    cplx  C++ vs numpy, k_m:     worst 2.94e-14  (vertical-161)

so the twin sits within 3x of the real path's own floor, on the same deck that
sets it. The gate below is 1e-12: two decades of headroom over the measured
value, because this number is toolchain-dependent and the PR lane does not
compile MSVC or clang. Per-toolchain agreement with numpy is the oracle;
cross-toolchain bit equality is NOT claimed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import momwire.sinusoidal_galerkin as sg
from momwire import (
    SinusoidalGalerkinSolver,
    _ground_refl,
    _sommerfeld_below,
)

SOIL_A = (13.0, 0.005)
C0 = 299792458.0
WL7 = C0 / 7e6

VERTICAL_41 = ([np.array([(0.0, 0.0, -1.15), (0.0, 0.0, -0.15)])], [[41]])
VERTICAL_161 = ([np.array([(0.0, 0.0, -1.15), (0.0, 0.0, -0.15)])], [[161]])
HORIZONTAL_6M = ([np.array([(-3.0, 0.0, -0.5), (3.0, 0.0, -0.5)])], [[81]])
L_BEND = (
    [
        np.array([(-1.0, 0.0, -0.5), (0.0, 0.0, -0.5)]),
        np.array([(0.0, 0.0, -0.5), (0.0, 0.0, -1.5)]),
    ],
    [[41], [41]],
)
T_JUNCTION = (
    [
        np.array([(-1.0, 0.0, -0.5), (1.0, 0.0, -0.5)]),
        np.array([(0.0, 0.0, -0.5), (0.0, 0.0, -1.5)]),
    ],
    [[41], [21]],
)
FEEDS = [(0, 0.5, 1 + 0j)]

DECKS = {
    "vertical-41": VERTICAL_41,
    "vertical-161": VERTICAL_161,
    "horizontal-6m": HORIZONTAL_6M,
    "L-bend": L_BEND,
    "T-junction": T_JUNCTION,
}

pytestmark = pytest.mark.skipif(
    not sg._HAVE_GALERKIN_FAR_FILL,
    reason="no compiled sinusoidal accelerator in this build",
)


def _solver(deck):
    wires, npe = deck
    return SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        feeds=FEEDS,
        wavelength=WL7,
        wire_radius=0.001,
    )


def _enter_medium(s):
    """The step-A test seam: k and eta overridden to the medium's."""
    eps_t = _ground_refl.eps_tilde(SOIL_A, s.omega, s.eps)
    s.k = _sommerfeld_below.k_medium(eps_t, s.k)
    s.eta = np.sqrt(s.mu / (s.eps * eps_t))
    assert np.iscomplexobj(s.k) and s.k.imag <= 0.0
    return s


def _impedance(deck, *, accel, medium):
    saved = sg._HAVE_GALERKIN_FAR_FILL
    sg._HAVE_GALERKIN_FAR_FILL = bool(accel) and saved
    try:
        s = _solver(deck)
        if medium:
            _enter_medium(s)
        with warnings.catch_warnings():
            # A `dtype=float` cast on a complex quantity is the
            # silent-truncation class step A exists to keep out.
            warnings.simplefilter("error", np.exceptions.ComplexWarning)
            return complex(s.compute_impedance()[0])
    finally:
        sg._HAVE_GALERKIN_FAR_FILL = saved


def test_the_complex_entry_point_is_in_this_build():
    """Guards the skip above from hiding a build that lost the symbol.

    Without this, every gate in the file would pass vacuously on a wheel
    whose twin failed to compile — the fill would fall back to numpy and
    "agrees with numpy" would be trivially true.
    """
    assert sg._HAVE_GALERKIN_FAR_FILL_CPLX


@pytest.mark.parametrize("name", sorted(DECKS))
def test_twin_agrees_with_the_numpy_oracle_at_k_m(name):
    deck = DECKS[name]
    z_accel = _impedance(deck, accel=True, medium=True)
    z_numpy = _impedance(deck, accel=False, medium=True)
    rel = abs(z_accel - z_numpy) / abs(z_numpy)
    assert rel < 1e-12, f"{name}: {rel:.3e}"
    # The medium is not free space — if k_m had been truncated to its real
    # part the answer would be the free-space one, which this separates.
    z_free = _impedance(deck, accel=True, medium=False)
    assert abs(z_accel - z_free) / abs(z_free) > 1e-3, (
        f"{name}: in-medium answer equals the free-space one"
    )


@pytest.mark.parametrize("name", sorted(DECKS))
def test_the_twin_is_actually_being_exercised(name):
    """The precondition every agreement gate above rests on.

    `_far_fill_accel` must have been called AND must have taken the complex
    entry point. Without this the suite passes if the dispatch silently
    routes the medium to numpy, which is exactly the pre-step-E behaviour.
    """
    calls = {"n": 0, "cplx": 0}
    real_fill = SinusoidalGalerkinSolver._far_fill_accel

    def spy(self, k, ctx, src_c, src_t, **kw):
        calls["n"] += 1
        if np.iscomplexobj(k) or np.iscomplexobj(self.eta):
            calls["cplx"] += 1
        return real_fill(self, k, ctx, src_c, src_t, **kw)

    SinusoidalGalerkinSolver._far_fill_accel = spy
    try:
        _impedance(DECKS[name], accel=True, medium=True)
    finally:
        SinusoidalGalerkinSolver._far_fill_accel = real_fill
    assert calls["n"] > 0, "the accelerated far fill was never called"
    assert calls["cplx"] > 0, "no call carried a complex k — dispatch took numpy"


def test_free_space_still_takes_the_real_entry_point():
    """The counter-gate: a real-k solve must not enter the twin.

    Bytes are the real contract here (momwire#762) and they are checked by
    rebuilding, but this pins the DISPATCH — the twin must not quietly become
    the path everything takes.
    """
    seen = []
    real_fill = SinusoidalGalerkinSolver._far_fill_accel

    def spy(self, k, ctx, src_c, src_t, **kw):
        seen.append(bool(np.iscomplexobj(k) or np.iscomplexobj(self.eta)))
        return real_fill(self, k, ctx, src_c, src_t, **kw)

    SinusoidalGalerkinSolver._far_fill_accel = spy
    try:
        _impedance(VERTICAL_41, accel=True, medium=False)
    finally:
        SinusoidalGalerkinSolver._far_fill_accel = real_fill
    assert seen, "the accelerated far fill was never called"
    assert not any(seen), "a free-space solve reached the complex entry point"


def test_im_k_positive_is_refused_by_the_kernel():
    """The growing-exponential branch, refused in C++ as well as in Python.

    The Python guard (`_complex_k`) is upstream of the kernel, so the kernel's
    own refusal is what protects a caller reaching it directly.
    """
    from momwire import _accelerators as acc

    z3 = np.zeros((1, 3), dtype=np.float64)
    with pytest.raises(RuntimeError, match="Im k <= 0"):
        acc.sinusoidal_galerkin_far_fill_cplx(
            z3,
            np.tile(np.array([[0.0, 0.0, 1.0]]), (1, 1)),
            np.array([0.001]),
            z3,
            np.tile(np.array([[0.0, 0.0, 1.0]]), (1, 1)),
            np.array([0.1]),
            complex(1.0, +0.5),  # Im k > 0
            complex(377.0, 0.0),
            np.array([0.0]),
            np.array([2.0]),
            np.ones((1, 1), dtype=np.complex128),
            np.array([0, 1], dtype=np.int64),
        )


def test_extended_kernel_in_the_medium_stays_on_numpy():
    """There is no complex EK twin, and the dispatch must know it.

    The EK block splits the reverse Bessel polynomials by hand on the grounds
    that x = jkR is purely imaginary, which is false at a complex k. So an EK
    solve in the medium must fall back rather than reach a kernel whose
    algebra does not hold there.
    """
    s = SinusoidalGalerkinSolver(
        wires=VERTICAL_41[0],
        n_per_edge_per_wire=VERTICAL_41[1],
        feeds=FEEDS,
        wavelength=WL7,
        wire_radius=0.001,
        extended_kernel=True,
    )
    _enter_medium(s)
    seen = []
    real_fill = SinusoidalGalerkinSolver._far_fill_accel

    def spy(self, k, ctx, src_c, src_t, **kw):
        seen.append(True)
        return real_fill(self, k, ctx, src_c, src_t, **kw)

    SinusoidalGalerkinSolver._far_fill_accel = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s.compute_impedance()
    finally:
        SinusoidalGalerkinSolver._far_fill_accel = real_fill
    assert not seen, "an EK solve in the medium reached the accelerated fill"
