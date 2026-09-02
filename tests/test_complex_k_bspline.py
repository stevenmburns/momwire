"""The in-medium thin-wire kernel: complex k through the B-spline fill
(momwire#553 unit 1).

A segment pair buried in a lossy medium sees the free-space thin-wire
kernel analytically continued to the medium's complex wavenumber
k_m = k₀·√ε̃, ε̃ = ε_r − jσ/(ωε₀), on the Im k_m ≤ 0 branch that makes
e^{−jk_m R} DECAY under e^{+jωt}. King & Smith (1981) eqs. (3.1)/(3.5) state
the continuation for the tubular kernel outright — the functional form does
not change — and their §7.4 (eqs. 4.1–4.5) puts the whole coincidence
singularity in the k-INDEPENDENT real part, which is why the closed-form
static extraction survives verbatim and only the smooth remainder's series
coefficients go complex. This file gates that claim rather than assuming it.

ORACLES (four, none of them the licensed engine)
------------------------------------------------
1. **Adaptive quadrature** of the very integrand the kernels discretise,
   real and imaginary parts separately, via `scipy.integrate.dblquad` at
   1e-13. The kernels are run at a high `n_qp` here so the residual is
   kernel CORRECTNESS and not Gauss–Legendre truncation (which G-U1-7
   measures on its own, at the shipped n_qp=4).
2. **The exact tubular kernel**, King & Smith (1981) eq. (3.5):
   K(z,z′) = (1/2π)∫_{−π}^{π} e^{−jkr}/(4πr) dθ′,
   r² = ζ² + 4a²sin²(θ′/2) — the thing the extended kernel is the O(a²)
   truncation of, integrated numerically in θ′ with its own self-convergence
   reported (G-U1-4).
3. **The Hertzian closed form** of the phase-0 prototype's regime-2 direct
   term (`scratch/524-phase0/proto/EQUATIONS.md`):
   E = C₁(∇∇/k_m² + I)·(e^{−jk_m R}/R), C₁ = −jωμ₀/4π — transcribed, not
   remembered — against a two-tent Galerkin mutual impedance assembled from
   the widened moments with the (jωμ₀, 1/(jωε̃)) prefactors composed here in
   the test (G-U1-5).
4. **Physics**: monotone extinction along a σ ladder, and |e^{−jk_m R}| < 1
   (G-U1-6).

The soils and frequencies are the shared #524 phase-0 spec
(`scratch/524-phase0/SPEC.md`): A = (13, 0.005), B = (20, 0.03),
C = (5, 0.001) S/m, at 7 and 21 MHz. Soil B at 7 MHz is the stressor —
|k_m|/k₀ = 8.92.
"""

import inspect

import numpy as np
import pytest
from scipy.integrate import dblquad, quad

import momwire._bspline_kernels as _bk
import momwire.bspline as _bs
from momwire._bspline_ek_moments import D_ek_moment
from momwire._bspline_kernels import (
    _EK,
    _complex_k,
    _ek_factor,
    _ek_reg_extra,
    _ek_reg_kernel,
    _refuse_complex_k,
    _seg_seg_full_moments_offedge,
    _seg_seg_full_moments_offedge_swept,
    _seg_seg_reg_geometry,
    _seg_seg_reg_moments,
    _seg_seg_reg_moments_from_geometry,
    _seg_seg_reg_moments_from_geometry_swept,
    _seg_seg_static_moments,
)
from momwire._stable import expm1_neg_jkR
from momwire.bspline import BSplineSolver
from momwire.hmatrix import HMatrixSolver

# ----------------------------------------------------------------------
# The media
# ----------------------------------------------------------------------

EPS0 = 8.8541878128e-12
MU0 = 4e-7 * np.pi
C0 = 299792458.0

# scratch/524-phase0/SPEC.md, verbatim.
SOILS = {"A": (13.0, 0.005), "B": (20.0, 0.03), "C": (5.0, 0.001)}
FREQS = {"7MHz": 7.0e6, "21MHz": 21.0e6}


def eps_tilde(eps_r, sigma, f_hz):
    """ε̃ = ε_r − jσ/(ωε₀) (EQUATIONS.md conventions section)."""
    return eps_r - 1j * sigma / (2.0 * np.pi * f_hz * EPS0)


def k_medium(eps_r, sigma, f_hz):
    """k_m = k₀√ε̃ on the Im ≤ 0 branch. Returns (k0, eps_tilde, k_m)."""
    k0 = 2.0 * np.pi * f_hz / C0
    et = eps_tilde(eps_r, sigma, f_hz)
    km = k0 * np.sqrt(et)
    if km.imag > 0:  # principal sqrt already lands here; assert, don't trust
        km = np.conj(km)
    return k0, et, complex(km)


MEDIA = {f"{s}/{fn}": k_medium(*SOILS[s], FREQS[fn]) for s in SOILS for fn in FREQS}
# The stressor: |k_m|/k0 = 8.92 at 7 MHz.
WORST = "B/7MHz"

# A real reference wavenumber for the free-space comparisons.
K0_7 = 2.0 * np.pi * 7.0e6 / C0

WHOLE_BLOCK = _EK(a=None, group_i=None, group_j=None)


def test_the_media_are_on_the_decaying_branch():
    """The convention gate on the test file's OWN k_m table: if these were
    on the Im > 0 branch every measured tolerance below would be measuring
    a growing exponential."""
    for name, (k0, et, km) in MEDIA.items():
        assert et.imag < 0, f"{name}: Im eps_tilde = {et.imag}"
        assert km.imag <= 0, f"{name}: Im k_m = {km.imag}"
        # |e^{-jk_m R}| = e^{Im(k_m) R} < 1 for every R > 0.
        R = np.geomspace(1e-4, 100.0, 51)
        assert np.all(np.abs(np.exp(-1j * km * R)) < 1.0)
    assert abs(MEDIA[WORST][2]) / MEDIA[WORST][0] == pytest.approx(8.9213, rel=1e-4)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------


def _straight(n, h, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0)):
    """n consecutive length-h segments from `origin` along `axis`."""
    o = np.asarray(origin, dtype=float)
    t = np.asarray(axis, dtype=float)
    t = t / np.linalg.norm(t)
    lo = np.stack([o + i * h * t for i in range(n)])
    hi = np.stack([o + (i + 1) * h * t for i in range(n)])
    return lo, hi


def _rel(got, ref):
    return float(np.abs(np.asarray(got) - np.asarray(ref)).max() / np.abs(ref).max())


# ======================================================================
# G-U1-1 — real k is byte-stable, and never enters the complex branch
# ======================================================================
#
# The whole of `tests/test_extended_kernel_bspline.py` (and
# `test_kernel_moments.py`, `test_accel_fallback.py`, `test_hmatrix.py`)
# is the wide half of this gate: unmodified by #553 U1 and green. What
# follows is the narrow half — that the ONE thing #553 adds to the real-k
# fill, the `np.iscomplexobj(k)` predicate, answers False on every real
# spelling and costs the real path no bits.

_REAL_K_SPELLINGS = [
    ("python float", 1.0),
    ("np.float64", np.float64(1.0)),
    ("float64 0-d array", np.array(1.0)),
    ("float64 array", np.array([1.0, 2.0])),
    ("python int", 1),
]
_COMPLEX_K_SPELLINGS = [
    ("python complex", 1 - 0.5j),
    ("np.complex128", np.complex128(1 - 0.5j)),
    ("complex 0-d array", np.array(1 - 0.5j)),
    ("complex array", np.array([1 - 0.5j, 2 - 1j])),
    ("complex array, Im = 0", np.array([1 + 0j])),
]


@pytest.mark.parametrize(
    "name,k", _REAL_K_SPELLINGS, ids=[n for n, _ in _REAL_K_SPELLINGS]
)
def test_gu1_1_the_dispatch_predicate_says_real(name, k):
    assert _complex_k(k) is False


@pytest.mark.parametrize(
    "name,k", _COMPLEX_K_SPELLINGS, ids=[n for n, _ in _COMPLEX_K_SPELLINGS]
)
def test_gu1_1_the_dispatch_predicate_says_complex(name, k):
    assert _complex_k(k) is True


class _ComplexKSpy:
    """Records every `_complex_k` verdict taken during a fill."""

    def __init__(self, real):
        self._real = real
        self.verdicts = []

    def __call__(self, k):
        v = self._real(k)
        self.verdicts.append(v)
        return v


@pytest.fixture
def complex_k_spy(monkeypatch):
    spy = _ComplexKSpy(_bk._complex_k)
    monkeypatch.setattr(_bk, "_complex_k", spy)
    return spy


_ACCEL_REDUCED_SYMBOLS = (
    "seg_seg_reg_moments_bspline_swept",
    "seg_seg_full_moments_bspline",
    "seg_seg_full_moments_bspline_swept",
)

# momwire#778 gave the PLAIN off-edge kernel a complex-k instantiation, so this
# one entry point is now REACHED in the medium rather than bypassed. It is
# spied separately from the tuple above precisely so the bypass gate keeps
# asserting zero for everything else — had it simply been appended, the "every
# accelerator stays idle" claim would have been weakened instead of split.
_ACCEL_CPLX_SYMBOL = "seg_seg_full_moments_bspline_cplx"


class _AccelSpy:
    """Counting proxy over the accelerator module (the #270 pattern)."""

    def __init__(self, real, names):
        self._real = real
        self.counts = dict.fromkeys(names, 0)

    def __getattr__(self, name):
        target = getattr(self._real, name)
        if name not in self.counts:
            return target

        def counted(*args, **kwargs):
            self.counts[name] += 1
            return target(*args, **kwargs)

        return counted


@pytest.fixture
def accel_spy(monkeypatch):
    spy = _AccelSpy(_bk._acc, _ACCEL_REDUCED_SYMBOLS)
    monkeypatch.setattr(_bk, "_acc", spy)
    return spy


_G1_DECK = dict(
    wires=[np.array([[0.0, 0.0, -2.5], [0.0, 0.0, 2.5]])],
    n_per_edge_per_wire=[[21]],
    wire_radius=0.02,
)


def test_gu1_1_a_real_solve_never_takes_the_complex_branch(complex_k_spy, accel_spy):
    """A real-k solve asks the predicate (so the wiring is live) and is told
    "real" every single time — and the C++ reduced kernels still fire, which
    is what "the real path is unchanged" means operationally."""
    lam = C0 / 30.0e6
    BSplineSolver(**_G1_DECK, wavelength=lam, degree=2).compute_impedance()
    assert complex_k_spy.verdicts, "the dispatch predicate was never consulted"
    assert not any(complex_k_spy.verdicts), "a real solve took the complex branch"
    fired = {n: c for n, c in accel_spy.counts.items() if c}
    assert fired, f"no C++ reduced kernel fired on a real solve: {accel_spy.counts}"


def _all_six(k, ks, h=0.12, a=0.005):
    """Every k-taking entry point once, plus the k-free static one."""
    ends = np.arange(4) * h
    lo_i, hi_i = _straight(3, h)
    lo_j, hi_j = _straight(3, h, origin=(0.0, 0.4, 0.0))
    geo = _seg_seg_reg_geometry(ends, a, max_d=2, n_qp=4)
    return (
        _seg_seg_static_moments(ends, a, 2),
        _seg_seg_reg_moments(ends, a, k, 2, 4),
        _seg_seg_reg_moments_from_geometry(geo, k),
        _seg_seg_reg_moments_from_geometry_swept(geo, ks),
        _seg_seg_full_moments_offedge(lo_i, hi_i, lo_j, hi_j, a, k, 2, 4),
        _seg_seg_full_moments_offedge_swept(lo_i, hi_i, lo_j, hi_j, a, ks, 2, 4),
    )


def test_gu1_1_real_k_output_is_bit_identical_to_the_pre_553_dispatch(monkeypatch):
    """Mutation control on the one thing #553 adds to the real-k fill.

    Compute with the shipped predicate, then force it to the pre-#553
    constant-False and compute again: every entry point must return the
    SAME BYTES. Were the predicate ever to mis-classify a real k the two
    would differ by the numpy-vs-C++ reduction-order gap (~1e-14), which
    `array_equal` sees and `allclose` would not."""
    k = K0_7
    ks = np.array([k, 2 * k])
    got = _all_six(k, ks)
    monkeypatch.setattr(_bk, "_complex_k", lambda _k: False)
    ref = _all_six(k, ks)
    for i, (g, r) in enumerate(zip(got, ref)):
        assert np.array_equal(g, r), f"entry point {i} moved"


def test_gu1_1_complex_k_bypasses_every_double_k_accelerator_flag(accel_spy):
    """The `double k` C++ twins must not be reached in the medium — they
    would truncate the wavenumber silently.

    momwire#778 narrowed this gate rather than deleting it. The PLAIN off-edge
    kernel now HAS a complex-k instantiation and is asserted separately below;
    every entry point that still takes `double k` must stay at zero, which is
    what this asserts. Widening the tuple instead would have let a future
    complex-k leak into the swept or reg kernels pass unnoticed.
    """
    h, a = 0.12, 0.005
    km = MEDIA[WORST][2]
    ends = np.arange(4) * h
    lo_i, hi_i = _straight(3, h)
    lo_j, hi_j = _straight(3, h, origin=(0.0, 0.4, 0.0))
    ks = np.array([km, 2 * km])
    geo = _seg_seg_reg_geometry(ends, a, max_d=2, n_qp=4)
    _seg_seg_reg_moments_from_geometry(geo, km)
    _seg_seg_reg_moments_from_geometry_swept(geo, ks)
    _seg_seg_full_moments_offedge_swept(lo_i, hi_i, lo_j, hi_j, a, ks, 2, 4)
    assert accel_spy.counts == dict.fromkeys(_ACCEL_REDUCED_SYMBOLS, 0)


def test_gu1_1_complex_k_now_reaches_the_plain_offedge_accelerator(monkeypatch):
    """momwire#778: the plain off-edge kernel's COMPLEX_K instantiation IS
    taken in the medium, and the real-k entry point is NOT.

    The pre-#778 gate spied only on a fixed tuple of `double k` symbols, so
    once the complex twin existed that gate went vacuous for this path — it
    asserted zero on names the call no longer used. This is the replacement
    that actually watches the symbol the medium now dispatches to.
    """
    spy = _AccelSpy(_bk._acc, (_ACCEL_CPLX_SYMBOL, "seg_seg_full_moments_bspline"))
    monkeypatch.setattr(_bk, "_acc", spy)
    h, a = 0.12, 0.005
    km = MEDIA[WORST][2]
    lo_i, hi_i = _straight(3, h)
    lo_j, hi_j = _straight(3, h, origin=(0.0, 0.4, 0.0))
    _seg_seg_full_moments_offedge(lo_i, hi_i, lo_j, hi_j, a, km, 2, 4)
    if _bk._HAVE_BSPLINE_OFFEDGE_CPLX_ACCEL:
        assert spy.counts[_ACCEL_CPLX_SYMBOL] == 1, spy.counts
    assert spy.counts["seg_seg_full_moments_bspline"] == 0, spy.counts


def test_gu1_1_the_complex_offedge_accelerator_matches_its_numpy_twin(monkeypatch):
    """momwire#778's continuation is exp(-jkR) = exp(Im(k)*R)*exp(-j*Re(k)*R),
    a real scale factor — NOT the King & Smith (3.20b,c) |k| substitution.
    Gated against the numpy twin the C++ path replaces, at an n_qp the old
    ceiling would not have served."""
    h, a = 0.12, 0.005
    km = MEDIA[WORST][2]
    lo_i, hi_i = _straight(6, h)
    lo_j, hi_j = _straight(6, h, origin=(0.0, 0.4, 0.0))
    for n_qp in (4, 8, 32):
        got = _seg_seg_full_moments_offedge(lo_i, hi_i, lo_j, hi_j, a, km, 2, n_qp)
        monkeypatch.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_CPLX_ACCEL", False)
        ref = _seg_seg_full_moments_offedge(lo_i, hi_i, lo_j, hi_j, a, km, 2, n_qp)
        monkeypatch.undo()
        rel = np.abs(got - ref).max() / np.abs(ref).max()
        assert rel < 1e-13, f"n_qp={n_qp}: relative {rel:.3e}"


def test_gu1_1_the_complex_offedge_accelerator_refuses_the_growing_branch():
    """Im k > 0 is the growing-exponential branch. The Python predicate raises
    before dispatch, and the C++ entry point re-asserts it because a test can
    reach that symbol directly."""
    if not _bk._HAVE_BSPLINE_OFFEDGE_CPLX_ACCEL:
        pytest.skip("no complex-k off-edge accelerator in this build")
    h, a = 0.12, 0.005
    lo_i, hi_i = _straight(3, h)
    lo_j, hi_j = _straight(3, h, origin=(0.0, 0.4, 0.0))
    gl_xi, gl_w = np.polynomial.legendre.leggauss(4)
    with pytest.raises(RuntimeError, match="Im k > 0"):
        _bk._acc.seg_seg_full_moments_bspline_cplx(
            np.ascontiguousarray(lo_i, dtype=np.float64),
            np.ascontiguousarray(hi_i, dtype=np.float64),
            np.ascontiguousarray(lo_j, dtype=np.float64),
            np.ascontiguousarray(hi_j, dtype=np.float64),
            a * a,
            complex(0.5, +0.1),
            2,
            np.ascontiguousarray(0.5 * (gl_xi + 1.0)),
            np.ascontiguousarray(0.5 * gl_w),
        )


def test_gu1_1_complex_k_bypasses_the_ek_accelerator_flags(accel_spy, monkeypatch):
    """The three k-TAKING EK twins must stay idle in the medium...

    ...while the k-FREE one keeps its C++ path, medium or not. That
    asymmetry is the whole shape of the continuation: the static half of
    the split carries no k (King & Smith §7.4), so it is served by the same
    accelerated closed form at k_m as at k_0, and only the remainder
    changes backend.
    """
    ek_k_names = (
        "seg_seg_reg_moments_bspline_swept_ek",
        "seg_seg_full_moments_bspline_ek",
        "seg_seg_full_moments_bspline_swept_ek",
    )
    static_name = "seg_seg_static_moments_bspline_uniform_ek"
    spy = _AccelSpy(_bk._acc, ek_k_names + (static_name,))
    monkeypatch.setattr(_bk, "_acc", spy)
    h, a = 0.12, 0.005
    km = MEDIA[WORST][2]
    ends = np.arange(4) * h
    lo_i, hi_i = _straight(3, h)
    lo_j, hi_j = _straight(3, h, origin=(0.0, 0.4, 0.0))
    gi = np.zeros(3, dtype=np.int64)
    pair = _EK(a=None, group_i=gi, group_j=gi + 1)
    geo = _seg_seg_reg_geometry(ends, a, max_d=2, n_qp=4, ek=WHOLE_BLOCK)
    _seg_seg_reg_moments_from_geometry(geo, km)
    _seg_seg_reg_moments_from_geometry_swept(geo, np.array([km]))
    _seg_seg_full_moments_offedge(lo_i, hi_i, lo_j, hi_j, a, km, 2, 4, ek=pair)
    _seg_seg_full_moments_offedge_swept(
        lo_i, hi_i, lo_j, hi_j, a, np.array([km]), 2, 4, ek=pair
    )
    assert [spy.counts[n] for n in ek_k_names] == [0, 0, 0], spy.counts
    _seg_seg_static_moments(ends, a, 2, ek=WHOLE_BLOCK)
    if _bk._HAVE_BSPLINE_STATIC_EK_ACCEL:
        assert spy.counts[static_name] == 1, spy.counts


# ======================================================================
# G-U1-2 — a complex k with Im = 0 agrees with the real k
# ======================================================================
#
# NOT a byte gate: complex128 and float64 are different dtypes, and the
# complex path is numpy where the real one is C++, so the two reduce in
# different orders. Pinning bytes across dtypes is the same trap #270's
# cross-backend note already names.
#
# Measured relative worsts over the grid f ∈ {3, 7, 30, 300} MHz × h ∈
# {0.02, 0.12, 1, 5} m × Δ/a ∈ {2, 6, 24, 100} × n_qp ∈ {4, 8} × degrees
# 0..2 × {reduced, EK} × off-edge gaps {a, 0.35, 5} m (the worst sits at
# the oscillatory end, |k|h ≈ 31):
#
#   static moments (k-free, no k to widen)         0.0
#   same-edge reg, single k                        1.268e-12
#   same-edge reg, swept                           6.539e-13
#   off-edge full, single k                        3.281e-15
#   off-edge full, swept                           7.243e-15
#   EK trio (`_ek_factor`/`_extra`/`_reg_kernel`)  0.0  (exactly)
ZERO_IMAG_AGREEMENT = 1e-11

# Complex swept-k vs complex single-k, both on numpy but with different
# chunking/`optimize=` reduction orders. Measured worsts: 4.462e-16 (reg),
# 0.0 (off-edge).
SWEPT_SINGLE_AGREEMENT = 1e-13

# The G-U1-2 grid the tests below walk (a subset of the measurement grid
# above, sized for a fast inner loop but including its worst corner).
_G2_GRID = [
    ("7MHz, h=0.12", 2.0 * np.pi * 7.0e6 / C0, 0.12),
    ("30MHz, h=1.0", 2.0 * np.pi * 30.0e6 / C0, 1.0),
    ("300MHz, h=5.0", 2.0 * np.pi * 300.0e6 / C0, 5.0),
]


def test_gu1_2_static_moments_take_no_k_at_all():
    """`_seg_seg_static_moments` is the k → 0 extraction: King & Smith §7.4
    puts the coincidence singularity in the kernel's k-INDEPENDENT real
    part, so this closed form is valid at complex k VERBATIM — and the way
    that is true in code is that there is no `k` parameter to widen."""
    sig = inspect.signature(_seg_seg_static_moments)
    assert "k" not in sig.parameters, sig
    src = inspect.getsource(_seg_seg_static_moments)
    assert "k" not in {t.strip() for t in src.split()}
    # And the same for the generated EK static family it adds.
    assert "k" not in inspect.signature(D_ek_moment).parameters


@pytest.mark.parametrize("case", _G2_GRID, ids=[c[0] for c in _G2_GRID])
@pytest.mark.parametrize("ek", [None, WHOLE_BLOCK], ids=["reduced", "ek"])
def test_gu1_2_same_edge_reg_moments(ek, case, record_property):
    _name, k, h = case
    a = h / 24.0
    ends = np.arange(5) * h
    ref = _seg_seg_reg_moments(ends, a, k, 2, 4, ek=ek)
    got = _seg_seg_reg_moments(ends, a, complex(k, 0.0), 2, 4, ek=ek)
    assert got.dtype == np.complex128
    rel = _rel(got, ref)
    record_property("g2_reg_single", rel)
    assert rel <= ZERO_IMAG_AGREEMENT, rel


@pytest.mark.parametrize("case", _G2_GRID, ids=[c[0] for c in _G2_GRID])
@pytest.mark.parametrize("ek", [None, WHOLE_BLOCK], ids=["reduced", "ek"])
def test_gu1_2_reg_moments_from_geometry_and_swept(ek, case, record_property):
    _name, k, h = case
    a = h / 24.0
    ends = np.arange(5) * h
    geo = _seg_seg_reg_geometry(ends, a, max_d=2, n_qp=4, ek=ek)
    ref = _seg_seg_reg_moments_from_geometry(geo, k)
    got = _seg_seg_reg_moments_from_geometry(geo, complex(k, 0.0))
    record_property("g2_reg_from_geometry", _rel(got, ref))
    assert _rel(got, ref) <= ZERO_IMAG_AGREEMENT, _rel(got, ref)

    ks = np.array([k, 2.0 * k, 0.5 * k])
    ref_s = _seg_seg_reg_moments_from_geometry_swept(geo, ks)
    got_s = _seg_seg_reg_moments_from_geometry_swept(geo, ks.astype(np.complex128))
    record_property("g2_reg_swept", _rel(got_s, ref_s))
    assert _rel(got_s, ref_s) <= ZERO_IMAG_AGREEMENT, _rel(got_s, ref_s)
    # And the swept complex path agrees with the single-k complex path — to
    # tolerance, not to the bit: the swept branch chunks and reduces with
    # `optimize=True`, the single-k one does not.
    for i, kk in enumerate(ks):
        one = _seg_seg_reg_moments_from_geometry(geo, complex(kk, 0.0))
        assert _rel(got_s[i], one) <= SWEPT_SINGLE_AGREEMENT, _rel(got_s[i], one)


@pytest.mark.parametrize("case", _G2_GRID, ids=[c[0] for c in _G2_GRID])
@pytest.mark.parametrize("ek", ["off", "pair"])
def test_gu1_2_offedge_full_moments(ek, case, record_property):
    _name, k, h = case
    a = h / 24.0
    lo_i, hi_i = _straight(3, h)
    lo_j, hi_j = _straight(3, h, origin=(0.0, 3.0 * h, 0.0))
    spec = None
    if ek == "pair":
        gi = np.zeros(3, dtype=np.int64)
        spec = _EK(a=None, group_i=gi, group_j=gi)
    ref = _seg_seg_full_moments_offedge(lo_i, hi_i, lo_j, hi_j, a, k, 2, 4, ek=spec)
    got = _seg_seg_full_moments_offedge(
        lo_i, hi_i, lo_j, hi_j, a, complex(k, 0.0), 2, 4, ek=spec
    )
    record_property("g2_offedge_single", _rel(got, ref))
    assert _rel(got, ref) <= ZERO_IMAG_AGREEMENT, _rel(got, ref)

    ks = np.array([k, 2.0 * k])
    ref_s = _seg_seg_full_moments_offedge_swept(
        lo_i, hi_i, lo_j, hi_j, a, ks, 2, 4, ek=spec
    )
    got_s = _seg_seg_full_moments_offedge_swept(
        lo_i, hi_i, lo_j, hi_j, a, ks.astype(np.complex128), 2, 4, ek=spec
    )
    record_property("g2_offedge_swept", _rel(got_s, ref_s))
    assert _rel(got_s, ref_s) <= ZERO_IMAG_AGREEMENT, _rel(got_s, ref_s)
    assert _rel(got_s[0], got) <= SWEPT_SINGLE_AGREEMENT, _rel(got_s[0], got)


def test_gu1_2_ek_trio_is_already_complex_clean():
    """`_ek_factor`/`_ek_reg_extra`/`_ek_reg_kernel` were written in `kr`
    and `1j` and needed no widening — this is the verification the scope
    note calls for, not a rewrite."""
    R = np.geomspace(1e-3, 10.0, 41)
    a, k = 0.01, K0_7
    for fn in (_ek_factor, _ek_reg_extra, _ek_reg_kernel):
        ref = fn(R, a, k)
        got = fn(R, a, complex(k, 0.0))
        assert np.asarray(got).dtype == np.complex128
        assert _rel(got, ref) <= ZERO_IMAG_AGREEMENT, (fn.__name__, _rel(got, ref))
    # The a = 0 collapse is EXACT at complex k too, not merely to rounding:
    # `_ek_factor` is 1.0 and `_ek_reg_extra` 0.0 in IEEE, so the EK
    # remainder is bit-identical to the reduced one (the #249 §2.1 claim,
    # re-run in the medium).
    km = MEDIA[WORST][2]
    assert np.all(_ek_factor(R, 0.0, km) == 1.0)
    assert np.all(_ek_reg_extra(R, 0.0, km) == 0.0)
    # The reduced branch's own spelling — `expm1_neg_jkR`'s complex bracket
    # since momwire#799, not the literal `exp(-jkR) - 1` this used to name.
    reduced = expm1_neg_jkR(km, R) / (4 * np.pi * R)
    assert np.array_equal(_ek_reg_kernel(R, 0.0, km), reduced)


# ======================================================================
# G-U1-3 — the moments against adaptive quadrature of their own integrand
# ======================================================================
#
# `n_qp` is deliberately high here (see the module docstring): this gate is
# about the kernel being the right integral at complex k, not about the
# shipped quadrature order.
#
# Measured worst relative error, over all three soils x both frequencies x
# degrees 0..2 x the three geometries:
#
#   same-edge reg remainder                     see G3_REG_TOL comment
#   same-edge FULL (static extraction + reg)    see G3_FULL_TOL comment
#   off-edge full, gap ~ a                      see G3_FULL_TOL comment
#   off-edge full, gap ~ lambda_0/4             see G3_FULL_TOL comment
#
G3_NQP = 12
G3_N = 3
G3_QUAD_KW = dict(epsabs=1e-14, epsrel=1e-12)


def _brute_moment(o_i, t_i, h_i, o_j, t_j, h_j, p, q, a, k, subtract_static):
    """∫₀^{h_i}∫₀^{h_j} u^p v^q · G(R) dv du by adaptive quadrature, real
    and imaginary parts separately. `G = (e^{−jkR} − 1)/(4πR)` when
    `subtract_static`, else `e^{−jkR}/(4πR)`. R² = |r_i(u) − r_j(v)|² + a²."""

    def integrand(v, u, part):
        d = (o_i + u * t_i) - (o_j + v * t_j)
        R = np.sqrt(d @ d + a * a)
        e = np.exp(-1j * k * R)
        num = (e - 1.0) if subtract_static else e
        val = (u**p) * (v**q) * num / (4.0 * np.pi * R)
        return val.real if part == 0 else val.imag

    re, _ = dblquad(integrand, 0.0, h_i, 0.0, h_j, args=(0,), **G3_QUAD_KW)
    im, _ = dblquad(integrand, 0.0, h_i, 0.0, h_j, args=(1,), **G3_QUAD_KW)
    return re + 1j * im


def _brute_block(lo_i, hi_i, lo_j, hi_j, a, k, max_d, subtract_static):
    n_i, n_j = lo_i.shape[0], lo_j.shape[0]
    out = np.zeros((max_d + 1, max_d + 1, n_i, n_j), dtype=np.complex128)
    for i in range(n_i):
        h_i = float(np.linalg.norm(hi_i[i] - lo_i[i]))
        t_i = (hi_i[i] - lo_i[i]) / h_i
        for j in range(n_j):
            h_j = float(np.linalg.norm(hi_j[j] - lo_j[j]))
            t_j = (hi_j[j] - lo_j[j]) / h_j
            for p in range(max_d + 1):
                for q in range(max_d + 1):
                    out[p, q, i, j] = _brute_moment(
                        lo_i[i],
                        t_i,
                        h_i,
                        lo_j[j],
                        t_j,
                        h_j,
                        p,
                        q,
                        a,
                        k,
                        subtract_static,
                    )
    return out


# Two tolerances, because the gate splits into two questions that the
# residual would otherwise conflate.
#
# (a) IS THE KERNEL THE RIGHT INTEGRAL at complex k? Asked where
#     Gauss-Legendre is not the limiting factor — Δ/a = 2, the fat end of
#     the usable window (#248's floor), n_qp = 16. Measured worsts over all
#     three soils x both frequencies x degrees 0..2:
#       same-edge reg remainder                 3.07e-16
#       same-edge static extraction + reg       9.91e-16
#       off-edge full, gap = a                  1.57e-16
#       off-edge full, gap = lambda_0/4         4.93e-16
#     Machine precision. Pinned at 1e-13 (~100x margin).
G3_TRUTH_TOL = 1e-13
#
# (b) At the THIN end (Δ/a = 24) the residual is Gauss-Legendre truncation
#     on the near-diagonal 1/R peak, which is a property of the shipped
#     fill and not of the continuation — and the way that is DEMONSTRATED
#     rather than asserted is that the residual is k-independent: the same
#     geometry at real k = |k_m| reads the same number. Measured ratios
#     complex/real over the six SPEC media at n_qp = 8: 1.002 to 1.016.
G3_THIN_PARITY = 1.5
G3_NQP_THIN = 8  # the C++ reduced kernel's own n_qp ceiling
G3_DA_TRUTH = 2.0
G3_DA_THIN = 24.0


def _g3_geometry(km, da):
    """~lambda_m/40 panels at the requested Δ/a."""
    h = (2.0 * np.pi / abs(km)) / 40.0
    return h, h / da


@pytest.mark.parametrize("medium", list(MEDIA))
def test_gu1_3_same_edge_reg_remainder_vs_adaptive_quadrature(medium, record_property):
    """`(e^{−jkR} − 1)/(4πR)` moments on the same-edge Toeplitz block."""
    _k0, _et, km = MEDIA[medium]
    h, a = _g3_geometry(km, G3_DA_TRUTH)
    ends = np.arange(G3_N + 1) * h
    got = _seg_seg_reg_moments(ends, a, km, 2, G3_NQP)
    lo, hi = _straight(G3_N, h)
    ref = _brute_block(lo, hi, lo, hi, a, km, 2, subtract_static=True)
    rel = _rel(got, ref)
    record_property("g3_reg_rel", rel)
    assert rel <= G3_TRUTH_TOL, f"{medium}: {rel:.3e}"


@pytest.mark.parametrize("medium", list(MEDIA))
def test_gu1_3_same_edge_full_kernel_vs_adaptive_quadrature(medium, record_property):
    """static extraction + reg remainder == the FULL `e^{−jkR}/(4πR)`
    moment, including the log-singular diagonal. This is the gate on King &
    Smith §7.4's claim that the k-free closed-form extraction survives the
    continuation verbatim — if the static family needed complex
    coefficients, the diagonal would be the first place it showed."""
    _k0, _et, km = MEDIA[medium]
    h, a = _g3_geometry(km, G3_DA_TRUTH)
    ends = np.arange(G3_N + 1) * h
    got = _seg_seg_static_moments(ends, a, 2) + _seg_seg_reg_moments(
        ends, a, km, 2, G3_NQP
    )
    lo, hi = _straight(G3_N, h)
    ref = _brute_block(lo, hi, lo, hi, a, km, 2, subtract_static=False)
    rel = _rel(got, ref)
    record_property("g3_full_same_edge_rel", rel)
    assert rel <= G3_TRUTH_TOL, f"{medium}: {rel:.3e}"


@pytest.mark.parametrize("gap", ["near", "far"])
@pytest.mark.parametrize("medium", list(MEDIA))
def test_gu1_3_offedge_full_kernel_vs_adaptive_quadrature(medium, gap, record_property):
    _k0, _et, km = MEDIA[medium]
    h, a = _g3_geometry(km, G3_DA_TRUTH)
    # "near" = the touching regime the a-regularization is for; "far" = a
    # quarter FREE-SPACE wavelength, the ordinary off-edge pair.
    offset = a if gap == "near" else 0.25 * (2.0 * np.pi / _k0)
    lo_i, hi_i = _straight(G3_N, h)
    lo_j, hi_j = _straight(G3_N, h, origin=(offset, 0.0, 0.0))
    got = _seg_seg_full_moments_offedge(lo_i, hi_i, lo_j, hi_j, a, km, 2, G3_NQP)
    ref = _brute_block(lo_i, hi_i, lo_j, hi_j, a, km, 2, subtract_static=False)
    rel = _rel(got, ref)
    record_property(f"g3_full_offedge_{gap}_rel", rel)
    assert rel <= G3_TRUTH_TOL, f"{medium}/{gap}: {rel:.3e}"


@pytest.mark.parametrize("medium", list(MEDIA))
def test_gu1_3_the_thin_wire_residual_is_quadrature_not_the_continuation(
    medium, record_property
):
    """At Δ/a = 24 the same-edge reg block misses the adaptive-quadrature
    truth by ~7.5e-05 at n_qp = 8 — and misses it by the SAME amount at
    real k = |k_m| on the same geometry. That is Gauss-Legendre truncation
    on the 1/R peak, k-independent to ~1%, not an error the continuation
    introduced.
    """
    _k0, _et, km = MEDIA[medium]
    h, a = _g3_geometry(km, G3_DA_THIN)
    ends = np.arange(G3_N + 1) * h
    lo, hi = _straight(G3_N, h)
    cplx = _rel(
        _seg_seg_reg_moments(ends, a, km, 2, G3_NQP_THIN),
        _brute_block(lo, hi, lo, hi, a, km, 2, subtract_static=True),
    )
    real = _rel(
        _seg_seg_reg_moments(ends, a, abs(km), 2, G3_NQP_THIN),
        _brute_block(lo, hi, lo, hi, a, abs(km), 2, subtract_static=True),
    )
    record_property("g3_thin_cplx", cplx)
    record_property("g3_thin_real", real)
    assert 1.0 / G3_THIN_PARITY <= cplx / real <= G3_THIN_PARITY, (cplx, real)


# ======================================================================
# G-U1-4 — the extended kernel against the exact tubular kernel
# ======================================================================
#
# King & Smith (1981) eq. (3.5):
#     K(z,z') = (1/2pi) integral_{-pi}^{pi} e^{-jkr}/(4 pi r) dtheta',
#     r^2 = zeta^2 + 4 a^2 sin^2(theta'/2)
# The extended kernel is its O(a^2) truncation and the reduced kernel its
# O(a^0) one, so the gate is NOT "EK equals the tube" — it is that the
# truncation ORDER survives the continuation to complex k: the EK gap
# falls like (a/zeta)^4 where the reduced one falls like (a/zeta)^2, at
# k_m exactly as at k_0.

_G4_QUAD_KW = dict(epsabs=1e-15, epsrel=1e-13, limit=400)


def _tube_kernel(zeta, a, k):
    """Eq. (3.5) by adaptive theta' quadrature. Returns (value, abserr)."""

    def integrand(th, part):
        r = np.sqrt(zeta * zeta + 4.0 * a * a * np.sin(0.5 * th) ** 2)
        val = np.exp(-1j * k * r) / (4.0 * np.pi * r)
        return val.real if part == 0 else val.imag

    # Even in theta' -> integrate [0, pi] and drop the 1/2 with the 1/(2pi).
    re, ere = quad(integrand, 0.0, np.pi, args=(0,), **_G4_QUAD_KW)
    im, eim = quad(integrand, 0.0, np.pi, args=(1,), **_G4_QUAD_KW)
    return (re + 1j * im) / np.pi, np.hypot(ere, eim) / np.pi


def _ek_pointwise(zeta, a, k):
    """The EK kernel assembled the way the fill assembles it: the k-free
    static spelling `_ek_factor(R, a, 0)` plus `_ek_reg_kernel`."""
    R = np.sqrt(zeta * zeta + a * a)
    static = (1.0 / (4.0 * np.pi * R)) * (
        1.0 - 0.5 * a * a / (R * R) + 0.75 * (a**4) / (R**4)
    )
    return static + _ek_reg_kernel(R, a, k)


def _reduced_pointwise(zeta, a, k):
    R = np.sqrt(zeta * zeta + a * a)
    return np.exp(-1j * k * R) / (4.0 * np.pi * R)


# Measured (see the gate's own printout): the theta' quadrature's own error
# estimate stays below this fraction of the value it reports, over the whole
# zeta ladder including the touching end.
# Measured worst over the whole zeta ladder and every SPEC k_m: 9.91e-14
# (at zeta/a = 0.5, where the theta' integrand is most peaked); 1.11e-14
# everywhere else.
G4_ORACLE_SELF_CONVERGENCE = 1e-11
# The EK gap must be at least this much smaller than the reduced gap at the
# far end of the zeta ladder — the O(a^4)-vs-O(a^2) statement. Measured
# ratios at zeta/a = 256: 8.74e+04 (real k0), 8.74e+04 to 9.09e+04 across
# the six SPEC media. Pinned two decades below the measurement, so the gate
# reads the ORDER and not the constant.
G4_EK_IMPROVEMENT = 1.0e3

# zeta = 0 is deliberately absent: the exact tube kernel diverges
# logarithmically there (r = 2a|sin(theta'/2)| -> 0), which is the #249
# limit the module docstring already records. The ladder starts inside the
# touching regime and runs out to zeta/a = 256.
_G4_ZETA_OVER_A = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0])
# EK only beats the reduced kernel from zeta/a >= 1 (#249 §3: "below
# delta/a ~ 1 EK stops improving, and by 0.5 it is WORSE"). Measured here:
# at zeta/a = 0.5 the reduced kernel is 4.5x better, at 0.25 EK is 2.2x
# better again. Those two rungs are recorded, not asserted on.
_G4_EK_WINS_FROM = 3  # index of zeta/a = 2.0


@pytest.mark.parametrize("medium", list(MEDIA))
def test_gu1_4_ek_truncation_order_survives_the_continuation(medium, record_property):
    _k0, _et, km = MEDIA[medium]
    a = 0.001
    zetas = a * _G4_ZETA_OVER_A
    ek_gap, red_gap, self_conv = [], [], []
    for z in zetas:
        ref, err = _tube_kernel(z, a, km)
        self_conv.append(err / abs(ref))
        ek_gap.append(abs(_ek_pointwise(z, a, km) - ref) / abs(ref))
        red_gap.append(abs(_reduced_pointwise(z, a, km) - ref) / abs(ref))
    ek_gap = np.array(ek_gap)
    red_gap = np.array(red_gap)
    record_property(f"g4_self_conv_{medium}", float(max(self_conv)))
    record_property(f"g4_ek_gap_{medium}", list(map(float, ek_gap)))
    record_property(f"g4_red_gap_{medium}", list(map(float, red_gap)))
    assert max(self_conv) <= G4_ORACLE_SELF_CONVERGENCE, max(self_conv)
    # Both truncations converge, monotonically, from where EK is valid...
    tail = slice(_G4_EK_WINS_FROM, None)
    assert all(np.diff(ek_gap[tail]) < 0), ek_gap
    assert all(np.diff(red_gap[tail]) < 0), red_gap
    # ...EK is the better one there...
    for i in range(_G4_EK_WINS_FROM, len(zetas)):
        assert ek_gap[i] < red_gap[i], (i, ek_gap[i], red_gap[i])
    # ...and by the far end it is better by the order, not by a constant.
    assert red_gap[-1] / ek_gap[-1] >= G4_EK_IMPROVEMENT, red_gap[-1] / ek_gap[-1]


def test_gu1_4_the_ek_gap_slope_is_the_real_k_one(record_property):
    """The load-bearing comparison: the O(a^4)/O(a^2) slopes measured at the
    worst SPEC k_m equal the ones measured at real k_0 — the continuation
    changed the kernel's VALUES, not its truncation order."""
    a = 0.001
    zetas = a * np.array([8.0, 16.0, 32.0, 64.0, 128.0, 256.0])

    def slopes(k):
        ek, red = [], []
        for z in zetas:
            ref, _ = _tube_kernel(z, a, k)
            ek.append(abs(_ek_pointwise(z, a, k) - ref) / abs(ref))
            red.append(abs(_reduced_pointwise(z, a, k) - ref) / abs(ref))
        lz = np.log(zetas / a)
        return (
            float(np.polyfit(lz, np.log(ek), 1)[0]),
            float(np.polyfit(lz, np.log(red), 1)[0]),
        )

    ek_r, red_r = slopes(K0_7)
    ek_c, red_c = slopes(MEDIA[WORST][2])
    record_property("g4_slopes_real", (ek_r, red_r))
    record_property("g4_slopes_complex", (ek_c, red_c))
    assert red_r == pytest.approx(-2.0, abs=0.15), red_r
    assert red_c == pytest.approx(-2.0, abs=0.15), red_c
    assert ek_r == pytest.approx(-4.0, abs=0.3), ek_r
    assert ek_c == pytest.approx(-4.0, abs=0.3), ek_c


# ======================================================================
# G-U1-5 — the Hertzian mutual-impedance limit
# ======================================================================
#
# Two tent (degree-1 B-spline) basis functions, each over two segments of
# length h, filled through the WIDENED numpy moment path; the Galerkin
# mixed-potential prefactors (jw mu_0, 1/(jw eps_tilde)) composed here.
# The oracle is the phase-0 prototype's regime-2 direct term, transcribed
# from scratch/524-phase0/proto/EQUATIONS.md:
#
#     E = C1 (grad grad / k_m^2 + I) . (e^{-j k_m R} / R),  C1 = -j w mu0 / 4pi
#
# Reaction of two short elements of current moment (integral of W) = h:
#     Z_exact = - h_m h_n  t_m . E(R0; t_n)
# which, writing g = e^{-j k R}/R, A = t_m.t_n, B = (t_m.Rhat)(t_n.Rhat),
#     Z_exact = h_m h_n (j w mu0 / 4pi) [ A g + (1/k^2)( g'(A - B)/R + B g'' ) ]
# with g' = -g(jk + 1/R) and g'' = g(jk + 1/R)^2 + g/R^2.


def _hertzian_Z(R_vec, t_m, t_n, k, omega, hm, hn):
    R = float(np.linalg.norm(R_vec))
    rh = R_vec / R
    g = np.exp(-1j * k * R) / R
    gp = -g * (1j * k + 1.0 / R)
    gpp = g * (1j * k + 1.0 / R) ** 2 + g / (R * R)
    A = float(t_m @ t_n)
    B = float(t_m @ rh) * float(t_n @ rh)
    bracket = A * g + (gp * (A - B) / R + B * gpp) / (k * k)
    return hm * hn * (1j * omega * MU0 / (4.0 * np.pi)) * bracket


def _tent_mutual_Z(o_m, t_m, o_n, t_n, h, a, k, omega, eps_c, n_qp=12):
    """Galerkin mutual impedance of two 2-segment tents, assembled from the
    widened moment kernel.

    Tent m occupies [o_m, o_m + 2h t_m]; W = u/h on its first segment and
    1 - u/h on its second (u local, from each segment's left end), so
    W' = +1/h then -1/h — zero net charge, the divergence-conforming shape
    whose Hertzian limit is a dipole of moment h.
    """
    lo_m, hi_m = _straight(2, h, origin=o_m, axis=t_m)
    lo_n, hi_n = _straight(2, h, origin=o_n, axis=t_n)
    M = _seg_seg_full_moments_offedge(lo_m, hi_m, lo_n, hi_n, a, k, 1, n_qp)
    # W coefficients per segment in the local monomial basis {1, u}.
    cw = np.array([[0.0, 1.0 / h], [1.0, -1.0 / h]])  # (segment, power)
    dw = np.array([1.0 / h, -1.0 / h])  # W' per segment
    z_a = 0.0 + 0j
    z_phi = 0.0 + 0j
    for i in range(2):
        for j in range(2):
            for p in range(2):
                for q in range(2):
                    z_a += cw[i, p] * cw[j, q] * M[p, q, i, j]
            z_phi += dw[i] * dw[j] * M[0, 0, i, j]
    td = float(np.asarray(t_m) @ np.asarray(t_n))
    return 1j * omega * MU0 * td * z_a + z_phi / (1j * omega * eps_c)


# The ladder shortens each tent from lambda_m/200 to lambda_m/1600 at a
# fixed lambda_m/4 separation. Measured |Z_tent - Z_hertzian| / |Z_hertzian|
# over all six SPEC media, both configurations:
#
#   h                lambda_m/200   /400       /800       /1600
#   parallel worst   4.321e-04      1.081e-04  2.703e-05  6.758e-06
#   collinear worst  9.488e-04      2.370e-04  5.925e-05  1.481e-05
#
# Second order in h, ratio 4.00 at every rung of every case — the tent pair
# IS converging to the Hertzian closed form and not merely sitting near it.
# Pinned at the converged end with ~6.8x margin on the measured worst.
G5_TOL = 1e-4
G5_ORDER = 3.5  # measured 4.00 everywhere
G5_LADDER = (1.0, 0.5, 0.25, 0.125)


@pytest.mark.parametrize("config", ["parallel", "collinear"])
@pytest.mark.parametrize("medium", list(MEDIA))
def test_gu1_5_hertzian_mutual_impedance_limit(medium, config, record_property):
    k0, et, km = MEDIA[medium]
    f = FREQS[medium.split("/")[1]]
    omega = 2.0 * np.pi * f
    # k_m is the primary quantity here, so the medium's permittivity is
    # derived FROM it and the Galerkin prefactors stay exactly consistent
    # with the kernel's k. (Deriving it the other way round, EPS0 * et,
    # differs by 5.4e-10 relative: the 2019 SI's mu0 and eps0 do not
    # reproduce the defined c to better than that, and a 1e-14-level gate
    # would be reading the CODATA round-off, not the fill.)
    eps_c = km**2 / (omega**2 * MU0)
    assert abs(eps_c - EPS0 * et) / abs(eps_c) < 1e-8

    lam_m = 2.0 * np.pi / abs(km)
    sep = lam_m / 4.0
    t = np.array([0.0, 0.0, 1.0])
    a = lam_m * 1e-5  # thin

    errs = []
    for frac in G5_LADDER:
        h = frac * lam_m / 200.0
        if config == "parallel":
            o_m = np.array([0.0, 0.0, -h])
            o_n = np.array([sep, 0.0, -h])
        else:
            o_m = np.array([0.0, 0.0, -h])
            o_n = np.array([0.0, 0.0, sep - h])
        got = _tent_mutual_Z(o_m, t, o_n, t, h, a, km, omega, eps_c)
        # Centroid separation (the tents peak at o + h t).
        R_vec = (o_m + h * t) - (o_n + h * t)
        ref = _hertzian_Z(R_vec, t, t, km, omega, h, h)
        errs.append(abs(got - ref) / abs(ref))
    record_property(f"g5_{medium}_{config}", list(map(float, errs)))
    # The Hertzian form is the h -> 0 limit: the ladder must converge...
    assert errs[-1] < errs[0], errs
    # ...at second order (each halving of h cuts the gap by ~4).
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert min(ratios) > G5_ORDER, ratios
    # ...and land inside the pinned tolerance.
    assert errs[-1] <= G5_TOL, f"{medium}/{config}: {errs[-1]:.3e}"


# ======================================================================
# G-U1-6 — the sigma ladder extinguishes the kernel, monotonically
# ======================================================================

G6_SIGMAS = (0.005, 0.05, 0.5, 5.0, 50.0, 500.0)
G6_EPS_R = 13.0
G6_F = 7.0e6


@pytest.mark.parametrize("ek", [None, WHOLE_BLOCK], ids=["reduced", "ek"])
def test_gu1_6_sigma_ladder_monotone_extinction(ek, record_property):
    """Fixed geometry, rising conductivity: the full-kernel moment's
    magnitude falls monotonically to zero (the kernel-level statement of
    phase-0's G3), and every k_m on the ladder decays."""
    h = 0.25
    a = h / 24.0
    lo_i, hi_i = _straight(3, h)
    lo_j, hi_j = _straight(3, h, origin=(2.0, 0.0, 0.0))
    ends = np.arange(4) * h
    mags_off, mags_same = [], []
    for sigma in G6_SIGMAS:
        _k0, et, km = k_medium(G6_EPS_R, sigma, G6_F)
        assert km.imag < 0
        R = np.geomspace(1e-3, 10.0, 41)
        assert np.all(np.abs(np.exp(-1j * km * R)) < 1.0), sigma
        spec = ek
        if ek is not None:
            gi = np.zeros(3, dtype=np.int64)
            spec = _EK(a=None, group_i=gi, group_j=gi)
        off = _seg_seg_full_moments_offedge(
            lo_i, hi_i, lo_j, hi_j, a, km, 2, 8, ek=spec
        )
        mags_off.append(float(np.abs(off[0, 0]).max()))
        same = _seg_seg_static_moments(ends, a, 2, ek=ek) + _seg_seg_reg_moments(
            ends, a, km, 2, 8, ek=ek
        )
        mags_same.append(float(np.abs(same[0, 0]).max()))
    record_property("g6_offedge", mags_off)
    record_property("g6_same_edge", mags_same)
    assert all(mags_off[i] > mags_off[i + 1] for i in range(len(mags_off) - 1)), (
        mags_off
    )
    # Extinction, not merely a downward drift: five decades of sigma kill
    # the off-edge moment outright. Measured 1.541e-03 -> 1.476e-105, i.e.
    # a ratio of 9.6e-103 — the pair sits 2 m apart and the skin depth at
    # sigma = 500 S/m, 7 MHz is 8.5 mm.
    assert mags_off[-1] / mags_off[0] < 1e-90, mags_off[-1] / mags_off[0]
    # The SAME-EDGE block does NOT extinguish with it, and the contrast is
    # the point: its coincidence core is the k-FREE 1/R static term, which
    # survives inside the skin depth however lossy the medium gets. It falls
    # monotonically too, but only by 5.6x across the same five decades
    # (measured 0.11468 -> 0.02065 reduced, 0.11507 -> 0.02239 EK; ratios
    # 0.180 and 0.195). A same-edge block that collapsed like the off-edge
    # one would mean the static extraction had picked up a spurious k
    # dependence.
    assert all(mags_same[i] > mags_same[i + 1] for i in range(len(mags_same) - 1)), (
        mags_same
    )
    assert mags_same[-1] / mags_same[0] > 0.03, mags_same
    assert mags_same[-1] / mags_same[0] > 1e80 * (mags_off[-1] / mags_off[0])


# ======================================================================
# G-U1-7 — n_qp adequacy: a measurement with a conclusion
# ======================================================================
#
# VERDICT: n_qp does not escalate in the medium. The reasoning and the full
# table are in the `_bspline_kernels` in-medium comment; the gates below are
# its anti-regression form.
#
# The first gate is a PARITY ratio, not an error tolerance: at equal |k|h
# the in-medium quadrature error must not exceed the free-space one by more
# than a small factor, because that (and not the medium per se) is what
# would force an escalation. Measured worsts on this gate's own grid
# (|k|h in {0.05 ... 2.0} x the six SPEC media): 1.075 reduced (at |k|h = 2,
# soil A / 7 MHz) and 3.284 EK (at |k|h = 1, soil B / 7 MHz, where the EK
# remainder's ABSOLUTE error is 4.2e-05 against the reduced kernel's
# 3.6e-03). The wider measurement grid behind the docstring reads 1.15 and
# 3.3. A ratio pinned at "5x the measured worst" would be a bound of 5.4 —
# meaningless for a quantity whose correct value is 1 — so these are pinned
# at a factor of two above the measurement instead.
NQP_PARITY_REDUCED = 2.0  # measured worst 1.075 here, 1.15 on the wider grid
NQP_PARITY_EK = 8.0  # measured worst 3.284 here, 3.3 on the wider grid
# Measured 2.535e-04 at 20 segments per quarter lambda_m; ~5.9x margin.
NQP4_ABS_AT_U4_MESH = 1.5e-3


def _reg_gl_error(k, kh, n_qp, ek=None, da=24.0, n=5, ref_nqp=20):
    h = kh / abs(k)
    a = h / da
    ends = np.arange(n + 1) * h
    got = _seg_seg_reg_moments(ends, a, k, 2, n_qp, ek=ek)
    ref = _seg_seg_reg_moments(ends, a, k, 2, ref_nqp, ek=ek)
    return _rel(got, ref)


@pytest.mark.parametrize("ek", [None, WHOLE_BLOCK], ids=["reduced", "ek"])
def test_gu1_7_in_medium_quadrature_costs_what_free_space_costs(ek, record_property):
    da = 24.0 if ek is None else 2.0
    limit = NQP_PARITY_REDUCED if ek is None else NQP_PARITY_EK
    worst, where = 0.0, None
    for kh in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
        base = _reg_gl_error(complex(K0_7, 0.0), kh, 4, ek=ek, da=da)
        for name, (_k0, _et, km) in MEDIA.items():
            r = _reg_gl_error(km, kh, 4, ek=ek, da=da) / base
            if r > worst:
                worst, where = r, (kh, name)
    record_property("g7_worst_ratio", (worst, where))
    assert worst <= limit, f"{where}: {worst:.3f}"


def test_gu1_7_nqp4_is_adequate_at_the_u4_in_medium_mesh(record_property):
    """U4 meshes against lambda_m, so the panel phase |k_m|h is 2pi/(4*N)
    for N segments per quarter in-medium wavelength — the same number the
    free-space mesher produces. At N = 20 that is |k_m|h = 0.0785."""
    kh = 2.0 * np.pi / 80.0
    worst = max(_reg_gl_error(km, kh, 4) for _k0, _et, km in MEDIA.values())
    record_property("g7_nqp4_at_u4_mesh", worst)
    assert worst <= NQP4_ABS_AT_U4_MESH, worst


def test_gu1_7_the_free_space_mesh_is_the_defect_u4_fixes(record_property):
    """Documented, not tolerated: a deck meshed against lambda_0 runs the
    in-medium kernel at |n| times the panel phase, and the n_qp=4 error
    rises with it. The fix is U4's meshing rule, not a quadrature order —
    this gate pins the SIZE of the effect so U4 can be checked against it."""
    k0, _et, km = MEDIA[WORST]
    n_index = abs(km) / k0
    assert n_index == pytest.approx(8.9213, rel=1e-4)
    kh_lam0 = 2.0 * np.pi / 80.0
    good = _reg_gl_error(km, kh_lam0, 4)
    bad = _reg_gl_error(km, kh_lam0 * n_index, 4)
    record_property("g7_under_mesh", (good, bad, bad / good))
    # Measured: 2.535e-04 correctly meshed, 2.416e-03 under-meshed, ratio
    # 9.53. Bracketed on both sides — a gate that only capped the number
    # would pass if U4 (or anything else) quietly made the effect vanish,
    # and the effect is real physics: it must be FIXED by meshing, not by
    # this file stopping to notice it.
    assert 5.0 < bad / good < 20.0, (good, bad)
    assert 1e-3 < bad < 1e-2, bad


# ======================================================================
# G-U1-8 — the refusals fire, by name
# ======================================================================


def test_gu1_8_im_k_positive_refuses_naming_the_convention():
    with pytest.raises(ValueError, match=r"Im k <= 0 so that e\^\{-jkR\} decays"):
        _complex_k(1.0 + 0.5j)
    # And through a real entry point, not just the predicate.
    ends = np.arange(4) * 0.1
    with pytest.raises(ValueError, match=r"Im k > 0"):
        _seg_seg_reg_moments(ends, 0.005, 1.0 + 0.5j, 2, 4)
    lo, hi = _straight(2, 0.1)
    lo_j, hi_j = _straight(2, 0.1, origin=(0.5, 0.0, 0.0))
    with pytest.raises(ValueError, match=r"Im k > 0"):
        _seg_seg_full_moments_offedge(lo, hi, lo_j, hi_j, 0.005, 1.0 + 0.5j, 2, 4)
    with pytest.raises(ValueError, match=r"Im k > 0"):
        _seg_seg_full_moments_offedge_swept(
            lo, hi, lo_j, hi_j, 0.005, np.array([1.0 - 0.5j, 1.0 + 0.5j]), 2, 4
        )
    # It refuses rather than conjugating: the caller's k comes back untouched
    # nowhere, and the message says where the branch belongs.
    with pytest.raises(ValueError, match=r"_sommerfeld\._d12"):
        _complex_k(np.array([1.0 + 1e-30j]))


def test_gu1_8_a_swept_complex_k_is_never_silently_truncated():
    """`np.asarray(k, dtype=float)` drops the imaginary part with a warning,
    not an error. The widened path must keep it; the un-widened paths must
    refuse."""
    h, a = 0.12, 0.005
    km = MEDIA[WORST][2]
    ends = np.arange(4) * h
    geo = _seg_seg_reg_geometry(ends, a, max_d=2, n_qp=4)
    swept = _seg_seg_reg_moments_from_geometry_swept(geo, np.array([km]))
    single = _seg_seg_reg_moments_from_geometry(geo, km)
    truncated = _seg_seg_reg_moments_from_geometry(geo, km.real)
    assert swept.dtype == np.complex128
    assert _rel(swept[0], single) <= SWEPT_SINGLE_AGREEMENT
    assert _rel(swept[0], truncated) > 1e-3, "the imaginary part made no difference"


def test_gu1_8_the_enrichment_fill_refuses_complex_k():
    lam = C0 / 30.0e6
    solver = BSplineSolver(
        **_G1_DECK, wavelength=lam, degree=2, use_singular_enrichment=True
    )
    geom = solver._build_geometry()
    solver.k = complex(solver.k, -0.3)
    # It refuses BEFORE touching its arguments — the two Nones would be a
    # TypeError one line later if the guard were not first.
    with pytest.raises(ValueError, match=r"singular-enrichment fill does not serve"):
        solver._enrichment_Z_assemble(geom, None, None)


def test_gu1_8_the_swept_same_edge_fill_refuses_complex_k():
    lam = C0 / 30.0e6
    solver = BSplineSolver(**_G1_DECK, wavelength=lam, degree=2)
    prep = solver._same_edge_prep(solver._build_geometry())
    ks = np.array([2 * np.pi / lam - 0.1j])
    with pytest.raises(ValueError, match=r"swept B-spline same-edge fill"):
        list(solver._same_edge_prep_swept_chunks(prep, ks))


def test_gu1_8_every_swept_entry_point_refuses_complex_k():
    """The whole silent-truncation class, not just the one site: each of
    these methods opened with `np.asarray(k_array, dtype=float)`, which
    drops the imaginary part with a ComplexWarning nobody sees."""
    lam = C0 / 30.0e6
    ks = np.array([2 * np.pi / lam - 0.1j, 2 * np.pi / lam - 0.2j])
    bs = BSplineSolver(**_G1_DECK, wavelength=lam, degree=2)
    hm = HMatrixSolver(**_G1_DECK, wavelength=lam, degree=2, nsegs=21)
    with pytest.raises(ValueError, match=r"compute_impedance_swept does not serve"):
        bs.compute_impedance_swept(ks)
    with pytest.raises(ValueError, match=r"_port_solutions_swept does not serve"):
        list(bs._port_solutions_swept(ks))
    with pytest.raises(ValueError, match=r"compute_impedance_swept does not serve"):
        hm.compute_impedance_swept(ks)
    with pytest.raises(ValueError, match=r"_port_solutions_swept does not serve"):
        list(hm._port_solutions_swept(ks))


def test_gu1_8_hmatrix_zblock_refuses_complex_k():
    lam = C0 / 30.0e6
    solver = HMatrixSolver(**_G1_DECK, wavelength=lam, degree=2, nsegs=21)
    idx = np.arange(3, dtype=np.int64)
    with pytest.raises(ValueError, match=r"HMatrixSolver\.zblock does not serve"):
        solver.zblock(idx, idx, k=complex(2 * np.pi / lam, -0.2))


def test_gu1_8_refuse_complex_k_names_the_caller():
    _refuse_complex_k(1.0, "anything")  # real: silent
    _refuse_complex_k(np.array([1.0, 2.0]), "anything")
    with pytest.raises(ValueError, match=r"the widget does not serve complex k"):
        _refuse_complex_k(1.0 - 1j, "the widget")


def test_gu1_8_the_solver_family_is_untouched_by_the_widening():
    """The widened path is the kernel layer only: no solver grew a medium
    in unit 1, and `bspline`'s module namespace is the place a stray
    complex-k solver knob would first show up."""
    assert not hasattr(BSplineSolver, "medium")
    assert not hasattr(BSplineSolver, "eps_tilde")
    sig = inspect.signature(BSplineSolver.__init__)
    assert not [p for p in sig.parameters if "medium" in p]
    assert _bs._refuse_complex_k is _refuse_complex_k
