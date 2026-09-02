"""The razor fill's moment kernel at an in-medium (complex) k — momwire#796.

`razor_seg_moments` took `double k` and `_FusedMoments.evaluate` handed it
`float(k)`, so a complex wavenumber died in the cast with a `TypeError` and no
fallback. Nothing served today reaches that line — razor's Sommerfeld contact
decks keep every segment above the plane, so the kernel's k is the free-space
one — but the below-plane fill hands every in-medium pair `k = k0*sqrt(eps_c)`
with `Im k <= 0`, and #778 had already opened this door on bspline's off-edge
kernel.

What this module gates:

  * **the two lanes agree at complex k**, on BOTH of razor's branches. They do
    not transfer identically: the reduced branch takes #778's real scale
    factor on the trig part of `exp(-jkR) - 1` (the `- 1` and the k-free
    statics untouched), while the EK branch's `C1 = 1 + jkR` and
    `C2 = 3*C1 - (kR)^2` become genuinely complex in `kR` and pick up terms in
    `Im(k)*R`. The C++ is written term for term against `_ek_factor` /
    `_ek_reg_extra`, which are numpy-generic and already correct there.
    NOT bitwise, for the reason the #742 module states: the kernel's per-pair
    reduction is not numpy's. Measured on this box across the six media below,
    both branches, `need_m1` both ways, the worst deviations are

        max|dM0| / max|M0|    2.5e-16
        max|dM1| / max|M1|    1.8e-15

    and the bars sit at 1e-10 — not for slack, but because the post-merge
    macOS lane measured 8.4e-13 on the reduced deck's M0 (the libm seam under
    the `exp(-jkR) - 1` cancellation at kR ~ 1e-4; the note above the bars
    has the mechanism), which a decade of headroom over the Linux numbers
    did not cover.

  * **real k did not move.** The <false> instantiation is textually the
    pre-#796 loop; that it is also bit-identical after a rebuild is the #762
    protocol and was run by hand for this change (12/12 arrays `array_equal`,
    both branches, three wavenumbers). What is gated HERE is the cheaper
    standing claim: a real k still reaches the real-k kernel and not the new
    one.

  * **the fallback is real.** The #822 lesson: a build with the real-k fill
    but WITHOUT `razor_seg_moments_cplx` must take the numpy lane on a complex
    k — never `float(k)` — and the forced-off twin asserts the fused
    complex-k count is zero, not merely that the answer looks right.

  * **`Im k > 0` is refused as a `ValueError`**, from the Python gate and from
    the C++ entry point independently. That is the growing-exponential branch;
    the entry throws `std::invalid_argument` (pybind11 maps it to ValueError)
    rather than the `std::runtime_error` its #778 bspline twin throws, which
    predates #796's ask for a ValueError.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import RazorSolver
from momwire import razor as _razor
from momwire._accel import _CANCELLABLE_KERNELS
from momwire._accel import acc as _acc

WL = 20.0
L_DIPOLE = 0.485 * WL
NSEGS = 24

needs_accel = pytest.mark.skipif(
    not _razor._use_razor_fill_accel(),
    reason="razor fill accelerator not built, or forced off for this run",
)
needs_cplx = pytest.mark.skipif(
    not _razor._use_razor_cplx_accel(),
    reason="razor complex-k kernel not built, or forced off for this run",
)

# The media table, transcribed from `test_complex_k_bspline.py` (itself
# `scratch/524-phase0/SPEC.md`, verbatim) so the two kernels are exercised on
# the same soils rather than on two private lists that can drift apart.
EPS0 = 8.8541878128e-12
C0 = 299792458.0
SOILS = {"A": (13.0, 0.005), "B": (20.0, 0.03), "C": (5.0, 0.001)}
FREQS = {"7MHz": 7.0e6, "21MHz": 21.0e6}


def _k_medium(eps_r, sigma, f_hz):
    """k_m = k0*sqrt(eps_tilde) on the Im <= 0 branch."""
    k0 = 2.0 * np.pi * f_hz / C0
    et = eps_r - 1j * sigma / (2.0 * np.pi * f_hz * EPS0)
    km = k0 * np.sqrt(et)
    if km.imag > 0:  # the principal sqrt already lands here; assert, don't trust
        km = np.conj(km)
    return complex(km)


MEDIA = {f"{s}/{fn}": _k_medium(*SOILS[s], FREQS[fn]) for s in SOILS for fn in FREQS}

# Both branches of the fill. The EK deck's radius is set so the pairs are
# coaxial-eligible (observer on its own wire's axis at equal radius), which is
# what puts the NEC Eq 89 factor in play rather than the reduced remainder.
DECKS = {
    "reduced": dict(
        wires=[[(0.0, 0.0, -L_DIPOLE / 2), (0.0, 0.0, L_DIPOLE / 2)]],
        nsegs=NSEGS,
        wire_radius=5e-4,
    ),
    "ek": dict(
        wires=[[(0.0, 0.0, -L_DIPOLE / 2), (0.0, 0.0, L_DIPOLE / 2)]],
        nsegs=NSEGS,
        wire_radius=L_DIPOLE / NSEGS / 4.0,
        extended_kernel=True,
    ),
}

# The bars are NOT the Linux numbers with a decade of headroom, and the reason
# is the first thing this module learned on main: the post-merge macOS lane
# (which a PR never runs) read 8.2e-13..8.4e-13 on the reduced deck's M0, on
# EVERY medium and both `need_m1` settings, with the EK deck and both M1s
# still passing. That is not the kernel. It is the libm seam under the
# `exp(-jkR) - 1` cancellation: the thin reduced deck's near-self pairs sit at
# kR ~ 1e-4, so a one-ulp disagreement between numpy's complex exp and the
# kernel's exp*cos is amplified ~1e4 times by the subtraction of 1, and on
# macOS the two lanes do not share a libm the way they do under glibc (where
# they agree to 2.5e-16 / 1.8e-15). The fat EK deck's kR is ~1e3 larger and
# the amplification is gone, which is why it passed unchanged. The per-medium
# sensitivity is largest at SMALL |k| (soil C at 7 MHz: 2.8e-12 per ulp,
# 20x soil B's), which the max-normalisation hides in the observed spread;
# a runner two ulps apart would land near 5e-12. 1e-10 keeps the module's
# stated headroom (~100x over the macOS reading) and sits three decades under
# anything the solved impedance can feel.
M0_BAR = 1e-10  # max|dM0| / max|M0|; Linux 2.5e-16, macOS 8.4e-13
M1_BAR = 1e-10  # max|dM1| / max|M1|; Linux 1.8e-15, macOS passes (reduced-deck M1 read under the 1e-11 bar)


def _grab(name):
    """The (obs, geom, a, ek) that one real solve's prepare was handed.

    Taken off a live solve rather than assembled here so the source set under
    test is the one the solver actually integrates — the momwire#745 property,
    which a hand-built geometry would quietly give up.
    """
    grabbed = {}
    original = RazorSolver._seg_moments_prepare

    def spy(self, obs, geom, a, *, ek=None):
        if not grabbed:
            grabbed.update(obs=obs, geom=geom, a=a, ek=ek, solver=self)
        return original(self, obs, geom, a, ek=ek)

    RazorSolver._seg_moments_prepare = spy
    try:
        RazorSolver(
            wavelength=WL, feed_arclength=0.25 * L_DIPOLE, **DECKS[name]
        ).compute_impedance()
    finally:
        RazorSolver._seg_moments_prepare = original
    return grabbed


def _both_prepared(name, monkeypatch):
    """The same source set prepared BOTH ways: fused token and numpy chunks."""
    g = _grab(name)
    solver, obs, geom, a, ek = g["solver"], g["obs"], g["geom"], g["a"], g["ek"]
    fused = _razor._FusedMoments(obs, geom, a, ek, solver.n_qp_source)
    monkeypatch.setattr(_razor, "_FORCE_NUMPY", True)
    chunks = solver._seg_moments_prepare(obs, geom, a, ek=ek)
    monkeypatch.setattr(_razor, "_FORCE_NUMPY", False)
    assert isinstance(chunks, _razor._PreparedChunks)
    return solver, fused, chunks, obs.shape[0]


# ==========================================================================
# The build carries it
# ==========================================================================
def test_this_build_carries_the_complex_kernel():
    """The capability flag is the kernel's OWN symbol, never a shared one.

    A .so built before #796 exports `razor_seg_moments` and not its complex
    twin, so a flag shared with #742 would advertise a contract it cannot
    serve. Skipped only where the extension genuinely is not built.
    """
    if _acc is None:
        pytest.skip("accelerator extension not built")
    assert bool(getattr(_acc, "razor_cplx_796", False))
    assert hasattr(_acc, "razor_seg_moments_cplx")
    assert _razor._HAVE_RAZOR_CPLX_ACCEL


def test_the_complex_kernel_is_cancellable():
    """An aborted complex-k fill must surface as `SolveAborted` like its
    real-k sibling, which means being listed for the cancel remap. Structural
    rather than behavioural: the remap itself is gated for `razor_seg_moments`
    in the #742 module and the two share one kernel body and one poll."""
    assert "razor_seg_moments_cplx" in _CANCELLABLE_KERNELS


# ==========================================================================
# The two lanes agree, at complex k, on both branches
# ==========================================================================
@needs_cplx
@pytest.mark.parametrize("need_m1", (True, False))
@pytest.mark.parametrize("medium", sorted(MEDIA))
@pytest.mark.parametrize("branch", sorted(DECKS))
def test_the_fused_complex_fill_agrees_with_the_numpy_lane(
    monkeypatch, branch, medium, need_m1
):
    solver, fused, chunks, n_obs = _both_prepared(branch, monkeypatch)
    k = MEDIA[medium]
    ref0, ref1 = solver._seg_moments_from_prepared(chunks, k, n_obs, need_m1=need_m1)
    got0, got1 = solver._seg_moments_from_prepared(fused, k, n_obs, need_m1=need_m1)
    # Measure both planes BEFORE asserting either, so a lane that misses one
    # bar still reports the other (the macOS reading above had no M1 number
    # because the M0 assertion fired first).
    d0 = float(np.abs(got0 - ref0).max() / np.abs(ref0).max())
    if need_m1:
        d1 = float(np.abs(got1 - ref1).max() / np.abs(ref1).max())
    else:
        assert got1 is None and ref1 is None
        d1 = None
    report = f"{branch}/{medium}: max|dM0|/max|M0| = {d0:.3e}"
    if d1 is not None:
        report += f", max|dM1|/max|M1| = {d1:.3e}"
    assert d0 < M0_BAR, report
    if d1 is not None:
        assert d1 < M1_BAR, report


@needs_cplx
@pytest.mark.parametrize("branch", sorted(DECKS))
def test_the_real_k_gate_still_takes_the_real_kernel(monkeypatch, branch):
    """A real k must not start going through the complex entry.

    The two agree numerically, so an answer check would not notice; count the
    entries instead (the #822 lesson applied to the gate rather than to the
    dispatch).
    """
    solver, fused, _chunks, n_obs = _both_prepared(branch, monkeypatch)
    calls = {"real": 0, "cplx": 0}
    real_fn, cplx_fn = _acc.razor_seg_moments, _acc.razor_seg_moments_cplx

    def count_real(*a, **kw):
        calls["real"] += 1
        return real_fn(*a, **kw)

    def count_cplx(*a, **kw):
        calls["cplx"] += 1
        return cplx_fn(*a, **kw)

    monkeypatch.setattr(_acc, "razor_seg_moments", count_real)
    monkeypatch.setattr(_acc, "razor_seg_moments_cplx", count_cplx)
    solver._seg_moments_from_prepared(fused, 2.0 * np.pi / WL, n_obs)
    assert (calls["real"], calls["cplx"]) == (1, 0)
    solver._seg_moments_from_prepared(fused, MEDIA["B/7MHz"], n_obs)
    assert (calls["real"], calls["cplx"]) == (1, 1)


# ==========================================================================
# The fallback, and that it is really the fallback
# ==========================================================================
@needs_accel
@pytest.mark.parametrize("branch", sorted(DECKS))
def test_a_build_without_the_complex_kernel_takes_the_numpy_lane(monkeypatch, branch):
    """The #822 lesson. Two independent tells: the fused complex-k kernel is
    never entered, AND the answer is bit-identical to the numpy lane's — which
    it can only be if the numpy lane is literally what ran."""
    solver, fused, chunks, n_obs = _both_prepared(branch, monkeypatch)
    calls = {"n": 0}
    cplx_fn = _acc.razor_seg_moments_cplx

    def counting(*a, **kw):
        calls["n"] += 1
        return cplx_fn(*a, **kw)

    monkeypatch.setattr(_acc, "razor_seg_moments_cplx", counting)
    monkeypatch.setattr(_razor, "_HAVE_RAZOR_CPLX_ACCEL", False)
    k = MEDIA["B/7MHz"]
    got0, got1 = solver._seg_moments_from_prepared(fused, k, n_obs)
    assert calls["n"] == 0, "the fused complex kernel ran with its flag off"
    monkeypatch.setattr(_razor, "_HAVE_RAZOR_CPLX_ACCEL", True)
    ref0, ref1 = solver._seg_moments_from_prepared(chunks, k, n_obs)
    assert np.array_equal(got0, ref0)
    assert np.array_equal(got1, ref1)


@needs_accel
def test_the_fallback_never_casts_a_complex_k(monkeypatch):
    """The literal #796 crash: `float(k)` on a complex k. A build without the
    complex kernel must reach the numpy lane instead of raising TypeError."""
    solver, fused, _chunks, n_obs = _both_prepared("reduced", monkeypatch)
    monkeypatch.setattr(_razor, "_HAVE_RAZOR_CPLX_ACCEL", False)
    m0, _m1 = solver._seg_moments_from_prepared(fused, MEDIA["A/21MHz"], n_obs)
    assert np.iscomplexobj(m0) and np.isfinite(m0).all()


# ==========================================================================
# Im k > 0 is refused, twice over
# ==========================================================================
@needs_cplx
def test_the_gate_refuses_a_growing_exponential(monkeypatch):
    solver, fused, _chunks, n_obs = _both_prepared("reduced", monkeypatch)
    with pytest.raises(ValueError, match="Im k > 0"):
        solver._seg_moments_from_prepared(fused, complex(0.3, +0.1), n_obs)


@needs_cplx
def test_the_entry_point_refuses_a_growing_exponential(monkeypatch):
    """Independently of the gate: the C++ re-asserts the convention, so a
    caller reaching the kernel directly cannot evade it."""
    _solver, fused, _chunks, _n_obs = _both_prepared("reduced", monkeypatch)
    with pytest.raises(ValueError, match="Im k > 0"):
        _acc.razor_seg_moments_cplx(
            fused.obs,
            fused.seg_p0,
            fused.seg_t,
            fused.seg_h,
            fused.a,
            fused.xg,
            fused.wg,
            complex(0.3, +0.1),
            True,
            fused.group_i,
            fused.group_j,
            fused.a_ek,
            0,
        )
