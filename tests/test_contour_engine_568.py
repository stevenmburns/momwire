"""The shared adaptive-contour engine in C++ (momwire#568 unit 1).

`_sommerfeld_below`'s five contour functions -- `_gauss_segment`,
`_adaptive_segment`, `_wynn_epsilon`, `_head`, `_tail_below` and their
`_run_contour` driver -- are where the buried and transmitted fills spend
their minutes. Unit 1 ports that ENGINE to C++, templated on the integrand,
so U2 (below fills) and U3 (transmitted fills) can instantiate it with their
own integrands. Nothing in the production dispatch changes here: the numpy
engine is still what ships, and it is the reference these gates measure
against.

What this file gates, and why each gate exists:

* **The complex Bessel pair.** The engine's integrands all call
  `_sommerfeld._bessel_j0_j1x`, which is `scipy.special.jv(0|1, x)` at
  COMPLEX x. C++ has no scipy, so `_contour_engine_inline.h` carries its own
  J0(x) / J1(x)/x. It is gated point-by-point against scipy over the whole
  production argument domain.
* **An analytic ground truth.** The Sommerfeld identity
  `int_0^inf J0(lam rho) e^{-gamma h} lam/gamma dlam = e^{-jkR}/R` has the
  same branch point, the same oscillatory tail and the same exponential decay
  as the production integrands, and a closed form. The C++ engine is run on
  it over the SPEC soils and gated against that closed form.
* **Parity with the numpy engine** on a synthetic NC = 6 integrand whose twin
  is written below, so both engines chew the same mathematics.
* **One test per measured phase-0 guard**, exercising the C++ path.

TOLERANCES, NEVER BYTE EQUALITY. The two engines take the same adaptive
decisions but are not bit-twins: the Gauss dot product reduces in a different
order, complex division is libgcc's rather than numpy's, and the Bessel pair
is a different algorithm entirely.
"""

import time

import numpy as np
import pytest
from scipy.special import jv

import momwire._sommerfeld_below as below
from momwire._sommerfeld import _bessel_j0_j1x

accel = pytest.importorskip("momwire._accelerators")

if not getattr(accel, "contour_engine_568", False):
    pytest.skip(
        "the accelerator predates momwire#568 unit 1 (stale .so?)",
        allow_module_level=True,
    )

EPS0 = 8.8541878128e-12
C0 = 299792458.0

# The production fine machine, node-for-node: `_sommerfeld_below._GX/_GW`.
GX, GW = np.polynomial.legendre.leggauss(below._GAUSS_N)
GXC, GWC = np.polynomial.legendre.leggauss(below._GAUSS_N_COARSE)


def eps_tilde(eps_r, sigma, f_hz):
    """eps~ = eps_r - j sigma/(omega eps0), as `test_sommerfeld_below` writes it."""
    return eps_r - 1j * sigma / (2.0 * np.pi * f_hz * EPS0)


# The SPEC soils, taken from `golden_below_below_524.SOILS` rather than
# re-invented: A is the reference loam, B the high-loss stressor, C the dry
# rock. Both SPEC frequencies.
SPEC_SOILS = {"A": (13.0, 0.005), "B": (20.0, 0.03), "C": (5.0, 0.001)}
SPEC_FREQS = (7.0e6, 21.0e6)


def medium(soil, f_hz):
    """(eps~, k_p, k_m, lambda_m) for one SPEC cell."""
    er, sg = SPEC_SOILS[soil]
    et = eps_tilde(er, sg, f_hz)
    k_p = 2.0 * np.pi * f_hz / C0
    k_m = below.k_medium(et, k_p)
    return et, k_p, k_m, 2.0 * np.pi / abs(k_m)


# ======================================================================
# The numpy twins
# ======================================================================
#
# Both are written HERE, not imported, so that a change to the production
# integrands can never silently redefine what these gates compare.


def np_identity(lam, rho, h, k):
    """The Sommerfeld-identity integrand, stacked (1, n) for `_run_contour`.

        f(lam) = J0(lam rho) e^{-gamma h} lam / gamma,  gamma = sqrt(lam^2 - k^2)

    Principal sqrt (Re gamma >= 0) is the branch the engine's FIRST-quadrant
    head detour selects: under e^{+j omega t} the cut runs downward from +k,
    an upward detour never crosses it, and at lam = 0 the signed zero puts
    sqrt on +jk -- the outgoing plane wave.
    """
    lam = np.asarray(lam, dtype=np.complex128)
    g = np.sqrt(lam * lam - k * k)
    b0, _ = _bessel_j0_j1x(lam * rho)
    return np.stack([b0 * np.exp(-g * h) * lam / g])


def np_synth6(lam, rho, h, k_p, k_m):
    """The synthetic NC = 6 integrand's twin -- the same spelling, in the same
    order, as `mw568::Synth6` in `_accelerators.cpp`.

    Deliberately not the production kernel: a pole-free rational weight
    `1/(gamma_m + lam + k_p)` stands in for the Sommerfeld D-pair, so U2 is
    free to write the real integrand however it likes, while the SHAPE the
    engine cares about is all here -- a gamma-decay factor, both Bessel
    pieces, and components spanning several decades so the vector max-norm
    test in `_adaptive_segment` actually has something to choose between.
    """
    lam = np.asarray(lam, dtype=np.complex128)
    gm = np.sqrt(lam * lam - k_m * k_m)
    e = np.exp(-gm * h)
    x = lam * rho
    b0, b1x = _bessel_j0_j1x(x)
    w = 1.0 / (gm + lam + k_p)
    l2 = lam * lam
    return np.stack(
        [
            w * e * b0 * lam,
            w * e * (b1x - b0) * l2,
            -(w * e * b1x * x * gm * lam),
            -(w * e * b1x * l2),
            w * e * b0,
            w * e * (b0 + 2.0 * b1x) * lam * gm,
        ]
    )


def closed_form(k, rho, h):
    """e^{-jkR}/R -- what the Sommerfeld identity's contour integral IS."""
    R = float(np.hypot(rho, h))
    return np.exp(-1j * k * R) / R


def cpp_identity(rho, h, k_p, k_m, use_k_m=False, rtol=1e-11, max_panels=4000):
    return accel.contour_engine_sommerfeld_identity(
        rho,
        h,
        k_p,
        k_m,
        use_k_m,
        rtol,
        below._ADAPT_DEPTH,
        below._DETOUR,
        GX,
        GW,
        max_panels,
    )


def cpp_synth6(rho, h, k_p, k_m, rtol=1e-11, max_panels=4000):
    return accel.contour_engine_synth6(
        rho,
        h,
        k_p,
        k_m,
        rtol,
        below._ADAPT_DEPTH,
        below._DETOUR,
        GX,
        GW,
        max_panels,
    )


# ======================================================================
# G-568-1 — the complex Bessel pair against scipy
# ======================================================================
#
# The production argument domain: on the head contour x = lam*rho is complex
# with |Im x| held under ~2 by the detour cap (the gate goes to 4 for margin);
# on the tail x is real and can run to the hundreds.
#
# The error is measured RELATIVE TO THE BESSEL ENVELOPE
# sqrt(2/(pi|x|)) e^{|Im x|}, not to |J0(x)| alone, and that floor is not a
# weakening — it is the only well-posed relative measure ACROSS A ZERO, where
# the function passes through 0 and no independent implementation can agree
# with scipy relatively. The two are measured and pinned separately so the
# distinction is on the record rather than assumed:
#
#   envelope-relative, whole domain   worst 6.8e-14  (J0), 7.0e-14 (J1/x)
#   PURE relative, |J0| > 0.1*env     worst 5.6e-14  (J0), 1.5e-13 (J1)
#   PURE relative, everywhere         worst 2.8e-12 at x = 11.7917, which is
#                                     1.3e-4 from the FOURTH zero of J0, where
#                                     |J0| = 3.1e-5 -- four decades under the
#                                     envelope, i.e. inside the notch
#
# 60,001 real points x in [1e-6, 500] plus 60,000 complex with |Im x| <= 4.
# Both gates pin at 1e-12: ~15x margin on the envelope-relative measure and
# ~7x on the strict one. These last digits move by a factor of ~2 between
# builds (GCC's inlining decides where the FMAs land), so the pins are set
# an order of magnitude out, not at the measurement.

G_568_1_TOL = 1e-12

# Where a pure-relative comparison is meaningful: within one decade of the
# Bessel envelope, i.e. not inside a zero's cancellation notch.
G_568_1_ZERO_GUARD = 0.1


def _bessel_envelope(x):
    return np.sqrt(2.0 / (np.pi * np.abs(x))) * np.exp(np.abs(x.imag))


def _bessel_worst(x):
    """(worst envelope-relative J0, worst envelope-relative J1/x)."""
    x = np.asarray(x, dtype=np.complex128)
    got0, got1x = accel.bessel_j0_j1x_complex(x)
    ref0 = jv(0, x)
    ref1x = jv(1, x) / x
    env = _bessel_envelope(x)
    e0 = np.abs(got0 - ref0) / np.maximum(np.abs(ref0), env)
    e1 = np.abs(got1x - ref1x) / np.maximum(np.abs(ref1x), env / np.abs(x))
    return float(e0.max()), float(e1.max())


def test_g5681_bessel_matches_scipy_relatively_away_from_the_zeros(
    record_property,
):
    """The strict measure, with the zeros' cancellation notches excluded by
    name rather than by a floor: nothing here is divided by a number the
    envelope says should not be small."""
    x = np.linspace(1e-6, 500.0, 60001).astype(np.complex128)
    got0, got1x = accel.bessel_j0_j1x_complex(x)
    ref0 = jv(0, x)
    ref1 = jv(1, x)
    env = _bessel_envelope(x)
    keep0 = np.abs(ref0) > G_568_1_ZERO_GUARD * env
    keep1 = np.abs(ref1) > G_568_1_ZERO_GUARD * env
    w0 = float(np.max(np.abs(got0[keep0] - ref0[keep0]) / np.abs(ref0[keep0])))
    w1 = float(
        np.max(np.abs(got1x[keep1] * x[keep1] - ref1[keep1]) / np.abs(ref1[keep1]))
    )
    record_property("pure_rel_j0", w0)
    record_property("pure_rel_j1", w1)
    record_property("sampled_fraction", float(keep0.mean()))
    assert keep0.mean() > 0.5, "the guard must not throw away the domain"
    assert max(w0, w1) < G_568_1_TOL, f"strict: {w0:.3e} / {w1:.3e}"


def test_g5681_bessel_matches_scipy_on_the_real_axis(record_property):
    """The tail's whole domain: real x from the series switch out to 500."""
    x = np.linspace(1e-6, 500.0, 60001)
    w0, w1 = _bessel_worst(x)
    record_property("worst_j0", w0)
    record_property("worst_j1x", w1)
    assert w0 < G_568_1_TOL, f"J0 worst {w0:.3e}"
    assert w1 < G_568_1_TOL, f"J1/x worst {w1:.3e}"


@pytest.mark.parametrize(
    "lo,hi",
    [(1e-6, 1.0), (1.0, 8.0), (8.0, 25.0), (25.0, 100.0), (100.0, 500.0)],
)
def test_g5681_bessel_holds_in_every_regime(lo, hi, record_property):
    """One band per algorithm switch -- series, Miller, asymptotic -- plus the
    handoffs at 8 and 25, so a regression cannot hide inside one branch."""
    x = np.linspace(lo, hi, 8001)
    w0, w1 = _bessel_worst(x)
    record_property("band", f"[{lo}, {hi}]")
    record_property("worst_j0", w0)
    record_property("worst_j1x", w1)
    assert max(w0, w1) < G_568_1_TOL, f"[{lo},{hi}]: {w0:.3e} / {w1:.3e}"


def test_g5681_bessel_matches_scipy_off_the_real_axis(record_property):
    """The head's domain: |Im x| <= 4 (twice what the detour cap allows),
    |x| out to 500."""
    rng = np.random.default_rng(568)
    x = rng.uniform(0.0, 500.0, 60000) + 1j * rng.uniform(-4.0, 4.0, 60000)
    w0, w1 = _bessel_worst(x)
    record_property("worst_j0", w0)
    record_property("worst_j1x", w1)
    assert max(w0, w1) < G_568_1_TOL, f"complex: {w0:.3e} / {w1:.3e}"


def test_g5681_bessel_keeps_the_small_x_series_switch():
    """`_bessel_j0_j1x` switches to 1 - x^2/4 and 1/2 - x^2/16 at |x| < 1e-6 so
    rho -> 0 is safe. The C++ twin switches at the same place, to the same two
    expressions -- gate it directly, because scipy's own jv is not the
    reference on that branch (it is what the branch EXISTS to avoid)."""
    x = np.array([1e-9, 1e-8 + 1e-9j, -3e-7, 5e-7 - 2e-7j], dtype=np.complex128)
    got0, got1x = accel.bessel_j0_j1x_complex(x)
    want0, want1x = _bessel_j0_j1x(x)
    assert np.max(np.abs(got0 - want0)) < 1e-15
    assert np.max(np.abs(got1x - want1x)) < 1e-15


# ======================================================================
# G-568-2 — the engine against the Sommerfeld identity's closed form
# ======================================================================
#
# The identity carries the same structure the production integrands do -- a
# branch point at lam = k that the head detours over, an oscillatory J0 tail
# panelled at its zeros, an e^{-gamma h} decay -- and it has an answer:
# e^{-jkR}/R. Running the C++ engine on it gates head, tail, the adaptive
# bisection and the Bessel pair TOGETHER against arithmetic, not against
# another implementation.
#
# The sweep is every SPEC soil x both SPEC frequencies x three geometries,
# with the integrand built on the free-space k_p AND on the in-medium k_m
# (Im k_m <= 0), because the a/marks logic reads both.
#
# MEASURED at rtol 1e-11: worst 1.1e-11 over the SPEC matrix (soil B at
# 7 MHz, use_k_m) and 1.8e-11 over the grazing rows -- i.e. between one and
# two times rtol, which is what an engine that has actually converged looks
# like. Pinned at 100x rtol, the bar the brief sets.

G_568_2_RTOL = 1e-11
G_568_2_TOL = 100.0 * G_568_2_RTOL


@pytest.mark.parametrize("soil", sorted(SPEC_SOILS))
@pytest.mark.parametrize("f_hz", SPEC_FREQS)
@pytest.mark.parametrize("use_k_m", [False, True])
def test_g5682_the_identity_over_the_spec_matrix(soil, f_hz, use_k_m, record_property):
    _et, k_p, k_m, lam_m = medium(soil, f_hz)
    k = k_m if use_k_m else complex(k_p, 0.0)
    worst = 0.0
    for rho_wl, h_wl in ((0.3, 0.2), (1.0, 0.05), (0.02, 0.01)):
        rho, h = rho_wl * lam_m, h_wl * lam_m
        val, hp, tp, conv, acc = cpp_identity(
            rho, h, k_p, k_m, use_k_m, rtol=G_568_2_RTOL
        )
        ref = closed_form(k, rho, h)
        assert conv, f"{soil}/{f_hz:g} ({rho_wl}, {h_wl}) did not converge"
        assert not acc, "a converged tail must not have gone through Wynn"
        assert hp > 0 and tp > 0
        worst = max(worst, abs(val - ref) / abs(ref))
    record_property("worst_rel", float(worst))
    assert worst < G_568_2_TOL, f"{soil}/{f_hz:g} use_k_m={use_k_m}: {worst:.3e}"


@pytest.mark.parametrize("soil", sorted(SPEC_SOILS))
def test_g5682_the_identity_survives_grazing_h(soil, record_property):
    """h -> 0 at fixed rho is where the tail stops decaying and the panel
    budget starts to matter. These still converge inside it; the Wynn
    territory beyond is G-568-5."""
    _et, k_p, k_m, lam_m = medium(soil, 7.0e6)
    worst = 0.0
    for h_wl in (0.05, 0.01, 0.004):
        rho, h = 1.0 * lam_m, h_wl * lam_m
        val, _hp, tp, conv, acc = cpp_identity(
            rho, h, k_p, k_m, True, rtol=G_568_2_RTOL
        )
        ref = closed_form(k_m, rho, h)
        assert conv and not acc, f"{soil} h={h_wl}: conv={conv} accel={acc}"
        assert tp < 4000
        worst = max(worst, abs(val - ref) / abs(ref))
    record_property("worst_rel", float(worst))
    assert worst < G_568_2_TOL, f"{soil}: {worst:.3e}"


# ======================================================================
# G-568-3 — engine parity with numpy on the shared synthetic integrand
# ======================================================================
#
# Both engines, same mathematics, same Gauss rule, same rtol/depth/detour.
# They follow the same adaptive decisions but are NOT bit-twins, so the pin
# is a tolerance.
#
# MEASURED worst componentwise relative over the geometries below: 1.4e-13
# (and 1.1e-13 on the coarse self-convergence machine, 5.3e-14 at NC = 1).
# Pinned at 1e-9 -- four decades over the realized number, which is the right
# kind of margin for a cross-build gate that must never be re-tolerance-d in
# a hurry.

G_568_3_TOL = 1e-9

PARITY_GEOMS = (
    (0.3, 0.2),
    (1.0, 0.05),
    (2.0, 0.5),
    (0.02, 0.01),
)


@pytest.mark.parametrize("rho_wl,h_wl", PARITY_GEOMS)
def test_g5683_the_engines_agree_on_the_synthetic_integrand(
    rho_wl, h_wl, record_property
):
    _et, k_p, k_m, lam_m = medium("A", 7.0e6)
    rho, h = rho_wl * lam_m, h_wl * lam_m

    def f(lam):
        return np_synth6(lam, rho, h, k_p, k_m)

    ref, _hp, _tp, _conv, _acc = below._run_contour(
        f, k_p, k_m, rho, h, 1e-11, below._ADAPT_DEPTH, below._DETOUR, GX, GW
    )
    got, _chp, _ctp, _cconv, _cacc = cpp_synth6(rho, h, k_p, k_m)
    rel = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)))
    record_property("worst_rel", float(rel))
    assert rel < G_568_3_TOL, f"({rho_wl}, {h_wl}): {rel:.3e}"


def test_g5683_parity_holds_on_the_coarse_self_convergence_machine(
    record_property,
):
    """`_six_integrals_below(selfconv=True)` re-runs the whole contour on
    Gauss-16, rtol x100 and a shallower detour. U2 will need that twin too, so
    the engine has to take the coarse rule as an argument and land on the same
    answer numpy does."""
    _et, k_p, k_m, lam_m = medium("B", 21.0e6)
    rho, h = 1.0 * lam_m, 0.05 * lam_m

    def f(lam):
        return np_synth6(lam, rho, h, k_p, k_m)

    ref, _, _, _, _ = below._run_contour(
        f,
        k_p,
        k_m,
        rho,
        h,
        1e-9,
        below._ADAPT_DEPTH_COARSE,
        below._DETOUR_COARSE,
        GXC,
        GWC,
    )
    got, _, _, _, _ = accel.contour_engine_synth6(
        rho,
        h,
        k_p,
        k_m,
        1e-9,
        below._ADAPT_DEPTH_COARSE,
        below._DETOUR_COARSE,
        GXC,
        GWC,
        4000,
    )
    rel = float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)))
    record_property("worst_rel", float(rel))
    assert rel < G_568_3_TOL, f"coarse machine: {rel:.3e}"


def test_g5683_the_identity_path_agrees_with_numpy_too(record_property):
    """The NC = 1 instantiation, against the numpy engine rather than the
    closed form -- so a template bug that only bites at NC = 1 cannot hide
    behind the identity's own tolerance."""
    _et, k_p, k_m, lam_m = medium("A", 7.0e6)
    worst = 0.0
    for rho_wl, h_wl in ((0.3, 0.2), (1.0, 0.05)):
        rho, h = rho_wl * lam_m, h_wl * lam_m

        def f(lam):
            return np_identity(lam, rho, h, k_m)

        ref, _, _, _, _ = below._run_contour(
            f,
            k_p,
            k_m,
            rho,
            h,
            1e-11,
            below._ADAPT_DEPTH,
            below._DETOUR,
            GX,
            GW,
        )
        got, _, _, _, _ = cpp_identity(rho, h, k_p, k_m, True)
        worst = max(worst, abs(got - ref[0]) / abs(ref[0]))
    record_property("worst_rel", float(worst))
    assert worst < G_568_3_TOL, f"NC=1 parity: {worst:.3e}"


# ======================================================================
# G-568-4 — health-counter parity
# ======================================================================
#
# `Health` is what lets a caller tell a converged answer from a lucky one, so
# the port has to reproduce it. Panel counts need not be identical (they are
# the outcome of floating-point comparisons that differ in the last bits), but
# a 10x discrepancy would mean the adaptive logic diverged.
#
# MEASURED over PARITY_GEOMS: every head and tail count IDENTICAL, and both
# flags identical. The band below is deliberately looser than the measurement
# so a last-bit tie-break on some other box cannot turn this red.

G_568_4_BAND = 0.20


@pytest.mark.parametrize("rho_wl,h_wl", PARITY_GEOMS)
def test_g5684_the_health_counters_track_numpy(rho_wl, h_wl, record_property):
    _et, k_p, k_m, lam_m = medium("A", 7.0e6)
    rho, h = rho_wl * lam_m, h_wl * lam_m

    def f(lam):
        return np_synth6(lam, rho, h, k_p, k_m)

    _v, hp, tp, conv, acc = below._run_contour(
        f, k_p, k_m, rho, h, 1e-11, below._ADAPT_DEPTH, below._DETOUR, GX, GW
    )
    _cv, chp, ctp, cconv, cacc = cpp_synth6(rho, h, k_p, k_m)
    record_property("numpy_panels", f"{hp}/{tp}")
    record_property("cpp_panels", f"{chp}/{ctp}")
    assert bool(cconv) == bool(conv)
    assert bool(cacc) == bool(acc)
    assert abs(chp - hp) <= max(1, G_568_4_BAND * hp), f"head {chp} vs {hp}"
    assert abs(ctp - tp) <= max(1, G_568_4_BAND * tp), f"tail {ctp} vs {tp}"


# ======================================================================
# G-568-5 — one test per measured guard
# ======================================================================
#
# Each of the three tail guards in `_tail_below` was a real defect in the
# #524 phase-0 prototype. The C++ port transcribes all three; these are the
# tests that would go red if one were dropped.
#
# ---------------------------------------------------------------------
# MUTATION PAIR 1 — the `+ 1e-9` inside the panel-index floor.
#
# `next_zero` computes `floor((z - off)/lat + 1e-9) + 1`. Drop the 1e-9 and a
# z sitting exactly on the Bessel-zero lattice floors back onto ITSELF: znext
# == z, the panel has zero length, its contribution is exactly 0.0, and two
# consecutive zeros trip the quiet counter. The tail then reports CONVERGED
# having integrated nothing past its start. Phase 0 measured the symptom as a
# tail truncating at lam ~ 1.4 when the answer needed lam ~ 40.
#
# CAUGHT BY: `test_g5685_the_zero_length_panel_nudge_holds`, which starts the
# tail exactly on the lattice. Without the nudge it would return converged
# with tail_panels == 2 and a value wrong in the first digit; the test asserts
# BOTH a healthy panel count and agreement with the closed form, so neither
# half can be satisfied by accident.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 7, 12])
def test_g5685_the_zero_length_panel_nudge_holds(n, record_property):
    """Put the tail's first abscissa exactly on the J0-zero lattice.

    The lattice is `off + n*lat` with `lat = pi/rho`, `off = -pi/(4 rho)`, and
    the tail starts at `a = 1.1 max(|k_p|, |k_m|)`. Choosing k_p so that `a`
    lands on a lattice point is what makes `(z - off)/lat` an exact integer --
    the input the missing nudge cannot survive.
    """
    rho = 1.0
    lat = np.pi / rho
    off = -0.25 * np.pi / rho
    a = off + n * lat
    k_p = a / 1.1
    k_m = complex(0.4 * k_p, -0.2 * k_p)
    assert (a - off) / lat == float(n), "the abscissa must sit ON the lattice"

    h = 0.35
    val, _hp, tp, conv, acc = cpp_identity(rho, h, k_p, k_m)
    ref = closed_form(complex(k_p, 0.0), rho, h)
    rel = abs(val - ref) / abs(ref)
    record_property("tail_panels", int(tp))
    record_property("rel", float(rel))
    assert conv and not acc
    # Zero-length panels would trip the quiet counter after exactly two.
    assert tp > 4, f"tail stalled after {tp} panels -- the nudge is gone"
    assert rel < G_568_2_TOL, f"n={n}: {rel:.3e}"


# ---------------------------------------------------------------------
# MUTATION PAIR 2 — the 1.5/h e-folding cap.
#
# Panels are `min(next_zero(z), z + 1.5/h)`. Drop the cap and a single
# 24-node Gauss rule has to span one whole Bessel half-period, pi/rho. At the
# geometry below that is 62.8 in lam against a decay length of 1/h = 0.5 --
# 126 e-foldings of e^{-lam h} under one rule. Phase 0 measured 8e-4 of error
# from FOURTEEN e-foldings.
#
# CAUGHT BY: `test_g5685_the_efolding_cap_sets_the_panel_length`. The
# assertion that discriminates is the closed-form comparison at 100x rtol
# (1e-9): an uncapped rule lands four to five decades above it. The panel-count
# assertion pins the mechanism -- with the cap the tail needs many short
# panels, without it a handful of long ones.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("rho,h", [(0.05, 2.0), (0.02, 3.0), (0.1, 1.0)])
def test_g5685_the_efolding_cap_sets_the_panel_length(rho, h, record_property):
    """A geometry where 1.5/h, not the zero lattice, decides panel length."""
    k_p = 0.15
    k_m = complex(0.5, -0.2)
    lat = np.pi / rho
    cap = 1.5 / h
    assert cap < lat, "this geometry must be cap-governed, not lattice-governed"

    val, _hp, tp, conv, acc = cpp_identity(rho, h, k_p, k_m)
    ref = closed_form(complex(k_p, 0.0), rho, h)
    rel = abs(val - ref) / abs(ref)
    record_property("tail_panels", int(tp))
    record_property("rel", float(rel))
    assert conv and not acc
    # Uncapped, one lattice panel would already cover the whole decay.
    assert tp > 5, f"only {tp} panels -- the cap is not governing"
    assert rel < G_568_2_TOL, f"rho={rho} h={h}: {rel:.3e}"


# ---------------------------------------------------------------------
# MUTATION PAIR 3 — the `ref` floor under the quiet test.
#
# The quiet test is `cmax == 0.0 or cmax < rtol * max(|acc|, ref, 1e-300)`.
# Drop `ref` (the head's scale) and a tail whose own magnitude sits decades
# below the head can never go quiet relatively: each panel is small, but it is
# not small COMPARED TO THE RUNNING TAIL SUM, which is equally small. The tail
# then runs until the integrand underflows to exactly 0.0 and the
# `cmax == 0.0` arm catches it -- hundreds of wasted panels, and on a kernel
# that does not underflow, the full budget and a spurious `accel`.
#
# CAUGHT BY: `test_g5685_the_ref_floor_terminates_on_the_absolute_bar`. At
# rho = 0.01, h = 50 the tail is ~13 decades under the head and the floored
# test goes quiet in two panels. Without the floor the same geometry needs
# ~477 panels (measured by hand from the e^{-lam h} underflow point), so the
# `tp <= 6` assertion is the discriminator.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("rho,h", [(0.01, 50.0), (0.001, 80.0)])
def test_g5685_the_ref_floor_terminates_on_the_absolute_bar(rho, h, record_property):
    """The tail sits decades below the head; it must stop on the absolute bar
    rather than chase its own ulps to the budget."""
    k_p = 0.15
    k_m = complex(0.5, -0.2)
    val, _hp, tp, conv, acc = cpp_identity(rho, h, k_p, k_m)
    ref = closed_form(complex(k_p, 0.0), rho, h)
    rel = abs(val - ref) / abs(ref)
    record_property("tail_panels", int(tp))
    record_property("rel", float(rel))
    assert conv, "the ref floor is what lets this terminate at all"
    assert not acc, "the budget must not have been exhausted"
    assert tp <= 6, f"{tp} panels -- the ref floor is not being applied"
    assert rel < G_568_2_TOL, f"rho={rho} h={h}: {rel:.3e}"


# ---------------------------------------------------------------------
# The fourth guard: the Wynn fallback when the panel budget runs out.
#
# `_tail_below` accelerates the last <= 41 partial sums when it has NOT gone
# quiet inside its budget, and flags `accel`. Phase 0 never needed it on a
# SPEC geometry; it is kept for the h -> 0 limit.
#
# TWO things are gated here, and the second one is a FINDING rather than a
# requirement: (a) the flag and the panel count behave, and the accelerated
# answer lands near the closed form at a geometry where the residual tail is
# still exponentially small (measured 1.7e-8); (b) at h = 0 EXACTLY -- where
# the tail is only conditionally convergent -- the accelerated answer is 2.8e-1
# off the closed form at a 200-panel budget and 5.7e-2 at the full 4000, and
# the C++ engine reproduces the numpy engine's wrong answer to 1.5e-14. That is
# the port being FAITHFUL, not the port being accurate: the numpy engine has
# the same cliff, and on this integrand Wynn's epsilon buys essentially nothing
# over the raw truncation. `accel = True` should be read as "this is a
# truncation", not as "this was accelerated". Recorded for momwire#568.
# ---------------------------------------------------------------------

G_568_5_WYNN_TOL = 1e-6


def test_g5685_the_wynn_fallback_fires_when_the_budget_runs_out(
    record_property,
):
    """rho = 1, h = 0.02, 300 panels: the tail needs more than the budget, so
    the engine extrapolates instead of truncating and says so."""
    rho, h, mp = 1.0, 0.02, 300
    k_p = 0.15
    k_m = complex(0.5, -0.2)
    val, _hp, tp, conv, acc = cpp_identity(rho, h, k_p, k_m, max_panels=mp)
    ref = closed_form(complex(k_p, 0.0), rho, h)
    rel = abs(val - ref) / abs(ref)
    record_property("rel", float(rel))
    assert not conv, "this geometry must exhaust its budget"
    assert acc, "an exhausted budget must go through Wynn"
    assert tp == mp
    assert rel < G_568_5_WYNN_TOL, f"Wynn landed at {rel:.3e}"


def test_g5685_wynn_at_h_zero_reproduces_the_numpy_engine(record_property):
    """h = 0 exactly. The engine cannot converge and the extrapolated answer
    is far from the closed form -- so what is gated is PARITY: the C++ Wynn
    column and numpy's must agree, on a sequence long enough (41 partial sums)
    that agreement is not a coincidence."""
    rho, h, mp = 1.0, 0.0, 200
    k_p = 0.15
    k_m = complex(0.5, -0.2)

    def f(lam):
        return np_identity(lam, rho, h, complex(k_p, 0.0))

    ref, _hp, tp, conv, acc = below._run_contour(
        f,
        k_p,
        k_m,
        rho,
        h,
        1e-11,
        below._ADAPT_DEPTH,
        below._DETOUR,
        GX,
        GW,
        max_panels=mp,
    )
    got, _chp, ctp, cconv, cacc = cpp_identity(rho, h, k_p, k_m, max_panels=mp)
    assert (conv, acc, tp) == (False, True, mp)
    assert (bool(cconv), bool(cacc), ctp) == (False, True, mp)
    rel = abs(got - ref[0]) / abs(ref[0])
    record_property("wynn_parity_rel", float(rel))
    # The distance from the closed form is a property of the ALGORITHM, not of
    # this port; recorded, not asserted tight.
    cf0 = closed_form(complex(k_p, 0.0), rho, h)
    record_property("wynn_vs_closed_form", float(abs(got - cf0) / abs(cf0)))
    assert rel < G_568_3_TOL, f"Wynn parity: {rel:.3e}"


# ======================================================================
# G-568-6 — the engine is reentrant, and it is faster
# ======================================================================


def test_g5686_the_engine_is_stateless_across_calls():
    """No mutable static, no global: the same arguments must give the same
    answer whatever ran in between. U2/U3 call this from OpenMP threads with
    the GIL released, where a hidden static would be a data race rather than a
    wrong number."""
    _et, k_p, k_m, lam_m = medium("A", 7.0e6)
    rho, h = 0.3 * lam_m, 0.2 * lam_m
    first = cpp_synth6(rho, h, k_p, k_m)
    # Interleave calls with very different contours (different regimes of the
    # Bessel switch, different panel logic, a Wynn-accelerated one).
    cpp_identity(0.01, 50.0, 0.15, complex(0.5, -0.2))
    cpp_identity(1.0, 0.0, 0.15, complex(0.5, -0.2), max_panels=60)
    cpp_synth6(2.0 * lam_m, 0.5 * lam_m, k_p, k_m)
    second = cpp_synth6(rho, h, k_p, k_m)
    assert np.array_equal(first[0], second[0])
    assert first[1:] == second[1:]


def test_g5686_the_cpp_engine_beats_numpy(record_property):
    """Informational, but pinned loosely so a catastrophic regression (an
    accidental O(n^2), a dropped fast path) cannot pass unnoticed.

    MEASURED 17-20x single-threaded over the four PARITY_GEOMS on an i7-8550U
    (19.3x aggregate, 77.4 ms of numpy against 4.0 ms of C++). The bar here is
    2x: the ratio is machine- and load-dependent, this box throttles, and a
    tight pin on a wall-clock number is how a suite starts flaking. Anything
    that lands under 2x is not "a slow box", it is a broken build.

    Where the win comes from, because it is not what one first guesses:
    numpy's cost is dominated by ufunc CALL OVERHEAD on 24-element arrays --
    ~20 ufunc dispatches per Gauss rule, at microseconds each -- while the C++
    cost is per-point transcendentals in the integrand. The Bessel pair is
    ~55-60% of the C++ side, which is why `_contour_engine_inline.h` carries a
    real-argument specialization (the tail's abscissa is real by
    construction). U2/U3 should read a similar or better ratio: the production
    integrands make MORE numpy calls per Gauss rule than this synthetic one,
    and only the C++ side can then run the (rho, h) grid under OpenMP.
    """
    _et, k_p, k_m, lam_m = medium("A", 7.0e6)
    rho, h = 1.0 * lam_m, 0.05 * lam_m

    def f(lam):
        return np_synth6(lam, rho, h, k_p, k_m)

    def timed(fn, reps):
        fn()
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    t_np = timed(
        lambda: below._run_contour(
            f,
            k_p,
            k_m,
            rho,
            h,
            1e-11,
            below._ADAPT_DEPTH,
            below._DETOUR,
            GX,
            GW,
        ),
        3,
    )
    t_c = timed(lambda: cpp_synth6(rho, h, k_p, k_m), 10)
    record_property("numpy_s", float(t_np))
    record_property("cpp_s", float(t_c))
    record_property("ratio", float(t_np / t_c))
    assert t_c < t_np / 2.0, f"only {t_np / t_c:.1f}x faster than numpy"
