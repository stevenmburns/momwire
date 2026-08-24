"""The transmitted fills on the C++ contour engine (momwire#568 unit 3).

U1 landed the engine and gated it against scipy, against the Sommerfeld
identity's closed form, and against the numpy engine on a SYNTHETIC integrand.
U2 put the below/below family on it. U3 puts the transmitted one on it:
`_integrand_six_transmitted` (five surfaces, two gammas, a `swap` twin), the
`_six_integrals_transmitted` driver with its coarse self-convergence twin, the
per-point loop of `t_surfaces_direct` (this is where the OpenMP region lives,
and it is the arc's biggest single cluster), and the projected pair table
behind both directions of travel.

The numpy spellings stay, unchanged, as the references. Every gate here runs
BOTH machines in one process — `trans._FORCE_NUMPY` flips the dispatch — and
compares them componentwise and relatively.

What this file gates, and why each gate exists:

* **The INTEGRAND pointwise**, not merely the integral. Two of this family's
  available mutations are invisible to an integrated gate: index 1 spelled
  `γ² + k²` instead of `λ²` moves the integral by ~3.5e-11 (under any pin an
  integrated gate can honestly carry) while moving the integrand by ~1e-5 at
  the bottom of the contour; and index 4 collapsed onto index 0 agrees
  wherever γ_m happens to track γ_p. `_accelerators.transmitted_integrand_six`
  exists for these two gates and for nothing else.
* **Value parity** on the six integrals, the five surfaces and the projected
  table, over the SPEC cells and off-SPEC stressors. This is the whole
  correctness claim: the shipped ladders in `test_sommerfeld_transmitted.py`
  are measured against oracles and prototypes at 1e-7 to 5e-3, three to eight
  decades looser than the C++/numpy spread, so they would not notice a defect
  this file is sized to catch.
* **The grazing band.** The transmitted tail costs ~16·cot θ panels and the
  budget is a CLIFF, not a soft landing — a truncated tail was measured 4.5e+3
  relative wrong. So there is a gate at the grid's own grazing floor (~4000
  panels, health clean) and one just under it, where BOTH machines must agree
  that they failed. `Health.nonconvergent` is contract here, not decoration.
* **The paths that are NOT quadrature.** The side-of-interface refusals must
  raise with the same words in the same node order — including the asymmetry
  that an observer may sit ON the interface and a source may not — and the
  grid's four domain refusals must survive the dispatch unchanged.
* **The reciprocity transpose**, read through both entry points, on both
  machines. One C++ kernel serves both directions with `transposed` swapping
  T_ρ^V and T_z^H, so this gate compares a transpose against itself rather
  than two ports against each other.
* **The families this unit did not build keep their bytes.** U3 shares
  `mw568_below::gamma_cut` with U2 and shares nothing at all with the ±=+
  family; both are re-read from inside this build.

TOLERANCES, NEVER BYTE EQUALITY. The two machines take the same adaptive
decisions — measured: identical panel counts, node for node — but are not
bit-twins: the Gauss dot product reduces in a different order, complex
division and `exp` are libgcc's rather than numpy's, and the Bessel pair is a
different algorithm entirely (U1's G-568-1).
"""

import math

import numpy as np
import pytest

from momwire import _sommerfeld as som
from momwire import _sommerfeld_below as below
from momwire import _sommerfeld_transmitted as trans

import golden_transmitted_524 as gold

accel = pytest.importorskip("momwire._accelerators")

if not getattr(accel, "transmitted_fills_568", False):
    pytest.skip(
        "the accelerator predates momwire#568 unit 3 (stale .so?)",
        allow_module_level=True,
    )

EPS0 = 8.8541878128e-12
C0 = 299792458.0
MU0 = som._MU0
KEYS = trans._SURF_KEYS_T
GROUND_Z = 0.0


def eps_tilde(eps_r, sigma, f_hz):
    """ε̃ = ε_r − jσ/(ωε₀), as `test_sommerfeld_transmitted` writes it."""
    return eps_r - 1j * sigma / (2.0 * np.pi * f_hz * EPS0)


def medium(cell):
    """(ε̃, k_p, ω, k_m, λ_p) for a golden cell key like "A/7MHz"."""
    sname, fname = cell.split("/")
    er, sg = gold.SOILS[sname]
    f = float(fname[:-3]) * 1e6
    om = 2.0 * np.pi * f
    et = eps_tilde(er, sg, f)
    kp = om / C0
    return et, kp, om, below.k_medium(et, kp), 2.0 * np.pi / kp


SPEC_CELLS = sorted(gold.SURFACES)


class force_numpy:
    """Run a block on the numpy reference, restoring the dispatch after.

    A context manager rather than a fixture because every gate here needs BOTH
    machines inside ONE test — a fixture that pinned the module flag for a
    whole test could only ever measure one of them.
    """

    def __enter__(self):
        self._was = trans._FORCE_NUMPY
        trans._FORCE_NUMPY = True
        assert not trans._use_transmitted_accel()

    def __exit__(self, *exc):
        trans._FORCE_NUMPY = self._was
        return False


@pytest.fixture(autouse=True)
def _accel_on():
    """Every test starts on the accelerated path, whatever the last one did."""
    was = trans._FORCE_NUMPY
    trans._FORCE_NUMPY = False
    assert trans._use_transmitted_accel(), "the dispatch must be live for these gates"
    yield
    trans._FORCE_NUMPY = was


def cwise(got, ref):
    """Worst COMPONENTWISE relative difference. Not of-family: the five
    transmitted surfaces span decades at a single point — the fifth one has a
    deep null in the near zone — and grading a small component against its
    largest neighbour would hide a factor of ten in it. That is U2's second
    inversion and the easiest one to reintroduce by accident."""
    got = np.asarray(got)
    ref = np.asarray(ref)
    return float(np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-300)))


def obs_point(r_wl, th_deg, lam_p):
    """(ρ, z) for an observer at R = r_wl·λ_p, elevation th_deg, over the
    source's ground projection — the grid's own coordinates."""
    r = r_wl * lam_p
    th = math.radians(th_deg)
    return max(r * math.cos(th), 0.0), max(r * math.sin(th), 0.0)


# ======================================================================
# G-568-11 — the integrand and the six integrals, C++ against numpy
# ======================================================================
#
# MEASURED worst componentwise relative on the six INTEGRALS, per SPEC cell,
# over the geometry list below at rtol 1e-9:
#
#   A/7  5.87e-12   A/21 2.07e-12   B/7  1.05e-11
#   B/21 2.66e-12   C/7  4.34e-12   C/21 1.75e-12
#
# and over the off-SPEC stressors: sea water 8.32e-11, near-lossless rock
# 1.35e-12, ε̃ just off 1 1.31e-11, 160 m 9.42e-11, ε̃ = 1 exactly 7.55e-11.
# Grand worst 9.42e-11. The swapped twin reads 6.34e-12.
#
# A DECADE OR TWO ABOVE U2's below/below spread (2.79e-12) and the reason is
# structural rather than a defect: the worst cell is always index 0 or index 4,
# the two components carrying a bare γ (∂/∂z → ×(−γ_p), ∂/∂z′ → ×(+γ_m)). γ
# grows like λ, so those two weight the integrand toward the far tail, where
# thousands of independently-rounded Gauss panels accumulate — and the worst
# points are the ones where the observer sits ON the interface (z = 0), which
# removes the observer leg's decay entirely and lengthens the tail further.
#
# Pinned at 1e-8 — ~106x over the realized worst. Deliberately NOT set at the
# measurement: these last digits move by a factor of a few between builds (GCC
# decides where the FMAs land), and a cross-build parity pin must never be the
# thing someone re-tolerances in a hurry. The gates it protects sit far above
# it anyway — G-U3-1 at 1e-7 against the phase-0 prototype, the grid ladders
# at 1e-3/5e-3.
#
# The INTEGRAND is pinned ten times tighter, at 1e-11 against a measured
# 5.81e-14 (per cell: 1.72e-14 … 5.82e-14, both branches), because it has no
# adaptive machinery in it at all: it is a fixed
# sequence of exp, sqrt, divide and Bessel calls, and the only spread available
# is the last bits of those. That tightness is what makes it a mutation gate.

G_568_11_TOL = 1e-8
G_568_11_INTEGRAND_TOL = 1e-11

# (R/λ_p, θ deg, |z′| m). The last two are the product geometry and its
# limits: deep grazing at working range, where the tail runs longest, and the
# observer ON the interface, which this family tabulates (θ = 0 is a legitimate
# limit here — the source leg alone holds the tail down).
PARITY_GEOMS = (
    (0.001, 45.0, 0.02),
    (0.01, 45.0, 0.15),
    (0.1, 80.0, 0.15),
    (0.5, 5.0, 0.15),
    (1.0, 0.5, 0.15),
    (2.0, 30.0, 1.0),
    (2.0, 0.2, 0.15),
    (0.05, 0.0, 0.15),
)

# λ values for the pointwise integrand gates. Six decades down to 1e-6 because
# that is where the (∂²/∂z² + k²) cancellation lives (see the index-1 mutation
# note); the complex ones are the head detour's abscissa, the real ones the
# tail's.
INTEGRAND_LAM = np.concatenate(
    [
        np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
        np.logspace(-1.0, 3.0, 30),
        np.array([1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0]) * (1.0 + 0.3j),
    ]
).astype(np.complex128)


def integrand_pair(cell, rho, z, zp, swap=False, lam=None):
    """(C++, numpy) integrand tables (n, 6) at the same λ."""
    et, kp, om, km, lam_p = medium(cell)
    lam = INTEGRAND_LAM if lam is None else lam
    got = accel.transmitted_integrand_six(
        lam, float(rho), float(z), float(zp), float(kp), complex(km), swap
    )
    ref = trans._integrand_six_transmitted(lam, rho, z, zp, kp, km, swap=swap).T
    return got, ref


@pytest.mark.parametrize("cell", SPEC_CELLS)
def test_g56811_the_integrand_agrees_pointwise(cell, record_property):
    """The transcription itself, read at λ rather than through an integral.

    Both branches: `swap=False` is the product path (source below carrying
    γ_m, observer above carrying γ_p) and `swap=True` its above→below twin.
    Six decades of λ, real and complex, so the head detour's abscissa and the
    tail's are both covered.
    """
    worst = 0.0
    for swap, (rho, z, zp) in ((False, (2.0, 0.4, -0.15)), (True, (2.0, -0.4, 0.15))):
        got, ref = integrand_pair(cell, rho, z, zp, swap=swap)
        worst = max(worst, cwise(got, ref))
    record_property("worst_rel", float(worst))
    assert worst < G_568_11_INTEGRAND_TOL, f"{cell}: {worst:.3e}"


@pytest.mark.parametrize("cell", SPEC_CELLS)
def test_g56811_the_six_integrals_agree_over_the_spec_matrix(cell, record_property):
    et, kp, om, km, lam_p = medium(cell)
    worst = 0.0
    for r_wl, th_deg, zp in PARITY_GEOMS:
        rho, z = obs_point(r_wl, th_deg, lam_p)
        got = trans._six_integrals_transmitted(et, kp, rho, z, -zp, 1e-9)
        with force_numpy():
            ref = trans._six_integrals_transmitted(et, kp, rho, z, -zp, 1e-9)
        worst = max(worst, cwise(got, ref))
    record_property("worst_rel", float(worst))
    assert worst < G_568_11_TOL, f"{cell}: {worst:.3e}"


@pytest.mark.parametrize(
    "eps_r,sigma,f_hz",
    [
        (81.0, 5.0, 7.0e6),  # sea water: |k_m| ~ 95 k_p, the tail's stressor
        (3.0, 1e-5, 21.0e6),  # near-lossless dry rock
        (1.05, 1e-6, 14.0e6),  # ε̃ just off 1: the Fresnel pair nearly degenerates
        (13.0, 0.005, 1.8e6),  # 160 m, well off the SPEC frequencies
    ],
)
def test_g56811_the_six_integrals_agree_off_spec(eps_r, sigma, f_hz, record_property):
    """Off the SPEC matrix on purpose. Sea water pushes |k_m| to ~95 k_p, which
    is where the head's `a` and the tail's e-folding cap part company; ε̃ ≈ 1 is
    where k_m²γ_p + k_p²γ_m and γ_m + γ_p nearly degenerate onto one another
    and a port that regrouped either would show up first."""
    et = eps_tilde(eps_r, sigma, f_hz)
    kp = 2.0 * np.pi * f_hz / C0
    lam_p = 2.0 * np.pi / kp
    worst = 0.0
    for r_wl, th_deg, zp in PARITY_GEOMS:
        rho, z = obs_point(r_wl, th_deg, lam_p)
        got = trans._six_integrals_transmitted(et, kp, rho, z, -zp, 1e-9)
        with force_numpy():
            ref = trans._six_integrals_transmitted(et, kp, rho, z, -zp, 1e-9)
        worst = max(worst, cwise(got, ref))
    record_property("worst_rel", float(worst))
    assert worst < G_568_11_TOL, f"({eps_r}, {sigma}, {f_hz:g}): {worst:.3e}"


def test_g56811_epsilon_one_has_no_short_circuit_here(record_property):
    """ε̃ = 1 is NOT the below family's zero.

    `_six_integrals_below` short-circuits at ε̃ = 1 because D₁ = D₂ = 0
    identically there — a remainder over a vanished contrast is exactly zero.
    The transmitted integral is the WHOLE field, so at ε̃ = 1 it is the
    free-space dipole and nothing about it vanishes. The C++ path must not
    acquire an inherited short circuit, and `zero_kernel` must stay at 0 on
    both machines.
    """
    kp = 2.0 * np.pi * 7.0e6 / C0
    lam_p = 2.0 * np.pi / kp
    worst = 0.0
    hc = below.Health()
    hn = below.Health()
    for r_wl, th_deg, zp in PARITY_GEOMS:
        rho, z = obs_point(r_wl, th_deg, lam_p)
        got = trans._six_integrals_transmitted(1.0, kp, rho, z, -zp, 1e-9, health=hc)
        with force_numpy():
            ref = trans._six_integrals_transmitted(
                1.0, kp, rho, z, -zp, 1e-9, health=hn
            )
        assert np.count_nonzero(ref) == 6, "eps~ = 1 is the free-space dipole, not zero"
        worst = max(worst, cwise(got, ref))
    record_property("worst_rel", float(worst))
    assert hc.zero_kernel == hn.zero_kernel == 0
    assert hc.evaluations == hn.evaluations == len(PARITY_GEOMS)
    assert worst < G_568_11_TOL, f"{worst:.3e}"


@pytest.mark.parametrize("cell", SPEC_CELLS)
def test_g56811_the_swapped_twin_agrees(cell, record_property):
    """`swap=True` — source ABOVE carrying γ_p, observer BELOW carrying γ_m.

    It exists to GATE the reciprocity transpose rather than to be called by
    the product path, so it is the branch a port is most likely to leave
    untested. Both legs move under it: the exponential's two γ exchange AND
    the two derivative factors exchange sign-and-medium.
    """
    et, kp, om, km, lam_p = medium(cell)
    worst = 0.0
    for rho_wl, z, zp in ((0.5, -0.15, 0.3), (0.02, -0.02, 1.0), (1.5, -0.5, 0.05)):
        rho = rho_wl * lam_p
        got = trans._six_integrals_transmitted(et, kp, rho, z, zp, 1e-9, swap=True)
        with force_numpy():
            ref = trans._six_integrals_transmitted(et, kp, rho, z, zp, 1e-9, swap=True)
        worst = max(worst, cwise(got, ref))
    record_property("worst_rel", float(worst))
    assert worst < G_568_11_TOL, f"{cell}: {worst:.3e}"


def test_g56811_the_swapped_twin_is_the_same_integral(record_property):
    """The reciprocity identity, read through the C++ integrand alone.

    V_T is symmetric under exchanging which leg carries which γ, so the
    swapped contour at (z, z′) must reproduce the unswapped one at (z′, z) —
    with the TWO ρ-z derivatives crossed, because ∂/∂z of one is ∂/∂z′ of the
    other. That crossing is the fifth surface's whole reason for existing, and
    it is what `_combine_transmitted_proj`'s `transposed` flag implements.

    MEASURED through the C++ path: the two scalars (index 1 = V_T's
    (∂²/∂z² + k²) reading, index 5 = U_T) agree to 2.8e-15 and 1.8e-15, and
    the crossed pair to 2.4e-14 / 9.6e-16.
    """
    et, kp, om, km, lam_p = medium("A/7MHz")
    rho, z, zp = 0.5 * lam_p, 0.4, -0.15
    up = trans._six_integrals_transmitted(et, kp, rho, z, zp, 1e-11)
    down = trans._six_integrals_transmitted(et, kp, rho, zp, z, 1e-11, swap=True)
    scal = max(
        abs(up[1] - down[1]) / abs(up[1]),
        abs(up[5] - down[5]) / abs(up[5]),
    )
    crossed = max(
        abs(up[0] - down[4]) / abs(up[0]),
        abs(up[4] - down[0]) / abs(up[4]),
    )
    record_property("scalars_rel", float(scal))
    record_property("crossed_rel", float(crossed))
    assert scal < 1e-11, f"the two scalars are not the same integral: {scal:.3e}"
    assert crossed < 1e-11, f"the rho-z derivatives did not cross: {crossed:.3e}"
    # ... and index 0 and index 4 are genuinely different numbers, so
    # "everything agrees with everything" cannot be how this passes.
    assert abs(up[0] - up[4]) > 0.1 * abs(up[0])


def test_g56811_the_coarse_self_convergence_machine_is_the_same_call(record_property):
    """`selfconv=True` re-runs the whole contour on Gauss-16, rtol ×100 and a
    shallower detour, at the SAME 6000-panel budget. Both rules, both depths
    and both detours are ARGUMENTS to the C++ entry point for exactly this
    reason: the fine machine and the coarse one are one call, not two ports.

    MEASURED: values 1.05e-11, the two machines' self-convergence ESTIMATES
    agreeing to 1.6e-3 by the metric below. The estimate is a fine-vs-coarse
    DIFFERENCE, so it inherits the coarse machine's own noise divided by the
    ~1e-9 spread it is estimating; a few parts in a thousand is what two
    independent evaluations of that difference should agree to, and pinning at
    the measurement would be pinning that noise. 1e-1 is the bar, and it still
    catches the failure that matters — a coarse machine wired to the wrong
    rule, depth, detour or panel budget moves the estimate by decades.

    THE 1e-12 FLOOR IS NOT A FUDGE and it is the one thing here worth
    reading twice. At (R, θ, |z′|) = (2 λ_p, 30°, 1 m) the fine and coarse
    machines agree to MACHINE PRECISION — the two estimates are 2.4e-15 and
    7.7e-16 — so a purely relative comparison of them would be grading
    rounding noise against rounding noise and reads 2.1e+0. The quantity
    being compared is an upper bound that the shipped gate asserts below
    1e-6; two evaluations of it that are both five decades under that bar are
    not disagreeing about anything. So they must agree RELATIVELY or both sit
    under 1e-12, and the floor is named rather than hidden inside a
    `max(..., tiny)`.
    """
    et, kp, om, km, lam_p = medium("B/7MHz")
    # Five decades under the 1e-6 bound `test_gu3_10_the_contour_reports_its_
    # own_health` asserts the estimate below.
    selfconv_floor = 1e-12
    worst_val = worst_spread = 0.0
    for r_wl, th_deg, zp in PARITY_GEOMS:
        rho, z = obs_point(r_wl, th_deg, lam_p)
        hc = below.Health()
        got = trans._six_integrals_transmitted(
            et, kp, rho, z, -zp, 1e-9, health=hc, selfconv=True
        )
        with force_numpy():
            hn = below.Health()
            ref = trans._six_integrals_transmitted(
                et, kp, rho, z, -zp, 1e-9, health=hn, selfconv=True
            )
        worst_val = max(worst_val, cwise(got, ref))
        den = max(hn.worst_selfconv, hc.worst_selfconv, selfconv_floor)
        worst_spread = max(
            worst_spread, abs(hc.worst_selfconv - hn.worst_selfconv) / den
        )
        assert hc.worst_selfconv_at == hn.worst_selfconv_at
    record_property("worst_rel", float(worst_val))
    record_property("worst_selfconv_spread_rel", float(worst_spread))
    assert worst_val < G_568_11_TOL, f"value: {worst_val:.3e}"
    assert worst_spread < 1e-1, f"self-convergence estimate: {worst_spread:.3e}"


# ---------------------------------------------------------------------
# MUTATION PAIR — the two lines of `mw568_trans::SixTransmitted` that a
# port can get wrong without anything failing loudly.
#
# (a) INDEX 1 IS `lam*lam`, NEVER `g_p*g_p + kp2`. (∂²/∂z² + k²) is γ² + k²
#     in either medium, and γ² + k² is a difference of two O(k²) numbers whose
#     result is O(λ²): at the bottom of the contour it cancels away eleven
#     digits. The numpy module records that the swapped and unswapped
#     spellings of the SAME quantity were measured 3.5e-11 apart before its
#     line said λ².
#
#     VERIFIED BY HAND ONCE (2026-08-23, this box). Rebuilding with
#     `const cd zzk = g_p * g_p + kp2;` turns the INTEGRAND gate red on every
#     one of the six SPEC cells, and NOTHING ELSE ANYWHERE:
#
#       test_g56811_the_integrand_agrees_pointwise
#         [A/7MHz]  4.63e-06   [A/21MHz] 2.49e-05
#         [B/7MHz]  4.63e-06   [B/21MHz] 2.49e-05
#         [C/7MHz]  4.63e-06   [C/21MHz] 2.49e-05
#
#     (the worst cell is λ = 1e-6, where ε·k_p²/λ² is exactly that; it is the
#     21 MHz cells that read higher because k_p is three times larger there).
#     The other 57 gates in this file stay green and, measured, essentially
#     UNMOVED: the six integrals read 2.69e-12 where they read 2.66e-12 clean,
#     the five surfaces 3.762e-11 where they read 3.760e-11, the grazing gate
#     2.58e-11 to its last digit. So do all 75 gates of the shipped ladder,
#     default lane and slow: `test_gu3_1_surfaces_vs_the_phase0_prototype`
#     reads 5.2e-10 either way against its 1e-7 bar.
#
#     THAT is why `transmitted_integrand_six` exists. An 11-digit cancellation
#     inside an integrand is invisible once the integral has averaged over it;
#     a mutation that no gate catches is a mutation that ships.
#
# (b) INDEX 4 IS THE FIFTH SURFACE and must not collapse onto index 0.
#     ∂/∂z′ multiplies by +γ_m where ∂/∂z multiplies by −γ_p, and the two are
#     different functions of λ, so T_z^H is not −cosφ·T_ρ^V. That collapse is
#     a ±=+ identity that does not survive here.
#
#     VERIFIED BY HAND ONCE (2026-08-23, this box). Rebuilding with
#     `out[4] = a * dr * d_z` (index 0's factor) turns 36 of this file's 63
#     gates red, and with them the shipped ladder:
#
#       test_g56812_the_fifth_surface_is_not_a_multiple_of_the_first
#                                                  both cells, ratio spread 0.0
#       test_g56811_the_integrand_agrees_pointwise all six cells,
#                                                  4.58e+00 to 2.62e+01
#       test_g56811_the_six_integrals_agree_*      all ten, plus eps~ = 1
#       test_g56811_the_swapped_twin_*             all seven
#       test_g56813 / test_g56814 / test_g56815    the grazing band, the
#                                                  observer-on-interface limit,
#                                                  all six surface cells
#       test_g56816_*                              the projected table, both
#                                                  directions, both grids
#
#     and, in tests/test_sommerfeld_transmitted.py, 7 of 21 default-lane gates
#     and 30 of 54 slow ones:
#
#       test_gu3_1_surfaces_vs_the_phase0_prototype   RED on every SPEC cell at
#                                                     TzH 2.01e+00 to 2.29e+00
#                                                     against the COMMITTED
#                                                     phase-0 goldens
#       test_gu3_5_eps_one_is_the_free_space_dipole_exactly           9.32e-01
#       test_gu3_2_composed_field_vs_empymod          28 of 32 entries
#       test_gu3_4_above_to_below_vs_empymod          RED
#       test_gu3_8_grid_matches_direct_and_the_goldens RED
#
#     The empymod and phase-0-golden lines are the ones that matter: O(1)
#     failures against INDEPENDENT oracles, not merely against the numpy twin.
#     That is what makes the fifth surface load-bearing physics rather than a
#     shared convention.
# ---------------------------------------------------------------------


@pytest.mark.parametrize("cell", ["A/7MHz", "B/21MHz"])
def test_g56812_the_fifth_surface_is_not_a_multiple_of_the_first(cell, record_property):
    """Index 4 read against index 0 through the C++ integrand.

    ∂/∂z′ → ×(+γ_m) and ∂/∂z → ×(−γ_p) share every other factor, so the ratio
    of the two components IS −γ_m/γ_p pointwise. A port that collapsed index 4
    onto index 0 would give ratio 1; a port that used −γ_p/γ_m or +γ_p would
    give the wrong ratio. Gated against the ratio itself, not against a
    magnitude.
    """
    et, kp, om, km, lam_p = medium(cell)
    lam = np.array([1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 50.0], dtype=np.complex128)
    got, _ref = integrand_pair(cell, 2.0, 0.4, -0.15, lam=lam)
    ratio = got[:, 4] / got[:, 0]
    # d_zp / d_z = (+gamma_m) / (-gamma_p): different medium AND opposite sign.
    want = som._gamma(lam, km) / (-som._gamma(lam, kp))
    worst = float(np.max(np.abs(ratio - want) / np.abs(want)))
    record_property("worst_ratio_rel", float(worst))
    assert worst < 1e-11, f"{cell}: index 4 is not carrying gamma_m: {worst:.3e}"
    # ... and the ratio is not 1, so a collapsed index 4 could not pass.
    spread = float(np.max(np.abs(ratio - 1.0)))
    record_property("ratio_spread_from_one", spread)
    assert spread > 0.5, f"index 4 collapsed onto index 0: {spread:.3e}"


# ======================================================================
# G-568-13 — the grazing band, where the tail budget is a cliff
# ======================================================================
#
# The transmitted tail is panelled on the J₀(λρ) zero lattice and must reach
# λ ~ 35/(z + |z′|), so it costs ~16·cot θ_true panels and `_MAX_TAIL_PANELS_T`
# is 6000. `grazing_floor` solves that law for the θ at which the grid's worst
# node lands exactly on the budget.
#
# MEASURED at soil A / 7 MHz, r_max = 2 λ_p = 85.65 m, |z′| = 0.15 m
# (θ_floor = 0.0524°) — panel counts IDENTICAL between the two machines at
# every point:
#
#   θ/θ_floor   cot θ_true   tail panels   converged   worst rel
#      0.5          453         4909         yes       2.27e-11
#      1.0          375         4029         yes       7.01e-12
#      1.5          320         3411         yes       2.58e-11
#      3.0          222         2324         yes       2.47e-11
#
# and at |z′| = 0.02 m (θ_floor = 0.1394°), θ/θ_floor = 0.5 puts cot θ_true at
# 690 and the tail hits the budget: 6000 panels, converged = False on BOTH
# machines, and the Wynn-accelerated values still agree to 1.49e-11 (they are
# the same truncation of the same series, so they agree — which is exactly why
# agreement is NOT evidence of correctness out there, and why the gate asserts
# the FLAG rather than the value).
#
# Pinned at 1e-7 — ~4000x over the realized worst. Looser than G-568-11's 1e-8
# on purpose: 4000 panels of independently-rounded Gauss sums is the widest
# spread the two machines have anywhere, and the point of THIS gate is health
# parity and honest convergence rather than a tighter number. What it catches
# is a tail that stops early, a panel lattice that drifts, or a Wynn
# extrapolation that fires unannounced — all of which move it by decades.

G_568_13_TOL = 1e-7


def test_g56813_the_grazing_band_converges_honestly_and_agrees(record_property):
    """Deep in cot θ, at and around the grid's own grazing floor.

    This is where an integrand transcription error shows up and nowhere else:
    a wrong γ, a regrouped Fresnel denominator or a mis-signed derivative all
    survive the head and the first few tail panels, and only a tail that runs
    for thousands of panels integrates the difference into something visible.
    """
    et, kp, om, km, lam_p = medium("A/7MHz")
    r_max = trans._R_CAP_LAMBDA_P * lam_p
    zp = 0.15
    floor = trans.grazing_floor(r_max, zp)
    assert floor > 0.0, "this cell is supposed to have a positive grazing floor"
    worst = 0.0
    panels = []
    for mult in (0.5, 1.0, 1.5, 3.0):
        th = floor * mult
        rho = r_max * math.cos(th)
        z = max(r_max * math.sin(th), 0.0)
        hc = below.Health()
        got = trans._six_integrals_transmitted(
            et, kp, rho, z, -zp, 1e-11, health=hc, where=("cpp", mult)
        )
        with force_numpy():
            hn = below.Health()
            ref = trans._six_integrals_transmitted(
                et, kp, rho, z, -zp, 1e-11, health=hn, where=("numpy", mult)
            )
        # Honest convergence, not a Wynn extrapolation of a truncated tail.
        assert hc.nonconvergent == 0, (mult, hc.as_dict())
        assert hn.nonconvergent == 0, (mult, hn.as_dict())
        assert hc.accelerated == hn.accelerated == 0
        assert hc.max_tail_panels == hn.max_tail_panels, (
            f"the two machines panelled the tail differently at {mult}x the "
            f"floor: {hc.max_tail_panels} vs {hn.max_tail_panels}"
        )
        panels.append(int(hc.max_tail_panels))
        worst = max(worst, cwise(got, ref))
    record_property("worst_rel", float(worst))
    record_property("tail_panels", repr(panels))
    # The gate is only a grazing gate if the tail really did run long.
    assert min(panels) > 1500, f"the tail did not get deep enough: {panels}"
    assert max(panels) < trans._MAX_TAIL_PANELS_T
    assert worst < G_568_13_TOL, f"{worst:.3e}"


def test_g56813_a_truncated_tail_is_flagged_on_both_paths(record_property):
    """Under the floor, where the budget runs out.

    `_tail_below` falls back to Wynn epsilon on the truncated partial sums and
    on this family that is CATASTROPHICALLY wrong — 4.5e+3 relative, measured.
    The two machines truncate the same series at the same panel and therefore
    agree with each other, which is exactly why agreement out here proves
    nothing and the FLAG is what is gated: if the C++ path ever returned an
    unconverged answer without saying so, every fill gate that asserts
    `nonconvergent == 0` would go quiet at once.
    """
    et, kp, om, km, lam_p = medium("A/7MHz")
    r_max = trans._R_CAP_LAMBDA_P * lam_p
    zp = 0.02
    th = 0.5 * trans.grazing_floor(r_max, zp)
    rho = r_max * math.cos(th)
    z = max(r_max * math.sin(th), 0.0)
    hc = below.Health()
    trans._six_integrals_transmitted(et, kp, rho, z, -zp, 1e-11, health=hc)
    with force_numpy():
        hn = below.Health()
        trans._six_integrals_transmitted(et, kp, rho, z, -zp, 1e-11, health=hn)
    record_property("health_cpp", repr(hc.as_dict()))
    record_property("health_numpy", repr(hn.as_dict()))
    assert hc.nonconvergent == hn.nonconvergent == 1, (hc.as_dict(), hn.as_dict())
    assert hc.accelerated == hn.accelerated == 1
    assert hc.max_tail_panels == hn.max_tail_panels == trans._MAX_TAIL_PANELS_T


# ======================================================================
# G-568-14 — the paths that are not quadrature
# ======================================================================


@pytest.mark.parametrize(
    "rho,z,zp,swap",
    [
        (-1.0, 0.5, -0.15, False),  # rho < 0
        (1.0, -0.5, -0.15, False),  # observer below the interface too
        (1.0, 0.5, 0.15, False),  # source above it
        (1.0, 0.5, 0.0, False),  # SOURCE ON the interface: phase 3
        (1.0, 0.5, -0.15, True),  # the swapped convention, both sides wrong
    ],
)
def test_g56814_the_side_refusals_say_the_same_thing_on_both_paths(rho, z, zp, swap):
    """ρ ≥ 0, the source strictly on its own side, the observer at or beyond
    the other. The C++ entry point does not carry these checks: they are
    exact, they are cheap, and their words are contract."""
    et, kp, om, km, lam_p = medium("A/7MHz")
    msgs = []
    for use_numpy in (True, False):
        ctx = force_numpy() if use_numpy else None
        if ctx:
            ctx.__enter__()
        try:
            with pytest.raises(ValueError) as e:
                trans._six_integrals_transmitted(et, kp, rho, z, zp, 1e-9, swap=swap)
            msgs.append(str(e.value))
        finally:
            if ctx:
                ctx.__exit__()
    assert msgs[0] == msgs[1], msgs


def test_g56814_an_observer_on_the_interface_is_served_not_refused(record_property):
    """z = 0 is the asymmetry, and it is the geometry rather than a
    convenience: the source leg |z′| holds the tail down on its own and the
    transmitted field is continuous across the interface, so the observer ON
    the interface is a tabulated row. A port that hardened the source check
    into a symmetric one would refuse the grid's entire θ = 0 row.
    """
    et, kp, om, km, lam_p = medium("C/7MHz")
    worst = 0.0
    for r_wl in (0.001, 0.05, 1.0):
        rho = r_wl * lam_p
        got = trans._six_integrals_transmitted(et, kp, rho, 0.0, -0.15, 1e-9)
        with force_numpy():
            ref = trans._six_integrals_transmitted(et, kp, rho, 0.0, -0.15, 1e-9)
        assert np.all(np.isfinite(got))
        worst = max(worst, cwise(got, ref))
    record_property("worst_rel", float(worst))
    assert worst < G_568_11_TOL, f"{worst:.3e}"


def test_g56814_the_batch_refuses_in_node_order():
    """One bad node in a batch refuses with that node's numbers, on both
    machines. The C++ batch is handed whole arrays, so the per-node check has
    to run in Python IN NODE ORDER — otherwise the accelerated path would
    report a different node's geometry, or (worse) refuse nothing and hand
    the kernel a source on the interface with no tail decay at all."""
    et, kp, om, km, lam_p = medium("A/7MHz")
    rho = np.array([1.0, 2.0, 3.0, 4.0])
    z = np.array([0.5, 0.5, 0.5, 0.5])
    zp = np.array([-0.15, -0.15, 0.0, -0.15])  # node 2 is on the interface
    msgs = []
    for use_numpy in (True, False):
        ctx = force_numpy() if use_numpy else None
        if ctx:
            ctx.__enter__()
        try:
            with pytest.raises(ValueError, match="STRICTLY on one side") as e:
                trans.t_surfaces_direct(et, kp, rho, z, zp, rtol=1e-9, omega=om)
            msgs.append(str(e.value))
        finally:
            if ctx:
                ctx.__exit__()
    assert msgs[0] == msgs[1], msgs
    assert "0.0" in msgs[0], msgs[0]


# ======================================================================
# G-568-15 — the five surfaces and the health counters
# ======================================================================
#
# MEASURED over the 14-node mixed lattice below (a θ = 0 row mixed in with
# random log-R / θ nodes spanning the whole served rectangle), all six SPEC
# cells: worst per-surface relative 3.76e-11 (C/7 and B/7 at 7 MHz; the 21 MHz
# cells read 4.8e-13 to 1.8e-12), and on the health side every single counter
# IDENTICAL — evaluations 14, nonconvergent 0, accelerated 0, zero_kernel 0,
# max_tail_panels (295/305/918/943/957) and max_head_panels (15/16) equal cell
# for cell, `worst_selfconv_at` the same node, `worst_selfconv` agreeing to
# 1.24e-3 relative or better.
#
# The panel counts get an EQUALITY assertion here rather than U2's ±20 % band,
# because the measurement is exact on this family and the equality is the
# evidence that the cot-θ tail cost law did not move: #568 promises each panel
# gets cheaper, not that there are fewer of them.

G_568_15_TOL = 1e-8


@pytest.mark.parametrize("cell", SPEC_CELLS)
def test_g56815_the_five_surfaces_agree(cell, record_property):
    """`t_surfaces_direct` end to end — the per-point loop that carries the
    OpenMP region, over the whole served rectangle including the θ = 0 row."""
    et, kp, om, km, lam_p = medium(cell)
    rng = np.random.default_rng(568)
    R = np.exp(rng.uniform(math.log(0.001 * lam_p), math.log(2.0 * lam_p), 14))
    th = np.concatenate([[0.0], np.radians(rng.uniform(0.3, 90.0, 13))])
    rho = np.maximum(R * np.cos(th), 0.0)
    z = np.maximum(R * np.sin(th), 0.0)

    hc = below.Health()
    got = trans.t_surfaces_direct(
        et, kp, rho, z, -0.15, rtol=1e-9, omega=om, health=hc, selfconv=True
    )
    with force_numpy():
        hn = below.Health()
        ref = trans.t_surfaces_direct(
            et, kp, rho, z, -0.15, rtol=1e-9, omega=om, health=hn, selfconv=True
        )

    worst = max(cwise(got[k], ref[k]) for k in KEYS)
    record_property("worst_rel", float(worst))
    record_property("health_cpp", repr(hc.as_dict()))
    record_property("health_numpy", repr(hn.as_dict()))
    assert worst < G_568_15_TOL, f"{cell}: {worst:.3e}"

    assert hc.evaluations == hn.evaluations == 14
    assert hc.nonconvergent == hn.nonconvergent == 0
    assert hc.accelerated == hn.accelerated == 0
    assert hc.zero_kernel == hn.zero_kernel == 0
    assert hc.worst_selfconv_at == hn.worst_selfconv_at
    assert hc.max_tail_panels == hn.max_tail_panels, "the tail cost law moved"
    assert hc.max_head_panels == hn.max_head_panels


def test_g56815_the_shapes_and_dtypes_survive_the_dispatch():
    """Broadcasting is part of the contract: `t_surfaces_direct` returns
    arrays shaped like the broadcast (ρ, z, z′), and a batch entry point is
    exactly where a ravel can leak out. z′ broadcasts against a 2-D (ρ, z)
    here, which is the shape `TransmittedGrid.__init__` fills a rung with."""
    et, kp, om, km, lam_p = medium("C/7MHz")
    rho = np.array([[0.2], [0.7]]) * lam_p
    z = np.array([0.05, 0.5, 3.0])
    got = trans.t_surfaces_direct(et, kp, rho, z, -0.15, rtol=1e-9, omega=om)
    with force_numpy():
        ref = trans.t_surfaces_direct(et, kp, rho, z, -0.15, rtol=1e-9, omega=om)
    for k in KEYS:
        assert got[k].shape == (2, 3) == ref[k].shape
        assert got[k].dtype == np.complex128
    assert max(cwise(got[k], ref[k]) for k in KEYS) < G_568_15_TOL


# ======================================================================
# G-568-16 — the projected pair table
# ======================================================================
#
# MEASURED worst relative over the pair sets below, both directions of travel:
# 2.53e-14 on the one-rung grid and 5.44e-15 on the nine-rung one (which adds
# the cubic ln|z′| leg), 2.61e-16 on the coincident-pair branch, 0.0 at the
# domain edges. The projection is a fixed 4x4 (or 4x4x4) stencil and a handful
# of multiplies, so the two spellings differ only in reduction order and in
# `std::exp` versus `np.exp`; there is no adaptive decision here to diverge.
#
# The reciprocity transpose reads 4.24e-14 (one rung) and 3.00e-15 (nine rungs)
# across BOTH machines — on the C++ side it is ONE kernel with a flag, so this
# gate compares a transpose against itself rather than two ports against each
# other.
#
# Pinned at 1e-11 — ~600x over the realized worst, and still eight decades
# under the grid's own 1e-3/5e-3 interpolation bars, so this gate can only ever
# fire on a real port defect and never on the tabulation's accuracy.

G_568_16_TOL = 1e-11


@pytest.fixture(scope="module")
def proj_grids():
    """Two real tabulations on the reference soil: one rung, and a nine-rung
    ladder so the cubic ln|z′| leg is exercised too. Module-scoped: the fills
    are the expensive part of this file and every projection gate wants the
    same ones.

    Deliberately small in RADIUS only. Every layout decision they exercise is
    the shipped one — the log-R axis with its two pad rows, the 3° θ axis down
    to the grazing floor, the log-spaced ladder — and the C++ view reads the
    axes from the grid rather than assuming any of them.
    """
    et, kp, om, km, lam_p = medium("A/7MHz")
    one = trans.TransmittedGrid(et, kp, 0.01 * lam_p, 0.15, 0.15, rtol=1e-9, omega=om)
    many = trans.TransmittedGrid(et, kp, 0.002 * lam_p, 0.10, 0.20, rtol=1e-9, omega=om)
    assert one.n_zp == 1 and many.n_zp >= 4
    return one, many, (et, kp, om, km, lam_p)


def pair_set(seed, m, s, grid, zp_lo, zp_hi):
    """(above, t_above, below, t_below) with every pair inside the grid.

    The above points are placed in the grid's own polar coordinates about the
    origin and the below points are kept near it, so R and θ land in the served
    rectangle by construction rather than by luck.
    """
    rng = np.random.default_rng(seed)
    r = np.exp(rng.uniform(math.log(1.4 * grid.r_min), math.log(0.6 * grid.r_max), m))
    th = rng.uniform(max(grid.th_min, np.radians(2.0)), 0.5 * np.pi, m)
    az = rng.uniform(0.0, 2.0 * np.pi, m)
    above = np.c_[
        r * np.cos(th) * np.cos(az), r * np.cos(th) * np.sin(az), r * np.sin(th)
    ]
    span = 0.02 * grid.r_max
    below_pts = np.c_[
        rng.uniform(-span, span, s),
        rng.uniform(-span, span, s),
        -rng.uniform(zp_lo, zp_hi, s),
    ]
    t_above = rng.normal(size=(m, 3))
    t_below = rng.normal(size=(s, 3))
    # A purely vertical source tangent and a purely horizontal one: the two
    # ends of the tangent reading, including the sinφ term that is odd under
    # the direction reversal.
    t_below[0] = [0.0, 0.0, 1.0]
    t_below[-1] = [0.6, 0.8, 0.0]
    t_above[0] = [0.0, 0.0, 1.0]
    return above, t_above, below_pts, t_below


@pytest.mark.parametrize("seed,m,s", [(1, 9, 7), (2, 3, 11), (3, 12, 1)])
@pytest.mark.parametrize("which", ["one", "many"])
def test_g56816_the_projected_table_agrees(
    proj_grids, which, seed, m, s, record_property
):
    one, many, (et, kp, om, km, lam_p) = proj_grids
    grid = one if which == "one" else many
    lo, hi = (0.15, 0.15) if which == "one" else (0.105, 0.195)
    above, t_above, below_pts, t_below = pair_set(seed, m, s, grid, lo, hi)

    up = trans.transmitted_field_proj_below_to_above(
        above, t_above, below_pts, t_below, GROUND_Z, kp, km, grid
    )
    down = trans.transmitted_field_proj_above_to_below(
        below_pts, t_below, above, t_above, GROUND_Z, kp, km, grid
    )
    with force_numpy():
        up_ref = trans.transmitted_field_proj_below_to_above(
            above, t_above, below_pts, t_below, GROUND_Z, kp, km, grid
        )
        down_ref = trans.transmitted_field_proj_above_to_below(
            below_pts, t_below, above, t_above, GROUND_Z, kp, km, grid
        )
    assert up.shape == (m, s) == up_ref.shape
    assert down.shape == (s, m) == down_ref.shape
    worst = max(cwise(up, up_ref), cwise(down, down_ref))
    record_property("worst_rel", float(worst))
    assert worst < G_568_16_TOL, f"{which}/seed {seed}: {worst:.3e}"


@pytest.mark.parametrize("which", ["one", "many"])
def test_g56816_the_reciprocity_transpose_is_an_identity(
    proj_grids, which, record_property
):
    """D_{a→b}(r_b; r_a) = D_{b→a}(r_a; r_b)^T, on BOTH machines.

    Any deviation is a bug rather than a tolerance: with the verified (7f)
    both directions are literally the same integral over the same tables. On
    the C++ side they are also the same KERNEL, differing only in a flag that
    swaps T_ρ^V with T_z^H — which is why a family with four surfaces could
    not have served both directions.
    """
    one, many, (et, kp, om, km, lam_p) = proj_grids
    grid = one if which == "one" else many
    lo, hi = (0.15, 0.15) if which == "one" else (0.105, 0.195)
    above, t_above, below_pts, t_below = pair_set(7, 8, 6, grid, lo, hi)
    worst = 0.0
    for use_numpy in (False, True):
        ctx = force_numpy() if use_numpy else None
        if ctx:
            ctx.__enter__()
        try:
            up = trans.transmitted_field_proj_below_to_above(
                above, t_above, below_pts, t_below, GROUND_Z, kp, km, grid
            )
            down = trans.transmitted_field_proj_above_to_below(
                below_pts, t_below, above, t_above, GROUND_Z, kp, km, grid
            )
            worst = max(worst, cwise(down, up.T))
        finally:
            if ctx:
                ctx.__exit__()
    record_property("worst_rel", float(worst))
    assert worst < G_568_16_TOL, (
        f"{which}: the transpose is not an identity {worst:.3e}"
    )


def test_g56816_the_coincident_pair_takes_the_same_branch(proj_grids, record_property):
    """ρ → 0 degenerates the incidence azimuth and both spellings fall back to
    d̂ = (1, 0). Source points stacked directly under the observer, one
    vertical tangent and one horizontal, so the fallback is read on both."""
    one, _many, (et, kp, om, km, lam_p) = proj_grids
    above = np.array([[0.0, 0.0, 0.5 * one.r_max], [0.0, 0.0, 0.3 * one.r_max]])
    below_pts = np.array([[0.0, 0.0, -0.15], [0.0, 0.0, -0.15]])
    t_above = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t_below = np.array([[0.0, 0.0, 1.0], [0.6, 0.8, 0.0]])
    got = trans.transmitted_field_proj_below_to_above(
        above, t_above, below_pts, t_below, GROUND_Z, kp, km, one
    )
    with force_numpy():
        ref = trans.transmitted_field_proj_below_to_above(
            above, t_above, below_pts, t_below, GROUND_Z, kp, km, one
        )
    worst = cwise(got, ref)
    record_property("worst_rel", float(worst))
    assert worst < G_568_16_TOL, f"{worst:.3e}"


def test_g56816_a_duck_typed_surface_source_stays_on_numpy(proj_grids, monkeypatch):
    """The twin flattens a real tabulation's value table. Anything else that
    can `eval(R, theta, zp)` — the suite's direct-surface stand-in — has no
    `_vals`, takes the numpy body, and must keep doing so: the three-way
    grid → direct → goldens gates rest on the direct side NOT sharing the grid
    side's interpolation code."""
    one, _many, (et, kp, om, km, lam_p) = proj_grids

    class DirectSurfaces:
        regime = "below-above"

        def __init__(self, et, kp, om, r_max):
            self.et, self.kp, self.om, self.r_max = et, kp, om, r_max

        def eval(self, R, theta, zp):
            R = np.asarray(R, dtype=float)
            theta = np.asarray(theta, dtype=float)
            return trans.t_surfaces_direct(
                self.et,
                self.kp,
                np.maximum(R * np.cos(theta), 0.0),
                np.maximum(R * np.sin(theta), 0.0),
                zp,
                rtol=1e-9,
                omega=self.om,
            )

    surf = DirectSurfaces(et, kp, om, one.r_max)
    assert getattr(surf, "_vals", None) is None
    above = np.array([[0.2 * one.r_max, 0.0, 0.2 * one.r_max]])
    below_pts = np.array([[0.0, 0.0, -0.15]])
    t1 = np.array([[1.0, 0.0, 0.0]])

    # A spy, not an inference: comparing the two dispatches' NUMBERS would not
    # prove this, because the surfaces underneath a duck-typed source come from
    # `t_surfaces_direct`, which has a C++ path of its own.
    def boom(*a, **k):
        raise AssertionError("the twin was handed a duck-typed surface source")

    monkeypatch.setattr(trans._acc, "transmitted_field_proj_batch", boom)
    got = trans.transmitted_field_proj_below_to_above(
        above, t1, below_pts, t1, GROUND_Z, kp, km, surf
    )
    assert got.shape == (1, 1) and np.all(np.isfinite(got))
    # ... and the real tabulation DOES reach it, so the spy is live.
    with pytest.raises(AssertionError, match="duck-typed"):
        trans.transmitted_field_proj_below_to_above(
            above, t1, below_pts, t1, GROUND_Z, kp, km, one
        )


@pytest.fixture(scope="module")
def grazing_grid():
    """A grid whose grazing floor is ABOVE zero, so the θ-floor refusal can be
    read through the dispatch.

    `grazing_floor` solves the tail-cost law at the outermost bottom-row node,
    so a positive floor needs r_max > 375·|z′|_min — a shallow source under a
    wide grid. This one is deliberately shallow (0.2 mm) rather than
    deliberately wide, because widening the radius is what makes a transmitted
    fill expensive and the floor is the only thing this fixture is for. The
    bottom row still costs its full ~4000 panels per node, which is the point:
    a floor that did not bite would not be a floor.
    """
    et, kp, om, km, lam_p = medium("A/7MHz")
    grid = trans.TransmittedGrid(et, kp, 0.01 * lam_p, 2e-4, 2e-4, rtol=1e-9, omega=om)
    assert grid.th_min > 0.0
    return grid, (et, kp, om, km, lam_p)


@pytest.mark.parametrize(
    "what,match",
    [
        ("past_cap", "tabulated to R"),
        ("under_floor", "tabulated from R"),
        ("off_ladder", "ladder is log-spaced"),
        ("wrong_side_above", "at or above"),
        ("wrong_side_below", "STRICTLY below"),
        ("foreign_grid", "needs a TransmittedGrid"),
    ],
)
def test_g56816_the_refusals_survive_the_dispatch(proj_grids, what, match):
    """The grid's domain refusals plus the two endpoint ones, through the C++
    projection. None of that prose is transcribed into C++: the kernel reports
    the query's extremes in R, θ and |z′| and `TransmittedGrid.eval` raises
    them, so the two paths cannot disagree about which geometries are served.

    Every one of these is a REFUSAL rather than a clamp because the
    transmitted surface is the WHOLE field — there is no negligible tail to
    freeze the way `SommerfeldGrid.eval` freezes the ±=+ remainder — so a
    dispatch that quietly clamped would return a confident wrong number.
    """
    one, _many, (et, kp, om, km, lam_p) = proj_grids
    t1 = np.array([[1.0, 0.0, 0.0]])
    below_pts = np.array([[0.0, 0.0, -0.15]])
    g = one
    if what == "past_cap":
        above = np.array([[5.0 * one.r_max, 0.0, 0.1]])
    elif what == "under_floor":
        above = np.array([[0.1 * one.r_min, 0.0, 0.0]])
    elif what == "off_ladder":
        above = np.array([[0.2 * one.r_max, 0.0, 0.2 * one.r_max]])
        below_pts = np.array([[0.0, 0.0, -0.4]])
    elif what == "wrong_side_above":
        above = np.array([[0.2 * one.r_max, 0.0, -0.05]])
    elif what == "wrong_side_below":
        above = np.array([[0.2 * one.r_max, 0.0, 0.2 * one.r_max]])
        below_pts = np.array([[0.0, 0.0, +0.4]])
    else:
        above = np.array([[0.2 * one.r_max, 0.0, 0.2 * one.r_max]])
        g = som.get_grid(et, kp, 1.0, om)

    msgs = []
    for use_numpy in (True, False):
        ctx = force_numpy() if use_numpy else None
        if ctx:
            ctx.__enter__()
        try:
            with pytest.raises(ValueError, match=match) as e:
                trans.transmitted_field_proj_below_to_above(
                    above, t1, below_pts, t1, GROUND_Z, kp, km, g
                )
            msgs.append(str(e.value))
        finally:
            if ctx:
                ctx.__exit__()
    assert msgs[0] == msgs[1], msgs


def test_g56816_the_theta_floor_refusal_survives_the_dispatch(grazing_grid):
    """The fourth refusal, which needs a grid with a positive grazing floor.

    It is a COST law rather than a physics one — the tail must reach
    λ ~ 35/(z + |z′|) and the budget is 6000 panels — and the C++ path has to
    report the query's MINIMUM θ for it to fire. A kernel that reported only
    the maximum would serve the whole grazing band silently, which is the one
    place a truncated tail is waiting.
    """
    grid, (et, kp, om, km, lam_p) = grazing_grid
    t1 = np.array([[1.0, 0.0, 0.0]])
    below_pts = np.array([[0.0, 0.0, -2e-4]])
    # Two above points: one comfortably inside, one under the floor. The
    # refusal has to come from the second even though the first is served.
    r = 0.5 * grid.r_max
    th_bad = 0.3 * grid.th_min
    above = np.array(
        [
            [r * math.cos(0.5), 0.0, r * math.sin(0.5)],
            [r * math.cos(th_bad), 0.0, r * math.sin(th_bad)],
        ]
    )
    msgs = []
    for use_numpy in (True, False):
        ctx = force_numpy() if use_numpy else None
        if ctx:
            ctx.__enter__()
        try:
            with pytest.raises(ValueError, match="COST law") as e:
                trans.transmitted_field_proj_below_to_above(
                    above,
                    np.repeat(t1, 2, axis=0),
                    below_pts,
                    t1,
                    GROUND_Z,
                    kp,
                    km,
                    grid,
                )
            msgs.append(str(e.value))
        finally:
            if ctx:
                ctx.__exit__()
    assert msgs[0] == msgs[1], msgs


def test_g56816_an_in_domain_query_at_the_edges_is_served_not_refused(
    proj_grids, record_property
):
    """The complement of the refusals: R exactly at the cap, R exactly at the
    inner floor and θ exactly at the grazing floor are IN the domain, and the
    extremes the kernel reports must not push them out of it by a rounding
    step."""
    one, _many, (et, kp, om, km, lam_p) = proj_grids
    t1 = np.array([[1.0, 0.0, 0.0]])
    below_pts = np.array([[0.0, 0.0, -0.15]])
    th = max(one.th_min, 0.0)
    above = np.array(
        [
            [one.r_max * math.cos(th), 0.0, one.r_max * math.sin(th)],
            [one.r_min * math.cos(th), 0.0, one.r_min * math.sin(th)],
        ]
    )
    got = trans.transmitted_field_proj_below_to_above(
        above, np.repeat(t1, 2, axis=0), below_pts, t1, GROUND_Z, kp, km, one
    )
    with force_numpy():
        ref = trans.transmitted_field_proj_below_to_above(
            above, np.repeat(t1, 2, axis=0), below_pts, t1, GROUND_Z, kp, km, one
        )
    assert np.all(np.isfinite(got))
    record_property("worst_rel", float(cwise(got, ref)))
    assert cwise(got, ref) < G_568_16_TOL


# ======================================================================
# G-568-17 — the families this unit did not build keep their bytes
# ======================================================================
#
# The standing arc-wide trap. U3 shares exactly one line of C++ with U2 —
# `mw568_below::gamma_cut`, deliberately shared rather than re-transcribed
# because the spelling IS the branch choice and two copies could drift — and
# shares nothing at all with the ±=+ family. Both are re-read from inside this
# build rather than inferred from a diff.
#
# MEASURED: 4.13e-12 on the ±=+ six integrals and 2.14e-13 on the below/below
# fills reproducing their own numpy body — both of the same order U2's file
# records, which is what says `gamma_cut` did not move under it.


def test_g56817_the_below_below_fills_did_not_move(record_property):
    """U2's below/below twin, re-read from the U3 build.

    The two units share `gamma_cut`. If U3 had "simplified" it — collapsed the
    two-sqrt product to sqrt(λ² − k²), say, which puts the cut on the segment
    BETWEEN the branch points and makes the head's first-quadrant detour
    illegal — this is the gate that would say so, because the below family
    would stop reproducing its own numpy body.
    """
    et = eps_tilde(13.0, 0.005, 7.0e6)
    kp = 2.0 * np.pi * 7.0e6 / C0
    worst = 0.0
    for rho, h in ((0.3, 0.2), (4.0, 0.9), (18.0, 0.05)):
        got = below._six_integrals_below(et, kp, rho, h, 1e-9)
        was = below._FORCE_NUMPY
        below._FORCE_NUMPY = True
        try:
            ref = below._six_integrals_below(et, kp, rho, h, 1e-9)
        finally:
            below._FORCE_NUMPY = was
        worst = max(worst, cwise(got, ref))
    record_property("worst_rel", float(worst))
    assert worst < 1e-9, f"the below/below fills moved: {worst:.3e}"


def test_g56817_the_above_above_family_did_not_move(record_property):
    """`_six_integrals` and its C++ batch carry every fig-13/fig-14 landmark
    that assumes a REAL decay-medium k. U3 shares `_gamma` and
    `_bessel_j0_j1x` with them on the numpy side and must not have disturbed
    either."""
    et = complex(20.0, -77.036)
    k2 = 0.14671
    rho = np.array([0.3, 1.0, 2.5])
    h = np.array([0.2, 0.05, 0.9])
    got = som._six_integrals_batch(et, k2, rho, h, rtol=1e-9)
    ref = np.stack([som._six_integrals(et, k2, rho[i], h[i], 1e-9) for i in range(3)])
    worst = cwise(got, ref)
    record_property("worst_rel", float(worst))
    assert worst < 1e-9, f"the ±=+ six integrals moved: {worst:.3e}"
