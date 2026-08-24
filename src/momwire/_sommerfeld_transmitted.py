"""The transmitted Sommerfeld family: a source on one side of the interface
read at an observer on the other (momwire#553 unit 3, the #524 phase-1
regime-1 arc).

`_sommerfeld` carries the ±=+ family (both points above) and
`_sommerfeld_below` its ±=− twin (both points below). Both are REMAINDERS —
a correction added to a direct term and an image term. This module is
neither. Above a buried source there is no direct term and no image term:
the transmitted integral IS the whole field,

    V_T(ρ,z,z′) = 2∫₀^∞ e^{−γ_m|z′|−γ_p z}/(k_m²γ_p + k_p²γ_m) J₀(λρ) λ dλ
    U_T(ρ,z,z′) = 2∫₀^∞ e^{−γ_m|z′|−γ_p z}/(γ_m + γ_p)         J₀(λρ) λ dλ

and every shortcut whose licence is "it is only a correction" is wrong here
by construction. That sentence is the unit's design rule and it is where
each of the four inversions recorded below comes from.

Five surfaces, not four
-----------------------
The observation-medium-plus components (AGARD-LS-131 paper 8, eqs 7a–7e,
transcribed in `scratch/524-phase0/proto/EQUATIONS.md`) are

    E_ρ^V = C₁ ∂²V_T/∂ρ∂z                    E_z^V = C₁ (∂²/∂z² + k_p²)V_T
    E_ρ^H = C₁ cosφ (∂²V_T/∂ρ² + U_T)        E_φ^H = −C₁ sinφ ((1/ρ)∂V_T/∂ρ + U_T)
    E_z^H = −C₁ cosφ ∂²V_T/∂ρ∂z′

The last one is the difference. At ±=+ the ∂/∂z′ derivative collapses onto
the ∂/∂z one (R_z^H = −cosφ·R_ρ^V), which is why momwire has always shipped
four surfaces; the transmitted family's two exponential legs carry DIFFERENT
γ, so ∂/∂z′ → ×(+γ_m) is not ∂/∂z → ×(−γ_p) and the fifth surface is
genuinely its own quantity. Phase 0 named it the highest-risk quantity in
the kit: it is the one component the licensed engine departs from the
prototype on by O(1) while agreeing to its own noise floor on everything
else, with empymod siding with the prototype. It is gated hardest here and
the engine is not its oracle at any depth.

Every surface is tabulated from ANALYTIC under-integral derivatives —
∂/∂ρ → −λJ₁, ∂²/∂ρ² via the Bessel ODE identity, ∂/∂z → ×(−γ_p),
∂/∂z′ → ×(+γ_m), (∂²/∂z² + k_p²) → ×λ² — never by differentiating an
interpolant. That is the LLNL-TR-490316 rule the phase-0 comment records:
evaluate the components directly, interpolate the components, never curl or
difference an interpolated field.

The z′ ladder
-------------
(ρ, z) surfaces over a LADDER of source depths, with e^{−jk_m|z′|} divided
out of everything — NEC-4's sub-region-2 trick, one factor. Phase 0
measured why: the raw scalars need >33 log-spaced nodes over
z′ ∈ [0.02, 8] m at every soil because the in-medium phase wiggles through
the range, and with that one factor removed the same 1e-3 cubic target
falls to ≤13–17 nodes over the product range. Past ~0.25 λ_m the two-ray
(λ₁/λ₂ saddle) structure returns and one phase divide cannot remove it, so
this grid REFUSES a deeper source and names the extension rate rather than
extrapolating into it.

Above → below is the same integral
----------------------------------
V_T is symmetric under exchanging which leg carries which γ, so the
downward-transmitted field is the upward one's reciprocity TRANSPOSE:
D_{a→b}(r_below; r_above) = D_{b→a}(r_above; r_below)^T. It is served by a
role exchange over the SAME tables and gated as an IDENTITY — against an
independent empymod run in the crossed configuration, and against the
by-construction transpose, because the role exchange REVERSES the
horizontal direction d̂ and sinφ is odd. Any deviation is a bug, not a
tolerance.

The four inversions
-------------------
U2 recorded three ±=+ conveniences that invert below the interface because
its remainder dominates. All three invert here too, for the stronger
reason, and a FOURTH one turned up that U2 never had to price — this one
inside phase 0's own measurement:

1. **Range caps refuse, never clamp.** `SommerfeldGrid.eval` freezes the
   surface amplitude past `r1_max` because out there the ±=+ remainder is a
   negligible 1/R₁ tail. The transmitted surface is the entire field, so a
   frozen amplitude is a fabrication with no small parameter behind it.
2. **Error norms are per-surface relative with a named floor**, never
   pooled against an of-scale maximum: the five surfaces span decades over
   the served rectangle and a pooled norm hides whichever one is small.
3. **ε̃ is keyed EXACTLY.** `_somm_eps_bucket`'s 1 % ladder is calibrated on
   the ±=+ premise; here ε̃ enters the divided-out phase e^{−jk_m|z′|} that
   the whole ladder architecture rests on, as well as the Fresnel amplitude.
4. **THE LADDER CONSTANT ITSELF INVERTS.** Phase 0 measured ≈13 nodes per
   quarter λ_m at 1e-3, cubic, and that number is the architecture's own
   sizing rule — but it was measured on the two SCALARS V_T and U_T, and
   what a phase-1 ladder tabulates is their DERIVATIVES. Every derivative
   taken under the integral weights the integrand toward larger λ
   (∂/∂ρ → −λJ₁, ∂/∂z′ → ×γ_m ~ λ) and therefore toward faster
   z′-variation. Re-measured on the five surfaces over phase 0's own range
   and observers — with this evaluator reproducing phase 0's scalar numbers
   exactly, 9/13/17 against its ≤13/≤13/≤17 — the surfaces need 21/21/25.
   1.5–2.3× the nodes for the same 1e-3, of which the of-scale-vs-relative
   norm (inversion 2) accounts for only 17→21; the rest is the quantity.
   Sizing this unit's ladder from the scalar constant would have served
   ~1e-2 while reporting 1e-3, so `ladder_nodes` carries its own measured
   rule. This is the one that BIT.

Two more that did not bite, recorded because they were checked: the
`scaled_to` frequency-reuse law has no analogue here for a reason rather
than an absence (the ±=+ law needs ONE length to normalize against and the
transmitted tabulation has two, the observer axes in λ_p and the z′ ladder
in λ_m, whose ratio √ε̃ is itself frequency-dependent); and the R₁ → 0
analytic-limit node has no transmitted twin, because a remainder runs into
a finite static limit at the origin and the whole field does not — the
transmitted surface carries the source's own 1/R² and 1/R³ near zone. That
second one is why the radial axis is logarithmic; see `TransmittedGrid`.

Health, not hope
----------------
The contour is `_sommerfeld_below`'s, shared rather than copied: `_head`,
`_tail_below`, `_adaptive_segment`, `_run_contour` and `Health` are
imported, and the only generalization they needed was to say out loud that
their `h` argument is the TAIL'S DECAY LENGTH — |z| + |z′| for two points
below, |z′| + z for a crossing pair — since the transmitted integrand
decays as e^{−γ_m|z′|−γ_p z} and both γ → λ. The below/below numbers are
pinned bit-for-bit against that rename.

C++ twins (momwire#568 unit 3)
------------------------------
Everything in this module is numpy and stays that way — the numpy spellings
are the REFERENCES, and the gates against their C++ twins are tolerances,
never bytes. What has a twin, behind the single `transmitted_fills_568`
capability flag (`_use_transmitted_accel`):

    _six_integrals_transmitted        transmitted_six_integrals_batch
    t_surfaces_direct                 its per-point loop, through the same
                                      batch — THIS is where the OpenMP region
                                      lives, and it is the arc's biggest
                                      single cluster
    transmitted_field_proj_*          transmitted_field_proj_batch, ONE kernel
                                      for both directions with `transposed`
                                      swapping T_ρ^V and T_z^H

`divide_out_transmitted`, `ladder_nodes`, `grazing_floor` and the two VECTOR
combiners (`_combine_transmitted`, `_combine_transmitted_transposed`) stay
numpy: closed forms or fully-vectorized pair algebra costing less than the
call that would dispatch them, and the vector spellings have no per-pair
consumer — `bspline`'s Galerkin fill reads the PROJECTED table. So does
everything that is a CONTRACT rather than arithmetic — the side-of-interface
refusals, the `Health` bookkeeping and `TransmittedGrid.eval`'s four
refusals — so each of those messages and counters has exactly one home.

The tail budget is the thing a reader of this section must not lose: a
truncated transmitted tail is not a graceful degradation (4.5e+3 relative,
measured), so `Health.nonconvergent` is contract on BOTH paths and the C++
batch reports `converged` per node for the Python layer to tally in node
order.

Measured on an i7-8550U: one contour node 10–16× the numpy one (the grazing
end, where the tail runs thousands of panels, is the best case), and the
slow transmitted ladder 446.0 s → 30.6 s at unchanged memory.
"""

from __future__ import annotations

import math
import os

import numpy as np

from ._accel import acc as _acc
from ._sommerfeld import (
    _GRID_CACHE,
    _GRID_CACHE_MAX,
    _MU0,
    _bessel_j0_j1x,
    _evict_fifo,
    _gamma,
    _somm_r1_bucket_wl,
    SommerfeldGrid,
)
from ._sommerfeld_below import (
    # Re-exported: one Health type for both buried families, so a caller
    # that fills a below/below grid and a transmitted one tallies into the
    # same counters instead of two shapes that have to be merged.
    Health,  # noqa: F401
    _ADAPT_DEPTH,
    _ADAPT_DEPTH_COARSE,
    _DETOUR,
    _DETOUR_COARSE,
    _GAUSS_N,  # noqa: F401  (named so the two families' rules cannot drift)
    _GW,
    _GWC,
    _GX,
    _GXC,
    _run_contour,
    k_medium,
    _c1_moment,
)

# The five transmitted surfaces, in the order (7a)–(7e) writes them. The
# first four are the below/below quartet's opposite numbers; `TzH` is the
# genuinely fifth one, ∂/∂z′, with no ±=+ analogue.
_SURF_KEYS_T = ("TrhoV", "TzV", "TrhoH", "TphiH", "TzH")

# ---------------------------------------------------------------------------
# The served domain — every number measured (momwire#553 U3)
# ---------------------------------------------------------------------------

# The z′ ladder's deep end, in IN-MEDIUM wavelengths λ_m = 2π/|k_m|.
# Phase 0's measured cliff: cubic interpolation in log|z′| on the divided-out
# scalars holds 1e-3 to ≈0.25 λ_m and past it the two-ray (λ₁/λ₂ saddle)
# structure returns, which one phase divide cannot remove (>33 nodes over
# [0.02, 8] m at every soil; a spherical-phase divide is no better).
_ZPRIME_MAX_LAMBDA_M = 0.25

# Ladder sizing, measured on the FIVE SURFACES rather than phase 0's two
# scalars — see `ladder_nodes` for the 1.5–2.3x that difference is worth and
# why it is this unit's fourth inversion.
_LADDER_PER_DECADE = 18.0
_LADDER_PER_QUARTER = 8.0
_LADDER_MIN = 9

# The observer lattice: log-spaced in R = √(ρ² + z²) and uniform in
# θ = atan2(z, ρ), both keyed to the AIR wavelength λ_p = 2π/k_p. Above the
# interface every wave propagates in air, so the observer-side structure is
# k_p-paced; k_m enters the AMPLITUDE (the Fresnel denominators) and the
# divided-out |z′| leg, not the observer axes.
#
# Measured worst per-surface relative interpolation error along a 97-point
# line, soil A / 7 MHz, z′ = −0.05 m, scored at the skipped nodes:
#
#   Δln R = 0.106   θ = 1°     3.0e-4      (extrapolated from 2.0e-3 at 0.168)
#                   θ = 5°     5.8e-4
#                   θ = 45°    2.8e-4
#                   θ = 90°    1.0e-4
#   Δθ = 3°        R = 0.1 m  3.2e-5
#                   R = 1 m    1.6e-5
#                   R = 10 m   7.4e-5
#                   R = 30 m   2.6e-5
#
# A UNIFORM (ρ, z) rectangle keyed to λ_p — the obvious lattice, and the one
# tried first — fails by three orders of magnitude and the failure is
# structural, not a spacing choice: at Δ = 0.012 λ_p it reads 8.0 (soil B) and
# 2.6e+1 (soil A) near the origin against 3.7e-3 in the far field, and
# refining to 0.012 from 0.050 λ_p barely moves it. The transmitted surface
# carries the source's own 1/R² near zone — the ±=± remainder families do
# not, which is exactly what their finite R₁ → 0 analytic limits are — so no
# uniform lattice can hold both the near decade and the far wavelength.
_R_STEP_LOG = 0.10
_DTH_DEG = 3.0

# Two rows past each end of the log-R axis so every IN-DOMAIN query gets the
# centred leg of the 4-point Lagrange stencil instead of the clamped end
# interval (U2 measured ~40x at the outer edge for the same trick). The θ
# axis is not padded: θ < 0 and θ > 90° are not geometries.
_PAD_ROWS = 2

# The observer's near floor, in AIR wavelengths: 0.043 m at 7 MHz, 0.014 m at
# 21 MHz. Inside it the tabulated quantity is the bare near zone and the log
# axis would have to keep going; going lower is cheap (10 nodes per e-fold)
# but it is a serve decision, so it refuses by name rather than clamping onto
# a node that does not exist.
#
# momwire#553 U5 is that serve decision, and it took the extension rather than
# the refusal: a buried radial under a base-fed vertical puts the monopole's
# lowest quadrature node a few millimetres above the radial's ground
# projection, so R runs three e-folds under this default on the ×1 anchor and
# five on the ×8 rung. `r_min` is therefore a CONSTRUCTOR ARGUMENT with this
# as its default, bucketed DOWN to whole `_R_STEP_LOG` steps of it so that
# every extended lattice's nodes coincide exactly with the default one's — the
# same coincidence `r_max`'s bucket exploits, in the other direction.
#
# What that costs and what it buys, measured on the anchor cell (soil A /
# 7 MHz, |z'| = 0.15 m, `test_gu5_transmitted_extended_r_min_interpolates`):
# 10 nodes per e-fold, and worst per-surface relative interpolation error
# against direct evaluation over R ∈ [5e-4, 0.043] m is 3.3e-6 — two
# decades INSIDE the 5.8e-4 the shipped domain carries, because at a fixed
# source depth the observer's R → 0 limit is not a near zone at all: the true
# separation is bounded below by |z'|, the field is finite there, and the
# divided-out surface goes to zero linearly in R. The 1/R² near zone the
# refusal names is the |z'| → 0 corner, which the z' ladder floors separately.
_R_MIN_LAMBDA_P = 0.001

# How far below `_R_MIN_LAMBDA_P` an extension may be asked for, in AIR
# wavelengths. 1e-6 λ_p is 4.3e-5 m at 7 MHz — six e-folds, 60 extra rows —
# and it is a COST bound, not an accuracy one: the measurement above holds
# because |z'| bounds the true separation, and a caller that wants lower still
# is really telling this family that its z' is about to go to zero too.
_R_MIN_FLOOR_LAMBDA_P = 1e-6

# The serve cap, in AIR wavelengths: 85.7 m at 7 MHz, 28.6 m at 21 MHz. The
# cap is a serve-domain decision and NOT a cost knob — the log axis makes
# range nearly free (7 nodes per doubling) and the far field is the easy end
# once the two-leg factor is divided out (measured 1.8e-5 over
# R ∈ [0.75, 2] λ_p at Δln R = 0.106, against 5.8e-4 inside it). It is set at
# the far edge of what was MEASURED, and past it `eval` refuses: the
# transmitted surface is the whole field, so there is no negligible tail to
# freeze the way `SommerfeldGrid.eval` freezes the ±=+ remainder. What 2 λ_p
# gives up: an HF deck whose above/below pair separation exceeds 85.7 m at
# 7 MHz or 28.6 m at 21 MHz — a 160 m top-loaded vertical over a buried
# screen would need the ladder extended, at 7 nodes per doubling.
_R_CAP_LAMBDA_P = 2.0

# The tail panel budget for this family, and the MEASURED cost law that sets
# the grazing floor with it.
#
# The transmitted tail must reach λ ≈ 35/h before e^{−λh} is negligible, and
# it is panelled on the J₀(λρ) zero lattice π/ρ, so it costs
#
#     panels ≈ (35/π)·ρ/h ≈ 12·cot θ_true,    θ_true = atan2(z + |z′|, ρ)
#
# — the transmitted twin of the below family's "~6.4/tan θ per node", with
# the SAME 1/tanθ law and a different constant. Measured, soil A / 7 MHz:
# 3127 panels at cot 285, 6262 at cot 571, 25629 at cot 2142, 51337 at cot
# 4283 (the last one 12.6 s for a single node).
#
# Running out of that budget is NOT a graceful degradation here, and this is
# the trap worth carrying forward. `_tail_below` falls back to Wynn epsilon
# on the truncated partial sums, which phase 0 kept for the h → 0 limit and
# never needed. On this family it fires and it is CATASTROPHICALLY wrong:
# truncating at 4000 panels reaches λ ≈ 147 where the answer needs λ ≈ 1750,
# and the accelerated value comes back off by 4.5e+3 RELATIVE (measured at
# ρ = 85.65 m, z = 0, z′ = −0.02 m) — not off by a tolerance, off by three
# and a half decades, silently, with `converged = False` as the only tell.
# So the fill is sized to converge honestly and `Health.nonconvergent` is
# asserted at zero by the gates rather than treated as a warning.
_MAX_TAIL_PANELS_T = 6000
# 16, not the measured 12: the law is an asymptotic and the budget is a
# CLIFF (see the Wynn note above), so the floor carries a 1.33x margin. At
# 12 exactly the bottom row's outermost node landed on the budget and two
# nodes per fill came back non-convergent.
_TAIL_PANELS_PER_COT = 16.0


# ---------------------------------------------------------------------------
# The C++ twins (momwire#568 unit 3)
# ---------------------------------------------------------------------------
#
# Everything below this line in the module is numpy, unchanged, and stays the
# REFERENCE. `_accelerators` carries a twin of the contour driver (the six
# integrals at a batch of (ρ, z, z′), OpenMP across nodes on U1's shared
# engine) and a twin of the pair projection; where a twin exists it is
# preferred and the numpy body is the fallback, following the
# `_bspline_kernels` pattern and U2's below/below precedent exactly.
#
# ITS OWN capability flag, deliberately neither `contour_engine_568` nor
# `below_fills_568`: a .so built at U1 or U2 exports those symbols but none of
# these, and a shared flag would have it claim a contract it cannot serve.
_HAVE_TRANSMITTED_FILLS_ACCEL = _acc is not None and bool(
    getattr(_acc, "transmitted_fills_568", False)
)

# The tests' handle on the dispatch — the parity gates have to drive BOTH
# machines inside one process, and a cross-process env var cannot do that.
# `MOMWIRE_TRANSMITTED_FORCE_NUMPY` is the same switch for a whole run (a
# timing comparison, a bisect); `monkeypatch.setattr(trans, "_FORCE_NUMPY",
# True)` is the same switch for one test. `_use_transmitted_accel` reads both
# at CALL time, so neither is baked in at import.
_FORCE_NUMPY = bool(os.environ.get("MOMWIRE_TRANSMITTED_FORCE_NUMPY"))


def _use_transmitted_accel():
    """True when the transmitted fills should ride the C++ contour engine."""
    return _HAVE_TRANSMITTED_FILLS_ACCEL and not _FORCE_NUMPY


def _six_transmitted_accel(k_p, k_m, rho, z, zp, swap, rtol, selfconv):
    """`transmitted_six_integrals_batch` with this module's machines filled in.

    Both rules, both depths and both detours are arguments rather than
    constants on the C++ side, so the fine machine and the coarse
    self-convergence twin are the same call — exactly as `_run_contour` takes
    them here. The rtol pair is computed HERE, not there: `min(rtol, 1e-11)`
    and its ×100 coarse partner are `_six_integrals_transmitted`'s own
    spelling and belong next to it. So is `_MAX_TAIL_PANELS_T`, which is this
    family's budget and not the engine's default.
    """
    rtol_fine = min(rtol, 1e-11)
    return _acc.transmitted_six_integrals_batch(
        float(k_p),
        complex(k_m),
        np.ascontiguousarray(rho, dtype=float),
        np.ascontiguousarray(z, dtype=float),
        np.ascontiguousarray(zp, dtype=float),
        bool(swap),
        rtol_fine,
        _ADAPT_DEPTH,
        _DETOUR,
        _GX,
        _GW,
        bool(selfconv),
        rtol_fine * 100.0,
        _ADAPT_DEPTH_COARSE,
        _DETOUR_COARSE,
        _GXC,
        _GWC,
        _MAX_TAIL_PANELS_T,
    )


# ---------------------------------------------------------------------------
# The integrand stack
# ---------------------------------------------------------------------------


def _integrand_six_transmitted(lam, rho, z, zp, k_p, k_m, swap=False):
    """The six λ-integrands of the transmitted family, stacked (6, n):

      0: ∂²V_T/∂ρ∂z        1: (∂²/∂z² + k_p²)V_T   2: ∂²V_T/∂ρ²
      3: (1/ρ)∂V_T/∂ρ      4: ∂²V_T/∂ρ∂z′          5: U_T

    All five derivatives are taken ANALYTICALLY under the integral sign, on
    the exponential and Bessel factors of (7f)/(7g):

        ∂/∂ρ J₀(λρ) = −λJ₁(λρ)
        ∂²/∂ρ² J₀(λρ) = λ²(J₁(λρ)/(λρ) − J₀(λρ))   [the Bessel ODE identity,
                                                     with `_bessel_j0_j1x`'s
                                                     series switch guarding
                                                     small λρ]
        ∂/∂z e^{−γ_p z}    = −γ_p·e        (z > 0)
        ∂/∂z′ e^{−γ_m|z′|} = +γ_m·e        (z′ < 0, so ∂|z′|/∂z′ = −1)
        (∂²/∂z² + k_p²)    = γ_p² + k_p²  = λ²

    Index 4 is the whole reason this family has five surfaces and not four:
    the two legs carry different γ, so ∂/∂z′ is not −∂/∂z and index 4 is not
    a multiple of index 0.

    `swap=True` builds the above→below twin — source ABOVE at z′ > 0
    carrying γ_p, observer BELOW at z < 0 carrying γ_m. Its scalar V_T is
    LITERALLY the same integral as the unswapped one with (z, z′) exchanged
    (the reciprocity identity phase 0 measured at 0.0), which is why the
    product path serves above→below as a transpose and this branch exists to
    GATE that rather than to be called by it.
    """
    lam = np.asarray(lam, dtype=np.complex128)
    g_p = _gamma(lam, k_p)
    g_m = _gamma(lam, k_m)
    if swap:
        e = 2.0 * np.exp(-g_p * abs(zp) - g_m * abs(z)) * lam
        d_z = +g_m
        d_zp = -g_p
    else:
        e = 2.0 * np.exp(-g_m * abs(zp) - g_p * z) * lam
        d_z = -g_p
        d_zp = +g_m
    # (∂²/∂z² + k²) → γ² + k² → λ² in EITHER medium, and it is written as
    # λ² rather than as the sum. Not a shortcut: γ² + k² is a difference of
    # two O(k²) numbers whose result is O(λ²), so at the bottom of the
    # contour (λ ~ 1e-3, |k_m| ~ 0.6) it cancels away eleven digits — the
    # swapped and unswapped spellings of the SAME quantity were measured
    # 3.5e-11 apart before this line said λ². The prototype carries the sum
    # and agrees to 5e-10 either way, which is how the cancellation was
    # found rather than inherited.
    zzk = lam * lam
    a = e / (k_m * k_m * g_p + k_p * k_p * g_m)  # the V_T kernel, (7f)
    u = e / (g_m + g_p)  # the U_T kernel, (7g)
    x = lam * rho
    b0, b1x = _bessel_j0_j1x(x)
    j1 = b1x * x
    dr = -lam * j1  # ∂/∂ρ
    return np.stack(
        [
            a * dr * d_z,
            a * zzk * b0,
            a * lam * lam * (b1x - b0),
            a * (-lam * lam * b1x),
            a * dr * d_zp,
            u * b0,
        ]
    )


def _check_transmitted_sides(rho, z, zp, swap):
    """The convention guards, by name — the ONE home for this prose.

    The SOURCE must be strictly on its own side of the interface (`swap`
    exchanges which side that is) and the observer on the other side, at or
    beyond it.

    The asymmetry in that sentence is not sloppiness, it is the geometry.
    `_sommerfeld_below` refuses θ → 0 because h = |z| + |z′| → 0 there strips
    the tail of its e^{−λh} decay entirely. The transmitted tail decays as
    e^{−λ(|z′| + z)} and the SOURCE leg alone is enough: an observer ON the
    interface still has |z′| > 0 holding the tail down, and the transmitted
    field is continuous across z = 0. So z = 0 is a legitimate limit and is
    tabulated, while a SOURCE on the interface is the crossing/contact
    geometry of momwire#524 phase 3 and refuses.

    Split out of `_six_integrals_transmitted` so the batch driver can run it
    per node, in node order, before the C++ call — these words are contract
    and the two paths must refuse the same geometries with the same message.
    """
    if rho < 0.0:
        raise ValueError(f"need rho >= 0, got {rho!r}")
    above, below_ = (zp, z) if swap else (z, zp)
    if not above >= 0.0 or not below_ < 0.0:
        raise ValueError(
            "the transmitted family needs the source STRICTLY on one side of "
            "the interface (z = 0) and the observer at or beyond the other: "
            f"got above-side height {above!r}, below-side height {below_!r}. "
            "The observer may sit ON the interface — the source leg alone "
            "keeps the tail decaying and the field is continuous there — but "
            "a SOURCE on the interface is the crossing/contact geometry of "
            "momwire#524 phase 3, and its tail has no decay at all"
        )


def _six_integrals_transmitted(
    eps_t,
    k2,
    rho,
    z,
    zp,
    rtol=1e-11,
    health=None,
    selfconv=False,
    where=None,
    swap=False,
):
    """The six transmitted λ-integrals at one (ρ, z, z′); returns (6,) complex.

    Convention guards, by name, in `_check_transmitted_sides`: the SOURCE
    must be strictly on its own side of the interface (`swap` exchanges which
    side that is) and the observer on the other side, at or above it. An
    observer ON the interface is legitimate and tabulated; a SOURCE on it is
    momwire#524 phase 3.

    The contour is `_sommerfeld_below`'s, unchanged. Its tail takes a DECAY
    LENGTH, and the transmitted integrand's is |z′| + z: both γ → λ at large
    λ, so e^{−γ_m|z′|−γ_p z} → e^{−λ(|z′|+z)} exactly as the below family's
    e^{−γ_m h} → e^{−λh}. That one argument is the whole of the sharing.
    """
    eps_t = complex(eps_t)
    rho = float(rho)
    z = float(z)
    zp = float(zp)
    k_p = float(k2)
    k_m = k_medium(eps_t, k_p)
    if k_m.imag > 0.0:  # pragma: no cover - k_medium already enforces it
        raise AssertionError(f"Im k_m must be <= 0 under e^(+j omega t), got {k_m!r}")
    _check_transmitted_sides(rho, z, zp, swap)

    if _use_transmitted_accel():
        vals, tp, hp, conv, accel, sconv = _six_transmitted_accel(
            k_p,
            k_m,
            np.array([rho]),
            np.array([z]),
            np.array([zp]),
            swap,
            rtol,
            selfconv,
        )
        if health is not None:
            health.note(where, int(tp[0]), int(hp[0]), bool(conv[0]), bool(accel[0]))
            if selfconv:
                health.note_selfconv(
                    where if where is not None else (rho, z, zp), float(sconv[0])
                )
        return vals[0]

    def f(lam):
        return _integrand_six_transmitted(lam, rho, z, zp, k_p, k_m, swap=swap)

    h_decay = abs(zp) + abs(z)
    fine, hp, tp, conv, accel = _run_contour(
        f,
        k_p,
        k_m,
        rho,
        h_decay,
        min(rtol, 1e-11),
        _ADAPT_DEPTH,
        _DETOUR,
        _GX,
        _GW,
        max_panels=_MAX_TAIL_PANELS_T,
    )
    if health is not None:
        health.note(where, tp, hp, conv, accel)
    if selfconv:
        coarse, _, _, _, _ = _run_contour(
            f,
            k_p,
            k_m,
            rho,
            h_decay,
            min(rtol, 1e-11) * 100.0,
            _ADAPT_DEPTH_COARSE,
            _DETOUR_COARSE,
            _GXC,
            _GWC,
            max_panels=_MAX_TAIL_PANELS_T,
        )
        scale = np.maximum(np.abs(fine), 1e-300)
        rel = float(np.max(np.abs(fine - coarse) / scale))
        if health is not None:
            health.note_selfconv(where if where is not None else (rho, z, zp), rel)
    return fine


def _six_integrals_transmitted_many(
    eps_t,
    k2,
    rho,
    z,
    zp,
    rtol=1e-11,
    health=None,
    selfconv=False,
    where=None,
    swap=False,
):
    """`_six_integrals_transmitted` over parallel (ρ, z, z′) arrays; (n, 6).

    `t_surfaces_direct`'s hot loop, and the only place in this module where
    the parallelism is worth taking: one node's contour is completely
    independent of every other, and a fill is thousands of them. With the
    accelerator it is one C++ call — OpenMP across nodes, the GIL released,
    U1's engine (allocation-free, no mutable static) on each thread. Without
    it, it is EXACTLY the per-point Python loop it replaces, calling the same
    `_six_integrals_transmitted`, in the same order.

    The scheduling is `dynamic` on the C++ side and that matters more here
    than it did for the below family: the tail cost runs as cot θ_true, so
    the bottom row of a (ln R, θ) fill costs thousands of panels per node
    where the top row costs tens, and the nodes arrive sorted by row.

    Everything that is not quadrature stays on this side: the
    side-of-interface refusals with their own words, applied per node IN NODE
    ORDER so the two paths refuse the same geometry first, and the `Health`
    bookkeeping, likewise in node order so `worst_selfconv_at` lands on the
    same node either way. `Health.nonconvergent` in particular is contract
    rather than decoration in this family — a truncated tail was measured
    4.5e+3 relative wrong — so `converged` comes back per node and is tallied
    here rather than being collapsed into a single flag in C++.
    """
    eps_t = complex(eps_t)
    rho = np.ascontiguousarray(rho, dtype=float).ravel()
    z = np.ascontiguousarray(z, dtype=float).ravel()
    zp = np.ascontiguousarray(zp, dtype=float).ravel()
    n = rho.size
    if z.size != n or zp.size != n:
        raise ValueError(
            f"rho, z and zp must have the same length, got {(n, z.size, zp.size)!r}"
        )
    if where is None:
        where = [None] * n

    if not _use_transmitted_accel():
        out = np.empty((n, 6), dtype=np.complex128)
        for i in range(n):
            out[i] = _six_integrals_transmitted(
                eps_t,
                k2,
                rho[i],
                z[i],
                zp[i],
                rtol,
                health=health,
                selfconv=selfconv,
                where=where[i],
                swap=swap,
            )
        return out

    k_p = float(k2)
    k_m = k_medium(eps_t, k_p)
    if k_m.imag > 0.0:  # pragma: no cover - k_medium already enforces it
        raise AssertionError(f"Im k_m must be <= 0 under e^(+j omega t), got {k_m!r}")
    for i in range(n):
        _check_transmitted_sides(float(rho[i]), float(z[i]), float(zp[i]), swap)

    vals, tp, hp, conv, accel, sconv = _six_transmitted_accel(
        k_p, k_m, rho, z, zp, swap, rtol, selfconv
    )
    if health is not None:
        for i in range(n):
            health.note(where[i], int(tp[i]), int(hp[i]), bool(conv[i]), bool(accel[i]))
            if selfconv:
                w = where[i]
                if w is None:
                    w = (float(rho[i]), float(z[i]), float(zp[i]))
                health.note_selfconv(w, float(sconv[i]))
    return vals


# ---------------------------------------------------------------------------
# The divide-out
# ---------------------------------------------------------------------------


def divide_out_transmitted(k_p, k_m, rho, z, zp):
    """g = e^{−j(k_m|z′| + k_p R)}/R with R = √(ρ² + (z + |z′|)²) — the factor
    the transmitted surfaces are the amplitude of, so `surface × g` is the
    field.

    Two legs, because the transmitted ray has two: |z′| straight up through
    the ground at k_m, then the rest of the way through the AIR at the real
    k_p, over the true source-to-observer distance R, with the 1/R spherical
    spreading every point-source field carries. It is the transmitted twin of
    the below family's measured two-leg blend, and the e^{−jk_m|z′|} half is
    not a choice at all — it is the phase-0 measurement the whole ladder
    architecture rests on.

    The (ρ, z) half IS a choice and it was MEASURED against three
    alternatives, worst per-surface relative interpolation error along the
    shipped log-R axis at Δln R = 0.168 — deliberately one step COARSER than
    the shipped 0.10, so the candidates separate — soil A / 7 MHz,
    z′ = −0.05 m:

        divide-out                               θ=1°    θ=5°    θ=45°  θ=90°
        e^{−j(k_m|z′| + k_p R)}/R     (shipped)  2.0e-3  1.2e-3  4.8e-4 1.0e-4
        ... × (−1 + 3ja + 3a²), a=1/k_pR        2.9e-3  3.0e-3  1.8e-4 2.0e-4
        ... × (1 − ja − a²)                      (worse than both, 2–20x)

    The near-field-polynomial candidates are the interesting losers. They
    collapse the table's dynamic range spectacularly — 1.2e+2 against
    1.8e+5 for the shipped factor, because they remove the 1/R³ static
    growth exactly — and they do win in the steep band. But they LOSE in the
    grazing band, which is the product band (a buried radial read at an
    elevated observer is a small-θ pair), and the metric that decides is
    per-surface relative error, which does not care about dynamic range. So
    the plain two-leg factor ships and the polynomial is recorded as
    measured-and-rejected rather than as an idea nobody had.

    A pure e^{−jk_m|z′|} with no observer-side factor at all was measured
    too and is far worse everywhere the origin is near: 2.6e+1 against
    8.0e+0 on a uniform λ_p rectangle at soil B / 7 MHz, and 5.4e-3 against
    3.8e-3 out in the far field of soil A.

    Both consumers must pair with THIS factor: handing a surface the wrong g
    is a silent decade rather than an error, which is why every entry point
    that takes surfaces also takes k_p and k_m.
    """
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    zp = np.asarray(zp, dtype=float)
    dz = z + np.abs(zp)
    r = np.sqrt(rho * rho + dz * dz)
    return np.exp(-1j * (k_m * np.abs(zp) + k_p * r)) / r


# ---------------------------------------------------------------------------
# The five surfaces, evaluated directly
# ---------------------------------------------------------------------------


def t_surfaces_direct(
    eps_t,
    k2,
    rho,
    z,
    zp,
    rtol=1e-9,
    omega=None,
    mu=_MU0,
    health=None,
    selfconv=False,
    swap=False,
):
    """Direct (no-grid) evaluation of the five transmitted surfaces at
    (ρ, z, z′). Returns a dict of complex arrays shaped like the broadcast
    inputs.

    `k2` is the FREE-SPACE wavenumber (real), as everywhere in this tree;
    k_m is derived inside. Writing P = 1/g for the reciprocal of
    `divide_out_transmitted`:

        T_ρ^V = C₁·P·∂²V_T/∂ρ∂z
        T_z^V = C₁·P·(∂²/∂z² + k_p²)V_T
        T_ρ^H = C₁·P·(∂²V_T/∂ρ² + U_T)
        T_φ^H = −C₁·P·((1/ρ)∂V_T/∂ρ + U_T)
        T_z^H = −C₁·P·∂²V_T/∂ρ∂z′

    with C₁ = −jωμ/4π and ω defaulting to k₂·c. There is no prefactor
    pattern to get wrong here and no k_o²/k_s² ratio: (7a)–(7e) are written
    in the observation medium directly, and the observation medium is air.

    O(ms) per point. This is the grid's fill function and the tests' oracle
    hook, never a per-pair call.
    """
    k_p = float(k2)
    if omega is None:
        from ._sommerfeld import _C_LIGHT

        omega = k_p * _C_LIGHT
    eps_t = complex(eps_t)
    k_m = k_medium(eps_t, k_p)

    rho_b, z_b, zp_b = np.broadcast_arrays(
        np.asarray(rho, dtype=float),
        np.asarray(z, dtype=float),
        np.asarray(zp, dtype=float),
    )
    shape = rho_b.shape
    rf = rho_b.ravel()
    zf = z_b.ravel()
    pf = zp_b.ravel()

    # The per-point loop, through `_six_integrals_transmitted_many` — one C++
    # call with OpenMP across nodes where the accelerator is live, and exactly
    # this loop, in this order, where it is not.
    six = _six_integrals_transmitted_many(
        eps_t,
        k_p,
        rf,
        zf,
        pf,
        rtol,
        health=health,
        selfconv=selfconv,
        where=[(float(rf[i]), float(zf[i]), float(pf[i])) for i in range(rf.size)],
        swap=swap,
    )
    v_rz, v_zzk, v_rr, v_r1, v_rzp, u = six.T
    cm = _c1_moment(omega, mu)
    phase = 1.0 / divide_out_transmitted(k_p, k_m, rf, zf, pf)
    out = {
        "TrhoV": cm * phase * v_rz,
        "TzV": cm * phase * v_zzk,
        "TrhoH": cm * phase * (v_rr + u),
        "TphiH": -cm * phase * (v_r1 + u),
        "TzH": -cm * phase * v_rzp,
    }
    return {kk: out[kk].reshape(shape) for kk in _SURF_KEYS_T}


# ---------------------------------------------------------------------------
# The azimuth combination
# ---------------------------------------------------------------------------


def _combine_transmitted(surf, g, dhx, dhy, p):
    """The (7a)–(7e) azimuth combination as a VECTOR, summed over sources:
    (N_obs, 3) complex.

    Linear in the source, so a complex current moment p = I·dl enters
    through the two bilinears p·d̂ and p×d̂|_z and p_z, with no
    magnitude/direction split — `_field_point._reflected_wave_combination`'s
    argument, unchanged: a complex horizontal moment has no direction to
    normalize.

    The one structural difference from the ±=± families: E_z's horizontal
    row reads the FIFTH surface rather than −cosφ·T_ρ^V. That collapse is a
    ±=+ identity and it does not hold here.
    """
    p = np.asarray(p, dtype=complex)
    cph = p[None, :, 0] * dhx + p[None, :, 1] * dhy
    sph = p[None, :, 0] * dhy - p[None, :, 1] * dhx
    pz = p[None, :, 2]

    e_rho = g * (pz * surf["TrhoV"] + cph * surf["TrhoH"])
    e_phi = g * sph * surf["TphiH"]
    e_z = g * (pz * surf["TzV"] + cph * surf["TzH"])

    out = np.empty((dhx.shape[0], 3), dtype=complex)
    out[:, 0] = np.sum(dhx * e_rho - dhy * e_phi, axis=1)
    out[:, 1] = np.sum(dhy * e_rho + dhx * e_phi, axis=1)
    out[:, 2] = np.sum(e_z, axis=1)
    return out


def _combine_transmitted_transposed(surf, g, dhx, dhy, p):
    """`_combine_transmitted`'s reciprocity TRANSPOSE, summed over sources.

    Written out rather than built as a 3×3 and transposed, because the
    transpose is small and explicit: with ĉ = d̂x, ŝ = d̂y the dyad is

        [ ĉ²T_ρ^H − ŝ²T_φ^H   ĉŝ(T_ρ^H + T_φ^H)   ĉ T_ρ^V ]
        [ ĉŝ(T_ρ^H + T_φ^H)   ŝ²T_ρ^H − ĉ²T_φ^H   ŝ T_ρ^V ]
        [ ĉ T_z^H             ŝ T_z^H             T_z^V   ]

    whose 2×2 horizontal block is already symmetric, so transposing swaps
    exactly one pair: T_ρ^V (the vertical source's radial row) with T_z^H
    (the horizontal source's vertical row) — the FIFTH surface. That is the
    whole content of "above→below is the transpose", and it is why a family
    with four surfaces could not have served both directions.

    `dhx, dhy` must be the horizontal direction from the BELOW point to the
    ABOVE point in BOTH directions of travel — the surfaces are tabulated
    with the below point as the source, and sinφ is odd. Reversing it is
    the one silent sign error available here, which is why the caller
    computes it from the geometry rather than from source-minus-observer.
    """
    p = np.asarray(p, dtype=complex)
    cph = p[None, :, 0] * dhx + p[None, :, 1] * dhy
    sph = p[None, :, 0] * dhy - p[None, :, 1] * dhx
    pz = p[None, :, 2]

    e_rho = g * (pz * surf["TzH"] + cph * surf["TrhoH"])
    e_phi = g * sph * surf["TphiH"]
    e_z = g * (pz * surf["TzV"] + cph * surf["TrhoV"])

    out = np.empty((dhx.shape[0], 3), dtype=complex)
    out[:, 0] = np.sum(dhx * e_rho - dhy * e_phi, axis=1)
    out[:, 1] = np.sum(dhy * e_rho + dhx * e_phi, axis=1)
    out[:, 2] = np.sum(e_z, axis=1)
    return out


# ---------------------------------------------------------------------------
# The ladder + grid
# ---------------------------------------------------------------------------


def ladder_nodes(zp_min, zp_max, lam_m):
    """The z′ ladder for a requested depth range: log-spaced |z′| nodes.

    Returns `(nodes, n)` with `nodes` ASCENDING in |z′|.

    A single depth is ONE rung and no z′ interpolation — the common product
    case, a radial screen buried at one depth, and the reason this family's
    fill is affordable at all. Otherwise the count comes from the measured
    rule below.

    THE SIZING RULE IS NOT PHASE 0'S, AND THE DIFFERENCE IS THE UNIT'S
    FOURTH INVERSION. Phase 0 measured ≈13 nodes per quarter λ_m at 1e-3,
    cubic — on the two SCALARS V_T and U_T. What a phase-1 ladder actually
    carries is their DERIVATIVES, and every derivative taken under the
    integral sign weights the integrand toward larger λ (∂/∂ρ → −λJ₁,
    ∂/∂z′ → ×γ_m ~ λ), i.e. toward faster z′-variation. Re-measured on the
    five surfaces over phase 0's own range [0.02, 1] m and its own four
    observers, with this module's evaluator reproducing phase 0's scalar
    numbers exactly:

        cell        scalars (phase 0's metric)   five surfaces
        A/7 MHz     9   (phase 0: ≤13)           21
        B/7 MHz     13  (phase 0: ≤13)           21
        A/21 MHz    17  (phase 0: ≤17)           25

    1.5–2.3× more nodes for the same 1e-3. Sizing a ladder from the scalar
    constant would have quietly served ~1e-2. (The of-scale-vs-relative norm
    — U2's second inversion — accounts for only 17→21 of that; the rest is
    the quantity, not the metric.)

    So: `_LADDER_PER_DECADE` nodes per decade of log span plus
    `_LADDER_PER_QUARTER` per quarter λ_m of extent, which covers every
    measured cell with ~1.2–1.4× margin, floored at `_LADDER_MIN`.
    """
    lo = float(abs(zp_min))
    hi = float(abs(zp_max))
    if lo <= 0.0 or hi < lo:
        raise ValueError(f"need 0 < |z'|_min <= |z'|_max, got {(zp_min, zp_max)!r}")
    if hi <= lo * (1.0 + 1e-9):
        return np.array([lo]), 1
    decades = math.log10(hi / lo)
    quarters = hi / (0.25 * lam_m)
    n = int(
        math.ceil(_LADDER_PER_DECADE * decades + _LADDER_PER_QUARTER * quarters) + 1
    )
    n = max(n, _LADDER_MIN)
    return np.exp(np.linspace(math.log(lo), math.log(hi), n)), n


class TransmittedGrid:
    """The below→above tabulation: (ln R, θ) surfaces over a z′ ladder.

    Three coordinate decisions, each a MEASUREMENT (momwire#553 U3).

    **The observer axes are POLAR and the radial one is LOGARITHMIC.** The
    brief's rectangular (ρ, z) lattice keyed to λ_p was tried first and it
    does not work, by three orders of magnitude: a uniform λ_p-keyed
    rectangle read 8.0 … 1.7e+3 relative near the origin against 3.7e-3 in
    the far field, at every spacing from 0.012 to 0.05 λ_p. The reason is
    the fourth inversion's twin. The ±=+ and ±=− families tabulate a
    REMAINDER, whose surfaces run into a finite analytic limit at R₁ → 0
    (eqs 169–172); the transmitted surface is the whole field, so it carries
    the source's own 1/R² and 1/R³ near zone and there is no limit node to
    fill — the tabulated quantity genuinely diverges toward the source. A
    uniform lattice cannot cover a decade-spanning near zone and a
    wavelength-paced far zone at once; a logarithmic radial axis covers
    both, and it is what the below family's `dr_near`/`dr_far` split was
    approximating. Measured per-axis interpolation error at the shipped
    spacings, worst per surface over a 97-point line, soil A/7 MHz:

        Δln R = 0.106  θ = 1°   3e-4      θ = 5°   5.8e-4   θ = 45°  2.8e-4
        Δθ = 3°        R = 0.1 m 3.2e-5   R = 1 m  1.6e-5   R = 10 m 7.4e-5

    **R is the OBSERVER's distance and θ its elevation, both measured from
    the source's ground projection** — R = √(ρ² + z²), θ = atan2(z, ρ) — and
    not the source-to-observer pair. That keeps the served domain a clean
    rectangle at every rung: z = R sinθ ≥ 0 for every θ ∈ [0°, 90°], so
    there are no unfillable nodes. Defining θ from the true separation
    (z + |z′|) instead puts a curved z > 0 boundary through the lattice and
    the lower-left corner of every rung is then geometry that does not
    exist.

    **θ runs down to 0° and there is no grazing floor.** `_sommerfeld_below`
    refuses below 1° because h = |z| + |z′| → 0 strips the tail of its
    decay. Here the source leg |z′| holds the tail down on its own, the
    transmitted field is continuous across z = 0, and the measurement says
    so: the θ-line from 0° to 90° interpolates to ≤7.4e-5 at Δθ = 3° with
    the θ = 0 node filled by direct evaluation like any other.

    Past `r_max` this grid REFUSES rather than clamping (`SommerfeldGrid`
    freezes the amplitude out there because its remainder is a negligible
    tail; the transmitted surface is the entire field). Deeper than the
    ladder it REFUSES and names the extension rate.
    """

    regime = "below-above"

    def __init__(
        self,
        eps_t,
        k2,
        r_max,
        zp_min,
        zp_max,
        rtol=1e-9,
        omega=None,
        mu=_MU0,
        health=None,
        r_min=None,
    ):
        from ._sommerfeld import _C_LIGHT

        self.eps_t = complex(eps_t)
        self.k2 = float(k2)
        self.omega = self.k2 * _C_LIGHT if omega is None else float(omega)
        self.mu = float(mu)
        self.k_m = k_medium(self.eps_t, self.k2)
        lam_p = 2.0 * np.pi / self.k2
        lam_m = 2.0 * np.pi / abs(self.k_m)
        self.lam_p = lam_p
        self.lam_m = lam_m

        self.r_min = r_min_bucket(r_min, lam_p)
        self.r_cap = _R_CAP_LAMBDA_P * lam_p
        self.r_max = min(max(float(r_max), 4.0 * self.r_min), self.r_cap)

        zp_top = abs(float(zp_max))
        if zp_top > _ZPRIME_MAX_LAMBDA_M * lam_m * (1.0 + 1e-9):
            raise ValueError(_deeper_than_ladder_message(zp_top, lam_m))
        self.zp_nodes, self.n_zp = ladder_nodes(zp_min, zp_max, lam_m)
        self.zp_min = float(self.zp_nodes[0])
        self.zp_max = float(self.zp_nodes[-1])
        if self.n_zp != 1 and self.n_zp < 4:  # pragma: no cover - rule floors at 9
            raise AssertionError(f"ladder of {self.n_zp} rungs cannot carry a cubic")

        self.lnr0 = math.log(self.r_min) - _PAD_ROWS * _R_STEP_LOG
        self.dlnr = _R_STEP_LOG
        self.n_r = (
            int(math.ceil(math.log(self.r_max / self.r_min) / _R_STEP_LOG))
            + 1
            + 2 * _PAD_ROWS
        )
        self.th_min = grazing_floor(self.r_max, self.zp_min)
        self.th0 = self.th_min
        n_int = max(int(round((90.0 - math.degrees(self.th_min)) / _DTH_DEG)), 4)
        self.n_th = n_int + 1
        self.dth = (0.5 * np.pi - self.th_min) / n_int

        r_nodes = np.exp(self.lnr0 + self.dlnr * np.arange(self.n_r))
        th_nodes = self.th0 + self.dth * np.arange(self.n_th)
        rr, tt = np.meshgrid(r_nodes, th_nodes, indexing="ij")
        rho = np.maximum(rr * np.cos(tt), 0.0)
        zz = np.maximum(rr * np.sin(tt), 0.0)

        vals = np.empty(
            (self.n_zp, len(_SURF_KEYS_T), self.n_r, self.n_th), dtype=np.complex128
        )
        for i, zpn in enumerate(self.zp_nodes):
            surf = t_surfaces_direct(
                self.eps_t,
                self.k2,
                rho,
                zz,
                -float(zpn),
                rtol=rtol,
                omega=self.omega,
                mu=self.mu,
                health=health,
            )
            vals[i] = np.stack([surf[k] for k in _SURF_KEYS_T])
        self._vals = vals
        if self.n_zp > 1:
            self.lnz0 = math.log(self.zp_min)
            self.dlnz = (math.log(self.zp_max) - self.lnz0) / (self.n_zp - 1)
        else:
            self.lnz0 = math.log(self.zp_min)
            self.dlnz = 0.0

    @property
    def nodes(self):
        """Total direct evaluations this grid cost."""
        return int(self.n_zp * self.n_r * self.n_th)

    def eval(self, R, theta, zp):
        """Interpolate the five transmitted surfaces at (R, θ, z′).

        R = √(ρ² + z²) and θ = atan2(z, ρ) are the OBSERVER's polar
        coordinates about the source's ground projection; z′ < 0 is the
        source depth. Cubic Lagrange in ln R and in θ, and — when the
        ladder has more than one rung — cubic in ln|z′| on the
        divided-out surfaces, which is phase 0's measured scheme.

        REFUSES, never clamps: past `r_max`, below `r_min`, and outside the
        ladder. A clamp anywhere here returns a confident wrong number,
        because there is no negligible tail in this family to freeze.
        """
        R = np.asarray(R, dtype=float)
        theta = np.asarray(theta, dtype=float)
        zp = np.asarray(zp, dtype=float)
        rb, tb, pb = np.broadcast_arrays(R, theta, zp)
        shape = rb.shape
        rf = rb.ravel()
        tf = tb.ravel()
        pf = np.abs(pb.ravel())

        if rf.size:
            if float(np.min(rf)) < self.r_min * (1.0 - 1e-9):
                raise ValueError(
                    "below→above grid tabulated from R = "
                    f"{self.r_min:.6g} m ({_R_MIN_LAMBDA_P:g} free-space "
                    f"wavelengths) but was queried at R = "
                    f"{float(np.min(rf)):.6g}: inside that radius the "
                    "transmitted surface is the source's own near zone, "
                    "which diverges as 1/R² — there is no analytic limit "
                    "node to clamp to, unlike the ±=± remainder families. "
                    "Add rungs at the bottom of the log axis (they cost "
                    f"{1.0 / _R_STEP_LOG:.0f} nodes per e-fold) or refuse "
                    "the pair"
                )
            if float(np.max(rf)) > self.r_max * (1.0 + 1e-9):
                raise ValueError(
                    "below→above grid tabulated to R = "
                    f"{self.r_max:.6g} m ({self.r_max / self.lam_p:.3g} "
                    "free-space wavelengths) but was queried at R = "
                    f"{float(np.max(rf)):.6g}: the transmitted surface is "
                    "the WHOLE field above a buried source, not a "
                    "remainder, so there is no negligible tail to freeze "
                    "and no honest clamp. Size the grid for the geometry "
                    "or refuse the geometry"
                )
            if float(np.max(tf)) > 0.5 * np.pi + 1e-12 or float(np.min(tf)) < -1e-12:
                raise ValueError(
                    "below→above grid theta must lie in [0, pi/2]: theta is "
                    "the OBSERVER's elevation over the source's ground "
                    "projection, so a theta outside that range means the "
                    "observer is not above the interface"
                )
            if float(np.min(tf)) < self.th_min * (1.0 - 1e-9):
                raise ValueError(
                    "below→above grid tabulated from theta = "
                    f"{math.degrees(self.th_min):.4g} deg but was queried at "
                    f"{math.degrees(float(np.min(tf))):.4g} deg. That floor is "
                    "a COST law, not a physics one: the transmitted tail is "
                    "panelled on the J0(lam rho) zeros and must reach "
                    "lam ~ 35/(z + |z'|), so it costs about "
                    f"{_TAIL_PANELS_PER_COT:g}*cot(theta_true) panels per node "
                    f"and this grid's budget is {_MAX_TAIL_PANELS_T:d}. Below "
                    "the floor the tail truncates, and a truncated tail here "
                    "does not degrade gracefully -- the Wynn fallback was "
                    "measured 4.5e+3 relative wrong. Raise the observer, "
                    "deepen the shallowest rung, or shrink the cap"
                )
            if float(np.min(pf)) < self.zp_min * (1.0 - 1e-9) or float(
                np.max(pf)
            ) > self.zp_max * (1.0 + 1e-9):
                raise ValueError(
                    _outside_ladder_message(
                        float(np.min(pf)),
                        float(np.max(pf)),
                        self.zp_min,
                        self.zp_max,
                        self.lam_m,
                    )
                )

        fr = (np.log(np.maximum(rf, 1e-300)) - self.lnr0) / self.dlnr
        i0 = np.clip(np.floor(fr).astype(int) - 1, 0, self.n_r - 4)
        wr = SommerfeldGrid._lagrange4(fr - i0)
        ft = (np.clip(tf, 0.0, 0.5 * np.pi) - self.th0) / self.dth
        j0 = np.clip(np.floor(ft).astype(int) - 1, 0, self.n_th - 4)
        wt = SommerfeldGrid._lagrange4(ft - j0)

        ii = i0[:, None] + np.arange(4)[None, :]
        jj = j0[:, None] + np.arange(4)[None, :]

        if self.n_zp == 1:
            block = self._vals[0][:, ii[:, :, None], jj[:, None, :]]
            out = np.einsum("snij,ni,nj->sn", block, wr, wt)
        else:
            fz = (np.log(np.maximum(pf, 1e-300)) - self.lnz0) / self.dlnz
            k0 = np.clip(np.floor(fz).astype(int) - 1, 0, self.n_zp - 4)
            wz = SommerfeldGrid._lagrange4(fz - k0)
            kk = k0[:, None] + np.arange(4)[None, :]
            block = self._vals[
                kk[:, :, None, None],
                :,
                ii[:, None, :, None],
                jj[:, None, None, :],
            ]
            out = np.einsum("nkijs,nk,ni,nj->sn", block, wz, wr, wt)

        return {key: out[s].reshape(shape) for s, key in enumerate(_SURF_KEYS_T)}

    def scaled_to(self, k2, omega, mu):
        raise NotImplementedError(
            "the transmitted grid has no normalized-master rescaling, and "
            "here that absence has a reason rather than an absence: "
            "rather than an absence: the +-=+ law S = omega*mu*G(eps_t; "
            "R1/lambda, theta) needs ONE length to normalize against, and "
            "the transmitted tabulation has two — the observer axes in "
            "lambda_p and the z' ladder in lambda_m — whose ratio sqrt(eps_t) "
            "is itself frequency-dependent, so no rescaling of one lattice "
            "carries the other"
        )


def r_min_bucket(r_min, lam_p):
    """The inner end of the log-R axis, snapped DOWN to a whole `_R_STEP_LOG`
    step of the default `_R_MIN_LAMBDA_P·λ_p`.

    `None` (or anything at or above the default) gives the default back
    unchanged, so every pre-U5 grid keeps its exact lattice. Anything smaller
    lands on `default·exp(−m·Δln R)` for an integer m ≥ 1: the extended axis's
    nodes then coincide EXACTLY with the default axis's, which is what makes
    an extension a longer table rather than a different one — and what lets
    the cache key carry m instead of a float that never repeats.

    Below `_R_MIN_FLOOR_LAMBDA_P·λ_p` it raises rather than extending: see
    that constant for why the bound is a cost one.
    """
    default = _R_MIN_LAMBDA_P * lam_p
    if r_min is None:
        return default
    r_min = float(r_min)
    if r_min >= default:
        return default
    floor = _R_MIN_FLOOR_LAMBDA_P * lam_p
    if r_min < floor:
        raise ValueError(
            f"the transmitted grid was asked to tabulate down to R = "
            f"{r_min:.6g} m, past the extension floor {floor:.6g} m "
            f"({_R_MIN_FLOOR_LAMBDA_P:g} free-space wavelengths). Extending "
            f"the log axis costs {1.0 / _R_STEP_LOG:.0f} rows per e-fold and "
            "the measurement behind it (per-surface interpolation error "
            "3.3e-6 over the extended decade) rests on the SOURCE DEPTH "
            "bounding the true separation from below, so an observer this "
            "close to a source's ground projection is really a shallow-|z'| "
            "geometry: deepen the source, raise the observer, or refuse the "
            "pair"
        )
    steps = math.ceil(math.log(default / r_min) / _R_STEP_LOG)
    return default * math.exp(-steps * _R_STEP_LOG)


def grazing_floor(r_max, zp_min):
    """The θ floor, in radians, that keeps every node inside the tail budget.

    Not a preference and not a copy of the below family's 1°: solve the
    measured cost law `panels ≈ _TAIL_PANELS_PER_COT · ρ/(z + |z′|)` at the
    grid's worst node — the largest R on the bottom θ row — for the θ that
    lands exactly on `_MAX_TAIL_PANELS_T`:

        sin θ_min = max(0, _TAIL_PANELS_PER_COT/_MAX_TAIL_PANELS_T
                           − |z′|_min / r_max)

    The shallowest LADDER RUNG appears because the source leg holds the tail
    down all by itself: a grid whose shallowest source is 15 cm and whose cap
    is 85.7 m gets θ_min = 0.014°, and one whose shallowest source is 2 cm
    gets 0.101°. Both are two orders of magnitude below the below/below
    family's 1° floor, because there h = |z| + |z′| goes to zero at grazing
    and here it cannot.

    A floor of exactly 0 is returned when the ladder is deep enough to pay
    for the whole bottom row, and then the θ = 0 row — the observer ON the
    interface — is tabulated like any other.
    """
    cot_max = _MAX_TAIL_PANELS_T / _TAIL_PANELS_PER_COT
    s = 1.0 / cot_max - abs(float(zp_min)) / float(r_max)
    return float(math.asin(min(max(s, 0.0), 1.0)))


def grid_extent(k2, r_max, zp_min, r_min=None):
    """`(r_min, r_max, th_min)` a `TransmittedGrid` would actually tabulate
    for this request — the two bucketings and the grazing floor, WITHOUT the
    fill.

    A serve-time caller has to check its geometry against the domain it will
    get, and the domain it will get is not the domain it asked for: `r_max`
    buckets UP (which RAISES the grazing floor, since the floor solves the
    tail-cost law at the outermost bottom-row node) and `r_min` buckets DOWN.
    Checking against the request instead of against the answer would let a
    geometry pass the serve check and then be refused from inside an
    80-second fill, with a message about a grid rather than about the deck.
    """
    lam_p = 2.0 * np.pi / float(k2)
    rb = _somm_r1_bucket_wl(min(float(r_max) / lam_p, _R_CAP_LAMBDA_P)) * lam_p
    rmin = r_min_bucket(r_min, lam_p)
    rmax = min(max(rb, 4.0 * rmin), _R_CAP_LAMBDA_P * lam_p)
    return rmin, rmax, grazing_floor(rmax, zp_min)


def _deeper_than_ladder_message(zp_top, lam_m):
    return (
        f"the transmitted grid was asked for a source depth {zp_top:.6g} m, "
        f"past the tabulated maximum {_ZPRIME_MAX_LAMBDA_M * lam_m:.6g} m "
        f"({_ZPRIME_MAX_LAMBDA_M:g} in-medium wavelengths). Beyond a quarter "
        "lambda_m the two-ray (lambda_1/lambda_2 saddle) structure of the "
        "transmitted integral returns and the single e^(-j k_m |z'|) divide-"
        "out no longer flattens it — momwire#524 phase 0 measured >33 nodes "
        "over the deep range at every soil, and a spherical-phase divide is "
        "no better. Extending the ladder is honest work and it costs about "
        f"{_LADDER_PER_QUARTER:g} extra rungs per additional quarter lambda_m "
        f"plus {_LADDER_PER_DECADE:g} per decade of log span, each rung a "
        "full (R, theta) fill; extrapolating past it is not"
    )


def _outside_ladder_message(lo, hi, zp_min, zp_max, lam_m):
    return (
        f"the transmitted grid tabulates source depths |z'| in [{zp_min:.6g}, "
        f"{zp_max:.6g}] m but was queried over [{lo:.6g}, {hi:.6g}] m. The z' "
        "ladder is log-spaced and cubic on the divided-out surfaces, so a "
        "query outside it is an extrapolation in the one variable the "
        "architecture is built around; size the ladder for the deck's "
        "buried depths (one depth is ONE rung and no interpolation at all) "
        f"or refuse the pair. Deeper rungs cost about {_LADDER_PER_QUARTER:g} "
        f"per quarter lambda_m = {0.25 * lam_m:.4g} m"
    )


def get_grid_below_above(
    eps_t,
    k2,
    r_max,
    zp_min,
    zp_max,
    omega,
    mu=_MU0,
    rtol=1e-9,
    health=None,
    r_min=None,
):
    """Cached `TransmittedGrid`, sharing `_sommerfeld`'s grid cache.

    The cache is the shared one with the regime discriminator U2 added, now
    carrying a third value: `("below-above", ...)` cannot collide with
    `("below", ...)` or with the ±=+ key even at identical (ε̃, k₂, ω, μ).

    **ε̃ is keyed EXACTLY** — U2's third inversion, and for a stronger
    reason here. `_somm_eps_bucket`'s 1 % Im ladder is calibrated on the
    ±=+ premise that the remainder is a small correction; below/below broke
    that because the remainder dominates. The transmitted family has no
    correction at all, and ε̃ enters TWICE over: through the divided-out
    phase e^{−jk_m|z′|}, which is the whole architecture, and through the
    Fresnel denominators k_m²γ_p + k_p²γ_m and γ_m + γ_p that set the
    amplitude. A bucketed ε̃ moves the ladder's own divide-out.

    **The z′ range is keyed exactly too**, which is not the r_max
    argument. `r_max` buckets UP because the log-R lattice is anchored at
    `r_min` with a fixed Δln R, so a grid tabulated further out has node
    positions that coincide exactly with the smaller one's. The ladder's
    nodes are log-spaced BETWEEN its two ends, so widening the range moves
    every rung — there is no such coincidence to exploit, and a deck's
    buried depths are a fixed set anyway.
    """
    k2 = float(k2)
    eps_t = complex(eps_t)
    lam_p = 2.0 * np.pi / k2
    rb_wl = _somm_r1_bucket_wl(min(float(r_max) / lam_p, _R_CAP_LAMBDA_P))
    # `r_min` buckets DOWN onto the default lattice's own log steps, so the
    # key carries a repeatable number rather than a float that never recurs.
    r_min_b = r_min_bucket(r_min, lam_p)
    key = (
        "below-above",
        eps_t,
        k2,
        rb_wl,
        r_min_b,
        float(abs(zp_min)),
        float(abs(zp_max)),
        float(omega),
        float(mu),
    )
    grid = _GRID_CACHE.get(key)
    if grid is None:
        _evict_fifo(_GRID_CACHE, _GRID_CACHE_MAX)
        grid = TransmittedGrid(
            eps_t,
            k2,
            rb_wl * lam_p,
            zp_min,
            zp_max,
            rtol=rtol,
            omega=float(omega),
            mu=float(mu),
            health=health,
            r_min=r_min_b,
        )
        _GRID_CACHE[key] = grid
    return grid


# ---------------------------------------------------------------------------
# The point evaluations
# ---------------------------------------------------------------------------


def _check_crossing_sides(above_pts, below_pts, ground_z, what):
    """The two endpoint refusals for a crossing pair set, by name.

    Split out of `_crossing_geometry` so the C++ projection twin can run them
    without building the pair tables it is about to replace: these words are
    contract and there is exactly one copy of them.
    """
    z = above_pts[:, 2] - ground_z
    depth = ground_z - below_pts[:, 2]
    if z.size and float(np.min(z)) < 0.0:
        raise ValueError(
            f"{what}: the above-interface endpoints must lie at or above "
            f"ground_z = {ground_z!r}, got minimum height {float(np.min(z)):.6g}"
        )
    if depth.size and float(np.min(depth)) <= 0.0:
        raise ValueError(
            f"{what}: the below-interface endpoints must lie STRICTLY below "
            f"ground_z = {ground_z!r}, got minimum depth "
            f"{float(np.min(depth)):.6g} (a non-positive depth is a point on "
            "or above the interface — that geometry is momwire#524 phase 3)"
        )
    return z, depth


def _crossing_geometry(above_pts, below_pts, ground_z, grid, what):
    """(ρ, z, z′, R, θ, d̂) for a crossing pair, with both sides checked.

    `d̂` is the horizontal unit vector from the BELOW point's ground
    projection to the ABOVE point's, in BOTH directions of travel — the
    surfaces are tabulated with the below point as the source. ρ → 0
    degenerates the azimuth and any horizontal direction serves there
    (T_ρ^H(90°) = −T_φ^H); a point source is a complex MOMENT with no
    direction of its own, so (1, 0) serves, exactly as `_field_point` does
    for the same cell.
    """
    above_pts = np.asarray(above_pts, dtype=float)
    below_pts = np.asarray(below_pts, dtype=float)
    z, depth = _check_crossing_sides(above_pts, below_pts, ground_z, what)
    dx = above_pts[:, 0][:, None] - below_pts[None, :, 0]
    dy = above_pts[:, 1][:, None] - below_pts[None, :, 1]
    rho = np.hypot(dx, dy)
    zz = np.broadcast_to(z[:, None], rho.shape)
    zp = np.broadcast_to(-depth[None, :], rho.shape)
    r_obs = np.hypot(rho, zz)
    theta = np.arctan2(zz, rho)
    tiny = 1e-12 * grid.r_max
    safe = rho > tiny
    inv = np.where(safe, 1.0 / np.where(safe, rho, 1.0), 0.0)
    dhx = np.where(safe, dx * inv, 1.0)
    dhy = np.where(safe, dy * inv, 0.0)
    return rho, zz, zp, r_obs, theta, dhx, dhy


def _require_transmitted_grid(grid, who):
    if getattr(grid, "regime", None) != "below-above":
        raise ValueError(
            f"{who} needs a TransmittedGrid: the reflected-wave and "
            "below/below grids tabulate four REMAINDER surfaces over a "
            "different divide-out, and the transmitted family has five and "
            "is the whole field"
        )


def _combine_transmitted_proj(surf, g, dhx, dhy, t_obs, t_src, transposed):
    """The (7a)–(7e) combination kept as a PAIR TABLE rather than summed:
    `[m, n] = t̂_m · D(r_m, r_n) · t̂_n`, `(n_obs, n_src)` complex.

    The same algebra `_combine_transmitted` / `_combine_transmitted_transposed`
    run, with the source moment read as the source TANGENT and the observer
    projection applied instead of a sum — the shape a Galerkin fill needs
    (`_sommerfeld_below.remainder_field_proj_below` is the below family's own
    twin of this, and `_sommerfeld.remainder_field_proj` the ±=+ one). Written
    here, next to the two vector spellings, so the fifth surface's placement
    is decided in ONE file: `transposed` swaps T_ρ^V with T_z^H and nothing
    else, which is the whole content of the reciprocity transpose.
    """
    cph = t_src[None, :, 0] * dhx + t_src[None, :, 1] * dhy
    sph = t_src[None, :, 0] * dhy - t_src[None, :, 1] * dhx
    pz = t_src[None, :, 2]

    if transposed:
        e_rho = g * (pz * surf["TzH"] + cph * surf["TrhoH"])
        e_z = g * (pz * surf["TzV"] + cph * surf["TrhoV"])
    else:
        e_rho = g * (pz * surf["TrhoV"] + cph * surf["TrhoH"])
        e_z = g * (pz * surf["TzV"] + cph * surf["TzH"])
    e_phi = g * sph * surf["TphiH"]

    return (
        t_obs[:, 0][:, None] * (dhx * e_rho - dhy * e_phi)
        + t_obs[:, 1][:, None] * (dhy * e_rho + dhx * e_phi)
        + t_obs[:, 2][:, None] * e_z
    )


def _proj_accel(
    above, t_above, below_pts, t_below, ground_z, k_p, k_m, grid, transposed
):
    """`transmitted_field_proj_batch` on an (above, below) pair set.

    Returns the `(n_above, n_below)` table in ROLE order; the downward caller
    transposes it into its own `(n_obs, n_src)`. One C++ kernel serves both
    directions, with `transposed` swapping T_ρ^V and T_z^H — if it were two
    kernels the reciprocity gate would be comparing two ports rather than a
    transpose against itself.

    THE FOUR REFUSALS ARE NOT TRANSCRIBED INTO C++. The kernel reports the
    query's extremes in R, θ and |z′| and they go straight back through
    `TransmittedGrid.eval`, which raises in its own words — one copy of that
    prose, and no way for the two paths to disagree about which geometries
    are served. Two points; the interpolation it also does is discarded and
    costs microseconds. That matters more in this family than in the
    remainder ones: every one of those four is a REFUSAL rather than a clamp
    precisely because the transmitted surface is the whole field, so a path
    that quietly clamped would return a confident wrong number.
    """
    out, r_lo, r_hi, th_lo, th_hi, zp_lo, zp_hi = _acc.transmitted_field_proj_batch(
        above,
        t_above,
        below_pts,
        t_below,
        float(ground_z),
        float(k_p),
        complex(k_m),
        bool(transposed),
        float(grid.lnr0),
        float(grid.dlnr),
        float(grid.th0),
        float(grid.dth),
        float(grid.lnz0),
        float(grid.dlnz),
        float(grid.r_max),
        grid._vals,
    )
    if above.shape[0] and below_pts.shape[0]:
        grid.eval(
            np.array([r_lo, r_hi]),
            np.array([th_lo, th_hi]),
            np.array([-zp_lo, -zp_hi]),
        )
    return out


def _proj_accel_live(grid):
    """True when this grid can be handed to the C++ projection twin.

    A duck-typed `eval`-able (the tests' direct-surface stand-in) has no
    `_vals` to flatten and takes the numpy body — which is also what keeps
    the three-way grid → direct → goldens gates honest, since the direct side
    must not share the grid side's interpolation code.
    """
    return _use_transmitted_accel() and getattr(grid, "_vals", None) is not None


def transmitted_field_proj_below_to_above(
    obs, t_obs, src, t_src, ground_z, k_p, k_m, grid
):
    """Projected transmitted table `t̂_m · D · t̂_n` for observers ABOVE the
    interface and sources BELOW it: `(n_obs, n_src)` complex.

    The WHOLE field per unit source moment, not a remainder — a Galerkin fill
    consuming this must not add a direct or an image term to it, which is the
    contract difference `_field_point`'s crossing regimes still refuse over.

    C++ TWIN (momwire#568 unit 3): `transmitted_field_proj_batch`, which
    inlines `_crossing_geometry`, `TransmittedGrid.eval`,
    `divide_out_transmitted` and `_combine_transmitted_proj` with no
    intermediates. The endpoint refusals and the grid's four domain refusals
    stay here.
    """
    _require_transmitted_grid(grid, "transmitted_field_proj_below_to_above")
    if _proj_accel_live(grid):
        above = np.asarray(obs, dtype=float)
        below_pts = np.asarray(src, dtype=float)
        _check_crossing_sides(
            above, below_pts, ground_z, "transmitted_field_proj_below_to_above"
        )
        return _proj_accel(
            above,
            np.asarray(t_obs, dtype=float),
            below_pts,
            np.asarray(t_src, dtype=float),
            ground_z,
            k_p,
            k_m,
            grid,
            transposed=False,
        )
    rho, zz, zp, r_obs, theta, dhx, dhy = _crossing_geometry(
        obs, src, ground_z, grid, "transmitted_field_proj_below_to_above"
    )
    surf = grid.eval(r_obs, theta, zp)
    g = divide_out_transmitted(k_p, k_m, rho, zz, zp)
    return _combine_transmitted_proj(
        surf,
        g,
        dhx,
        dhy,
        np.asarray(t_obs, dtype=float),
        np.asarray(t_src, dtype=float),
        transposed=False,
    )


def transmitted_field_proj_above_to_below(
    obs, t_obs, src, t_src, ground_z, k_p, k_m, grid
):
    """Projected transmitted table for observers BELOW the interface and
    sources ABOVE it: `(n_obs, n_src)` complex.

    The reciprocity TRANSPOSE of `transmitted_field_proj_below_to_above` over
    the SAME tables, read at the same (R, θ, z′) — the above point supplies R
    and θ, the below point supplies z′ — with the dyad transposed and the
    horizontal direction still taken from the below point to the above one.
    Any disagreement with the upward direction's transpose is a BUG, not a
    tolerance, and it is gated as an identity.

    C++ TWIN (momwire#568 unit 3): the SAME kernel the upward direction
    reads, with `transposed=True`. It returns its table in role order, so the
    transpose here is the same `.T` the numpy body applies to the pair-shaped
    arrays — the identity is not re-implemented, it is re-read.
    """
    _require_transmitted_grid(grid, "transmitted_field_proj_above_to_below")
    if _proj_accel_live(grid):
        above = np.asarray(src, dtype=float)
        below_pts = np.asarray(obs, dtype=float)
        _check_crossing_sides(
            above, below_pts, ground_z, "transmitted_field_proj_above_to_below"
        )
        return _proj_accel(
            above,
            np.asarray(t_src, dtype=float),
            below_pts,
            np.asarray(t_obs, dtype=float),
            ground_z,
            k_p,
            k_m,
            grid,
            transposed=True,
        ).T
    # The geometry table comes back as (above, below) = (source, observer), so
    # every pair-shaped array transposes into this call's (n_obs, n_src).
    rho, zz, zp, r_obs, theta, dhx, dhy = _crossing_geometry(
        src, obs, ground_z, grid, "transmitted_field_proj_above_to_below"
    )
    surf = grid.eval(r_obs, theta, zp)
    g = divide_out_transmitted(k_p, k_m, rho, zz, zp)
    return _combine_transmitted_proj(
        {k: v.T for k, v in surf.items()},
        g.T,
        dhx.T,
        dhy.T,
        np.asarray(t_obs, dtype=float),
        np.asarray(t_src, dtype=float),
        transposed=True,
    )


def transmitted_field_below_to_above(obs, src, src_moment, ground_z, k_p, k_m, grid):
    """E at observers ABOVE the interface from sources BELOW it:
    (N_obs, 3) complex, summed over sources.

    This is the WHOLE field, not a remainder — there is no direct term and
    no image term for the caller to add, and adding one would double-count.
    That difference in contract is why `_field_point`'s crossing regimes
    still refuse after this unit: `reflected_field_at` promises a remainder.

    `src_moment` (S, 3) are complex current moments p = I·dl. `k_p` and
    `k_m` are both arguments because the surfaces are the amplitude of
    `divide_out_transmitted`'s two-leg factor and pairing them with any
    other g is a silent decade.
    """
    _require_transmitted_grid(grid, "transmitted_field_below_to_above")
    rho, zz, zp, r_obs, theta, dhx, dhy = _crossing_geometry(
        obs, src, ground_z, grid, "transmitted_field_below_to_above"
    )
    surf = grid.eval(r_obs, theta, zp)
    g = divide_out_transmitted(k_p, k_m, rho, zz, zp)
    return _combine_transmitted(surf, g, dhx, dhy, src_moment)


def transmitted_field_above_to_below(obs, src, src_moment, ground_z, k_p, k_m, grid):
    """E at observers BELOW the interface from sources ABOVE it:
    (N_obs, 3) complex, summed over sources.

    The reciprocity TRANSPOSE of `transmitted_field_below_to_above` over the
    SAME tables — D_{a→b}(r_b; r_a) = D_{b→a}(r_a; r_b)^T — and not a second
    family. The tables are read at the SAME (R, θ, z′) the upward direction
    would use (the above point supplies R and θ, the below point supplies
    z′), with the dyad transposed and the horizontal direction still taken
    from the below point to the above one.

    Any deviation from the upward direction's numbers is a BUG here, not a
    tolerance: with the verified (7f) both directions are literally the same
    integral.
    """
    _require_transmitted_grid(grid, "transmitted_field_above_to_below")
    rho, zz, zp, r_obs, theta, dhx, dhy = _crossing_geometry(
        src, obs, ground_z, grid, "transmitted_field_above_to_below"
    )
    surf = grid.eval(r_obs, theta, zp)
    g = divide_out_transmitted(k_p, k_m, rho, zz, zp)
    # The geometry table is (above, below) = (source, observer) here, so the
    # pair axes are transposed relative to the returned (N_obs, N_src).
    out = _combine_transmitted_transposed(
        {k: v.T for k, v in surf.items()},
        g.T,
        dhx.T,
        dhy.T,
        src_moment,
    )
    return out
