"""The below/below Sommerfeld remainder: source and observer both IN the
ground (momwire#553 unit 2, the #524 phase-1 regime-2 arc).

`_sommerfeld` implements the ±=+ family — source and observer both ABOVE a
lossy half-space — as the NEC-2 theory manual writes it. This module is its
±=− twin: both points below the interface, in the lossy medium, where the
total field is

    E_total = E_direct(k_m) + A_m·E_image(k_m) + E_remainder,
    A_m = (k_p² − k_m²)/(k_p² + k_m²),

with k_p the free-space wavenumber (real) and k_m = k_p√ε̃ the in-medium one
(Im k_m ≤ 0, so e^{−jk_m R} decays under e^{+jωt}). The remainder is the
same four-surface family the reflected wave is carried by — I_ρ^V, I_z^V,
I_ρ^H, I_φ^H over (R₁, θ) — and it is combined into a field by the SAME
eqs 143–147 dyad algebra. Three things differ, and all three are measured
rather than assumed (momwire#524 phase 0, `scratch/524-phase0/` in the
antennaknobs tree):

**The kernel pair is the swapped `_d12`.** `_d12(λ, k_p, k_m)` — the ±=+
call with its arguments exchanged — IS the below/below D₁/D₂ pair, to 0.0
relative difference over every SPEC soil × frequency and the whole contour
domain. The two regimes are one generalized kernel in a SHARED medium s and
an OTHER medium o; ±=+ reads (s, o) = (p, m) and ±=− reads (m, p). That much
IS a swap, and the below→above family — genuinely a different integral
family through different contours — is not.

**The decay medium's k is complex.** This is why nothing here threads a mode
through `_six_integrals`. Every landmark of the fig-13/fig-14 contours is
built on a REAL decay-medium wavenumber: `k2 = float(k2)`, `kmax`, `kcap`,
each fig-14 waypoint, the steepest-descent tail directions, the `k1_dead`
test. Under the swap the roles invert and not one of them carries over
unaudited, so the below family gets its own contour — the open-literature
recipe (Mosig & Michalski 2021) the #524 phase-0 prototype verified benign
at every SPEC soil and far past them (seawater class): zero tail failures,
worst self-convergence 4.4e-9. The ±=+ path keeps its bytes.

**One z-derivative changes sign.** h = |z + z′| = |z| + |z′| for two points
below, so z + z′ < 0 and ∂/∂z e^{−γ_m|z+z′|} = +γ_m·e, where the ±=+ case
has −γ₂·e. Exactly one of the six integrands (∂²V/∂ρ∂z, the one carrying a
single z-derivative) flips, and with it I_ρ^V and — through
R_z^H = −cosφ·R_ρ^V, which survives the swap — the z row of the dyad. The
R₁ → 0 limits carry the same flip through the static coefficients, which is
the classic hiding spot for a sign error, so `_limits_r1_zero_shared` is
written once for both regimes and GATED at ±=+ against `_limits_r1_zero`.

Health, not hope
----------------
The prototype's contour is only benign because three specific traps are
guarded, all three pre-measured in phase 0 and all three re-guarded here:
ε̃ → 1 makes the D-pair vanish identically and a purely relative tail test
can never go quiet on ulp noise (short-circuit); a Bessel-zero panel lattice
can hand back a zero-length panel that reads as convergence (the +1e-9
nudge); and one Gauss rule must never leap many e-foldings of e^{−λh} (the
1.5/h panel cap). `Health` counts what actually happened so a caller can
tell a converged answer from a lucky one.

C++ twins (momwire#568 unit 2)
------------------------------
Everything in this module is numpy and stays that way — the numpy spellings
are the REFERENCES, and the gates against their C++ twins are tolerances,
never bytes. What has a twin, behind the single `below_fills_568` capability
flag (`_use_below_accel`):

    _six_integrals_below        below_six_integrals_batch
    iv_surfaces_direct_below    its per-point loop, through the same batch —
                                THIS is where the OpenMP region lives
    remainder_field_proj_below  remainder_field_proj_batch_below

`divide_out_below` and the R₁ → 0 limits stay numpy: closed forms costing
less than the call that would dispatch them. So does everything that is a
CONTRACT rather than arithmetic — the ε̃ = 1 short circuit, the (ρ, h) domain
raise, the `Health` bookkeeping and `SommerfeldGridBelow.eval`'s three
refusals — so each of those messages and counters has exactly one home.

Measured on an i7-8550U: one contour node 7–14× the numpy one, a whole
`iv_surfaces_direct_below` batch ~18× with OpenMP across nodes on top of that,
and the slow below/below ladder 392.9 s → 23.0 s at unchanged memory.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from ._accel import acc as _acc
from ._sommerfeld import (
    _GRID_CACHE,
    _GRID_CACHE_MAX,
    _MU0,
    _SOMM_TH_SPLIT_DEG,
    _SURF_KEYS,
    _bessel_j0_j1x,
    _d12,
    _evict_fifo,
    _somm_r1_bucket_wl,
    SommerfeldGrid,
)

# Head-contour Gauss rule and adaptive depth (the phase-0 prototype's fine
# machine; its coarse twin — the self-convergence estimator — is below).
_GAUSS_N = 24
_ADAPT_DEPTH = 16
_GAUSS_N_COARSE = 16
_ADAPT_DEPTH_COARSE = 12

# Detour-height numerator: H = min(0.35a, _DETOUR/ρ) caps the J₀ growth
# e^{|Im λ|ρ} along the head detour at e^2 = 7.39 everywhere.
_DETOUR = 2.0
_DETOUR_COARSE = 1.2

# Tail panel budget. Phase 0 measured 986 panels worst-case over the whole
# SPEC matrix at 12,000; this is the same order of headroom without letting
# a pathological cell run for minutes.
_MAX_TAIL_PANELS = 4000

# Cap on the R₁ a below/below grid will tabulate, in IN-MEDIUM wavelengths
# λ_m = 2π/|k_m|. See `SommerfeldGridBelow` for why this is a product-regime
# number and not the ±=+ family's 15λ.
#
# 2, not 4, and the reason is a MEASUREMENT rather than a preference. The
# grid → direct probe over the full served rectangle (R₁ ∈ [0, cap] × θ ∈
# [1°, 90°], boundaries included, all six SPEC cells) puts a clean cliff at
# exactly 2 λ_m — worst per R₁ band, over every cell and every θ band:
#
#   R₁ ∈ [0.0, 0.2) λ_m    5.8e-5
#   R₁ ∈ [0.2, 1.0) λ_m    2.0e-4
#   R₁ ∈ [1.0, 2.0) λ_m    1.6e-4
#   R₁ ∈ [2.0, 4.0) λ_m    2.4e-3      ← 12x the rest
#
# The far annulus is 3-12x worse than everything inside it and would have
# had to carry a pin 12x looser than the domain deserves. Densifying it
# instead costs ~34 % more nodes for the steep band alone (Δθ 2.5° → 1.25°)
# and ~75 % with the grazing band too, on a fill that is already ~85 s of
# numpy — so the honest trade is the smaller domain, gated at 2e-4, with a
# refusal by name past it rather than a loose pin over it. Densifying the
# far annulus is a recorded follow-up for whoever needs the bigger domain,
# and it wants the C++ twin first.
#
# What 2 λ_m buys at the SPEC soils: 20 m (A/7 MHz), 9.6 m (B/7), 36 m
# (C/7), 7.7 / 5.0 / 12.7 m at 21 MHz. Buried radial screens live inside
# that at HF; anything larger refuses and says so.
_SOMM_BELOW_R1_CAP_LAMBDA_M = 4.0

# The inner/far break, in λ_m. Everything below it is the lattice the grid
# had before momwire#838 part 2, byte for byte; the far annulus above it is
# new, more finely resolved in θ, and filled LAZILY per θ band.
#
# The 2 λ_m cap above was NOT a radial-resolution limit, which is what the
# follow-up note had been carried as. Measured one axis at a time over
# R₁ ∈ [2, 4) λ_m (`scratch/probe838_r1_cap.py`):
#
#   R₁ axis, θ on a node      dr = 0.05 λ_m already reads 1.6e-5 / 4.9e-6 /
#                             1.3e-5 (A/7, B/7, C/21). NOT the lever.
#   θ axis, steep band        Δθ = 2.5° reads 1.5e-3 / 8.3e-4 / 2.0e-3 —
#                             the 2.4e-3 cliff the old cap comment banked.
#   θ axis, grazing band      Δθ = 1.0° also misses, on C/21 (5.9e-4).
#
# So θ is the whole of it, and the old comment already knew: it names
# Δθ 2.5° → 1.25° as the fix and rejects it on COST, not effectiveness.
# ("θ density is not the lever" is part 1's SUB-1° FLOOR finding, where
# `_MAX_TAIL_PANELS` binds — a different question.)
#
# Far-zone lattice, and what it buys against the 2e-4 bar the inner domain
# is gated at (worst over A/7, B/7, C/21):
#
#   steep  60° / 72 cells = 0.83333°    3.6e-5    5.5x
#   graze  29° / 72 cells = 0.40278°    3.6e-5    5.5x
#   band  0.9° /  4 cells = 0.225°      2.7e-5    7.4x   (unchanged)
#
# Δθ is spelled as a CELL COUNT, not a decimal, because part 1's rule is
# that it must divide its band exactly — and neither 0.833 nor 0.4 does
# (60/0.833 = 72.03, 29/0.4 = 72.5). A count cannot get that wrong.
#
# COST, per deck class (soil A at 7 MHz; construction itself is 0.62 s and
# is unchanged by part 2, because none of this is filled at construction):
#
#   nothing past 2 λ_m, nothing under 1°   +0.00 s   pays NOTHING
#   grazing, inner zones only (part 1)     +0.80 s
#   far zone, steep θ only                 +0.34 s
#   far zone AND grazing (BLE's 135 ft)    +0.86 s
#
# That first row is the point of deferring per θ BAND rather than per zone:
# the far annulus is ~4× the panels of the whole rest of the grid and ~68 %
# of that is its own sub-1° band, so a deck that reaches far but not low
# skips the expensive part entirely.
_SOMM_BELOW_R1_NEAR_CAP_LAMBDA_M = 2.0
_SOMM_BELOW_FAR_STEEP_CELLS = 72
_SOMM_BELOW_FAR_GRAZE_CELLS = 72

# Regions the constructor leaves EMPTY, filled on first use. Both are
# expensive and most decks never reach them: the sub-1 deg band in each R₁
# zone (0, 3, 6) and the whole far annulus (6, 7, 8). Panels go as
# 6.4/tan θ, so the far zone's own band alone is ~68 % of its panels from
# ~5 % of its nodes — which is why the far zone is deferred per θ BAND and
# not as a block. Nothing under 2 λ_m with nothing under 1° pays anything.
_DEFERRED_REGIONS = frozenset({0, 3, 6, 7, 8})

# Rows added past each end of a region's R₁ axis so that every IN-DOMAIN
# query gets a centred 4x4 Lagrange stencil instead of the clamped
# end-interval one. `SommerfeldGrid.eval` clips the stencil origin into
# range, which turns a query in the last cell into the [2, 3] leg of the
# cubic; measured on this family that is worth ~40x at the outer edge
# (1.7e-5 against 4.6e-7 for the same point two cells further in). Two rows
# per end is ~2 % of the node count and removes the edge as a special case.
_SOMM_BELOW_PAD_ROWS = 2

# The measured lattice (momwire#553 U2). Every number here came off a dense
# per-point probe against `iv_surfaces_direct_below`, relative to each probe
# point's own scale — the pooled metric hides near-field cells, because the
# surfaces span ~4 decades once the lateral wave takes over.
#
# theta is the whole story. At dr = 0.05 λ_m the steep band [30°, 90°] reads
# 2.4e-1 / 5.2e-1 / 1.1e-1 (soils A/B/C at 7 MHz) with Δθ = 10°, 2.2e-2 /
# 4.7e-2 / 6.7e-3 with 5°, and 2.7e-3 / 4.2e-3 / 6.4e-4 with 2.5° — where the
# ±=+ family's own bar is ~2e-3. Radial spacing barely moves it: 0.1, 0.05
# and 0.025 λ_m all read 2.2e-1 … 2.5e-1 in that band at Δθ = 10°. Once θ is
# resolved the radial term does start to show — the grazing band at Δθ = 1°
# reads 1.4e-3 at dr = 0.1 λ_m against 4.7e-4 at 0.05 — so 0.05 λ_m it is.
_SOMM_BELOW_DR_LAMBDA_M = 0.05
_SOMM_BELOW_DR_NEAR_LAMBDA_M = 0.01
_SOMM_BELOW_R_BREAK_LAMBDA_M = 0.2
_SOMM_BELOW_DTH_GRAZE_DEG = 1.0
_SOMM_BELOW_DTH_STEEP_DEG = 2.5

# The grazing floor, in degrees, below which the grid REFUSES.
#
# θ = 0 is h = 0: the tail loses its e^{−γ_m h} decay entirely and what is
# left is a conditionally convergent integral this contour does not evaluate,
# so there is no honest θ = 0 node to fill. Nor would one help. Measured
# drift of the four surfaces at fixed R₁, against their value at θ = 5°:
#
#   R₁ = 0.2 λ_m   θ 2° → 0.05°   0.07 → 0.17 of scale
#   R₁ = 1.0 λ_m   θ 2° → 0.05°   0.16 → 0.44
#   R₁ = 3.5 λ_m   θ 2° → 0.05°   1.11 → 3.26
#
# — growth roughly linear in log θ, i.e. the lateral wave's logarithm, and
# the cost of chasing it is ~6.4/tan θ tail panels per node (R₁-independent:
# the panel lattice is π/ρ and the tail must reach ~C/h; measured at 0.1°,
# 3542 / 3574 / 3618 panels at R₁/λ_m = 0.2 / 1.0 / 2.0).
#
# An earlier reading of this block said "no uniform lattice resolves that"
# and offered a log-spaced band or an asymptotic lateral-wave branch as the
# follow-up. **That is true of a single GLOBAL Δθ and false of a banded
# one** (momwire#838). This grid is already banded — a grazing band and a
# steep band, split at `_SOMM_TH_SPLIT_DEG` — so the answer was a third
# band, uniform like the other two, over [0.1°, 1°].
#
# Measured (`scratch/probe838_grazing_band.py`): 4-point Lagrange along each
# candidate's own coordinate, scored against `iv_surfaces_direct_below` at a
# COMMON set of 17 off-node queries, worst over soils A/B/C × R₁/λ_m ∈
# {0.2, 1, 2} × the four surfaces. Scoring each lattice at its own cell
# midpoints instead compares different points and flatters the log band.
#
#   nodes   uniform Δθ    worst rel   |   log (geometric)   worst rel
#     4       0.300°       3.1e-07    |                      8.8e-04
#     5       0.225°       9.9e-08    |                      3.6e-04
#     7       0.150°       1.9e-08    |                      1.0e-04
#    10       0.100°       3.9e-09    |                      2.2e-05
#
# Uniform wins by three to four orders at equal node count AND fewer panels
# (log spacing puts its extra nodes at the small-θ end, which is where a
# node is most expensive). At the band's own 4.7e-4 bar the log lattice is
# the one that fails at four nodes.
# Δθ must DIVIDE the band exactly. (1.0 − 0.1)/Δθ non-integral makes the
# last node overshoot `th_band_hi` (0.25 gives nodes 0.1 … 1.1), the two
# bands then share no node, and every old-domain cell at θ = 1° moves —
# measured 1.2e-07 before this was caught. 0.225 = 0.9/4 is the five-node
# rung of the table above.
_SOMM_BELOW_DTH_BAND_DEG = 0.225
_SOMM_BELOW_TH_BAND_HI_DEG = 1.0

# The grazing floor, in degrees, below which the grid REFUSES.
#
# θ = 0 is h = 0: the tail loses its e^{−γ_m h} decay entirely and what is
# left is a conditionally convergent integral this contour does not evaluate,
# so there is no honest θ = 0 node to fill.
#
# What sets the floor is NOT convergence — it is `_MAX_TAIL_PANELS`, and the
# cap is reached silently (the contour stops there, bumps
# `Health.nonconvergent`, and returns the truncated value anyway; filed as
# momwire#841).
#
# Worst tail panels over the SPEC soils A/B/C × 7/21 MHz × R₁/λ_m ∈
# {0, 0.02, 0.05, 0.2, 1, 2} — the sweep matters, a single deck understates
# this badly (soil A at 7 MHz alone reads 3542 at 0.1°, i.e. 90 %, where the
# true worst is 96.7 % on soil C at 21 MHz):
#
#   0.15°   2603 / 4000   65.1 %   all converged
#   0.12°   3237 / 4000   80.9 %   all converged
#   0.10°   3868 / 4000   96.7 %   all converged      <- the floor
#   0.09°   4000 / 4000  100.0 %   25 points CAPPED
#
# **0.1° is the last rung that converges on every SPEC soil**, so that is
# where the floor goes and no lower. It serves every geometry momwire#838
# asks for — BLE 1937's 45 ft tip is 0.64° and its 135 ft tip 0.21°.
#
# The margin is thin (3.3 %) and the failure past it is silent, so
# `test_every_band_node_converges_under_the_panel_cap` asserts both the
# convergence and the headroom on every band node. Below the floor, measured
# against the same point with the cap lifted to 40000, the truncation is
# benign for a while and then is not: 3.8e-09 at 0.08°, 2.6e-06 at 0.05°, and
# 9.3e-04 at 0.023° — the NEC-5 Validation Manual screen's angle, explicitly
# NOT served, and the first rung where the cap costs more than this band's own
# interpolation bar.
_SOMM_BELOW_TH_MIN_DEG = 0.1

_GX, _GW = np.polynomial.legendre.leggauss(_GAUSS_N)
_GXC, _GWC = np.polynomial.legendre.leggauss(_GAUSS_N_COARSE)


# ---------------------------------------------------------------------------
# The C++ twins (momwire#568 unit 2)
# ---------------------------------------------------------------------------
#
# Everything below this line in the module is numpy, unchanged, and stays the
# REFERENCE. `_accelerators` carries a twin of the contour driver (the six
# integrals at a batch of (ρ, h), OpenMP across nodes on U1's shared engine)
# and a twin of the point projection; where a twin exists it is preferred and
# the numpy body is the fallback, following the `_bspline_kernels` pattern.
#
# ITS OWN capability flag, deliberately NOT `contour_engine_568`: a .so built
# at U1 exports the engine's test entry points but none of these symbols, and
# a shared flag would have it claim a contract it cannot serve.
_HAVE_BELOW_FILLS_ACCEL = _acc is not None and bool(
    getattr(_acc, "below_fills_568", False)
)

# The tests' handle on the dispatch — the parity gates have to drive BOTH
# machines inside one process, and a cross-process env var cannot do that.
# `MOMWIRE_BELOW_FORCE_NUMPY` is the same switch for a whole run (a timing
# comparison, a bisect); `monkeypatch.setattr(below, "_FORCE_NUMPY", True)` is
# the same switch for one test. `_use_below_accel` reads both at CALL time, so
# neither is baked in at import.
_FORCE_NUMPY = bool(os.environ.get("MOMWIRE_BELOW_FORCE_NUMPY"))


def _use_below_accel():
    """True when the below/below fills should ride the C++ contour engine."""
    return _HAVE_BELOW_FILLS_ACCEL and not _FORCE_NUMPY


def _six_below_accel(k_p, k_m, rho, h, rtol, selfconv):
    """`below_six_integrals_batch` with this module's machines filled in.

    Both rules, both depths and both detours are arguments rather than
    constants on the C++ side, so the fine machine and the coarse
    self-convergence twin are the same call — exactly as `_run_contour` takes
    them here. The rtol pair is computed HERE, not there: `min(rtol, 1e-11)`
    and its ×100 coarse partner are `_six_integrals_below`'s own spelling and
    belong next to it.
    """
    rtol_fine = min(rtol, 1e-11)
    return _acc.below_six_integrals_batch(
        float(k_p),
        complex(k_m),
        np.ascontiguousarray(rho, dtype=float),
        np.ascontiguousarray(h, dtype=float),
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
        _MAX_TAIL_PANELS,
    )


# ---------------------------------------------------------------------------
# Quadrature health
# ---------------------------------------------------------------------------


@dataclass
class Health:
    """What the contour actually did, tallied across calls.

    A below/below answer is only as good as its tail: `nonconvergent` > 0
    means a panel budget ran out rather than the integrand going quiet, and
    `worst_selfconv` is the fine-vs-coarse relative spread wherever a caller
    asked for one. Phase 0's whole gate suite reported 0 / 0 / 4.4e-9 on
    these contours; anything else is news.
    """

    evaluations: int = 0
    nonconvergent: int = 0
    accelerated: int = 0
    max_tail_panels: int = 0
    max_head_panels: int = 0
    worst_selfconv: float = 0.0
    worst_selfconv_at: tuple | None = None
    zero_kernel: int = 0
    _rows: list = field(default_factory=list, repr=False)

    def note(self, where, panels, head_panels, converged, accelerated):
        self.evaluations += 1
        if not converged:
            self.nonconvergent += 1
        if accelerated:
            self.accelerated += 1
        self.max_tail_panels = max(self.max_tail_panels, panels)
        self.max_head_panels = max(self.max_head_panels, head_panels)

    def note_selfconv(self, where, rel):
        if rel > self.worst_selfconv:
            self.worst_selfconv = rel
            self.worst_selfconv_at = where

    def as_dict(self):
        return {
            "evaluations": self.evaluations,
            "nonconvergent": self.nonconvergent,
            "accelerated": self.accelerated,
            "max_tail_panels": self.max_tail_panels,
            "max_head_panels": self.max_head_panels,
            "worst_selfconv": self.worst_selfconv,
            "worst_selfconv_at": self.worst_selfconv_at,
            "zero_kernel": self.zero_kernel,
        }


# ---------------------------------------------------------------------------
# Medium
# ---------------------------------------------------------------------------


def k_medium(eps_t, k2):
    """k_m = k₂√ε̃ on the Im ≤ 0 branch — the derivation of `_six_integrals`
    :356-359 verbatim, so the two regimes cannot drift apart on which root
    of ε̃ they mean."""
    k2 = float(k2)
    km = k2 * np.sqrt(complex(eps_t))
    if km.imag > 0:
        km = np.conj(km)
    return complex(km)


def lambda_medium(eps_t, k2):
    """λ_m = 2π/|k_m|, the in-medium wavelength `k_medium` propagates at.

    Equivalently λ_m = λ₀/|n|, with the refractive-index magnitude
    |n| = |√ε̃| = [ε_r² + 3.23e8·(σ/f_MHz)²]^(1/4) — ε̃ = ε_r − jσ/(ωε₀),
    f_MHz = f/1e6, the constant folding in ω = 2π·f_MHz·1e6 and ε₀. This
    docstring is the ONE place that spelling lives; `k_medium` derives |n|
    on its Im ≤ 0 branch and this is just its reciprocal-wavelength
    reading, so nothing here re-derives ε̃ or re-picks a branch.

    Why it matters, verbatim from #553 U1's G-U1-7 (`_bspline_kernels.py`):
    GL quadrature cost is set by |k|·h, not by the medium as such — but a
    deck meshed against λ₀ instead of λ_m runs its buried segments at
    |k_m|h = |n|·|k₀|h, |n|× the panel phase the mesher meant to deliver.
    That is a MESHING defect, not a quadrature one — `k_medium`/`_bspline_
    kernels` never widen n_qp for it, and `segments_per_quarter_wavelength`
    below is where the honesty about it lives instead.
    """
    return 2.0 * np.pi / abs(k_medium(eps_t, k2))


def segments_per_quarter_wavelength(seg_lengths, eps_t, k2):
    """How many segments of each given length would tile a quarter
    in-medium wavelength — `auto_mesh`'s own "N per quarter-wave" density
    (antennaknobs `builder.py`), read against λ_m instead of λ₀.

    Advisory, and ONLY advisory: this never raises. NEC-5 itself serves
    under-meshed buried decks — the seam rule forbids refusing what the
    host serves — so under-meshing is not this module's call to block; a
    caller (U5's serve path, the app's `auto_mesh`, a test) reads the
    number back and decides what to do with it, and the arc's envelope
    discipline (tolerances vs ladder limits) is where coarse-mesh honesty
    actually lives. `seg_lengths` is an array-like of segment lengths in
    metres; the return is the same shape, dimensionless, and NaN/inf on a
    zero-length input rather than a raise.

    A wire meshed at N segments per quarter FREE-SPACE wavelength
    (h = λ₀/(4N)) reads here as N/|n| segments per quarter λ_m — under-
    resolved by exactly the refractive index the free-space mesher never
    knew about. At ε̃ → 1, |n| = 1 and this is the free-space reading
    exactly (`_sommerfeld_below`'s ε̃ → 1 short circuit, `test_gu2_6_eps_
    one_is_exactly_zero`, applies here too: no soil, no correction).
    """
    seg_lengths = np.asarray(seg_lengths, dtype=float)
    lam_m = lambda_medium(eps_t, k2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.25 * lam_m / seg_lengths


def _c1_moment(omega, mu):
    """C₁ = −jωμ/4π for a unit current moment Iℓ = 1.

    `_sommerfeld._c1` is this divided by k₂² — the eq-123 spelling, which
    folds the ±=+ family's k₂² into the normalization. The below family's
    prefactors carry k_p²/k_m² rather than a bare k₂², so it is clearer to
    name the moment constant itself and write the ratio out.
    """
    return -1j * float(omega) * float(mu) / (4.0 * np.pi)


# ---------------------------------------------------------------------------
# The integrand stack
# ---------------------------------------------------------------------------


def _integrand_six_below(lam, rho, h, k_p, k_m):
    """The six λ-integrands of the below/below remainder, stacked (6, n) in
    `_integrand_six`'s own order so the two families' surface assemblies
    read the same:

      0: ∂²V/∂ρ²      [D₂ e^{−γ_m h} λ³ (J₁/x − J₀)]
      1: ∂²V/∂z²      [D₂ γ_m² e^{−γ_m h} J₀ λ]
      2: ∂²V/∂ρ∂z     [−D₂ γ_m e^{−γ_m h} J₁ λ²]      ← the ±=− sign
      3: (1/ρ)∂V/∂ρ   [−D₂ e^{−γ_m h} (J₁/x) λ³]
      4: V            [D₂ e^{−γ_m h} J₀ λ]
      5: U            [D₁ e^{−γ_m h} J₀ λ]

    (D₁, D₂, γ_m) = `_d12(λ, k_p, k_m)` — the SWAPPED call. `_d12(λ, k1, k2)`
    builds its kernels around γ₂ = γ(k2) as the decay γ and k1 as the other
    medium, so passing (k_p, k_m) puts the ground in the decay slot: exactly
    the below/below pair, measured at 0.0 relative difference against the
    phase-0 prototype's independent generalized form.

    Index 2 is the whole of the ±=− sign story. h = |z + z′| and z + z′ < 0
    below, so ∂/∂z e^{−γ_m|z+z′|} = +γ_m·e where the ±=+ case gets −γ₂·e;
    with ∂J₀/∂ρ = −λJ₁ the product lands at −D₂γ_m J₁λ², the negative of
    `_integrand_six`'s index 2. The Bessel form only — there is no Hankel
    twin here, because there is no fig-14 contour here.
    """
    lam = np.asarray(lam, dtype=np.complex128)
    d1, d2, g_m = _d12(lam, k_p, k_m)
    e = np.exp(-g_m * h)
    x = lam * rho
    b0, b1x = _bessel_j0_j1x(x)
    l2 = lam * lam
    l3 = l2 * lam
    common = d2 * e
    return np.stack(
        [
            common * (b1x - b0) * l3,
            common * g_m * g_m * b0 * lam,
            -common * g_m * (b1x * x) * l2,
            -common * b1x * l3,
            common * b0 * lam,
            d1 * e * b0 * lam,
        ]
    )


# ---------------------------------------------------------------------------
# The contour: first-quadrant half-sine head + Bessel-zero tail
# ---------------------------------------------------------------------------


def _gauss_segment(f, z0, z1, gx, gw):
    mid = 0.5 * (z0 + z1)
    half = 0.5 * (z1 - z0)
    return f(mid + half * gx) @ (gw * half)


def _adaptive_segment(f, z0, z1, rtol, depth, gx, gw, whole=None, ref=0.0):
    """Recursive bisection Gauss on z0→z1, vector-valued.

    `ref` floors the relative test: on a high-loss ground the remainder can
    sit decades below a neighbouring section's magnitude, and a purely
    local relative target then chases noise. Passing the head's own scale
    in as `ref` ties the target to the answer's scale instead.
    """
    if whole is None:
        whole = _gauss_segment(f, z0, z1, gx, gw)
    mid = 0.5 * (z0 + z1)
    left = _gauss_segment(f, z0, mid, gx, gw)
    right = _gauss_segment(f, mid, z1, gx, gw)
    better = left + right
    err = float(np.max(np.abs(better - whole)))
    scale = max(float(np.max(np.abs(better))), ref)
    if depth <= 0 or err <= rtol * max(scale, 1e-300):
        return better
    return _adaptive_segment(
        f, z0, mid, rtol, depth - 1, gx, gw, left, ref
    ) + _adaptive_segment(f, mid, z1, rtol, depth - 1, gx, gw, right, ref)


def _wynn_epsilon(seq):
    """Wynn's epsilon over a sequence of complex vectors; the best even
    column, elementwise. Only reachable when the tail has NOT gone quiet
    inside its panel budget — phase 0 never needed it on a SPEC geometry
    (every one has |z| + |z′| ≥ 0.12 m, so e^{−λh} decays honestly), and it
    is kept for the h → 0 limit, where it is the difference between an
    answer and a truncation.
    """
    s = np.asarray(seq)
    n = s.shape[0]
    if n < 5:
        return s[-1]
    prev = np.zeros_like(s)
    cur = s.copy()
    best = s[-1].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(1, n):
            d = cur[1:] - cur[:-1]
            d = np.where(np.abs(d) < 1e-300, 1e-300, d)
            nxt = prev[:-1] + 1.0 / d
            prev, cur = cur[:-1], nxt
            if k % 2 == 0 and cur.shape[0] >= 1:
                cand = cur[-1]
                if np.all(np.isfinite(cand)):
                    best = cand
            if cur.shape[0] <= 1:
                break
    return best


def _head(f, a, rho, marks, rtol, depth, detour, gx, gw):
    """∫ over [0, a] deformed into the FIRST quadrant.

    λ(t) = t + jH·sin(πt/a) with a = 1.1·max(k_p, |k_m|), past BOTH branch
    points. e^{+jωt} puts both cuts running downward from +k_p / +k_m, so
    an upward detour crosses neither and encloses no Zenneck pole — there
    is no pole extraction anywhere in this module, and phase 0 measured why
    it is not needed (neither Sommerfeld denominator comes within 1.05 /
    0.063 of zero anywhere on the path, at any SPEC soil).

    H shrinks as ρ grows because J₀(λρ) grows like e^{|Im λ|ρ} off the real
    axis; the cap holds that growth at e². Sub-interval ends are seeded at
    sevenths of [0, a] and around each branch point's real part, so the
    adaptive bisection starts with the fast structure already bracketed.
    """
    H = min(0.35 * a, detour / max(rho, 1e-12))
    H = max(H, 1e-6 * a)

    def ft(t):
        t = np.asarray(t, dtype=float)
        lam = t + 1j * H * np.sin(np.pi * t / a)
        dl = 1.0 + 1j * H * (np.pi / a) * np.cos(np.pi * t / a)
        return f(lam) * dl

    edges = {0.0, a}
    for i in range(1, 7):
        edges.add(a * i / 7.0)
    for mk in marks:
        for w in (0.0, -0.15, 0.15, -0.4, 0.4):
            v = mk * (1.0 + w)
            if 0.0 < v < a:
                edges.add(v)
    e = sorted(edges)
    total = None
    for i in range(len(e) - 1):
        part = _adaptive_segment(ft, e[i], e[i + 1], rtol, depth, gx, gw)
        total = part if total is None else total + part
    return total, len(e) - 1


def _tail_below(f, a, rho, h, ref, rtol, gx, gw, max_panels=_MAX_TAIL_PANELS):
    """∫ from a to ∞ along the real axis, panelled at the J₀(λρ) zeros.

    Three guards, each one a measured phase-0 trap:

    * **Zero-length panels.** `floor((z − off)/lat) + 1` can round a z that
      sits exactly on the lattice back onto itself, and two consecutive
      zero-length panels contribute exactly 0.0 — which the quiet counter
      reads as convergence. The measured symptom was a tail truncating at
      λ ≈ 1.4 where the answer needed λ ≈ 40, reported as CONVERGED. The
      `+1e-9` inside the floor is the fix.
    * **e-folding cap.** A single Gauss rule spanning 14 e-foldings of
      e^{−λh} was wrong by 8e-4; panels are capped at 1.5/h as well as at
      the zero spacing.
    * **Noise-vs-noise.** `contrib_max == 0.0` is tested explicitly, and
      the relative test is floored by `ref` (the head's scale), so a
      remainder sitting decades below the divided-out factor still
      terminates on an absolute bar rather than chasing its own ulps.
    """
    dl_decay = (1.5 / h) if h > 0 else np.inf
    if not np.isfinite(dl_decay):
        dl_decay = max(1.0, 10.0 * a)
    lat = (np.pi / rho) if rho > 0 else np.inf
    off = (-0.25 * np.pi / rho) if rho > 0 else 0.0

    def next_zero(zz):
        if not np.isfinite(lat):
            return np.inf
        n = math.floor((zz - off) / lat + 1e-9) + 1
        return off + n * lat

    z = a
    sums = []
    acc = None
    parts = 0
    quiet = 0
    converged = False
    ref_scale = float(np.max(np.abs(ref))) if ref is not None else 0.0
    while parts < max_panels:
        znext = min(next_zero(z), z + dl_decay)
        contrib = _adaptive_segment(f, z, znext, rtol, 6, gx, gw, ref=ref_scale)
        acc = contrib if acc is None else acc + contrib
        sums.append(acc.copy())
        z = znext
        parts += 1
        cmax = float(np.max(np.abs(contrib)))
        smax = max(float(np.max(np.abs(acc))), ref_scale, 1e-300)
        if cmax == 0.0 or cmax < rtol * smax:
            quiet += 1
            if quiet >= 2:
                converged = True
                break
        else:
            quiet = 0
    accel = False
    if not converged and len(sums) >= 8:
        acc = _wynn_epsilon(np.array(sums[-min(len(sums), 41) :]))
        accel = True
    return acc, parts, converged, accel


def _run_contour(
    f, k_p, k_m, rho, h, rtol, depth, detour, gx, gw, max_panels=_MAX_TAIL_PANELS
):
    """Head + tail for ONE contour. `h` is the TAIL'S DECAY LENGTH, not a
    coordinate: the below/below integrand decays as e^{−γ_m h} with
    h = |z| + |z′|, and `_sommerfeld_transmitted` reuses this machinery with
    h = |z′| + z because e^{−γ_m|z′|−γ_p z} → e^{−λ(|z′|+z)} at large λ just
    as surely. `max_panels` is the transmitted family's other handle: its
    grazing rows need a bigger budget than any below/below geometry does,
    and the default keeps this family's numbers bit-for-bit unchanged.
    """
    a = 1.1 * max(abs(k_p), abs(k_m))
    marks = (k_p, abs(k_m.real))
    head, hp = _head(f, a, rho, marks, rtol, depth, detour, gx, gw)
    tail, tp, conv, accel = _tail_below(
        f, a, rho, h, head, rtol, gx, gw, max_panels=max_panels
    )
    return head + tail, hp, tp, conv, accel


def _six_integrals_below(
    eps_t, k2, rho, h, rtol=1e-11, health=None, selfconv=False, where=None
):
    """The six below/below λ-integrals at one (ρ, h); returns (6,) complex.

    `h = |z| + |z′|` (both points below), ρ the horizontal separation.
    `selfconv` re-runs the whole contour on a coarser machine (Gauss-16,
    rtol ×100, a shallower detour) and records the componentwise relative
    spread in `health` — the estimate phase 0 reports as an upper bound
    roughly five decades above the realized error, never as the error.
    """
    eps_t = complex(eps_t)
    rho = float(rho)
    h = float(h)
    if rho < 0 or h < 0 or (rho == 0.0 and h == 0.0):
        raise ValueError(f"need rho, h >= 0 and R1 > 0, got {(rho, h)!r}")
    if eps_t == 1.0:
        # ε̃ = 1: k_m = k_p, D₁ = D₂ = 0 identically. The two subtracted
        # static terms are grouped differently and leave ulp noise, on
        # which a relative tail test is noise-vs-noise and never goes
        # quiet — phase 0 measured 1164 s instead of 0.4 s for the same
        # (correct) answer. Short-circuit, exactly as `_six_integrals`
        # does at its own :351.
        if health is not None:
            health.zero_kernel += 1
        return np.zeros(6, dtype=np.complex128)

    k_p = float(k2)
    k_m = k_medium(eps_t, k_p)

    if _use_below_accel():
        vals, tp, hp, conv, accel, sconv = _six_below_accel(
            k_p, k_m, np.array([rho]), np.array([h]), rtol, selfconv
        )
        if health is not None:
            health.note(where, int(tp[0]), int(hp[0]), bool(conv[0]), bool(accel[0]))
            if selfconv:
                health.note_selfconv(
                    where if where is not None else (rho, h), float(sconv[0])
                )
        return vals[0]

    def f(lam):
        return _integrand_six_below(lam, rho, h, k_p, k_m)

    fine, hp, tp, conv, accel = _run_contour(
        f, k_p, k_m, rho, h, min(rtol, 1e-11), _ADAPT_DEPTH, _DETOUR, _GX, _GW
    )
    if health is not None:
        health.note(where, tp, hp, conv, accel)
    if selfconv:
        coarse, _, _, _, _ = _run_contour(
            f,
            k_p,
            k_m,
            rho,
            h,
            min(rtol, 1e-11) * 100.0,
            _ADAPT_DEPTH_COARSE,
            _DETOUR_COARSE,
            _GXC,
            _GWC,
        )
        scale = np.maximum(np.abs(fine), 1e-300)
        rel = float(np.max(np.abs(fine - coarse) / scale))
        if health is not None:
            health.note_selfconv(where if where is not None else (rho, h), rel)
    return fine


def _six_integrals_below_many(
    eps_t, k2, rho, h, rtol=1e-11, health=None, selfconv=False, where=None
):
    """`_six_integrals_below` over parallel (ρ, h) arrays; returns (n, 6).

    The grid fill's hot loop, and the only place in this module where the
    parallelism is worth taking: one node's contour is completely independent
    of every other, and a fill is thousands of them. With the accelerator it
    is one C++ call — OpenMP across nodes, the GIL released, U1's engine
    (allocation-free, no mutable static) on each thread. Without it, it is
    EXACTLY the per-point Python loop it replaces, calling the same
    `_six_integrals_below`, in the same order.

    Everything that is not quadrature stays on this side: the ε̃ = 1 short
    circuit (exact zeros, one `zero_kernel` bump per node, as the per-point
    spelling gives), the (ρ, h) domain raise with its own words, and the
    `Health` bookkeeping, which is applied in node order so `worst_selfconv_at`
    lands on the same node either way.
    """
    eps_t = complex(eps_t)
    rho = np.ascontiguousarray(rho, dtype=float).ravel()
    h = np.ascontiguousarray(h, dtype=float).ravel()
    n = rho.size
    if where is None:
        where = [None] * n

    if not _use_below_accel():
        out = np.empty((n, 6), dtype=np.complex128)
        for i in range(n):
            out[i] = _six_integrals_below(
                eps_t,
                k2,
                rho[i],
                h[i],
                rtol,
                health=health,
                selfconv=selfconv,
                where=where[i],
            )
        return out

    for i in range(n):
        r, hh = float(rho[i]), float(h[i])
        if r < 0 or hh < 0 or (r == 0.0 and hh == 0.0):
            raise ValueError(f"need rho, h >= 0 and R1 > 0, got {(r, hh)!r}")
    if eps_t == 1.0:
        if health is not None:
            health.zero_kernel += n
        return np.zeros((n, 6), dtype=np.complex128)

    k_p = float(k2)
    k_m = k_medium(eps_t, k_p)
    vals, tp, hp, conv, accel, sconv = _six_below_accel(
        k_p, k_m, rho, h, rtol, selfconv
    )
    if health is not None:
        for i in range(n):
            health.note(where[i], int(tp[i]), int(hp[i]), bool(conv[i]), bool(accel[i]))
            if selfconv:
                w = where[i] if where[i] is not None else (float(rho[i]), float(h[i]))
                health.note_selfconv(w, float(sconv[i]))
    return vals


# ---------------------------------------------------------------------------
# The R₁ → 0 limits
# ---------------------------------------------------------------------------


def _limits_r1_zero_shared(k_s, k_o, sign_rz, theta, omega, mu):
    """The analytic R₁ → 0 surface limits for EITHER regime, written once.

    Both families' limits come from the same place: the λ → ∞ static tails
    of the kernels, γ_s → λ − k_s²/2λ, which give

        D₁ → c₂/λ,        c₂ = (k_o² − k_s²)/(k_o² + k_s²)
        D₂ → c₃/λ³,       c₃ = k_s²(k_o² − k_s²)/(k_o² + k_s²)²

    and three elementary Hankel integrals (∫e^{−λh}J₀ dλ = 1/R₁,
    ∫e^{−λh}J₁ dλ = (1 − h/R₁)/ρ, ∫e^{−λh}J₁/λ dλ = (R₁ − h)/ρ). `sign_rz`
    is the ∂/∂z sign of the regime — +1 above (h = z + z′ > 0), −1 below
    (h = |z + z′|, z + z′ < 0) — and it multiplies exactly the one surface
    that carries a single z-derivative.

    Reading (s, o) = (k₂, k₁) with `sign_rz = +1` reproduces
    `_sommerfeld._limits_r1_zero` — eqs 169–172 — and that is a GATE, not a
    remark: the below regime's groupings differ from the shipped ones only
    by which medium is which, which is precisely where a sign error would
    hide unnoticed.
    """
    theta = np.asarray(theta, dtype=float)
    ks2 = k_s * k_s
    ko2 = k_o * k_o
    cm = _c1_moment(omega, mu)
    c2 = (ko2 - ks2) / (ko2 + ks2)
    c3 = ks2 * (ko2 - ks2) / (ko2 + ks2) ** 2
    ratio = ko2 / ks2  # the (5a)/(5b) prefactor k_o²/k_s²
    s = np.sin(theta)
    co = np.cos(theta)
    # (1 − sinθ)/cosθ and (1 − sinθ)/cos²θ, θ → π/2 limits 0 and 1/2.
    near = np.abs(co) < 1e-8
    co_safe = np.where(near, 1.0, co)
    q1 = np.where(near, 0.0, (1.0 - s) / co_safe)
    q2 = np.where(near, 0.5, (1.0 - s) / (co_safe * co_safe))
    return {
        "IrhoV": sign_rz * cm * ratio * c3 * q1,
        "IzV": np.full_like(q1, cm * ratio * c3, dtype=np.complex128),
        "IrhoH": cm * (c2 - c3 + c3 * q2),
        "IphiH": -cm * (c2 - c3 * q2),
    }


def _limits_r1_zero_below(eps_t, k2, theta, omega, mu):
    """The below/below R₁ → 0 limits: `_limits_r1_zero_shared` at
    (s, o) = (k_m, k_p) with the ±=− z-derivative sign."""
    k_p = float(k2)
    k_m = k_medium(eps_t, k_p)
    return _limits_r1_zero_shared(k_m, k_p, -1.0, theta, omega, mu)


# ---------------------------------------------------------------------------
# The four surfaces
# ---------------------------------------------------------------------------


def divide_out_below(k_p, k_m, rho, hh):
    """g = e^{−j(k_p ρ + k_m h)}/R₁ — the factor the below surfaces are the
    amplitude of, so `surface × g` is the field. R₁ = √(ρ² + h²).

    MEASURED choice (momwire#553 U2), and NOT the obvious one. Three
    candidates were scored by per-point relative interpolation error
    against direct evaluation on the shipped lattice, across the SPEC
    soils:

    * `g_m = e^{−jk_m R₁}/R₁` — what the direct and image terms carry;
    * `g_p = e^{−jk_p R₁}/R₁` — the ±=+ family's free-space factor;
    * this one, the two-leg blend.

    g_m and g_p are indistinguishable (they differ by under 5 % on the
    median and not at all on the worst): a factor that depends on R₁ alone
    is smooth along the lattice's radial axis either way, so neither buys
    anything. The blend is a different animal because it depends on θ, and
    θ is where this family's interpolation error actually lives. Measured
    worst, grazing band [1°, 30°] at dr = 0.05 λ_m:

        Δθ = 5°   g_m 2.09e-1 (soil A) / 2.83e-1 (B)
                  blend 1.06e-2 / 2.68e-3          — 20x / 105x
        Δθ = 1°   g_m 4.74e-4 / 6.08e-4
                  blend 3.37e-5 / 4.44e-5          — 14x / 14x

    and steep band [30°, 90°], 3x to 3.5x at every Δθ tried.

    The physics is why: below the interface the near remainder is the
    image wave, travelling R₁ in the medium at k_m, while the far
    remainder is the LATERAL wave, which travels the horizontal leg ρ in
    the AIR at the real k_p and only the vertical legs h in the lossy
    medium. Neither pure factor can be right at both ends. This one
    degenerates to g_m at θ = 90° (ρ = 0) and to g_p at θ = 0, and it
    tracks the true carrier in between — which is exactly the θ-dependence
    the lattice was otherwise being asked to interpolate.

    Both consumers must pair with THIS factor. `remainder_field_proj_below`
    and `remainder_field_below` take k_p and k_m for that reason, and
    handing either the wrong g is a silent decade rather than an error —
    hence the gate that drives both spellings off one grid.
    """
    rho = np.asarray(rho, dtype=float)
    hh = np.asarray(hh, dtype=float)
    r1 = np.sqrt(rho * rho + hh * hh)
    return np.exp(-1j * (k_p * rho + k_m * hh)) / r1


def iv_surfaces_direct_below(
    eps_t,
    k2,
    R1,
    theta,
    rtol=1e-9,
    omega=None,
    mu=_MU0,
    health=None,
    selfconv=False,
):
    """Direct (no-grid) evaluation of the four below/below surfaces at
    (R₁, θ). Returns a dict of complex arrays shaped like R₁.

    `k2` is the FREE-SPACE wavenumber (real), as everywhere in this tree;
    k_m is derived inside. R₁ = √(ρ² + h²) with h = |z| + |z′| and
    θ = atan2(h, ρ) ∈ [0, π/2]; ω defaults to k₂·c.

    The surfaces are the (5a)–(5d) components of the remainder divided by
    `divide_out_below`, i.e. the below twin of `iv_surfaces_direct`'s
    `phase = R₁e^{jk₂R₁}` normalization. Writing P = 1/g = R₁·e^{j(k_pρ +
    k_m h)} for that reciprocal:

        I_ρ^V = C₁·(k_p²/k_m²)·P·∂²V/∂ρ∂z
        I_z^V = C₁·(k_p²/k_m²)·P·(∂²V/∂z² + k_m²V)
        I_ρ^H = C₁·P·(∂²V/∂ρ² + U)
        I_φ^H = −C₁·P·((1/ρ)∂V/∂ρ + U)

    with C₁ = −jωμ/4π. The prefactor pattern is the SAME generalized one
    the ±=+ assembly uses (a k_o²/k_s² on the V-source rows, none on the H
    rows); only (s, o) exchange. Verified against the phase-0 prototype's
    independent generalized evaluator, which in turn reproduces momwire's
    own four surfaces at ±=+ to 1.4e-10.

    O(ms) per point. This is the grid's fill function and the tests'
    oracle hook, never a per-pair call.
    """
    k_p = float(k2)
    if omega is None:
        from ._sommerfeld import _C_LIGHT

        omega = k_p * _C_LIGHT
    eps_t = complex(eps_t)
    k_m = k_medium(eps_t, k_p)

    R1 = np.asarray(R1, dtype=float)
    theta = np.asarray(theta, dtype=float)
    R1b, thb = np.broadcast_arrays(R1, theta)
    out_shape = R1b.shape
    R1f = R1b.ravel()
    thf = thb.ravel()

    out = {kk: np.zeros(R1f.shape, dtype=np.complex128) for kk in _SURF_KEYS}

    zero = R1f == 0.0
    if np.any(zero):
        lim = _limits_r1_zero_below(eps_t, k_p, thf[zero], omega, mu)
        for kk in _SURF_KEYS:
            out[kk][zero] = lim[kk]

    nz = np.nonzero(~zero)[0]
    if nz.size:
        r1 = R1f[nz]
        rho = np.maximum(r1 * np.cos(thf[nz]), 0.0)
        h = np.maximum(r1 * np.sin(thf[nz]), 0.0)
        # The R₁ > 0 rows, and ONLY those: the R₁ = 0 row above is the
        # analytic `_limits_r1_zero_below` closed form, which is cheaper than
        # the call that would dispatch it.
        thnz = thf[nz]
        six = _six_integrals_below_many(
            eps_t,
            k_p,
            rho,
            h,
            rtol,
            health=health,
            selfconv=selfconv,
            where=[(float(r1[i]), float(thnz[i])) for i in range(nz.size)],
        )
        v_rr, v_zz, v_rz, v_r1, v, u = six.T
        cm = _c1_moment(omega, mu)
        km2 = k_m * k_m
        kp2 = k_p * k_p
        ratio = kp2 / km2
        phase = 1.0 / divide_out_below(k_p, k_m, rho, h)
        out["IrhoV"][nz] = cm * phase * ratio * v_rz
        out["IzV"][nz] = cm * phase * ratio * (v_zz + km2 * v)
        out["IrhoH"][nz] = cm * phase * (v_rr + u)
        out["IphiH"][nz] = -cm * phase * (v_r1 + u)

    return {kk: out[kk].reshape(out_shape) for kk in _SURF_KEYS}


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


class SommerfeldGridBelow(SommerfeldGrid):
    """The below/below tabulation: `SommerfeldGrid`'s lattice, re-keyed.

    Three decisions separate this from its ±=+ parent, and each one is a
    MEASUREMENT with a number behind it (momwire#553 U2):

    **The length unit is λ_m = 2π/|k_m|, not λ_p = 2π/k₂.** Below the
    interface every wave in the problem propagates in the ground: the
    direct and image terms oscillate at k_m and the remainder's saddle
    sits there too. Keying the lattice to λ_p under-resolves the surfaces
    by exactly |k_m|/k₂ — a factor of 8.9 on the SPEC's worst soil — and
    the measured interpolation error says so. λ_p enters only through the
    lateral wave, whose beat length 2π/|k_m − k₂| is capped against as in
    the parent (and measured NOT to bind once the lattice is λ_m-keyed).

    **`r1_max` is capped at a few λ_m and past it this grid REFUSES.** The
    parent's 15λ cap rests on the remainder being a negligible ~1/R₁ tail
    past ~3–4λ, so freezing the surface amplitude at the cap is an accurate
    stand-in (`SommerfeldGrid.eval` clamps r1 silently, matching the C++
    `proj_one`). Below the interface that argument is simply FALSE:
    phase 0 measured |E_remainder|/|E_direct| RISING with range, to 2.3× at
    ρ = 10 m in soil A. A frozen amplitude there would be a fabrication, so
    `eval` raises by name instead of clamping. The cap is sized for the
    product regime this unit serves — buried radials and ground screens,
    a few λ_m across.

    **No boundary-layer keying and no `_r0_fill`.** The parent's finest
    scale is 1/|k₁| = 1/|k_m|, a layer far thinner than λ_p (2e-3 λ on sea
    water) that the R₁ = 0 node would otherwise poison. Below, the shared
    medium IS the ground, so that same 1/|k_m| is λ_m/2π ≈ 0.16 λ_m — the
    first cell resolves it ~16 times over. There is nothing to key down to
    and nothing to poison; measured continuity across the first cells says
    the same.

    No normalized-master reuse either: `scaled_to` refuses. The parent's
    frequency-axis reuse rests on S = ω·μ·G(ε̃; R₁/λ, θ) with a
    λ-proportional lattice, and neither the scaling law nor the lattice
    proportionality was re-measured for the below family in this unit.
    """

    regime = "below"

    def __init__(self, eps_t, k2, r1_max, rtol=1e-9, omega=None, mu=_MU0, health=None):
        self.eps_t = complex(eps_t)
        self.k2 = float(k2)
        from ._sommerfeld import _C_LIGHT

        self.omega = self.k2 * _C_LIGHT if omega is None else float(omega)
        self.mu = float(mu)
        self.k_m = k_medium(self.eps_t, self.k2)
        lam_m = 2.0 * np.pi / abs(self.k_m)
        self.lam_m = lam_m
        cap = _SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m
        self.r1_max = min(max(float(r1_max), 0.35 * lam_m), cap)
        self.r1_cap = cap

        # The lateral-wave beat, capped against exactly as the parent does.
        beat = 2.0 * np.pi / max(abs(self.k_m - self.k2), 1e-30)

        self.r_break = _SOMM_BELOW_R_BREAK_LAMBDA_M * lam_m
        # Three R₁ zones since momwire#838 part 2: inner, near, and the far
        # annulus [r_near, r1_max]. `r_near` is the old cap, so everything
        # under it is the lattice this grid always had.
        self.r_near = min(_SOMM_BELOW_R1_NEAR_CAP_LAMBDA_M * lam_m, self.r1_max)
        self.th_min = math.radians(_SOMM_BELOW_TH_MIN_DEG)
        split = _SOMM_TH_SPLIT_DEG
        dr_in = _SOMM_BELOW_DR_NEAR_LAMBDA_M * lam_m
        dr_out = min(_SOMM_BELOW_DR_LAMBDA_M * lam_m, beat / 6.0)
        self.beat_binds = beat / 6.0 < _SOMM_BELOW_DR_LAMBDA_M * lam_m
        th0 = _SOMM_BELOW_TH_MIN_DEG
        band_hi = _SOMM_BELOW_TH_BAND_HI_DEG
        dthb = _SOMM_BELOW_DTH_BAND_DEG
        dthg = _SOMM_BELOW_DTH_GRAZE_DEG
        dths = _SOMM_BELOW_DTH_STEEP_DEG
        self.th_band_hi = math.radians(band_hi)
        self._th_split = math.radians(split)
        # THREE θ bands per R₁ zone since momwire#838, not two. The sub-1°
        # band is a SEPARATE region rather than a finer `dthg` over the whole
        # grazing band, and that is forced rather than chosen: refining the
        # existing band's Δθ moves its node set (0.1 + k·0.25 never lands on
        # 2°), so every interpolated cell in the old domain would move. A
        # separate region leaves the old two byte for byte and meets them at
        # a shared node — the same seam the grid already has at `split`.
        #
        # Region order is (R₁ zone) × (θ band), θ fastest: 0-2 inner, 3-5
        # outer. `eval` below and `proj_one_below` in `_accel_mw568.cpp`
        # both index it that way; they are two copies of one layout, which
        # is why the C++ side takes `th_band_hi` explicitly instead of
        # inferring a band count from `len(reg_vals)`.
        # Far-annulus Δθ, from cell counts so it divides its band exactly.
        dthg_far = (split - band_hi) / _SOMM_BELOW_FAR_GRAZE_CELLS
        dths_far = (90.0 - split) / _SOMM_BELOW_FAR_STEEP_CELLS
        # Region order is (R₁ zone) × (θ band), θ fastest: 0-2 inner,
        # 3-5 near, 6-8 far. `_interp` here and `proj_one_below` in
        # `_accel_mw568.cpp` are two copies of this one layout, which is why
        # the C++ side takes `th_band_hi` explicitly rather than inferring a
        # band count from `len(reg_vals)`.
        layout = [
            (0.0, self.r_break, dr_in, th0, band_hi, dthb),
            (0.0, self.r_break, dr_in, band_hi, split, dthg),
            (0.0, self.r_break, dr_in, split, 90.0, dths),
            (self.r_break, self.r_near, dr_out, th0, band_hi, dthb),
            (self.r_break, self.r_near, dr_out, band_hi, split, dthg),
            (self.r_break, self.r_near, dr_out, split, 90.0, dths),
            (self.r_near, self.r1_max, dr_out, th0, band_hi, dthb),
            (self.r_near, self.r1_max, dr_out, band_hi, split, dthg_far),
            (self.r_near, self.r1_max, dr_out, split, 90.0, dths_far),
        ]

        # The band regions (0 and 3) are filled LAZILY. They are ~320 nodes at
        # ~48 ms each — 4.2x the whole rest of the grid — and a deck with
        # nothing under 1 deg never reads them. Filling them eagerly made an
        # existing below-grid test go 0.50 s -> 16.2 s in the xdist lane,
        # where OpenMP is pinned to one thread per worker. Cost now falls
        # only on the decks momwire#838 is for.
        self._band_idx = (0, 3, 6)
        self._regions = []
        pad = _SOMM_BELOW_PAD_ROWS
        for ridx, (r0, r1, dr, th0, th1, dth) in enumerate(layout):
            # Pad the R₁ axis past both ends (see `_SOMM_BELOW_PAD_ROWS`): a
            # query anywhere in [r0, r1] then has two lattice rows on each
            # side and gets the CENTRED leg of the cubic. The low end only
            # pads where there is room — R₁ < 0 is not a geometry.
            pad_lo = pad if r0 > 0.0 else 0
            r_start = r0 - pad_lo * dr
            n_r = max(int(np.ceil((r1 - r0) / dr)) + 1 + pad_lo + pad, 4)
            n_th = int(round((th1 - th0) / dth)) + 1
            r_nodes = r_start + dr * np.arange(n_r)
            th_nodes = np.radians(th0 + dth * np.arange(n_th))
            if ridx in _DEFERRED_REGIONS:
                surf = None  # deferred; see `_fill_region`
            else:
                rr, tt = np.meshgrid(r_nodes, th_nodes, indexing="ij")
                surf = iv_surfaces_direct_below(
                    self.eps_t,
                    self.k2,
                    rr,
                    tt,
                    rtol=rtol,
                    omega=self.omega,
                    mu=self.mu,
                    health=health,
                )
            self._regions.append(
                {
                    "r0": r_start,
                    "dr": dr,
                    "n_r": n_r,
                    "th0": np.radians(th0),
                    "dth": np.radians(dth),
                    "n_th": n_th,
                    # An unfilled band region carries a NaN table, not an
                    # absent one: the C++ kernel takes every region up front
                    # and would otherwise reject the shape. NaN rather than
                    # zeros on purpose — the conservative theta bound in
                    # `remainder_field_proj_below` means a query can never
                    # route here unfilled, and if that reasoning is ever
                    # wrong the answer must be visibly wrong rather than
                    # plausibly zero.
                    "vals": (
                        np.full(
                            (len(_SURF_KEYS), n_r, n_th), np.nan, dtype=np.complex128
                        )
                        if surf is None
                        else np.stack([surf[key] for key in _SURF_KEYS])
                    ),
                    "filled": surf is not None,
                    "r_nodes": r_nodes,
                    "th_nodes": th_nodes,
                }
            )
        self._rtol = rtol
        self._health = health

    def eval(self, R1, theta):
        """Interpolate the four below surfaces; REFUSES past `r1_max`.

        The parent clamps an over-range R₁ to `r1_max` because out there
        its remainder is a negligible tail. This family's remainder is not
        negligible anywhere the product cares about — it grows relative to
        the direct term with range — so a clamp would return a confident
        wrong number. θ is range-checked for the same reason: the parent's
        silent clip to [0, π/2] would absorb a wrong-side geometry into a
        plausible-looking answer instead of naming it.
        """
        R1 = np.asarray(R1, dtype=float)
        theta = np.asarray(theta, dtype=float)
        if R1.size and float(np.min(R1)) < 0.0:
            raise ValueError("query R1 must be non-negative")
        if R1.size and float(np.max(R1)) > self.r1_max * (1.0 + 1e-9):
            raise ValueError(
                f"below/below grid tabulated to r1_max = {self.r1_max:.6g} "
                f"({self.r1_max / self.lam_m:.3g} in-medium wavelengths) but "
                f"was queried at R1 = {float(np.max(R1)):.6g}: past the "
                "tabulation the below/below remainder is NOT a negligible "
                "tail (it grows relative to the direct term with range), so "
                "there is no honest clamp — size the grid for the geometry "
                "or refuse the geometry"
            )
        if theta.size and float(np.max(theta)) > 0.5 * np.pi + 1e-12:
            raise ValueError(
                "below/below grid theta must lie in [th_min, pi/2]; a theta "
                "past pi/2 means the geometry is not two points below the "
                "interface"
            )
        if theta.size and float(np.min(theta)) < self.th_min * (1.0 - 1e-9):
            raise ValueError(
                "below/below grid tabulated from theta = "
                f"{math.degrees(self.th_min):.4g} deg but was queried at "
                f"{math.degrees(float(np.min(theta))):.4g} deg: below the "
                "grazing floor the surfaces carry the lateral wave's "
                "logarithmic structure (measured drift 1.1 to 3.3 of scale "
                "between 2 and 0.05 deg at R1 = 3.5 in-medium wavelengths), "
                "and theta = 0 has no node at all -- h = 0 leaves the tail "
                "without its exponential decay. Raise the pair's depth sum "
                "relative to its horizontal separation, or refuse the "
                "geometry"
            )
        return self._interp(
            np.minimum(R1, self.r1_max), np.clip(theta, self.th_min, 0.5 * np.pi)
        )

    def _fill_region(self, idx):
        """Materialize one deferred region, once. Idempotent.

        The values are exactly what an eager fill would have produced: the
        node sets were fixed in the constructor and only the evaluation
        moved. What is deferred and why is `_DEFERRED_REGIONS`.
        """
        reg = self._regions[idx]
        if reg["filled"]:
            return
        rr, tt = np.meshgrid(reg["r_nodes"], reg["th_nodes"], indexing="ij")
        surf = iv_surfaces_direct_below(
            self.eps_t,
            self.k2,
            rr,
            tt,
            rtol=self._rtol,
            omega=self.omega,
            mu=self.mu,
            health=self._health,
        )
        reg["vals"] = np.stack([surf[key] for key in _SURF_KEYS])
        reg["filled"] = True

    def _ensure_band(self):
        """The sub-1 deg band, in every R₁ zone. Kept as a named entry point
        because the momwire#838 part 1 gates and the C++ prefill both mean
        exactly this set."""
        for idx in self._band_idx:
            self._fill_region(idx)

    @property
    def _band_filled(self):
        return all(self._regions[i]["filled"] for i in self._band_idx)

    def _ensure_for(self, mx_r1, mn_th, mx_th):
        """Fill every region a batch bounded by these extremes could touch.

        The C++ kernel takes all nine tables up front and reports only the
        query set's extremes, so this is deliberately CONSERVATIVE: R₁ zones
        from the innermost up to the one holding `mx_r1`, crossed with every
        θ band the interval [mn_th, mx_th] meets. A deck that stays inside
        2 λ_m never materializes the far annulus; one with nothing under 1°
        never materializes the band.
        """
        zones = [0]
        if mx_r1 > self.r_break:
            zones.append(3)
        if mx_r1 > self.r_near:
            zones.append(6)
        bands = []
        if mn_th < self.th_band_hi:
            bands.append(0)
        if mx_th >= self.th_band_hi and mn_th <= self._th_split:
            bands.append(1)
        if mx_th > self._th_split:
            bands.append(2)
        for z in zones:
            for b in bands:
                self._fill_region(z + b)

    def _interp(self, R1, theta):
        """The parent's bicubic, over THIS family's three-θ-band layout.

        Not `super().eval`: the parent routes on one θ split and reads a
        six-region grid as the momwire#159 far zone, which this layout is
        not. Overriding here rather than generalizing the parent keeps the
        ±=+ family's routing byte for byte — measured, not assumed (the
        golden ±=+ anchors are unmoved by momwire#838).
        """
        r_b, th_b = np.broadcast_arrays(R1, theta)
        shape = r_b.shape
        r_f = np.minimum(r_b.ravel(), self.r1_max)
        th_f = th_b.ravel()

        # (R₁ zone) × (θ band), θ fastest — the layout the constructor built.
        # STRICT `<` at the band edge: θ = th_band_hi belongs to the OLD
        # grazing band, which is what makes the old domain bit-identical
        # there. The two bands' node at 1° is the same direct evaluation, but
        # `th_min + Δθ·n` need not reproduce `th_band_hi` to the last bit, and
        # a query landing on the fine band's copy would differ in low bits.
        band = np.where(
            th_f < self.th_band_hi, 0, np.where(th_f <= self._th_split, 1, 2)
        )
        zone = np.where(r_f <= self.r_break, 0, np.where(r_f <= self.r_near, 3, 6))
        region_of = zone + band

        # Materialize whatever this query actually reaches (momwire#838).
        for idx in np.unique(region_of):
            if not self._regions[idx]["filled"]:
                self._fill_region(int(idx))

        out = np.empty((len(_SURF_KEYS), r_f.size), dtype=np.complex128)
        for idx, reg in enumerate(self._regions):
            sel = np.nonzero(region_of == idx)[0]
            if sel.size == 0:
                continue
            fr = (r_f[sel] - reg["r0"]) / reg["dr"]
            ft = (th_f[sel] - reg["th0"]) / reg["dth"]
            i0 = np.clip(np.floor(fr).astype(int) - 1, 0, reg["n_r"] - 4)
            j0 = np.clip(np.floor(ft).astype(int) - 1, 0, reg["n_th"] - 4)
            wr = self._lagrange4(fr - i0)
            wt = self._lagrange4(ft - j0)
            ii = i0[:, None] + np.arange(4)[None, :]
            jj = j0[:, None] + np.arange(4)[None, :]
            block = reg["vals"][:, ii[:, :, None], jj[:, None, :]]
            out[:, sel] = np.einsum("snij,ni,nj->sn", block, wr, wt)

        return {key: out[s].reshape(shape) for s, key in enumerate(_SURF_KEYS)}

    def scaled_to(self, k2, omega, mu):
        raise NotImplementedError(
            "the below/below grid has no normalized-master rescaling: the "
            "±=+ family's frequency reuse rests on S = omega*mu*G(eps_t; "
            "R1/lambda, theta) with a lambda-proportional lattice, and "
            "momwire#553 U2 measured neither for this family"
        )


def get_grid_below(eps_t, k2, r1_max, omega, mu=_MU0, health=None):
    """Cached `SommerfeldGridBelow`, sharing `_sommerfeld`'s grid cache.

    The cache is EXTENDED, not forked: `_sommerfeld.get_grid`'s key gained
    a leading regime discriminator in the same change, so an above and a
    below grid at identical (ε̃, k₂, r1, ω, μ) are two entries in one dict
    instead of one entry serving whichever regime asked first. Forking the
    cache would have doubled the eviction bookkeeping and left two places
    to get the FIFO bound wrong.

    No normalized-master layer here (see `SommerfeldGridBelow`): the
    physical grid is what gets cached.

    **The ε̃ ladder does NOT apply below, and this is the third appearance
    of one inversion.** `_somm_eps_bucket` quantizes Im(ε̃) onto a 1 %
    geometric ladder so a frequency sweep collapses to a few fills. Its
    calibration is explicit that a relative Im perturbation δ moves the
    ±=+ surfaces by only ~(0.08–0.14)·δ of scale, i.e. ≲7e-4 at the 0.5 %
    worst offset — comfortably under that family's ~2e-3 interpolation bar.
    Every word of that rests on the ±=+ premise that the remainder is a
    small correction to direct + image.

    Below the interface it is not, and the arithmetic inverts with it. The
    surfaces carry e^{+j(k_p ρ + k_m h)} with k_m = k_p√ε̃, so a relative
    Im shift δ moves the phase by ~½·δ·|k_m|R₁, which at the 2 λ_m cap is
    ~½·δ·4π ≈ 6·δ — GROWING with R₁ instead of being damped by 0.1. At the
    ladder's 0.4 % worst offset that is a ~2e-2 phase error, and because
    the remainder carries the whole field out there (measured: 12×, 168×
    direct+image at working range) it lands on the composed answer at the
    same size. Measured at (B/7 MHz, R₁ = 1.31 λ_m, θ = 2.18°): grid built
    at the bucketed ε̃ = 20 − 77.34644j against direct at the caller's
    exact 20 − 77.03616j reads 5.98e-3 / 1.52e-2 / 4.13e-3 / 6.59e-3 per
    surface, against 1.5e-6 … 5.7e-5 when both sides share one ε̃.

    And it buys nothing here. ε̃ = ε_r − jσ/(ωε₀) is frequency-dependent,
    so a sweep re-keys on ω anyway (this family has no normalized master to
    amortize across frequencies), and a single deck is a single ε̃. So the
    below family keys and FILLS on the caller's exact ε̃. The above/above
    ladder in `_sommerfeld.get_grid` is untouched — its premise still holds
    there.

    `r1_max` is still bucketed up: the lattice spacing is fixed, so a grid
    tabulated further out has node positions that coincide exactly with the
    smaller one's and interpolates identically inside it.
    """
    k2 = float(k2)
    eps_t = complex(eps_t)
    lam_m = 2.0 * np.pi / abs(k_medium(eps_t, k2))
    r1b_wl = _somm_r1_bucket_wl(min(float(r1_max) / lam_m, _SOMM_BELOW_R1_CAP_LAMBDA_M))
    key = ("below", eps_t, k2, r1b_wl, float(omega), float(mu))
    grid = _GRID_CACHE.get(key)
    if grid is None:
        _evict_fifo(_GRID_CACHE, _GRID_CACHE_MAX)
        grid = SommerfeldGridBelow(
            eps_t, k2, r1b_wl * lam_m, omega=float(omega), mu=float(mu), health=health
        )
        _GRID_CACHE[key] = grid
    return grid


# ---------------------------------------------------------------------------
# The point projection
# ---------------------------------------------------------------------------


def remainder_field_proj_below(obs, t_obs, src, t_src, ground_z, k_p, k_m, grid):
    """Projected below/below remainder table t_m · F(r_m, r_n) · t_n.

    The eqs 143–147 azimuth combination, unchanged: the surfaces already
    carry the ±=− signs, so the dyad algebra is literally the ±=+ one
    (R_z^H = −cosφ·R_ρ^V survives the swap — measured, phase 0). What
    changes is what goes into it:

    * `hh = |z − z_g| + |z′ − z_g|`, the two DEPTHS added, where the ±=+
      spelling adds two heights and gets the same formula only because
      both are positive there. Written as depths so a wrong-side point
      cannot quietly produce a plausible h.
    * `g = e^{−j(k_p ρ + k_m h)}/R₁`, the two-leg blend of
      `divide_out_below` — the surface is the amplitude of THIS factor, and
      pairing it with any other g is a silent decade of error. Both
      wavenumbers are arguments for exactly that reason.

    Both endpoints must be below `ground_z` and this RAISES if they are
    not. `remainder_field_proj` would have clamped θ into range and
    returned a number; the near-interface case (a wire crossing the
    interface, contact) is momwire#524 phase 3, not something to absorb.

    C++ TWIN (momwire#568 unit 2). This used to read "numpy only: the C++
    `remainder_field_proj_batch` takes a `double k` and cannot carry a complex
    wavenumber". It now has `remainder_field_proj_batch_below` beside it — a
    TWIN rather than a widening of the ±=+ kernel, because the two divide-outs
    are not the same expression: above/above divides out e^{−jkR₁}/R₁, a
    function of R₁ alone, and this family divides out the two-leg blend, which
    depends on θ. Widening in place would have put that choice inside the ±=+
    family's hottest inner loop for a family that does not need it; the ±=+
    path keeps its bytes instead.

    The twin serves only a real tabulation. A duck-typed `eval`-able (the
    tests' direct-surface stand-in) has no `_regions` to flatten and takes the
    numpy body, which is also what keeps the grid-vs-direct gates honest.
    """
    obs = np.asarray(obs, dtype=float)
    src = np.asarray(src, dtype=float)
    t_obs = np.asarray(t_obs, dtype=float)
    t_src = np.asarray(t_src, dtype=float)
    if getattr(grid, "regime", None) != "below":
        raise ValueError(
            "remainder_field_proj_below needs a SommerfeldGridBelow: the "
            "above/above grid tabulates a different surface family over a "
            "different divide-out"
        )
    d_obs = ground_z - obs[:, 2]
    d_src = ground_z - src[:, 2]
    if float(np.min(d_obs)) <= 0.0 or float(np.min(d_src)) <= 0.0:
        raise ValueError(
            "the below/below remainder needs BOTH endpoints strictly below "
            f"ground_z = {ground_z!r}: got min observer depth "
            f"{float(np.min(d_obs)):.6g}, min source depth "
            f"{float(np.min(d_src)):.6g} (a non-positive depth is a point on "
            "or above the interface — that geometry is momwire#524 phase 3)"
        )

    if _use_below_accel() and getattr(grid, "_regions", None) is not None:
        from ._sommerfeld import grid_cpp_args

        # The sub-1 deg band and the whole far annulus fill lazily
        # (momwire#838), and C++ takes every region table up front, so an
        # unfilled region goes in as NaN. The ORDER below is what makes that
        # safe: this pass only has to produce the query's EXTREMES, which are
        # geometry and do not read a table; `grid.eval` then raises the domain
        # refusals off them, and only a deck that is both in-domain and
        # actually reaching gets the regions it touches filled and the kernel
        # re-run. A refused geometry fills nothing.
        filled_before = tuple(r["filled"] for r in grid._regions)

        def _run():
            return _acc.remainder_field_proj_batch_below(
                obs,
                t_obs,
                src,
                t_src,
                float(ground_z),
                float(k_p),
                complex(k_m),
                float(grid.th_min),
                float(grid.th_band_hi),
                *grid_cpp_args(grid),
            )

        out, mx_r1, mn_th, mx_th = _run()
        # The three domain refusals are NOT transcribed into C++. The kernel
        # reports the query's extremes and they go straight back through
        # `SommerfeldGridBelow.eval`, which raises in its own words — one copy
        # of that prose, and no way for the two paths to disagree about which
        # geometries are served. Two points; the interpolation it also does is
        # discarded and costs microseconds.
        if obs.shape[0] and src.shape[0]:
            grid.eval(np.array([mx_r1, mx_r1]), np.array([mn_th, mx_th]))
            grid._ensure_for(mx_r1, mn_th, mx_th)
            if tuple(r["filled"] for r in grid._regions) != filled_before:
                out, mx_r1, mn_th, mx_th = _run()
        return out

    th_src = np.hypot(t_src[:, 0], t_src[:, 1])
    safe_t = th_src > 1e-12
    ux = np.where(safe_t, t_src[:, 0] / np.where(safe_t, th_src, 1.0), 1.0)
    uy = np.where(safe_t, t_src[:, 1] / np.where(safe_t, th_src, 1.0), 0.0)
    tz_src = t_src[:, 2]

    dx = obs[:, 0][:, None] - src[:, 0][None, :]
    dy = obs[:, 1][:, None] - src[:, 1][None, :]
    rho = np.hypot(dx, dy)
    hh = d_obs[:, None] + d_src[None, :]
    r1 = np.sqrt(rho * rho + hh * hh)
    surf = grid.eval(r1, np.arctan2(hh, rho))
    g = divide_out_below(k_p, k_m, rho, hh)

    tiny = 1e-12 * grid.r1_max
    safe_r = rho > tiny
    inv_rho = np.where(safe_r, 1.0 / np.where(safe_r, rho, 1.0), 0.0)
    dhx = np.where(safe_r, dx * inv_rho, ux[None, :])
    dhy = np.where(safe_r, dy * inv_rho, uy[None, :])
    cphi = ux[None, :] * dhx + uy[None, :] * dhy
    sphi = ux[None, :] * dhy - uy[None, :] * dhx

    e_rho = g * (
        tz_src[None, :] * surf["IrhoV"] + th_src[None, :] * cphi * surf["IrhoH"]
    )
    e_phi = g * th_src[None, :] * sphi * surf["IphiH"]
    e_z = g * (tz_src[None, :] * surf["IzV"] - th_src[None, :] * cphi * surf["IrhoV"])
    return (
        t_obs[:, 0][:, None] * (dhx * e_rho - dhy * e_phi)
        + t_obs[:, 1][:, None] * (dhy * e_rho + dhx * e_phi)
        + t_obs[:, 2][:, None] * e_z
    )


def remainder_field_below(obs, src, src_moment, ground_z, k_p, k_m, grid):
    """The below/below remainder as a VECTOR at each observer point:
    (N_obs, 3) complex, summed over sources.

    `remainder_field_proj_below` kept as a vector, the way `_field_point`
    keeps `remainder_field_proj` as one — a complex current moment
    p = I·dl enters through the two bilinears p·d̂ and p×d̂|_z with no
    magnitude/direction split, because a complex horizontal moment has no
    direction to normalize.
    """
    obs = np.asarray(obs, dtype=float)
    src = np.asarray(src, dtype=float)
    p = np.asarray(src_moment, dtype=complex)
    if getattr(grid, "regime", None) != "below":
        raise ValueError(
            "remainder_field_below needs a SommerfeldGridBelow: the "
            "above/above grid tabulates a different surface family over a "
            "different divide-out"
        )
    d_obs = ground_z - obs[:, 2]
    d_src = ground_z - src[:, 2]
    if float(np.min(d_obs)) <= 0.0 or float(np.min(d_src)) <= 0.0:
        raise ValueError(
            "the below/below remainder needs BOTH endpoints strictly below "
            f"ground_z = {ground_z!r}: got min observer depth "
            f"{float(np.min(d_obs)):.6g}, min source depth "
            f"{float(np.min(d_src)):.6g} (a non-positive depth is a point on "
            "or above the interface — that geometry is momwire#524 phase 3)"
        )

    dx = obs[:, 0][:, None] - src[:, 0][None, :]
    dy = obs[:, 1][:, None] - src[:, 1][None, :]
    rho = np.hypot(dx, dy)
    hh = d_obs[:, None] + d_src[None, :]
    r1 = np.sqrt(rho * rho + hh * hh)
    surf = grid.eval(r1, np.arctan2(hh, rho))
    g = divide_out_below(k_p, k_m, rho, hh)

    tiny = 1e-12 * grid.r1_max
    safe_r = rho > tiny
    inv_rho = np.where(safe_r, 1.0 / np.where(safe_r, rho, 1.0), 0.0)
    dhx = np.where(safe_r, dx * inv_rho, 1.0)
    dhy = np.where(safe_r, dy * inv_rho, 0.0)

    cph = p[None, :, 0] * dhx + p[None, :, 1] * dhy
    sph = p[None, :, 0] * dhy - p[None, :, 1] * dhx

    e_rho = g * (p[None, :, 2] * surf["IrhoV"] + cph * surf["IrhoH"])
    e_phi = g * sph * surf["IphiH"]
    e_z = g * (p[None, :, 2] * surf["IzV"] - cph * surf["IrhoV"])

    out = np.empty((obs.shape[0], 3), dtype=complex)
    out[:, 0] = np.sum(dhx * e_rho - dhy * e_phi, axis=1)
    out[:, 1] = np.sum(dhy * e_rho + dhx * e_phi, axis=1)
    out[:, 2] = np.sum(e_z, axis=1)
    return out


def image_coefficient_below(eps_t):
    """A_m = (k_p² − k_m²)/(k_p² + k_m²) = (1 − ε̃)/(1 + ε̃).

    MEASURED, not derived: phase 0 scored all eight combinations of
    {image dyad literal/mirror} × {sign A_m} × {sign of the remainder}
    against the empymod oracle over 170 in-medium points and this one won
    by 420×, with every loser an O(1) failure. It is the negative of the
    ±=+ family's C₂ = (ε̃ − 1)/(ε̃ + 1), which is exactly the kind of
    coincidence a sign scan exists to keep honest.
    """
    eps_t = complex(eps_t)
    return (1.0 - eps_t) / (1.0 + eps_t)
