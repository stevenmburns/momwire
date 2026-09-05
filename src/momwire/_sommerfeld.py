"""Sommerfeld-integral engine for the NEC-style Sommerfeld/Norton ground.

Implements the ground-remainder Sommerfeld integrals of the NEC-2 theory
manual (docs/nec2_theory_manual.pdf §IV.1–IV.2): the six λ-integrals of
eqs 148–153 with the D₁/D₂ kernels of eqs 154–155, evaluated on the
deformed contours of figs 13–14, and assembled into the four
interpolation surfaces I_ρ^V, I_z^V, I_ρ^H, I_φ^H of eqs 156–159 (with
the analytic R₁ → 0 limits of eqs 169–172). See
docs/sommerfeld-ground-plan.md Phase 1.

Clean-room note: implemented from the public-domain theory-manual
equations only; no GPL Sommerfeld code (nec2c, nec2++/PyNEC) was
consulted. Validation is data-level: the manual's figure extrema
(tests/oracle_sommerfeld_figs.py), closed-form identities, and
nec2c-captured golden impedances.

Conventions (matching momwire and `_ground_refl`):

  e^{+jωt} time dependence; ε̃ = εr − jσ/(ωε₀) with Im(ε̃) ≤ 0 for a
  passive ground; k₂ = free-space wavenumber (real); k₁ = k₂√ε̃ with
  Im(k₁) ≤ 0.

  γᵢ(λ) = (λ² − kᵢ²)^{1/2} with NEC's vertical branch cuts (fig 13:
  downward from +kᵢ, upward from −kᵢ), realized as
  γ = √(−j(λ−k))·√(j(λ+k)) with principal square roots. On the real
  axis this gives the radiation branch γ = +j√(k²−λ²) for |λ| < k, so
  e^{−γ(z+z′)} is the outgoing wave — pinned by the Sommerfeld-identity
  test (the same contours must reproduce e^{−jk₂R}/R exactly).

Geometry per pair: ρ = horizontal distance, h = z + z′ (both source and
observer above the interface, h ≥ 0), R₁ = √(ρ² + h²) = distance from
the image point, θ = atan2(h, ρ). The Bessel (J₀) form of the integrals
is used for ρ < 2h, the Hankel (H₀⁽²⁾) form otherwise (a widened version
of NEC's ρ < h/2 rule — see `_six_integrals`).

All distances are in the length unit implied by k₂ (SI meters when k₂
is rad/m). The C₁ = −jωμ₀/(4πk₂²) unit-dipole normalization of eq 123
is applied with ω = k₂c by default.
"""

import math
import os

import numpy as np
from scipy.special import hankel2, jv

from ._accel import acc as _acc
from ._cancel import SolveAborted
from ._constants import C_LIGHT

# Far-pair grid-extent cap, in wavelengths (issue #157). The interpolation
# grid's radius `r1_max` is sized to the geometry's largest image-point
# distance, and grid-fill cost grows ~quadratically in that radius (both the
# radial and, near grazing, the theta node counts scale with it). A wire
# parked hundreds of wavelengths away — the NEC TL-anchor idiom, or any
# genuinely large structure over real ground (rhombics, long-wire arrays) —
# would make construction do millions of oscillatory Sommerfeld integrals and
# effectively hang. Capping the radius bounds that cost: beyond the cap the
# smooth remainder g·surf is a negligible ~1/R1 tail (the space wave, carried
# by the separate reflection-coefficient/image term, dominates and decays no
# faster), so `proj_one`'s existing r1 -> r1_max clamp — g keeps the true
# distance, only the slowly-varying surface amplitude freezes at the cap — is
# an accurate, bounded stand-in for the true far interaction. The 15-lambda
# default is calibrated: the remainder is empirically negligible beyond
# ~3-4 lambda (a 6-lambda long wire, two dipoles 8 lambda apart, and a
# 171-lambda TL anchor over finite ground all give bit-identical impedance
# for every cap >= ~8 lambda vs a 25-lambda grid), so 15 lambda leaves ~4x
# margin and grids any real HF-over-ground structure exactly while bounding
# the pathological remote-wire fill to a few seconds. Overridable via the
# environment for validation/benchmarking.
_SOMM_R1_CAP_LAMBDA = float(os.environ.get("MOMWIRE_SOMM_R1_CAP_LAMBDA") or "15.0")

# Radius where the grid switches from the near tabulation (NEC fig-12 spacings,
# extent-keyed theta) to a coarse far zone (issue #159). Empirically the fine
# structure of the four surfaces — the lateral-wave interference near grazing —
# lives at moderate R1 (~0.5-3 lambda) and decays beyond: dense scans at
# R1 = 5-10 lambda show a 2.5 deg theta / 0.2 lambda R1 lattice interpolates to
# <= 7e-4 of surface scale for every tested ground (incl. the lossless eps=16
# stress case), vs the <= 2e-3 near-zone bar. Keying the fine spacings to the
# full extent (the pre-#159 layout) made node count grow ~quadratically with
# geometry size for nothing: at the 15-lambda #157 cap the split cuts the fill
# ~7.6x with measured interpolation error identical to the near-keyed grid.
# 4 lambda matches where #157 measured the remainder itself becoming
# negligible, and grids with r1_max <= the split build bit-identically to the
# pre-#159 layout. Overridable for validation (raise it to force the old
# layout on any extent).
#
# Addendum (2026-08-01, issue #161): that "<= 7e-4 for every tested ground"
# reading stands as measured, but for the LOSSLESS eps=16 stress case it was
# measured against a direct evaluation that was itself wrong past R1 ~ 2.84
# lambda at grazing — the fig-14 waypoint had dropped below k1's branch point
# there, and the mis-branched contour returned a nearly flat surface, which a
# coarse lattice of course interpolates well. Against the corrected surface
# the same far lattice reads 2.2e-2 (R1 4-5), 5.7e-2 (5-7) and 9.3e-3 (7-10)
# of scale, all of it at theta < 5 deg; the grid was always this far from the
# truth there, only now it is visible. Lossy grounds are bit-identical before
# and after (10-1.26j: 1.3e-3/1.4e-3/1.5e-3; 3-1.2j: 1.9e-3) and still meet
# the 2e-3 bar, so no spacing is moved here — sizing the far lattice for a
# near-lossless ground at grazing is #159's call, not #161's.
_SOMM_R1_NEAR_LAMBDA = float(os.environ.get("MOMWIRE_SOMM_R1_NEAR_LAMBDA") or "4.0")

# Far-zone lattice (wavelengths / degrees) — see the calibration note above.
_SOMM_DR_FAR_LAMBDA = 0.2
_SOMM_DTH_FAR_DEG = 2.5

# Steep-band R1 node-count cap for the momwire#443 boundary-layer keying:
# its dr never goes below r_break/(cap - 1), so the eps~ -> infinity decks
# (layer ~ 1e-7 lambda) cannot explode the fill. 513 puts the floor at
# ~3.9e-4 lambda, which fully resolves every physical ground (sea water,
# the worst, needs 5e-4).
_SOMM_N0_CAP = 513

# The grazing/steep theta split, degrees, all zones (see the momwire#443
# layout comment in SommerfeldGrid.__init__ for why it sits at 30).
_SOMM_TH_SPLIT_DEG = 30.0

# Frequency-axis grid reuse (issue #159, phase 2). In wavelength coordinates
# the four surfaces obey S = omega * mu * G(eps_t; R1/lambda, theta): omega
# and mu enter iv_surfaces_direct only through the linear eq-123
# normalization C1, and the measured k2-scaling at omega = k2*c is exactly
# linear. Every lattice parameter (cap, beat keying, dth2, far spacings) is
# lambda-proportional too, so one normalized master fill serves any
# frequency via a coordinate scale plus one scalar multiply. The only true
# frequency dependence left is eps_t = eps_r - j*sigma/(omega*eps0), whose
# imaginary part drifts ~1/omega across a sweep — so Im(eps_t) is served from
# a geometric LADDER of rungs (momwire#159 phase 2), and since momwire#902 a
# query BETWEEN two rungs is the linear blend of the masters at both, filled
# on ONE shared lattice, rather than the nearest rung alone:
#
#   step 1 % , nearest rung (before #902)   worst offset 0.5 %   error 2.0e-3
#   step 10 %, blend of the bracketing pair  worst = midpoint     error 5.8e-4
#   step 20 %, blend                         "                    error 2.2e-3
#
# (max over every lattice node, per surface relative to its own scale, soil
# eps_r 20 / sigma 0.0303 at 14 MHz, r1 = 1 lambda — the #902 study.) The
# blend's error is second order in the rung spacing where the snap's is first
# order, so the 10 % ladder is 3.4x MORE accurate than the 1 % snap was and
# needs a tenth of the rungs: a 2.5 % band (20 m) sits inside one rung pair
# three times in four, and a sweep costs the pair's two fills at its first
# point and nothing after — where the 1 % snap paid one fill per rung
# crossing, five stalls of ~1.2 s spread through a 50-point sweep (that is
# #902's symptom, inside EZNEC). Re(eps_t) does not move with frequency and
# is ~8x more sensitive, so it is keyed exactly. Set the env override to 0
# to disable the ladder (exact fill per eps_t). Wide multi-octave sweeps
# still fill per rung pair — Im genuinely changes several-fold there; that is
# physics, not caching.
#
# The two masters of a pair share a lattice keyed at the pair's DEMANDING
# rung (the larger |Im|, i.e. the larger |k1|: the near-interface layer 1/|k1|
# and the lateral-wave beat 2 pi/|k1 - k2| both key the spacing DOWN as |k1|
# grows), so the blend is elementwise on identical node sets and no
# resampling error enters. A rung therefore has one master per pair it
# belongs to, keyed (rung, lattice rung): a sweep that straddles a rung pays
# four fills rather than three, once. That cost was preferred to resampling
# the neighbour's master, which would put a second interpolation under the
# blend.
_SOMM_EPS_IM_BUCKET = float(os.environ.get("MOMWIRE_SOMM_EPS_IM_BUCKET") or "0.10")

# The r1 bucket is taken with this much SLACK above the geometry's own r1/lambda
# (momwire#902): a sweep moves r1/lambda upward with frequency, and a geometry
# sitting just under a 1.25^n edge crossed it mid-band and refilled (Bydipole1
# at 14.307 MHz, one of the five stalls). A grid tabulated to the next bucket
# serves any smaller r1 with the same accuracy, so the only cost of the slack
# is the occasional one-bucket-larger fill (measured +10 % at 1 -> 1.25 lambda),
# and a sweep narrower than the slack never straddles an edge. Env override.
_SOMM_R1_SWEEP_SLACK = float(os.environ.get("MOMWIRE_SOMM_R1_SWEEP_SLACK") or "0.05")

# Reference scales the normalized masters are filled at (lambda_ref = 1).
_K2_REF = 2.0 * np.pi

_C_LIGHT = C_LIGHT  # momwire#456: one owner, in `momwire._constants`
_MU0 = 4e-7 * np.pi

# Gauss–Legendre rule shared by all contour sections.
_GAUSS_N = 24
_GX, _GW = np.polynomial.legendre.leggauss(_GAUSS_N)

# Recursion cap for the adaptive sections (branch-point neighborhoods).
_ADAPT_DEPTH = 14


def max_image_distance(seg_l, seg_r, ground_z, chunk_rows=None):
    """Grid-sizing radius for a Sommerfeld build: the largest obs-to-image
    -point distance over every pair of segment ENDPOINTS (issue #331).

    Four call sites (`BSplineSolver._Z_sommerfeld_remainder`,
    `._Q_sommerfeld_remainder_enrich`, `HMatrixSolver._somm_nodes`,
    `SinusoidalGalerkinSolver._field_tensor_sommerfeld_remainder`) all sized
    their grid the same way: the obs-to-image distance is convex in the two
    segment parameters, so its max over all pairs is attained at endpoint
    pairs, and `ex = concat(seg_l, seg_r)` is the `2N` real+image endpoint
    set (image = the same points, reflected through `ground_z` inside the
    distance formula below). This is that one shared computation, INCLUDING
    the `* 1.001` grid-oversize fudge every site applied before handing the
    result to `_somm_grid` / `get_grid` — the return value is exactly what
    the deleted `r1_max = ...` line produced, no more.

    Row-chunks the endpoint axis instead of materializing the full
    `(2N, 2N)` distance matrix (a ~160-190 byte/pair transient that peaked
    ~4 GB at N=4,700 — issue #331). This is exact, not an approximation:
    max is order- and grouping-insensitive over non-NaN reals (these
    distances are sums of squares under a sqrt, so never NaN), so
    accumulating `r1 = max(r1, chunk.max())` over row bands is bit-identical
    to the single full-matrix `.max()` it replaces.

    `chunk_rows` bounds the transient at a few MB regardless of N: for `n`
    endpoints, one row-chunk of `rows` rows allocates O(1) `(rows, n)`
    float64 arrays (dxe/dye/hze plus their squares, sum, and sqrt) at 8
    bytes each. The default `rows = max(1, 4_000_000 // (16 * n))` caps a
    single `(rows, n)` array at `rows * n * 8 <= 2 MB`, so the handful of
    such arrays alive at once stays a few MB total — far below the old
    `8 * (2N)**2` full-matrix transient. Pass an explicit value to force a
    particular chunking (e.g. to exercise a partial tail chunk in tests).
    """
    ex = np.concatenate([seg_l, seg_r])
    n = ex.shape[0]
    if n == 0:
        return 0.0
    x = ex[:, 0]
    y = ex[:, 1]
    z = ex[:, 2]
    if chunk_rows is None:
        chunk_rows = max(1, 4_000_000 // (16 * n))
    r1 = 0.0
    for i0 in range(0, n, chunk_rows):
        i1 = min(i0 + chunk_rows, n)
        dxe = x[i0:i1, None] - x[None, :]
        dye = y[i0:i1, None] - y[None, :]
        hze = (z[i0:i1, None] - ground_z) + (z[None, :] - ground_z)
        chunk_max = float(np.sqrt(dxe * dxe + dye * dye + hze * hze).max())
        if chunk_max > r1:
            r1 = chunk_max
    return r1 * 1.001


def _gamma(lam, k):
    """(λ² − k²)^{1/2} with vertical cuts down from +k / up from −k."""
    return np.sqrt(-1j * (lam - k)) * np.sqrt(1j * (lam + k))


def _d12(lam, k1, k2):
    """NEC eqs 154–155 kernels (the 2s of eqs 141–142 are inside)."""
    g1 = _gamma(lam, k1)
    g2 = _gamma(lam, k2)
    k1s = k1 * k1
    k2s = k2 * k2
    d1 = 2.0 / (g1 + g2) - 2.0 * k2s / (g2 * (k1s + k2s))
    d2 = 2.0 / (k1s * g2 + k2s * g1) - 2.0 / (g2 * (k1s + k2s))
    return d1, d2, g2


def _bessel_j0_j1x(x):
    """(J₀(x), J₁(x)/x) with a series switch at small |x| (ρ → 0 safe)."""
    x = np.asarray(x, dtype=np.complex128)
    small = np.abs(x) < 1e-6
    xs = np.where(small, 1.0, x)
    j0 = np.where(small, 1.0 - 0.25 * x * x, jv(0, xs))
    j1x = np.where(small, 0.5 - x * x / 16.0, jv(1, xs) / xs)
    return j0, j1x


def _integrand_six(lam, rho, h, k1, k2, form):
    """The six λ-integrands of NEC eqs 148–153, stacked (6, n):

      0: ∂²V′₂₂/∂ρ²      [D₂ e^{−γ₂h} J₀″(λρ) λ³,  J₀″ = J₁/x − J₀]
      1: ∂²V′₂₂/∂z²      [D₂ γ₂² e^{−γ₂h} J₀ λ]
      2: ∂²V′₂₂/∂ρ∂z     [+D₂ γ₂ e^{−γ₂h} J₁ λ²,   J₀′ = −J₁]
      3: (1/ρ)∂V′₂₂/∂ρ   [−D₂ e^{−γ₂h} (J₁/x) λ³]
      4: V′₂₂            [D₂ e^{−γ₂h} J₀ λ]
      5: U′₂₂            [D₁ e^{−γ₂h} J₀ λ]

    `form` = "J" (Bessel, integrate 0→∞) or "H" (Hankel, ½H₀⁽²⁾ for J₀
    over the full fig-14 contour). Identity 0 + 3 + λ²·4 = 0 (Laplacian
    of J₀) holds pointwise and is unit-tested.
    """
    lam = np.asarray(lam, dtype=np.complex128)
    d1, d2, g2 = _d12(lam, k1, k2)
    e = np.exp(-g2 * h)
    x = lam * rho
    if form == "J":
        b0, b1x = _bessel_j0_j1x(x)
    else:
        b0 = 0.5 * hankel2(0, x)
        b1x = 0.5 * hankel2(1, x) / x
    l2 = lam * lam
    l3 = l2 * lam
    common = d2 * e
    return np.stack(
        [
            common * (b1x - b0) * l3,
            common * g2 * g2 * b0 * lam,
            common * g2 * (b1x * x) * l2,
            -common * b1x * l3,
            common * b0 * lam,
            d1 * e * b0 * lam,
        ]
    )


def _gauss_segment(f, z0, z1):
    """∫ f over the straight segment z0→z1, f returning (6, n)."""
    mid = 0.5 * (z0 + z1)
    half = 0.5 * (z1 - z0)
    nodes = mid + half * _GX
    return f(nodes) @ (_GW * half)


def _adaptive_segment(f, z0, z1, rtol, depth=_ADAPT_DEPTH, whole=None):
    """Recursive bisection Gauss quadrature on z0→z1 (vector-valued).

    The tolerance is RELATIVE to the local segment magnitude: near the
    small-|λρ| part of the Hankel contour the integrand reaches ~1/(k₂ρ)²
    (canceled by neighboring sections down to the ~C₃/R₁ answer), so an
    absolute target is unreachable there while a relative one keeps the
    post-cancellation error at ~rtol·(peak/answer) — set rtol accordingly
    small (1e−11 leaves ~1e−7 after a 1e4 cancellation).
    """
    if whole is None:
        whole = _gauss_segment(f, z0, z1)
    mid = 0.5 * (z0 + z1)
    left = _gauss_segment(f, z0, mid)
    right = _gauss_segment(f, mid, z1)
    better = left + right
    err = np.max(np.abs(better - whole))
    scale = np.max(np.abs(better))
    if depth <= 0 or err <= rtol * max(scale, 1e-300):
        return better
    return _adaptive_segment(f, z0, mid, rtol, depth - 1, left) + _adaptive_segment(
        f, mid, z1, rtol, depth - 1, right
    )


def _tail(f, z0, direction, panel, rtol, ref_scale, panel0=None, max_panels=800):
    """Panel-by-panel tail ∫ from z0 toward `direction`·∞; stops when two
    consecutive panels are below rtol·scale.

    `panel` is the asymptotic panel length (0.2π/max(ρ,h) resolves the
    Bessel/exponential oscillation out there). At small R₁ that length is
    enormous compared to the k-scale structure near the tail's start, so
    the first panels ramp geometrically from `panel0` (≈ the k-scale) up
    to `panel` — a single Gauss rule leaping from |λ| ~ k to ~1/R₁ was
    the dominant Hankel-form error at R₁ ≲ 1e−3 wavelengths. Decay along
    NEC's tail directions is exponential (rate ≥ ~R₁ per unit |λ|
    relative to `panel`), so plain summation converges without Shanks
    acceleration."""
    total = 0.0
    quiet = 0
    z = z0
    step = panel if panel0 is None else min(panel0, panel)
    for _ in range(max_panels):
        z_next = z + step * direction
        contrib = _gauss_segment(f, z, z_next)
        total = total + contrib
        z = z_next
        step = min(2.0 * step, panel)
        scale = max(np.max(np.abs(total)), ref_scale)
        contrib_max = np.max(np.abs(contrib))
        # `== 0.0` matters: at eps_t = 1 the integrands are identically
        # zero and `0 < rtol * 0` would never trip the quiet counter,
        # burning all max_panels on an exactly-zero tail.
        if contrib_max == 0.0 or contrib_max < rtol * scale:
            quiet += 1
            if quiet >= 2:
                break
        else:
            quiet = 0
    return total


def _six_integrals(eps_t, k2, rho, h, rtol=1e-9, form=None):
    """Evaluate the six integrals at one (ρ, h) pair; returns (6,) complex.

    Contour selection: Bessel form (fig 13) for ρ < 2h — a widened
    version of the manual's ρ < h/2 rule, see the inline comment — and
    the Hankel form (fig 14) otherwise. `form` ("J"/"H") overrides the
    rule where both converge — the cross-form agreement test uses it.
    """
    eps_t = complex(eps_t)
    if eps_t == 1.0:
        # Free space: D1 = D2 = 0 identically. Short-circuit rather than
        # integrate ulp noise (whose relative convergence test never
        # trips — the tails would burn max_panels on ~1e-16 values).
        return np.zeros(6, dtype=np.complex128)
    k2 = float(k2)
    k1 = k2 * np.sqrt(eps_t)
    if k1.imag > 0:
        k1 = np.conj(k1)
    rho = float(rho)
    h = float(h)
    if rho < 0 or h < 0 or (rho == 0.0 and h == 0.0):
        raise ValueError(f"need rho, h >= 0 and R1 > 0, got {(rho, h)!r}")

    scale = max(rho, h)
    panel = 0.2 * np.pi / scale
    kmax = max(abs(k1), k2)
    # Contour landmarks never need to chase branch points the integrand
    # can't reach: e^{-gamma_2 h} / the Hankel tail kill everything beyond
    # ~50/scale, so for enormous |k1| (PEC-limit eps ~ 1e16) the k1-shaped
    # waypoints cap there. Any cut crossing past the cap sits between two
    # numerical zeros. No-op for physical grounds (|k1| <~ 10 k2).
    # The fig-13 end-of-adaptive landmark below takes this as-is (its
    # horizontal tail runs ABOVE both downward cuts and so cannot cross
    # one); the fig-14 waypoint d needs a branch-point-aware version of
    # the same cap — see the block before `d` is chosen (issue #161).
    kcap = 1.2 * k2 + 50.0 / scale
    kmax_eff = min(kmax, kcap)
    qtol = min(rtol, 1e-11)  # per-segment relative tolerance

    # The Bessel form has no λρ → 0 pole; its horizontal tail decays like
    # e^{−λh} with |J₀| bounded by e^{Im λ·ρ} ≤ e^{ρ/h}, so it converges
    # for any ρ ≲ h at ~48·ρ/h panels. Widen NEC's ρ < h/2 rule to
    # ρ < 2h to shrink the Hankel region (whose small-|λρ| cancellation
    # costs accuracy at very small R₁); both forms are unit-tested to
    # agree in the overlap band.
    use_bessel = rho < 2.0 * h if form is None else form == "J"
    if use_bessel:

        def f(lam):
            return _integrand_six(lam, rho, h, k1, k2, "J")

        # Fig 13: 0 → p(1+j) diagonal, then horizontal at Im λ = p. The
        # horizontal passes above the k₂/k₁ cuts (they run downward);
        # adaptive quadrature to past the branch points, then panel tail.
        p = min(1.0 / rho if rho > 0 else np.inf, 1.0 / h)
        brk = p * (1.0 + 1.0j)
        end_adapt = 1.3 * kmax_eff + 3.0 * p + 1.0j * p
        total = _adaptive_segment(f, 0.0 + 0.0j, brk, qtol)
        if end_adapt.real > brk.real:
            total = total + _adaptive_segment(f, brk, end_adapt, qtol)
            tail_start = end_adapt
        else:
            tail_start = brk
        total = total + _tail(
            f, tail_start, 1.0 + 0.0j, panel, rtol, np.max(np.abs(total))
        )
        return total

    def f(lam):
        return _integrand_six(lam, rho, h, k1, k2, "H")

    # Fig 14 contour. Tail slope matches the steepest-descent direction
    # λᵢ/λᵣ = ∓ρ/h; waypoints from the manual (p. 52–53).
    r1 = np.hypot(rho, h)
    dir_right = (h - 1.0j * rho) / r1
    dir_left = (-h - 1.0j * rho) / r1
    a = -0.4j * k2
    b = (0.6 + 0.2j) * k2
    c = (1.02 + 0.2j) * k2
    # Waypoint d, where the run turns into the descending tail. It has to
    # clear the k₁ branch point: γ₁'s cut runs straight DOWN from +k₁, so a
    # d left of k₁.real starts the tail on the far side of that cut and
    # γ₁ flips sign over the live part of the contour. The `kcap` cap above
    # is keyed to max(ρ, h) and at grazing that falls below k₁ — issue #161,
    # an isolated jump of ~1.2e−1 of scale (lossless εr=16, θ = 0.5°) at the
    # R₁ where kcap crosses k₁.real, and the capped side is the wrong one
    # (contours with d.real from 1.01·k₁ to 1.5·k₁ agree to ~1e−11).
    #
    # Capping short of k₁ is only safe once its neighborhood is numerically
    # dead. What the a→d run carries there is e^{−γ₂h}·H₀⁽²⁾(λρ) ~
    # e^{−(k₁ᵣh − k₁ᵢρ)}: along the real axis H₀⁽²⁾ merely oscillates, so
    # the decay is keyed to h alone, plus (for a lossy ground) the depth of
    # the branch point below the axis. Measured across εr = 16/100/1e4 and
    # 10−1.26j/81−9000j, that exponent tracks the capped contour's error to
    # within an O(1) prefactor: ≥ 50 puts it at the quadrature floor.
    #
    # The one case that stays capped while live is a PEC-limit ε (~1e16) at
    # h → 0: chasing a branch point out to |k₁| costs ~|k₁|ρ/2π oscillations
    # on c→d and the adaptive bisection bottoms out at 2^14 panels, so
    # |k₁| > 200·k₂ (|ε̃| > 4e4 — past any physical ground, sea water
    # included at |k₁| ≈ 95·k₂) keeps the old cap. There the whole ground
    # remainder is ~1/√ε of the image term, so the residual is invisible.
    k1_dead = k1.real * h - k1.imag * rho >= 50.0
    if k1_dead or abs(k1) > 200.0 * k2:
        cap_d = kcap
    else:
        cap_d = max(kcap, 1.01 * k1.real)
    if 1.01 * k1.real <= cap_d:
        d = 1.01 * k1.real + 0.99j * max(k1.imag, -cap_d)
    else:
        d = cap_d + 0.0j
    if d.real < 1.1 * k2:
        d = 1.1 * k2 + 1.0j * d.imag

    total = _adaptive_segment(f, a, b, qtol)
    total = total + _adaptive_segment(f, b, c, qtol)
    total = total + _adaptive_segment(f, c, d, qtol)
    ref = np.max(np.abs(total))
    p0 = 0.5 * kmax
    total = total + _tail(f, d, dir_right, panel, rtol, ref, panel0=p0)
    # The left tail runs −∞ → a on the contour; _tail integrates outward
    # from a, so its contribution enters with a minus sign.
    total = total - _tail(f, a, dir_left, panel, rtol, ref, panel0=p0)
    return total


_FORM_CODE = {None: 0, "J": 1, "H": 2}


def _six_integrals_batch(eps_t, k2, rho, h, rtol=1e-9, form=None, cancel_flag=0):
    """`_six_integrals` over parallel (ρ, h) arrays; returns (n, 6) complex.

    Routes through the C++ accelerator (`somm_six_integrals_batch`,
    OpenMP across nodes) when it is loaded and falls back to the Python
    per-node loop otherwise — same contours, same 24-point Gauss rule,
    cross-checked in tests/test_sommerfeld_accel.py.

    `cancel_flag` is a raw int32 address in the C++ kernels' convention
    (0 = no cancellation; see `CancelToken.ptr`): the kernel polls it per
    node and raises `SolveAborted`, and the Python fallback loop polls
    the same address so cancellation behaves identically on both paths.
    """
    rho = np.ascontiguousarray(rho, dtype=float).ravel()
    h = np.ascontiguousarray(h, dtype=float).ravel()
    eps_t = complex(eps_t)
    if eps_t == 1.0:
        return np.zeros((rho.size, 6), dtype=np.complex128)
    if _acc is not None and hasattr(_acc, "somm_six_integrals_batch"):
        return _acc.somm_six_integrals_batch(
            eps_t, float(k2), rho, h, float(rtol), _FORM_CODE[form], int(cancel_flag)
        )
    flag = None
    if cancel_flag:
        import ctypes

        flag = ctypes.cast(int(cancel_flag), ctypes.POINTER(ctypes.c_int32))
    out = np.empty((rho.size, 6), dtype=np.complex128)
    for i in range(rho.size):
        if flag is not None and flag.contents.value:
            raise SolveAborted()
        out[i] = _six_integrals(eps_t, k2, rho[i], h[i], rtol, form)
    return out


def _c1(k2, omega, mu):
    """NEC eq 123 normalization for a unit current moment Iℓ = 1."""
    return -1j * omega * mu / (4.0 * np.pi * k2 * k2)


def _limits_r1_zero(eps_t, k2, theta, omega, mu):
    """Analytic R₁ → 0 surface limits, NEC eqs 169–172."""
    eps_t = complex(eps_t)
    theta = np.asarray(theta, dtype=float)
    k1s = k2 * k2 * eps_t
    k2s = k2 * k2
    c1 = _c1(k2, omega, mu)
    c2 = (k1s - k2s) / (k1s + k2s)
    c3 = k2s * (k1s - k2s) / (k1s + k2s) ** 2
    s = np.sin(theta)
    co = np.cos(theta)
    # (1 − sinθ)/cosθ and (1 − sinθ)/cos²θ, θ → π/2 limits 0 and 1/2.
    near = np.abs(co) < 1e-8
    co_safe = np.where(near, 1.0, co)
    q1 = np.where(near, 0.0, (1.0 - s) / co_safe)
    q2 = np.where(near, 0.5, (1.0 - s) / (co_safe * co_safe))
    return {
        "IrhoV": c1 * c3 * k1s * q1,
        "IzV": np.full_like(q1, c1 * c3 * k1s, dtype=np.complex128),
        "IrhoH": c1 * k2s * (c2 - c3 + c3 * q2),
        "IphiH": -c1 * k2s * (c2 - c3 * q2),
    }


def iv_surfaces_direct(
    eps_t, k2, R1, theta, rtol=1e-9, omega=None, mu=_MU0, cancel_flag=0
):
    """Direct (no-grid) evaluation of the four NEC interpolation surfaces
    I_ρ^V, I_z^V, I_ρ^H, I_φ^H (eqs 156–159) at points (R₁, θ).

    R₁ in the length unit of 1/k₂; θ = atan2(z+z′, ρ) in radians,
    0 ≤ θ ≤ π/2. Returns a dict of complex arrays shaped like R₁.
    Unit dipole moment; ω defaults to k₂·c (SI).

    This is the Phase 2 grid's fill function and the tests' oracle
    hook — O(ms) per point, not for per-pair use in assembly.
    """
    if omega is None:
        omega = k2 * _C_LIGHT
    R1 = np.asarray(R1, dtype=float)
    theta = np.asarray(theta, dtype=float)
    R1b, thb = np.broadcast_arrays(R1, theta)
    out_shape = R1b.shape
    R1f = R1b.ravel()
    thf = thb.ravel()

    eps_t = complex(eps_t)
    k1s = k2 * k2 * eps_t
    k2s = k2 * k2
    c1 = _c1(k2, omega, mu)

    keys = ("IrhoV", "IzV", "IrhoH", "IphiH")
    out = {kk: np.zeros(R1f.shape, dtype=np.complex128) for kk in keys}

    zero = R1f == 0.0
    if np.any(zero):
        lim = _limits_r1_zero(eps_t, k2, thf[zero], omega, mu)
        for kk in keys:
            out[kk][zero] = lim[kk]

    nz = np.nonzero(~zero)[0]
    if nz.size:
        r1 = R1f[nz]
        rho = np.maximum(r1 * np.cos(thf[nz]), 0.0)
        h = np.maximum(r1 * np.sin(thf[nz]), 0.0)
        six = _six_integrals_batch(eps_t, k2, rho, h, rtol, cancel_flag=cancel_flag)
        v_rr, v_zz, v_rz, v_r1, v, u = six.T
        phase = r1 * np.exp(1j * k2 * r1)
        out["IrhoV"][nz] = c1 * phase * k1s * v_rz
        out["IzV"][nz] = c1 * phase * k1s * (v_zz + k2s * v)
        out["IrhoH"][nz] = c1 * phase * k2s * (v_rr + u)
        out["IphiH"][nz] = -c1 * phase * k2s * (v_r1 + u)

    return {kk: out[kk].reshape(out_shape) for kk in keys}


def greens_free_space_check(k2, rho, h, form, rtol=1e-9):
    """Contour/branch self-test: ∫₀^∞ (λ/γ₂) e^{−γ₂h} J₀(λρ) dλ over the
    module's own contours must equal the Sommerfeld identity value
    e^{−jk₂R}/R. `form` picks the fig-13 ("J") or fig-14 ("H") path
    regardless of the ρ < h/2 production rule, so both machines are
    testable on overlapping points. Returns (numeric, exact).
    """
    k2 = float(k2)
    rho = float(rho)
    h = float(h)
    r = np.hypot(rho, h)

    def f6(lam):
        lam = np.asarray(lam, dtype=np.complex128)
        g2 = _gamma(lam, k2)
        x = lam * rho
        if form == "J":
            b0, _ = _bessel_j0_j1x(x)
        else:
            b0 = 0.5 * hankel2(0, x)
        val = (lam / g2) * np.exp(-g2 * h) * b0
        return np.stack([val] * 6)

    scale = max(rho, h)
    panel = 0.2 * np.pi / scale
    if form == "J":
        p = min(1.0 / rho if rho > 0 else np.inf, 1.0 / h, 10.0 * k2)
        brk = p * (1.0 + 1.0j)
        end_adapt = 1.3 * k2 + 3.0 * p + 1.0j * p
        total = _adaptive_segment(f6, 0.0 + 0.0j, brk, rtol)
        if end_adapt.real > brk.real:
            total = total + _adaptive_segment(f6, brk, end_adapt, rtol)
            start = end_adapt
        else:
            start = brk
        total = total + _tail(f6, start, 1.0 + 0.0j, panel, rtol, np.max(np.abs(total)))
    else:
        dir_right = (h - 1.0j * rho) / r
        dir_left = (-h - 1.0j * rho) / r
        a = -0.4j * k2
        b = (0.6 + 0.2j) * k2
        c = (1.02 + 0.2j) * k2
        d = 1.3 * k2 + 0.0j
        total = _adaptive_segment(f6, a, b, rtol)
        total = total + _adaptive_segment(f6, b, c, rtol)
        total = total + _adaptive_segment(f6, c, d, rtol)
        ref = np.max(np.abs(total))
        total = total + _tail(f6, d, dir_right, panel, rtol, ref, panel0=0.5 * k2)
        total = total - _tail(f6, a, dir_left, panel, rtol, ref, panel0=0.5 * k2)

    exact = np.exp(-1j * k2 * r) / r
    return complex(total[0]), complex(exact)


# ---------------------------------------------------------------------------
# Interpolation grid (Phase 2)
# ---------------------------------------------------------------------------

_SURF_KEYS = ("IrhoV", "IzV", "IrhoH", "IphiH")


class SommerfeldGrid:
    """NEC-style bivariate interpolation grid over `iv_surfaces_direct`.

    Uniform (R₁, θ) near regions per theory-manual fig 12, spacings in
    wavelengths of k₂ (Δθ in degrees), with `r_near` = min(r1_max,
    `_SOMM_R1_NEAR_LAMBDA`·λ) and every zone split at `_SOMM_TH_SPLIT_DEG`
    = 30° so the inner zone's two bands can carry different R₁ lattices
    (momwire#443):

      1: R₁ ∈ [0, 0.2λ],      θ ∈ [0°, 30°],  ΔR₁ = 0.01λ, Δθ = 10°
      2: R₁ ∈ [0, 0.2λ],      θ ∈ [30°, 90°], ΔR₁ = 0.01λ‡, Δθ = 10°
      3: R₁ ∈ [0.2λ, r_near], θ ∈ [0°, 30°],  ΔR₁ = 0.05λ†, Δθ = 5°
      4: R₁ ∈ [0.2λ, r_near], θ ∈ [30°, 90°], ΔR₁ = 0.1λ†,  Δθ = 10°

    († capped at one sixth of the lateral-wave beat length 2π/|k₁ − k₂| —
    the manual's own caveat that grid 2 needs finer ΔR₁ for high-εr
    low-loss grounds, applied to both outer near regions.)
    (‡ keyed down to the near-interface boundary layer width 1/(4|k₁|),
    floored at 0.2λ/(`_SOMM_N0_CAP` − 1) — the momwire#443 contact fix,
    see the layout comment in `__init__`.)

    When the geometry extends past `r_near`, two coarse far regions cover
    the rest (issue #159 — the surfaces' fine lateral-wave structure has
    decayed out there, see the `_SOMM_R1_NEAR_LAMBDA` note):

      5: R₁ ∈ [r_near, r1_max], θ ∈ [0°, 30°],  ΔR₁ = 0.2λ, Δθ = 2.5°
      6: R₁ ∈ [r_near, r1_max], θ ∈ [30°, 90°], ΔR₁ = 0.2λ, Δθ = 10°

    Two modernizations vs NEC: `r1_max` is sized to the geometry that
    will query the grid (instead of a hard 1λ plus Norton asymptotics
    beyond), and the spacing keying above. Values at R₁ = 0 come from
    the analytic eqs 169–172 limits via `iv_surfaces_direct`.

    `eval(R1, theta)` interpolates all four surfaces with a 4×4 Lagrange
    (bivariate cubic) stencil, vectorized over query batches; measured
    accuracy vs direct evaluation is ~1e−4 (unit-tested at 1e−3, NEC's
    own bar). Queries must satisfy 0 ≤ R₁ ≤ r1_max (tiny overshoot is
    clamped) and 0 ≤ θ ≤ π/2.
    """

    def __init__(
        self,
        eps_t,
        k2,
        r1_max,
        rtol=1e-6,
        omega=None,
        mu=_MU0,
        cancel_flag=0,
        lattice_eps=None,
    ):
        self.eps_t = complex(eps_t)
        # The eps the LATTICE is keyed to (spacings via |k1|); the surfaces are
        # filled at `eps_t` regardless. A rung pair (momwire#902) fills both
        # masters on the lattice of its more demanding rung so the blend is
        # elementwise; a lattice keyed to a larger |k1| than the fill's own is
        # always at least as fine, never coarser. None: the fill's own eps.
        self.lattice_eps = self.eps_t if lattice_eps is None else complex(lattice_eps)
        self.k2 = float(k2)
        self.omega = k2 * _C_LIGHT if omega is None else float(omega)
        self.mu = float(mu)
        lam = 2.0 * np.pi / self.k2
        # Clamp to [0.35 lambda, cap]: never smaller than the near grid, never
        # larger than the far-pair cap (issue #157) that bounds fill cost.
        self.r1_max = min(max(float(r1_max), 0.35 * lam), _SOMM_R1_CAP_LAMBDA * lam)

        k1 = self.k2 * np.sqrt(self.lattice_eps)
        if k1.imag > 0:
            k1 = np.conj(k1)
        # Lateral-wave beat keying only matters while the interface wave
        # is a visible feature: for |k1|/k2 beyond any physical ground
        # (PEC-limit tests, |eps| ~ 1e16) the surfaces are ~1/sqrt(eps)
        # small and the keying would explode the node count — skip it.
        if abs(k1) <= 12.0 * self.k2:
            beat = 2.0 * np.pi / max(abs(k1 - self.k2), 1e-30)
        else:
            beat = np.inf

        # The near/far split (issue #159): the fine tabulation stops at
        # r_near; for grids that small it equals r1_max and the layout is
        # bit-identical to the pre-split one.
        self.r_near = min(self.r1_max, _SOMM_R1_NEAR_LAMBDA * lam)

        # Region 2's θ spacing is keyed to the NEAR extent: near grazing
        # the surfaces vary on the height scale h = R₁·sinθ, so a fixed
        # Δθ grows ever coarser in h as R₁ grows (NEC never met this —
        # its grid stopped at 1λ). Keep r_near·Δθ ≲ 0.07λ. Beyond r_near
        # the lateral-wave structure has decayed and the far regions'
        # fixed 2.5° suffices — keying to the full extent (pre-#159) grew
        # the node count ~quadratically with geometry size.
        dth2_target = min(5.0, np.degrees(0.07 * lam / self.r_near))
        n_th2 = int(np.ceil(_SOMM_TH_SPLIT_DEG / dth2_target)) + 1
        dth2 = _SOMM_TH_SPLIT_DEG / (n_th2 - 1)

        self.r_break = 0.2 * lam

        # The near zone is split at th_split like the outer zones, because
        # its two bands need different R1 lattices (momwire#443). Ground
        # contact queries the surfaces at R1 = z + z' << lambda and theta
        # near 90 deg, where they carry a boundary layer of width
        # ~1/|k1| = lambda/(2 pi sqrt|eps~|): on sea water that is 2e-3
        # lambda — entirely inside the first 0.01-lambda cell — and
        # interpolating across it cost 0.13 ohm on every shipped contact
        # answer (#282 stage 2's measurement). The steep band (region 1)
        # resolves it at ~4 nodes per layer width, capped so the
        # eps~ -> infinity decks (|k1|/k2 ~ 1e5, layer 4e-7 lambda) cannot
        # explode the node count. The grazing band (region 0) keeps the
        # 0.01-lambda spacing: there the layer variable h = R1 sin(theta)
        # stretches it out in R1, no physical deck queries small R1 at
        # grazing (two points at grazing-small R1 means both are ON the
        # plane — radial screens, which are refused), and its rows are
        # where the direct evaluator is ~100x slower, so keying them would
        # multiply the fill cost for queries nothing makes. The split
        # sits at 30 deg, not the outer zones' historical 20, because the
        # evaluator's slow contours reach past 20 deg: at 30 the steep
        # band's keyed rows are all on the fast path (sea water fills in
        # ~0.7 s against 3.6 s per slow row), and the band the slow rows
        # do cover keeps exactly its pre-#443 node budget. Grounds whose
        # layer the 0.01-lambda spacing already resolves (poor/average
        # soil, the lossless dielectric) key to 0.01 lambda exactly.
        layer = 1.0 / max(abs(k1), 1e-300)
        dr0 = min(0.01 * lam, max(layer / 4.0, self.r_break / (_SOMM_N0_CAP - 1)))
        # When the layer is unresolvable even at the cap, the R1 = 0 node
        # (the analytic eqs 169-172 limit, O(1)) poisons the first cells,
        # whose true values are the smooth small tail the layer decays to.
        # Fill that node at half a cell instead: every real query sits far
        # past the layer there, so the interpolant should follow the tail,
        # not bleed the layer top across the cells the queries do hit.
        self._r0_fill = 0.5 * dr0 if layer < 0.25 * dr0 else 0.0

        split = _SOMM_TH_SPLIT_DEG
        layout = [
            (0.0, self.r_break, 0.01 * lam, 0.0, split, 10.0),
            (0.0, self.r_break, dr0, split, 90.0, 10.0),
            (self.r_break, self.r_near, min(0.05 * lam, beat / 6.0), 0.0, split, dth2),
            (self.r_break, self.r_near, min(0.1 * lam, beat / 6.0), split, 90.0, 10.0),
        ]
        if self.r1_max > self.r_near * (1.0 + 1e-9):
            dr_far = _SOMM_DR_FAR_LAMBDA * lam
            layout += [
                (self.r_near, self.r1_max, dr_far, 0.0, split, _SOMM_DTH_FAR_DEG),
                (self.r_near, self.r1_max, dr_far, split, 90.0, 10.0),
            ]

        self._regions = []
        for reg_idx, (r0, r1, dr, th0, th1, dth) in enumerate(layout):
            n_r = max(int(np.ceil((r1 - r0) / dr)) + 1, 4)
            n_th = int(round((th1 - th0) / dth)) + 1
            r_nodes = r0 + dr * np.arange(n_r)  # last row may pad past r1
            th_nodes = np.radians(th0 + dth * np.arange(n_th))
            rr, tt = np.meshgrid(r_nodes, th_nodes, indexing="ij")
            if reg_idx == 1 and self._r0_fill > 0.0:
                # The unresolvable-layer case: node 0 keeps lattice
                # position R1 = 0 but is FILLED at half a cell (see the
                # layout comment).
                rr = rr.copy()
                rr[0, :] = self._r0_fill
            surf = iv_surfaces_direct(
                self.eps_t,
                self.k2,
                rr,
                tt,
                rtol=rtol,
                omega=self.omega,
                mu=self.mu,
                cancel_flag=cancel_flag,
            )
            vals = np.stack([surf[key] for key in _SURF_KEYS])
            self._regions.append(
                {
                    "r0": r0,
                    "dr": dr,
                    "n_r": n_r,
                    "th0": np.radians(th0),
                    "dth": np.radians(dth),
                    "n_th": n_th,
                    "vals": vals,
                }
            )

    @staticmethod
    def _lagrange4(u):
        """Cubic Lagrange weights for nodes at 0, 1, 2, 3 evaluated at u."""
        u0 = u
        u1 = u - 1.0
        u2 = u - 2.0
        u3 = u - 3.0
        return np.stack(
            [
                -u1 * u2 * u3 / 6.0,
                u0 * u2 * u3 / 2.0,
                -u0 * u1 * u3 / 2.0,
                u0 * u1 * u2 / 6.0,
            ],
            axis=-1,
        )

    def eval(self, R1, theta):
        """Interpolate the four surfaces at (R1, theta); returns a dict of
        complex arrays shaped like the broadcast inputs."""
        R1 = np.asarray(R1, dtype=float)
        theta = np.asarray(theta, dtype=float)
        r_b, th_b = np.broadcast_arrays(R1, theta)
        shape = r_b.shape
        r_f = r_b.ravel()
        th_f = np.clip(th_b.ravel(), 0.0, 0.5 * np.pi)

        # A negative R1 is a genuine bug; an R1 past r1_max is now expected —
        # a far pair beyond the grid cap (issue #157). Clamp it, matching the
        # C++ proj_one path (g keeps the true distance, surf freezes at r1_max).
        if np.any(r_f < 0.0):
            raise ValueError("query R1 must be non-negative")
        r_f = np.minimum(r_f, self.r1_max)

        th_split = np.radians(_SOMM_TH_SPLIT_DEG)
        # Near/far select: for 4-region grids r_near == r1_max, so the
        # clamped queries all land near and this reduces to the old routing.
        grazing = th_f <= th_split
        inner = np.where(grazing, 0, 1)
        near = np.where(grazing, 2, 3)
        far = np.where(grazing, 4, 5)
        region_of = np.where(
            r_f <= self.r_break, inner, np.where(r_f <= self.r_near, near, far)
        )

        out = np.empty((len(_SURF_KEYS), r_f.size), dtype=np.complex128)
        for idx, reg in enumerate(self._regions):
            sel = np.nonzero(region_of == idx)[0]
            if sel.size == 0:
                continue
            fr = (r_f[sel] - reg["r0"]) / reg["dr"]
            ft = (th_f[sel] - reg["th0"]) / reg["dth"]
            i0 = np.clip(np.floor(fr).astype(int) - 1, 0, reg["n_r"] - 4)
            j0 = np.clip(np.floor(ft).astype(int) - 1, 0, reg["n_th"] - 4)
            wr = self._lagrange4(fr - i0)  # (n, 4)
            wt = self._lagrange4(ft - j0)
            # gather the 4x4 stencils: vals (4, nR, nTh)
            ii = i0[:, None] + np.arange(4)[None, :]  # (n, 4)
            jj = j0[:, None] + np.arange(4)[None, :]
            block = reg["vals"][:, ii[:, :, None], jj[:, None, :]]  # (4, n, 4, 4)
            out[:, sel] = np.einsum("snij,ni,nj->sn", block, wr, wt)

        return {key: out[s].reshape(shape) for s, key in enumerate(_SURF_KEYS)}

    def scaled_to(self, k2, omega, mu):
        """A physical-units copy of this grid rescaled to another
        (k2, omega, mu) — the frequency-reuse view (issue #159 phase 2).

        Valid because S = ω·μ·G(ε̃; R₁/λ, θ) (ω and μ enter only through
        the linear eq-123 normalization C₁; the k₂-scaling at ω = k₂c is
        exactly linear, verified numerically) and the lattice is
        λ-proportional: lengths scale by λ_new/λ_old, values by
        (ω·μ)/(ω_old·μ_old), angles and node counts are untouched. The
        value tables are fresh scaled copies, so the source grid (a cached
        normalized master) is never mutated.
        """
        k2 = float(k2)
        scale = self.k2 / k2  # = lambda_new / lambda_old
        factor = (float(omega) * float(mu)) / (self.omega * self.mu)
        g = object.__new__(SommerfeldGrid)
        g.eps_t = self.eps_t
        g.lattice_eps = self.lattice_eps
        g.k2 = k2
        g.omega = float(omega)
        g.mu = float(mu)
        g.r1_max = self.r1_max * scale
        g.r_break = self.r_break * scale
        g.r_near = self.r_near * scale
        g._r0_fill = self._r0_fill * scale
        g._regions = [
            {
                "r0": reg["r0"] * scale,
                "dr": reg["dr"] * scale,
                "n_r": reg["n_r"],
                "th0": reg["th0"],
                "dth": reg["dth"],
                "n_th": reg["n_th"],
                "vals": reg["vals"] * factor,
            }
            for reg in self._regions
        ]
        return g

    def blend(self, other, t, eps_t):
        """(1 − t)·self + t·other on the SHARED lattice: the rung pair's
        member for the exact `eps_t` between the two rungs (momwire#902).

        Elementwise on the value tables, so the two grids must be the same
        lattice — same regions, node counts and spacings — which is what
        filling both members of a pair with `lattice_eps` set to the pair's
        demanding rung guarantees. A mismatch is refused rather than
        resampled: resampling would put a second interpolation under the
        blend. The blend carries the exact eps_t it stands for, and the
        lattice eps it was built on.
        """
        if len(self._regions) != len(other._regions):
            raise ValueError("blend: the two grids are not one lattice")
        for a, b in zip(self._regions, other._regions):
            if (
                a["n_r"] != b["n_r"]
                or a["n_th"] != b["n_th"]
                or a["r0"] != b["r0"]
                or a["dr"] != b["dr"]
                or a["th0"] != b["th0"]
                or a["dth"] != b["dth"]
            ):
                raise ValueError("blend: the two grids are not one lattice")
        t = float(t)
        g = object.__new__(SommerfeldGrid)
        g.eps_t = complex(eps_t)
        g.lattice_eps = self.lattice_eps
        g.k2 = self.k2
        g.omega = self.omega
        g.mu = self.mu
        g.r1_max = self.r1_max
        g.r_break = self.r_break
        g.r_near = self.r_near
        g._r0_fill = self._r0_fill
        g._regions = [
            {
                "r0": a["r0"],
                "dr": a["dr"],
                "n_r": a["n_r"],
                "th0": a["th0"],
                "dth": a["dth"],
                "n_th": a["n_th"],
                "vals": (1.0 - t) * a["vals"] + t * b["vals"],
            }
            for a, b in zip(self._regions, other._regions)
        ]
        return g


def grid_cpp_args(grid):
    """Flatten a `SommerfeldGrid` into the positional args the C++ remainder
    kernels take after (ground_z, k): (r1_max, r_break, th_split, r_near,
    reg_r0, reg_dr, reg_th0, reg_dth, reg_vals). The four (near-only grids)
    or six (with the #159 far zone) region value tables are made
    C-contiguous complex128 once; callers that sample the same grid many
    times (the ACA path) should hoist this out of their loop.
    """
    regs = grid._regions
    reg_vals = [np.ascontiguousarray(r["vals"], dtype=np.complex128) for r in regs]
    return (
        float(grid.r1_max),
        float(grid.r_break),
        float(math.radians(_SOMM_TH_SPLIT_DEG)),
        float(grid.r_near),
        np.array([r["r0"] for r in regs], dtype=float),
        np.array([r["dr"] for r in regs], dtype=float),
        np.array([r["th0"] for r in regs], dtype=float),
        np.array([r["dth"] for r in regs], dtype=float),
        reg_vals,
    )


def remainder_field_proj(obs, t_obs, src, t_src, ground_z, k, grid, cancel_flag=0):
    """Projected smooth-remainder field table t_m · F(r_m, r_n) · t_n.

    The theory-manual eqs 143-147 azimuth combination of the four grid
    surfaces: per (observer point m, source point n), decompose the
    source tangent into vertical + horizontal parts, combine the
    interpolated surfaces with the incidence-azimuth factors, and
    project the resulting E-field on the observer tangent. F is the
    field of a unit current MOMENT (Il = 1, eq 123 normalization), so
    quadrature callers weight rows/columns by their own basis shapes
    and dz measures. Shared by the bspline Galerkin remainder block,
    the sinusoidal remainder tensor, and the fast solvers' rectangular
    remainder sampler — one home for the dyad algebra.

    obs (M, 3) / t_obs (M, 3), src (S, 3) / t_src (S, 3); returns
    (M, S) complex. Callers chunk the observer axis to bound the
    working set (four surfaces x M x S complexes live at once).

    Routes through the C++ accelerator (`remainder_field_proj_batch`,
    the Phase-4b fused interpolate+project kernel, OpenMP over observer
    rows) when it is loaded — this assembly is ~90% of a Sommerfeld
    solve — and falls back to the vectorized numpy body otherwise. Both
    paths poll `cancel_flag` (raw int32 address; 0 = no cancellation).
    """
    if _acc is not None and hasattr(_acc, "remainder_field_proj_batch"):
        return _acc.remainder_field_proj_batch(
            obs,
            t_obs,
            src,
            t_src,
            float(ground_z),
            float(k),
            *grid_cpp_args(grid),
            int(cancel_flag),
        )

    th_src = np.hypot(t_src[:, 0], t_src[:, 1])
    safe_t = th_src > 1e-12
    ux = np.where(safe_t, t_src[:, 0] / np.where(safe_t, th_src, 1.0), 1.0)
    uy = np.where(safe_t, t_src[:, 1] / np.where(safe_t, th_src, 1.0), 0.0)
    tz_src = t_src[:, 2]

    dx = obs[:, 0][:, None] - src[:, 0][None, :]
    dy = obs[:, 1][:, None] - src[:, 1][None, :]
    rho = np.hypot(dx, dy)
    hh = (obs[:, 2] - ground_z)[:, None] + (src[:, 2] - ground_z)[None, :]
    r1 = np.sqrt(rho * rho + hh * hh)
    surf = grid.eval(r1, np.arctan2(hh, rho))
    g = np.exp(-1j * k * r1) / r1

    tiny = 1e-12 * grid.r1_max
    safe_r = rho > tiny
    inv_rho = np.where(safe_r, 1.0 / np.where(safe_r, rho, 1.0), 0.0)
    # rho -> 0: the incidence azimuth degenerates; I_rho^H(90 deg)
    # = -I_phi^H there (unit-tested in the engine suite), so any
    # d-hat works — use the source horizontal direction.
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


# ---------------------------------------------------------------------------
# Module-level grid caches (shared by every solver that consumes the grid)
# ---------------------------------------------------------------------------
#
# Grid fills cost seconds while the grids themselves are a few hundred kB,
# and the engine wrappers build a fresh solver per impedance() call — an
# instance cache never survives an interactive knob-turn. Two levels
# (issue #159 phase 2):
#
#   _NORM_CACHE — the expensive artifacts: normalized masters filled at
#     k2 = _K2_REF (lambda_ref = 1), keyed (ladder rung, r1_wl_bucket,
#     lattice rung). Frequency-independent: one rung PAIR serves a whole
#     band sweep (momwire#902).
#   _GRID_CACHE — cheap physical-units views (the pair's blend at the exact
#     eps_t, then `SommerfeldGrid.scaled_to`: an elementwise blend, a
#     coordinate scale and one scalar multiply, ~sub-ms), keyed
#     (eps_t, k2, r1_wl_bucket, omega, mu) so repeat solves at one
#     frequency return the identical object, as before.
#
# `r1_max` is bucketed UP in ~25% geometric steps (in wavelengths) before
# keying: a grid tabulated to a larger radius is valid (and marginally
# finer in theta) for any smaller one, so nearby geometries (knob turns)
# share one fill. Im(eps_t) is quantized onto the _SOMM_EPS_IM_BUCKET
# geometric ladder — see the calibration note at the constant. Hoisted
# here from bspline.py so SinusoidalSolver and the fast solvers hit the
# same caches (docs/sommerfeld-everywhere-plan.md Phase 1).
_GRID_CACHE: dict = {}
_GRID_CACHE_MAX = 128
_NORM_CACHE: dict = {}
_NORM_CACHE_MAX = 32


def _evict_fifo(cache: dict, limit: int) -> None:
    while len(cache) >= limit:
        cache.pop(next(iter(cache)))


def _somm_r1_bucket_wl(r1_wl: float) -> float:
    """Round a radius in wavelengths up to the next 1.25^n (floor 0.1)."""
    x = max(float(r1_wl), 0.1)
    n = math.ceil(math.log(x, 1.25) - 1e-12)
    bucket = 1.25**n
    if bucket < r1_wl:  # float fuzz at an exact bucket edge
        bucket *= 1.25
    return float(bucket)


def _somm_r1_bucket(r1_max: float, k: float) -> float:
    """Round `r1_max` up to the next 1.25^n wavelengths (floor 0.1 wl)."""
    lam = 2.0 * np.pi / k
    return lam * _somm_r1_bucket_wl(r1_max / lam)


def _somm_eps_bucket(eps_t: complex) -> complex:
    """Quantize Im(eps_t) onto the _SOMM_EPS_IM_BUCKET geometric ladder.

    Re is keyed exactly (it does not move with frequency and is ~8x more
    sensitive). Nonstandard values — free space, nonpassive Im > 0,
    Re <= 0, or a disabled ladder — pass through exactly.
    """
    step = 1.0 + _SOMM_EPS_IM_BUCKET
    if step <= 1.0:
        return eps_t
    re, im = eps_t.real, eps_t.imag
    if not (re > 0.0) or im >= 0.0:  # lossless (im == 0) included: exact
        return eps_t
    n = round(math.log(-im, step))
    return complex(re, -(step**n))


def _somm_eps_rungs(eps_t: complex):
    """The ladder rungs bracketing Im(eps_t) and the blend weight: (lo, hi, t)
    with Im(lo) ≤ Im(eps_t) ≤ Im(hi) in magnitude and
    eps_t = (1 − t)·lo + t·hi in Im. On a rung, for a disabled ladder, or
    for the nonstandard values `_somm_eps_bucket` passes through, lo == hi
    == the exact value and t = 0: one master, no blend (momwire#902).
    """
    eps_t = complex(eps_t)
    step = 1.0 + _SOMM_EPS_IM_BUCKET
    re, im = eps_t.real, eps_t.imag
    if step <= 1.0 or not (re > 0.0) or im >= 0.0:
        return eps_t, eps_t, 0.0
    x = math.log(-im, step)
    n = math.floor(x)
    lo_im, hi_im = step**n, step ** (n + 1)
    # Float fuzz at an exact rung: read it as the rung, from either side.
    if abs(-im - lo_im) <= 1e-12 * lo_im:
        return complex(re, -lo_im), complex(re, -lo_im), 0.0
    if abs(-im - hi_im) <= 1e-12 * hi_im:
        return complex(re, -hi_im), complex(re, -hi_im), 0.0
    t = (-im - lo_im) / (hi_im - lo_im)
    return complex(re, -lo_im), complex(re, -hi_im), float(t)


def _norm_master(rung, r1b_wl, lattice_eps, cancel_flag=0):
    """The normalized master for one ladder rung on one lattice, cached in
    `_NORM_CACHE` under (rung, r1 bucket, lattice rung). A cancelled fill
    raises SolveAborted out of the constructor before the insert."""
    nkey = ("above", rung, r1b_wl, lattice_eps)
    master = _NORM_CACHE.get(nkey)
    if master is None:
        _evict_fifo(_NORM_CACHE, _NORM_CACHE_MAX)
        master = SommerfeldGrid(
            rung,
            _K2_REF,
            r1b_wl,  # lambda_ref = 1: wavelengths ARE physical units
            omega=_K2_REF * _C_LIGHT,
            mu=_MU0,
            cancel_flag=cancel_flag,
            lattice_eps=lattice_eps,
        )
        _NORM_CACHE[nkey] = master
    return master


def get_grid(eps_t, k2, r1_max, omega, mu=_MU0, cancel_flag=0):
    """Cached `SommerfeldGrid` for (eps_t, k2, r1_max, omega, mu).

    Two-level FIFO-bounded module cache: frequency-independent normalized
    masters (filled once per (ladder rung, r1-bucket, lattice rung)) and a
    cheap per-(eps_t, k2, omega, mu) view — the rung pair's blend at the
    exact eps_t, rescaled — see the cache note above. A cancelled fill
    raises SolveAborted out of the constructor before either cache insert,
    so no partial grid is ever cached.
    """
    # Cap (in wavelengths, #157) before bucketing so every geometry beyond
    # the cap keys to the same capped grid instead of minting a distinct
    # oversized cache entry; the sweep slack (#902) goes on before the cap.
    k2 = float(k2)
    lam = 2.0 * np.pi / k2
    r1_wl = float(r1_max) / lam * (1.0 + _SOMM_R1_SWEEP_SLACK)
    r1b_wl = _somm_r1_bucket_wl(min(r1_wl, _SOMM_R1_CAP_LAMBDA))
    eps_t = complex(eps_t)
    lo, hi, t = _somm_eps_rungs(eps_t)
    # The leading regime discriminator (momwire#553 U2): the below/below
    # family tabulates a DIFFERENT surface family, over a different
    # divide-out, on a lattice keyed to a different wavelength — and it
    # keys on the same (eps_t, k2, r1, omega, mu). Without the tag the two
    # regimes collide in this dict and whichever asked first would serve
    # both. `_sommerfeld_below.get_grid_below` writes ("below", ...) keys
    # into this same cache rather than forking it, so there is one FIFO
    # bound and one eviction rule. The view is keyed on the EXACT eps_t
    # when it is a blend and on the rung when it is one (momwire#902).
    key = ("above", lo if lo == hi else eps_t, k2, r1b_wl, float(omega), float(mu))
    grid = _GRID_CACHE.get(key)
    if grid is None:
        # The pair's lattice is its demanding rung's (the larger |Im|).
        master = _norm_master(lo, r1b_wl, hi, cancel_flag)
        if lo != hi:
            other = _norm_master(hi, r1b_wl, hi, cancel_flag)
            master = master.blend(other, t, eps_t)
        _evict_fifo(_GRID_CACHE, _GRID_CACHE_MAX)
        grid = master.scaled_to(k2, omega, mu)
        _GRID_CACHE[key] = grid
    return grid
