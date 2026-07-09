"""Reflection-coefficient finite ground: per-pair Fresnel weight tables.

NEC's IPERF=0 "finite ground, reflection coefficient approximation"
(gn_card(0)) weights each image-field contribution by the Fresnel plane-wave
reflection coefficients evaluated at the specular angle of the (image source
midpoint → observer midpoint) ray, treating the coefficients as constant over
a segment pair. This module builds those per-segment-pair weight tables for
the mixed-potential solvers; the quadrature kernels (J moment tensors over
mirrored segments) are untouched.

Field-weighting dyad (matches NEC-2's EFLD, not the transverse v̂v̂+ĥĥ form):
with p̂ the horizontal unit vector normal to the plane of incidence, the
image field components in the incidence plane (vertical z and horizontal
in-plane) are scaled by ρ_v, and the horizontal out-of-plane (p̂) component
by −ρ_h. In momwire's image convention — image block assembled with mirrored
tangents M·t = (tx, ty, −tz) and subtracted with one global minus sign — the
PEC limit ε̃ → ∞ gives ρ_v → 1, ρ_h → −1, the dyad → identity, and the
weighted image reduces exactly to the existing PEC-image build.

Sign conventions: e^{+jωt} time dependence (momwire's exp(−jkR)/4πR kernel),
so ε̃ = εr − j·σ/(ω·ε₀) and Im(ε̃) ≤ 0 for passive ground.

All tables are dense (N_seg, N_seg) numpy arrays — O(N²) small-constant
geometry, negligible next to the J-block quadrature fill.
"""

import numpy as np

# Guard for coincident observer/image points (a segment pair lying *in* the
# ground plane). Real geometries sit above ground so rmag ≥ 2·min height;
# this only prevents 0/0 from degenerate input.
_TINY = 1e-30


def eps_tilde(ground_eps, omega, eps0):
    """Complex relative permittivity ε̃ from a `ground_eps` solver spec.

    `ground_eps` is either a complex ε̃ directly (Im ≤ 0 expected for a
    passive ground in the e^{+jωt} convention) or an (eps_r, sigma) tuple
    with sigma in S/m, folded as ε̃ = εr − j·σ/(ω·ε₀).
    """
    if isinstance(ground_eps, (tuple, list)):
        if len(ground_eps) != 2:
            raise ValueError(
                f"ground_eps tuple must be (eps_r, sigma), got {ground_eps!r}"
            )
        eps_r, sigma = ground_eps
        return complex(float(eps_r), -float(sigma) / (omega * eps0))
    return complex(ground_eps)


def fresnel_rho(eps_t, cos_th):
    """Fresnel plane-wave reflection coefficients ρ_v (TM, in-plane) and
    ρ_h (TE, out-of-plane) at incidence angle θ from the vertical.

    Vectorized over `cos_th` (any shape). Principal-branch sqrt of
    ε̃ − sin²θ has Im ≤ 0 for Im(ε̃) ≤ 0 — the decaying-transmitted-wave
    branch, matching NEC.
    """
    cos_th = np.asarray(cos_th)
    sin2 = 1.0 - cos_th * cos_th
    root = np.sqrt(eps_t - sin2)
    rho_v = (eps_t * cos_th - root) / (eps_t * cos_th + root)
    rho_h = (cos_th - root) / (cos_th + root)
    return rho_v, rho_h


def specular_ray_tables(centers, ground_z, src_centers=None):
    """Per-pair specular-ray geometry: incidence-angle cosine and the
    horizontal unit vector p̂ normal to the plane of incidence.

    For observer segment m (midpoint r_m) and source segment n imaged
    across z = ground_z: the specular ray runs from the image midpoint
    r'_n = (x_n, y_n, 2·ground_z − z_n) to r_m. Square by default
    (sources = observers); pass `src_centers` (REAL, unmirrored source
    midpoints) for a rectangular observer×source block.

    Returns (cos_th, px, py), each (N_obs, N_src):
      cos_th : cosine of the incidence angle (ray direction · ẑ);
      px, py : components of p̂_mn = ẑ × d̂_horizontal (p̂_z ≡ 0). For a
               near-vertical ray the incidence plane is undefined; p̂
               falls back to x̂, which is harmless because ρ_v = −ρ_h at
               θ = 0 makes the field dyad isotropic there.
    """
    c = np.asarray(centers, dtype=float)
    cs = c if src_centers is None else np.asarray(src_centers, dtype=float)

    dx = c[:, 0][:, None] - cs[:, 0][None, :]
    dy = c[:, 1][:, None] - cs[:, 1][None, :]
    dz = c[:, 2][:, None] + cs[:, 2][None, :] - 2.0 * ground_z

    hyp = np.hypot(dx, dy)
    rmag = np.sqrt(dx * dx + dy * dy + dz * dz)
    cos_th = dz / np.maximum(rmag, _TINY)

    # p̂ = (−dy, dx, 0)/hyp; degenerate (vertical ray) → x̂.
    safe = hyp > _TINY
    inv_hyp = np.where(safe, 1.0 / np.where(safe, hyp, 1.0), 1.0)
    px = np.where(safe, -dy * inv_hyp, 1.0)
    py = np.where(safe, dx * inv_hyp, 0.0)

    return cos_th, px, py


def specular_pair_tables(
    centers, tangents, ground_z, src_centers=None, src_tangents=None
):
    """Frequency-independent per-pair specular geometry.

    For observer segment m (midpoint r_m, unit tangent t_m) and source
    segment n imaged across z = ground_z: the specular ray runs from the
    image midpoint r'_n = (x_n, y_n, 2·ground_z − z_n) to r_m.

    Square by default (sources = observers). Pass `src_centers` /
    `src_tangents` (REAL, unmirrored source geometry) for a rectangular
    observer×source block — the fast solvers' per-block fills use this.

    Returns (cos_th, td_img, P), each (N_obs, N_src):
      cos_th : cosine of the incidence angle (ray direction · ẑ);
      td_img : t_m · M·t_n, M = diag(1, 1, −1) — today's PEC mirror table;
      P      : (t_m · p̂_mn)(t_n · p̂_mn), the out-of-plane horizontal dyad
               component, with p̂_mn = ẑ × d̂_horizontal. For a near-vertical
               ray the incidence plane is undefined; p̂ falls back to x̂,
               which is harmless because ρ_v = −ρ_h at θ = 0 makes the dyad
               isotropic there.
    """
    t = np.asarray(tangents, dtype=float)
    ts = t if src_tangents is None else np.asarray(src_tangents, dtype=float)

    cos_th, px, py = specular_ray_tables(centers, ground_z, src_centers)

    # (t_m · p̂_mn) and (t_n · p̂_mn); p̂ has no z-component so M·t_n · p̂ =
    # t_n · p̂.
    tm_p = t[:, 0][:, None] * px + t[:, 1][:, None] * py
    tn_p = ts[:, 0][None, :] * px + ts[:, 1][None, :] * py
    P = tm_p * tn_p

    td_img = t @ (ts * np.array([1.0, 1.0, -1.0])).T

    return cos_th, td_img, P


def a_term_weights(rho_v, rho_h, td_img, P):
    """A-term per-pair weight table replacing the PEC mirror tangent-dot:
    w_A = ρ_v·(td_img − P) − ρ_h·P (NEC field dyad applied to the image
    vector potential's direction, projected on the observer tangent).
    """
    return rho_v * (td_img - P) - rho_h * P


# Φ-term (image charge) weighting candidates — the one place the mixed-
# potential form cannot copy NEC's field-level weighting exactly; see
# docs/refl-coef-ground-plan.md Phase 1.
PHI_MODES = ("rho_v", "image", "normal", "blend")


def phi_term_weights(mode, eps_t, rho_v):
    """Per-pair (or scalar-broadcast) weight for the image charge term.

    "rho_v"  : ρ_v(θ_mn) at the pair's specular angle (plane-wave TM);
    "image"  : quasi-static half-space image coefficient (ε̃−1)/(ε̃+1);
    "normal" : ρ_v at normal incidence, (√ε̃−1)/(√ε̃+1);
    "blend"  : mean of "rho_v" and "image".

    All → 1 in the PEC limit ε̃ → ∞.

    "normal" is the solver default — the Phase 1 evaluation
    (scripts/compare_refl_coef_ground.py against the PyNEC gn 0 goldens)
    measured, over the dipole 0.1–0.5λ acceptance window,
    max |ΔZ| / mean |ΔZ| of: normal 2.45/1.75 Ω, rho_v 6.97/2.87 Ω,
    blend 7.13/3.27 Ω, image 11.70/4.82 Ω (PEC image: 41.2/19.4 Ω), with
    the momwire-vs-NEC cross-solver floor itself ≈ 1.4 Ω. Per-pair
    specular ρ_v loses because grazing-angle pairs push ρ_v → −1, which
    is unphysical for the quasi-static-dominated charge interaction;
    the full quasi-static coefficient overweights instead. The θ=0
    plane-wave value sits between and tracks the oracle best.
    """
    if mode == "rho_v":
        return rho_v
    if mode == "image":
        return (eps_t - 1.0) / (eps_t + 1.0)
    if mode == "normal":
        r = np.sqrt(eps_t)
        return (r - 1.0) / (r + 1.0)
    if mode == "blend":
        return 0.5 * (rho_v + (eps_t - 1.0) / (eps_t + 1.0))
    raise ValueError(f"unknown ground_phi_mode {mode!r}; expected one of {PHI_MODES}")


def phi_mode_coeffs(mode, eps_t):
    """Every Φ mode as w_Φ = c0 + c1·ρ_v(θ) with complex constants (c0, c1).

    This is the form the C++ fused off-edge assembler consumes (it computes
    ρ_v per segment pair in-kernel anyway for the A-term dyad, so per-pair Φ
    modes cost nothing extra there). Must stay consistent with
    `phi_term_weights` — guarded by a test.
    """
    if mode == "rho_v":
        return complex(0.0), complex(1.0)
    if mode == "image":
        return (eps_t - 1.0) / (eps_t + 1.0), complex(0.0)
    if mode == "normal":
        r = np.sqrt(eps_t)
        return (r - 1.0) / (r + 1.0), complex(0.0)
    if mode == "blend":
        return 0.5 * (eps_t - 1.0) / (eps_t + 1.0), complex(0.5)
    raise ValueError(f"unknown ground_phi_mode {mode!r}; expected one of {PHI_MODES}")
