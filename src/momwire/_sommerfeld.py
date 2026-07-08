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

import numpy as np
from scipy.special import hankel2, jv

from ._accel import acc as _acc
from ._cancel import SolveAborted

_C_LIGHT = 299792458.0
_MU0 = 4e-7 * np.pi

# Gauss–Legendre rule shared by all contour sections.
_GAUSS_N = 24
_GX, _GW = np.polynomial.legendre.leggauss(_GAUSS_N)

# Recursion cap for the adaptive sections (branch-point neighborhoods).
_ADAPT_DEPTH = 14


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
    if 1.01 * k1.real <= kcap:
        d = 1.01 * k1.real + 0.99j * max(k1.imag, -kcap)
    else:
        d = kcap + 0.0j
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

    Three uniform (R₁, θ) regions per theory-manual fig 12, spacings in
    wavelengths of k₂ (Δθ in degrees):

      1: R₁ ∈ [0, 0.2λ],      θ ∈ [0°, 90°], ΔR₁ = 0.01λ, Δθ = 10°
      2: R₁ ∈ [0.2λ, r1_max], θ ∈ [0°, 20°], ΔR₁ = 0.05λ†, Δθ = 5°
      3: R₁ ∈ [0.2λ, r1_max], θ ∈ [20°, 90°], ΔR₁ = 0.1λ†,  Δθ = 10°

    († capped at one sixth of the lateral-wave beat length 2π/|k₁ − k₂| —
    the manual's own caveat that grid 2 needs finer ΔR₁ for high-εr
    low-loss grounds, applied to both outer regions.)

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
        self, eps_t, k2, r1_max, rtol=1e-6, omega=None, mu=_MU0, cancel_flag=0
    ):
        self.eps_t = complex(eps_t)
        self.k2 = float(k2)
        self.omega = k2 * _C_LIGHT if omega is None else float(omega)
        self.mu = float(mu)
        lam = 2.0 * np.pi / self.k2
        self.r1_max = max(float(r1_max), 0.35 * lam)

        k1 = self.k2 * np.sqrt(self.eps_t)
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

        # Region 2's θ spacing is keyed to the grid extent: near grazing
        # the surfaces vary on the height scale h = R₁·sinθ, so a fixed
        # Δθ grows ever coarser in h as R₁ grows (NEC never met this —
        # its grid stopped at 1λ). Keep r1_max·Δθ ≲ 0.07λ.
        dth2_target = min(5.0, np.degrees(0.07 * lam / self.r1_max))
        n_th2 = int(np.ceil(20.0 / dth2_target)) + 1
        dth2 = 20.0 / (n_th2 - 1)

        self._regions = []
        r_break = 0.2 * lam
        for r0, r1, dr, th0, th1, dth in (
            (0.0, r_break, 0.01 * lam, 0.0, 90.0, 10.0),
            (r_break, self.r1_max, min(0.05 * lam, beat / 6.0), 0.0, 20.0, dth2),
            (r_break, self.r1_max, min(0.1 * lam, beat / 6.0), 20.0, 90.0, 10.0),
        ):
            n_r = max(int(np.ceil((r1 - r0) / dr)) + 1, 4)
            n_th = int(round((th1 - th0) / dth)) + 1
            r_nodes = r0 + dr * np.arange(n_r)  # last row may pad past r1
            th_nodes = np.radians(th0 + dth * np.arange(n_th))
            rr, tt = np.meshgrid(r_nodes, th_nodes, indexing="ij")
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

        if np.any(r_f < 0.0) or np.any(r_f > self.r1_max * (1.0 + 1e-9) + 1e-12):
            raise ValueError("query R1 outside [0, r1_max] of this grid")
        r_f = np.minimum(r_f, self.r1_max)

        r_break = self._regions[1]["r0"]
        th_split = np.radians(20.0)
        region_of = np.where(r_f <= r_break, 0, np.where(th_f <= th_split, 1, 2))

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


def remainder_field_proj(obs, t_obs, src, t_src, ground_z, k, grid):
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
    """
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
    e_z = g * (
        tz_src[None, :] * surf["IzV"] - th_src[None, :] * cphi * surf["IrhoV"]
    )
    return (
        t_obs[:, 0][:, None] * (dhx * e_rho - dhy * e_phi)
        + t_obs[:, 1][:, None] * (dhy * e_rho + dhx * e_phi)
        + t_obs[:, 2][:, None] * e_z
    )


# ---------------------------------------------------------------------------
# Module-level grid cache (shared by every solver that consumes the grid)
# ---------------------------------------------------------------------------
#
# Grid fills cost seconds while the grids themselves are a few tens of
# kB, and the engine wrappers build a fresh solver per impedance() call —
# an instance cache never survives an interactive knob-turn. `r1_max` is
# bucketed UP in ~25% geometric steps before keying: a grid tabulated to
# a larger radius is valid (and marginally finer in theta) for any
# smaller one, so nearby geometries share one fill instead of each
# paying seconds. The bound covers a full web sweep (one entry per k;
# ~21-41 points) plus a couple of ground choices, so a knob-turn that
# re-runs the same sweep hits every entry. Hoisted here from bspline.py
# so SinusoidalSolver and the fast solvers hit the same cache
# (docs/sommerfeld-everywhere-plan.md Phase 1).
_GRID_CACHE: dict = {}
_GRID_CACHE_MAX = 128


def _evict_fifo(cache: dict, limit: int) -> None:
    while len(cache) >= limit:
        cache.pop(next(iter(cache)))


def _somm_r1_bucket(r1_max: float, k: float) -> float:
    """Round `r1_max` up to the next 1.25^n wavelengths (floor 0.1 wl)."""
    lam = 2.0 * np.pi / k
    x = max(r1_max / lam, 0.1)
    n = math.ceil(math.log(x, 1.25) - 1e-12)
    bucket = lam * 1.25**n
    if bucket < r1_max:  # float fuzz at an exact bucket edge
        bucket *= 1.25
    return float(bucket)


def get_grid(eps_t, k2, r1_max, omega, mu=_MU0, cancel_flag=0):
    """Cached `SommerfeldGrid` keyed `(eps_t, k2, r1_bucket, omega, mu)`.

    FIFO-bounded module cache. A cancelled fill raises SolveAborted out
    of the constructor before the cache insert, so no partial grid is
    ever cached.
    """
    r1b = _somm_r1_bucket(float(r1_max), float(k2))
    key = (complex(eps_t), float(k2), r1b, float(omega), float(mu))
    grid = _GRID_CACHE.get(key)
    if grid is None:
        _evict_fifo(_GRID_CACHE, _GRID_CACHE_MAX)
        grid = SommerfeldGrid(
            eps_t, k2, r1b, omega=omega, mu=mu, cancel_flag=cancel_flag
        )
        _GRID_CACHE[key] = grid
    return grid
