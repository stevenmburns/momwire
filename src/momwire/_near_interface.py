"""Designed near-interface evaluation of the transmitted family — the
crossing serve's kernel layer (momwire#524 phase 2).

The six scalars {U_T, V_T, W_T, ∂zW_T, ∂z∂z′V_T, ∂z′W_T} evaluated at ONE
(ρ, z, z′) each, with the corner z, z′ → 0, ρ → 0 reached by DESIGN, not
clamp (`scratch` derivation `DERIVATION-NEAR-INTERFACE.md`, pinned by its
probes 21–22 to machine class including the corner):

  head [0, 1.1K]   : the shipped first-quadrant detour (`_head`) — branch
                     points + transmitted pole handled as production does;
  mid  [1.1K, 8K]  : real-axis adaptive Gauss (smooth, pole/cut-free);
  tail [8K, ∞)     : rotated rays λ = Λ + t·e^{±jπ/4}. ρ = 0 uses one
                     up-ray (J₀ = 1); ρ > 0 splits J₀ into Hankel halves,
                     H1 up / H2 down. |integrand| ~ e^{−t(s+ρ)/√2} —
                     uniform through the corner, z′ = 0 and z = 0 exact.

Two scope guards ride the tail (both measured on the high-σ adjudication
ladder, neither moving anything pinned):

  * exact-underflow ray panels count as QUIET — at σ-class |k_m| the whole
    tail can underflow e^{−1330} to exactly 0.0 for far pairs, and an
    all-zero panel is the tail being zero, not a stall;
  * the far-pair kill cap λ ≤ 60/s — beyond it the integrand is e^{−60}
    of the total, dead range the adaptive head/mid would otherwise grind
    at full depth. Inactive for every near-interface pair.

This module deliberately has no grid: every value is a direct contour
evaluation, O(ms)/point, because the crossing fill's pair count is small
and the corner cannot be interpolated from a grid that must exclude it.

Convention gate: e^{+jωt}, ε̃ = ε_r − jσ/ωε₀, asserted at import.
"""

from __future__ import annotations

import numpy as np
from scipy.special import hankel1, hankel2

from ._sommerfeld_below import _adaptive_segment, _head
from ._sommerfeld_transmitted import (
    _ADAPT_DEPTH,
    _DETOUR,
    _GW,
    _GX,
    _bessel_j0_j1x,
    _gamma,
    k_medium,
)

KEYS = ("U", "V", "W", "dzW", "dzpV", "dzpW")
_LAM_MULT = 8.0
_RAY = np.exp(1j * np.pi / 4.0)
_MAX_RAY_PANELS = 90
_FAR_PAIR_KILL = 60.0

# --- convention gate (e^{+jωt}: the lossy k_m must make e^{−jk_m R} decay) --
_kp_gate = 2.0 * np.pi / 42.831
_km_gate = k_medium(13.0 - 12.84j, _kp_gate)
assert _km_gate.imag < 0.0, "e^{+j omega t} broken: e^{-j k_m R} must decay"
assert abs(np.exp(-1j * _km_gate * 5.0) / 5.0) < abs(
    np.exp(-1j * _km_gate * 1.0) / 1.0
), "lossy-medium decay gate failed"


def _core(lam, z, zp, k_p, k_m):
    """The six spectral factors × (2 Ẽ λ), WITHOUT the Bessel factor.

    z′ ≤ 0, so e^{−γ_m |z′|} = e^{+γ_m z′}. Stacked (6, n):
    0 U, 1 V, 2 W, 3 ∂zW (= −γ₊W̃ under the integral), 4 ∂z∂z′V (= −γ₊γ₋Ṽ),
    5 ∂z′W (= +γ₋W̃). Derivative bookkeeping: ∂z ↔ −γ₊, ∂z′ ↔ +γ₋.
    """
    lam = np.asarray(lam, dtype=np.complex128)
    g_p = _gamma(lam, k_p)
    g_m = _gamma(lam, k_m)
    e = 2.0 * np.exp(g_m * zp - g_p * z) * lam
    u = e / (g_p + g_m)
    v = e / (k_m * k_m * g_p + k_p * k_p * g_m)
    w = (g_p - g_m) * v
    return np.stack([u, v, w, -g_p * w, -g_p * g_m * v, g_m * w])


def _ray_integral(f_core, factor, lam0, direction, scale, rtol):
    """∫ f_core(λ)·factor(λ) over λ = λ0 + t·direction, t ∈ [0, ∞).
    Geometric panels, each adaptive Gauss; stops when two consecutive
    panels contribute < rtol of the running total.

    Two length scales coexist on the ray: the 1/λ (log-family) structure
    at scale ~λ0 near t = 0, and the e^{−t(s+ρ)/√2} decay at `scale`.
    Panels START at the λ0 scale and double toward the decay scale —
    starting at the decay scale under-resolves the log content when s + ρ
    is tiny (measured: 44 % on dW/dln s at s = 1e-5).
    """

    def ft(t):
        t = np.asarray(t, dtype=float)
        lam = lam0 + t * direction
        return f_core(lam) * factor(lam) * direction

    acc = None
    t_lo = 0.0
    step = min(0.25 * scale, lam0)
    quiet = 0
    for _ in range(_MAX_RAY_PANELS):
        t_hi = t_lo + step
        part = _adaptive_segment(ft, t_lo, t_hi, rtol, _ADAPT_DEPTH, _GX, _GW)
        acc = part if acc is None else acc + part
        ref = float(np.max(np.abs(acc)))
        # ref == 0.0: the whole ray underflows to exact 0 (high-σ far
        # pairs); consecutive all-zero panels are quiet, the tail IS zero.
        if ref == 0.0 or float(np.max(np.abs(part))) < rtol * ref:
            quiet += 1
            if quiet >= 2:
                return acc if acc is not None else part
        else:
            quiet = 0
        t_lo = t_hi
        step *= 2.0
    raise RuntimeError("rotated tail did not go quiet inside the panel budget")


def six_point(eps_t, k2, rho, z, zp, rtol=1e-10, lam_mult=_LAM_MULT):
    """The six designed integrals at ONE (ρ, z, z′), z ≥ 0 ≥ z′,
    R = hypot(ρ, z − z′) > 0. Returns (6,) complex."""
    k_p = float(k2)
    k_m = k_medium(complex(eps_t), k_p)
    rho, z, zp = float(rho), float(z), float(zp)
    if not (z >= 0.0 and zp <= 0.0):
        raise ValueError(f"need z >= 0 >= zp, got {(z, zp)!r}")
    s = z - zp
    if rho < 0.0 or s + rho <= 0.0:
        raise ValueError(f"need R > 0, got rho={rho!r}, s={s!r}")

    kk = max(k_p, abs(k_m))
    a_head = 1.1 * kk
    lam_top = lam_mult * kk
    # Far-pair kill cap (σ = 5 class, |k_m| ≫ k_p): beyond λ ~ 60/s the
    # integrand is e^{−60} of the total — dead range. Cap the extents
    # there, keeping the k_p branch point + transmitted pole (|λ_p| ~ k_p)
    # inside the head. Inactive for s ≤ 60/λ_top — every near-interface
    # pair — so nothing the corner probes pinned changes.
    if s > 0.0 and _FAR_PAIR_KILL / s < lam_top:
        lam_kill = _FAR_PAIR_KILL / s
        a_head = max(2.2 * k_p, min(a_head, lam_kill))
        lam_top = max(1.5 * a_head, lam_kill)

    def f_core(lam):
        return _core(lam, z, zp, k_p, k_m)

    def f_j0(lam):
        b0, _ = _bessel_j0_j1x(lam * rho)
        return f_core(lam) * b0

    head, _hp = _head(
        f_j0,
        a_head,
        rho,
        (k_p, abs(k_m.real)),
        rtol,
        _ADAPT_DEPTH,
        _DETOUR,
        _GX,
        _GW,
    )
    mid = _adaptive_segment(f_j0, a_head, lam_top, rtol, _ADAPT_DEPTH, _GX, _GW)

    scale = np.sqrt(2.0) / (s + rho)
    if rho == 0.0:
        tail = _ray_integral(f_core, lambda lam: 1.0, lam_top, _RAY, scale, rtol)
    else:
        up = _ray_integral(
            f_core,
            lambda lam: 0.5 * hankel1(0, lam * rho),
            lam_top,
            _RAY,
            scale,
            rtol,
        )
        dn = _ray_integral(
            f_core,
            lambda lam: 0.5 * hankel2(0, lam * rho),
            lam_top,
            np.conj(_RAY),
            scale,
            rtol,
        )
        tail = up + dn
    return head + mid + tail


def designed_tables(eps_t, k2, rho, z, zp, rtol=1e-10, lam_mult=_LAM_MULT):
    """Broadcast wrapper over `six_point`. Accepts z′ = 0 and z = 0
    exactly (no clamp); refuses only R = 0. Returns dict over `KEYS`.

    Duplicate (ρ, z, z′) triples are evaluated ONCE per call (ε̃, k₂,
    rtol are fixed inside a call, so the triple determines the value):
    a symmetric deck's cross mesh repeats triples IEEE-exactly — the
    4-radial fan's is exactly 4.00× duplicated (probe40's census,
    momwire#680 U1) — and a cache hit returns the very same floats, so
    the memo is bit-identical to the unmemoized loop by construction.
    Keys are exact float tuples: no rounding, no tolerance, nothing to
    convention-gate.
    """
    rho_b, z_b, zp_b = np.broadcast_arrays(
        np.asarray(rho, float), np.asarray(z, float), np.asarray(zp, float)
    )
    out = np.empty((6,) + rho_b.shape, dtype=np.complex128)
    memo = {}
    it = np.nditer(rho_b, flags=["multi_index"])
    for _ in it:
        ix = it.multi_index
        key = (float(rho_b[ix]), float(z_b[ix]), float(zp_b[ix]))
        got = memo.get(key)
        if got is None:
            got = memo[key] = six_point(
                eps_t,
                k2,
                key[0],
                key[1],
                key[2],
                rtol=rtol,
                lam_mult=lam_mult,
            )
        out[(slice(None),) + ix] = got
    return dict(zip(KEYS, out))


def radius_tables(eps_t, k2, rho, z, zp, wire_radius, rtol=1e-10):
    """`designed_tables` with the thin-wire offset folded in:
    ρ_eff = hypot(ρ, a) — the same-edge moments' R = √(Δz² + a²)
    convention extended to the cross family (the derivation's radius
    rule: every cross-family evaluation whose pair distance can reach
    the a-scale carries the offset; at R ≫ a it is invisible)."""
    rho_eff = np.hypot(np.asarray(rho, float), float(wire_radius))
    return designed_tables(eps_t, k2, rho_eff, z, zp, rtol=rtol)
