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
evaluation, because the corner cannot be interpolated from a grid that
must exclude it. Two ROUTES evaluate that same integrand on that same
path and differ only in where the panels come from (momwire#895):

  point   : `six_point` (or its C++ twin) — panels chosen adaptively per
            (ρ, z, z′) triple, O(ms)/point;
  column  : `six_columns` — panels fixed ONCE per ρ column and shared by
            every (z, z′) in it, converged for the column's smallest s,
            O(µs)/point after the column's setup.

The column route is not an approximation and not a grid: in `_core` the
only z and z′ dependence is e^{γ₋z′ − γ₊z} and the only ρ dependence is
the Bessel / Hankel factor, so a column shares the path, the nodes, the
weights, the path derivative AND the Bessel factor exactly, and a point is
one exponential per node plus a (6 × K)·K product. The corner is served by
the same contour as everything else, which is why it does not bite.

Measured (#895): every point of a real crossing set sits in a column of
≥ 32 mast heights, so on BLE 45 ft the route runs 17.9× (N = 4)
and 19.3× (N = 16) the C++ point twin over the same unique triples,
single-threaded — 47 µs/point against 0.84 ms — for the same impedance to
every printed digit.

Convention gate: e^{+jωt}, ε̃ = ε_r − jσ/ωε₀, asserted at import.
"""

from __future__ import annotations

import functools
import os

import numpy as np
from scipy.special import hankel1, hankel2
from threadpoolctl import ThreadpoolController

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

try:  # the C++ twin (momwire#680 U2) is optional; this walk is the reference
    from . import _near_interface_accel as _nia
except ImportError:  # pragma: no cover - pure-Python install
    _nia = None

KEYS = ("U", "V", "W", "dzW", "dzpV", "dzpW")
_LAM_MULT = 8.0
_RAY = np.exp(1j * np.pi / 4.0)
_MAX_RAY_PANELS = 90
_FAR_PAIR_KILL = 60.0

# Column-rule resolution: each seeded interval (`_sub_seed`) is split into
# 2^p fixed Gauss panels. p carries no structure — `_sub_seed` does — so it
# is pure margin, and the measured margin at p = 0 is already four decades:
# the #680 ledger reads 1.5e-15 against the reference walk, a 240-point
# random ladder over 1.8–50 MHz, ε_r 1.5–30, σ 1e-4–5, ρ 1e-5–60 m reads
# 2.0e-11 worst, and BLE N = 4/16 whole-deck reads 2.1e-14 against the C++
# twin. p = 1 costs 2.3× the wall and moves none of those numbers, which is
# what the converged-pair gate says: the rule is converged AT p = 0.
_COLUMN_P = 0

# Largest (nz × K) exponential block a column materialises at once. K grows
# with ρ (one panel per J₀ oscillation), so a long-radial column times a tall
# mast is the shape that would allocate hundreds of MB in one go. Rows are
# independent, so chunking z is invisible to the answer.
_COLUMN_Z_CHUNK = 1 << 22

# The twin's capability flag — its OWN symbol (`near_interface_680`), never a
# shared one: a .so built at an earlier arc exports the #568 entries but not
# this one, and a shared flag would claim a contract it cannot serve.
_HAVE_NEAR_INTERFACE_ACCEL = _nia is not None and bool(
    getattr(_nia, "near_interface_680", False)
)

# The tests' handle on the dispatch — parity gates drive BOTH machines inside
# one process. `MOMWIRE_NEAR_INTERFACE_FORCE_NUMPY` is the whole-run switch (a
# timing comparison, a bisect); `monkeypatch.setattr(ni, "_FORCE_NUMPY", True)`
# is the per-test one. `_use_near_interface_accel` reads both at CALL time.
_FORCE_NUMPY = bool(os.environ.get("MOMWIRE_NEAR_INTERFACE_FORCE_NUMPY"))

# The route `designed_tables` fills through (momwire#895). Same switch shape
# as `_FORCE_NUMPY` one level up: `MOMWIRE_NEAR_INTERFACE_ROUTE` is the
# whole-run spelling (a timing comparison, a bisect),
# `monkeypatch.setattr(ni, "_ROUTE", "point")` the per-test one, and both are
# read at CALL time. `point` selects the per-point walk — the C++ twin when
# built, else `six_point` — which is also the parity gates' reference.
_ROUTES = ("column", "point")
_ROUTE = os.environ.get("MOMWIRE_NEAR_INTERFACE_ROUTE", "column")


def _use_near_interface_accel():
    """The C++ walk twin serves when built and not forced off."""
    return _HAVE_NEAR_INTERFACE_ACCEL and not _FORCE_NUMPY


def _use_column_route():
    """The fixed per-column rule serves unless the point route is asked for.

    An unrecognised spelling REFUSES rather than falling back: a typo in the
    whole-run env switch would otherwise silently select the route it was
    set to avoid, which is exactly the bisect this switch exists for.
    """
    if _ROUTE not in _ROUTES:
        raise ValueError(
            f"near-interface route {_ROUTE!r} is not one of {_ROUTES} "
            "(MOMWIRE_NEAR_INTERFACE_ROUTE)"
        )
    return _ROUTE == "column"


# The BLAS pool is held to the PHYSICAL core count for the length of a
# column fill (momwire#898). The column route's only BLAS call is the small
# (nz × K)·(K × 6) complex gemm per column; OpenBLAS threads it across the
# logical count, and the pool then SPINS after the call on the hyperthread
# siblings of the core running the next column's numpy / scipy work (the
# exponential, the Bessel and Hankel factors), which is where the route's
# time goes. Measured on a 4c/8t box, one 109-z column: BLAS limited to 1,
# 2 and 4 threads all read ~6 ms; 8 reads ~11 ms; the gemm itself is
# 0.06 ms either way. So the artifact is the siblings, not threading as
# such, and physical cores is the fix — the same policy antennaknobs'
# server applies process-wide (its thread-policy block; antennaknobs#1050,
# #1051, #1052 are the measurements behind it).
#
# The limit never RAISES a count: a caller who already pinned lower (the
# served app at physical, a bisect at OPENBLAS_NUM_THREADS=1) keeps theirs.
# The controller is built once (a library scan, ~0.7 ms) and the context
# costs ~20 µs to enter and leave, so it wraps a whole fill call, not a
# column. It sets the process-wide OpenBLAS count and restores it on exit.


@functools.lru_cache(maxsize=1)
def _blas_controller():
    """The process's one `ThreadpoolController`, built on first use — after
    numpy and scipy have loaded their BLAS, which is what it scans for."""
    return ThreadpoolController()


@functools.lru_cache(maxsize=1)
def _physical_cpu_count():
    """Physical cores, not HT siblings: psutil when present (portable), else
    the logical count — never "logical / 2", which misfires on parts
    without HT (antennaknobs' server has the history)."""
    try:
        import psutil  # noqa: PLC0415 — optional, imported on use
    except ImportError:
        return max(1, os.cpu_count() or 1)
    return psutil.cpu_count(logical=False) or max(1, os.cpu_count() or 1)


def _blas_physical_cores():
    """Context manager holding every loaded BLAS to at most the physical
    core count, and to no more than it already has."""
    ctl = _blas_controller()
    n = _physical_cpu_count()
    for lib in ctl.lib_controllers:
        if lib.user_api == "blas":
            n = min(n, lib.num_threads)
    return ctl.limit(limits=n, user_api="blas")


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


def _fixed_gauss(edges, p):
    """Gauss nodes/weights on each [e_i, e_{i+1}], split into 2^p panels.

    The fixed stand-in for `_adaptive_segment`: same `_GX`/`_GW` rule, same
    per-interval scope, bisection depth read off `p` instead of off the
    integrand. p = 0 is one 24-point panel per seeded interval, i.e. the
    adaptive walk's own starting point before any bisection.
    """
    xs, ws = [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        sub = np.linspace(e0, e1, 2**p + 1)
        for a, b in zip(sub[:-1], sub[1:]):
            mid, half = 0.5 * (a + b), 0.5 * (b - a)
            xs.append(mid + half * _GX)
            ws.append(_GW * half)
    return np.concatenate(xs), np.concatenate(ws)


def _sub_seed(edges, rho):
    """Sub-seed a sorted edge list so no interval spans more than a doubling
    in λ or one oscillation of J₀(λρ).

    The adaptive walk gets both of these for free by bisecting on the
    integrand; a FIXED rule has to seed them, and both rules are already in
    the house rather than invented here:

      * doubling — `_ray_integral`'s own panel rule ("panels START at the λ0
        scale and double"), for its own stated reason: the structure scale
        of the log family is λ itself. It is what resolves γ_p's branch
        point at λ = k_p, which `_head` seeds edges AROUND (±15 %, ±40 %)
        and then leaves in a single interval reaching the first seventh of
        a_head — a factor 12.5 in λ at σ = 3 S/m class, measured 1.8e-8;
      * one oscillation — `_tail_below`'s J₀ zero lattice, read at 2π/ρ
        (every second zero) rather than π/ρ. Without it a 41 m radial puts
        12 oscillations in one 24-point panel and reads 110 % wrong; 4 per
        panel, which is a 13.6 m one, already costs 8e-14.

    On a BLE 45 ft column neither fires in the HEAD — `_head`'s own edges
    already satisfy both — so the whole of this is the mid's seeding there,
    and the head only starts paying where it was genuinely under-resolved.
    """
    lat = (2.0 * np.pi / rho) if rho > 0.0 else np.inf
    out = [edges[0]]
    for e in edges[1:]:
        while True:
            lo = out[-1]
            step = min(e - lo, lat, lo) if lo > 0.0 else min(e - lo, lat)
            if lo + step >= e:
                break
            out.append(lo + step)
        out.append(e)
    return out


def _column_rule(rho, k_p, k_m, s_min, lam_mult=_LAM_MULT, p=None):
    """Nodes λ_k and weights w_k for one (ρ, z′) column, path derivative and
    Bessel/Hankel factor folded in. Returns (λ, w), both (K,) complex.

    Every path decision below is `six_point`'s, cited by the line it
    mirrors, evaluated ONCE for the column instead of once per point. The
    only thing the column has to choose for itself is `s_min`: the smallest
    s = z − z′ in the column, which is the slowest-decaying member and so
    the one the extents must be converged for. Points with larger s then
    carry nodes past their own decay, where the integrand underflows to
    exact 0.0 — the tail being zero, the walk's own quiet rule.
    """
    # `_COLUMN_P` is read at CALL time, like the two route switches: a
    # default argument would bind it at import and the resolution ladder
    # could not be driven from a test or a probe.
    p = _COLUMN_P if p is None else int(p)
    # `six_point`: kk / a_head / lam_top, verbatim.
    kk = max(k_p, abs(k_m))
    a_head = 1.1 * kk
    lam_top = lam_mult * kk
    # `six_point`'s far-pair kill cap, on the column's SMALLEST s: that is
    # the largest 60/s, so the extents are the least capped any member
    # would ask for and no member loses range it needed.
    if s_min > 0.0 and _FAR_PAIR_KILL / s_min < lam_top:
        lam_kill = _FAR_PAIR_KILL / s_min
        a_head = max(2.2 * k_p, min(a_head, lam_kill))
        lam_top = max(1.5 * a_head, lam_kill)

    # --- head: `_head`'s detour, H rule and seeded edges. H depends on ρ
    # alone, so it is column-shared exactly.
    H = min(0.35 * a_head, _DETOUR / max(rho, 1e-12))
    H = max(H, 1e-6 * a_head)
    edges = {0.0, a_head}
    for i in range(1, 7):
        edges.add(a_head * i / 7.0)
    for mk in (k_p, abs(k_m.real)):  # `six_point`'s `marks` argument
        for w in (0.0, -0.15, 0.15, -0.4, 0.4):
            v = mk * (1.0 + w)
            if 0.0 < v < a_head:
                edges.add(v)
    t, wt = _fixed_gauss(_sub_seed(sorted(edges), rho), p)
    lam_h = t + 1j * H * np.sin(np.pi * t / a_head)
    dl_h = 1.0 + 1j * H * (np.pi / a_head) * np.cos(np.pi * t / a_head)
    b0, _ = _bessel_j0_j1x(lam_h * rho)  # `six_point`'s `f_j0`
    w_h = wt * dl_h * b0

    # --- mid: the real axis [a_head, lam_top], same J₀ factor. `six_point`
    # hands the WHOLE range to one `_adaptive_segment`, so unlike the head
    # it carries no seeding at all and `_sub_seed` supplies all of it.
    t, wt = _fixed_gauss(_sub_seed([a_head, lam_top], rho), p)
    lam_m = t.astype(np.complex128)
    b0, _ = _bessel_j0_j1x(lam_m * rho)
    w_m = wt * b0

    # --- tail: `_ray_integral`'s geometric panels — starting at the λ₀
    # scale and doubling toward the decay scale, which is what resolves the
    # 1/λ log content when s + ρ is tiny — run out to 60 decay lengths
    # instead of to the adaptive quiet test. e^{−60} = 9e−27 of the total:
    # the same dead-range constant `_FAR_PAIR_KILL` uses, 16 decades inside
    # any rtol a caller asks for, which is why this rule is rtol-free.
    scale = np.sqrt(2.0) / (s_min + rho)
    step = min(0.25 * scale, lam_top)
    t_edges = [0.0]
    while t_edges[-1] < _FAR_PAIR_KILL * scale:
        t_edges.append(t_edges[-1] + step)
        step *= 2.0
    tt, wtt = _fixed_gauss(t_edges, p)
    if rho == 0.0:
        # `six_point`: the single up-ray, J₀(0) = 1, no Hankel split.
        lam_t = lam_top + tt * _RAY
        w_t = wtt * _RAY
    else:
        up = lam_top + tt * _RAY
        dn = lam_top + tt * np.conj(_RAY)
        lam_t = np.concatenate([up, dn])
        w_t = np.concatenate(
            [
                wtt * _RAY * 0.5 * hankel1(0, up * rho),
                wtt * np.conj(_RAY) * 0.5 * hankel2(0, dn * rho),
            ]
        )
    return np.concatenate([lam_h, lam_m, lam_t]), np.concatenate([w_h, w_m, w_t])


def _column_factors(lam, w, k_p, k_m):
    """`_core` at every node with its z-dependent exponential factored out
    and the weights folded in: F (6, K) and the two γ (K,), such that
    six(z, z′) = F @ exp(γ_m z′ − γ_p z). Index order is `_core`'s."""
    g_p = _gamma(lam, k_p)
    g_m = _gamma(lam, k_m)
    u = 2.0 * lam / (g_p + g_m)
    v = 2.0 * lam / (k_m * k_m * g_p + k_p * k_p * g_m)
    wv = (g_p - g_m) * v
    return np.stack([u, v, wv, -g_p * wv, -g_p * g_m * v, g_m * wv]) * w, g_p, g_m


def six_columns(eps_t, k2, rho, zs, zp, rtol=1e-10, lam_mult=_LAM_MULT, p=None):
    """The six designed integrals for ONE ρ column: every (z, z′) pair at
    that ρ, z ≥ 0 ≥ z′, R = hypot(ρ, z − z′) > 0 for each. `zp` is either a
    scalar shared by every z or an array paired with `zs` element-wise.
    Returns (len(zs), 6) complex, row i for (zs[i], zp[i]).

    A column is a ρ, not a (ρ, z′) (momwire#899): in `_core` z and z′ both
    enter only through e^{γ₋z′ − γ₊z}, so every path decision — the nodes,
    the weights, the derivative, the Bessel factor — is shared across BOTH,
    and a point is one exponential per node whichever z′ it carries. The
    only thing the members share beyond ρ is `s_min`, the column's smallest
    z − z′, which sets the extents (see `_column_rule`).

    Same domain contract and same refusals as `six_point`. `rtol` is
    accepted for signature parity and does not move the rule: the fixed
    rule's resolution is `p` and its extents are the e^{−60} dead-range
    constant, both rtol-free (see `_column_rule`). What says the rule is
    converged is therefore the p / p+1 pair, not a tolerance argument.

    A member's value depends on which column it was evaluated in, through
    `s_min` alone: measured 7.9e-16 worst between a ledger point read as a
    column of one and the same point read inside a 31-z column. So this
    route does not promise the same BITS for a triple served in two
    differently-grouped calls, only the same answer; the memo in
    `designed_tables` is what keeps one call's duplicates identical.
    """
    k_p = float(k2)
    k_m = k_medium(complex(eps_t), k_p)
    rho = float(rho)
    zs = np.atleast_1d(np.asarray(zs, dtype=float))
    zps = np.broadcast_to(np.asarray(zp, dtype=float), zs.shape)
    bad = (zs < 0.0) | (zps > 0.0)
    if np.any(bad):
        i = int(np.argmax(bad))
        raise ValueError(f"need z >= 0 >= zp, got {(float(zs[i]), float(zps[i]))!r}")
    s = zs - zps
    if rho < 0.0 or np.any(s + rho <= 0.0):
        s_bad = float(s[np.argmin(s)])
        raise ValueError(f"need R > 0, got rho={rho!r}, s={s_bad!r}")

    lam, w = _column_rule(rho, k_p, k_m, float(np.min(s)), lam_mult=lam_mult, p=p)
    F, g_p, g_m = _column_factors(lam, w, k_p, k_m)
    # Underflow to exact 0.0 is the answer, not a warning: a member whose s
    # is far above the column's s_min carries nodes past its own decay, and
    # the high-σ far pair underflows e^{−1330} on every node of the tail.
    # The exponent is built per DISTINCT z': g_m * z' is one (K,) product
    # shared by every z that carries that z', never an (nz x K) product of
    # its own — that product was the #899 study's +44 % (`group_columns`).
    # A column of one z' is the #895 path, the same bits.
    out = np.empty((zs.size, 6), dtype=np.complex128)
    step = max(1, _COLUMN_Z_CHUNK // lam.size)
    with np.errstate(under="ignore"):
        for zp_one in np.unique(zps):
            rows = np.flatnonzero(zps == zp_one)
            base = g_m * zp_one  # (K,)
            for i0 in range(0, rows.size, step):
                sel = rows[i0 : i0 + step]
                e = np.exp(base[None, :] - g_p[None, :] * zs[sel, None])  # (nz, K)
                out[sel] = e @ F.T
    return out


def designed_tables(eps_t, k2, rho, z, zp, rtol=1e-10, lam_mult=_LAM_MULT, memo=None):
    """Broadcast wrapper over the designed evaluation. Accepts z′ = 0 and
    z = 0 exactly (no clamp); refuses only R = 0. Returns dict over `KEYS`.

    Duplicate (ρ, z, z′) triples are evaluated ONCE per call (ε̃, k₂,
    rtol are fixed inside a call, so the triple determines the value):
    a symmetric deck's cross mesh repeats triples IEEE-exactly — the
    4-radial fan's is exactly 4.00× duplicated (probe40's census,
    momwire#680 U1) — and a cache hit returns the very same floats, so
    the memo is bit-identical to the unmemoized loop by construction.
    Keys are exact float tuples: no rounding, no tolerance, nothing to
    convention-gate.

    `memo` (momwire#688): an optional CALLER-OWNED dict extends the dedup
    across calls — the crossing fill's admissibility split makes many
    designed calls per fill (near batch, far-block samples), and a
    symmetric deck repeats triples ACROSS those calls exactly as it does
    within one. The caller must hold (ε̃, k₂, rtol) fixed for the memo's
    lifetime — the key is the triple alone, exactly as within a call.
    Filled entries are (6,) complex arrays; unfilled sentinel is None.

    The memo layer stays HERE whichever route serves: the dedup happens
    first, and only the unique triples reach `six_columns` (#895), the C++
    batch (#680 U2, which is also U3's parallel unit), or the `six_point`
    loop. The numpy `six_point` walk is the reference for all three; the
    twin is gated against it at 1e-12 RELATIVE and the column route at the
    reference's own 1e-10, never bit.

    On the column route the unique triples are grouped by `group_columns`
    (exact ρ) and each group evaluated once, which is where the 11–15× comes from: a real crossing
    set puts every point in a column of ≥ 32 mast heights. The grouping was
    by (ρ, z′) until momwire#899; see `group_columns` for why it is not ρ
    alone. The dedup, the key order and the scatter below are the same on
    every route — only the arithmetic that fills `memo` differs.

    The dedup is one `np.unique` over the asked triples (momwire#899): the
    asked set is ~3× the unique set on a crossing fill, and the two Python
    passes over it that stood here were 10–15 % of the column route.
    """
    rho_b, z_b, zp_b = np.broadcast_arrays(
        np.asarray(rho, float), np.asarray(z, float), np.asarray(zp, float)
    )
    if memo is None:
        memo = {}
    keys, inverse = _unique_triples(rho_b, z_b, zp_b)
    unique = []
    for key in keys:
        if memo.get(key) is None:
            if key not in memo:
                unique.append(key)
            memo[key] = None  # first-appearance order, filled below
    if _use_column_route() and unique:
        # First-appearance order is already fixed above, so the grouping is
        # free to reorder: it decides only which rule serves a triple, never
        # which key the memo holds or in what order it was seen.
        with _blas_physical_cores():
            for members in group_columns(unique).values():
                r = members[0][0]
                zs = [m[1] for m in members]
                zps = [m[2] for m in members]
                vals = six_columns(eps_t, k2, r, zs, zps, rtol=rtol, lam_mult=lam_mult)
                for key, row in zip(members, vals):
                    memo[key] = row
    elif _use_near_interface_accel() and unique:
        k_p = float(k2)
        k_m = k_medium(complex(eps_t), k_p)
        tri = np.asarray(unique, dtype=float).reshape(-1, 3)
        vals = _nia.near_interface_six_batch(
            k_p,
            k_m,
            np.ascontiguousarray(tri[:, 0]),
            np.ascontiguousarray(tri[:, 1]),
            np.ascontiguousarray(tri[:, 2]),
            float(rtol),
            float(lam_mult),
            _ADAPT_DEPTH,
            _DETOUR,
            _GX,
            _GW,
        )
        for key, row in zip(unique, vals):
            memo[key] = row
    else:
        for key in unique:
            memo[key] = six_point(
                eps_t,
                k2,
                key[0],
                key[1],
                key[2],
                rtol=rtol,
                lam_mult=lam_mult,
            )
    if keys:
        vals = np.stack([memo[key] for key in keys])  # (n_unique, 6)
        out = np.ascontiguousarray(vals[inverse].T).reshape((6,) + rho_b.shape)
    else:
        out = np.empty((6,) + rho_b.shape, dtype=np.complex128)
    return dict(zip(KEYS, out))


def group_columns(keys):
    """Group unique (rho, z, z') triples into the columns `designed_tables`
    evaluates: exact rho. Returns {rho: [triples]}, first-seen order inside
    a group. The ONE grouping production uses; a replay (probe4) calls this
    too, so it can never mirror a grouping that exists at no commit.

    Why rho alone, measured on the real BLE asked sets (the #899 study,
    clock-free: setups, and node-evaluations sum(n_members * K) with K from
    `_column_rule` itself):

        BLE N = 4, 7628 triples     columns   singletons   sum n*K
          by (rho, z')                 138          68     7.02e6
          by rho                        84          40     7.54e6
          by (rho, s within x2)        412          48     6.59e6

    A column's K is set by s_min (tail extents, kill cap), so merging widens
    the rule for some members: +7.5 % node-evaluations here against 54 fewer
    setups of ~1.5 ms each. Banding by s to avoid the widening is a loss —
    a single (rho, z') column already spans ~16 factor-two bands of s, so
    the band splits far more than it merges. The +26 % that grouping by rho
    first measured (Skylake, E3 block) was NOT the widened rule: it was the
    per-member z' making the exponent's argument a second full (nz x K)
    complex product, which the same study's E2 block isolated at +44 % on
    the unchanged grouping. `six_columns` now evaluates the exponent per
    distinct z' inside a column, so that product is gone.
    """
    columns = {}
    for key in keys:
        columns.setdefault(key[0], []).append(key)
    return columns


def _unique_triples(rho_b, z_b, zp_b):
    """The distinct (ρ, z, z′) triples of a broadcast ask, as Python-float
    tuples in FIRST-APPEARANCE order, plus the (flat) index of each asked
    point into that list. One `np.unique` over the asked set; the memo's
    key contract (float triple, first seen first) is unchanged, and −0.0
    and 0.0 fold together here exactly as they do as dict keys."""
    tri = np.stack([rho_b.ravel(), z_b.ravel(), zp_b.ravel()], axis=1)
    if tri.shape[0] == 0:
        return [], np.empty(0, dtype=np.intp)
    rows, first, inverse = np.unique(
        tri, axis=0, return_index=True, return_inverse=True
    )
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    keys = [tuple(r) for r in rows[order].tolist()]
    return keys, rank[np.asarray(inverse).ravel()]


def radius_tables(eps_t, k2, rho, z, zp, wire_radius, rtol=1e-10, memo=None):
    """`designed_tables` with the thin-wire offset folded in:
    ρ_eff = hypot(ρ, a) — the same-edge moments' R = √(Δz² + a²)
    convention extended to the cross family (the derivation's radius
    rule: every cross-family evaluation whose pair distance can reach
    the a-scale carries the offset; at R ≫ a it is invisible). `memo`
    keys on the FOLDED ρ_eff (see `designed_tables`)."""
    rho_eff = np.hypot(np.asarray(rho, float), float(wire_radius))
    return designed_tables(eps_t, k2, rho_eff, z, zp, rtol=rtol, memo=memo)
