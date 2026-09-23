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
  column  : `six_columns` (or its C++ twin) — panels fixed ONCE per ρ
            column and shared by every (z, z′) in it, converged for the
            column's smallest s, O(µs)/point after the column's setup.

Each route has a C++ twin and each twin is the twin of ITS OWN numpy walk,
gated against it and against nothing else (momwire#680 U2, #899 item 1);
both live in `_near_interface_accel` behind their own capability flags, and
`MOMWIRE_NEAR_INTERFACE_FORCE_NUMPY` turns both off together, because
"the numpy walks are the machine" is one question and not two. The column
twin carries the RULE as well as the sum — a third of the numpy column
route's wall is `_column_rule` + `_column_factors` — and it parallelises
across columns, which is where the column route's thread scaling comes
from (the numpy one has none; momwire#898 is why).

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

from . import _accel
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

# The C++ twin (momwire#680 U2) is optional; this walk is the reference.
#
# Imported through `_accel` rather than directly (momwire#1032) so it comes
# from the SAME instruction-set variant `_accelerators` did. The two are
# separate .so files: pairing an AVX2 one with an SSE2 one would fault on
# exactly the CPU the split exists for, and it would fault from the module
# nobody was looking at. `import_companion` returns None on a pure-Python
# install exactly as the old `except ImportError` did.
_nia = _accel.import_companion("_near_interface_accel")

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

# The twins' capability flags — each its OWN symbol (`near_interface_680`,
# `near_interface_columns_899`), never a shared one: a .so built at an earlier
# arc exports the #568 entries but not the point twin's, and one built between
# the two arcs exports the point twin's but not the column twin's. A shared
# flag would claim a contract it cannot serve.
_HAVE_NEAR_INTERFACE_ACCEL = _nia is not None and bool(
    getattr(_nia, "near_interface_680", False)
)
_HAVE_NEAR_INTERFACE_COLUMNS_ACCEL = _nia is not None and bool(
    getattr(_nia, "near_interface_columns_899", False)
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


def _use_column_accel():
    """The C++ COLUMN twin serves the column route when built and not forced
    off. Same switch as the point twin's, deliberately: `_FORCE_NUMPY` means
    "the numpy walks are the machine", and a timing comparison or a bisect
    that turned off one twin and not the other would answer neither
    question."""
    return _HAVE_NEAR_INTERFACE_COLUMNS_ACCEL and not _FORCE_NUMPY


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


def _refuse_bad_members(rho, zs, zps):
    """`six_point`'s domain contract over a SET of members: z ≥ 0 ≥ z′ and
    R > 0, refused in the walk's own words with the offending member's own
    numbers. Returns s = z − z′ for the caller that needs it.

    One copy, because two machines evaluate columns (`six_columns` and the
    C++ twin) and a refusal is part of the contract, not an implementation
    detail: a second spelling could drift into refusing a different set.
    """
    bad = (zs < 0.0) | (zps > 0.0)
    if np.any(bad):
        i = int(np.argmax(bad))
        raise ValueError(f"need z >= 0 >= zp, got {(float(zs[i]), float(zps[i]))!r}")
    s = zs - zps
    rhos = np.broadcast_to(np.asarray(rho, dtype=float), s.shape)
    bad = (rhos < 0.0) | (s + rhos <= 0.0)
    if np.any(bad):
        i = int(np.argmax(bad))
        raise ValueError(f"need R > 0, got rho={float(rhos[i])!r}, s={float(s[i])!r}")
    return s


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
    s = _refuse_bad_members(rho, zs, zps)

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


# The array memo's hash: the three coordinates' IEEE bit patterns folded into
# one uint64 (odd multipliers, xor, a final xor-shift so the high bits reach
# the low ones). It only ORDERS the store; equality is always decided on the
# full row, so a collision costs a comparison and never a wrong value.
_HASH_M = (
    np.uint64(0x9E3779B97F4A7C15),
    np.uint64(0xC2B2AE3D27D4EB4F),
    np.uint64(0x165667B19E3779F9),
)

# `TripleMemo`'s merge policy (momwire#1168 U1). Each insert is sorted on its
# own and appended as a pending run; stored rows are re-sorted only when the
# pending rows exceed max(_MEMO_MERGE_MIN_ROWS, main / _MEMO_MERGE_FRACTION)
# (merge into main), or the pending runs exceed _MEMO_MAX_PENDING_RUNS (the
# runs compacted into one). The fraction makes main grow geometrically between
# merges, so a fill re-sorts main O(log n) times; the run cap bounds the
# searchsorted passes a lookup makes. Counted on bspline hub_deck(16) x8
# (419 inserts, 107k rows): 1 merge + 25 compactions, 0.82 M rows re-sorted.
# The #1168 audit's prototype re-sorted its pending store on EVERY insert —
# 417 re-sorts, 12.4 M rows — and ran bspline slower than the dict did.
_MEMO_MERGE_MIN_ROWS = 1 << 16
_MEMO_MERGE_FRACTION = 4
_MEMO_MAX_PENDING_RUNS = 16


def _row_hash(keys):
    """uint64 hash of normalised (n, 3) float rows (see `_HASH_M`)."""
    b = keys.view(np.uint64)
    h = b[:, 0] * _HASH_M[0]
    h ^= b[:, 1]
    h *= _HASH_M[1]
    h ^= b[:, 2]
    h *= _HASH_M[2]
    h ^= h >> np.uint64(29)
    return h


class TripleMemo:
    """The caller-owned cross-call memo of `designed_tables` (momwire#688),
    held as arrays (momwire#1168 U1).

    The contract is the dict's it replaces, `_designed_tables_reference`:
    the key is the exact (ρ, z, z′) float triple, −0.0 and 0.0 are one key
    (rows are stored as `row + 0.0`), a NaN row never matches, a hit returns
    the very floats first evaluated, and a triple is evaluated only when the
    memo does not hold it. So it decides only HOW a row is found, never which
    rows are fresh nor their order — and `designed_tables` through it is
    bit-identical to the dict route.

    One difference, on the failure path only: a call that raises part-way
    inserts nothing (the dict left None sentinels that made every later ask
    of those keys raise). The refusals are per member, so the next ask of
    the same triple refuses again.

    Storage is a sorted main run plus sorted pending runs, each run the
    parallel arrays (hash, key, value, insertion sequence), sorted by hash;
    a lookup is one searchsorted per run and a full-row compare, walking
    forward over equal hashes. See `_MEMO_MERGE_MIN_ROWS` for the merge
    policy. `stats` counts the work (lookups, hits, collisions: hash-equal
    compares with a different key; merges and compactions: re-sorts of
    stored rows).

    Iterating gives the keys as float tuples in insertion order, which
    `designed_tables` makes first-appearance order — the dict's order.
    """

    def __init__(self):
        self._main = self._empty_run()
        self._pending = []
        self._n_pending = 0
        self._seq = 0
        self.stats = dict.fromkeys(
            ("lookups", "hits", "collisions", "inserts", "merges", "compactions"),
            0,
        )

    @staticmethod
    def _empty_run():
        return (
            np.empty(0, dtype=np.uint64),
            np.empty((0, 3), dtype=float),
            np.empty((0, 6), dtype=np.complex128),
            np.empty(0, dtype=np.int64),
        )

    def __len__(self):
        return self._main[0].size + self._n_pending

    def _runs(self):
        return (self._main, *self._pending)

    def lookup(self, rows):
        """(hit, block) for (n, 3) float rows: `block[i]` holds the stored
        value where `hit[i]`, and is uninitialised elsewhere."""
        n = rows.shape[0]
        hit = np.zeros(n, dtype=bool)
        block = np.empty((n, 6), dtype=np.complex128)
        self.stats["lookups"] += n
        if n == 0 or len(self) == 0:
            return hit, block
        keys = rows + 0.0
        hq = _row_hash(keys)
        todo = np.arange(n)
        for h, k, v, _s in self._runs():
            if todo.size == 0:
                break
            if h.size == 0:
                continue
            found = self._find(h, k, hq[todo], keys[todo])
            ok = found >= 0
            block[todo[ok]] = v[found[ok]]
            hit[todo[ok]] = True
            # A key lives in exactly one run (only misses are inserted), so a
            # row found here is not looked for again.
            todo = todo[~ok]
        self.stats["hits"] += int(np.count_nonzero(hit))
        return hit, block

    def _find(self, h, k, hq, q):
        """Index into one run of each query's key, −1 where absent."""
        found = np.full(hq.size, -1, dtype=np.intp)
        live = np.arange(hq.size)
        pos = np.searchsorted(h, hq, side="left")
        while live.size:
            p = pos[live]
            keep = p < h.size
            live, p = live[keep], p[keep]
            keep = h[p] == hq[live]
            live, p = live[keep], p[keep]
            eq = np.all(k[p] == q[live], axis=1)
            found[live[eq]] = p[eq]
            live = live[~eq]
            self.stats["collisions"] += live.size
            pos[live] += 1
        return found

    def insert(self, rows, vals):
        """Store (m, 3) rows the memo does not hold, with their (m, 6) values,
        in the order given."""
        m = rows.shape[0]
        if m == 0:
            return
        keys = rows + 0.0
        h = _row_hash(keys)
        o = np.argsort(h, kind="stable")
        seq = np.arange(self._seq, self._seq + m, dtype=np.int64)
        self._seq += m
        self._pending.append((h[o], keys[o], np.asarray(vals)[o], seq[o]))
        self._n_pending += m
        self.stats["inserts"] += m
        if self._n_pending > max(
            _MEMO_MERGE_MIN_ROWS, self._main[0].size // _MEMO_MERGE_FRACTION
        ):
            self._main = self._merged(self._runs())
            self._pending = []
            self._n_pending = 0
            self.stats["merges"] += 1
        elif len(self._pending) > _MEMO_MAX_PENDING_RUNS:
            self._pending = [self._merged(self._pending)]
            self.stats["compactions"] += 1

    @staticmethod
    def _merged(runs):
        """One run from sorted runs. The stable sort is timsort on uint64,
        which merges the presorted runs rather than sorting from scratch."""
        o = np.argsort(np.concatenate([r[0] for r in runs]), kind="stable")
        return tuple(np.concatenate([r[f] for r in runs])[o] for f in range(4))

    def keys(self):
        """The stored keys as float tuples, in insertion order."""
        runs = self._runs()
        seq = np.concatenate([r[3] for r in runs])
        k = np.concatenate([r[1] for r in runs])[np.argsort(seq)]
        return [tuple(r) for r in k.tolist()]

    def values(self):
        """The stored (6,) values, in insertion order."""
        runs = self._runs()
        seq = np.concatenate([r[3] for r in runs])
        return list(np.concatenate([r[2] for r in runs])[np.argsort(seq)])

    def __iter__(self):
        return iter(self.keys())


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

    `memo` (momwire#688): an optional CALLER-OWNED `TripleMemo` extends the
    dedup across calls — the crossing fill's admissibility split makes many
    designed calls per fill (near batch, far-block samples), and a
    symmetric deck repeats triples ACROSS those calls exactly as it does
    within one. The caller must hold (ε̃, k₂, rtol) fixed for the memo's
    lifetime — the key is the triple alone, exactly as within a call. A
    dict is refused: it is `_designed_tables_reference`'s memo, kept as the
    in-process reference for the array one (momwire#1168 U1), and a dict
    passed here would otherwise be the silent slow route.

    The memo layer stays HERE whichever route serves: the dedup happens
    first, and only the unique triples reach one of the four machines
    (`_evaluate_fresh`). The dispatch, in order:

      column route (default), twin built  : `near_interface_six_columns`,
                                            the whole grouping in ONE call;
      column route, twin off              : the `six_columns` loop, under
                                            the #898 BLAS pin;
      point route, twin built             : `near_interface_six_batch`;
      point route, twin off               : the `six_point` loop.

    The numpy `six_point` walk is the reference for all of them; each twin
    is gated against ITS OWN numpy walk at 1e-12 RELATIVE and the column
    route against the reference's own 1e-10, never bit.

    On the column route the unique triples are grouped by `group_columns`
    (exact ρ) and each group evaluated once, which is where the 11–15× comes from: a real crossing
    set puts every point in a column of ≥ 32 mast heights. The grouping was
    by (ρ, z′) until momwire#899; see `group_columns` for why it is not ρ
    alone. The dedup, the fresh set's order and the scatter below are the
    same on every route — only the arithmetic that fills the fresh rows
    differs.

    The dedup is one `np.unique` over the asked triples (momwire#899): the
    asked set is ~3× the unique set on a crossing fill, and the two Python
    passes over it that stood here were 10–15 % of the column route. The
    cross-call lookup is `TripleMemo`'s array search (momwire#1168 U1): the
    dict it replaced cost a tuple build and two hash probes per unique row,
    ~57 % of this function's time on razor hub_deck(16) x8.
    """
    if memo is not None and not isinstance(memo, TripleMemo):
        raise TypeError(
            f"designed_tables takes a TripleMemo, got {type(memo).__name__}; "
            "a dict memo is _designed_tables_reference's"
        )
    rho_b, z_b, zp_b = np.broadcast_arrays(
        np.asarray(rho, float), np.asarray(z, float), np.asarray(zp, float)
    )
    rows, inverse = _unique_rows(rho_b, z_b, zp_b)
    if memo is None:
        block = np.empty((rows.shape[0], 6), dtype=np.complex128)
        fresh_pos = np.arange(rows.shape[0])
    else:
        hit, block = memo.lookup(rows)
        fresh_pos = np.flatnonzero(~hit)  # ascending: first-appearance order
    if fresh_pos.size:
        sub = rows[fresh_pos]
        vals = _evaluate_fresh(eps_t, k2, sub, rtol, lam_mult)
        block[fresh_pos] = vals
        if memo is not None:
            memo.insert(sub, vals)
    if rows.shape[0]:
        out = np.ascontiguousarray(block[inverse].T).reshape((6,) + rho_b.shape)
    else:
        out = np.empty((6,) + rho_b.shape, dtype=np.complex128)
    return dict(zip(KEYS, out))


def _evaluate_fresh(eps_t, k2, sub, rtol, lam_mult):
    """The six values of each (m, 3) row of `sub` (distinct triples, in
    first-appearance order), as (m, 6), row i for sub[i] — through the route
    `designed_tables` documents. Each branch hands its machine the same
    arguments in the same order as `_designed_tables_reference` does, which
    is what keeps the two routes' bits equal: on the column route a member's
    value depends on its column's membership (`six_columns`), so the fresh
    set and its grouping are the contract, not the values alone.
    """
    if _use_column_route() and _use_column_accel():
        # The twin's parallel unit is the COLUMN, so the whole grouping goes
        # in ONE call: a call per column would hand OpenMP one column at a
        # time and give back exactly the scaling the numpy route lacks. The
        # concatenation keeps each group contiguous and `offsets` says where
        # each starts.
        #
        # No BLAS pin here — the twin has no gemm to pin. It gets the
        # PHYSICAL core count instead, which is #898's finding applied to its
        # own arithmetic: libmvec exp/sincos saturates a core's FPU, so the
        # hyperthread siblings contend rather than add (this kernel measured
        # 40 ms at 4 threads and 46 ms at 8 on a 4c/8t box). The count is
        # passed IN rather than guessed there: the policy, and `psutil`, live
        # on this side.
        k_p = float(k2)
        k_m = k_medium(complex(eps_t), k_p)
        rho_c, sizes, member_order = _column_blocks(sub)
        offsets = np.zeros(sizes.size + 1, dtype=np.intp)
        offsets[1:] = np.cumsum(sizes)
        zs = np.ascontiguousarray(sub[member_order, 1])
        zps = np.ascontiguousarray(sub[member_order, 2])
        # Refused HERE, in the walk's words with the offending member's
        # numbers: the twin refuses the same set (before it builds a single
        # column), but from C++ it cannot spell the values.
        _refuse_bad_members(np.repeat(rho_c, sizes), zs, zps)
        vals = _nia.near_interface_six_columns(
            k_p,
            k_m,
            rho_c,
            offsets,
            zs,
            zps,
            float(lam_mult),
            int(_COLUMN_P),
            float(_DETOUR),
            _physical_cpu_count(),
            _GX,
            _GW,
        )
        out = np.empty((sub.shape[0], 6), dtype=np.complex128)
        out[member_order] = vals
        return out
    if _use_column_route():
        triples = [tuple(r) for r in sub.tolist()]
        got = {}
        with _blas_physical_cores():
            for members in group_columns(triples).values():
                vals = six_columns(
                    eps_t,
                    k2,
                    members[0][0],
                    [m[1] for m in members],
                    [m[2] for m in members],
                    rtol=rtol,
                    lam_mult=lam_mult,
                )
                got.update(zip(members, vals))
        return np.stack([got[t] for t in triples])
    if _use_near_interface_accel():
        k_p = float(k2)
        k_m = k_medium(complex(eps_t), k_p)
        return np.asarray(
            _nia.near_interface_six_batch(
                k_p,
                k_m,
                np.ascontiguousarray(sub[:, 0]),
                np.ascontiguousarray(sub[:, 1]),
                np.ascontiguousarray(sub[:, 2]),
                float(rtol),
                float(lam_mult),
                _ADAPT_DEPTH,
                _DETOUR,
                _GX,
                _GW,
            )
        )
    return np.stack(
        [
            six_point(eps_t, k2, r, zz, zzp, rtol=rtol, lam_mult=lam_mult)
            for r, zz, zzp in sub.tolist()
        ]
    )


def _designed_tables_reference(
    eps_t, k2, rho, z, zp, rtol=1e-10, lam_mult=_LAM_MULT, memo=None
):
    """`designed_tables` with the dict memo it had before momwire#1168 U1,
    kept as the in-process reference the array memo (`TripleMemo`) is gated
    bit-identical against. `memo` is a dict keyed on the exact float triple;
    a filled entry is a (6,) complex array and the unfilled sentinel is None,
    left behind by a call that raised part-way (a later ask of that key then
    raises in the restack below). No production caller reaches it.
    """
    rho_b, z_b, zp_b = np.broadcast_arrays(
        np.asarray(rho, float), np.asarray(z, float), np.asarray(zp, float)
    )
    if memo is None:
        memo = {}
    rows, inverse = _unique_rows(rho_b, z_b, zp_b)
    keys = [tuple(r) for r in rows.tolist()]
    # `block` is the (n_unique, 6) the scatter needs. Filling it AS the values
    # are produced is momwire#904 lever 2: the fill branches already hold every
    # fresh row, so re-reading them out of the memo one key at a time to
    # restack them was pure round trip. Cached rows still come from the dict —
    # they are the only ones this call did not compute.
    block = np.empty((len(keys), 6), dtype=np.complex128)
    unique, fresh_idx, n_filled = [], [], 0
    for i, key in enumerate(keys):
        cached = memo.get(key)
        if cached is None:
            if key not in memo:
                unique.append(key)
                fresh_idx.append(i)
            memo[key] = None  # first-appearance order, filled below
        else:
            block[i] = cached
            n_filled += 1
    if _use_column_route() and unique:
        # First-appearance order is already fixed above, so the grouping is
        # free to reorder: it decides only which rule serves a triple, never
        # which key the memo holds or in what order it was seen.
        fresh_pos = np.asarray(fresh_idx, dtype=np.intp)
        if _use_column_accel():
            # The twin's parallel unit is the COLUMN, so the whole grouping
            # goes in ONE call: a call per column would hand OpenMP one
            # column at a time and give back exactly the scaling the numpy
            # route lacks. The concatenation keeps each group contiguous and
            # `offsets` says where each starts, so the scatter below reads
            # the members in the order they were handed over.
            #
            # No BLAS pin here — the twin has no gemm to pin. It gets the
            # PHYSICAL core count instead, which is #898's finding applied
            # to its own arithmetic: libmvec exp/sincos saturates a core's
            # FPU, so the hyperthread siblings contend rather than add (this
            # kernel measured 40 ms at 4 threads and 46 ms at 8 on a 4c/8t
            # box). The count is passed IN rather than guessed there: the
            # policy, and `psutil`, live on this side.
            k_p = float(k2)
            k_m = k_medium(complex(eps_t), k_p)
            # momwire#904 lever 1: the rows are already a float array on this
            # side, so the twin is handed slices of it. What stood here turned
            # the unique rows into key tuples, grouped the tuples, and then
            # turned the tuples back into a float array — a round trip through
            # Python objects for data that never stopped being an array.
            sub = rows[fresh_pos]
            rho_c, sizes, member_order = _column_blocks(sub)
            offsets = np.zeros(sizes.size + 1, dtype=np.intp)
            offsets[1:] = np.cumsum(sizes)
            zs = np.ascontiguousarray(sub[member_order, 1])
            zps = np.ascontiguousarray(sub[member_order, 2])
            # Refused HERE, in the walk's words with the offending member's
            # numbers: the twin refuses the same set (before it builds a
            # single column), but from C++ it cannot spell the values.
            _refuse_bad_members(np.repeat(rho_c, sizes), zs, zps)
            vals = _nia.near_interface_six_columns(
                k_p,
                k_m,
                rho_c,
                offsets,
                zs,
                zps,
                float(lam_mult),
                int(_COLUMN_P),
                float(_DETOUR),
                _physical_cpu_count(),
                _GX,
                _GW,
            )
            placed = fresh_pos[member_order]
            block[placed] = vals
            n_filled += placed.size
            for j, row in zip(placed, vals):
                memo[keys[j]] = row
        else:
            groups = group_columns(unique)
            with _blas_physical_cores():
                for members in groups.values():
                    r = members[0][0]
                    zs = [m[1] for m in members]
                    zps = [m[2] for m in members]
                    vals = six_columns(
                        eps_t, k2, r, zs, zps, rtol=rtol, lam_mult=lam_mult
                    )
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
        if n_filled != len(keys):
            # A route that does not fill `block` as it goes (the numpy column
            # loop, the point batch, the `six_point` walk), or a memo carrying
            # a None sentinel from a call that raised part-way. Restack from
            # the memo, which is what stood here before momwire#904 — and on
            # the sentinel it raises exactly as it did then rather than
            # scattering an uninitialised row.
            block = np.stack([memo[key] for key in keys])  # (n_unique, 6)
        out = np.ascontiguousarray(block[inverse].T).reshape((6,) + rho_b.shape)
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


def _unique_rows(rho_b, z_b, zp_b):
    """The distinct (ρ, z, z′) rows of a broadcast ask as a float (n, 3)
    ARRAY in first-appearance order, plus the flat index of each asked point
    into it. `_unique_triples` is this plus the tuple build.

    A lexsort over the three columns, not `np.unique(axis=0)`. Same rows and
    same inverse — `axis=0` sorts a structured VOID view of the rows, and on
    the real BLE asked sets that machinery is most of the dedup: measured
    48.5 ms against 7.5 ms here over the N = 16 solve's 74,756 asked rows,
    with rows and inverse array-equal on every call (momwire#904).

    The two agree on the awkward values as well, which is why the swap is
    safe: −0.0 and 0.0 fold together in both (and so as dict keys), and NaN
    rows stay distinct in both, because `!=` is true of every NaN pair here
    exactly as the void compare makes it.
    """
    tri = np.ascontiguousarray(
        np.stack([rho_b.ravel(), z_b.ravel(), zp_b.ravel()], axis=1)
    )
    n = tri.shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=np.intp)
    idx = np.lexsort((tri[:, 2], tri[:, 1], tri[:, 0]))
    srt = tri[idx]
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    np.any(srt[1:] != srt[:-1], axis=1, out=new_group[1:])
    gid = np.cumsum(new_group) - 1
    first = idx[new_group]  # one representative per group, sorted order
    inverse = np.empty(n, dtype=np.intp)
    inverse[idx] = gid
    order = np.argsort(first, kind="stable")  # groups, first-appearance order
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return tri[first[order]], rank[inverse]


def _unique_triples(rho_b, z_b, zp_b):
    """The distinct (ρ, z, z′) triples of a broadcast ask, as Python-float
    tuples in FIRST-APPEARANCE order, plus the (flat) index of each asked
    point into that list. The memo's key contract (float triple, first seen
    first) is unchanged, and −0.0 and 0.0 fold together here exactly as they
    do as dict keys.

    Kept at this exact signature because it IS the #899 contract gate's
    subject; `designed_tables` calls `_unique_rows` directly so it can group
    and scatter on the array without the tuple round trip."""
    rows, inverse = _unique_rows(rho_b, z_b, zp_b)
    if rows.shape[0] == 0:
        return [], inverse
    return [tuple(r) for r in rows.tolist()], inverse


def _column_blocks(sub):
    """`group_columns`' partition of `sub` (an (m, 3) float array), as index
    arithmetic: the distinct ρ in first-appearance order, each group's size,
    and the member order within the concatenation.

    Exactly `group_columns`' ordering — groups in first-seen ρ order, members
    in first-seen order inside a group — so the twin is handed the same
    columns in the same order it was handed before, and the rule each column
    picks is unchanged. Returns (rho_c, sizes, member_order)."""
    rho = np.ascontiguousarray(sub[:, 0])
    uniq, first, inv = np.unique(rho, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    gid = rank[np.asarray(inv).ravel()]
    member_order = np.argsort(gid, kind="stable")  # stable: keeps first-seen
    sizes = np.bincount(gid, minlength=order.size).astype(np.intp)
    return uniq[order], sizes, member_order


def radius_tables(eps_t, k2, rho, z, zp, wire_radius, rtol=1e-10, memo=None):
    """`designed_tables` with the thin-wire offset folded in:
    ρ_eff = hypot(ρ, a) — the same-edge moments' R = √(Δz² + a²)
    convention extended to the cross family (the derivation's radius
    rule: every cross-family evaluation whose pair distance can reach
    the a-scale carries the offset; at R ≫ a it is invisible). `memo`
    keys on the FOLDED ρ_eff (see `designed_tables`)."""
    rho_eff = radius_fold(rho, wire_radius)
    return designed_tables(eps_t, k2, rho_eff, z, zp, rtol=rtol, memo=memo)


def radius_fold(rho, wire_radius):
    """`radius_tables`' ρ → ρ_eff = hypot(ρ, a), the one spelling of it: a
    caller that dedups the folded triples itself (the crossing fill's chunked
    main sandwich) must fold them exactly as the memo keys them."""
    return np.hypot(np.asarray(rho, float), float(wire_radius))
