"""The exact ring kernel's point moments as a power series in k with
geometry-only coefficients (momwire#1454, step 3).

`_exact_kernel.point_moments` integrates the whole kernel again at every
wavenumber. Split it about a reference distance R_c,

    e^{-jkR}/R = e^{-jkR_c} Σ_{n≥0} (-jk)^n (R - R_c)^n / (n! R),

and the moment of a coaxial source segment at an observer point becomes

    N_q(c, h; k) = e^{-jkR_c} Σ_{n≥0} (-jk)^n / n! · μ_{n,q}(c, h),
    μ_{n,q}(c, h) = (1/4π) ∫₀^h v^q ⟨(R - R_c)^n / R⟩(c + v) dv,

with ⟨·⟩ the average round the ring, R² = z² + β², β = 2a sin θ, θ = φ/2.
With R_c = 0 the n = 0 term is the static kernel 1/R (log-singular at
z = 0) and n ≥ 1 is the dynamic remainder (e^{-jkR} - 1)/R, which is
bounded. Every μ is REAL and depends on geometry alone, so a geometry's
coefficients are computed once and a new frequency costs one Horner
evaluation (and at most one complex exponential) per unique pair. The
derivation, the error bounds and the measurements are
`docs/design/exact-kernel-static-dynamic-split.md`; it is our own, from
public mathematics (the binomial series, the Wallis integrals, the AGM, the
trapezoid rule on a periodic integrand), and no NEC-4 / NEC-5 material was
used.

Three rules, chosen per pair from its geometry alone (`pair_kind`):

* **NEAR** -- the pair lies closer to the observer than its own length, the
  observer inside it included. R_c = 0. Even powers of R are polynomials in
  z, integrated by the Gauss-Legendre rule in v. For the odd powers the
  z-integral is done in closed form under the ring average,

      ∫ R^{-1} dz = asinh(z/β),     ∫ z R^p dz = R^{p+2}/(p+2),
      ∫ R^p dz = [z R^p + p β² ∫ R^{p-2} dz] / (p + 1),

  every endpoint difference in a form that does not cancel, and what is
  left is one integral over θ ∈ [0, π/2]. The θ rule is Gauss-Legendre on
  chunks graded geometrically out from θ = 0, the first as long as the
  nearest endpoint's |z|/2a, because the integrand's branch points sit at
  sin θ = ±i|z|/2a. When the observer is on the segment the integrand keeps
  a log singularity at θ = 0 (asinh(z/β) = sgn z [ln(|z| + R) - ln β]); the
  chunk that holds it is integrated by the product rule that is exact for
  (polynomial)·ln θ, the weights the log split of `_exact_kernel` uses.
* **CENTRED** -- at least its own length away, and at least 6a away. The
  closed form's endpoint difference cancels like c/h there, so the rule is
  Gauss-Legendre in v. R_c is the middle of the pair's range of R, so the
  series runs in k·(R - R_c), which is at most k(h/2 + a/6) however far
  along the wire the pair is: a 59 m wire's last pair needs the same few
  terms as its second. The ring average is the 12-point trapezoid rule
  (six distinct points), exact to (a²/s₀)^12 on a periodic integrand.
* **PLAIN FAR** -- at least its own length away but inside 6a (a fat
  wire's neighbours). R_c = 0, Gauss-Legendre in v on ⟨R^{n-1}⟩ from the AGM
  and the half-integer recurrence (|z| < 6a) or the ring-mean expansion
  (beyond): the two forms `_exact_kernel.ring_kernel` uses, power by power.

Term counts are fixed when a pair is built, from the bound x^n/n! at the
largest x = k·|R - R_c| the rule is used for (`X_MAX` with R_c = 0,
`X_CENTRED` otherwise). `serves` is the matching test on a group and a
wavenumber: a group whose segments are too long for those bounds at some
frequency keeps the direct rule at that frequency.

Real k is assumed nowhere in this module's arithmetic: the coefficients are
real and geometry-only, and `evaluate` takes any complex k. What does assume
it is the BOUND -- the term count is sized for |k|·|R - R_c| with no growth
from e^{Im(k) R} -- and the callers, which refuse a complex k before here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from math import comb

import numpy as np

from . import _exact_kernel as _ek

# With R_c = 0 the series is used while k·R ≤ X_MAX over the pair. The terms
# of e^{-jx} peak at x^n/n! ≈ e^x/√(2πx): at x = 4 the largest is 10.7, so
# the sum keeps all but one digit.
X_MAX = 4.0
# About its own centre a pair's series is used while k·|R - R_c| ≤ X_CENTRED.
X_CENTRED = 1.0
# A series stops where x^n/n! falls below this.
SERIES_TOL = 1e-17
# The most terms any pair can be built with (x = X_MAX needs 34).
N_CAP = 40
# Trapezoid points round the ring for a CENTRED pair (half are distinct).
N_RING = 12
# A built pair's residency bound: two moments of N_CAP real coefficients,
# plus its key, offset, centre and value at the current k.
PAIR_NBYTES = 2 * 8 * N_CAP + 96

# Switch for the solvers' use of the series, read at call time. OFF unless
# asked for (`MOMWIRE_EXACT_KERNEL_SERIES=1`, or set here by a gate that
# holds the two rules against each other): the series agrees with the direct
# rule to ~1e-15 of a moment, which is inside the drift budget but does flip
# the last printed digit of a few printouts (the PR's byte gate lists them),
# and a printed digit moving is a decision, not a side effect.
ENABLED = os.environ.get("MOMWIRE_EXACT_KERNEL_SERIES", "0") == "1"


def n_terms(x):
    """N with x^N/N! ≤ SERIES_TOL (at least 4): the series runs n = 0..N."""
    n, term = 0, 1.0
    while term > SERIES_TOL and n < N_CAP - 1:
        n += 1
        term *= x / n
    return max(n, 4)


def group_reach(h, a):
    """(r_plain, d_centred): the largest R a pair with R_c = 0 can see on a
    group whose longest segment is max(h), and the largest |R - R_c| a
    CENTRED pair can. A NEAR pair reaches under 2h along the axis; a PLAIN
    FAR one under 6a + h."""
    h = np.asarray(h, dtype=np.float64)
    hm = float(h.max()) if h.size else 0.0
    r_plain = float(np.hypot(max(2.0 * hm, _ek._T_SWITCH * a + hm), 2.0 * a))
    return r_plain, 0.5 * hm + a


def serves(k, reach):
    """Whether the series is used for a group of `group_reach` at k."""
    r_plain, d_centred = reach
    return bool(abs(k) * r_plain <= X_MAX and abs(k) * d_centred <= X_CENTRED)


# --------------------------------------------------------------------------
# the ring averages ⟨R^{n-1}⟩ / 4π, n = 0..N
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _far_binom():
    """binom((n-1)/2, i)·C(i, i/2) for i = 0, 2, .., 8: (N_CAP, 5)."""
    out = np.zeros((N_CAP, 5))
    for n in range(N_CAP):
        nu = 0.5 * (n - 1)
        b = 1.0  # binom(nu, i)
        for i in range(9):
            if i % 2 == 0:
                out[n, i // 2] = b * comb(i, i // 2)
            b = b * (nu - i) / (i + 1)
    return out


def ring_powers(z, a, N):
    """(N + 1, ...) real: ⟨R^{n-1}⟩/(4π) at |z|, n = 0..N.

    |z| < 6a: ⟨R^p⟩ = (2/π) ρ^p J_{p/2}(m), the half-integer orders from one
    AGM and the recurrence, the integer orders from the same recurrence.
    Beyond: ⟨(s₀ + ε)^{p/2}⟩ expanded about the ring mean s₀ = z² + 2a² to
    ε⁸. Both are `_exact_kernel`'s forms, taken one power of k at a time.
    """
    z = np.abs(np.asarray(z, dtype=np.float64))
    out = np.empty((N + 1,) + z.shape, dtype=np.float64)
    near = z < _ek._T_SWITCH * a
    if np.any(near):
        zn = z[near]
        rho2 = zn * zn + 4.0 * a * a
        rho = np.sqrt(rho2)
        m = 4.0 * a * a / rho2
        with np.errstate(divide="ignore"):
            Km, Em = _ek._agm_KE(zn * zn / rho2)
        Rh = [Km, Em]
        Ji = [np.full_like(m, np.pi / 2.0), (2.0 - m) * np.pi / 4.0]
        for i in range(1, (N - 1) // 2 + 2):
            nu = i - 0.5
            c0, c1, c2 = (
                2.0 * nu * (m - 1.0),
                (2.0 * nu + 1.0) * (2.0 - m),
                2.0 * nu + 2.0,
            )
            Rh.append((c0 * Rh[i - 1] + c1 * Rh[i]) / c2)
            nu = float(i)
            Ji.append(
                (
                    2.0 * nu * (m - 1.0) * Ji[i - 1]
                    + (2.0 * nu + 1.0) * (2.0 - m) * Ji[i]
                )
                / (2.0 * nu + 2.0)
            )
        scale = 2.0 / (4.0 * np.pi**2)
        rp = 1.0 / rho
        for n in range(N + 1):
            J = Rh[n // 2] if n % 2 == 0 else Ji[(n - 1) // 2]
            out[n][near] = scale * (rp * J)
            rp = rp * rho
    if not np.all(near):
        zf = z[~near]
        s0 = zf * zf + 2.0 * a * a
        r = np.sqrt(s0)
        e2 = (a * a / s0) ** 2
        B = _far_binom()
        rp = 1.0 / r
        for n in range(N + 1):
            acc = B[n, 4]
            for i in (3, 2, 1, 0):
                acc = acc * e2 + B[n, i]
            out[n][~near] = rp * acc / _ek._FOURPI
            rp = rp * r
    return out


# --------------------------------------------------------------------------
# one pair's coefficients
# --------------------------------------------------------------------------
def pair_is_near(c, h):
    """The switch between the closed form and a rule in v: the pair lies
    closer to the observer than its own length (the observer inside it
    included)."""
    if c <= 0.0 <= c + h:
        return True
    return min(abs(c), abs(c + h)) < h


NEAR, CENTRED, PLAIN_FAR = 0, 1, 2


def pair_kind(c, h, a):
    """Which of the three rules a pair takes."""
    if pair_is_near(c, h):
        return NEAR
    if min(abs(c), abs(c + h)) >= _ek._T_SWITCH * a:
        return CENTRED
    return PLAIN_FAR


def _theta_chunks(t):
    """[0, π/2] cut geometrically outward from θ = 0, the first chunk as
    long as t = |z|/2a of the nearest non-zero endpoint."""
    top = 0.5 * np.pi
    out = []
    lo, hi = 0.0, min(t, top)
    while True:
        out.append((lo, hi))
        if hi >= top:
            return out
        lo, hi = hi, min(hi * _ek._GRADE, top)


def _near_odd(c, h, a, N):
    """μ_{n,q} for even n = 0, 2, .. ≤ N (odd powers p = n - 1) and q = 0, 1
    of a NEAR pair, by the closed form in z and the graded rule in θ.
    Returns (J + 1, 2), row j for n = 2j."""
    z0, z1 = float(c), float(c) + float(h)
    J = N // 2
    s0 = (z0 > 0.0) - (z0 < 0.0)
    s1 = (z1 > 0.0) - (z1 < 0.0)
    # asinh(z1/β) - asinh(z0/β) = [..] - nlog·ln β: the observer is on or
    # inside the segment when nlog > 0, and the integrand is log-singular
    # at θ = 0 then.
    nlog = s1 - s0
    t = min(abs(v) for v in (z0, z1) if v != 0.0) / (2.0 * a)
    d2 = h * (z1 + z0)  # R1² - R0²
    acc = np.zeros((J + 1, 2))
    for lo, hi in _theta_chunks(t):
        first = nlog and lo == 0.0
        if first:
            # the chunk holding θ = 0: I = A + B ln θ, both smooth, by the
            # split rule's product weights (ln θ = ln hi + ln x)
            x, w = _ek._gl01(_ek._N_SPLIT)
            wl = hi * (w * np.log(hi) + _ek._gl01_log(_ek._N_SPLIT))
        else:
            x, w = _ek._gl01(_ek._N_GL)
        th = lo + (hi - lo) * x
        w = (hi - lo) * w
        b = 2.0 * a * np.sin(th)
        b2 = b * b
        R0 = np.sqrt(z0 * z0 + b2)
        R1 = np.sqrt(z1 * z1 + b2)
        if nlog == 0:
            # same side: ln((|zb| + Rb)/(|za| + Ra)), za the nearer end
            if z0 > 0.0:
                I0 = np.log1p(h * (1.0 + (z1 + z0) / (R1 + R0)) / (z0 + R0))
            else:
                I0 = np.log1p(h * (1.0 - (z1 + z0) / (R1 + R0)) / (R1 - z1))
        elif first:
            I0 = -nlog * np.log(b / th)
            if s1:
                I0 = I0 + np.log(z1 + R1)
            if s0:
                I0 = I0 + np.log(R0 - z0)
            B0 = np.full_like(th, -float(nlog))
        else:
            I0 = np.arcsinh(z1 / b) - np.arcsinh(z0 / b)
        # E_p = R1^p - R0^p and D_p = z1 R1^p - z0 R0^p, p odd, kept as sums
        # of same-signed terms
        E = d2 / (R1 + R0)
        R0p, R1p = R0, R1
        for j in range(J + 1):
            p = 2 * j - 1
            if j:
                D = h * R1p + z0 * E
                I0 = (D + p * b2 * I0) / (p + 1)
                if first:
                    B0 = p * b2 * B0 / (p + 1)
                E = R1 * R1 * E + R0p * d2  # E_{p+2}
                R0p, R1p = R0p * R0 * R0, R1p * R1 * R1
            I1 = E / (p + 2)
            acc[j, 0] += w @ I0
            acc[j, 1] += w @ (I1 - z0 * I0)
            if first:
                lg = wl @ B0
                acc[j, 0] += lg
                acc[j, 1] -= z0 * lg
    return acc * (2.0 / np.pi) / _ek._FOURPI


# cos φ at the six distinct points of the 12-point trapezoid rule, φ = 15°,
# 45°, .. 165°: (√6 + √2)/4, 1/√2, (√6 - √2)/4 and their negatives, written
# out so that both routes hold the same six numbers.
_RING_COS = np.array(
    [
        0.9659258262890683,
        0.7071067811865476,
        0.25881904510252074,
        -0.25881904510252074,
        -0.7071067811865476,
        -0.9659258262890683,
    ]
)
assert _RING_COS.size == N_RING // 2


def _centred(c, h, a, nq):
    """(R_c, μ) of a CENTRED pair: Gauss-Legendre in v, the trapezoid rule
    round the ring, powers of R - R_c."""
    zmin, zmax = sorted((abs(c), abs(c + h)))
    Rc = 0.5 * (zmin + float(np.hypot(zmax, 2.0 * a)))
    N = n_terms(X_CENTRED)
    xg, wg = _ek._gl01(_ek._N_GL)
    v = h * xg
    z = c + v
    R = np.sqrt((z * z + 2.0 * a * a)[:, None] - 2.0 * a * a * _RING_COS[None, :])
    d = R - Rc
    term = 1.0 / R  # (R - R_c)^n / R
    mu = np.empty((N + 1, nq))
    for n in range(N + 1):
        T = term.mean(axis=1) / _ek._FOURPI
        for q in range(nq):
            mu[n, q] = T @ (h * wg * v**q)
        term = term * d
    return Rc, mu


def point_series(c, h, a, nq=2):
    """(R_c, μ): the (N + 1, nq) real coefficients μ_{n,q}(c, h) of ONE
    observer and one coaxial source segment, and the reference distance
    they are taken about. `c` is the segment's start relative to the
    observer along the direction it runs, `h` its length
    (`_exact_kernel.point_moments`' geometry)."""
    if nq > 2:
        raise ValueError("point_series serves nq ≤ 2")
    c, h = float(c), float(h)
    kind = pair_kind(c, h, a)
    if kind == CENTRED:
        return _centred(c, h, a, nq)
    N = n_terms(X_MAX)
    xg, wg = _ek._gl01(_ek._N_GL)
    v = h * xg
    P = ring_powers(c + v, a, N)  # (N + 1, n_gl)
    mu = np.stack([P @ (h * wg * v**q) for q in range(nq)], -1)
    if kind == NEAR:
        mu[0::2, :] = _near_odd(c, h, a, N)[:, :nq]
    return 0.0, mu


def pair_reach(c, h, a):
    """The largest |R - R_c| over the pair: what k multiplies in its
    series' bound."""
    zmin, zmax = sorted((abs(c), abs(c + h)))
    r = float(np.hypot(zmax, 2.0 * a))
    if pair_kind(c, h, a) == CENTRED:
        return 0.5 * (r - zmin)
    return r


def evaluate(Rc, mu, k):
    """N_q(k) = e^{-jkR_c} Σ_n (-jk)^n/n! μ_{n,q} by Horner, highest term
    first."""
    mu = np.asarray(mu)
    acc = np.zeros(mu.shape[1:], dtype=np.complex128)
    x = -1j * k
    for n in range(mu.shape[0] - 1, 0, -1):
        acc = (acc + mu[n]) * (x / n)
    acc = acc + mu[0]
    return acc * np.exp(x * Rc) if Rc else acc


# --------------------------------------------------------------------------
# the row source (numpy reference of the C++ `ExactKernelPointSeries`)
# --------------------------------------------------------------------------
class PointSeriesRows:
    """`_exact_kernel.PointRows` with the wavenumber moved from the
    constructor to the call: `rows(x_obs, k)` returns the same (nq, m, n)
    moments, from coefficients kept per unique (c, h) for the object's
    lifetime.

    ``built`` counts the pairs whose coefficients were integrated (the
    static-cache misses), ``reused`` the unique pairs a call found already
    built, ``evaluated`` the Horner evaluations; ``kinds`` the built pairs
    by rule (NEAR, CENTRED, PLAIN_FAR)."""

    def __init__(self, x0, sgn, h, a, nq, max_cached=None):
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.sgn = np.asarray(sgn, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)
        self.n = self.x0.shape[0]
        self.a, self.nq = a, nq
        self.q = 1e-11 * float(np.max(self.h)) if self.n else 1.0
        self.reach = group_reach(self.h, a)
        self.max_cached = max_cached
        self.built = self.reused = self.evaluated = 0
        self.kinds = [0, 0, 0]
        self._cache = {}

    def rows(self, x_obs, k):
        if not serves(k, self.reach):
            raise ValueError("PointSeriesRows: k is beyond the series' bounds")
        x_obs = np.asarray(x_obs, dtype=np.float64)
        m, n, nq = x_obs.shape[0], self.n, self.nq
        c = self.sgn[None, :] * (self.x0[None, :] - x_obs[:, None])
        hh = np.broadcast_to(self.h[None, :], (m, n))
        key = np.stack([np.round(c / self.q), np.round(hh / self.q)], -1).reshape(-1, 2)
        uniq, first, inv = np.unique(
            key, axis=0, return_index=True, return_inverse=True
        )
        cf, hf = c.ravel()[first], hh.ravel()[first]
        vals = np.empty((nq, uniq.shape[0]), dtype=np.complex128)
        for u, t in enumerate(map(tuple, uniq.tolist())):
            got = self._cache.get(t)
            if got is None:
                got = point_series(cf[u], hf[u], self.a, nq)
                self.built += 1
                self.kinds[pair_kind(cf[u], hf[u], self.a)] += 1
                if self.max_cached is None or len(self._cache) < self.max_cached:
                    self._cache[t] = got
            else:
                self.reused += 1
            vals[:, u] = evaluate(*got, k)
        self.evaluated += uniq.shape[0]
        return vals[:, inv.ravel()].reshape(nq, m, n)


# --------------------------------------------------------------------------
# the C++ route (`_accel_exact_kernel_series.cpp`)
# --------------------------------------------------------------------------
_acc = _ek._acc
_HAVE_ACCEL = _acc is not None and hasattr(_acc, "exact_kernel_series_1454")


def _rules():
    xg, wg = _ek._gl01(_ek._N_GL)
    xs, ws = _ek._gl01(_ek._N_SPLIT)
    return _acc.ExactKernelSeriesRules(xg, wg, xs, ws, _ek._gl01_log(_ek._N_SPLIT))


class PointSeriesAccel:
    """`PointSeriesRows` through the C++ kernel: the same `rows(x_obs, k)`
    and counters, the cache bounded the same way, the pair keyed on the
    lesser of its mirror images (as `_exact_kernel.PointRowsAccel`). It also
    keeps the built pairs' values at the last k it was asked for, so a
    repeat call at one frequency evaluates nothing."""

    def __init__(self, x0, sgn, h, a, nq, max_cached=None, *, n_threads=0):
        h = np.asarray(h, dtype=np.float64)
        self.n = int(h.shape[0])
        self.nq = nq
        q = 1e-11 * float(np.max(h)) if self.n else 1.0
        self._impl = _acc.ExactKernelPointSeries(
            np.asarray(x0, dtype=np.float64),
            np.asarray(sgn, dtype=np.float64),
            h,
            float(a),
            int(nq),
            -1 if max_cached is None else int(max_cached),
            q,
            _rules(),
            X_MAX,
            X_CENTRED,
            SERIES_TOL,
            int(n_threads),
        )
        self.reach = group_reach(h, a)

    built = property(lambda self: int(self._impl.built))
    reused = property(lambda self: int(self._impl.reused))
    evaluated = property(lambda self: int(self._impl.evaluated))
    kinds = property(lambda self: [int(v) for v in self._impl.kinds])
    nbytes = property(lambda self: int(self._impl.nbytes))

    def rows(self, x_obs, k):
        x_obs = np.ascontiguousarray(x_obs, dtype=np.float64)
        out = np.empty((self.nq, x_obs.shape[0], self.n), dtype=np.complex128)
        self._impl.rows(x_obs, k, out)
        return out


def accel_serves(k, nq):
    """Whether the C++ series route takes a group's moments (and is switched
    on). Complex k is the callers' refusal, as for the direct route."""
    return _HAVE_ACCEL and _ek._USE_ACCEL and nq <= 2 and not np.iscomplexobj(k)


def point_series_rows(x0, sgn, h, a, k, nq, max_cached=None):
    """The series source for a group: C++ where it serves, else the numpy
    `PointSeriesRows`. `k` only chooses the route; the source serves any
    wavenumber `serves` admits."""
    if accel_serves(k, nq):
        return PointSeriesAccel(x0, sgn, h, a, nq, max_cached=max_cached)
    return PointSeriesRows(x0, sgn, h, a, nq, max_cached=max_cached)
