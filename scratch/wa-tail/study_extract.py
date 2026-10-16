"""Large-lambda extraction for the below/below Sommerfeld remainder.

Each of the six integrands of `_integrand_six_below` is written as

    f_i(lam) = g_i(u) * exp(-lam h) * B_i(lam rho),      u = 1/lam,

where B_i is one of J0, J1, J1/x or (J1/x - J0), and g_i is a power series
in u whose coefficients depend on (k_p, k_m, h) only. The first N terms of
that series, each regularised at lam -> 0 by (1 - exp(-lam b))^M, are
subtracted from the integrand everywhere on the contour, and their exact
integrals over [0, inf) are added back in closed form. REPORT.md has the
derivation.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import comb, jv

# ---------------------------------------------------------------------------
# Truncated power series in u, complex coefficients
# ---------------------------------------------------------------------------


class PS:
    """c[0] + c[1] u + ... + c[n-1] u^(n-1), truncated at order n."""

    def __init__(self, c, n):
        c = np.zeros(n, dtype=np.complex128) if c is None else np.asarray(c, complex)
        self.c = np.zeros(n, dtype=np.complex128)
        m = min(n, c.size)
        self.c[:m] = c[:m]
        self.n = n

    @classmethod
    def const(cls, v, n):
        return cls([v], n)

    def __add__(self, o):
        o = o if isinstance(o, PS) else PS.const(o, self.n)
        return PS(self.c + o.c, self.n)

    __radd__ = __add__

    def __neg__(self):
        return PS(-self.c, self.n)

    def __sub__(self, o):
        return self + (-(o if isinstance(o, PS) else PS.const(o, self.n)))

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if not isinstance(o, PS):
            return PS(self.c * o, self.n)
        return PS(np.convolve(self.c, o.c)[: self.n], self.n)

    __rmul__ = __mul__

    def inv(self):
        a = self.c
        if a[0] == 0:
            raise ZeroDivisionError("series with zero constant term")
        b = np.zeros(self.n, dtype=np.complex128)
        b[0] = 1.0 / a[0]
        for k in range(1, self.n):
            b[k] = -np.dot(a[1 : k + 1], b[k - 1 :: -1][:k]) / a[0]
        return PS(b, self.n)

    def __truediv__(self, o):
        if not isinstance(o, PS):
            return PS(self.c / o, self.n)
        return self * o.inv()

    def __rtruediv__(self, o):
        return PS.const(o, self.n) * self.inv()

    def shift(self, s):
        """Multiply by u^s (s may be negative only if the low terms are 0)."""
        c = np.zeros(self.n, dtype=np.complex128)
        if s >= 0:
            c[s:] = self.c[: self.n - s]
        else:
            if np.any(self.c[:-s] != 0):
                raise ValueError("negative shift drops nonzero terms")
            c[: self.n + s] = self.c[-s:]
        return PS(c, self.n)

    def exp(self):
        """exp of a series with zero constant term."""
        if self.c[0] != 0:
            raise ValueError("exp needs a zero constant term")
        out = PS.const(1.0, self.n)
        term = PS.const(1.0, self.n)
        for j in range(1, self.n):
            term = term * self / j
            out = out + term
        return out


def sqrt_one_minus(k, n):
    """s = sqrt(1 - k^2 u^2) as a series (gamma = lam * s on the cut-free
    branch: Re gamma > 0 for real lam -> +inf)."""
    c = np.zeros(n, dtype=np.complex128)
    x = complex(k) ** 2
    for j in range(0, (n + 1) // 2):
        if 2 * j >= n:
            break
        # binom(1/2, j) (-x)^j
        c[2 * j] = _binom_half(j) * (-x) ** j
    return PS(c, n)


def _binom_half(j):
    out = 1.0
    for i in range(j):
        out *= (0.5 - i) / (i + 1)
    return out


# Which Bessel combination each integrand carries, and its sign/role.
BESSEL = ("J1x_minus_J0", "J0", "J1", "J1x", "J0", "J0")


def coefficients(k_p, k_m, h, order):
    """Series g_i(u) (6 PS objects) such that integrand_i =
    g_i(1/lam) * exp(-lam h) * B_i(lam rho) exactly as a formal series.

    `order` is how many u-powers to carry (enough for the leading term of
    every integrand plus the extracted N).
    """
    n = order + 4  # headroom for the u^-2 shifts below
    sp = sqrt_one_minus(k_p, n)
    sm = sqrt_one_minus(k_m, n)
    kp2 = complex(k_p) ** 2
    km2 = complex(k_m) ** 2
    S = kp2 + km2
    # D1 = u * d1s, D2 = u * d2s (gamma_x = lam * s_x)
    d1s = 2.0 / (sp + sm) - (2.0 * km2 / S) / sm
    d2s = 2.0 / (kp2 * sm + km2 * sp) - (2.0 / S) / sm
    # D2 = O(lam^-3) exactly (REPORT.md, eq. D2): the u^0 and u^1 terms
    # cancel analytically and are roundoff here.
    d2s.c[:2] = 0.0
    # exp(-gamma_m h) = exp(-lam h) * exp(-h (s_m - 1)/u)
    E = ((sm - 1.0).shift(-1) * (-h)).exp()
    g = [
        (d2s * E).shift(-2),  # u D2 lam^3          -> u^-2 d2s
        (d2s * sm * sm * E).shift(-2),  # u D2 lam^3 s^2
        -(d2s * sm * E).shift(-2),  # -u D2 lam^3 s
        -(d2s * E).shift(-2),  # -u D2 lam^3
        d2s * E,  # u D2 lam
        d1s * E,  # u D1 lam
    ]
    return [PS(gi.c[:order], order) for gi in g]


# ---------------------------------------------------------------------------
# Closed forms: Phi_m(t) "=" int_0^inf lam^-m exp(-lam t) J0(lam rho) dlam,
# defined up to a polynomial in t of degree m-1, with dPhi_m/dt = -Phi_{m-1}.
# ---------------------------------------------------------------------------


def _L(t, rho):
    R = np.hypot(rho, t)
    return np.log(t + R), R


def phi(m, t, rho):
    L, R = _L(t, rho)
    if m == 0:
        return 1.0 / R
    if m == 1:
        return -L
    if m == 2:
        return t * L - R
    if m == 3:
        return -((0.5 * t * t - 0.25 * rho * rho) * L - 0.75 * t * R)
    if m == 4:
        return (
            L * (t**3 / 6.0 - rho * rho * t / 4.0)
            + 5.0 * rho * rho * R / 12.0
            - 11.0 * R**3 / 36.0
        )
    if m == 5:
        # dPhi5/dt = -Phi4; integrated by parts (see REPORT.md)
        return -(
            L * (t**4 / 24.0 - rho * rho * t * t / 8.0 + rho**4 / 64.0)
            - R * (25.0 * t**3 / 288.0 - 55.0 * rho * rho * t / 576.0)
        )
    raise ValueError(m)


def dphi_drho(m, t, rho):
    L, R = _L(t, rho)
    dL = rho / (R * (t + R))
    dR = rho / R
    if m == 0:
        return -rho / R**3
    if m == 1:
        return -dL
    if m == 2:
        return -rho / (t + R)
    if m == 3:
        return -(
            (-0.5 * rho) * L + (0.5 * t * t - 0.25 * rho * rho) * dL - 0.75 * t * dR
        )
    if m == 4:
        return (
            L * (-rho * t / 2.0)
            + dL * (t**3 / 6.0 - rho * rho * t / 4.0)
            + 5.0 * rho * R / 6.0
            + 5.0 * rho * rho * dR / 12.0
            - 11.0 * R * R * dR / 12.0
        )
    if m == 5:
        return -(
            L * (-rho * t * t / 4.0 + rho**3 / 16.0)
            + dL * (t**4 / 24.0 - rho * rho * t * t / 8.0 + rho**4 / 64.0)
            - dR * (25.0 * t**3 / 288.0 - 55.0 * rho * rho * t / 576.0)
            - R * (-110.0 * rho * t / 576.0)
        )
    raise ValueError(m)


def _psi(kind, n, t, rho):
    """The divergent-up-to-polynomial value of int lam^-n e^{-lam t} B."""
    if kind == "J0":
        return phi(n, t, rho)
    if kind == "J1":  # J1(lam rho) = -(1/lam) d/drho J0(lam rho)
        return -dphi_drho(n + 1, t, rho)
    if kind == "J1x":  # J1/(lam rho)
        return -dphi_drho(n + 2, t, rho) / rho
    if kind == "J1x_minus_J0":
        return _psi("J1x", n, t, rho) - _psi("J0", n, t, rho)
    raise ValueError(kind)


def reg_order(kind, n):
    """M for (1 - e^{-lam b})^M: enough to annihilate Phi's polynomial
    ambiguity (degree <= the Phi index - 1) in the finite difference."""
    m = {"J0": n, "J1": n + 1, "J1x": n + 2, "J1x_minus_J0": n + 2}[kind]
    # A convergent integral needs no regulariser at all.
    convergent = {
        "J0": n == 0,
        "J1": n <= 1,
        "J1x": n == 0,
        "J1x_minus_J0": n == 0,
    }[kind]
    return 0 if convergent else m


def closed(kind, n, h, rho, b):
    """int_0^inf lam^-n (1 - e^{-lam b})^M e^{-lam h} B(lam rho) dlam."""
    M = reg_order(kind, n)
    if M == 0:
        return _psi(kind, n, h, rho)
    tot = 0.0
    for j in range(M + 1):
        tot = tot + ((-1) ** j) * comb(M, j, exact=True) * _psi(kind, n, h + j * b, rho)
    return tot


def bessel(kind, x):
    x = np.asarray(x, dtype=np.complex128)
    small = np.abs(x) < 1e-6
    xs = np.where(small, 1.0, x)
    j0 = np.where(small, 1.0 - 0.25 * x * x, jv(0, xs))
    j1 = np.where(small, 0.5 * x, jv(1, xs))
    j1x = np.where(small, 0.5 - x * x / 16.0, jv(1, xs) / xs)
    return {"J0": j0, "J1": j1, "J1x": j1x, "J1x_minus_J0": j1x - j0}[kind]


class Extraction:
    """The subtracted terms for one (rho, h) node, and their closed forms."""

    def __init__(self, k_p, k_m, rho, h, N, b):
        self.rho, self.h, self.b, self.N = rho, h, b, N
        g = coefficients(k_p, k_m, h, order=N + 3)
        self.terms = []  # (component, kind, n, coeff)
        for i in range(6):
            c = g[i].c
            nz = np.nonzero(np.abs(c) > 0)[0]
            n0 = int(nz[0]) if nz.size else 0
            for n in range(n0, n0 + N):
                if n < c.size and c[n] != 0:
                    self.terms.append((i, BESSEL[i], n, c[n]))
        self.value = np.zeros(6, dtype=np.complex128)
        for i, kind, n, cn in self.terms:
            self.value[i] += cn * closed(kind, n, h, rho, b)
        # leading residual power per component (for WA's exponent)
        self.alpha = np.zeros(6)
        for i in range(6):
            c = g[i].c
            nz = np.nonzero(np.abs(c) > 0)[0]
            n0 = int(nz[0]) if nz.size else 0
            self.alpha[i] = -(n0 + N)

    def __call__(self, lam):
        lam = np.asarray(lam, dtype=np.complex128)
        e = np.exp(-lam * self.h)
        out = np.zeros((6,) + lam.shape, dtype=np.complex128)
        cache = {}
        for i, kind, n, cn in self.terms:
            if kind not in cache:
                cache[kind] = bessel(kind, lam * self.rho)
            M = reg_order(kind, n)
            reg = (-np.expm1(-lam * self.b)) ** M if M else 1.0
            out[i] += cn * lam ** (-n) * reg * e * cache[kind]
        return out


def eps_tilde(eps_r, sigma, f_hz):
    eps0 = 8.8541878128e-12
    return complex(eps_r, -sigma / (2.0 * math.pi * f_hz * eps0))


def k_free(f_hz):
    return 2.0 * math.pi * f_hz / 299792458.0
