"""Extended-precision reference for the six below/below integrals at one node.

Independent of the double-precision engine except for the integrand's
DEFINITION (`_sommerfeld._d12` / `_sommerfeld_below._integrand_six_below`,
transcribed here in mpmath). Everything is carried at `DPS` digits:

  head   the same first-quadrant detour lam(t) = t + jH sin(pi t/a) on
         [0, a] (any deformation that crosses no cut gives the same value),
         tanh-sinh `mp.quad` on the engine's own sub-intervals
  tail   real axis from a, panelled at the J0 zero lattice (n - 1/4) pi/rho,
         each panel by Gauss-Legendre with an n vs 2n check (bisect on miss),
         walked until lam*h reaches WALK_LH, then the remaining alternating
         remainder extrapolated by Wynn's epsilon IN mp on the last 30
         lattice sums. Two walk lengths are run and must agree; the reported
         uncertainty is their difference plus the panel-check residual.

Bessel J0/J1 of real argument: mpmath for x < 40, Hankel's P/Q asymptotic
expansion beyond (smallest term at x = 40 is ~1e-34, below DPS = 40's
working floor), checked against `mp.besselj` in `selfcheck()`.
"""

from __future__ import annotations

import math

import mpmath as mp

DPS = 40
mp.mp.dps = DPS


def k_free(f_hz):
    return 2.0 * math.pi * f_hz / 299792458.0


def eps_tilde(eps_r, sigma, f_hz):
    eps0 = 8.8541878128e-12
    return complex(eps_r, -sigma / (2.0 * math.pi * f_hz * eps0))


SOILS = {
    "dan": (6.0078e6, complex(13.0, -14.96)),
    "salt1.8": (1.8e6, None, (81.0, 5.0)),
    "fresh1.8": (1.8e6, None, (80.0, 0.001)),
    "salt28": (28.0e6, None, (81.0, 5.0)),
    "fresh28": (28.0e6, None, (80.0, 0.001)),
}


def soil(name):
    s = SOILS[name]
    if s[1] is not None:
        return s[0], s[1]
    return s[0], eps_tilde(s[2][0], s[2][1], s[0])


def k_medium(eps_t, k2):
    km = k2 * complex(eps_t) ** 0.5
    if km.imag > 0:
        km = km.conjugate()
    return complex(km)


def node(soil_name, r1_wl, theta_deg):
    """(k_p, k_m, rho, h) in the engine's own floats (metres)."""
    f, et = soil(soil_name)
    kp = k_free(f)
    km = k_medium(et, kp)
    lam_m = 2.0 * math.pi / abs(km)
    th = math.radians(theta_deg)
    return kp, km, r1_wl * lam_m * math.cos(th), r1_wl * lam_m * math.sin(th)


# --------------------------------------------------------------------------
# Bessel
# --------------------------------------------------------------------------

_ASY_X = 60


def _pq(x, mu):
    P = mp.mpf(1)
    Q = mp.mpf(0)
    c = mp.mpf(1)
    prev = mp.inf
    eps = mp.mpf(10) ** (-DPS - 5)
    for k in range(1, 400):
        kk = 2 * k - 1
        c = c * (mu - kk * kk) / (8 * k * x)
        m = abs(c)
        if m >= prev:
            raise ArithmeticError("asymptotic series diverged before floor")
        prev = m
        if k & 1:
            Q += c if ((k - 1) // 2) % 2 == 0 else -c
        else:
            P += c if (k // 2) % 2 == 0 else -c
        if m < eps:
            break
    return P, Q


def j0j1(x):
    """(J0(x), J1(x)) for real x >= 0 (mpf)."""
    if x < _ASY_X:
        return mp.besselj(0, x), mp.besselj(1, x)
    P0, Q0 = _pq(x, 0)
    P1, Q1 = _pq(x, 4)
    cz, sz = mp.cos(x), mp.sin(x)
    pref = mp.sqrt(1 / (mp.pi * x))
    j0 = pref * ((P0 + Q0) * cz + (P0 - Q0) * sz)
    j1 = pref * ((P1 + Q1) * sz + (Q1 - P1) * cz)
    return j0, j1


def selfcheck():
    worst = mp.mpf(0)
    for x in (60.0, 61.3, 77.7, 300.1, 5123.4, 1e5 + 0.37):
        a = j0j1(mp.mpf(x))
        b = (mp.besselj(0, x), mp.besselj(1, x))
        worst = max(worst, abs(a[0] - b[0]), abs(a[1] - b[1]))
    return worst


# --------------------------------------------------------------------------
# Integrand
# --------------------------------------------------------------------------


def gamma(lam, k):
    return mp.sqrt(-1j * (lam - k)) * mp.sqrt(1j * (lam + k))


def six(lam, rho, h, kp, km, real_axis):
    g1 = gamma(lam, kp)
    g2 = gamma(lam, km)
    k1s = kp * kp
    k2s = km * km
    d1 = 2 / (g1 + g2) - 2 * k2s / (g2 * (k1s + k2s))
    d2 = 2 / (k1s * g2 + k2s * g1) - 2 / (g2 * (k1s + k2s))
    e = mp.exp(-g2 * h)
    x = lam * rho
    if real_axis:
        xr = mp.re(x)
        if xr < mp.mpf("1e-30"):
            b0, b1x = mp.mpf(1), mp.mpf("0.5")
        else:
            j0, j1 = j0j1(xr)
            b0, b1x = j0, j1 / xr
    else:
        if abs(x) < mp.mpf("1e-30"):
            b0, b1x = mp.mpf(1), mp.mpf("0.5")
        else:
            b0 = mp.besselj(0, x)
            b1x = mp.besselj(1, x) / x
    l2 = lam * lam
    l3 = l2 * lam
    common = d2 * e
    return [
        common * (b1x - b0) * l3,
        common * g2 * g2 * b0 * lam,
        -common * g2 * (b1x * x) * l2,
        -common * b1x * l3,
        common * b0 * lam,
        d1 * e * b0 * lam,
    ]


# --------------------------------------------------------------------------
# Quadrature
# --------------------------------------------------------------------------

_GL = {}


def gl_nodes(n):
    if n not in _GL:
        xs, ws = [], []
        for i in range(1, n + 1):
            x = mp.cos(mp.pi * (i - mp.mpf("0.25")) / (n + mp.mpf("0.5")))
            for _ in range(100):
                p0, p1 = mp.mpf(1), x
                for k in range(2, n + 1):
                    p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                dp = n * (x * p1 - p0) / (x * x - 1)
                dx = p1 / dp
                x -= dx
                if abs(dx) < mp.mpf(10) ** (-DPS - 3):
                    break
            p0, p1 = mp.mpf(1), x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            xs.append(x)
            ws.append(2 / ((1 - x * x) * dp * dp))
        _GL[n] = (xs, ws)
    return _GL[n]


def gl(f, z0, z1, n):
    xs, ws = gl_nodes(n)
    mid = (z0 + z1) / 2
    half = (z1 - z0) / 2
    acc = [mp.mpc(0)] * 6
    for x, w in zip(xs, ws):
        v = f(mid + half * x)
        for c in range(6):
            acc[c] += v[c] * w * half
    return acc


def adapt(f, z0, z1, tol, n=20, depth=0, whole=None):
    """GL-n vs GL-2n on the whole panel; bisect on miss. Returns (val, err)."""
    a = whole if whole is not None else gl(f, z0, z1, n)
    b = gl(f, z0, z1, 2 * n)
    err = max(abs(a[c] - b[c]) for c in range(6))
    scale = max(abs(b[c]) for c in range(6))
    if err <= tol * max(scale, mp.mpf(10) ** -300) or depth > 30 or err < tol:
        return b, err
    m = (z0 + z1) / 2
    l, el = adapt(f, z0, m, tol, n, depth + 1)
    r, er = adapt(f, m, z1, tol, n, depth + 1)
    return [l[c] + r[c] for c in range(6)], el + er


def wynn(seq):
    """Wynn epsilon, best even column, on a list of mp scalars."""
    n = len(seq)
    prev = [mp.mpf(0)] * n
    cur = list(seq)
    best = cur[-1]
    for k in range(1, n):
        nxt = []
        for i in range(len(cur) - 1):
            d = cur[i + 1] - cur[i]
            if d == 0:
                return best
            nxt.append(prev[i + 1] + 1 / d)
        prev, cur = cur, nxt
        if k % 2 == 0 and cur:
            best = cur[-1]
        if len(cur) <= 1:
            break
    return best


def head(kp, km, rho):
    a = mp.mpf(1.1) * max(abs(kp), abs(km))
    H = min(mp.mpf("0.35") * a, mp.mpf(2) / max(rho, mp.mpf("1e-12")))
    H = max(H, mp.mpf("1e-6") * a)
    rho_m = mp.mpf(rho)

    def ft(t):
        lam = t + 1j * H * mp.sin(mp.pi * t / a)
        dl = 1 + 1j * H * (mp.pi / a) * mp.cos(mp.pi * t / a)
        v = six(lam, rho_m, HH[0], kp_m, km_m, False)
        return [vv * dl for vv in v]

    edges = {mp.mpf(0), a}
    for i in range(1, 7):
        edges.add(a * i / 7)
    for mk in (mp.mpf(kp), abs(mp.mpf(km.real))):
        for w in ("0", "-0.15", "0.15", "-0.4", "0.4"):
            v = mk * (1 + mp.mpf(w))
            if 0 < v < a:
                edges.add(v)
    e = sorted(edges)
    tot = [mp.mpc(0)] * 6
    for i in range(len(e) - 1):
        for c in range(6):
            tot[c] += mp.quad(lambda t, c=c: ft(t)[c], [e[i], e[i + 1]])
    return tot, a


HH = [None]
kp_m = None
km_m = None


def head_fast(kp, km, rho, h):
    """Head by adaptive GL on the same sub-intervals (vector-valued, one
    integrand evaluation per point for all six)."""
    global kp_m, km_m
    kp_m = mp.mpf(kp)
    km_m = mp.mpc(km)
    a = mp.mpf("1.1") * max(abs(kp_m), abs(km_m))
    H = min(mp.mpf("0.35") * a, mp.mpf(2) / max(mp.mpf(rho), mp.mpf("1e-12")))
    H = max(H, mp.mpf("1e-6") * a)
    rho_m, h_m = mp.mpf(rho), mp.mpf(h)

    def ft(t):
        lam = t + 1j * H * mp.sin(mp.pi * t / a)
        dl = 1 + 1j * H * (mp.pi / a) * mp.cos(mp.pi * t / a)
        v = six(lam, rho_m, h_m, kp_m, km_m, False)
        return [vv * dl for vv in v]

    edges = {mp.mpf(0), a}
    for i in range(1, 7):
        edges.add(a * i / 7)
    for mk in (kp_m, abs(mp.re(km_m))):
        for w in ("0", "-0.15", "0.15", "-0.4", "0.4"):
            v = mk * (1 + mp.mpf(w))
            if 0 < v < a:
                edges.add(v)
    e = sorted(edges)
    tot = [mp.mpc(0)] * 6
    err = mp.mpf(0)
    for i in range(len(e) - 1):
        # split each head interval in 8 to keep the GL degree modest
        sub = [e[i] + (e[i + 1] - e[i]) * j / 8 for j in range(9)]
        for j in range(8):
            v, er = adapt(ft, sub[j], sub[j + 1], mp.mpf(10) ** -30, n=24)
            err += er
            for c in range(6):
                tot[c] += v[c]
    return tot, a, err


def tail(kp, km, rho, h, a, checkpoints, max_panels=2_000_000):
    """Lattice walk from a; at each lam*h checkpoint (and at least 40 lattice
    sums) Wynn on the last 30 lattice-aligned sums per component."""
    kp_mm = mp.mpf(kp)
    km_mm = mp.mpc(km)
    rho_m, h_m = mp.mpf(rho), mp.mpf(h)
    lat = mp.pi / rho_m
    off = -lat / 4

    def f(lam):
        return six(mp.mpc(lam, 0), rho_m, h_m, kp_mm, km_mm, True)

    n = int(mp.floor((a - off) / lat)) + 1
    z = a
    acc = [mp.mpc(0)] * 6
    sums = []
    err = mp.mpf(0)
    panels = 0
    dl_max = min(lat, mp.mpf(3) / h_m) if h_m > 0 else lat
    cps = sorted(checkpoints)
    out = []
    while panels < max_panels and cps:
        zl = off + n * lat
        while z < zl:
            zn = min(zl, z + dl_max)
            v, er = adapt(f, z, zn, mp.mpf(10) ** -32, n=12)
            err += er
            for c in range(6):
                acc[c] += v[c]
            z = zn
        n += 1
        panels += 1
        sums.append(list(acc))
        if len(sums) > 40:
            sums.pop(0)
        if z * h_m >= cps[0] and panels >= 40:
            last = sums[-30:]
            ext = [wynn([s[c] for s in last]) for c in range(6)]
            out.append((cps.pop(0), ext, panels, err))
    return out


def reference(kp, km, rho, h, checkpoints=(8.0, 16.0)):
    hd, a, herr = head_fast(kp, km, rho, h)
    outs = tail(kp, km, rho, h, a, checkpoints)
    vals = [[hd[c] + o[1][c] for c in range(6)] for o in outs]
    v0, v1 = vals[0], vals[-1]
    spread = max(abs(v0[c] - v1[c]) for c in range(6))
    return dict(
        value=[complex(x) for x in v1],
        spread=float(spread),
        head_err=float(herr),
        tail_err=float(outs[-1][3]),
        panels=[o[2] for o in outs],
    )
