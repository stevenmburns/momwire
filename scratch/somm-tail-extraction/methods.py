"""The four ways of evaluating one below/below node, instrumented.

    base   the current engine: `_run_contour` as `_six_integrals_below`'s
           numpy path calls it (Gauss-24 head, Bessel-zero tail, quiet test,
           Wynn fallback), unchanged
    wa     the same head and panels, with the tail's partial sums
           extrapolated by the Mosig-Michalski weighted-averages method
           (Sidi's W-algorithm with analytic remainder estimates)
    extN   N large-lambda terms subtracted and added back in closed form
           (`extract.Extraction`), the residual on the current tail
    extN+wa  both

Every method counts the lambda points it evaluates; a point costs one
evaluation of the six integrands (and, under extraction, of the subtracted
terms, which reuse the same Bessel calls in a production kernel).
"""

from __future__ import annotations

import math
import time

import numpy as np

from momwire import _sommerfeld_below as SB

import extract as X

GX, GW = SB._GX, SB._GW


class Counter:
    def __init__(self, f):
        self.f = f
        self.points = 0

    def __call__(self, lam):
        lam = np.asarray(lam)
        self.points += lam.size
        return self.f(lam)


def _sidi_w(xi, sums, omega):
    """Sidi's W-algorithm: S_n = S + omega_n * sum_i c_i xi_n^-i, solved for
    S from all points given (elementwise over the trailing axis)."""
    t = 1.0 / np.asarray(xi, float)
    Mk = sums / omega
    Nk = 1.0 / omega
    n = len(t)
    for k in range(1, n):
        dt = (t[k:] - t[: n - k])[:, None]
        Mk = (Mk[1:] - Mk[:-1]) / dt
        Nk = (Nk[1:] - Nk[:-1]) / dt
    return (Mk / Nk)[0]


def tail(
    f,
    a,
    rho,
    h,
    scale,
    rtol,
    *,
    mode,
    alpha=None,
    max_panels=SB._MAX_TAIL_PANELS,
    wa_order=10,
):
    """`_tail_below`'s panel walk; `mode` "quiet" is its own semantics (quiet
    test, Wynn fallback), "wa" adds the weighted-averages estimate and stops
    when two successive estimates agree to rtol * scale."""
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
    xs, ss, idx = [], [], []  # lattice-aligned partial sums for WA
    est_prev = None
    agree = 0
    used_wa = False
    while parts < max_panels:
        zl = next_zero(z)
        znext = min(zl, z + dl_decay)
        contrib = SB._adaptive_segment(f, z, znext, rtol, 6, GX, GW, ref=scale)
        acc = contrib if acc is None else acc + contrib
        sums.append(acc.copy())
        z = znext
        parts += 1
        cmax = float(np.max(np.abs(contrib)))
        smax = max(float(np.max(np.abs(acc))), scale, 1e-300)
        if cmax == 0.0 or cmax < rtol * smax:
            quiet += 1
            if quiet >= 2:
                converged = True
                break
        else:
            quiet = 0
        if mode == "wa" and znext == zl and np.isfinite(lat):
            xs.append(z)
            ss.append(acc.copy())
            idx.append(int(round((z - off) / lat)))
            if len(xs) >= 3:
                k = min(len(xs), wa_order + 1)
                xi = np.array(xs[-k:])
                sign = np.where(np.array(idx[-k:]) % 2 == 0, 1.0, -1.0)
                # remainder estimate omega_n = (-1)^n xi^(alpha - 1/2) e^{-xi h}
                om = (
                    sign[:, None]
                    * xi[:, None] ** (alpha[None, :] - 0.5)
                    * np.exp(-xi[:, None] * h)
                )
                est = _sidi_w(xi, np.array(ss[-k:]), om)
                if est_prev is not None and len(xs) >= 4:
                    d = float(np.max(np.abs(est - est_prev)))
                    if d < rtol * smax:
                        agree += 1
                        if agree >= 2:
                            acc = est
                            converged = True
                            used_wa = True
                            break
                    else:
                        agree = 0
                est_prev = est
    accel = False
    if not converged and len(sums) >= 8:
        acc = SB._wynn_epsilon(np.array(sums[-min(len(sums), 41) :]))
        accel = True
    return acc, parts, converged, accel or used_wa


def soil(name):
    if name == "dan":
        return 6.0078e6, complex(13.0, -14.96)
    if name == "salt1.8":
        return 1.8e6, X.eps_tilde(81.0, 5.0, 1.8e6)
    if name == "fresh28":
        return 28.0e6, X.eps_tilde(80.0, 0.001, 28.0e6)
    if name == "salt28":
        return 28.0e6, X.eps_tilde(81.0, 5.0, 28.0e6)
    if name == "fresh1.8":
        return 1.8e6, X.eps_tilde(80.0, 0.001, 1.8e6)
    raise ValueError(name)


def node(soil_name, r1_wl, theta_deg):
    f, et = soil(soil_name)
    kp = X.k_free(f)
    km = SB.k_medium(et, kp)
    lam_m = 2.0 * np.pi / abs(km)
    th = math.radians(theta_deg)
    return kp, km, r1_wl * lam_m * math.cos(th), r1_wl * lam_m * math.sin(th)


def bessel_exp():
    # amplitude exponent of B_i(lam rho) at large lam
    return np.array([-0.5, -0.5, -0.5, -1.5, -0.5, -0.5])


def run(
    kp,
    km,
    rho,
    h,
    method,
    *,
    N=0,
    rtol=1e-11,
    max_panels=SB._MAX_TAIL_PANELS,
    b_fac=1.0,
):
    """One node by one method -> dict(value, tail_panels, head_panels, points,
    wall, converged, accel)."""

    def f6(lam):
        return SB._integrand_six_below(lam, rho, h, kp, km)

    t0 = time.perf_counter()
    if method == "base":
        cf = Counter(f6)
        val, hp, tp, conv, accel = SB._run_contour(
            cf,
            kp,
            km,
            rho,
            h,
            rtol,
            SB._ADAPT_DEPTH,
            SB._DETOUR,
            GX,
            GW,
            max_panels=max_panels,
        )
        return dict(
            value=val,
            tail_panels=tp,
            head_panels=hp,
            points=cf.points,
            wall=time.perf_counter() - t0,
            converged=bool(conv),
            accel=bool(accel),
        )
    a = 1.1 * max(abs(kp), abs(km))
    alpha0 = np.array([0.0, 0.0, 0.0, 0.0, -2.0, 0.0])
    if N > 0:
        ex = X.Extraction(kp, km, rho, h, N, b=b_fac / a)

        def fres(lam):
            return f6(lam) - ex(lam)

        cf = Counter(fres)
        add = ex.value
        alpha = ex.alpha
    else:
        cf = Counter(f6)
        add = np.zeros(6, dtype=np.complex128)
        alpha = alpha0
    marks = (kp, abs(km.real))
    head, hp = SB._head(cf, a, rho, marks, rtol, SB._ADAPT_DEPTH, SB._DETOUR, GX, GW)
    # The tail's relative test is against the ANSWER's scale; under
    # extraction the head of the residual can sit far below it.
    scale = float(np.max(np.abs(head + add)))
    mode = "wa" if method.endswith("wa") else "quiet"
    tl, tp, conv, accel = tail(
        cf,
        a,
        rho,
        h,
        scale,
        rtol,
        mode=mode,
        alpha=alpha + bessel_exp(),
        max_panels=max_panels,
    )
    return dict(
        value=head + tl + add,
        tail_panels=tp,
        head_panels=hp,
        points=cf.points,
        wall=time.perf_counter() - t0,
        converged=bool(conv),
        accel=bool(accel),
    )
