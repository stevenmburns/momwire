"""The series and the direct rule against 30-digit arithmetic (momwire#1454
step 3): the measured error on both sides of each switch.

    python reference.py            (needs mpmath; minutes)

The reference is N_q(c, h; k) = ∫₀^h v^q K(c + v) dv with the static kernel
in closed form, K_s(z) = K(m)/(2π²ρ) (the complete elliptic integral from
the AGM, so a z of 1e-14 a loses nothing), integrated by tanh-sinh on
breakpoints graded onto the singular point, and the dynamic remainder
(1/2π)∮(e^{-jkR} - 1)/(4πR) dφ by nested tanh-sinh. Errors are relative to
the pair's static moment, the first moment's in units of h.
"""

import mpmath as mp
import numpy as np

from momwire import _exact_kernel as ek, _exact_kernel_series as es

mp.mp.dps = 30


def ref(c, h, a, k, nq=2):
    c, h, a, k = map(mp.mpf, (c, h, a, k))

    def Ks(z):
        if z == 0:
            return mp.mpf(0)
        rho = mp.sqrt(z * z + 4 * a * a)
        return (mp.pi / (2 * mp.agm(1, abs(z) / rho))) / (2 * mp.pi**2 * rho)

    def Kd(z):
        def f(th):
            R = mp.sqrt(z * z + 4 * a * a * mp.sin(th) ** 2)
            return mp.expm1(-1j * k * R) / R if R != 0 else -1j * k

        return mp.quad(f, mp.linspace(0, mp.pi / 2, 5)) * 2 / mp.pi / (4 * mp.pi)

    vs = -c
    pts = sorted({mp.mpf(0), h} | ({vs} if 0 < vs < h else set()))
    brk = set(pts)
    for side in (-1, 1):
        d = max(abs(pts[0] - vs), abs(pts[-1] - vs))
        while d > a * mp.mpf(10) ** -14:
            p = vs + side * d
            if pts[0] < p < pts[-1]:
                brk.add(p)
            d /= 4
    brk = sorted(brk)
    out = []
    for q in range(nq):
        s = mp.quad(lambda v, q=q: v**q * Ks(c + v), brk)
        d = mp.quad(lambda v, q=q: v**q * Kd(c + v), pts) if k != 0 else 0
        out.append(complex(s + d))
    return np.array(out)


CASES = [
    # a, h/a, c/h, k
    (0.01, 0.05, -0.5, 0.15),  # fat self term
    (0.01, 0.05, -0.999, 0.15),
    (0.01, 0.2, -0.5, 0.15),
    (0.01, 1, -0.5, 1.0),
    (0.01, 1, 0.0, 0.5),  # observer on the end
    (0.01, 1, 1e-6, 0.5),
    (0.001, 100, -0.5, 1.0),  # thin self term
    (0.001, 100, 0.5, 1.0),
    (0.01, 3, 0.5, 1.0),
    (0.01, 7, 0.3, 2.0),
    # the near/far switch, c/h = 1 (d = h)
    (0.01, 0.1, 0.999, 1.0),
    (0.01, 0.1, 1.0, 1.0),
    (0.01, 1, 0.999, 1.0),
    (0.01, 1, 1.0, 1.0),
    (0.001, 30, 0.999, 1.0),
    (0.001, 30, 1.0, 1.0),
    # the 6a switch (c = 6a): plain far below, centred above
    (0.01, 3, 1.999, 1.0),
    (0.01, 3, 2.0, 1.0),
    (0.01, 1, 5.999, 1.0),
    (0.01, 1, 6.0, 1.0),
    # far along a wire
    (0.001, 100, 30, 1.0),
    (0.001, 100, 300, 1.0),
]

KIND = ("near", "centred", "plain-far")

if __name__ == "__main__":
    print(
        f"{'a':>6} {'h/a':>5} {'c/h':>7} {'k':>5} {'rule':9} {'series':>9} {'direct':>9}"
    )
    for a, hr, cr, k in CASES:
        h = hr * a
        c = cr * h
        r = ref(c, h, a, k)
        new = es.evaluate(*es.point_series(c, h, a), k)
        if es.pair_is_near(c, h):
            old = ek.point_moments(c, h, a, k, 2)
        else:
            old = ek.point_moments_far(np.array([c]), np.array([h]), a, k, 2)[:, 0]
        sc = abs(r[0]) * np.array([1.0, h])
        e_new, e_old = (np.max(abs(v - r) / sc) for v in (new, old))
        print(
            f"{a:6g} {hr:5g} {cr:7g} {k:5g} {KIND[es.pair_kind(c, h, a)]:9} "
            f"{e_new:9.1e} {e_old:9.1e}",
            flush=True,
        )
