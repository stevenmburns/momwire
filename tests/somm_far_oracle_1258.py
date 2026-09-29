"""An independent evaluation of the four ground-remainder surfaces at large R1
(momwire#1258): the oracle `_sommerfeld.far_surfaces` and the grid's
continuation past its edge are measured against.

Independent of `_sommerfeld` in everything but the equations. The six
integrals of the NEC-2 theory manual (eqs 148-155; docs/nec2_theory_manual.pdf
§IV.1-IV.2) are restated here from the manual, not imported, and integrated
along the REAL lambda axis in the Bessel (J0) form by brute-force composite
Gauss-Legendre -- where `_six_integrals` deforms onto the fig-13/fig-14
contours with adaptive quadrature and, past ~20 wavelengths, loses the answer
to the Hankel form's e^{0.2 k2 rho} growth. The real axis is a valid path for
a passive ground (every branch point and the pole sit on or below it, and
gamma = sqrt(lambda^2 - k^2) is continuous along it on the principal branch).
Three details make it converge:

* the 1/gamma2 singularity at lambda = k2 is removed by lambda = k2 -+ t^2 on
  a neighbourhood either side of it;
* panels are sized to the faster of the J0 oscillation (2 pi/rho) and the
  e^{-gamma2 h} one below k2 (2 pi/h);
* the tail stops where e^{-gamma2 h} has decayed by e^{-45}, so the method
  needs h > 0 -- both points off the plane, which every query here is.

Measured against `iv_surfaces_direct` where that is sound (R1 <= 10
wavelengths): 1e-10 of scale, and doubling the panel density moves the answer
by ~1e-11 (`tests/test_somm_far_1258.py` holds both). Cost ~0.01-1 s a point.

Run as a script it prints the oracle against the served surfaces on the
#1257 test bed's ground, the table the momwire#1258 report quotes.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0, j1

MU0 = 4e-7 * np.pi
C_LIGHT = 299792458.0
_GX, _GW = np.polynomial.legendre.leggauss(32)
KEYS = ("IrhoV", "IzV", "IrhoH", "IphiH")


def _gamma(lam, k):
    # Principal branch on the real axis: +j sqrt(k^2 - lam^2) below k (the
    # outgoing wave), positive real above it; for a lossy k1 (Im k1 < 0) the
    # radicand has Im > 0 and the root Re > 0 throughout.
    return np.sqrt(lam.astype(complex) ** 2 - k * k + 0j)


def surfaces(eps_t, k2, R1, theta, density=1.0, omega=None, mu=MU0):
    """(I_rho^V, I_z^V, I_rho^H, I_phi^H) at one (R1, theta), dict by name.

    Same normalization as `_sommerfeld.iv_surfaces_direct` (eqs 156-159, the
    eq-123 C1 with omega = k2 c by default). `density` scales the panel count
    (the convergence check doubles it).
    """
    eps_t = complex(eps_t)
    k1 = k2 * np.sqrt(eps_t)
    if k1.imag > 0:
        k1 = np.conj(k1)
    k1s, k2s = k1 * k1, k2 * k2
    rho = R1 * np.cos(theta)
    h = R1 * np.sin(theta)
    if not h > 0.0:
        raise ValueError("the real-axis oracle needs h = R1 sin(theta) > 0")

    def integrands(lam):
        g1 = _gamma(lam, k1)
        g2 = _gamma(lam, k2)
        # eqs 154-155
        d1 = 2.0 / (g1 + g2) - 2.0 * k2s / (g2 * (k1s + k2s))
        d2 = 2.0 / (k1s * g2 + k2s * g1) - 2.0 / (g2 * (k1s + k2s))
        e = np.exp(-g2 * h)
        x = lam * rho
        b0 = j0(x)
        b1x = np.where(x > 1e-8, j1(x) / np.where(x > 1e-8, x, 1.0), 0.5)
        c = d2 * e
        # eqs 148-153: d2V/drho2, d2V/dz2, d2V/drho dz, (1/rho) dV/drho, V, U
        return np.stack(
            [
                c * (b1x - b0) * lam**3,
                c * g2 * g2 * b0 * lam,
                c * g2 * (b1x * x) * lam**2,
                -c * b1x * lam**3,
                c * b0 * lam,
                d1 * e * b0 * lam,
            ]
        )

    width = min(np.pi / max(rho, 1e-9), 0.5 * np.pi / h, 0.02 * k2) / density
    lam_end = np.sqrt(k2s + (45.0 / h) ** 2) + 2.0 * k2
    if h * abs(k1) < 60.0:
        lam_end = max(lam_end, 1.5 * abs(k1))
    total = np.zeros(6, dtype=complex)

    # The branch point k2, either side: lambda = k2 -+ t^2, dlambda = 2t dt.
    d = 0.05 * k2
    nt = int(np.ceil(np.sqrt(d / width) * 4.0 * density)) + 8
    tb = np.linspace(0.0, np.sqrt(d), nt + 1)
    for sgn in (-1.0, 1.0):
        for a, b in zip(tb[:-1], tb[1:], strict=True):
            t = 0.5 * (a + b) + 0.5 * (b - a) * _GX
            total += (integrands(k2 + sgn * t * t) * (2.0 * t)) @ (_GW * 0.5 * (b - a))

    for lo, hi in ((0.0, k2 - d), (k2 + d, lam_end)):
        n = int(np.ceil((hi - lo) / width))
        edges = np.linspace(lo, hi, n + 1)
        for s in range(0, n, 20000):
            a = edges[s : min(s + 20000, n)]
            b = edges[s + 1 : min(s + 20000, n) + 1]
            mid = 0.5 * (a + b)
            half = 0.5 * (b - a)
            lam = (mid[:, None] + half[:, None] * _GX[None, :]).ravel()
            wts = (half[:, None] * _GW[None, :]).ravel()
            total += integrands(lam) @ wts

    v_rr, v_zz, v_rz, v_r1, v, u = total
    if omega is None:
        omega = k2 * C_LIGHT
    c1 = -1j * omega * mu / (4.0 * np.pi * k2s)
    phase = R1 * np.exp(1j * k2 * R1)
    # eqs 156-159
    return {
        "IrhoV": c1 * phase * k1s * v_rz,
        "IzV": c1 * phase * k1s * (v_zz + k2s * v),
        "IrhoH": c1 * phase * k2s * (v_rr + u),
        "IphiH": -c1 * phase * k2s * (v_r1 + u),
    }


def _main():  # pragma: no cover - the report table
    from momwire import _sommerfeld as sm

    f = 299.8e6
    k = 2.0 * np.pi * f / C_LIGHT
    lam = 2.0 * np.pi / k
    om = k * C_LIGHT
    eps = 13.0 - 1j * 0.005 / (om * 8.8541878128e-12)
    grid = sm.get_grid(eps, k, 1000.0 * lam, om)
    scale = om * MU0 / (4.0 * np.pi)
    print(f"eps_t = {eps:.4f}, r1_max = {grid.r1_max / lam:.2f} wavelengths")
    print(
        "R1/lam  theta   served-oracle  frozen-oracle   (max over surfaces, / |C1 k^2|)"
    )
    for th in (0.05, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 45.0, 80.0):
        edge = grid.eval([grid.r1_max], [np.radians(th)])
        for rw in (16.0, 20.0, 50.0, 200.0, 1000.0):
            o = surfaces(eps, k, rw * lam, np.radians(th))
            g = grid.eval([rw * lam], [np.radians(th)])
            e = max(abs(o[q] - g[q][0]) for q in KEYS) / scale
            fz = max(abs(o[q] - edge[q][0]) for q in KEYS) / scale
            print(f"{rw:7.0f} {th:6.2f}   {e:.2e}       {fz:.2e}")


if __name__ == "__main__":  # pragma: no cover
    _main()
