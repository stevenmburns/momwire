"""Dense grazing scans of the four Sommerfeld surfaces, and a contour-family
comparison at the radius where the fig-14 waypoint used to cross k1's
branch point (issue #161).

Two probes:

  scan   The issue's own repro, at full density: |second difference| of each
         surface vs R1 at theta = 0.5 deg, reported as max / median. Over a
         window narrow enough that curvature is smooth this is a jump
         detector — an isolated jump shows up as a ratio in the thousands
         where a smooth surface reads a few. On the pre-#161 contour it read
         8.6e6 (lossless eps_r = 16) and 2.4e3 (default 10 - 1.26j), with the
         spike at R1 = 2.842 and 4.046 wavelengths respectively: exactly
         where `kcap = 1.2*k2 + 50/max(rho, h)` falls through k1.real and the
         descending tail starts on the far side of k1's cut.

  family The evidence for which side was wrong. The fig-14 contour is
         re-run with its waypoint dictated — at the old cap, and at 1.01 /
         1.2 / 1.5 times k1.real — and all four are compared against the
         fig-13 (Bessel) machine with its panel budget lifted, which shares
         no contour geometry with fig 14 and cannot have the defect.

tests/test_sommerfeld_engine.py carries the cheap CI-sized versions of both
(sections 3b and 6). This script is where the dense ones live.

Run from project root:

    PYTHONPATH=. .venv/bin/python scripts/probe_sommerfeld_waypoint_scan.py
"""

import numpy as np

from momwire import _sommerfeld as som

K2 = 2.0 * np.pi  # wavelength units
GROUNDS = (("lossless stress", 16.0 + 0.0j), ("default", 10.0 - 1.26j))


def _k1(eps_t):
    k1 = K2 * np.sqrt(complex(eps_t))
    return np.conj(k1) if k1.imag > 0 else k1


def _six_with_waypoint(eps_t, rho, h, d, rtol=1e-11):
    """The fig-14 contour with the caller placing the waypoint `d`."""
    k1 = _k1(eps_t)

    def f(lam):
        return som._integrand_six(lam, rho, h, k1, K2, "H")

    r1 = np.hypot(rho, h)
    panel = 0.2 * np.pi / max(rho, h)
    a = -0.4j * K2
    total = som._adaptive_segment(f, a, (0.6 + 0.2j) * K2, rtol)
    total = total + som._adaptive_segment(
        f, (0.6 + 0.2j) * K2, (1.02 + 0.2j) * K2, rtol
    )
    total = total + som._adaptive_segment(f, (1.02 + 0.2j) * K2, d, rtol)
    ref = np.max(np.abs(total))
    p0 = 0.5 * max(abs(k1), K2)
    total = total + som._tail(f, d, (h - 1j * rho) / r1, panel, rtol, ref, p0)
    return total - som._tail(f, a, (-h - 1j * rho) / r1, panel, rtol, ref, p0)


def _six_bessel(eps_t, rho, h, rtol=1e-11, max_panels=60000):
    """The fig-13 contour, panel budget lifted so it reaches grazing."""
    k1 = _k1(eps_t)

    def f(lam):
        return som._integrand_six(lam, rho, h, k1, K2, "J")

    p = min(1.0 / rho, 1.0 / h)
    brk = p * (1.0 + 1.0j)
    end_adapt = 1.3 * max(abs(k1), K2) + 3.0 * p + 1.0j * p
    total = som._adaptive_segment(f, 0.0 + 0.0j, brk, rtol)
    start = brk
    if end_adapt.real > brk.real:
        total = total + som._adaptive_segment(f, brk, end_adapt, rtol)
        start = end_adapt
    return total + som._tail(
        f,
        start,
        1.0 + 0.0j,
        0.2 * np.pi / max(rho, h),
        rtol,
        np.max(np.abs(total)),
        max_panels=max_panels,
    )


def scan(th_deg=0.5, lo=2.0, hi=4.0, n=2001):
    print(
        f"\n=== dense second-difference scan, theta = {th_deg} deg, "
        f"R1 in [{lo}, {hi}], dR1 = {(hi - lo) / (n - 1):.4f} ==="
    )
    r1 = np.linspace(lo, hi, n)
    th = np.full_like(r1, np.radians(th_deg))
    for name, eps_t in GROUNDS:
        surf = som.iv_surfaces_direct(eps_t, K2, r1, th, rtol=1e-9)
        for kk in som._SURF_KEYS:
            d2 = np.abs(np.diff(surf[kk], 2))
            med = np.median(d2)
            scale = np.max(np.abs(surf[kk]))
            print(
                f"  {name:16s} {kk:6s} max/median = {d2.max() / med:10.2f}  "
                f"at R1 = {r1[1:-1][d2.argmax()]:.4f}  "
                f"(max = {d2.max():.3e}, {d2.max() / scale:.2e} of scale)"
            )


def family(th_deg=0.5, radii=(2.7, 2.9, 3.5, 4.2)):
    print(f"\n=== contour families vs the fig-13 referee, theta = {th_deg} deg ===")
    th = np.radians(th_deg)
    for name, eps_t in GROUNDS:
        k1 = _k1(eps_t)
        for r1 in radii:
            rho, h = r1 * np.cos(th), r1 * np.sin(th)
            kcap = 1.2 * K2 + 50.0 / max(rho, h)
            ref = _six_bessel(eps_t, rho, h)
            scale = np.max(np.abs(ref))
            cols = []
            for tag, d in (
                ("old cap", kcap + 0.0j),
                ("1.01 k1", 1.01 * k1.real + 0.99j * k1.imag),
                ("1.2 k1", 1.20 * k1.real + 0.99j * k1.imag),
                ("1.5 k1", 1.50 * k1.real + 0.99j * k1.imag),
            ):
                v = _six_with_waypoint(eps_t, rho, h, d)
                cols.append(f"{tag} {np.max(np.abs(v - ref)) / scale:.2e}")
            shipped = som._six_integrals(eps_t, K2, rho, h, 1e-9)
            cols.append(f"SHIPPED {np.max(np.abs(shipped - ref)) / scale:.2e}")
            print(
                f"  {name:16s} R1={r1:5.2f} kcap={kcap:7.3f} k1.real={k1.real:7.3f}"
                f"  |  {'  '.join(cols)}"
            )


if __name__ == "__main__":
    scan()
    scan(lo=3.0, hi=5.0)
    family()
