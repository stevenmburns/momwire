"""The chapter-3 "table of doom": the toy solver on the REAL specimen
(a = 0.5 mm) at every scale, including the brute-force attempts. Chapter 3
quotes this table verbatim.

Run from the repo root:
  .venv/bin/python site/figures/tab_ch3_doom.py          # small N only, seconds
  .venv/bin/python site/figures/tab_ch3_doom.py --big    # adds N up to 15001
                                                         # (~4 GB RAM, ~5 min)

The fill is chunked over match points so the (N, N, q) quadrature transient
never materializes at big N; the mathematics is identical to
`toy_solver.toy_dipole`.
"""

import sys
import time

import numpy as np

from toy_solver import C0, EPS0

WAVELENGTH = 22.0
L = 2 * 0.962 * WAVELENGTH / 4
A = 0.0005


def toy_dipole_chunked(N, n_qp=4, chunk=256):
    k = 2 * np.pi / WAVELENGTH
    omega = 2 * np.pi * C0 / WAVELENGTH
    dz = L / N
    z_mid = np.linspace(-L / 2 + dz / 2, L / 2 - dz / 2, N)
    xg, wg = np.polynomial.legendre.leggauss(n_qp)
    z_src = z_mid[None, :, None] + (dz / 2) * xg[None, None, :]
    Z = np.empty((N, N), dtype=np.complex128)
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        R = np.sqrt((z_mid[i0:i1, None, None] - z_src) ** 2 + A**2)
        kern = np.exp(-1j * k * R) / (4 * np.pi * R**5) * (
            (1 + 1j * k * R) * (2 * R**2 - 3 * A**2) + (k * A * R) ** 2
        )
        Z[i0:i1] = (kern * (dz / 2) * wg).sum(axis=2)
    rhs = np.zeros(N, dtype=complex)
    rhs[N // 2] = -1j * omega * EPS0 / dz
    I = np.linalg.solve(Z, rhs)
    return 1.0 / I[N // 2]


if __name__ == "__main__":
    Ns = [21, 81, 301, 1001]
    if "--big" in sys.argv:
        Ns += [2501, 5001, 10001, 15001]
    print(f"specimen: L = {L:.4f} m, a = {A * 1e3:.1f} mm, lambda = {WAVELENGTH} m")
    print(f"{'N':>6}  {'dz/a':>6}  {'matrix':>9}  {'time':>7}  Z_in")
    for N in Ns:
        t0 = time.time()
        z = toy_dipole_chunked(N)
        dt = time.time() - t0
        print(
            f"{N:6d}  {(L / N) / A:6.1f}  {N * N * 16 / 1e9:7.2f} GB"
            f"  {dt:6.1f}s  {z.real:9.2f} {z.imag:+12.2f}j",
            flush=True,
        )
