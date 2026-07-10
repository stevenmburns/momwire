"""The Act I toy: the simplest method-of-moments dipole solver.

Pulse basis + point matching on Pocklington's equation with the reduced
thin-wire kernel (Richmond's form — Balanis, *Antenna Theory*, eq. 8.25).
Chapter 2 quotes `toy_dipole` verbatim; the figure scripts import it.
Deliberately naive: no singularity treatment, no better basis, no better
feed model. Acts II–IV of the primer are the story of everything this
file leaves out.
"""

import numpy as np

C0 = 299792458.0  # m/s
EPS0 = 8.8541878188e-12  # F/m


def toy_dipole(L, a, wavelength, N, n_qp=8):
    """Input impedance of a center-fed dipole: length L, wire radius a,
    N segments (odd, so one segment straddles the center), 1 V delta gap."""
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi * C0 / wavelength
    dz = L / N
    z_mid = np.linspace(-L / 2 + dz / 2, L / 2 - dz / 2, N)  # segment centers

    # Gauss-Legendre points on every source segment: z' has shape (1, N, q).
    xg, wg = np.polynomial.legendre.leggauss(n_qp)
    z_src = z_mid[None, :, None] + (dz / 2) * xg[None, None, :]

    # Distance from every match point z_m to every source point z', with the
    # thin-wire trick: the source current lives on the axis, the boundary
    # condition is enforced on the surface, so R can never fall below a.
    R = np.sqrt((z_mid[:, None, None] - z_src) ** 2 + a**2)

    # Richmond's integrand for E_z of a filament current (Balanis 8.25):
    #   e^{-jkR} / (4 pi R^5) * [ (1 + jkR)(2R^2 - 3a^2) + (kaR)^2 ]
    kern = np.exp(-1j * k * R) / (4 * np.pi * R**5) * (
        (1 + 1j * k * R) * (2 * R**2 - 3 * a**2) + (k * a * R) ** 2
    )
    Z = (kern * (dz / 2) * wg).sum(axis=2)  # quadrature -> (N, N) matrix

    # Delta-gap feed: E^i = V/dz on the center segment only, and Pocklington
    # says  Z @ I = -j omega eps0 E^i.
    rhs = np.zeros(N, dtype=complex)
    rhs[N // 2] = -1j * omega * EPS0 * (1.0 / dz)

    I = np.linalg.solve(Z, rhs)
    return 1.0 / I[N // 2], I, z_mid  # Z_in = V / I(feed), with V = 1
