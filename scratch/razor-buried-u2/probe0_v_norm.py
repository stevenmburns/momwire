"""Probe 0: the normalization c1*V(eps~=1) == g/(j w eps0) with g = e^{-jkR}/(4 pi R)."""

import numpy as np

from momwire import _near_interface
from momwire._sommerfeld_below import _c1_moment

EPS0 = 8.854187817e-12
MU0 = 4e-7 * np.pi
wl = 7.0
k = 2 * np.pi / wl
omega = k / np.sqrt(MU0 * EPS0)
c1 = _c1_moment(omega, MU0)
for R, z, zp in ((0.001, 0.0, 0.0), (0.3, 0.2, 0.0), (1.0, 0.0, -0.5)):
    rho = np.sqrt(max(R * R - (z - zp) ** 2, 0.0))
    six = _near_interface.six_point(1.0, k, rho, z, zp)
    g = np.exp(-1j * k * R) / (4 * np.pi * R)
    fam = g / (1j * omega * EPS0)
    print(R, [abs(c1 * v / fam) for v in six])
