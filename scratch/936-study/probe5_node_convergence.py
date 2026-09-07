"""momwire#936: does the DIRECT spelling of the W term converge AT THE NODE?

This is the question that decides (1) a seventh kernel column vs (2) the
direct spelling. probe1 compared the two spellings on a segment away from the
interface and got 1e-6. The module's warning is not about that: it is that a
"drop" spelling's retained ln(a)-class content diverges at the NODE when its
balancing end/corner terms are deleted. So the test has to put the segment END
on the interface, where W carries that content, and refine.

Two spellings of one integral, laddered on the graded node quadrature:

    DIRECT     sum w W F'   -   [W F]_ends
    BYPARTS    - sum w F (dW/dl)          (dW/dl by centred difference)

Both are exact by parts at any orientation (probe1). What is at stake is
whether the DIRECT one is numerically stable as the node is resolved -- if it
diverges while the by-parted one settles, the by-parts is load-bearing and
option (2) is unavailable for the tilted side.

Soil A, where W is nonzero. alpha = 45 deg, segment starting ON the interface.
"""

import numpy as np

from momwire import _ground_refl
from momwire._crossing_fill import _graded_u
from momwire._near_interface import designed_tables

SOIL, FREQ = (13.0, 0.005), 7.1e6
K2 = 2.0 * np.pi * FREQ / 299792458.0
ET = _ground_refl.eps_tilde(SOIL, 2.0 * np.pi * FREQ, 8.8541878128e-12)
A_WIRE = 1e-3
SRC = np.array([0.7, 0.0, -0.15])


def W_at(p):
    d = p - SRC
    rho = float(np.hypot(d[0], d[1]))
    return complex(designed_tables(ET, K2, rho, float(p[2]), float(SRC[2]))["W"])


def spellings(alpha_deg, h, growth, eps_fd=1e-6):
    a = np.radians(alpha_deg)
    t = np.array([np.sin(a), 0.0, np.cos(a)])
    p0 = np.array([0.0, 0.0, 0.0])  # the END IS ON THE INTERFACE
    u, w = _graded_u(h, "lo", A_WIRE, growth)
    pts = p0[None, :] + u[:, None] * t[None, :]
    F, Fd = u / h, np.full(len(u), 1.0 / h)
    Wv = np.array([W_at(p) for p in pts])
    direct = np.sum(w * Wv * Fd) - (W_at(p0 + h * t) * 1.0 - W_at(p0) * 0.0)
    dl = eps_fd * t
    dWdl = np.array([(W_at(p + dl) - W_at(p - dl)) / (2 * eps_fd) for p in pts])
    byparts = -np.sum(w * F * dWdl)
    return direct, byparts, len(u)


print(f"soil {SOIL}, alpha = 45 deg, segment END on the interface")
print(
    f"{'growth':>7s} {'nodes':>6s} {'|direct|':>12s} {'|byparts|':>12s} "
    f"{'|difference|':>13s}"
)
for growth in (4.0, 3.0, 2.0, 1.6, 1.35):
    d, b, n = spellings(45.0, 0.30, growth)
    print(f"{growth:7.2f} {n:6d} {abs(d):12.6f} {abs(b):12.6f} {abs(d - b):13.3e}")
print("\nsame, refining the SEGMENT toward the node (h shrinking):")
print(
    f"{'h (m)':>9s} {'nodes':>6s} {'|direct|':>12s} {'|byparts|':>12s} "
    f"{'|difference|':>13s}"
)
for h in (0.30, 0.15, 0.05, 0.01, 0.002):
    d, b, n = spellings(45.0, h, 2.0)
    print(f"{h:9.4f} {n:6d} {abs(d):12.6f} {abs(b):12.6f} {abs(d - b):13.3e}")
