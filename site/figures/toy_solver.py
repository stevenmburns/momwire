"""The Act I toy: the simplest method-of-moments dipole solver that *works*.

Pulse basis on the mixed-potential (Harrington) form of the electric-field
integral equation, with the reduced thin-wire kernel. The wire's n segments
are described by 2n+1 points — the n+1 endpoints and the n midpoints:

    endpoints:  o     o     o     o     o      (n+1 nodes, where charge sits)
    midpoints:     x     x     x     x         (n segments, where current sits)

The current is one constant (pulse) per segment, living at the midpoints. The
charge is dI/dz — for pulses, a point charge at each endpoint where the current
steps. Splitting the field into a vector-potential piece (from the current) and
a scalar-potential piece (from the charge) is what keeps this well-behaved:
matching the raw Pocklington kernel with pulses does not converge, but this does.

It converges *slowly* — first order in segment count — which is the whole point
of Act II. Chapter 2 quotes `toy_dipole` verbatim; the figure scripts import it.
This is essentially Harrington's classic straight-wire example (Harrington,
*Field Computation by Moment Methods*, 1968).
"""

import numpy as np

C0 = 299792458.0  # m/s
EPS0 = 8.8541878188e-12  # F/m
MU0 = 1.25663706127e-6  # H/m


def toy_dipole(L, a, wavelength, N):
    """Input impedance of a center-fed dipole: length L, wire radius a,
    N segments (odd, so one segment straddles the center), 1 V delta gap."""
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi * C0 / wavelength
    dz = L / N

    # 2N+1 points: interleaved endpoints (even index) and midpoints (odd index).
    pts = np.linspace(-L / 2, L / 2, 2 * N + 1)
    mid = pts[1::2]  # N segment centers  — where the current lives
    lo, hi = pts[0:-1:2], pts[2::2]  # each segment's two endpoints — where charge sits

    def psi(A, B):
        """Green's-function kernel between every point in A and B, with the
        analytic self-term where two points coincide (the wire's own surface)."""
        R = np.abs(A[:, None] - B[None, :])
        same = R < dz / 1e6
        R[same] = 1.0  # avoid 0/0; overwritten below
        out = np.exp(-1j * k * R) / (4 * np.pi * R)
        # self-patch: ∫ over a segment of length dz, observed on the surface.
        out[same] = np.log(dz / a) / (2 * np.pi * dz) - 1j * k / (4 * np.pi)
        return out

    # Vector potential (from the current): both segments tested at their centers.
    Z = 1j * omega * MU0 * dz**2 * psi(mid, mid)
    # Scalar potential (from the charge): the endpoint charges of segment n,
    # differenced across the endpoints of segment m. This is the -∇φ term.
    Z += (psi(hi, hi) - psi(lo, hi) - psi(hi, lo) + psi(lo, lo)) / (1j * omega * EPS0)

    # Delta-gap feed: 1 V across the center segment.
    v = np.zeros(N, dtype=complex)
    v[N // 2] = 1.0
    I = np.linalg.solve(Z, v)
    return 1.0 / I[N // 2], I, mid  # Z_in = V / I(feed), with V = 1
