"""momwire#935: the shape check -- a shallow buried dipole against NEC-5.

The floor move is only worth anything if it buys a GEOMETRY, so this is the
end-to-end version of the claim: the same 5.9114 m dipole the oracle ran, at
the same three depths, through momwire's solver.

    depth      theta = atan(2h/L)   old floor (0.1)   new floor (0.05)
    0.5 mm       0.00969 deg          refused           refused
    1.0 mm       0.01939 deg          refused           refused
    3.0 mm       0.05817 deg          refused           SERVED

momwire is laddered too. NEC-5's own ladder converged to ~0.16 ohm (the
Richardson bar in `tests/golden_grazing_shape_nec5.py`); a momwire column read
at one mesh would be the #845 mistake in the other direction.
"""

import math

import numpy as np

from momwire.bspline import BSplineSolver

C0 = 299792458.0
HALF, RAD, FREQ = 2.9557, 5.0e-4, 7.1e6
WL = C0 / FREQ
GROUND = (13.0, 0.005)


def z_at(depth_m, n):
    wire = np.array([[-HALF, 0.0, -depth_m], [HALF, 0.0, -depth_m]])
    z, _ = BSplineSolver(
        wires=[wire],
        n_per_edge_per_wire=[[n]],
        wire_radius=RAD,
        wavelength=WL,
        degree=2,
        feed_model="segment",
        feed_wire_index=0,
        feed_arclength=HALF,
        ground_z=0.0,
        ground_eps=GROUND,
        ground_model="sommerfeld",
    ).compute_impedance()
    return complex(z)


print(f"5.9114 m dipole, radius {RAD} m, {FREQ / 1e6} MHz, soil {GROUND}")
print(
    f"grazing floor = {__import__('momwire._sommerfeld_below', fromlist=['x'])._SOMM_BELOW_TH_MIN_DEG} deg\n"
)
for mm in (0.5, 1.0, 3.0):
    d = mm / 1000.0
    th = math.degrees(math.atan2(2 * d, 2 * HALF))
    print(f"depth {mm:4.1f} mm  (theta = {th:.5f} deg)")
    for n in (41, 81, 161, 241, 321):
        try:
            z = z_at(d, n)
            print(f"    n={n:4d}  {z.real:9.3f}{z.imag:+9.3f}j")
        except ValueError as e:
            print(f"    n={n:4d}  REFUSED: {str(e)[:95]}...")
            break
