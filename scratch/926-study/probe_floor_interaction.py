"""momwire#926, re-read after #935 moved the below/below grazing floor.

Two questions, both measured rather than reasoned:

  1. Does #935 change #926's arithmetic? #926 is the crossing-node grading
     advisory colliding with the h/a >= 2 stand-off floor on radials sloping
     UP from a grounded node. That floor lives in `_surface_height.py`, which
     names neither `_sommerfeld_below` nor the grazing floor -- so the answer
     should be no, and this reproduces the refusal to check.

  2. Is there a BURIED twin, where the binding constraint IS the grazing floor
     and #935 therefore halved it? A radial sloping DOWN from a node on the
     interface is the mirror image of #926's geometry, and the below/below
     floor bites it the way h/a bites the other.
"""

import math

import numpy as np

from momwire import _sommerfeld_below as below
from momwire import _surface_height
from momwire.bspline import BSplineSolver

C0 = 299792458.0
FREQ, RAD, SOIL = 7e6, 1e-3, (13.0, 0.005)
WL = C0 / FREQ
SPAN, RISE = 10.0, 0.5  # #926's deck: 5 % slope


def deck(finest_mm, sign=+1.0):
    """Four radials from a node at z=0, sloping `sign` over `SPAN`.

    Graded toward the node the way `CoarseCrossingNode` prescribes -- a
    geometric ramp from `finest` at the node to the design's own segment
    length away from it -- plus the 10 m mast.
    """
    s = finest_mm / 1000.0
    pts, ratio = [0.0], 1.0
    while pts[-1] < SPAN:
        pts.append(pts[-1] + s * ratio)
        ratio *= 1.35
    pts = np.array([p for p in pts if p < SPAN] + [SPAN])
    wires, npe = [], []
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        xyz = np.stack([pts * dx, pts * dy, sign * RISE * (pts / SPAN)], axis=1)
        wires.append(xyz)
        npe.append([1] * (len(pts) - 1))
    wires.append(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]))
    npe.append([20])
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        wire_radius=RAD,
        wavelength=WL,
        degree=2,
        feed_model="segment",
        feed_wire_index=4,
        feed_arclength=0.25,
        ground_z=0.0,
        ground_eps=SOIL,
        ground_model="sommerfeld",
    )


print(
    f"h/a validity floor      = {_surface_height.SURFACE_HEIGHT_CLASS.floor_h_over_a}"
)
print(
    f"below/below theta floor = {below._SOMM_BELOW_TH_MIN_DEG} deg (was 0.1 pre-#935)"
)
print(
    f"\n#926 arithmetic: on a {100 * RISE / SPAN:.0f} % slope, h = {RISE / SPAN} * s,"
)
print(
    f"  so h >= 2a = {2 * RAD * 1000:.0f} mm forbids any node panel shorter than "
    f"{2 * RAD / (RISE / SPAN) * 1000:.0f} mm.\n"
)

for sign, label in ((+1.0, "ABOVE (#926 as filed)"), (-1.0, "BURIED (the mirror)")):
    print(f"--- radials sloping {label} ---")
    for finest in (6.0, 40.0, 80.0):
        try:
            BSplineSolver(**deck(finest, sign)).compute_impedance()
            print(f"  finest {finest:5.1f} mm  SERVED")
        except ValueError as e:
            print(f"  finest {finest:5.1f} mm  REFUSED: {str(e)[:110]}")
        except NotImplementedError as e:
            # The buried mirror does not reach either floor: the crossing
            # fill refuses a TILTED buried segment outright, which is a more
            # fundamental gap than the one #926 is about.
            print(f"  finest {finest:5.1f} mm  UNIMPLEMENTED: {str(e)[:110]}")
    print()

# what the grazing floor alone implies for the buried mirror
for floor in (0.1, 0.05):
    h_min = SPAN * math.tan(math.radians(floor)) / 2.0
    print(
        f"buried mirror, floor {floor:.2f} deg -> shallowest legal depth over "
        f"{SPAN} m span = {h_min * 1000:.2f} mm -> node panel >= "
        f"{h_min / (RISE / SPAN) * 1000:.0f} mm"
    )
