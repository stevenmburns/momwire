"""momwire#936: momwire on the OP deck, guard bypassed, against NEC-5.

The cards are the ones on antennaknobs `scratch/slope-study/
op_plumb_buried.nec5.nec` (AC6LA's resonant cut, 0.956 x lambda/4 = 7.134 m
per axis at 45 deg, 1 mm copper, 15 cm normal rise, four quarter-wave radials
15 cm down). NEC-5 on this box reproduces the laptop's 58.085 - 3.661j to
0.0002 ohm, so the reference is verified here rather than transcribed.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from momwire import _crossing_fill as CF  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402

WL = 299792458.0 / 7.1e6
A_WIRE = 1e-3
COPPER = 5.7471e7
MAST = 7.1340
RADIAL = 10.55607
DEPTH = 0.15
Z_NEC5 = complex(58.085, -3.6608)


def build(loaded=True, mult=1):
    hub = np.array([0.0, 0.0, -DEPTH])
    node = np.array([0.0, 0.0, 0.0])
    wires = [
        np.array([hub, node]),
        np.array([node, [MAST, 0.0, MAST]]),
    ]
    npe = [[1 * mult], [11 * mult]]
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        wires.append(np.array([hub, [RADIAL * dx, RADIAL * dy, -DEPTH]]))
        npe.append([5 * mult])
    kw = dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[
            [(0, "start")] + [(i, "start") for i in range(2, 6)],
            [(0, "end"), (1, "start")],
        ],
        feeds=[(1, 0.5 * MAST * np.sqrt(2.0) / (11 * mult), 1 + 0j)],
        wavelength=WL,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )
    if loaded:
        kw["wire_conductivity"] = COPPER
    return kw


orig = CF._TILT_TOL
CF._TILT_TOL = 1e9
try:
    print(f"NEC-5 (verified here): {Z_NEC5}")
    for loaded in (True, False):
        for mult in (1, 2):
            try:
                z, _ = BSplineSolver(**build(loaded, mult)).compute_impedance()
                z = complex(z)
                dr = 100 * (z.real - Z_NEC5.real) / Z_NEC5.real
                print(
                    f"  momwire loaded={str(loaded):5s} x{mult}: "
                    f"{z.real:9.3f}{z.imag:+9.3f}j  dR {dr:+6.2f} %  "
                    f"|dZ| {abs(z - Z_NEC5):7.3f}"
                )
            except Exception as e:  # noqa: BLE001 - a probe; the reason is printed
                print(
                    f"  momwire loaded={loaded} x{mult}: "
                    f"{type(e).__name__}: {str(e)[:80]}"
                )
finally:
    CF._TILT_TOL = orig
