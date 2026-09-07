"""momwire#936: the OP deck's own lean control.

probe8 put momwire 5.18 % under NEC-5 on the OP deck at 45 deg lean, wider
than the ~3 % contact class. That is only evidence of a TILT error if the SAME
deck at lean 0 -- same mast length, same radials, same mesh, same loading --
misses by less. This sweeps the lean on both engines and reports the drift.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe3_op_deck as OP  # noqa: E402

from momwire import _crossing_fill as CF  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402

WL = 299792458.0 / 7.1e6
A_WIRE, COPPER = 1e-3, 5.7471e7
MAST_LEN = 7.1340 * np.sqrt(2.0)  # AC6LA's resonant cut, along the wire
RADIAL, DEPTH = 10.55607, 0.15


def tip(lean_deg):
    a = np.radians(lean_deg)
    return MAST_LEN * np.array([np.sin(a), 0.0, np.cos(a)])


def nec5(lean_deg):
    t = tip(lean_deg)
    lines = [
        "CM 936 OP lean control",
        "CE",
        f"GW 6,1,0.,0.,{-DEPTH},0.,0.,0.,.001",
        f"GW 1,11,0.,0.,0.,{t[0]:.5f},{t[1]:.5f},{t[2]:.5f},.001",
    ]
    for i, (dx, dy) in enumerate(((1, 0), (0, 1), (-1, 0), (0, -1)), start=2):
        lines.append(
            f"GW {i},5,0.,0.,{-DEPTH},{RADIAL * dx:.5f},{RADIAL * dy:.5f},{-DEPTH},.001"
        )
    lines += [
        "GE -1,0",
        "LD 5,0,1,32,5.7471E+7,1.",
        "FR 0,1,0,0,7.1",
        "GN 0,0,0,0,13.,.005,1.,0.",
        "EX 0,1,1,0,1.,0.",
        "XQ 0",
        "EN",
    ]
    return OP.run_nec5("\n".join(lines) + "\n")


def mw(lean_deg):
    hub = np.array([0.0, 0.0, -DEPTH])
    node = np.array([0.0, 0.0, 0.0])
    wires = [np.array([hub, node]), np.array([node, tip(lean_deg)])]
    npe = [[1], [11]]
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        wires.append(np.array([hub, [RADIAL * dx, RADIAL * dy, -DEPTH]]))
        npe.append([5])
    z, _ = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[
            [(0, "start")] + [(i, "start") for i in range(2, 6)],
            [(0, "end"), (1, "start")],
        ],
        feeds=[(1, MAST_LEN / 22.0, 1 + 0j)],
        wavelength=WL,
        wire_radius=A_WIRE,
        wire_conductivity=COPPER,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    ).compute_impedance()
    return complex(z)


orig = CF._TILT_TOL
CF._TILT_TOL = 1e9
try:
    print(f"{'lean':>6s} {'momwire':>22s} {'NEC-5':>22s} {'dR %':>8s} {'|dZ|':>8s}")
    base = None
    for lean in (0.0, 15.0, 30.0, 45.0):
        zm, zn = mw(lean), nec5(lean)
        dr = 100 * (zm.real - zn.real) / zn.real
        if base is None:
            base = dr
        print(
            f"{lean:6.1f} {zm.real:10.3f}{zm.imag:+10.3f}j "
            f"{zn.real:10.3f}{zn.imag:+10.3f}j {dr:+8.2f} {abs(zm - zn):8.3f}"
            f"   drift {dr - base:+6.2f} pp"
        )
finally:
    CF._TILT_TOL = orig
