"""momwire#936: the CONTROL. Is the tilted miss the tilt, or the deck class?

probe6 measured momwire -3.68 % in R against NEC-5 on a 45 deg leaning mast
over a buried screen, with the guard bypassed. That is inside the
contact-class residual the issue names -- but a number inside a class is only
evidence if the SAME deck at lean = 0, where the fill is known correct, misses
by the same amount. Otherwise the class is being used to excuse a tilt error.

So: one deck family, lean swept from 0, both engines, same mesh.
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

WL = 299792458.0 / OP.FREQ


def wires_at(lean_deg):
    a = np.radians(lean_deg)
    tip = OP.QW * np.array([np.sin(a), 0.0, np.cos(a)])
    hub = np.array([0.0, 0.0, -OP.DEPTH])
    node = np.array([0.0, 0.0, 0.0])
    out = []
    for i in range(OP.N_RAD):
        th = 2 * np.pi * i / OP.N_RAD
        out.append(
            (hub, np.array([OP.QW * np.cos(th), OP.QW * np.sin(th), -OP.DEPTH]), 20)
        )
    out.append((hub, node, 2))
    out.append((node, tip, 21))
    return out


def nec5(lean_deg):
    ws = wires_at(lean_deg)
    lines = ["CM 936 lean control", "CE"]
    for i, (p0, p1, n) in enumerate(ws, start=1):
        lines.append(
            f"GW {i} {n} {p0[0]:.6E} {p0[1]:.6E} {p0[2]:.6E} "
            f"{p1[0]:.6E} {p1[1]:.6E} {p1[2]:.6E} {OP.RAD_A:.6E}"
        )
    lines += [
        "GE -1 0",
        "GN 0 0 0 0 1.300000E+01 5.000000E-03 1.000000E+00 0.000000E+00 NOFILE",
        f"EX 0 {len(ws)} 1 0 1.000000E+00 0.000000E+00",
        f"FR 0 1 0 0 {OP.FREQ / 1e6:.6E} 0.000000E+00",
        "XQ 0",
        "EN",
    ]
    return OP.run_nec5("\n".join(lines) + "\n")


def mw(lean_deg):
    ws = wires_at(lean_deg)
    n_rad = OP.N_RAD
    build = dict(
        wires=[np.array([p0, p1]) for p0, p1, _ in ws],
        n_per_edge_per_wire=[[n] for _, _, n in ws],
        junctions=[
            [(i, "start") for i in range(n_rad)] + [(n_rad, "start")],
            [(n_rad, "end"), (n_rad + 1, "start")],
        ],
        feeds=[(n_rad + 1, 0.25, 1 + 0j)],
        wavelength=WL,
        wire_radius=OP.RAD_A,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )
    z, _ = BSplineSolver(**build).compute_impedance()
    return complex(z)


orig = CF._TILT_TOL
CF._TILT_TOL = 1e9
try:
    print(f"{'lean':>6s} {'momwire':>22s} {'NEC-5':>22s} {'dR %':>8s} {'|dZ|':>8s}")
    for lean in (0.0, 15.0, 30.0, 45.0):
        zm, zn = mw(lean), nec5(lean)
        dr = 100 * (zm.real - zn.real) / zn.real
        print(
            f"{lean:6.1f} {zm.real:10.3f}{zm.imag:+10.3f}j "
            f"{zn.real:10.3f}{zn.imag:+10.3f}j {dr:+8.2f} {abs(zm - zn):8.3f}"
        )
finally:
    CF._TILT_TOL = orig


# ===========================================================================
# MEASURED 2026-09-07, guard bypassed, soil 13/0.005, 7.1 MHz.
#
#   lean                momwire                  NEC-5     dR %     |dZ|
#    0.0     72.941   +35.970j     75.185   +39.952j    -2.98    4.571
#   15.0     71.231   +35.301j     73.473   +39.330j    -3.05    4.610
#   30.0     66.241   +33.074j     68.480   +37.218j    -3.27    4.710
#   45.0     58.571   +28.811j     60.809   +33.073j    -3.68    4.814
#
# THE CONTROL IS THE POINT. At lean = 0, where the fill is known correct and
# the guard permits the deck, momwire already misses NEC-5 by -2.98 % / 4.57
# ohm -- that is the contact-class residual, not a tilt effect. Sweeping to
# 45 deg adds 0.7 pp / 0.24 ohm on top of it: the tilt contributes about 5 %
# of a residual that is already there, and the answer stays inside the class.
#
# WHAT THIS DOES AND DOES NOT SETTLE. It says the assembled fill is not
# grossly wrong for tilt. It does NOT prove the guard is merely over-strict:
# probe1 measured the dropped term at 21-58 % OF THE TERM IT BELONGS TO, and
# a 0.24 ohm drift is exactly the size a real-but-small missing term would
# have. A flat-ish residual is consistent with both readings, so it cannot be
# quoted for either one alone.
