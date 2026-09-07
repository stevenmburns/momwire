"""momwire#936: with the guard bypassed, is the tilted answer RIGHT?

The study's terms are all written in tangent components already, and probe2
showed the non-W structure is orientation-general. probe5 showed the direct
and by-parted spellings of the W term agree to 1e-11 at the node. None of that
answers the only question that matters: does the assembled fill produce the
right IMPEDANCE for a tilted deck?

This asks it end to end against NEC-5 on the SAME geometry -- built here, run
here, so no external fixture is needed and no number is transcribed. If the
guard is over-strict, momwire lands within the contact-class residual (~3 % in
R, and it walks AWAY from momwire with NEC-5's mesh, so agreement at that
level is the bar, not equality). If something is genuinely missing, it will
not.
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


def momwire_deck(n_mult=1):
    """The same wires probe3 hands NEC-5, in momwire's spelling."""
    ws = OP.wires()
    wires = [np.array([p0, p1]) for p0, p1, _ in ws]
    npe = [[max(1, int(n * n_mult))] for _, _, n in ws]
    # hub is wire 0..3 start and wire 4 start; node is wire 4 end + wire 5 start
    n_rad = OP.N_RAD
    junctions = [
        [(i, "start") for i in range(n_rad)] + [(n_rad, "start")],
        [(n_rad, "end"), (n_rad + 1, "start")],
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=junctions,
        feeds=[(n_rad + 1, 0.25, 1 + 0j)],
        wavelength=WL,
        wire_radius=OP.RAD_A,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )


if __name__ == "__main__":
    z_n5 = complex(60.809, 33.073)  # probe3, re-derived on this box
    print(f"NEC-5 on this geometry (probe3): {z_n5}")
    orig = CF._TILT_TOL
    CF._TILT_TOL = 1e9
    try:
        for m in (1, 2):
            try:
                z, _ = BSplineSolver(**momwire_deck(m)).compute_impedance()
                z = complex(z)
                dr = 100 * (z.real - z_n5.real) / z_n5.real
                print(
                    f"  momwire x{m}: {z.real:9.3f}{z.imag:+9.3f}j   "
                    f"dR {dr:+6.2f} %   |dZ| {abs(z - z_n5):7.3f}"
                )
            except Exception as e:  # noqa: BLE001 - a probe; the reason is printed
                print(f"  momwire x{m}: {type(e).__name__}: {str(e)[:100]}")
    finally:
        CF._TILT_TOL = orig
