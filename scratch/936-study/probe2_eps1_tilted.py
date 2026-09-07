"""momwire#936 study: is the ASSEMBLED crossing block orientation-general
apart from the W move?

probe1 showed the by-parts identity is exact at any alpha and that the
substitution `dW/dl -> t_z dW/dz` is not. It did NOT establish which term of
the assembled sandwich makes that substitution -- so this asks the block.

At eps_tilde = 1 the interface vanishes: the crossing fill must reproduce the
free-space junction fill exactly (the repo's own adjudicator,
`test_g524_7_fan_eps1_collapse`). W and its derivatives are ZERO there, so
this test is BLIND to the W move by construction -- which is the point. It
isolates the rest: U, V, the Phi term and the ends/corner. If the eps1
collapse holds for a TILTED above member at the same residual class as the
untilted deck, then everything except the W move is already orientation-
general and the unit is exactly the W move. If it fails, the scope is wider.

The tilt guard is bypassed for the duration -- that is what is being probed.
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


def deck(lean_deg, ground):
    """One buried radial rising to a node at z=0, and an above mast that
    leaves the node leaning `lean_deg` from vertical."""
    a = np.radians(lean_deg)
    top = 10.0 * np.array([np.sin(a), 0.0, np.cos(a)])
    wires = [
        np.array([(5.0, 0.0, -0.15), (0.0, 0.0, -0.15), (0.0, 0.0, 0.0)]),
        np.array([(0.0, 0.0, 0.0), tuple(top)]),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[10, 2], [15]],
        junctions=[[(0, "end"), (1, "start")]],
        feeds=[(1, 4.3333333333, 1 + 0j)],
        wavelength=WL,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=ground,
        ground_model="sommerfeld",
    )


def collapse(lean_deg):
    build = deck(lean_deg, (1.0, 0.0))
    truth = {
        k: v
        for k, v in build.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    z_free, _ = BSplineSolver(**truth).compute_impedance()
    z_cross, _ = BSplineSolver(**build).compute_impedance()
    return complex(z_cross), complex(z_free)


orig = CF._TILT_TOL
CF._TILT_TOL = 1e9  # bypass the guard: the tilt is what we are probing
try:
    print("eps~ = 1 collapse: crossing fill vs the free-space junction fill")
    print(f"{'lean':>6s} {'crossing Z':>22s} {'free-space Z':>22s} {'|diff|':>10s}")
    for lean in (0.0, 5.0, 15.0, 30.0, 45.0):
        try:
            zc, zf = collapse(lean)
            print(
                f"{lean:6.1f} {zc.real:10.4f}{zc.imag:+10.4f}j "
                f"{zf.real:10.4f}{zf.imag:+10.4f}j {abs(zc - zf):10.4f}"
            )
        except Exception as e:  # noqa: BLE001 - a probe; the reason is printed
            print(f"{lean:6.1f} {type(e).__name__}: {str(e)[:70]}")
finally:
    CF._TILT_TOL = orig


# ===========================================================================
# MEASURED 2026-09-07. The guard bypassed; eps~ = 1.
#
#   lean             crossing Z           free-space Z     |diff|
#    0.0    22.5309 -494.8617j    22.5308 -494.8496j     0.0121
#    5.0    21.5860 -496.8371j    21.5859 -496.8247j     0.0124
#   15.0    19.6957 -501.2486j    19.6959 -501.2350j     0.0136
#   30.0    16.9690 -509.1381j    16.9695 -509.1217j     0.0164
#   45.0    14.5408 -518.9211j    14.5416 -518.9006j     0.0206
#
# THE COLLAPSE HOLDS FOR A TILTED ABOVE MEMBER. The residual at 45 deg is
# 0.0206 against 0.0121 at 0 deg -- the same node-mesh convergence class the
# untilted adjudicator measures, growing mildly with lean rather than
# diverging. A structural orientation error in U, V, the Phi term or the
# ends/corner would show here as a break, not a 1.7x drift in the fourth
# decimal.
#
# So everything in the crossing block EXCEPT the W-carrying terms is already
# orientation-general, and the unit is exactly the W move. That is what makes
# this a bounded change rather than a re-derivation: W and its derivatives are
# identically zero at eps~ = 1, so this test is blind to the W move BY
# CONSTRUCTION and can isolate the rest.
