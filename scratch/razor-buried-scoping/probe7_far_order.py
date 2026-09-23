"""Probe 7: is razor-2p still first order in the far mesh? (momwire#845)

Two #845 decks on TODAY's code: the 20 m centre-fed free-space dipole and
the 10.70 m base-fed monopole over soil A (Sommerfeld, contact). Reference =
bspline d2 at N = 961; its own last step is printed so a row it cannot
resolve is visible. Plus the WHOLLY-BURIED vertical of #812's gate (2 m at
-0.15 m), razor vs bspline at N = 11..161, the family the flip would serve.
"""

import json
import pathlib
import sys
import warnings

import numpy as np

MW = pathlib.Path("/home/smburns/antennas/antennaknobs/momwire")
sys.path.insert(0, str(MW / "tests"))
import momwire  # noqa: E402
from momwire import razor as _razor  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_buried_serve_553 import SOIL_A, WL7  # noqa: E402

assert str(MW) in momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_BELOW_PLANE = True


def z(cls, **kw):
    return complex(cls(**kw).compute_impedance()[0])


def dipole(n):
    return dict(
        wires=[np.array([(0, 0, -10.0), (0, 0, 10.0)])],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 10.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
    )


def monopole(n):
    return dict(
        wires=[np.array([(0, 0, 0.0), (0, 0, 10.70)])],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 0.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def buried(n):
    return dict(
        wires=[np.array([(0, 0, -2.15), (0, 0, -0.15)])],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


out = {}
for name, mk, ladder in (
    ("free_dipole", dipole, (15, 31, 61, 121, 241, 481)),
    ("somm_monopole", monopole, (9, 15, 31, 61, 121, 241)),
):
    ref = z(BSplineSolver, **mk(961))
    ref2 = z(BSplineSolver, **mk(481))
    print(f"{name}: ref bspline d2 N=961 {ref:.4f} (last step {abs(ref - ref2):.4f})")
    rows = []
    for n in ladder:
        zr = z(RazorSolver, **mk(n), nec5_quadrature=True)
        zb = z(BSplineSolver, **mk(n))
        rows.append((n, abs(zr - ref), abs(zb - ref)))
        print(
            f"  N={n:4d} razor-2p |dZ| {abs(zr - ref):8.4f}  bspline-d2 |dZ| {abs(zb - ref):8.4f}"
        )
    out[name] = rows

rows = []
print("wholly buried vertical, even N, feed on the centre knot")
for n in (10, 20, 40, 80, 160, 320):
    zr = z(RazorSolver, **buried(n), nec5_quadrature=True)
    zb = z(BSplineSolver, **buried(n))
    rows.append((n, [zr.real, zr.imag], [zb.real, zb.imag]))
    print(f"  N={n:4d} razor {zr:.4f}  bspline {zb:.4f}  |dZ| {abs(zr - zb):.4f}")
out["buried_vertical"] = rows
pathlib.Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
