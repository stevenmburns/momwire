"""Probe 9: is razor-2p's WHOLLY-BELOW fill (#812) still NEC-5's twin in soil?

A 2 m vertical buried at -0.15 m (z -2.15 -> -0.15), 1 mm, soil A, 7 MHz,
centre knot feed; N = 10..160. razor-2p (nec5_quadrature=True; EK is refused
below the plane, NEC-5 is EK-on natively), the licensed binary under GE -1,
and bspline d2. Above ground razor-2p tracks NEC-5 to a constant ~0.04 ohm at
EQUAL mesh (nec5-formulation-identified); the question is whether the below
family keeps that.
"""

import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, "/home/smburns/antennas/antennaknobs/src")
MW = pathlib.Path("/home/smburns/antennas/antennaknobs/momwire")
sys.path.insert(0, str(MW / "tests"))
import momwire  # noqa: E402
from antennaknobs.engines.nec5 import NEC5Engine, run_deck  # noqa: E402
from momwire import razor as _razor  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_buried_serve_553 import SOIL_A, WL7  # noqa: E402

assert str(MW) in momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_BELOW_PLANE = True
EXE = "/home/smburns/antennas/NEC5-downloads/nec5-linux/nec5cl"


def f(x):
    return f"{float(x):.6E}"


def nec5(n):
    d = (
        "\n".join(
            [
                "CM probe9",
                "CE",
                f"GW 1 {n} {f(0)} {f(0)} {f(-2.15)} {f(0)} {f(0)} {f(-0.15)} {f(0.001)}",
                "GE -1 0",
                f"GN 0 0 0 0 {f(13.0)} {f(0.005)} {f(1.0)} {f(0.0)} NOFILE",
                f"EX 0 1 {n // 2} 2 {f(1.0)} {f(0.0)}",
                f"FR 0 1 0 0 {f(7.0)} {f(0.0)}",
                "XQ 0",
                "EN",
            ]
        )
        + "\n"
    )
    return NEC5Engine._parse_input_parameters(run_deck(EXE, d, timeout=300))[0][0][2]


def mw(cls, n, **kw):
    d = dict(
        wires=[np.array([(0, 0, -2.15), (0, 0, -0.15)])],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 1.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )
    return complex(cls(**d, **kw).compute_impedance()[0])


out = []
for n in (10, 20, 40, 80, 160):
    zn = nec5(n)
    zr = mw(RazorSolver, n, nec5_quadrature=True)
    zb = mw(BSplineSolver, n)
    out.append(
        dict(
            N=n,
            nec5=[zn.real, zn.imag],
            razor=[zr.real, zr.imag],
            bspline=[zb.real, zb.imag],
        )
    )
    print(
        f"N={n:4d} NEC-5 {zn.real:9.3f}{zn.imag:+9.3f}j  razor-2p {zr.real:9.3f}{zr.imag:+9.3f}j "
        f"(|d| {abs(zr - zn):6.3f})  bspline {zb.real:9.3f}{zb.imag:+9.3f}j (|d| {abs(zb - zn):6.3f})",
        flush=True,
    )
pathlib.Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
