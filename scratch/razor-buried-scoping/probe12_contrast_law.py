"""Probe 12: how the crossing-node defect scales with the soil contrast eps~.

crossing_deck(1) at far mesh x4, ports as probe 2 (above knot z = 4.5, buried
knot z = -1.0). Per (eps_r, sigma): razor-2p 2-port non-reciprocity, razor and
bspline single-port Z at z = 4.5, and the licensed binary's Z (probe 8's deck,
GE -1). eps~ = eps_r - j sigma/(omega eps0) at 7 MHz.
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
from test_crossing_serve_524 import crossing_deck  # noqa: E402

assert str(MW) in momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_BELOW_PLANE = True
_razor._SERVE_CROSSING = True
EXE = "/home/smburns/antennas/NEC5-downloads/nec5-linux/nec5cl"
M = 4
W_EPS0 = 2 * np.pi * 7e6 * 8.8541878128e-12


def f(x):
    return f"{float(x):.6E}"


def mw_deck(eps, feeds):
    d = crossing_deck(1)
    b, a = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [
        [n * M for n in b[:-1]] + [b[-1]],
        [a[0]] + [n * M for n in a[1:]],
    ]
    d["feeds"] = feeds
    d["ground_eps"] = eps
    return d


def nec5_z(eps_r, sig):
    edges = [
        (-2.0, -0.5, 3 * M),
        (-0.5, -0.1, 2 * M),
        (-0.1, 0.0, 2),
        (0.0, 0.1, 2),
        (0.1, 0.5, 2 * M),
        (0.5, 10.0, 19 * M),
    ]
    lines = ["CM probe12", "CE"]
    for tag, (z0, z1, ns) in enumerate(edges, start=1):
        lines.append(
            f"GW {tag} {ns} {f(0)} {f(0)} {f(z0)} {f(0)} {f(0)} {f(z1)} {f(0.001)}"
        )
    lines += [
        "GE -1 0",
        f"GN 0 0 0 0 {f(eps_r)} {f(sig)} {f(1.0)} {f(0.0)} NOFILE",
        f"EX 0 6 {8 * M} 2 {f(1.0)} {f(0.0)}",
        f"FR 0 1 0 0 {f(7.0)} {f(0.0)}",
        "XQ 0",
        "EN",
    ]
    txt = run_deck(EXE, "\n".join(lines) + "\n", timeout=300)
    return NEC5Engine._parse_input_parameters(txt)[0][0][2]


out = []
cases = [
    (1.0, 0.0),
    (1.1, 0.0),
    (1.5, 0.0),
    (2.0, 0.0),
    (4.0, 0.0),
    (13.0, 0.0),
    (30.0, 0.0),
    (80.0, 0.0),
    (1.0, 0.002),
    (5.0, 0.001),
    (13.0, 0.005),
    (13.0, 0.05),
]
print(
    " eps_r  sigma   |eps~|  g=|e-1|/|e+1|  nonrec     razor Z             bspline Z           NEC-5 Z"
    "            dR(rz-n5) dX(rz-n5) dR(bs-n5)"
)
for eps_r, sig in cases:
    et = complex(eps_r, -sig / W_EPS0)
    g = abs(et - 1) / abs(et + 1)
    d2 = mw_deck((eps_r, sig), [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)])
    d2.pop("junctions")
    Y = np.asarray(RazorSolver(**d2, nec5_quadrature=True).compute_y_matrix())
    nonrec = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
    d1 = mw_deck((eps_r, sig), [(1, 4.5, 1 + 0j)])
    zb = complex(BSplineSolver(**d1).compute_impedance()[0])
    d1.pop("junctions")
    zr = complex(RazorSolver(**d1, nec5_quadrature=True).compute_impedance()[0])
    zn = nec5_z(eps_r, sig)
    out.append(
        dict(
            eps_r=eps_r,
            sigma=sig,
            g=g,
            nonrec=nonrec,
            razor=[zr.real, zr.imag],
            bspline=[zb.real, zb.imag],
            nec5=[zn.real, zn.imag],
        )
    )
    print(
        f"{eps_r:6.1f} {sig:6.3f} {abs(et):7.2f} {g:10.4f}   {nonrec:.3e}  {zr.real:8.2f}{zr.imag:+8.2f}j  "
        f"{zb.real:8.2f}{zb.imag:+8.2f}j  {zn.real:8.2f}{zn.imag:+8.2f}j  {zr.real - zn.real:+7.2f} "
        f"{zr.imag - zn.imag:+7.2f}  {zb.real - zn.real:+7.2f}",
        flush=True,
    )
pathlib.Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1))
