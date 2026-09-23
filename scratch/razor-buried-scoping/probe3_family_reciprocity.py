"""Probe 3: which family carries probe 2's flat 3 % non-reciprocity?

(a) WHOLLY-BELOW 2-port (razor #812 fill only, no cross blocks): a buried
    horizontal wire 4 m long at depth 0.5 m, ports at arclength 1.0 and 3.0.
(b) ABOVE-ONLY 2-port over the same Sommerfeld soil (razor's composing
    ground only): the same wire 0.5 m ABOVE the plane.
(c) the crossing deck again, ONE port at arclength 4.5 (knot-aligned at
    every rung) - the driving-point residual without #845's feed-snap noise.

Ports are knot-aligned at every rung (h = 4/N, N a multiple of 4).
"""

import json
import pathlib
import sys
import time
import warnings

import numpy as np

MW = pathlib.Path("/home/smburns/antennas/antennaknobs/momwire")
sys.path.insert(0, str(MW / "tests"))

import momwire  # noqa: E402
from momwire import razor as _razor  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_buried_serve_553 import SOIL_A, WL7  # noqa: E402
from test_crossing_serve_524 import crossing_deck  # noqa: E402

assert str(MW) in momwire.__file__, momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_BELOW_PLANE = True
_razor._SERVE_CROSSING = True

out = []


def two_port(z, n, ground):
    pts = np.array([(-2.0, 0.0, z), (2.0, 0.0, z)])
    d = dict(
        wires=[pts],
        n_per_edge_per_wire=[[n]],
        feeds=[(0, 1.0, 1 + 0j), (0, 3.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
    )
    if ground:
        d.update(ground_z=0.0, ground_eps=SOIL_A, ground_model="sommerfeld")
    return d


for label, z in (("below", -0.5), ("above", 0.5)):
    for n in (8, 16, 32, 64, 128):
        for name, cls, kw in (
            ("razor_2p", RazorSolver, {"nec5_quadrature": True}),
            ("bspline_d2", BSplineSolver, {}),
        ):
            t0 = time.perf_counter()
            Y = np.asarray(cls(**two_port(z, n, True), **kw).compute_y_matrix())
            dt = time.perf_counter() - t0
            nonrec = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
            zin = complex(np.linalg.inv(Y)[0, 0])
            out.append(
                dict(
                    case=label,
                    N=n,
                    solver=name,
                    z11=[zin.real, zin.imag],
                    nonrec=nonrec,
                    t=round(dt, 2),
                )
            )
            print(
                f"{label:5s} N={n:4d} {name:10s} Z11 {zin.real:9.3f}{zin.imag:+9.3f}j "
                f"nonrec {nonrec:.3e} {dt:6.2f} s",
                flush=True,
            )

for m in (1, 2, 4, 8, 16):
    d = crossing_deck(1)
    b, a = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [
        [n * m for n in b[:-1]] + [b[-1]],
        [a[0]] + [n * m for n in a[1:]],
    ]
    d["feeds"] = [(1, 4.5, 1 + 0j)]
    N = sum(d["n_per_edge_per_wire"][0]) + sum(d["n_per_edge_per_wire"][1])
    zs = {}
    for name, cls, kw in (
        ("razor_2p", RazorSolver, {"nec5_quadrature": True}),
        ("bspline_d2", BSplineSolver, {}),
    ):
        dd = dict(d)
        if cls is RazorSolver:
            dd.pop("junctions")
        t0 = time.perf_counter()
        z, _ = cls(**dd, **kw).compute_impedance()
        dt = time.perf_counter() - t0
        zs[name] = complex(z)
        out.append(
            dict(
                case="crossing_knotfeed",
                x=m,
                N=N,
                solver=name,
                z=[complex(z).real, complex(z).imag],
                t=round(dt, 2),
            )
        )
    dz = zs["razor_2p"] - zs["bspline_d2"]
    print(
        f"crossing feed@4.5 x{m:<2d} N={N:4d} razor {zs['razor_2p']:.4f} "
        f"bspline {zs['bspline_d2']:.4f}  dR {dz.real:+.3f} dX {dz.imag:+.3f}",
        flush=True,
    )

p = pathlib.Path(__file__).with_suffix(".json")
p.write_text(json.dumps(out, indent=1))
print("wrote", p)
