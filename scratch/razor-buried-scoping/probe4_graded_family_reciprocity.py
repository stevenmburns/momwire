"""Probe 4: probe 2's reciprocity instrument per FAMILY, on a graded mesh.

Probe 3's straight uniform wire is symmetric by construction (vacuous). Here
the crossing deck's own GRADED two-wire geometry (the mesh that makes razor's
free-space Y non-symmetric at 2.8e-4 -> 1.9e-7) is translated rigidly:

  below : shifted down 10.3 m -> wholly buried (the #812 below fill only),
          the junction becomes an ordinary buried junction at z = -10.3;
  above : shifted up 2.3 m -> wholly above the Sommerfeld ground (razor's
          composing ground only), junction at z = +2.3.

Ports as probe 2: the upper wire at arclength 4.5, the lower at 1.0 - knots at
every rung. Soil A. If a family's own non-reciprocity decays with mesh as the
free-space control does, that family is not where probe 2's flat 3 % lives.
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
from test_crossing_serve_524 import crossing_deck  # noqa: E402

assert str(MW) in momwire.__file__, momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_BELOW_PLANE = True
_razor._SERVE_CROSSING = True


def deck(m, shift):
    d = crossing_deck(1)
    b, a = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [
        [n * m for n in b[:-1]] + [b[-1]],
        [a[0]] + [n * m for n in a[1:]],
    ]
    d["wires"] = [w + np.array([0.0, 0.0, shift]) for w in d["wires"]]
    d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
    return d


out = []
for label, shift in (("below", -10.3), ("above", 2.3)):
    for m in (1, 2, 4, 8):
        for name, cls, kw in (
            ("razor_2p", RazorSolver, {"nec5_quadrature": True}),
            ("bspline_d2", BSplineSolver, {}),
        ):
            d = deck(m, shift)
            if cls is RazorSolver:
                d.pop("junctions")
            t0 = time.perf_counter()
            try:
                Y = np.asarray(cls(**d, **kw).compute_y_matrix())
            except Exception as e:  # noqa: BLE001 - probe reports any refusal verbatim
                print(
                    f"{label} x{m} {name}: {type(e).__name__}: {str(e)[:300]}",
                    flush=True,
                )
                continue
            dt = time.perf_counter() - t0
            nonrec = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
            Z = np.linalg.inv(Y)
            z11 = complex(Z[0, 0])
            out.append(
                dict(
                    case=label,
                    x=m,
                    solver=name,
                    z11=[z11.real, z11.imag],
                    nonrec=nonrec,
                    t=round(dt, 2),
                )
            )
            print(
                f"{label:5s} x{m:<2d} {name:10s} Z11 {z11.real:9.3f}{z11.imag:+9.3f}j "
                f"nonrec {nonrec:.3e} {dt:6.2f} s",
                flush=True,
            )

p = pathlib.Path(__file__).with_suffix(".json")
p.write_text(json.dumps(out, indent=1))
print("wrote", p)
