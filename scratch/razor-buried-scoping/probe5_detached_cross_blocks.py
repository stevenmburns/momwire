"""Probe 5: the cross blocks ALONE (no node row), on a DETACHED deck.

The crossing deck's graded wires pulled apart: the above wire lifted 0.3 m
(bottom at z = +0.3), the buried wire lowered 0.3 m and moved 0.5 m sideways
(top at z = -0.3, x = 0.5) - no junction, nothing in the plane.

razor REFUSES this deck in its constructor (the detached class is served by
neither #812 nor #813). For this probe only, `_refuse_buried_geometry` is
replaced IN-PROCESS by one that routes the deck through `_assemble_Z_crossing`
with zero crossing tents: the same-medium blocks from their own families and
the two `_crossing_fill` cross blocks on razor's path axes. Nothing on disk
changes.

bspline serves the detached class through the TRANSMITTED GRID
(`_sommerfeld_transmitted`), which is independent of `_crossing_fill`.

Reported: 2-port non-reciprocity (above port arclength 4.5, buried port 1.0,
knots at every rung) and Z11, razor vs bspline, soil A; free-space control.
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
from momwire import _medium_spec  # noqa: E402
from momwire import razor as _razor  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_crossing_serve_524 import crossing_deck  # noqa: E402

assert str(MW) in momwire.__file__, momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_BELOW_PLANE = True
_razor._SERVE_CROSSING = True


def _detached_route(self):
    self._below_plane = False
    self._crossing = False
    if self.ground_z is None:
        return
    media = self._wire_media()
    if _medium_spec.BELOW in media and _medium_spec.ABOVE in media:
        self._crossing = True  # zero crossing tents: cross blocks only
    elif _medium_spec.BELOW in media:
        self._below_plane = True


RazorSolver._refuse_buried_geometry = _detached_route


def deck(m, ground=True):
    d = crossing_deck(1)
    b, a = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [
        [n * m for n in b[:-1]] + [b[-1]],
        [a[0]] + [n * m for n in a[1:]],
    ]
    below, above = d["wires"]
    d["wires"] = [below + np.array([0.5, 0.0, -0.3]), above + np.array([0.0, 0.0, 0.3])]
    d.pop("junctions")
    d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
    if not ground:
        for k in ("ground_z", "ground_eps", "ground_model"):
            d.pop(k)
    return d


out = []
for m in (1, 2, 4, 8):
    for medium in ("free", "soilA"):
        for name, cls, kw in (
            ("razor_2p", RazorSolver, {"nec5_quadrature": True}),
            ("bspline_d2", BSplineSolver, {}),
        ):
            if medium == "free" and name == "bspline_d2":
                continue
            t0 = time.perf_counter()
            try:
                s = cls(**deck(m, medium != "free"), **kw)
                Y = np.asarray(s.compute_y_matrix())
            except Exception as e:  # noqa: BLE001 - probe reports any refusal verbatim
                print(
                    f"x{m} {medium} {name}: {type(e).__name__}: {str(e)[:400]}",
                    flush=True,
                )
                continue
            dt = time.perf_counter() - t0
            if name == "razor_2p" and medium == "soilA":
                assert s._crossing and not s._below_plane
            nonrec = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
            Z = np.linalg.inv(Y)
            z11, z22, z12 = complex(Z[0, 0]), complex(Z[1, 1]), complex(Z[0, 1])
            out.append(
                dict(
                    x=m,
                    medium=medium,
                    solver=name,
                    z11=[z11.real, z11.imag],
                    z22=[z22.real, z22.imag],
                    z12=[z12.real, z12.imag],
                    nonrec=nonrec,
                    t=round(dt, 2),
                )
            )
            print(
                f"x{m:<2d} {medium:5s} {name:10s} Z11 {z11.real:9.3f}{z11.imag:+9.3f}j "
                f"Z22 {z22.real:8.3f}{z22.imag:+9.3f}j Z12 {z12.real:7.3f}{z12.imag:+7.3f}j "
                f"nonrec {nonrec:.3e} {dt:6.2f} s",
                flush=True,
            )

p = pathlib.Path(__file__).with_suffix(".json")
p.write_text(json.dumps(out, indent=1))
print("wrote", p)
