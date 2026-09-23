"""Probe 2: a reference-free gate for razor's crossing serve.

crossing_deck(1) with TWO knot-aligned ports: one on the above wire at
arclength 4.5 (z = 4.5) and one on the below wire at arclength 1.0
(z = -1.0). Both sit on a knot at every far-mesh multiple, so razor's
knot snap does not move them between rungs (the #845 feed-quantisation
artefact is removed from the ladder).

Reported per rung, per medium:
  * Z11 (above port driving point, from inv(Y)),
  * non-reciprocity  |Y12 - Y21| / |Y12|   (0 for any Galerkin solver),
for razor-2p in: free space (same geometry, no ground), eps~ = 1, soil A;
and bspline d2 at soil A for Z11.

Path testing is only ASYMPTOTICALLY reciprocal (momwire#309). If the razor
cross blocks are a consistent discretisation of the same operator, the soil
non-reciprocity must decay with the mesh like the free-space one does.
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

rungs = [int(a) for a in (sys.argv[1:] or ["1", "2", "4", "8"])]


def deck(m, medium):
    d = crossing_deck(1)
    b, a = d["n_per_edge_per_wire"]
    d["n_per_edge_per_wire"] = [
        [n * m for n in b[:-1]] + [b[-1]],
        [a[0]] + [n * m for n in a[1:]],
    ]
    d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
    if medium == "free":
        for k in ("ground_z", "ground_eps", "ground_model"):
            d.pop(k)
    elif medium == "eps1":
        d["ground_eps"] = (1.0, 0.0)
    elif medium == "diel":
        d["ground_eps"] = (13.0, 0.0)
    return d


out = []
for m in rungs:
    for medium in ("free", "eps1", "soilA", "diel"):
        for name, cls, kw in (
            ("razor_2p", RazorSolver, {"nec5_quadrature": True}),
            ("bspline_d2", BSplineSolver, {}),
        ):
            if name == "bspline_d2" and medium in ("free", "eps1"):
                continue
            d = deck(m, medium)
            if cls is RazorSolver:
                d.pop("junctions")
            t0 = time.perf_counter()
            s = cls(**d, **kw)
            Y = np.asarray(s.compute_y_matrix())
            dt = time.perf_counter() - t0
            Zoc = np.linalg.inv(Y)
            nonrec = abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1])
            z11 = complex(Zoc[0, 0])
            z22 = complex(Zoc[1, 1])
            row = dict(
                x=m,
                medium=medium,
                solver=name,
                z11=[z11.real, z11.imag],
                z22=[z22.real, z22.imag],
                nonrec=nonrec,
                t=round(dt, 2),
            )
            out.append(row)
            print(
                f"x{m:<2d} {medium:5s} {name:10s} Z11 {z11.real:9.3f}{z11.imag:+9.3f}j "
                f"Z22 {z22.real:9.3f}{z22.imag:+9.3f}j  nonrec {nonrec:.3e}  {dt:6.2f} s",
                flush=True,
            )

tag = "_".join(str(r) for r in rungs)
p = pathlib.Path(__file__).with_name(f"probe2_reciprocity_{tag}.json")
p.write_text(json.dumps(out, indent=1))
print("wrote", p)
