"""Probe 1: re-run momwire#813 step 4's far-mesh ladder on TODAY's main.

crossing_deck(1) at soil A, far mesh scaled x1/x2/x4 (every edge except the
two node-adjacent ones, which reproduces the 09-03 N = 30/56/108), razor-2p
(nec5_quadrature=True) with the shelved family flags monkeypatched ON, against
bspline d2 and bspline d1 (the same tent basis under Galerkin testing).

Nothing in the momwire tree is modified: the flags are module globals read at
call time, set here in-process only.
"""

import json
import pathlib
import sys
import time
import warnings

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

rungs = [int(a) for a in (sys.argv[1:] or ["1", "2", "4"])]
out = []
for m in rungs:
    d = crossing_deck(1)
    b, a = d["n_per_edge_per_wire"]
    # node-adjacent edges: below last, above first
    d["n_per_edge_per_wire"] = [
        [n * m for n in b[:-1]] + [b[-1]],
        [a[0]] + [n * m for n in a[1:]],
    ]
    N = sum(d["n_per_edge_per_wire"][0]) + sum(d["n_per_edge_per_wire"][1])
    row = {"x": m, "N": N}
    for name, cls, kw in (
        ("bspline_d2", BSplineSolver, {}),
        ("bspline_d1", BSplineSolver, {"degree": 1}),
        ("razor_2p", RazorSolver, {"nec5_quadrature": True}),
    ):
        dd = dict(d)
        if cls is RazorSolver:
            dd.pop("junctions")
        t0 = time.perf_counter()
        z, _ = cls(**dd, **kw).compute_impedance()
        dt = time.perf_counter() - t0
        z = complex(z)
        row[name] = [z.real, z.imag, round(dt, 2)]
        print(
            f"x{m} N={N} {name:11s} {z.real:10.4f} {z.imag:+10.4f}j  {dt:7.2f} s",
            flush=True,
        )
    out.append(row)

p = pathlib.Path(__file__).with_suffix(".json")
p.write_text(json.dumps(out, indent=1))
print("wrote", p)
