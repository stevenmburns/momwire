"""Probe 15 (momwire#1149 U2): convergence onto bspline on the catalog
buried_radial_vertical. The deck is captured ONCE from antennaknobs'
MomwireEngine (the solver kwargs it builds, recorded by a stand-in class),
then every edge count is scaled x1/x2/x4 and both solvers are called
directly. The feed position is the engine's own; razor snaps it to a knot,
bspline takes it exactly, so the gap carries razor's feed quantisation.

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<ak>/src .venv/bin/python probe15... N
"""

import sys
import warnings
from pathlib import Path

from antennaknobs.designs.verticals.buried_radial_vertical import Builder as BRV
from antennaknobs.engines.momwire import MomwireEngine

import momwire
from momwire import razor as _razor
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

HERE = Path(__file__).resolve().parent
assert str(HERE.parents[1]) in momwire.__file__, momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_CROSSING = True
CAPTURED = {}


class _Recorder(BSplineSolver):
    def __init__(self, **kw):
        CAPTURED.update(kw)
        super().__init__(**kw)


n_rad = int(sys.argv[1]) if len(sys.argv) > 1 else 4
b = BRV()
b.n_radials = n_rad
e = MomwireEngine(b, solver=_Recorder, ground=("finite", 13.0, 0.005), ground_z=0.0)
e.impedance()
base = {k: v for k, v in CAPTURED.items() if k not in ("cancel",)}
print("feeds", base["feeds"], "edges", base["n_per_edge_per_wire"], flush=True)
prev = None
for m in (1, 2, 4):
    d = dict(base)
    d["n_per_edge_per_wire"] = [
        [n * m for n in ed] for ed in base["n_per_edge_per_wire"]
    ]
    zb = complex(BSplineSolver(**d).compute_impedance()[0])
    dr = {k: v for k, v in d.items() if k != "junctions"}
    zr = complex(RazorSolver(**dr, nec5_quadrature=True).compute_impedance()[0])
    gap = zr - zb
    tag = "" if prev is None else f" (/{abs(prev) / abs(gap):4.2f})"
    prev = gap
    print(
        f"BRV N={n_rad} x{m} razor {zr.real:8.3f}{zr.imag:+8.3f}j  bs {zb.real:8.3f}"
        f"{zb.imag:+8.3f}j  gap {gap.real:+7.3f}{gap.imag:+7.3f}j{tag}",
        flush=True,
    )
