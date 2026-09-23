"""Probe 14: AK's buried power-balance leg (tests/test_buried_flow_power_balance.py)
applied to razor-2p on the catalog BRV, N = 1/2/4 radials, soil A: radiated
fraction eta (hemispherical gain integral), R_in, and eta*R_in (radiated watts
per amp squared, the engine-agnostic quantity), razor vs bspline. Razor's
capability row flipped in-process; nothing on disk changes.
"""

import sys
import warnings

sys.path.insert(0, "/home/smburns/antennas/antennaknobs/tests")
from momwire import razor as _razor
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

from antennaknobs.designs.verticals.buried_radial_vertical import Builder as BRV
from antennaknobs.engines.momwire import MomwireEngine
from test_buried_flow_power_balance import radiated_fraction

warnings.filterwarnings("ignore")
_razor._SERVE_BELOW_PLANE = True
_razor._SERVE_CROSSING = True
_c = RazorSolver.capabilities
RazorSolver.capabilities = _c._replace(
    buried=True,
    refusals={
        k: v
        for k, v in _c.refusals.items()
        if k not in ("buried", "buried+crossing_junction")
    },
)

for n in (1, 2, 4):
    for name, s, kw in (
        ("bspline_d2", BSplineSolver, {}),
        ("razor_2p", RazorSolver, {"nec5_quadrature": True}),
    ):
        b = BRV()
        b.n_radials = n
        e = MomwireEngine(
            b, solver=s, solver_kwargs=kw, ground=("finite", 13.0, 0.005), ground_z=0.0
        )
        z = e.impedance()[0]
        eta = radiated_fraction(e.far_field())
        print(
            f"N={n} {name:10s} Z {z.real:8.3f}{z.imag:+8.3f}j  eta {eta:.4f}  eta*R {eta * z.real:7.3f}",
            flush=True,
        )
