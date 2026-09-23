"""Probe 10 (momwire#1149 U2): antennaknobs' buried power-balance leg on the
catalog buried_radial_vertical (N radials), soil A, through AK's own
MomwireEngine: radiated fraction eta (AK's hemispherical integral), R_in and
eta*R_in, razor-2p (fixed / unfixed) against bspline d2.

Run with this worktree's venv and antennaknobs on PYTHONPATH (read-only, no
bytecode written into that tree):

  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<ak>/src .venv/bin/python probe10...

Razor's capability row is flipped IN-PROCESS for the engine's pre-flight;
nothing on disk changes.

Usage: probe10_power_balance.py [N ...]
"""

import sys
import warnings
from pathlib import Path

import numpy as np
from antennaknobs.designs.verticals.buried_radial_vertical import Builder as BRV
from antennaknobs.engines.momwire import MomwireEngine
from antennaknobs.far_field import radiated_fraction

import momwire
from momwire import razor as _razor
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

HERE = Path(__file__).resolve().parent
assert str(HERE.parents[1]) in momwire.__file__, momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_CROSSING = True
_c = RazorSolver.capabilities
RazorSolver.capabilities = _c._replace(
    buried=True,
    refusals={k: v for k, v in _c.refusals.items() if k != "buried+crossing_junction"},
)
_fixed = RazorSolver._crossing_node_charges


def _zero(self, geom, tents, *a, **kw):
    return np.zeros((geom["n_basis_total"], len(tents)), dtype=np.complex128)


for n in [int(x) for x in sys.argv[1:]] or [1, 2, 4]:
    rows = {}
    for name, s, kw in (
        ("bspline_d2", BSplineSolver, {}),
        ("razor_fixed", RazorSolver, {"nec5_quadrature": True}),
        ("razor_unfixed", RazorSolver, {"nec5_quadrature": True}),
    ):
        RazorSolver._crossing_node_charges = (
            _zero if name == "razor_unfixed" else _fixed
        )
        b = BRV()
        b.n_radials = n
        e = MomwireEngine(
            b, solver=s, solver_kwargs=kw, ground=("finite", 13.0, 0.005), ground_z=0.0
        )
        z = e.impedance()[0]
        eta = radiated_fraction(e.far_field())
        rows[name] = (z, eta)
    zb, eb = rows["bspline_d2"]
    for name, (z, eta) in rows.items():
        print(
            f"N={n} {name:13s} Z {z.real:8.3f}{z.imag:+8.3f}j  eta {eta:.4f}  "
            f"eta*R {eta * z.real:7.3f}  dR {z.real - zb.real:+6.2f}  "
            f"d(eta*R) {100 * (eta * z.real / (eb * zb.real) - 1):+6.2f} %",
            flush=True,
        )
