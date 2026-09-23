"""Probe 11: the three AK catalog buried designs through AK's own MomwireEngine,
razor-2p vs bspline d2, with razor's family flags set IN-PROCESS (what a flip
of `_SERVE_BURIED` would serve today, minus the capability-row change).

The engine reads razor's capability row at construction; the row still says
buried=False, so the engine may refuse before the solver is built. That is
reported verbatim - it is part of what the flip changes.
"""

import sys
import time
import traceback
import warnings

sys.path.insert(0, "/home/smburns/antennas/antennaknobs/tests")
import momwire  # noqa: E402
from momwire import razor as _razor  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402

from antennaknobs.designs.specialty.buried_dipole import Builder as BuriedDipole  # noqa: E402
from antennaknobs.designs.verticals.buried_radial_vertical import (
    Builder as BuriedRadialVertical,
)  # noqa: E402
from antennaknobs.designs.verticals.elevated_buried_counterpoise import (  # noqa: E402
    Builder as ElevatedBuriedCounterpoise,
)
from antennaknobs.engines.momwire import MomwireEngine  # noqa: E402

assert "antennaknobs/momwire/src" in momwire.__file__, momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_BELOW_PLANE = True
_razor._SERVE_CROSSING = True
# the flip's capability-row half, in-process: buried=True, the two retired cells gone
_caps = RazorSolver.capabilities
RazorSolver.capabilities = _caps._replace(
    buried=True,
    refusals={
        k: v
        for k, v in _caps.refusals.items()
        if k not in ("buried", "buried+crossing_junction")
    },
)
SOIL = ("finite", 13.0, 0.005)

for cls in (BuriedRadialVertical, ElevatedBuriedCounterpoise, BuriedDipole):
    b = cls()
    for name, solver, kw in (
        ("bspline_d2", BSplineSolver, {}),
        ("razor_2p", RazorSolver, {"nec5_quadrature": True}),
    ):
        t0 = time.perf_counter()
        try:
            eng = MomwireEngine(
                b, solver=solver, solver_kwargs=kw, ground=SOIL, ground_z=0.0
            )
            zz = eng.impedance()
            z = complex(zz[0] if hasattr(zz, "__len__") else zz)
            dt = time.perf_counter() - t0
            print(
                f"{cls.__name__:28s} {name:10s} Z {z.real:9.3f}{z.imag:+9.3f}j  {dt:6.2f} s",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - probe reports any refusal verbatim
            dt = time.perf_counter() - t0
            print(
                f"{cls.__name__:28s} {name:10s} REFUSED/ERR after {dt:.2f} s: "
                f"{type(e).__name__}: {str(e)[:260]}",
                flush=True,
            )
            if not isinstance(e, (ValueError, NotImplementedError)):
                traceback.print_exc()
