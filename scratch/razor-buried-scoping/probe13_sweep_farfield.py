"""Probe 13: do the seams a flip would expose work on razor buried decks?

(1) momwire `compute_impedance_swept` on a wholly-below deck and on the
    crossing deck (razor, flags on) vs per-frequency single solves.
(2) AK's buried far field (U8, antennaknobs in_medium) through MomwireEngine on
    the catalog buried dipole and BRV with razor (capability row flipped
    in-process) vs bspline: peak gain and its elevation.
"""

import pathlib
import sys
import warnings

import numpy as np

MW = pathlib.Path("/home/smburns/antennas/antennaknobs/momwire")
sys.path.insert(0, str(MW / "tests"))
import momwire  # noqa: E402
from momwire import razor as _razor  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from test_buried_serve_553 import C0  # noqa: E402
from test_crossing_serve_524 import crossing_deck  # noqa: E402

from antennaknobs.designs.specialty.buried_dipole import Builder as BuriedDipole  # noqa: E402
from antennaknobs.designs.verticals.buried_radial_vertical import Builder as BRV  # noqa: E402
from antennaknobs.engines.momwire import MomwireEngine  # noqa: E402

assert str(MW) in momwire.__file__
warnings.filterwarnings("ignore")
_razor._SERVE_BELOW_PLANE = True
_razor._SERVE_CROSSING = True
_caps = RazorSolver.capabilities
RazorSolver.capabilities = _caps._replace(
    buried=True,
    refusals={
        k: v
        for k, v in _caps.refusals.items()
        if k not in ("buried", "buried+crossing_junction")
    },
)

# (1) swept
fr = np.array([6.8e6, 7.0e6, 7.2e6])
k_arr = 2 * np.pi * fr / C0
for label, d in (
    (
        "wholly-below",
        dict(
            wires=[np.array([(0, 0, -2.15), (0, 0, -0.15)])],
            n_per_edge_per_wire=[[20]],
            feeds=[(0, 1.0, 1 + 0j)],
            wavelength=C0 / 7e6,
            wire_radius=0.001,
            ground_z=0.0,
            ground_eps=(13.0, 0.005),
            ground_model="sommerfeld",
        ),
    ),
    ("crossing", {k: v for k, v in crossing_deck(1).items() if k != "junctions"}),
):
    try:
        zs = np.asarray(
            RazorSolver(**d, nec5_quadrature=True).compute_impedance_swept(k_arr)
        )
        zs = zs.reshape(len(fr), -1)[:, 0]
        singles = []
        for f in fr:
            dd = dict(d)
            dd["wavelength"] = C0 / f
            singles.append(
                complex(RazorSolver(**dd, nec5_quadrature=True).compute_impedance()[0])
            )
        print(
            f"swept {label:12s} max |swept - single| = {np.max(np.abs(zs - np.array(singles))):.3e} ohm",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001 - probe reports any failure verbatim
        print(f"swept {label}: {type(e).__name__}: {str(e)[:300]}", flush=True)

# (2) far field through AK
for cls in (BuriedDipole, BRV):
    b = cls()
    for name, solver, kw in (
        ("bspline_d2", BSplineSolver, {}),
        ("razor_2p", RazorSolver, {"nec5_quadrature": True}),
    ):
        try:
            eng = MomwireEngine(
                b,
                solver=solver,
                solver_kwargs=kw,
                ground=("finite", 13.0, 0.005),
                ground_z=0.0,
            )
            ff = eng.far_field(n_theta=45, n_phi=36, del_theta=2, del_phi=10)
            g = np.asarray(ff.rings)
            i = np.unravel_index(np.argmax(g), g.shape)
            print(
                f"far field {cls.__module__.split('.')[-1]:24s} {name:10s} peak {ff.max_gain:7.3f} dBi "
                f"at theta {np.asarray(ff.thetas)[i[0]]:.0f} deg; in-medium moment fraction "
                f"{ff.in_medium_moment_fraction}",
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 - probe reports any failure verbatim
            print(
                f"far field {cls.__module__.split('.')[-1]} {name}: {type(e).__name__}: {str(e)[:300]}",
                flush=True,
            )
