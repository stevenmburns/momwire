"""momwire#1156 probe 9: SG's wire_loss_power on buried decks.

For wire_conductivity on a buried dipole, a mixed deck (single feed on the
buried wire) and a crossing deck: SG's new readout, the INHERITED one (real
k = w/c shapes, the path a buried deck took before), and bspline's and
razor's, under refinement. Watts at the solve's own drive."""

import sys
import warnings


sys.path.insert(0, "tests")
sys.path.insert(0, "scratch/1156-sg-buried-loading")

from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from momwire.sinusoidal import SinusoidalSolver  # noqa: E402
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from probe5_mixed_refined import mixed  # noqa: E402
from test_jacket_in_soil_1154 import dipole  # noqa: E402
from test_razor_crossing_loading_1149 import crossing  # noqa: E402

SIG = 3.5e7
DECKS = {
    "dipole": lambda m: dipole(n=20 * m, wire_conductivity=SIG),
    "mixed": lambda m: dict(
        mixed(m), feeds=[(0, 2.5, 1 + 0j)], wire_conductivity=[SIG, SIG]
    ),
    "crossing": lambda m: dict(
        crossing(m), wire_radius=1e-3, wire_conductivity=[SIG, SIG]
    ),
}


def p_loss(cls, d, inherited=False):
    d = dict(d)
    extra = {}
    if cls is RazorSolver:
        d.pop("junctions", None)
        extra = dict(nec5_quadrature=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = cls(**d, **extra)
        _z, coeffs = s.compute_impedance()
        if inherited:
            return SinusoidalSolver.wire_loss_power(s, coeffs)[0]
        return s.wire_loss_power(coeffs)[0]


if __name__ == "__main__":
    for name in sys.argv[1:] or DECKS:
        for m in (1, 2, 4):
            d = DECKS[name](m)
            sg = p_loss(SinusoidalGalerkinSolver, d)
            try:
                old = f"{p_loss(SinusoidalGalerkinSolver, d, inherited=True):.6e}"
            except Exception as e:  # noqa: BLE001 - the probe reports what raised
                old = type(e).__name__
            bs = p_loss(BSplineSolver, d)
            rz = p_loss(RazorSolver, d)
            print(
                f"{name:9s} m={m} SG {sg:.6e} (inherited {old}) | bspline {bs:.6e} "
                f"rel {abs(sg / bs - 1):.2e} | razor {rz:.6e} rel {abs(sg / rz - 1):.2e}",
                flush=True,
            )
