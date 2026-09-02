"""momwire#647 Part B: is the DENSE sea-water overshoot an ORDER defect?

#647's headline is that the dense route's ground correction over sea water
(eps_r 81, sigma 5) overshoots by 1.637 at h/lambda = 1.09e-4 while every
other soil sits at 0.98-0.99, and that the overshoot grows with sigma. The
issue's own hypothesis is that `_quadrature.remainder_qp` keys the remainder
order on GEOMETRY alone (`c * len / R_min`) while at sigma = 5 the image
structure sharpens on the SKIN-DEPTH scale, which that predicate cannot see.

The question this probe answers, before anything is built on it: does the
dense route's sea-water answer MOVE when the remainder order is raised, and
where does it stop moving? No NEC5 binary is needed for that — it is the
convergence-trend reading #647 itself names as the way to gate this ("gate
the trend, never the size"). If sea water is already converged at the default
order, the order is not the mechanism and #647 is a different arc from the
fast-route keying gap; if it is not, the two share a root.

The quantity is the issue's own: Delta = Z(GN 0) - Z(GN 1), the ground
correction, on one horizontal wire at deep grazing.
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))

from momwire import BSplineSolver
from test_sommerfeld_ground import _GRAZE_WL, _grazing_wire

SOILS = [
    ("sea", 81.0, 5.0),
    ("vgood", 20.0, 0.0303),
    ("avg", 13.0, 0.005),
    ("poor", 5.0, 0.001),
    ("diel", 2.5, 1e-05),
]
ORDERS = [None, 32, 64, 96, 128, 192, 256]


def delta(h, eps_r, sigma, n_qp=None):
    """Z(finite ground) - Z(PEC): the ground correction, dense route."""
    common = dict(ground_model="sommerfeld")
    kwg = _grazing_wire(h, ground_eps=(eps_r, sigma), **common)
    kwp = _grazing_wire(h)  # PEC
    if n_qp is not None:
        kwg["n_qp_sommerfeld"] = n_qp
    zg, _ = BSplineSolver(**kwg).compute_impedance()
    zp, _ = BSplineSolver(**kwp).compute_impedance()
    return complex(zg) - complex(zp)


def main():
    for hl in (1.09e-4, 1e-3):
        h = hl * _GRAZE_WL
        print(f"\n{'=' * 78}")
        print(f"h/lambda = {hl:.2e}   Delta = Z(finite) - Z(PEC), dense route")
        print(f"{'=' * 78}")
        print(
            f"  {'soil':>6} {'eps_r':>6} {'sigma':>8} "
            + " ".join(f"{('dflt' if q is None else q):>10}" for q in ORDERS)
        )
        conv = {}
        for name, er, sg in SOILS:
            vals = [delta(h, er, sg, q) for q in ORDERS]
            conv[name] = vals[-1]
            print(
                f"  {name:>6} {er:>6.1f} {sg:>8.4g} "
                + " ".join(f"{abs(v):>10.4f}" for v in vals)
            )
            # How far each order sits from the highest-order answer.
            rel = [abs(v - vals[-1]) / abs(vals[-1]) for v in vals]
            print(
                f"  {'':>6} {'':>6} {'rel to 256':>8} "
                + " ".join(f"{r:>10.2e}" for r in rel)
            )
        print()
        print("  the reading: |Delta(default)/Delta(converged)| — the shape of")
        print("  #647's ratio column, with the CONVERGED answer standing in for")
        print("  the binary's truth:")
        for name, er, sg in SOILS:
            d0 = delta(h, er, sg, None)
            r = abs(d0) / abs(conv[name])
            ang = np.angle(d0 / conv[name], deg=True)
            print(
                f"    {name:>6}: {r:>6.3f}  {ang:>+6.1f} deg   "
                f"|err| {abs(d0 - conv[name]):>8.3f} ohm"
            )


main()
