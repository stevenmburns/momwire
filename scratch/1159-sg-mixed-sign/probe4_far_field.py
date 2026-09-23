"""momwire#1159 probe 4: the far field of a detached deck in soil, SG vs
bspline, BOTH ports driven. The pattern is `_far_moments` over
`element_currents` — the EZNEC serve's own route — on a theta grid over the
upper hemisphere at phi = 0 and 90 deg. Reports the relative gap of the
complex (M_theta, M_phi) and of the power pattern, per refinement.

Usage: probe4_far_field.py [deck] [m ...]   deck in {mixed, detached, hub}
Prints which momwire was imported (side-by-side against main's source).
"""

import sys
import warnings

import numpy as np

import momwire
from momwire._far_readout import Ground, _far_moments
from momwire.bspline import BSplineSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

sys.path.insert(0, "scratch/1156-sg-buried-loading")
sys.path.insert(0, "tests")
from probe5_mixed_refined import mixed  # noqa: E402
from test_razor_detached_1149 import detached, detached_hub  # noqa: E402

DECKS = {"mixed": mixed, "detached": detached, "hub": detached_hub}
THETA = np.radians(np.linspace(0.0, 89.0, 90))
PHI = np.radians([0.0, 45.0, 90.0])


def pattern(cls, d, ports=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = cls(**d)
        sol = s.compute_port_solution()
        c = sol.coeffs if ports is None else sol.coeffs[:, ports]
        coeffs = c.sum(axis=1) if c.ndim == 2 else c
        mid, moment, _n, _d = s.element_currents(coeffs)
    eps_r, sigma = d["ground_eps"]
    f = 299792458.0 / d["wavelength"]
    mt, mp = _far_moments(
        mid,
        moment,
        2 * np.pi / d["wavelength"],
        THETA,
        PHI,
        Ground("sommerfeld", eps_r, sigma),
        0.0,
        f,
    )
    return np.stack([mt, mp])


def gaps(a, b):
    field = np.linalg.norm(a - b) / np.linalg.norm(b)
    pa, pb = (np.abs(a) ** 2).sum(0), (np.abs(b) ** 2).sum(0)
    power = np.abs(pa - pb).max() / pb.max()
    return field, power


if __name__ == "__main__":
    print("momwire from", momwire.__file__)
    name = sys.argv[1] if len(sys.argv) > 1 else "mixed"
    ms = [int(x) for x in sys.argv[2:]] or [1, 2, 4]
    for m in ms:
        d = DECKS[name](m)
        for label, ports in (("both", None), ("port0", 0), ("port1", 1)):
            a = pattern(SinusoidalGalerkinSolver, d, ports)
            b = pattern(BSplineSolver, d, ports)
            fg, pg = gaps(a, b)
            print(
                f"{name} m={m} {label}: |M_sg - M_bs|/|M_bs| {fg:.3e}  "
                f"max|P_sg - P_bs|/max P_bs {pg:.3e}",
                flush=True,
            )
