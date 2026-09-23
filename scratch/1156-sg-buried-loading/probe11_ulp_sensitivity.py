"""momwire#1156 probe 11: the loaded crossing deck's swept first point sits
5e-7 ohm off a fresh single solve (bare: 5.9e-10). Is that the fill's
sensitivity to a one-ulp change of the wavelength, loaded vs bare?"""

import math
import sys
import warnings

import numpy as np

sys.path.insert(0, "tests")

from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from test_sg_buried_loading_1156 import SIG_JACKET, per_wire, xing  # noqa: E402


def zz(d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return complex(SinusoidalGalerkinSolver(**d).compute_impedance()[0])


for label, d in (
    ("bare", xing()),
    ("sigma", per_wire(xing(), wire_conductivity=3.5e7)),
    ("jacket", per_wire(xing(), insulation_radius=1.8e-3, insulation_eps_r=3.5)),
    ("sigma+jacket", per_wire(xing(), **SIG_JACKET)),
):
    lam = d["wavelength"]
    z0 = zz(d)
    z1 = zz(dict(d, wavelength=np.nextafter(lam, math.inf)))
    z2 = zz(dict(d, wavelength=np.nextafter(lam, -math.inf)))
    print(
        f"{label:13s} Z {z0:.6f}  one-ulp moves: {abs(z1 - z0):.3e} {abs(z2 - z0):.3e}"
    )

# The same jump with NO loading at all: bare wires at the jacket's kernel
# radius a' = a (b/a)^(1 - 1/eps_r). If it moves the same, the jump is the
# fill's (a branch that flips on the wavelength), not the loading term's.
a, b, er = 1e-3, 1.8e-3, 3.5
d = dict(xing(), wire_radius=a * (b / a) ** (1 - 1 / er))
lam = d["wavelength"]
z0 = zz(d)
z1 = zz(dict(d, wavelength=np.nextafter(lam, math.inf)))
print(f"{'bare at a_eq':13s} Z {z0:.6f}  one-ulp move: {abs(z1 - z0):.3e}")
