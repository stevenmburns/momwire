"""momwire#1156 probe 10: the crossing deck's fill, twice, loading OFF; and
the BARE crossing deck's swept vs single-frequency solve. Is the 5e-7 seen
in the gates the fill's own repeatability, independent of loading?"""

import math
import sys
import warnings

import numpy as np

sys.path.insert(0, "tests")

from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from test_sg_buried_loading_1156 import assembled, xing  # noqa: E402

s = SinusoidalGalerkinSolver(**xing())
G1, *_ = assembled(s, False)
G2, *_ = assembled(s, False)
s2 = SinusoidalGalerkinSolver(**xing())
G3, *_ = assembled(s2, False)
sc = np.abs(G1).max()
print(f"same instance twice: max|dG| {np.abs(G1 - G2).max():.3e} (|G| {sc:.3e})")
print(f"fresh instance     : max|dG| {np.abs(G1 - G3).max():.3e}")
i, j = np.unravel_index(np.argmax(np.abs(G1 - G3)), G1.shape)
print(f"  worst entry ({i},{j}) of {G1.shape}")
d = xing()
lams = (d["wavelength"], d["wavelength"] / 1.05)
ks = [2 * math.pi / lam for lam in lams]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sw = np.ravel(SinusoidalGalerkinSolver(**d).compute_impedance_swept(ks))
    si = [
        complex(
            SinusoidalGalerkinSolver(**dict(d, wavelength=lam)).compute_impedance()[0]
        )
        for lam in lams
    ]
    si2 = [
        complex(
            SinusoidalGalerkinSolver(**dict(d, wavelength=lam)).compute_impedance()[0]
        )
        for lam in lams
    ]
print("bare swept - single :", np.abs(sw - np.array(si)))
print("bare single - single:", np.abs(np.array(si) - np.array(si2)))
