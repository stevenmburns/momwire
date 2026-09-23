"""momwire#1159 probe 1: WHICH block carries the sign. At eps~ = 1 the mixed
deck is free space, so its assembled G must equal the free-space G on the
same wires block for block. Compare the four quadrants (above/below rows x
above/below columns) of SG's mixed G against its free-space G."""

import sys
import warnings

import numpy as np

sys.path.insert(0, "scratch/1156-sg-buried-loading")

from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from probe5_mixed_refined import mixed  # noqa: E402

one = mixed(2, eps=(1.0, 0.0))
free = {
    k: v for k, v in one.items() if k not in ("ground_z", "ground_eps", "ground_model")
}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    s1 = SinusoidalGalerkinSolver(**one)
    sf = SinusoidalGalerkinSolver(**free)
    g1 = s1._build_geometry()
    gf = sf._build_geometry()
    G1, _ = s1._assemble_Z(g1, s1.k)
    Gf, _ = sf._assemble_Z(gf, sf.k)
below = s1._below_segments(g1)
a, b = np.nonzero(~below)[0], np.nonzero(below)[0]
for name, r, c in (("aa", a, a), ("ab", a, b), ("ba", b, a), ("bb", b, b)):
    m1, mf = G1[np.ix_(r, c)], Gf[np.ix_(r, c)]
    plus = np.linalg.norm(m1 - mf) / np.linalg.norm(mf)
    minus = np.linalg.norm(m1 + mf) / np.linalg.norm(mf)
    print(f"{name}: |G1-Gf|/|Gf| = {plus:.3e}   |G1+Gf|/|Gf| = {minus:.3e}")
