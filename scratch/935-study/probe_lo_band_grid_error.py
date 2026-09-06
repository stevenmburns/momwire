"""momwire#935: the low band's interpolation error ON THE REAL GRID.

The lattice study measured 6.834e-10 for four uniform nodes over [0.05, 0.1]
deg, but that probe interpolated along theta ALONE against a common query set.
The shipped grid also interpolates along R1, and the gate's bar has to come
from the thing the gate measures. Queries sit at cell midpoints and thirds --
where a 4-point Lagrange stencil is worst -- and off the R1 nodes too.
"""

import numpy as np

from momwire import _ground_refl
from momwire import _sommerfeld_below as below
from momwire._sommerfeld import _SURF_KEYS

C0 = 299792458.0
EPS0 = 8.8541878128e-12
SOILS = {"A": (13.0, 0.005), "B": (20.0, 0.03), "C": (5.0, 0.001)}


def deck(soil, f):
    k2 = 2.0 * np.pi * f / C0
    om = 2.0 * np.pi * f
    eps_t = _ground_refl.eps_tilde(SOILS[soil], om, EPS0)
    return eps_t, k2, om, below.lambda_medium(eps_t, k2)


worst, where = 0.0, None
for soil in SOILS:
    for f in (7e6, 21e6):
        eps_t, k2, om, lam_m = deck(soil, f)
        g = below.SommerfeldGridBelow(
            eps_t, k2, below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m, omega=om
        )
        g._ensure_band_lo()
        reg = g._regions[below.region_index(below._ZONE_INNER, below._BAND_LO)]
        nodes = reg["th0"] + reg["dth"] * np.arange(reg["n_th"])
        d = np.diff(nodes)
        th_q = np.concatenate([nodes[:-1] + 0.5 * d, nodes[:-1] + 0.33 * d])
        for r1l in (0.2, 1.0, 1.9):
            r1 = np.full(th_q.shape, r1l * lam_m)
            got = g.eval(r1, th_q)
            ref = below.iv_surfaces_direct_below(
                eps_t, k2, r1, th_q, rtol=1e-9, omega=om
            )
            for k in _SURF_KEYS:
                a = np.asarray(got[k], dtype=complex)
                b = np.asarray(ref[k], dtype=complex)
                rel = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
                if rel > worst:
                    worst, where = rel, (soil, f / 1e6, r1l, k)
        print(f"  {soil}/{f / 1e6:.0f} MHz  running worst {worst:.3e}")

print(f"\nWORST {worst:.4e} at {where}")
print(
    f"the band's own bar (momwire#553 U2) is 4.7e-4 -> headroom {4.7e-4 / worst:.0f}x"
)
