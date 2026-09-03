"""momwire#838 part 1, phase 0: what shape should the sub-1 deg theta band be?

`_SOMM_BELOW_TH_MIN_DEG = 1.0`'s comment block offers two candidate designs
for reaching BLE 1937's geometry (0.64 deg at the 45 ft tip, 0.21 deg at
135 ft) and asserts that "no uniform lattice resolves that", the drift being
roughly linear in log theta. This probe tests the untested half of that:
does a LOG-spaced theta band actually interpolate to the band's own bar
(4.7e-4, the U2 figure for dtheta = 1 deg at dr = 0.05 lambda_m)?

It does -- and it is the wrong design anyway. A uniform lattice over the
same RESTRICTED band beats it by three to four orders at equal node count
and lower panel cost. The comment's claim is true of one GLOBAL dtheta and
false of a banded one, which is the structure the grid already has (it
carries a graze band and a steep band already, split at _SOMM_TH_SPLIT_DEG).

Both candidates are scored the same way: 4-point Lagrange along the band's
own coordinate, against `iv_surfaces_direct_below` at a COMMON set of
off-node queries, relative to each query point's own scale over the four
surfaces. The common query set matters -- scoring each lattice at its own
cell midpoints compares different points and flatters the log band.

    python scratch/probe838_grazing_band.py
"""

import numpy as np

from momwire import _ground_refl
from momwire import _sommerfeld_below as below
from momwire._sommerfeld import SommerfeldGrid, _SURF_KEYS

C0 = 299792458.0
F7 = 7e6
K7 = 2.0 * np.pi * F7 / C0
OM7 = 2.0 * np.pi * F7
EPS0 = 8.8541878128e-12
SOILS = {"A": (13.0, 0.005), "B": (5.0, 0.001), "C": (30.0, 0.02)}
R1LS = (0.2, 1.0, 2.0)
BAR = 4.7e-4

# Off-node, geometric across the band, interior enough that every candidate
# gives each query a full four-node stencil.
TH_Q = np.exp(np.linspace(np.log(0.115), np.log(0.87), 17))


def surf_at(eps_t, r1, th_deg):
    d = below.iv_surfaces_direct_below(
        eps_t, K7, np.full(th_deg.shape, r1), np.radians(th_deg), rtol=1e-9, omega=OM7
    )
    return np.stack([d[k] for k in _SURF_KEYS])


def interp(nodes_c, vals, q_c):
    """The grid's own scheme: 4-point Lagrange along a UNIFORM coordinate."""
    h = nodes_c[1] - nodes_c[0]
    f = (q_c - nodes_c[0]) / h
    j0 = np.clip(np.floor(f).astype(int) - 1, 0, len(nodes_c) - 4)
    w = SommerfeldGrid._lagrange4(f - j0)
    jj = j0[:, None] + np.arange(4)[None, :]
    return np.einsum("snj,nj->sn", vals[:, jj], w)


def lattice(kind, n, lo=0.1, hi=1.0):
    """theta nodes and the coordinate they are uniform in."""
    if kind == "log":
        c = np.linspace(np.log(lo), np.log(hi), n)
        return np.exp(c), c
    th = np.linspace(lo, hi, n)
    return th, th


def panels(th_deg):
    """Tail panels, the fill currency: ~6.4/tan(theta) per node (#553 U2)."""
    return float(np.sum(6.4 / np.tan(np.radians(th_deg))))


def main():
    print(
        f"bar {BAR:.1e}   band [0.1, 1.0] deg   {len(TH_Q)} common queries in "
        f"[{TH_Q[0]:.3f}, {TH_Q[-1]:.3f}] deg"
    )
    print(f"worst over soils A/B/C x R1/lam_m {R1LS} x the 4 surfaces\n")
    print(
        f"{'kind':>8} {'nodes':>6} {'dtheta':>8} {'worst rel':>11} {'vs bar':>6} {'panels':>9}"
    )
    ref_cache = {}
    for kind in ("uniform", "log"):
        for n in (4, 5, 7, 10, 13, 19):
            th_nodes, c_nodes = lattice(kind, n)
            worst = 0.0
            for soil, ground in SOILS.items():
                eps_t = _ground_refl.eps_tilde(ground, OM7, EPS0)
                lam_m = below.lambda_medium(eps_t, K7)
                for r1l in R1LS:
                    r1 = r1l * lam_m
                    key = (soil, r1l)
                    if key not in ref_cache:
                        ref_cache[key] = surf_at(eps_t, r1, TH_Q)
                    ref = ref_cache[key]
                    got = interp(
                        c_nodes,
                        surf_at(eps_t, r1, th_nodes),
                        np.log(TH_Q) if kind == "log" else TH_Q,
                    )
                    scale = np.abs(ref).max(axis=0)
                    worst = max(
                        worst, float((np.abs(got - ref) / scale[None, :]).max())
                    )
            dth = "geom" if kind == "log" else f"{0.9 / (n - 1):.3f}"
            print(
                f"{kind:>8} {n:6d} {dth:>8} {worst:11.3e} "
                f"{'PASS' if worst < BAR else 'FAIL':>6} {panels(th_nodes):9.0f}"
            )
        print()


if __name__ == "__main__":
    main()
