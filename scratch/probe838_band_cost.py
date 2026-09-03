"""momwire#838 part 1, phase 0: what the sub-1 deg band costs, and where the
theta floor honestly sits.

Two questions the design needs answered before it is chosen:

  * the per-node tail cost against theta (the comment block's ~6.4/tan theta,
    stated R1-independent -- checked here), and what that does to a whole
    grid fill;
  * where the floor actually is. It is NOT set by convergence: it is set by
    `_MAX_TAIL_PANELS`, and the way that cap is reached is silent -- the
    contour stops at the cap, bumps `Health.nonconvergent`, and returns a
    value anyway. This measures how wrong that value is, which is the number
    that says whether the floor is 0.1 deg or lower.

    python scratch/probe838_band_cost.py
"""

import time

import numpy as np

from momwire import _ground_refl
from momwire import _sommerfeld_below as below
from momwire._sommerfeld import _SOMM_TH_SPLIT_DEG, _SURF_KEYS

C0 = 299792458.0
F7 = 7e6
K7 = 2.0 * np.pi * F7 / C0
OM7 = 2.0 * np.pi * F7
EPS0 = 8.8541878128e-12
SOIL_A = (13.0, 0.005)


def per_node_cost(eps_t, lam_m):
    print(f"{'theta':>7} {'6.4/tan':>8} {'R1/lam_m':>9} {'panels':>7} {'sec/pt':>8}")
    for thd in (2.0, 1.0, 0.5, 0.2, 0.1):
        for r1l in (0.2, 1.0, 2.0):
            h = below.Health()
            t0 = time.perf_counter()
            below.iv_surfaces_direct_below(
                eps_t,
                K7,
                np.array([r1l * lam_m]),
                np.radians([thd]),
                rtol=1e-9,
                omega=OM7,
                health=h,
            )
            dt = time.perf_counter() - t0
            print(
                f"{thd:7.2f} {6.4 / np.tan(np.radians(thd)):8.0f} {r1l:9.2f} "
                f"{h.as_dict()['max_tail_panels']:7d} {dt:8.3f}"
            )


def grid_cost(eps_t, lam_m):
    """Node and panel counts for the grid today, and for the added band."""
    r_break = below._SOMM_BELOW_R_BREAK_LAMBDA_M * lam_m
    r1_max = below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m
    dr_in = below._SOMM_BELOW_DR_NEAR_LAMBDA_M * lam_m
    dr_out = below._SOMM_BELOW_DR_LAMBDA_M * lam_m
    pad = below._SOMM_BELOW_PAD_ROWS

    def n_r_of(r0, r1, dr):
        pad_lo = pad if r0 > 0.0 else 0
        return max(int(np.ceil((r1 - r0) / dr)) + 1 + pad_lo + pad, 4)

    n_rows = n_r_of(0.0, r_break, dr_in) + n_r_of(r_break, r1_max, dr_out)

    def band(t0, t1, dth):
        nth = int(round((t1 - t0) / dth)) + 1
        th = t0 + dth * np.arange(nth)
        return n_rows * nth, float(np.sum(6.4 / np.tan(np.radians(th)))) * n_rows

    tot_n = tot_p = 0.0
    print("today:")
    for label, t0, t1, dth in (
        ("graze 1-30", 1.0, _SOMM_TH_SPLIT_DEG, below._SOMM_BELOW_DTH_GRAZE_DEG),
        ("steep 30-90", _SOMM_TH_SPLIT_DEG, 90.0, below._SOMM_BELOW_DTH_STEEP_DEG),
    ):
        n, p = band(t0, t1, dth)
        print(f"  {label:12s} dth {dth:4.2f}  nodes {n:5d}  panels {p:10.0f}")
        tot_n += n
        tot_p += p
    print(f"  {'TOTAL':12s} {'':9s} nodes {tot_n:5.0f}  panels {tot_p:10.0f}")
    print("adding a uniform sub-band [0.1, 1.0] deg:")
    for dth in (0.3, 0.25, 0.2, 0.15):
        n, p = band(0.1, 1.0, dth)
        print(
            f"  dth {dth:4.2f}  nodes +{n:5d} (+{100 * n / tot_n:5.1f}%)  "
            f"panels +{p:10.0f} (+{100 * p / tot_p:5.1f}%)"
        )
    t0 = time.perf_counter()
    g = below.SommerfeldGridBelow(eps_t, K7, r1_max, omega=OM7)
    dt = time.perf_counter() - t0
    nodes = sum(r["n_r"] * r["n_th"] for r in g._regions)
    print(f"measured fill today: {dt:.2f} s for {nodes} nodes")


def floor_probe(eps_t, lam_m):
    """The cap is silent. How wrong is a capped point?"""
    print(f"cap _MAX_TAIL_PANELS = {below._MAX_TAIL_PANELS}")
    print(
        f"{'theta':>7} {'6.4/tan':>8} {'panels':>7} {'nonconv':>8} {'rel vs lifted cap':>18}"
    )
    orig = below._MAX_TAIL_PANELS
    for thd in (0.12, 0.10, 0.09, 0.08, 0.05, 0.023):
        h = below.Health()
        a = below.iv_surfaces_direct_below(
            eps_t,
            K7,
            np.array([lam_m]),
            np.radians([thd]),
            rtol=1e-9,
            omega=OM7,
            health=h,
        )
        try:
            below._MAX_TAIL_PANELS = 40000
            b = below.iv_surfaces_direct_below(
                eps_t, K7, np.array([lam_m]), np.radians([thd]), rtol=1e-9, omega=OM7
            )
        finally:
            below._MAX_TAIL_PANELS = orig
        av = np.array([a[k][0] for k in _SURF_KEYS])
        bv = np.array([b[k][0] for k in _SURF_KEYS])
        rel = float(np.abs(av - bv).max() / np.abs(bv).max())
        d = h.as_dict()
        print(
            f"{thd:7.3f} {6.4 / np.tan(np.radians(thd)):8.0f} "
            f"{d['max_tail_panels']:7d} {d['nonconvergent']:8d} {rel:18.3e}"
        )


def main():
    eps_t = _ground_refl.eps_tilde(SOIL_A, OM7, EPS0)
    lam_m = below.lambda_medium(eps_t, K7)
    print(f"soil A @ 7 MHz  eps_t = {eps_t:.4f}  lam_m = {lam_m:.4f} m\n")
    print("--- per-node cost ---")
    per_node_cost(eps_t, lam_m)
    print("\n--- whole-grid cost ---")
    grid_cost(eps_t, lam_m)
    print("\n--- where the floor sits ---")
    floor_probe(eps_t, lam_m)


if __name__ == "__main__":
    main()
