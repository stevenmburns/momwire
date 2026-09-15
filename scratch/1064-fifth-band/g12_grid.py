"""momwire#1064 G1a / G2a (capture) and G2b / G2c (direct), one tree per process.

  PYTHONPATH=<momwire src> python g12_grid.py --tree v055|main|branch --out g12_TREE.json [--direct]

For every medium (soils A/B/C x 7/21 MHz, and soil A at 3.5 MHz) on a grid
tabulated to the cap:

  * OLD set (theta >= 0.05 deg): every low- and mid-band node, those bands'
    cell midpoints and thirds, and theta in {1, 2, 5, 30, 60, 89.9} deg, at
    R1/lam_m in {0.02, 0.2, 1, 1.9, 3, 3.9}. Recorded: `grid.eval` (numpy) on
    all four surfaces, and the C++-dispatched `remainder_field_proj_below` on
    one pair per point.
  * FLOOR set (theta < 0.05 deg, main and branch only): the floor band's two
    nodes under the edge and its two cells' midpoints and thirds, at the same R1.
  * --direct (branch): the floor band's and the low band's cell midpoints and
    thirds against `iv_surfaces_direct_below` (rtol 1e-9), at R1/lam_m in
    {0.2, 1, 1.9, 3, 3.9}: the worst relative error per band.

Query points are computed here from literal degrees, never read from a grid, so
the three trees are asked the same floats. `g12_compare.py` reads the files.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import momwire
from momwire import _ground_refl
from momwire import _sommerfeld_below as below
from momwire._sommerfeld import _SURF_KEYS

C0 = 299792458.0
EPS0 = 8.8541878128e-12
SOILS = {"A": (13.0, 0.005), "B": (20.0, 0.03), "C": (5.0, 0.001)}
MEDIA = [(s, f) for s in "ABC" for f in (7e6, 21e6)] + [("A", 3.5e6)]
R1_OLD = (0.02, 0.2, 1.0, 1.9, 3.0, 3.9)
R1_DIRECT = (0.2, 1.0, 1.9, 3.0, 3.9)
D_LO = 0.05 / 3.0
FLOOR = 0.05 - 2.0 * D_LO


def cells(nodes):
    nodes = np.asarray(nodes, dtype=float)
    d = np.diff(nodes)
    return list(nodes[:-1] + 0.5 * d) + list(nodes[:-1] + 0.33 * d)


LOW_NODES = [0.05 + D_LO * k for k in range(4)]
MID_NODES = [0.1 + 0.225 * k for k in range(5)]
OLD_THETA = (
    LOW_NODES
    + MID_NODES
    + cells(LOW_NODES)
    + cells(MID_NODES)
    + [1.0, 2.0, 5.0, 30.0, 60.0, 89.9]
)
FLOOR_OWN = [FLOOR + D_LO * k for k in range(3)]  # 0.016667, 0.033333, 0.05
FLOOR_THETA = FLOOR_OWN[:2] + cells(FLOOR_OWN) + [0.049999999]


def medium(soil, f):
    k2 = 2.0 * math.pi * f / C0
    om = 2.0 * math.pi * f
    eps_t = _ground_refl.eps_tilde(SOILS[soil], om, EPS0)
    return eps_t, k2, om, below.lambda_medium(eps_t, k2), below.k_medium(eps_t, k2)


def pair(r1, th):
    rho, hh = r1 * math.cos(th), r1 * math.sin(th)
    obs = np.array([[0.0, 0.0, -0.5 * hh]])
    src = np.array([[rho, 0.0, -0.5 * hh]])
    t_obs = np.array([[1.0, 0.0, 0.0]])
    t_src = np.array([[0.6, 0.0, 0.8]])
    return obs, t_obs, src, t_src


def cz(v):
    v = complex(v)
    return [v.real, v.imag]


def capture(g, thetas_deg, r1s, lam_m, k2, k_m):
    out = []
    for r1l in r1s:
        for thd in thetas_deg:
            r1, th = r1l * lam_m, math.radians(thd)
            vals = g.eval(np.array([r1]), np.array([th]))
            obs, t_obs, src, t_src = pair(r1, th)
            kern = below.remainder_field_proj_below(
                obs, t_obs, src, t_src, 0.0, k2, k_m, g
            )
            out.append(
                dict(
                    r1_lam=r1l,
                    theta_deg=repr(thd),
                    surf={k: cz(vals[k][0]) for k in _SURF_KEYS},
                    kernel=cz(kern[0, 0]),
                )
            )
    return out


def direct(g, thetas_deg, eps_t, k2, om, lam_m):
    worst, where = 0.0, None
    th = np.radians(np.asarray(thetas_deg, dtype=float))
    for r1l in R1_DIRECT:
        r1 = np.full(th.shape, r1l * lam_m)
        got = g.eval(r1, th)
        ref = below.iv_surfaces_direct_below(eps_t, k2, r1, th, rtol=1e-9, omega=om)
        for k in _SURF_KEYS:
            a = np.asarray(got[k], dtype=complex)
            b = np.asarray(ref[k], dtype=complex)
            rel = np.abs(a - b) / np.maximum(np.abs(b), 1e-300)
            i = int(np.argmax(rel))
            if rel[i] > worst:
                worst, where = float(rel[i]), (r1l, float(thetas_deg[i]), k)
    return worst, where


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--direct", action="store_true")
    args = ap.parse_args()
    has_floor_band = hasattr(below, "_SOMM_BELOW_TH_BAND_FLOOR_HI_DEG")
    serves_floor = below._SOMM_BELOW_TH_MIN_DEG < 0.05
    rec = dict(
        tree=args.tree,
        momwire=str(Path(momwire.__file__).resolve().parent),
        floor_deg=below._SOMM_BELOW_TH_MIN_DEG,
        has_floor_band=has_floor_band,
        media={},
    )
    for soil, f in MEDIA:
        t0 = time.perf_counter()
        eps_t, k2, om, lam_m, k_m = medium(soil, f)
        g = below.SommerfeldGridBelow(
            eps_t, k2, below._SOMM_BELOW_R1_CAP_LAMBDA_M * lam_m, omega=om
        )
        m = dict(old=capture(g, OLD_THETA, R1_OLD, lam_m, k2, k_m))
        if serves_floor:
            m["floor"] = capture(g, FLOOR_THETA, R1_OLD, lam_m, k2, k_m)
        if args.direct:
            m["direct_floor_band"] = direct(g, cells(FLOOR_OWN), eps_t, k2, om, lam_m)
            m["direct_low_band"] = direct(g, cells(LOW_NODES), eps_t, k2, om, lam_m)
        m["seconds"] = round(time.perf_counter() - t0, 1)
        rec["media"][f"{soil}/{f / 1e6:g}MHz"] = m
        print(args.tree, soil, f, m["seconds"], "s", flush=True)
    args.out.write_text(json.dumps(rec))


if __name__ == "__main__":
    main()
