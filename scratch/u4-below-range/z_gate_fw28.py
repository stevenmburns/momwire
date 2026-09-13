"""U4 PX1 follow-up: Z gates at fresh water, 28 MHz (MEASUREMENTS.md, "PX1,
measured: one row misses"). Registered before any run.

The record's synthetic deck, rescaled. The mast scales with the free-space
wavelength (3.5 -> 28 MHz) and the screen with the in-medium wavelength (soil A
at 3.5 MHz -> fresh water at 28 MHz), so the radial tips again sit 4.76
lambda_m apart. The depth is the missed row's 0.02 m, and the segment counts
are the record's. `--radials 4` is the registered deck; `--radials 16` adds the
most oblique pairs a fan puts past the cap.

Spellings, guards and bar are T1's (tests/test_below_past_cap_zero_1053.py on
the src branch). Run against the src branch tree:

  PYTHONPATH=<u4src>/src python scratch/u4-below-range/z_gate_fw28.py \\
      --radials 4 --out scratch/u4-below-range/z_fw28_r4.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import time
import warnings
from pathlib import Path

import numpy as np

import momwire
from momwire import _below_interface, _ground_refl
from momwire import _sommerfeld_below as below
from momwire.bspline import BSplineSolver

C0 = 299792458.0
F_REC = 3.5e6
F = 28e6
SOIL = (80.0, 0.001)
LAM_M_REC = 15.965887973309902  # soil A at 3.5 MHz, the record's deck
DEPTH = 0.02
RADIUS = 0.001
SHIPPED_CAP = 4.0
EXTENDED_CAP = 5.0
BAR = 1e-2


def lam_m():
    om = 2.0 * math.pi * F
    eps_t = _ground_refl.eps_tilde(SOIL, om, 8.8541878128e-12)
    return below.lambda_medium(eps_t, om / C0)


MAST = 21.4 * F_REC / F
RADIAL = 38.0 * lam_m() / LAM_M_REC


def deck(refine, n_radials):
    lower = {1: [MAST / 11], 3: [MAST / 33, (MAST / 33) * 2]}[refine]
    edges = {1: [1, 10], 3: [1, 1, 31]}[refine]
    mast = np.array(
        [(0.0, 0.0, 0.0)] + [(0.0, 0.0, z) for z in lower] + [(0.0, 0.0, MAST)]
    )
    rise = np.array([(0.0, 0.0, 0.0), (0.0, 0.0, -DEPTH)])
    radials = []
    for i in range(n_radials):
        a = 2.0 * math.pi * i / n_radials
        tip = (RADIAL * math.cos(a), RADIAL * math.sin(a), -DEPTH)
        radials.append(np.array([(0.0, 0.0, -DEPTH), tip]))
    return dict(
        wires=[mast, rise, *radials],
        n_per_edge_per_wire=[edges, [refine]] + [[19 * refine]] * n_radials,
        feeds=[(0, MAST / 22, 1 + 0j)],
        wavelength=C0 / F,
        wire_radius=RADIUS,
        ground_z=0.0,
        junctions=[
            [(0, "start"), (1, "start")],
            [(1, "end")] + [(2 + i, "start") for i in range(n_radials)],
        ],
        ground_eps=SOIL,
        ground_model="sommerfeld",
    )


def solve(refine, n_radials, seen):
    """Z at this rung, with the plan's R1 and every grid's tabulation recorded
    in in-medium wavelengths, and the grid objects kept for eviction."""
    real_plan = _below_interface.serve_plan
    real_grid = below.get_grid_below
    sig = inspect.signature(real_plan)

    def plan(*a, **kw):
        out = real_plan(*a, **kw)
        k_m = sig.bind(*a, **kw).arguments["k_m"]
        seen.setdefault("plan_r1_wl", []).append(
            out["r1_below"] * abs(k_m) / (2.0 * math.pi)
        )
        return out

    def grid(*a, **kw):
        g = real_grid(*a, **kw)
        seen.setdefault("grid_r1_wl", []).append(g.r1_max / g.lam_m)
        seen.setdefault("grids", []).append(g)
        return g

    _below_interface.serve_plan = plan
    below.get_grid_below = grid
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.time()
            z = complex(BSplineSolver(**deck(refine, n_radials)).compute_impedance()[0])
            seen.setdefault("seconds", []).append(round(time.time() - t0, 1))
            return z
    finally:
        _below_interface.serve_plan = real_plan
        below.get_grid_below = real_grid


def evict(grids):
    cache = below._GRID_CACHE
    for key in [k for k, g in cache.items() if any(g is x for x in grids)]:
        del cache[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radials", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert below._SOMM_BELOW_R1_CAP_LAMBDA_M == SHIPPED_CAP
    rec = dict(
        momwire=momwire.__file__,
        radials=args.radials,
        f_hz=F,
        soil=SOIL,
        depth_m=DEPTH,
        lam_m=lam_m(),
        mast_m=MAST,
        radial_m=RADIAL,
        tip_span_lambda_m=2.0 * RADIAL / lam_m(),
        bar=BAR,
    )
    shipped, extended, ext3 = {}, {}, {}
    try:
        z_ship = solve(1, args.radials, shipped)
        evict(shipped["grids"])
        below._SOMM_BELOW_R1_CAP_LAMBDA_M = EXTENDED_CAP
        z_ext1 = solve(1, args.radials, extended)
        z_ext3 = solve(3, args.radials, ext3)
    except ValueError as exc:
        rec.update(verdict="REFUSED", error=str(exc)[:600])
        args.out.write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec))
        return
    finally:
        below._SOMM_BELOW_R1_CAP_LAMBDA_M = SHIPPED_CAP
    delta = abs(z_ship - z_ext1)
    step = abs(z_ext3 - z_ext1)
    guards = dict(
        plan_past_cap=max(shipped["plan_r1_wl"]) > SHIPPED_CAP,
        shipped_grid_at_cap=max(shipped["grid_r1_wl"]) <= SHIPPED_CAP * (1 + 1e-12),
        extended_grid_past_cap=max(extended["grid_r1_wl"]) > SHIPPED_CAP,
        delta_not_bit_zero=delta > 1e-9 * abs(z_ext1),
    )
    if not all(guards.values()):
        verdict = "GUARD-MISS"
    else:
        verdict = "HIT" if delta <= BAR * step else "MISS"
    for d in (shipped, extended, ext3):
        d.pop("grids", None)
    rec.update(
        z_shipped_r1=repr(z_ship),
        z_extended_r1=repr(z_ext1),
        z_extended_r3=repr(z_ext3),
        delta_ohm=delta,
        ladder_step_ohm=step,
        delta_over_step=delta / step,
        delta_over_abs_z=delta / abs(z_ext1),
        guards=guards,
        shipped=shipped,
        extended=extended,
        extended_r3=ext3,
        verdict=verdict,
    )
    args.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
