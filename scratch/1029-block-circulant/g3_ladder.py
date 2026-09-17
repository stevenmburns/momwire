"""momwire#1029 phase 1, gate G3: the cost ladder, dense against the route.

  PYTHONPATH=<momwire src>:<antennaknobs src> \\
      python scratch/1029-block-circulant/g3_ladder.py --rungs 12 24 48 96 150 \\
          --out g3.jsonl
  ... --one 48 --mode route      # one measurement, the worker the driver spawns

ONE PROCESS PER MEASUREMENT, because peak RSS is a process high-water mark
(`VmHWM`) and cannot be reset: a dense 150-radial fill in the same process
would put its own peak on every later row. The driver spawns the workers one
at a time — never two heavy things at once — and each worker:

  1. solves once COLD, which is what fills the module-level Sommerfeld grid,
     geometry and basis caches;
  2. solves again WARM with the fill and the solve timed separately, which is
     the row. `#1067`'s momwire column is a warm number and this is the same
     thing.

Registered in `PLAN-phase1.md` §6:

  G3a  warm seconds at 48 radials    <= 3.0 s
  G3b  warm seconds at 150 radials   <= 30.14 s (#1067's 3b75639 column),
       and >= 5x faster than dense
  G3c  peak RSS within 5 % of dense at every rung — REGISTERED, not
       discovered: the restricted fill keeps Z full size, so phase 1 buys
       time and not memory (§2's registered consequence).

Amendment 2's prediction for the 150-radial rung is 14 - 30 s, marginal at
the high bound against G3b's 30.14 s bar.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import box as box_info  # noqa: E402 — after the sys.path insert above
from feasibility import GROUND


def peak_rss_mb():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return round(float(line.split()[1]) / 1024.0, 1)
    return None


def build(n_radials, route):
    from antennaknobs.designs.verticals.buried_radial_vertical import Builder
    from antennaknobs.engines.momwire import MomwireEngine

    from momwire import BSplineSolver

    b = Builder()
    b.n_radials = n_radials
    extra = {"rotational_symmetry": True} if route else {}
    eng = MomwireEngine(
        b, solver=BSplineSolver, solver_kwargs={"degree": 2, **extra}, ground=GROUND
    )
    return eng._make_solver(wavelength=eng._wavelength_for(float(b.freq)))


def timed_pass(s, route):
    """One warm solve with the fill and the solve timed apart. The steps are
    `compute_impedance`'s own, in its order, so the split adds up to the
    total it would have returned."""
    t0 = time.perf_counter()
    geom = s._build_geometry()
    supp_seg, polys, kcl_A, wire_knots, wbg = s._build_basis_polynomials(geom)
    n_b = supp_seg.shape[0]
    t_basis = time.perf_counter() - t0

    if route:
        sectors, axial = s._rotational_dof_groups(wbg, n_b)
        rows = s._rotational_rows(geom)
        t0 = time.perf_counter()
        Z = s._compute_Z_operator_buried(geom, supp_seg, polys, rows=rows)
        t_fill = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        Z = s._compute_Z_operator(geom, supp_seg, polys)
        t_fill = time.perf_counter() - t0

    v, port_vectors, _vpf, all_voltages, kcl_con = s._feed_drive_and_readout(
        geom, wire_knots, wbg, n_b, kcl_A
    )
    t0 = time.perf_counter()
    if route:
        coeffs = s._rotational_solve(Z, v, kcl_con, sectors, axial)
    else:
        coeffs = s._solve_with_kcl(Z, v, kcl_con, overwrite=True)
    t_solve = time.perf_counter() - t0
    del Z
    z_in = complex(np.atleast_1d(s._per_feed_z(coeffs, port_vectors, all_voltages))[0])
    return {
        "t_basis_s": round(t_basis, 3),
        "t_fill_s": round(t_fill, 3),
        "t_solve_s": round(t_solve, 4),
        "t_warm_s": round(t_basis + t_fill + t_solve, 3),
        "z_in": [z_in.real, z_in.imag],
        "n_basis": int(n_b),
    }


def one(n_radials, route, passes=2):
    """`passes=2` is the registered measurement: a cold solve to fill the
    module caches, then the timed warm one. `passes=1` skips the cold solve
    and TIMES THE COLD PASS instead — the escape for a rung whose two passes
    do not both fit the address-space cap, since the first pass's arena is
    still mapped when the second allocates. A pass-1 row is an UPPER BOUND on
    the warm number, not the warm number; the rows say which they are.
    """
    t_cold = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = build(n_radials, route)
        if passes > 1:
            t0 = time.perf_counter()
            s.compute_impedance()  # cold: fills the module caches
            t_cold = time.perf_counter() - t0
        rec = timed_pass(s, route)
    rec.update(
        radials=n_radials,
        mode="route" if route else "dense",
        passes=passes,
        warm=passes > 1,
        t_cold_s=None if t_cold is None else round(t_cold, 3),
        peak_rss_mb=peak_rss_mb(),
        chunked=bool(s._buried_chunked_serves),
        hostname=platform.node(),
    )
    if route:
        smap = s._rotational_map
        rec["n_sectors"] = smap.n_sectors
        rec["copy_spread"] = s._rotational_copy_spread
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", type=int, nargs="+", default=[12, 24, 48, 96, 150])
    ap.add_argument("--one", type=int)
    ap.add_argument("--mode", choices=("dense", "route"))
    ap.add_argument("--passes", type=int, default=2, choices=(1, 2))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.one is not None:
        print(json.dumps(one(args.one, args.mode == "route", args.passes)), flush=True)
        return 0

    import antennaknobs

    import momwire

    meta = {
        "_meta": True,
        "gate": "G3",
        # The heads come from the environment because the box runs a worktree
        # rsync'd WITHOUT .git; the laptop that syncs it knows them and says so.
        "momwire_head": os.environ.get("MW1029_MOMWIRE_HEAD")
        or box_info.git_head(Path(momwire.__file__).parent),
        "antennaknobs_head": os.environ.get("MW1029_AK_HEAD")
        or box_info.git_head(Path(antennaknobs.__file__).parent),
        "momwire_path": str(Path(momwire.__file__).parent),
        "box": box_info.provenance(),
        "accel": box_info.accel_variant(),
        "passes": args.passes,
    }
    out = open(args.out, "w") if args.out else None  # noqa: SIM115
    if out:
        out.write(json.dumps(meta) + "\n")
    print(json.dumps(meta))
    rows = {}
    for n in args.rungs:
        for mode in ("dense", "route"):
            proc = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--one",
                    str(n),
                    "--mode",
                    mode,
                    "--passes",
                    str(args.passes),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                print(f"rung {n} {mode} FAILED:\n{proc.stderr[-2000:]}", flush=True)
                continue
            rec = json.loads(proc.stdout.strip().splitlines()[-1])
            rows[(n, mode)] = rec
            print(json.dumps(rec), flush=True)
            if out:
                out.write(json.dumps(rec) + "\n")
                out.flush()
        d, r = rows.get((n, "dense")), rows.get((n, "route"))
        if d and r:
            summary = {
                "_summary": True,
                "radials": n,
                "dense_warm_s": d["t_warm_s"],
                "route_warm_s": r["t_warm_s"],
                "speedup": round(d["t_warm_s"] / max(r["t_warm_s"], 1e-9), 2),
                "dense_fill_s": d["t_fill_s"],
                "route_fill_s": r["t_fill_s"],
                "dense_solve_s": d["t_solve_s"],
                "route_solve_s": r["t_solve_s"],
                "dense_rss_mb": d["peak_rss_mb"],
                "route_rss_mb": r["peak_rss_mb"],
                "rss_ratio": round(r["peak_rss_mb"] / max(d["peak_rss_mb"], 1e-9), 3),
                "z_in_rel": abs(complex(*r["z_in"]) - complex(*d["z_in"]))
                / abs(complex(*d["z_in"])),
            }
            print(json.dumps(summary), flush=True)
            if out:
                out.write(json.dumps(summary) + "\n")
                out.flush()
    if out:
        out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
