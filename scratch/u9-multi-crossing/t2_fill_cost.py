"""U9 fill-cost timing (PLAN.md Amendment 5): one deck, cold then warm, in one
fresh process, on whichever momwire PYTHONPATH selects. `run_t2.sh` alternates
the two trees (main 1ca8725 and the src branch) deck by deck and repeats.

  PYTHONPATH=<momwire src>:<antennaknobs src> python t2_fill_cost.py --deck NAME --out F.jsonl

Cold is the first solve in the process: every below/below grid is built and
filled. Warm is a second solve with the engine's result cache cleared, so the
grids are reused. This is the #983 harness's spelling (`scratch/914-study/
ladder_today.py`): MomwireEngine, BSplineSolver degree 2, soil A. Peak memory is
the process's max RSS, not tracemalloc, which would tax the Python side.
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import momwire
from momwire import BSplineSolver
from momwire import _sommerfeld_below as below

SOIL_A = ("finite", 13.0, 0.005)
CATALOG = {
    "brv_default": ("verticals.buried_radial_vertical", {}),
    "buried_dipole": ("specialty.buried_dipole", {}),
    "ebc_default": ("verticals.elevated_buried_counterpoise", {}),
    "brv48": ("verticals.buried_radial_vertical", {"n_radials": 48}),
    "brv_corner": (
        "verticals.buried_radial_vertical",
        {"depth": 0.02, "radial_factor": 1.5},
    ),
}


def catalog_pair(name):
    import antennaknobs.web.examples  # noqa: F401 -- binds register_all first
    from antennaknobs.designs import verticals  # noqa: F401
    from antennaknobs.engines.momwire import MomwireEngine
    import importlib

    design, over = CATALOG[name]
    b = importlib.import_module(f"antennaknobs.designs.{design}").Builder()
    for k, v in over.items():
        setattr(b, k, v)
    eng = MomwireEngine(
        b, solver=BSplineSolver, solver_kwargs={"degree": 2}, ground=SOIL_A
    )

    def cold():
        return complex(eng.impedance()[0])

    def warm():
        eng._solved_cache = None
        return complex(eng.impedance()[0])

    return cold, warm


def dipole935_pair():
    sys.path.insert(0, str(Path(momwire.__file__).resolve().parents[2] / "tests"))
    from test_grazing_band_lo_935 import _shape_z

    return (lambda: complex(_shape_z(0.003, 41))), (
        lambda: complex(_shape_z(0.003, 41))
    )


def commit_of(path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except OSError:
        return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True, choices=sorted([*CATALOG, "dipole935"]))
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    src = Path(momwire.__file__).resolve().parent
    rec = dict(
        deck=args.deck,
        rep=args.rep,
        momwire=str(src),
        commit=commit_of(src),
        floor_deg=below._SOMM_BELOW_TH_MIN_DEG,
        budget=below._MAX_TAIL_PANELS,
        t_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    cold, warm = (
        dipole935_pair() if args.deck == "dipole935" else catalog_pair(args.deck)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            t0 = time.perf_counter()
            z_cold = cold()
            rec["cold_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            z_warm = warm()
            rec["warm_s"] = time.perf_counter() - t0
            rec["z_cold"] = [z_cold.real, z_cold.imag]
            rec["z_warm"] = [z_warm.real, z_warm.imag]
        except Exception as exc:  # noqa: BLE001 - a record, not a handler
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    grids = [
        g
        for g in __import__(
            "momwire._sommerfeld", fromlist=["_GRID_CACHE"]
        )._GRID_CACHE.values()
        if getattr(g, "regime", None) == "below"
    ]
    rec["low_band_filled"] = any(
        g._regions[i]["filled"] for g in grids for i in g._band_lo_idx
    )
    rec["maxrss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
    )
    with args.out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))
    _ = np


if __name__ == "__main__":
    main()
