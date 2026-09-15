"""momwire#1064 G5 (and G1b's Z): one deck, cold then warm, in one fresh process,
on whichever momwire PYTHONPATH selects. U9 amendment 5's harness
(`scratch/u9-multi-crossing/t2_fill_cost.py`), with two decks under 0.05 deg
added: #935's dipole at 1 mm, and U9's two-node soil-A deck at 11 m.

  PYTHONPATH=<momwire src>:<antennaknobs src> python t5_fill_cost.py \\
      --deck NAME --rep N --tree v055|main|branch --out F.jsonl

Cold is the first solve in the process, so every below/below grid is built and
filled. Warm is a second solve reusing the grids: the catalog decks clear the
engine's result cache, and the solver-built decks build a new solver. Each row
records which grazing bands were filled, Z to the last bit, and max RSS.
"""

from __future__ import annotations

import argparse
import importlib
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

TESTS = Path(momwire.__file__).resolve().parents[2] / "tests"
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


def dipole_pair(depth_m):
    sys.path.insert(0, str(TESTS))
    from test_grazing_band_lo_935 import _shape_z

    def solve():
        return complex(_shape_z(depth_m, 41))

    return solve, solve


def twonode_pair(separation):
    sys.path.insert(0, str(TESTS))
    from test_crossing_serve_524 import two_node_deck

    def solve():
        z, _ = BSplineSolver(**two_node_deck(separation)).compute_impedance()
        return complex(np.asarray(z).ravel()[0])

    return solve, solve


DECKS = {
    **{name: (lambda name=name: catalog_pair(name)) for name in CATALOG},
    "dipole935": lambda: dipole_pair(0.003),
    "dipole1mm": lambda: dipole_pair(0.001),
    "twonode11": lambda: twonode_pair(11.0),
}


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


def bands_filled():
    from momwire._sommerfeld import _GRID_CACHE

    grids = [g for g in _GRID_CACHE.values() if getattr(g, "regime", None) == "below"]
    out = {}
    for name in ("_band_floor_idx", "_band_lo_idx", "_band_idx"):
        if not grids or not hasattr(grids[0], name):
            out[name] = None
            continue
        out[name] = any(
            g._regions[i]["filled"] for g in grids for i in getattr(g, name)
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True, choices=sorted(DECKS))
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    src = Path(momwire.__file__).resolve().parent
    rec = dict(
        deck=args.deck,
        rep=args.rep,
        tree=args.tree,
        momwire=str(src),
        commit=commit_of(src),
        floor_deg=below._SOMM_BELOW_TH_MIN_DEG,
        budget=below._MAX_TAIL_PANELS,
        t_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            cold, warm = DECKS[args.deck]()
            t0 = time.perf_counter()
            z_cold = cold()
            rec["cold_s"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            z_warm = warm()
            rec["warm_s"] = time.perf_counter() - t0
            rec["z_cold"] = [z_cold.real, z_cold.imag]
            rec["z_warm"] = [z_warm.real, z_warm.imag]
        except Exception as err:  # noqa: BLE001 - a record, not a handler
            rec["error"] = f"{type(err).__name__}: {err!s:.300}"
    rec["bands_filled"] = bands_filled()
    rec["maxrss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
    )
    with args.out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec)[:500])


if __name__ == "__main__":
    main()
