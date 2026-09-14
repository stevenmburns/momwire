"""U9 (d) (PLAN.md Amendment 3): one LPDA solve per process -- feed Z, wall
time, max RSS -- appended to a JSONL record.

  PYTHONPATH=<momwire src>:<antennaknobs src> python d2_lpda_solve.py \\
      --deck full|one --rung r1|far3 --engine momwire|nec5 [--nec5-exe PATH] --out F.jsonl

Decks are the files `d0_lpda_decks.py` wrote (the full deck is the corpus file
itself). Each is built through antennaknobs' `builder_from_file(path, refine=1)`
and `make_engine_factory`, the U2 ladder's own path, so both engines read the
same bytes. antennaknobs' AK#1464 vertex preflight is stubbed
(`momwire.bspline.below_reach_refusal`), named in PLAN.md; momwire's own scope and
serve plan still decide.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import time
import warnings
from pathlib import Path

import momwire
from momwire import bspline

bspline.below_reach_refusal = lambda *a, **k: None  # AK#1464 stub (PLAN.md Amendment 3)

HERE = Path(__file__).resolve().parent
FULL = (
    Path.home()
    / "antennas/nec-wild/community/cebik-w4rnl/models/LPDAs/nec"
    / "lpma3r5-4-6el86ft75o-buriedradials.nec"
)
DECKS = {
    ("full", "r1"): FULL,
    ("full", "far3"): HERE / "lpda_full_far3.nec",
    ("one", "r1"): HERE / "lpda_one_node.nec",
    ("one", "far3"): HERE / "lpda_one_node_far3.nec",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", choices=("full", "one"), required=True)
    ap.add_argument("--rung", choices=("r1", "far3"), required=True)
    ap.add_argument("--engine", choices=("momwire", "nec5"), required=True)
    ap.add_argument("--nec5-exe", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.nec5_exe:
        os.environ["NEC5_EXE"] = args.nec5_exe

    from antennaknobs.cli import _GROUND_UNSET, file_ground_default, make_engine_factory
    from antennaknobs.file_designs import builder_from_file

    path = DECKS[(args.deck, args.rung)]
    src = Path(momwire.__file__).resolve().parent
    rec = dict(
        deck=args.deck,
        rung=args.rung,
        engine=args.engine,
        path=str(path),
        momwire=str(src),
        momwire_commit=subprocess.run(
            ["git", "-C", str(src), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip(),
        nec5_exe=os.environ.get("NEC5_EXE"),
        t_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        try:
            b = builder_from_file(str(path), refine=1)
            eng = make_engine_factory(
                args.engine, file_ground_default(_GROUND_UNSET, b)
            )(b())
            z = complex(eng.impedance()[0])
            rec["z"] = [z.real, z.imag]
            rec["segments"] = sum(w.n_seg for w in b.file_deck_parsed.wires)
        except Exception as exc:  # noqa: BLE001 - a record, not a handler
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        rec["seconds"] = round(time.perf_counter() - t0, 2)
    rec["maxrss_mb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
    )
    rec["maxrss_children_mb"] = round(
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1
    )
    with args.out.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
