"""U9 D2 (PLAN.md): NEC-5 run natively on the LPDA deck files, as written, to
tell a deck problem from antennaknobs' multiport-route refusal. Printout output
only (the licence courtesy rule): the feed Z is read off with antennaknobs' own
parser.

  PYTHONPATH=<antennaknobs src> python d3_nec5_native.py --nec5-exe PATH --out F
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from antennaknobs.engines.nec5 import NEC5Engine, run_deck

HERE = Path(__file__).resolve().parent
DECKS = {
    "full": Path.home()
    / "antennas/nec-wild/community/cebik-w4rnl/models/LPDAs/nec"
    / "lpma3r5-4-6el86ft75o-buriedradials.nec",
    "one": HERE / "lpda_one_node.nec",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nec5-exe", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rec = dict(nec5_exe=args.nec5_exe, runs={})
    for name, path in DECKS.items():
        t0 = time.perf_counter()
        row = dict(path=str(path))
        try:
            text = run_deck(args.nec5_exe, path.read_text(), timeout=3600)
            blocks = NEC5Engine._parse_input_parameters(text)
            row["blocks"] = [
                [dict(tag=t, seg=s, z=[z.real, z.imag]) for t, s, z in rows]
                for rows in blocks
            ]
        except Exception as exc:  # noqa: BLE001 - a record, not a handler
            row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        row["seconds"] = round(time.perf_counter() - t0, 2)
        rec["runs"][name] = row
        print(name, json.dumps(row)[:400], flush=True)
    args.out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
