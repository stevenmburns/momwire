"""momwire#1027 step 2(b), P2b.5: the clean source ladder carried further.

`step2b_uniform.row` unchanged (every segment divided by F, panel boundaries
held, ports matched, radius and knot asserted on both engines), on the rungs
P2b.4 did not reach: a = 0.1 mm at F = 8, and a = 0.05 mm at F = 1 / 2 / 4 / 8.

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/1027-rod/step2b_uniform_more.py [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from step2b_uniform import row

RUNGS = [(0.0001, L, 8) for L in (0.15, 0.60)] + [
    (0.00005, L, f) for L in (0.15, 0.60) for f in (1, 2, 4, 8)
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    exe = os.environ["NEC5_EXE"]
    meta = dict(
        nec5_exe=exe, nec5_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest()
    )
    print(meta, flush=True)
    out = [row(L, f, a) for a, L, f in RUNGS]
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
