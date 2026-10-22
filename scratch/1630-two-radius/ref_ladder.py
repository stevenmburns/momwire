"""Licensed references for momwire#1630, black box: NEC-4.2 (segment source)
and NEC-5 (knot source) over the decks in decks.py. One JSON line per run,
flushed. Usage: python ref_ladder.py OUT.jsonl [r ...]"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import decks

NEC42 = Path.home() / "antennas/nec42-build/nec42cl-static"
NEC5 = Path.home() / "antennas/NEC5-downloads/nec5-linux/nec5cl"
RADII = {
    "main": (0.019, 0.001),
    "one": (0.001, 0.001),
    "above_half": (0.0095, 0.001),
    "below_half": (0.019, 0.0005),
}
_NUM = r"[-+]?\d\.\d+E[-+]\d+"


def z_of(text):
    """First ANTENNA INPUT PARAMETERS row's R, X (both engines print it)."""
    i = text.index("ANTENNA INPUT PARAMETERS")
    for line in text[i:].splitlines()[1:12]:
        nums = re.findall(_NUM, line)
        if len(nums) >= 8:
            return float(nums[4]), float(nums[5])
    raise ValueError("no impedance row")


def run(exe, deck_text):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        (p / "in.nec").write_text(deck_text)
        t0 = time.perf_counter()
        subprocess.run(
            [str(exe), "in.nec", "out.nec"], cwd=td, check=True, capture_output=True
        )
        dt = time.perf_counter() - t0
        return z_of((p / "out.nec").read_text(errors="replace")), dt


def main():
    """OUT.jsonl NRAD[,NRAD] r [r ...]"""
    out = Path(sys.argv[1])
    nrads = [int(x) for x in sys.argv[2].split(",")]
    rs = [int(x) for x in sys.argv[3:]] or [1, 2, 4, 8]
    with out.open("a") as fh:
        for n_rad in nrads:
            for variant in ("hub", "slope"):
                for cfg, (a_v, a_b) in RADII.items():
                    for r in rs:
                        jobs = [("nec42", "mid"), ("nec42", "bottom"), ("nec5", "mid")]
                        for eng, feed in jobs:
                            if cfg != "main" and feed == "bottom":
                                continue
                            mk = decks.nec42 if eng == "nec42" else decks.nec5
                            exe = NEC42 if eng == "nec42" else NEC5
                            text = mk(variant, r, feed, a_v=a_v, a_b=a_b, n_rad=n_rad)
                            try:
                                (R, X), dt = run(exe, text)
                                rec = dict(R=R, X=X, s=round(dt, 2))
                            except Exception as e:  # noqa: BLE001 - recorded, run goes on
                                rec = dict(error=repr(e)[:300])
                            rec.update(
                                engine=eng,
                                variant=variant,
                                n_rad=n_rad,
                                cfg=cfg,
                                r=r,
                                feed=feed,
                            )
                            fh.write(json.dumps(rec) + "\n")
                            fh.flush()


if __name__ == "__main__":
    main()
