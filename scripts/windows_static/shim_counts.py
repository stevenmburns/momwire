"""How often the MinGW vector-math shim ran, per deck (windows-static.yml).

    python scripts/windows_static/shim_counts.py LABEL=EXE [LABEL=EXE ...]

Each EXE is a build with MW_EXE_VMATH_SHIM_COUNT=ON, which reports its calls
on stderr at exit. time_exes.py's two timing decks, at one thread and at the
size policy's count. A run with no report, or with no fast-path call, fails.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from time_exes import hub, yagi

LINE = re.compile(r"^mw_vmath_shim: .*$", re.M)


def main(argv: list[str]) -> int:
    work = Path(tempfile.mkdtemp())
    decks = {"fs_yagi_6x201": yagi(6, 201), "buried_hub32x20": hub(32, 20)}
    bad = 0
    for name, text in decks.items():
        deck = work / f"{name}.nec"
        deck.write_text(text)
        for threads in ("1", "auto"):
            env = dict(os.environ)
            env.pop("MOMWIRE_THREADS", None)
            if threads != "auto":
                env["MOMWIRE_THREADS"] = threads
            for arg in argv:
                label, exe = arg.split("=", 1)
                r = subprocess.run(
                    [exe, str(deck), str(work / "t.out")],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                m = LINE.search(r.stderr)
                print(
                    f"{label} {name} threads={threads} rc={r.returncode}: "
                    f"{m.group(0) if m else 'NO REPORT'}"
                )
                fast = (
                    sum(int(v) for v in re.findall(r"fast=(\d+)", m.group(0)))
                    if m
                    else 0
                )
                if r.returncode != 0 or fast == 0:
                    bad = 1
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
