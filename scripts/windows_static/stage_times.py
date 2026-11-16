"""Where each executable's time goes, stage by stage (MOMWIRE_TIMES), on one machine.

    python scripts/windows_static/stage_times.py LABEL=EXE [LABEL=EXE ...]

time_exes.py's fs_yagi 6x201 and 32-radial buried hub, at one thread and at
the size policy's count, REPS runs each (interleaved), medians of each stage
of the executable's own `times:` line plus the wall time seen from outside.
`start` is the process's age at main() (loader + static initialisers), `rest`
the run half, printout text and file write. Indicative, as time_exes.py says.
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from time_exes import hub, yagi

REPS = 3
COLS = ("start", "seam", "ground", "fill", "factor", "rest", "other", "total", "wall")


def parse(stderr: str) -> dict[str, float]:
    for line in stderr.splitlines():
        if line.startswith("times:"):
            kv = dict(f.split("=", 1) for f in line.split()[1:] if "=" in f)
            g = lambda k: float(kv.get(k, 0.0))  # noqa: E731
            return {
                "start": g("start_ms"),
                "seam": g("seam_ms"),
                "ground": g("ground_ms"),
                "fill": g("fill_ms"),
                "factor": g("factor_ms"),
                "rest": g("rp_ms")
                + g("near_ms")
                + g("run_ms")
                + g("printout_ms")
                + g("write_ms")
                + g("load_ms"),
                "other": g("other_ms"),
                "total": g("total_ms"),
                "threads": g("threads"),
            }
    return {}


def main(argv: list[str]) -> int:
    exes = [a.split("=", 1) for a in argv]
    work = Path(tempfile.mkdtemp())
    decks = {"fs_yagi_6x201": yagi(6, 201), "buried_hub32x20": hub(32, 20)}
    for name, text in decks.items():
        (work / f"{name}.nec").write_text(text)
    bad = 0
    for name in decks:
        for threads in ("1", "auto"):
            env = dict(os.environ)
            for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MOMWIRE_THREADS"):
                env.pop(k, None)
            env["MOMWIRE_TIMES"] = "1"
            if threads != "auto":
                env["MOMWIRE_THREADS"] = threads
            got: dict[str, list[dict[str, float]]] = {label: [] for label, _ in exes}
            for _ in range(REPS):
                for label, exe in exes:
                    out = work / "t.out"
                    t0 = time.perf_counter()
                    r = subprocess.run(
                        [exe, str(work / f"{name}.nec"), str(out)],
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    wall = (time.perf_counter() - t0) * 1e3
                    row = parse(r.stderr)
                    if r.returncode != 0 or not row:
                        print(
                            f"FAILED {label} {name}: rc={r.returncode:#x} {r.stderr[-300:]!r}"
                        )
                        bad = 1
                        continue
                    row["wall"] = wall
                    got[label].append(row)
            print(f"\n{name}, MOMWIRE_THREADS={threads} (ms, medians of {REPS})")
            print("| exe | threads | " + " | ".join(COLS) + " |")
            print("|---|---|" + "---|" * len(COLS))
            for label, rows in got.items():
                if not rows:
                    continue
                med = {c: statistics.median(r[c] for r in rows) for c in COLS}
                print(
                    f"| {label} | {int(rows[0]['threads'])} | "
                    + " | ".join(f"{med[c]:.1f}" for c in COLS)
                    + " |"
                )
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
