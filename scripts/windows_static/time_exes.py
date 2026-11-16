"""Indicative wall-time table of standalone executables on one machine.

    python scripts/windows_static/time_exes.py OUT.jsonl LABEL=EXE [LABEL=EXE ...]

Each LABEL is `<build>/<basis>`, e.g. `mingw/razor2p`, and every executable
answers every deck here, each at the deck-sized thread count (MOMWIRE_THREADS
unset) and at one thread: one untimed run, then REPS timed runs, the minimum
and median reported. A CI runner is a shared VM, so the table is indicative
only: the paired comparison (same runner, same minute, interleaved) is the
point, not the absolute numbers. Real timing belongs on a quiet Windows box.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPS = 5


def yagi(n_el: int, segs: int) -> str:
    # free space, half-wave-ish elements at 14.1 MHz, driven element 2
    lines = ["CM free-space yagi for timing (windows-static.yml)", "CE"]
    for i in range(n_el):
        half = 5.3 - 0.08 * i
        x = 2.0 * i
        lines.append(
            f"GW {i + 1},{segs},{x:.3f},{-half:.4f},0.,{x:.3f},{half:.4f},0.,.00635"
        )
    lines += [
        "GE 0",
        "GN -1",
        "FR 0,1,0,0,14.1",
        f"EX 0,2,{segs // 2 + 1},0,1.,0.",
        "XQ 0",
        "EN",
    ]
    return "\n".join(lines) + "\n"


def hub(n_rad: int, segs: int) -> str:
    # ab_hub16's shape (bspline_printout_buried) with more radials: buried at
    # 0.15 m in Sommerfeld ground, fed on the first radial's first segment
    lines = ["CM buried radial hub for timing (windows-static.yml)", "CE"]
    for i in range(n_rad):
        a = 2.0 * math.pi * i / n_rad
        lines.append(
            f"GW {i + 1},{segs},0.,0.,-.15,{10 * math.cos(a):.6f},{10 * math.sin(a):.6f},-.15,.001"
        )
    lines += [
        "GE -1,-1",
        "LD 5,0,0,0,3.5e7",
        "FR 0,1,0,0,7.15",
        "GN 2,0,0,0,13.,.005",
        "EX 0,1,1,0,1.,0.",
        "XQ 0",
        "EN",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    out = Path(argv[0])
    exes = [a.split("=", 1) for a in argv[1:]]
    work = Path(tempfile.mkdtemp())
    decks = {
        "fs_yagi_6x41": yagi(6, 41),
        "fs_yagi_6x101": yagi(6, 101),
        "fs_yagi_6x201": yagi(6, 201),
        "buried_hub16x20": (
            ROOT / "tests/fixtures/bspline_printout_buried/ab_hub16.nec"
        ).read_text(),
        "buried_hub32x20": hub(32, 20),
    }
    for name, text in decks.items():
        (work / f"{name}.nec").write_text(text)
    rows = []
    bad = 0
    for name in decks:
        deck = work / f"{name}.nec"
        for threads in ("auto", "1"):
            env = dict(os.environ)
            env.pop("OPENBLAS_NUM_THREADS", None)
            env.pop("OMP_NUM_THREADS", None)
            env.pop("MOMWIRE_THREADS", None)
            if threads != "auto":
                env["MOMWIRE_THREADS"] = threads
            times: dict[str, list[float]] = {label: [] for label, _ in exes}
            # interleaved: rep k runs every executable before rep k+1
            for rep in range(REPS + 1):
                for label, exe in exes:
                    po = work / f"{name}-{label.replace('/', '_')}-{threads}.out"
                    t0 = time.perf_counter()
                    r = subprocess.run(
                        [exe, str(deck), str(po)], env=env, capture_output=True
                    )
                    dt = time.perf_counter() - t0
                    text = po.read_text(errors="replace") if po.exists() else ""
                    if (
                        r.returncode != 0
                        or "NEC ERROR" in text
                        or "not served" in text
                        or not text
                    ):
                        print(
                            f"FAILED {label} {name} t={threads}: rc={r.returncode} {r.stderr[-400:]!r}"
                        )
                        print("\n".join(text.splitlines()[:12]))
                        bad = 1
                        break
                    if rep:
                        times[label].append(dt)
            for label, ts in times.items():
                if not ts:
                    continue
                row = {
                    "deck": name,
                    "threads": threads,
                    "exe": label,
                    "min_s": round(min(ts), 4),
                    "median_s": round(statistics.median(ts), 4),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    # the table: one line per (deck, threads), every executable's median
    labels = [label for label, _ in exes]
    print("\n| deck | threads | " + " | ".join(labels) + " |")
    print("|---|---|" + "---|" * len(labels))
    for name in decks:
        for threads in ("auto", "1"):
            cells = []
            for label in labels:
                m = [
                    r
                    for r in rows
                    if r["deck"] == name
                    and r["threads"] == threads
                    and r["exe"] == label
                ]
                cells.append(f"{m[0]['median_s']:.3f}" if m else "-")
            print(f"| {name} | {threads} | " + " | ".join(cells) + " |")
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
