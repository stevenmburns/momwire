"""How often an executable hangs on a deck, and where (a diagnostic, not a gate).

    python scripts/windows_static/hang_probe.py N GDB DECK LABEL=EXE [LABEL=EXE ...]

Each EXE answers DECK N times, interleaved, 4 at once, at the size policy's
thread count (MOMWIRE_THREADS unset: a small deck takes the serial copy with
OpenBLAS at two threads). A run still going after 60 s is a hang: whether its
printout was already complete is reported (a hang at exit leaves it whole),
GDB attaches for every thread's backtrace (unstripped executables give
names), and the process is killed. Prints the counts per executable.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LIMIT_S = 60


def main(argv: list[str]) -> int:
    n, gdb, deck, *pairs = argv
    exes = [p.split("=", 1) for p in pairs]
    work = Path(tempfile.mkdtemp())
    env = dict(os.environ)
    env.pop("MOMWIRE_THREADS", None)
    env.pop("OPENBLAS_NUM_THREADS", None)
    env.pop("OMP_NUM_THREADS", None)
    jobs = [(i, label, exe) for i in range(int(n)) for label, exe in exes]

    def one(job):
        i, label, exe = job
        out = work / f"{label}-{i}.out"
        p = subprocess.Popen(
            [exe, deck, str(out)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            p.wait(timeout=LIMIT_S)
            return label, None
        except subprocess.TimeoutExpired:
            text = out.read_text(errors="replace") if out.exists() else ""
            whole = "RUN TIME" in text or text.rstrip().endswith("END")
            bt = subprocess.run(
                [gdb, "-batch", "-p", str(p.pid), "-ex", "thread apply all bt 25"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            p.kill()
            return label, (
                f"pid {p.pid}, printout {'complete' if whole else 'incomplete'} "
                f"({len(text)} chars)\n" + bt.stdout[-12000:]
            )

    counts = {label: [0, 0] for label, _ in exes}
    shown = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for label, hang in pool.map(one, jobs):
            counts[label][0] += 1
            if hang:
                counts[label][1] += 1
                if shown < 3:
                    shown += 1
                    print(f"=== HANG {label}: {hang}", flush=True)
    for label, (runs, hangs) in counts.items():
        print(f"{label}: {hangs} hangs in {runs} runs of {Path(deck).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
