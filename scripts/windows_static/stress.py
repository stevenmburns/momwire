"""Rare-event count: hangs and odd exit codes over the corpus, repeated (a diagnostic).

    python scripts/windows_static/stress.py ROUNDS GDB DIRS LABEL=EXE[|KEY=VAL...] ...

DIRS is a comma-separated list of directories; every `*.nec` under them is
answered ROUNDS times by each LABEL's executable (KEY=VAL pairs after `|`
are added to its environment), interleaved, 4 at once, at the size policy's
thread count (MOMWIRE_THREADS unset; a small deck takes the serial copy with
OpenBLAS at two threads, where run 43965973397's events were). An EVENT is
a run still going after 120 s (a hang: gdb's backtrace of every thread, and
whether the printout was already complete), or an exit status other than the
one that deck's runs return most often (printed in hex with stderr's tail).
Prints the counts per label. Exits 0: it measures, it does not gate.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LIMIT_S = 120


def main(argv: list[str]) -> int:
    rounds, gdb, dirs, *specs = argv
    decks = sorted({p for d in dirs.split(",") for p in Path(d).rglob("*.nec")})
    base = dict(os.environ)
    for k in (
        "MOMWIRE_THREADS",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MW_EXE_COPY",
    ):
        base.pop(k, None)
    labels = {}
    for spec in specs:
        label, rest = spec.split("=", 1)
        exe, *kvs = rest.split("|")
        env = dict(base)
        env.update(kv.split("=", 1) for kv in kvs)
        labels[label] = (exe, env)
    work = Path(tempfile.mkdtemp())
    jobs = [
        (r, i, label)
        for r in range(int(rounds))
        for i in range(len(decks))
        for label in labels
    ]

    def one(job):
        r, i, label = job
        exe, env = labels[label]
        out = work / f"{label}-{r}-{i}.out"
        p = subprocess.Popen(
            [exe, str(decks[i]), str(out)],
            env=env,
            # stdout piped, as fat_compare.py has it: run 44057069874's two
            # exit hangs came from there while this script, then writing
            # stdout to DEVNULL, saw none in 33,552 runs
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            _, err = p.communicate(timeout=LIMIT_S)
            out.unlink(missing_ok=True)
            return label, i, p.returncode, err[-600:].decode(errors="replace"), None
        except subprocess.TimeoutExpired:
            text = out.read_text(errors="replace") if out.exists() else ""
            whole = "RUN TIME" in text
            try:
                bt = subprocess.run(
                    [gdb, "-batch", "-p", str(p.pid), "-ex", "thread apply all bt 30"],
                    capture_output=True,
                    text=True,
                    timeout=180,
                ).stdout
            except Exception as e:  # noqa: BLE001 -- a diagnostic: report and go on
                bt = f"(gdb failed: {e})"
            p.kill()
            p.communicate()
            out.unlink(missing_ok=True)
            note = f"printout {'complete' if whole else 'incomplete'} ({len(text)} chars)\n{bt[-15000:]}"
            return label, i, None, "", note

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, jobs))
    usual: dict[tuple[str, int], int] = {}
    by = {}
    for label, i, rc, _, _ in results:
        if rc is not None:
            by.setdefault((label, i), Counter())[rc] += 1
    for key, c in by.items():
        usual[key] = c.most_common(1)[0][0]
    counts = {label: [0, 0, 0] for label in labels}  # runs, hangs, odd exits
    shown = 0
    for label, i, rc, err, hang in results:
        counts[label][0] += 1
        if hang is not None:
            counts[label][1] += 1
            if shown < 4:
                shown += 1
                print(f"=== HANG {label} {decks[i]}: {hang}", flush=True)
        elif rc != usual[(label, i)]:
            counts[label][2] += 1
            if shown < 8:
                shown += 1
                print(
                    f"=== EXIT {label} {decks[i]}: {rc & 0xFFFFFFFF:#010x} (usual "
                    f"{usual[(label, i)] & 0xFFFFFFFF:#x}); stderr: {err!r}",
                    flush=True,
                )
    for label, (runs, hangs, odd) in counts.items():
        print(
            f"{label}: {runs} runs over {len(decks)} decks x {rounds} rounds: {hangs} hangs, {odd} odd exits"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
