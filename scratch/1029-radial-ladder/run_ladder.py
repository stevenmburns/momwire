"""Re-time the buried radial-screen ladder: momwire against two NEC-5 builds.

    python run_ladder.py <out.jsonl> [radials ...]

WHY THIS DRIVER EXISTS AT ALL, rather than a loop in the shell. Three things it
adds that #983's harness scripts deliberately do not, and none of them touches
what those scripts measure:

  * **peak RSS.** `ladder_today.py` reports tracemalloc, which counts Python
    allocations and not the BLAS or C++ arenas. The brief wants RSS, so each arm
    runs as its own subprocess and the driver reads
    `getrusage(RUSAGE_CHILDREN).ru_maxrss` across it. The harness's tracemalloc
    figure is kept beside it, because #983's memory column is tracemalloc and
    quoting one as the other would be a different measurement.
  * **the arm order.** momwire -> x13 -> 3b75639 -> x13 -> momwire, twice per N,
    so each build is bracketed by the other within one N and the spread is a
    number rather than an assumption.
  * **RLIMIT_AS**, so a momwire arm that runs away dies cleanly instead of taking
    the box.

The harness scripts themselves are invoked unmodified, at momwire 1ca8725.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
H914 = HERE.parent / "914-study"
AK = Path("/home/smburns/stevenmburns/antennaknobs")
PY_EXE = str(AK / ".venv" / "bin" / "python")
T = Path("/home/smburns/nec5-timing")
AS_LIMIT = 35 * 1024**3  # 35 GB, per the brief's stop rule

PINNED = {
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "PYTHONUTF8": "1",
}


def _arm(cmd, env_extra, limit_as=None):
    """Run one arm; return (json row, wall, peak RSS in MB)."""
    env = dict(os.environ)
    env.update(PINNED)
    env.update(env_extra)

    def _pre():
        if limit_as:
            resource.setrlimit(resource.RLIMIT_AS, (limit_as, limit_as))

    # PER-ARM RSS, via `/usr/bin/time -v`. The first version of this driver read
    # `getrusage(RUSAGE_CHILDREN).ru_maxrss` around each arm, which is a
    # high-water mark over ALL children the driver has ever reaped: it never
    # decreases, so every arm after the largest one reported None and the column
    # was empty for nine rows in ten. `time -v` reports the maximum resident set
    # of one invocation and its own tree, which is the number the brief asks for.
    t0 = time.perf_counter()
    p = subprocess.run(
        ["/usr/bin/time", "-v", *cmd],
        capture_output=True,
        text=True,
        env=env,
        preexec_fn=_pre,
    )
    wall = time.perf_counter() - t0
    rss_mb = None
    for line in (p.stderr or "").splitlines():
        if "Maximum resident set size" in line:
            try:
                rss_mb = float(line.rsplit(":", 1)[1].strip()) / 1024.0
            except ValueError:
                pass
    row = None
    for line in (p.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                pass
    if row is None:
        row = {
            "failed": True,
            "returncode": p.returncode,
            "stderr": (p.stderr or "")[-400:],
        }
    return row, wall, rss_mb


def main(argv):
    out = Path(argv[0])
    radials = [int(x) for x in argv[1:]] or [12, 24, 36, 48, 72, 96, 113, 120, 150]
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            r = json.loads(line)
            done.add((r["radials"], r["arm"], r["round"]))
    log = out.open("a", encoding="utf-8")

    def emit(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(
            json.dumps(
                {
                    k: rec[k]
                    for k in ("radials", "arm", "round", "wall_s", "rss_mb")
                    if k in rec
                }
            ),
            flush=True,
        )

    for n in radials:
        for rnd in (1, 2):
            order = ["momwire", "x13", "3b75639", "x13", "momwire"]
            for i, arm in enumerate(order):
                key = (n, f"{arm}#{i}", rnd)
                if key in done:
                    continue
                if arm == "momwire":
                    cmd = [PY_EXE, str(H914 / "ladder_today.py"), "--radials", str(n)]
                    row, wall, rss = _arm(cmd, {}, limit_as=AS_LIMIT)
                else:
                    cmd = [PY_EXE, str(H914 / "nec5_ladder.py"), "--radials", str(n)]
                    row, wall, rss = _arm(cmd, {"NEC5_EXE": str(T / f"nec5cl-{arm}")})
                emit(
                    {
                        "radials": n,
                        "arm": f"{arm}#{i}",
                        "build": arm,
                        "round": rnd,
                        "wall_s": wall,
                        "rss_mb": rss,
                        **row,
                    }
                )
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
