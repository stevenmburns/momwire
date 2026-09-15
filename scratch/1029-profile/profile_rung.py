"""One warm solve of buried_radial_vertical, instrumented, for py-spy to attach to.

    python profile_rung.py --radials N --gate-dir D --rss-log F --out J

Settings are #1067's momwire arm exactly: BSplineSolver degree 2, ground
("finite", 13.0, 0.005), default swept_mem_mb, threads pinned by the caller.

WHY THE PHASES ARE FUNCTIONS, not a py-spy attach. The brief suggests attaching
py-spy by `--pid` after the cold solve so only the warm solve is sampled. That
cannot work on this box: `/proc/sys/kernel/yama/ptrace_scope` is **1**, so py-spy
may only profile a process it is the PARENT of, and `--pid` fails with "Permission
Denied" (sudo wants a password here). Measured, not assumed — the first 48-radial
run produced a 99-byte py-spy log and no profile.

So py-spy runs as the parent over the whole process, and the phases are separated
inside the profile instead: the cold solve happens inside `_phase_cold()` and the
warm one inside `_phase_warm()`. Every sample's stack therefore names its phase,
and slicing the speedscope by frame is exact — no clock alignment between two
tools, which is the part that would have been fragile.

The gate is kept for the case where ptrace_scope is ever 0, but is off by default.

The RSS log's phase column is what lines the memory trace up against the profile's
phases; the CSV is (monotonic_s, rss_mb, phase).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

PHASE = {"name": "startup"}
STOP = threading.Event()


def _rss_mb() -> float:
    with open("/proc/self/status", encoding="ascii") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def _sampler(path: Path, period: float = 0.5):
    t0 = time.monotonic()
    with path.open("w", encoding="ascii") as fh:
        fh.write("monotonic_s,rss_mb,phase\n")
        while not STOP.is_set():
            fh.write(f"{time.monotonic() - t0:.3f},{_rss_mb():.1f},{PHASE['name']}\n")
            fh.flush()
            STOP.wait(period)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radials", type=int, required=True)
    ap.add_argument("--gate-dir", type=Path, required=True)
    ap.add_argument("--rss-log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--cprofile",
        type=Path,
        default=None,
        help="write a cProfile of the WARM solve here (the phase table's instrument)",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="block after the cold solve until <gate-dir>/go appears; only useful where "
        "ptrace_scope is 0 and py-spy can attach by --pid",
    )
    a = ap.parse_args()

    a.gate_dir.mkdir(parents=True, exist_ok=True)
    t = threading.Thread(target=_sampler, args=(a.rss_log,), daemon=True)
    t.start()

    import sys

    sys.path.insert(0, "/home/smburns/stevenmburns/antennaknobs/scripts")
    import bench_converge as bnc

    import momwire.bspline as B
    from antennaknobs.engines.momwire import MomwireEngine
    from momwire import BSplineSolver

    PHASE["name"] = "build"
    cls = bnc.load_design("verticals.buried_radial_vertical")
    b = cls()
    b.n_radials = a.radials
    segs = sum(
        sum(int(c) for c in w[2].counts) if hasattr(w[2], "counts") else int(w[2])
        for w in b.build_wires()
    )
    eng = MomwireEngine(
        b,
        solver=BSplineSolver,
        solver_kwargs={"degree": 2},
        ground=("finite", 13.0, 0.005),
    )

    def _phase_cold(engine):
        return engine.impedance()[0]

    def _phase_warm(engine):
        return engine.impedance()[0]

    PHASE["name"] = "cold"
    t0 = time.perf_counter()
    z_cold = _phase_cold(eng)
    cold = time.perf_counter() - t0

    (a.gate_dir / "cold_done").write_text(str(os.getpid()), encoding="ascii")
    PHASE["name"] = "waiting-for-profiler"
    if a.gate:
        deadline = time.monotonic() + 300
        while not (a.gate_dir / "go").exists() and time.monotonic() < deadline:
            time.sleep(0.05)

    PHASE["name"] = "warm"
    eng._solved_cache = None
    if a.cprofile:
        # cProfile, not a sampler. py-spy is unusable on this workload: measured at
        # 48 radials, `--native --rate 100` ran 29 min for a 28 s solve and reported
        # 810 s behind; even without `--native` the warm solve went 13.3 s -> 75.2 s
        # (5.6x) and py-spy reported 2,947 s behind, because it walks every thread's
        # stack and OpenBLAS is running four. cProfile charges per PYTHON call, and
        # momwire's phases are a handful of Python calls each doing seconds of work
        # in C++ or NumPy, so it is both cheap here and exactly the right
        # attribution for a phase table. Its own overhead is measured, not assumed.
        import cProfile

        pr = cProfile.Profile()
        t0 = time.perf_counter()
        pr.enable()
        z_warm = _phase_warm(eng)
        pr.disable()
        warm = time.perf_counter() - t0
        pr.dump_stats(str(a.cprofile))
    else:
        t0 = time.perf_counter()
        z_warm = _phase_warm(eng)
        warm = time.perf_counter() - t0

    PHASE["name"] = "done"
    STOP.set()
    t.join(timeout=2)
    a.out.write_text(
        json.dumps(
            {
                "radials": a.radials,
                "segments": segs,
                "cold_s": cold,
                "warm_s": warm,
                "z_cold": [z_cold.real, z_cold.imag],
                "z_warm": [z_warm.real, z_warm.imag],
                "gated": bool(a.gate),
                "cprofile": str(a.cprofile) if a.cprofile else None,
                "extents_accel": bool(B._HAVE_PLAN_EXTENTS_ACCEL),
                "galerkin_accel": bool(B._HAVE_FIELD_GALERKIN_ACCEL),
            }
        )
        + "\n",
        encoding="ascii",
    )
    print(a.out.read_text(encoding="ascii"), end="")


main()
