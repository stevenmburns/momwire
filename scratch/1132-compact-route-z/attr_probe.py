"""Peak-RSS attribution for the #1029 sector route (momwire#1132 "check first").

A sampler thread reads VmRSS every `--dt` s; at each new high it records the
MAIN thread's Python stack (innermost momwire frames). The final report is the
peak, the stack that owned it, and the stack/RSS at each +`--step` MB rise.

  PYTHONPATH=<momwire src> python attr_probe.py --radials 12 [--dense]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
import warnings


def rss_mb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def hwm_mb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def stack_of(tid):
    fr = sys._current_frames().get(tid)
    if fr is None:
        return []
    out = []
    for fs in traceback.extract_stack(fr):
        if "momwire" in fs.filename or "attr_probe" in fs.filename:
            out.append(f"{fs.filename.rsplit('/', 1)[-1]}:{fs.lineno} {fs.name}")
    return out[-8:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radials", type=int, required=True)
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--step", type=float, default=25.0)
    args = ap.parse_args()

    from antennaknobs.designs.verticals.buried_radial_vertical import Builder
    from antennaknobs.engines.momwire import MomwireEngine

    import momwire
    from momwire import BSplineSolver

    b = Builder()
    b.n_radials = args.radials
    extra = {} if args.dense else {"rotational_symmetry": True}
    eng = MomwireEngine(
        b,
        solver=BSplineSolver,
        solver_kwargs={"degree": 2, **extra},
        ground=("finite", 13.0, 0.005),
    )
    s = eng._make_solver(wavelength=eng._wavelength_for(float(b.freq)))
    main_tid = threading.get_ident()
    base = rss_mb()
    state = {"peak": base, "stack": [], "rises": [], "last": base}
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            r = rss_mb()
            if r > state["peak"]:
                state["peak"] = r
                state["stack"] = stack_of(main_tid)
                if r - state["last"] >= args.step:
                    state["rises"].append((round(r, 1), state["stack"][-3:]))
                    state["last"] = r
            time.sleep(args.dt)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z, _c = s.compute_impedance()
    dt = time.perf_counter() - t0
    stop.set()
    th.join()
    z = complex(__import__("numpy").atleast_1d(z)[0])
    print(
        json.dumps(
            {
                "momwire": momwire.__file__,
                "radials": args.radials,
                "route": not args.dense,
                "z": [z.real, z.imag],
                "rss_before_mb": round(base, 1),
                "sampled_peak_mb": round(state["peak"], 1),
                "vmhwm_mb": round(hwm_mb(), 1),
                "seconds": round(dt, 2),
                "peak_stack": state["stack"],
                "rises": state["rises"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
