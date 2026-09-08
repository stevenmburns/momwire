"""momwire#914: where the buried-screen ladder stands with levers 1-3 IN.

All three levers the #914 issue lists are already built:
  unit 1  `pair_extents_below`      C++ (bspline.py:713, _accel_mw568.cpp:1339)
  unit 2  `assemble_field_galerkin` C++ (bspline.py:5345, _accel_mw568.cpp:1448)
  #915    the chunked buried route  removed the zero-padded
          `_build_J_blocks_subset` scatter (bspline.py:5892)

This re-measures the ladder so the record says what is LEFT rather than what
the pre-lever profile said. One process per arm, tracemalloc for memory.
"""

import argparse
import json
import sys
import time
import tracemalloc

sys.path.insert(0, "/home/smburns/stevenmburns/antennaknobs/scripts")

import bench_converge as bnc  # noqa: E402

import momwire.bspline as B  # noqa: E402
from antennaknobs.engines.momwire import MomwireEngine  # noqa: E402
from momwire import BSplineSolver  # noqa: E402


def run(radials, swept):
    cls = bnc.load_design("verticals.buried_radial_vertical")
    b = cls()
    b.n_radials = radials
    kw = {"degree": 2}
    if swept:
        kw["swept_mem_mb"] = swept
    tracemalloc.start()
    t = time.perf_counter()
    eng = MomwireEngine(
        b, solver=BSplineSolver, solver_kwargs=kw, ground=("finite", 13.0, 0.005)
    )
    z = eng.impedance()[0]
    cold = time.perf_counter() - t
    eng._solved_cache = None
    t = time.perf_counter()
    z2 = eng.impedance()[0]
    warm = time.perf_counter() - t
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "radials": radials,
        # NOT bnc.total_nominal_segs: it constructs a FRESH builder and so
        # ignores the n_radials override, reporting the 4-radial default.
        "segments": sum(
            sum(int(c) for c in t[2].counts) if hasattr(t[2], "counts") else int(t[2])
            for t in b.build_wires()
        ),
        "z": [z.real, z.imag],
        "z_warm": [z2.real, z2.imag],
        "cold_s": cold,
        "warm_s": warm,
        "peak_mb": peak / 1e6,
        "extents_accel": bool(B._HAVE_PLAN_EXTENTS_ACCEL),
        "galerkin_accel": bool(B._HAVE_FIELD_GALERKIN_ACCEL),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radials", type=int, required=True)
    ap.add_argument("--swept-mem-mb", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(run(a.radials, a.swept_mem_mb)))


main()
