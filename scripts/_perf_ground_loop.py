"""Tight loop of PyNEC ground solves, intended as a `perf record` target.

Picks a representative Yagi+ground config (the workload the web UI runs
when a user is dragging sliders in PyNEC + ground mode) and runs it for
a fixed number of iterations so perf can collect enough samples to show
the hot functions inside _PyNEC.so.

Usage standalone:
    PYTHONPATH=. .venv/bin/python scripts/_perf_ground_loop.py [iters]

Usage under perf (DWARF call-graph for inlined frames inside the .so):
    PYTHONPATH=. perf record -F 999 --call-graph=dwarf \\
        -o /tmp/ground.perf.data \\
        -- .venv/bin/python scripts/_perf_ground_loop.py 200
    perf report -i /tmp/ground.perf.data --no-children -g none

Usage under perf for cache events (L1/LLC/TLB hit rates):
    PYTHONPATH=. perf stat -e cycles,instructions,cache-references,\\
        cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,\\
        LLC-load-misses,dTLB-loads,dTLB-load-misses \\
        -- .venv/bin/python scripts/_perf_ground_loop.py 200
"""

from __future__ import annotations

import sys
import time

from web.pynec_backend import solve_yagi

REQ = {
    "solver": "pynec",
    "geometry": "yagi",
    "ground": True,
    "height_m": 10.0,
    "design_freq_mhz": 14.3,
    "measurement_freq_mhz": 14.3,
    "n_per_wire": 30,
    "n_directors": 0,
    "driver_length_factor": 0.962,
    "reflector_length_factor": 1.01,
    "spacing_wavelengths": 0.15,
    "wire_radius": 0.0005,
    "director_spacing_wavelengths": 0.2,
    "director_size_factor": 0.95,
}


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    # Warm
    solve_yagi(REQ)
    t0 = time.perf_counter()
    for _ in range(iters):
        solve_yagi(REQ)
    dt = time.perf_counter() - t0
    print(f"{iters} solves in {dt:.2f} s  ({dt / iters * 1e3:.1f} ms/solve)")


if __name__ == "__main__":
    main()
