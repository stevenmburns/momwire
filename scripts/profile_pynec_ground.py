"""Profile a PyNEC Yagi solve with the Sommerfeld-Norton ground model.

Targets the exact code path the web UI's /ws and /sweep endpoints take
when the user runs in Yagi + PyNEC + ground mode — that's ~100x slower
per solve than free space and we want to know which NEC entry point
dominates the wall time.

Reports:
  1. Median wall time, free-space vs ground (for the same geometry)
  2. cProfile output trimmed to the top consumers for the ground case

Usage (from project root, web package needs to be importable):
    PYTHONPATH=. .venv/bin/python scripts/profile_pynec_ground.py
    PYTHONPATH=. .venv/bin/python scripts/profile_pynec_ground.py --n 50
    PYTHONPATH=. .venv/bin/python scripts/profile_pynec_ground.py --dirs 3

Note: cProfile only sees Python frames — the actual NEC C/C++ code is
opaque, so the profile output collapses to a single `xq_card` entry.
For function-level visibility inside _PyNEC.so use a native profiler
(see scripts/_perf_ground_loop.py for a `perf record` driver).
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

from web.pynec_backend import solve_yagi


def base_req(
    n_per_wire: int, n_directors: int, ground: bool, design_freq_mhz: float = 14.3
) -> dict:
    return {
        "solver": "pynec",
        "geometry": "yagi",
        "ground": ground,
        "height_m": 10.0,
        "design_freq_mhz": design_freq_mhz,
        "measurement_freq_mhz": design_freq_mhz,
        "n_per_wire": n_per_wire,
        "n_directors": n_directors,
        "driver_length_factor": 0.962,
        "reflector_length_factor": 1.01,
        "spacing_wavelengths": 0.15,
        "wire_radius": 0.0005,
        "director_spacing_wavelengths": 0.2,
        "director_size_factor": 0.95,
    }


def time_runs(req: dict, repeats: int = 7) -> list[float]:
    solve_yagi(req)  # warm
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        solve_yagi(req)
        times.append((time.perf_counter() - t0) * 1e3)
    return sorted(times)


def profile_one(req: dict, label: str, top: int = 25) -> None:
    solve_yagi(req)  # warm
    pr = cProfile.Profile()
    pr.enable()
    solve_yagi(req)
    pr.disable()

    times = time_runs(req)
    median = times[len(times) // 2]
    print(
        f"\n=== {label}  median={median:.1f} ms  "
        f"range=[{times[0]:.1f}, {times[-1]:.1f}] ==="
    )

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(top)
    out = s.getvalue()
    keep = out.split("ncalls", 1)
    if len(keep) == 2:
        print("ncalls" + keep[1])
    else:
        print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="segments per wire")
    ap.add_argument("--dirs", type=int, default=0, help="number of directors")
    ap.add_argument(
        "--profile-dump",
        type=str,
        default=None,
        help="path to write the cProfile .prof file (for snakeviz)",
    )
    args = ap.parse_args()

    free_req = base_req(args.n, args.dirs, ground=False)
    ground_req = base_req(args.n, args.dirs, ground=True)

    print(f"Geometry: Yagi  N={args.n}  directors={args.dirs}")
    print("Warming up + timing (no profile overhead)...")
    free_times = time_runs(free_req)
    ground_times = time_runs(ground_req)
    print(
        f"  free space    : median {free_times[3]:6.1f} ms"
        f"  range [{free_times[0]:.1f}, {free_times[-1]:.1f}]"
    )
    print(
        f"  ground (S-N)  : median {ground_times[3]:6.1f} ms"
        f"  range [{ground_times[0]:.1f}, {ground_times[-1]:.1f}]"
        f"  ({ground_times[3] / free_times[3]:.0f}x free space)"
    )

    profile_one(ground_req, f"Ground solve  N={args.n}  dirs={args.dirs}")

    if args.profile_dump:
        # Re-run with profile and dump to .prof for snakeviz/snakeviz-style tools.
        solve_yagi(ground_req)  # warm
        pr = cProfile.Profile()
        pr.enable()
        solve_yagi(ground_req)
        pr.disable()
        pr.dump_stats(args.profile_dump)
        print(
            f"\nWrote profile to {args.profile_dump}"
            f"  (view with: snakeviz {args.profile_dump})"
        )


if __name__ == "__main__":
    main()
