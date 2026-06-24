"""Microbenchmark for the sincos-bound cross-edge quadrature kernel.

The C++ `seg_seg_quad_batch_3d` accelerator (reached via
`_triangular_kernels._seg_seg_offedge_quad_batch`) spends its inner loop in
`std::cos`/`std::sin` over the quadrature-pair phases — it is sincos-bound, so
its wall time is the most direct readout of whether the inner sincos got
vectorized (libmvec on Linux, SLEEF on macOS arm64) versus running scalar.

This times that kernel on a representative two-parallel-edge workload and prints
ns-per-quadrature-pair. It deliberately does NOT compare against a baseline
itself: the macOS iteration CI job builds the extension twice (with and without
SLEEF) and diffs two runs of this script. With `--json` it emits a one-line
machine-readable record for that diffing.

Usage:
    python scripts/bench_sincos.py            # human-readable
    python scripts/bench_sincos.py --json     # one JSON line on stdout
"""

import argparse
import json
import sys
import time

import numpy as np

from momwire import _triangular_kernels as tk


def build_workload(N=64, n_k=4):
    """Two parallel edges (offset 0.5 m in x), N segments each, n_k frequencies.

    The kernel cost is n_k * N * N * n_qp**2 sincos evaluations, so this is sized
    to run for a handful of milliseconds — long enough to time stably, short
    enough for CI.
    """
    y = np.linspace(0.0, 8.0, N + 1)
    edge_i_l = np.column_stack([np.zeros(N), y[:-1], np.zeros(N)])
    edge_i_r = np.column_stack([np.zeros(N), y[1:], np.zeros(N)])
    edge_j_l = np.column_stack([np.full(N, 0.5), y[:-1], np.zeros(N)])
    edge_j_r = np.column_stack([np.full(N, 0.5), y[1:], np.zeros(N)])
    k_array = np.linspace(0.3, 1.1, n_k)
    return edge_i_l, edge_i_r, edge_j_l, edge_j_r, k_array


def bench(reps=7, warmup=2, N=64, n_k=4, n_qp=8, a=0.0005):
    if not tk._HAVE_OFF_ACCEL:
        raise SystemExit(
            "seg_seg_quad_batch_3d accelerator not built — nothing to benchmark"
        )
    el, er, jl, jr, k_array = build_workload(N=N, n_k=n_k)

    def run():
        return tk._seg_seg_offedge_quad_batch(el, er, jl, jr, a, k_array, n_qp)

    for _ in range(warmup):
        run()

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)

    best = min(times)
    pairs = n_k * N * N * n_qp * n_qp  # quadrature pairs => sincos evaluations
    return {
        "best_s": best,
        "median_s": float(np.median(times)),
        "ns_per_pair": best / pairs * 1e9,
        "n_pairs": pairs,
        "config": {"N": N, "n_k": n_k, "n_qp": n_qp, "reps": reps},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit one JSON line")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--n-qp", type=int, default=8)
    args = ap.parse_args(argv)

    result = bench(reps=args.reps, N=args.N, n_qp=args.n_qp)

    if args.json:
        print(json.dumps(result))
    else:
        print(f"seg_seg_quad_batch_3d  ({result['n_pairs']:,} pairs)")
        print(f"  best   : {result['best_s'] * 1e3:.2f} ms")
        print(f"  median : {result['median_s'] * 1e3:.2f} ms")
        print(f"  per-pair: {result['ns_per_pair']:.3f} ns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
