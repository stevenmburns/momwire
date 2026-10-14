"""Where a WARM razor-2p solve's time goes, phase by phase.

    python scripts/core_split/profile_razor_solve.py OUT.json
        [--n 12 50 108 400 800] [--ground free pec] [--reps R]

One "solve" is what a served deck pays per run: construct a `RazorSolver`
(razor-2p: `nec5_quadrature=True, exact_kernel="auto"`), then
`compute_port_solution()`, then the knot-current readout of its column. The
geometry is the core-split notes' dipole: horizontal, 10 m, 6 m up, radius
5 mm, lambda = 22 m; `pec` adds a perfect ground plane at z = 0. Each
repetition perturbs the wavelength by a few ppm so no cache can answer it.

Phases are INCLUSIVE wall times of the wrapped methods, so the table is
read top-down: `init`, then inside `compute_port_solution` the geometry,
the prepare, the fill (of which `native` is the time inside the razor
kernels), the LU (`_bspline._lu_solve`), and `rest` = everything else.
`readout` is `currents_at_knots`. `cxx_solve` is the C++ path's single call
when `MOMWIRE_RAZOR_CXX=1` routes the solve there (then `fill` and `lu`
are inside it).

Threads are whatever OMP_NUM_THREADS / OPENBLAS_NUM_THREADS say; the script
records them.
"""

import argparse
import json
import os
import statistics
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--n", type=int, nargs="+", default=[12, 50, 108, 400, 800])
    ap.add_argument("--ground", nargs="+", default=["free", "pec"])
    ap.add_argument("--reps", type=int, default=0, help="0 = by size")
    args = ap.parse_args()

    from momwire import RazorSolver, razor
    from momwire._accel import acc

    acc_obj = acc
    rec = {}

    def wrap_method(cls, name, key):
        orig = getattr(cls, name)

        def w(*a, **k):
            t = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                rec[key] = rec.get(key, 0.0) + time.perf_counter() - t

        setattr(cls, name, w)

    wrap_method(RazorSolver, "__init__", "init")
    wrap_method(RazorSolver, "_build_geometry", "geometry")
    wrap_method(RazorSolver, "_assemble_Z_prepare", "prepare")
    wrap_method(RazorSolver, "_assemble_Z_from_prepared", "fill")
    wrap_method(RazorSolver, "_refuse_coincident_segments", "coincident")
    wrap_method(RazorSolver, "_port_columns", "port_columns")
    wrap_method(RazorSolver, "compute_port_solution", "solve")
    wrap_method(RazorSolver, "currents_at_knots", "readout")
    if hasattr(RazorSolver, "_cxx_port_solution"):
        wrap_method(RazorSolver, "_cxx_port_solution", "cxx_solve")

    orig_lu = razor._bspline._lu_solve

    def lu(*a, **k):
        t = time.perf_counter()
        try:
            return orig_lu(*a, **k)
        finally:
            rec["lu"] = rec.get("lu", 0.0) + time.perf_counter() - t

    razor._bspline._lu_solve = lu

    for name in dir(acc_obj):
        if not name.startswith(("razor_", "exact_kernel_point")):
            continue
        f = getattr(acc_obj, name)
        if not callable(f):
            continue

        def mk(f):
            def w(*a, **k):
                t = time.perf_counter()
                try:
                    return f(*a, **k)
                finally:
                    rec["native"] = rec.get("native", 0.0) + time.perf_counter() - t
                    rec["native_calls"] = rec.get("native_calls", 0) + 1

            return w

        setattr(acc_obj, name, mk(f))

    results = []
    for ground in args.ground:
        for n in args.n:
            reps = args.reps or (60 if n <= 108 else (20 if n <= 400 else 8))
            rows = []
            for r in range(reps + 3):
                rec.clear()
                t0 = time.perf_counter()
                s = RazorSolver(
                    wires=[[(0.0, -5.0, 6.0), (0.0, 5.0, 6.0)]],
                    nsegs=n,
                    wire_radius=0.005,
                    wavelength=22.0 * (1.0 + 3e-6 * r),
                    feed_arclength=5.0,
                    nec5_quadrature=True,
                    exact_kernel="auto",
                    ground_z=0.0 if ground == "pec" else None,
                )
                sol = s.compute_port_solution()
                s.currents_at_knots(sol.coeffs[:, 0])
                rec["total"] = time.perf_counter() - t0
                if r >= 3:
                    rows.append(dict(rec))
            keys = sorted({k for row in rows for k in row})
            med = {k: statistics.median(row.get(k, 0.0) for row in rows) for k in keys}
            med["n_basis"] = int(sol.coeffs.shape[0])
            med["ground"] = ground
            med["n"] = n
            med["reps"] = reps
            results.append(med)
            print(
                ground,
                n,
                " ".join(
                    f"{k}={med[k] * 1e3:.3f}ms"
                    for k in (
                        "total",
                        "init",
                        "solve",
                        "geometry",
                        "prepare",
                        "fill",
                        "native",
                        "lu",
                        "cxx_solve",
                        "readout",
                    )
                    if k in med
                ),
                f"calls={med.get('native_calls', 0)}",
                flush=True,
            )
    env = {
        k: os.environ.get(k)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MOMWIRE_RAZOR_CXX")
    }
    with open(args.out, "w") as fh:
        json.dump(
            {"env": env, "results": results, "numpy": np.__version__}, fh, indent=1
        )


if __name__ == "__main__":
    main()
