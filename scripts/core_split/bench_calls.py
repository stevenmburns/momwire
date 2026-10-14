"""Per-call time of razor's bound functions, for a paired base-vs-change run.

    OMP_NUM_THREADS=8 python scripts/core_split/bench_calls.py OUT.json \\
        [--window NAME=WINDOW.npz ...]

"tiny" inputs (4 segments, 2-4 observers) make the binding's own cost
(argument conversion, allocation, validation, GIL release) and the OpenMP
region's wake-up the whole call. Each --window is a real row window written
by capture_window.py (e.g. N = 108 and N = 801); its three chained calls are
timed on their captured inputs. Calls go to the RAW extension functions,
without `_accel`'s cancel wrapper. Each figure is the median of 15 timeit
repeats; run base and change interleaved, several times each, on a quiet box,
and compare medians of the runs.
"""

import argparse
import importlib.util
import json
import os
import timeit

import numpy as np

from momwire import _accel

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = (
    "razor_seg_moments",
    "razor_seg_moments_cplx",
    "razor_assemble_t1",
    "razor_assemble_t1_weighted",
    "razor_t2_rows",
    "razor_q_rows",
    "razor_cplx_brackets",
)


def _raw():
    acc = _accel.acc
    return {n: getattr(getattr(acc, n), "__wrapped__", getattr(acc, n)) for n in NAMES}


def tiny(n_obs=2, n_seg=4, n_qp=4, n_basis=3, n_path=2):
    """Small valid arguments for every razor call."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(n_obs * n_path, 3))
    p0 = rng.normal(size=(n_seg, 3))
    t = rng.normal(size=(n_seg, 3))
    t /= np.linalg.norm(t, axis=1)[:, None]
    h = np.full(n_seg, 0.1)
    a = np.full(n_seg, 0.005)
    xg, wg = np.polynomial.legendre.leggauss(n_qp)
    e, ed = np.empty(0, np.int64), np.empty(0)
    sa = np.arange(n_basis, dtype=np.int64)
    sb = sa + 1
    hb = np.full(n_basis, 0.1)
    fa, fb = np.zeros(n_basis, bool), np.ones(n_basis, bool)
    ones_b, w = np.ones(n_basis), np.ones(n_obs * n_path)
    M0 = (rng.normal(size=(n_obs * n_path, n_seg)) + 1j).astype(np.complex128)
    td = rng.normal(size=(3, n_basis))
    return {
        "razor_seg_moments": (obs, p0, t, h, a, xg, wg, 1.3, True, e, e, ed, 0, None),
        "razor_seg_moments_cplx": (
            *(obs, p0, t, h, a, xg, wg, 1.3 - 0.1j, True, e, e, ed, 0, None),
        ),
        "razor_assemble_t1": (M0, M0, sa, sb, hb, hb, fa, fb, obs, td, td, w, n_path),
        "razor_assemble_t1_weighted": (
            *(M0, M0, sa, sb, hb, hb, fa, fb, ones_b, ones_b, obs, obs, p0, t, w),
            *(n_path, 0, 13 - 0.5j, 0.0, 1 + 0j),
        ),
        "razor_t2_rows": (
            np.zeros((n_obs, n_basis), np.complex128, order="F"),
            np.ones((n_obs, n_basis), np.complex128),
            M0[:n_obs],
            np.empty((0, n_seg), np.complex128),
            np.arange(n_obs, dtype=np.int64),
            np.full(n_obs, -1, np.int64),
            *(sa, sb, ones_b, ones_b, 2.0, 3.0),
        ),
        "razor_q_rows": (
            np.ones((n_obs, n_seg, 2), np.complex128),
            np.zeros(n_obs * n_path, np.int64),
            *(sa, sb, hb, hb, fa, fb, ones_b, ones_b, w, n_path),
        ),
        "razor_cplx_brackets": (np.linspace(0.1, 1, 8), 1.3 - 0.1j),
    }


def window_calls(raw, path):
    """{call: (args, kwargs)} for a captured window's three chained calls."""
    spec = importlib.util.spec_from_file_location(
        "capture_window", os.path.join(HERE, "capture_window.py")
    )
    cap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cap)
    w = np.load(path)
    sm, t1, obs_row, t2 = cap.chain_args(w)
    M0, M1 = raw["razor_seg_moments"](*sm)
    t1[0], t1[1] = M0, M1
    T1 = raw["razor_assemble_t1"](*t1, obs_row=obs_row)
    t2[0] = np.zeros(tuple(w["t2_out_shape"]), np.complex128, order="F")
    t2[1] = T1
    return {
        "razor_seg_moments": (sm, {}),
        "razor_assemble_t1": (t1, {"obs_row": obs_row}),
        "razor_t2_rows": (t2, {}),
    }


def bench(fn, args, kwargs=None):
    kwargs = kwargs or {}

    def call():
        fn(*args, **kwargs)

    number = 1
    while timeit.timeit(call, number=number) < 0.05:
        number *= 2
    per = sorted(t / number for t in timeit.repeat(call, number=number, repeat=15))
    return {
        "median_ns": per[len(per) // 2] * 1e9,
        "min_ns": per[0] * 1e9,
        "number": number,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out")
    p.add_argument("--window", action="append", default=[], metavar="NAME=PATH")
    args = p.parse_args()
    raw = _raw()
    res = {
        "accelerator": _accel.acc.__file__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    for name, a in tiny().items():
        res[f"tiny.{name}"] = bench(raw[name], a)
    for spec in args.window:
        label, path = spec.split("=", 1)
        for name, (a, kw) in window_calls(raw, path).items():
            res[f"{label}.{name}"] = bench(raw[name], a, kw)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{k:44s} {v['median_ns']:14.0f} ns")


if __name__ == "__main__":
    main()
