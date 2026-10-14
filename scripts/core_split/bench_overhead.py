"""Per-call time of the razor bound functions on tiny and real inputs.

    python bench_overhead.py <capture_dir> <label> <out.json>

Tiny inputs isolate the binding overhead (argument conversion, allocation,
GIL release, validation); the captured 108-segment window is a real warm call.
Calls the RAW extension functions (no _accel cancel wrapper), median of 15
timeit repeats.
"""

import json
import os
import sys
import timeit

import numpy as np

from momwire import _accel

acc = _accel.acc
raw = {
    n: getattr(getattr(acc, n), "__wrapped__", getattr(acc, n))
    for n in [
        "razor_seg_moments",
        "razor_seg_moments_cplx",
        "razor_assemble_t1",
        "razor_assemble_t1_weighted",
        "razor_t2_rows",
        "razor_q_rows",
        "razor_cplx_brackets",
    ]
}


def load(d):
    A = {}
    for line in open(os.path.join(d, "manifest.txt")):
        name, dt, nd, *shape = line.split()
        A[name] = np.fromfile(os.path.join(d, name + ".bin"), np.dtype(dt)).reshape(
            [int(s) for s in shape]
        )
    return A


def tiny(n_obs=2, n_seg=4, n_qp=4, n_basis=3, n_path=2):
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(n_obs * n_path, 3))
    p0 = rng.normal(size=(n_seg, 3))
    t = rng.normal(size=(n_seg, 3))
    t /= np.linalg.norm(t, axis=1)[:, None]
    h = np.full(n_seg, 0.1)
    a = np.full(n_seg, 0.005)
    xg, wg = np.polynomial.legendre.leggauss(n_qp)
    e = np.empty(0, np.int64)
    ed = np.empty(0)
    sa = np.arange(n_basis, dtype=np.int64)
    sb = sa + 1
    M0 = (rng.normal(size=(n_obs * n_path, n_seg)) + 1j).astype(np.complex128)
    return dict(
        sm=(obs, p0, t, h, a, xg, wg, 1.3, True, e, e, ed, 0, None),
        smc=(obs, p0, t, h, a, xg, wg, 1.3 - 0.1j, True, e, e, ed, 0, None),
        t1=(
            M0,
            M0,
            sa,
            sb,
            np.full(n_basis, 0.1),
            np.full(n_basis, 0.1),
            np.zeros(n_basis, bool),
            np.ones(n_basis, bool),
            obs,
            rng.normal(size=(3, n_basis)),
            rng.normal(size=(3, n_basis)),
            np.ones(n_obs * n_path),
            n_path,
        ),
        t1w=(
            M0,
            M0,
            sa,
            sb,
            np.full(n_basis, 0.1),
            np.full(n_basis, 0.1),
            np.zeros(n_basis, bool),
            np.ones(n_basis, bool),
            np.ones(n_basis),
            np.ones(n_basis),
            obs,
            obs,
            p0,
            t,
            np.ones(n_obs * n_path),
            n_path,
            0,
            13 - 0.5j,
            0.0,
            1 + 0j,
        ),
        t2=(
            np.zeros((n_obs, n_basis), np.complex128, order="F"),
            np.ones((n_obs, n_basis), np.complex128),
            M0[:n_obs],
            np.empty((0, n_seg), np.complex128),
            np.arange(n_obs, dtype=np.int64),
            np.full(n_obs, -1, np.int64),
            sa,
            sb,
            np.ones(n_basis),
            np.ones(n_basis),
            2.0,
            3.0,
        ),
        q=(
            np.ones((n_obs, n_seg, 2), np.complex128),
            np.zeros(n_obs * n_path, np.int64),
            sa,
            sb,
            np.full(n_basis, 0.1),
            np.full(n_basis, 0.1),
            np.zeros(n_basis, bool),
            np.ones(n_basis, bool),
            np.ones(n_basis),
            np.ones(n_basis),
            np.ones(n_obs * n_path),
            n_path,
        ),
        br=(np.linspace(0.1, 1, 8), 1.3 - 0.1j),
    )


def real(A):
    sm = (
        A["sm_obs"],
        A["sm_seg_p0"],
        A["sm_seg_t"],
        A["sm_seg_h"],
        A["sm_a"],
        A["sm_xg"],
        A["sm_wg"],
        float(A["sm_k"][0]),
        bool(A["sm_need_m1"][0]),
        A["sm_group_i"],
        A["sm_group_j"],
        A["sm_a_ek"],
        0,
        None,
    )
    M0, M1 = raw["razor_seg_moments"](*sm)
    n_path = int(A["t1_n_path"][0])
    t1a = (
        M0,
        M1,
        A["t1_s_a"],
        A["t1_s_b"],
        A["t1_h_a"],
        A["t1_h_b"],
        A["t1_fall_a"],
        A["t1_fall_b"],
        A["t1_t_out"],
        A["t1_td_a"],
        A["t1_td_b"],
        A["t1_wts"],
        n_path,
    )
    T1 = raw["razor_assemble_t1"](*t1a, obs_row=A["t1_obs_row"])
    out = np.zeros(T1.shape, np.complex128, order="F")
    t2a = (
        out,
        T1,
        A["t2_M0w"],
        A["t2_M0k"],
        A["t2_plus"],
        A["t2_minus"],
        A["t1_s_a"],
        A["t1_s_b"],
        A["t2_q_a"],
        A["t2_q_b"],
        float(A["t2_c_a_imag"][0]),
        float(A["t2_c_phi_imag"][0]),
    )
    return sm, t1a, A["t1_obs_row"], t2a


def bench(fn, args, kw=None, number=None):
    kw = kw or {}
    f = lambda: fn(*args, **kw)  # noqa: E731
    if number is None:
        number = 1
        while timeit.timeit(f, number=number) < 0.05:
            number *= 2
    ts = timeit.repeat(f, number=number, repeat=15)
    per = sorted(t / number for t in ts)
    return {
        "median_ns": per[len(per) // 2] * 1e9,
        "min_ns": per[0] * 1e9,
        "number": number,
    }


def main():
    cap, label, outp = sys.argv[1], sys.argv[2], sys.argv[3]
    T = tiny()
    res = {"label": label, "file": acc.__file__}
    res["tiny.razor_seg_moments"] = bench(raw["razor_seg_moments"], T["sm"])
    res["tiny.razor_seg_moments_cplx"] = bench(raw["razor_seg_moments_cplx"], T["smc"])
    res["tiny.razor_assemble_t1"] = bench(raw["razor_assemble_t1"], T["t1"])
    res["tiny.razor_assemble_t1_weighted"] = bench(
        raw["razor_assemble_t1_weighted"], T["t1w"]
    )
    res["tiny.razor_t2_rows"] = bench(raw["razor_t2_rows"], T["t2"])
    res["tiny.razor_q_rows"] = bench(raw["razor_q_rows"], T["q"])
    res["tiny.razor_cplx_brackets"] = bench(raw["razor_cplx_brackets"], T["br"])
    sm, t1a, orow, t2a = real(load(cap))
    res["real108.razor_seg_moments"] = bench(raw["razor_seg_moments"], sm)
    res["real108.razor_assemble_t1"] = bench(
        raw["razor_assemble_t1"], t1a, {"obs_row": orow}
    )
    res["real108.razor_t2_rows"] = bench(raw["razor_t2_rows"], t2a)
    json.dump(res, open(outp, "w"), indent=1)
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{label:8s} {k:36s} {v['median_ns']:12.0f} ns")


if __name__ == "__main__":
    main()
