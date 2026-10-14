"""Capture one razor row window's three kernel calls from a real solve.

    python scripts/core_split/capture_window.py OUT.npz [--nsegs N]
        [--extended-kernel] [--prefix P]

Runs a razor-2p (`nec5_quadrature`) solve of a horizontal 10 m dipole 6 m up
in free space and records the INPUTS of the first
`razor_seg_moments` -> `razor_assemble_t1` -> `razor_t2_rows` chain whose
calls hand each other their outputs directly (M0/M1 into T1, T1 into the
fused T2). The arrays go into OUT.npz under `<prefix>sm_*`, `<prefix>t1_*`
and `<prefix>t2_*`, added to whatever OUT.npz already holds. Outputs are not
stored: their bits belong to the build that computes them.

The window feeds `bench_calls.py`, and at N = 12 it is the committed fixture
of tests/test_razor_core_standalone.py (once plain, once with
`--extended-kernel --prefix ek_`).
"""

import argparse
import os

import numpy as np

SM_NAMES = (
    "obs",
    "seg_p0",
    "seg_t",
    "seg_h",
    "a",
    "xg",
    "wg",
    "k",
    "need_m1",
    "group_i",
    "group_j",
    "a_ek",
    "cancel_flag",
    "b_obs",
)
T1_NAMES = (
    "M0",
    "M1",
    "s_a",
    "s_b",
    "h_a",
    "h_b",
    "fall_a",
    "fall_b",
    "t_out",
    "td_a",
    "td_b",
    "wts",
    "n_path",
)
T2_NAMES = (
    "out",
    "t1",
    "M0w",
    "M0k",
    "plus",
    "minus",
    "s_a",
    "s_b",
    "q_a",
    "q_b",
    "c_a_imag",
    "c_phi_imag",
)


def capture(nsegs, extended_kernel):
    """The first directly chained window's inputs, as a dict of arrays."""
    from momwire import RazorSolver, _accel

    acc = _accel.acc
    seen = {}
    names = ("razor_seg_moments", "razor_assemble_t1", "razor_t2_rows")
    orig = {n: getattr(acc, n) for n in names}

    def sm(*a, **k):
        r = orig["razor_seg_moments"](*a, **k)
        seen.setdefault("sm", (a, k, r))
        return r

    def t1(*a, **k):
        r = orig["razor_assemble_t1"](*a, **k)
        m = seen.get("sm")
        if m is not None and a[0] is m[2][0] and a[1] is m[2][1]:
            seen.setdefault("t1", (a, k, r))
        return r

    def t2(*a, **k):
        if "t1" in seen and a[1] is seen["t1"][2]:
            seen.setdefault("t2", a)
        return orig["razor_t2_rows"](*a, **k)

    acc.razor_seg_moments, acc.razor_assemble_t1, acc.razor_t2_rows = sm, t1, t2
    try:
        RazorSolver(
            wires=[[(0.0, -5.0, 6.0), (0.0, 5.0, 6.0)]],
            nsegs=nsegs,
            wire_radius=0.005,
            wavelength=22.0,
            feed_arclength=5.0,
            nec5_quadrature=True,
            extended_kernel=extended_kernel,
        ).compute_port_solution()
    finally:
        for n, f in orig.items():
            setattr(acc, n, f)
    if "t2" not in seen:
        raise SystemExit("no directly chained seg_moments -> T1 -> fused T2 window")

    out = {}
    a, k, _ = seen["sm"]
    sm_args = dict(zip(SM_NAMES, a), **k)
    for n in SM_NAMES:
        if n == "cancel_flag":
            continue
        v = sm_args.get(n)
        out["sm_" + n] = np.asarray(np.empty(0) if v is None else v)
    a, k, _ = seen["t1"]
    for n, v in zip(T1_NAMES[2:], a[2:]):
        out["t1_" + n] = np.asarray(v)
    out["t1_obs_row"] = np.asarray(k.get("obs_row", np.empty(0, np.int64)), np.int64)
    t2_args = dict(zip(T2_NAMES, seen["t2"]))
    out["t2_out_shape"] = np.asarray(t2_args["out"].shape, np.int64)
    for n in T2_NAMES[2:]:
        out["t2_" + n] = np.asarray(t2_args[n])
    return out


def chain_args(w, prefix=""):
    """The three calls' positional arguments for window `prefix` of `w`.

    T1's first two and T2's first two arguments are outputs of the call
    before; they are left as None for the caller to fill.
    """

    def g(name):
        return w[prefix + name]

    sm = [g("sm_" + n) for n in SM_NAMES if n != "cancel_flag"]
    sm[7] = float(sm[7]) if not np.iscomplexobj(sm[7]) else complex(sm[7])
    sm[8] = bool(sm[8])
    sm.insert(12, 0)  # cancel_flag
    b_obs = sm[13]
    sm[13] = b_obs if b_obs.size else None
    t1 = [None, None] + [g("t1_" + n) for n in T1_NAMES[2:]]
    t1[12] = int(t1[12])
    t2 = [None, None] + [g("t2_" + n) for n in T2_NAMES[2:]]
    t2[10], t2[11] = float(t2[10]), float(t2[11])
    return sm, t1, g("t1_obs_row"), t2


def python_chain(acc, w, prefix=""):
    """(M0, M1, T1, Z rows) from the bound kernels, Z rows column-major."""
    sm, t1, obs_row, t2 = chain_args(w, prefix)
    M0, M1 = acc.razor_seg_moments(*sm)
    t1[0], t1[1] = M0, M1
    T1 = acc.razor_assemble_t1(*t1, obs_row=obs_row)
    rows = np.zeros(tuple(w[prefix + "t2_out_shape"]), np.complex128, order="F")
    t2[0], t2[1] = rows, T1
    acc.razor_t2_rows(*t2)
    return M0, M1, T1, rows


def write_driver_inputs(directory, w, prefix=""):
    """Window `prefix` as razor_chain_driver.cpp reads it: <name>.bin files,
    C order, plus a manifest of `name dtype ndim shape...` lines."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "manifest.txt"), "w") as man:
        for key in sorted(w.files if hasattr(w, "files") else w):
            if not key.startswith(prefix) or key == prefix + "t2_out_shape":
                continue
            name = key[len(prefix) :]
            if prefix == "" and name.split("_", 1)[0] not in ("sm", "t1", "t2"):
                continue
            a = np.asarray(w[key])
            if a.ndim == 0:  # a scalar argument: one float64
                if a.dtype.kind == "c":
                    raise ValueError("the driver chains the real-k kernel only")
                a = a.reshape(1).astype(np.float64)
            a = np.ascontiguousarray(a)
            a.tofile(os.path.join(directory, name + ".bin"))
            man.write(f"{name} {a.dtype.str} {a.ndim} {' '.join(map(str, a.shape))}\n")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out")
    p.add_argument("--nsegs", type=int, default=108)
    p.add_argument("--extended-kernel", action="store_true")
    p.add_argument("--prefix", default="", help="key prefix, to keep several windows")
    args = p.parse_args()
    arrays = dict(np.load(args.out)) if os.path.exists(args.out) else {}
    for key, value in capture(args.nsegs, args.extended_kernel).items():
        arrays[args.prefix + key] = value
    np.savez_compressed(args.out, **arrays)
    pre = args.prefix
    print(
        f"{args.out}: window {pre!r}: M0 {arrays[pre + 'sm_obs'].shape[0]} x "
        f"{arrays[pre + 'sm_seg_h'].size}, Z rows {tuple(arrays[pre + 't2_out_shape'])}, "
        f"EK {'on' if arrays[pre + 'sm_group_i'].size else 'off'}"
    )


if __name__ == "__main__":
    main()
