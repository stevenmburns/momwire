"""Capture the first razor row window's three kernel calls from a real
RazorSolver (razor-2p) solve, then (with --compare) check the standalone
driver's outputs against them byte for byte.

    python capture_chain.py <dir> [nsegs]
    python capture_chain.py <dir> --compare
"""

import os
import sys

import numpy as np


def save(d, name, a):
    a = np.ascontiguousarray(a)
    a.tofile(os.path.join(d, name + ".bin"))
    with open(os.path.join(d, "manifest.txt"), "a") as fh:
        fh.write(f"{name} {a.dtype.str} {a.ndim} {' '.join(map(str, a.shape))}\n")


def capture(d, nsegs):
    from momwire import RazorSolver, _accel

    acc = _accel.acc
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "manifest.txt"), "w").close()
    st = {"sm": None, "t1": None, "t2": None}
    o_sm, o_t1, o_t2 = acc.razor_seg_moments, acc.razor_assemble_t1, acc.razor_t2_rows

    def sm(*a, **k):
        r = o_sm(*a, **k)
        if st["sm"] is None:
            st["sm"] = (a, k, r)
        return r

    def t1(*a, **k):
        r = o_t1(*a, **k)
        if st["t1"] is None and st["sm"] is not None:
            st["t1"] = (a, k, r)
        return r

    def t2(*a, **k):
        if st["t2"] is None and st["t1"] is not None and a[1] is not None:
            r = o_t2(*a, **k)
            st["t2"] = (a, k, np.array(a[0], copy=True))
            return r
        return o_t2(*a, **k)

    acc.razor_seg_moments, acc.razor_assemble_t1, acc.razor_t2_rows = sm, t1, t2
    s = RazorSolver(
        wires=[[(0.0, -5.0, 6.0), (0.0, 5.0, 6.0)]],
        nsegs=nsegs,
        wire_radius=0.005,
        wavelength=22.0,
        feed_arclength=5.0,
        nec5_quadrature=True,
        exact_kernel="auto",
    )
    s.compute_port_solution()
    acc.razor_seg_moments, acc.razor_assemble_t1, acc.razor_t2_rows = o_sm, o_t1, o_t2

    (a, k, (M0, M1)) = st["sm"]
    names = [
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
    ]
    args = dict(zip(names, a))
    for n in [
        "obs",
        "seg_p0",
        "seg_t",
        "seg_h",
        "a",
        "xg",
        "wg",
        "group_i",
        "group_j",
        "a_ek",
    ]:
        save(d, "sm_" + n, args[n])
    save(
        d,
        "sm_b_obs",
        np.asarray(args["b_obs"] if args["b_obs"] is not None else np.empty(0)),
    )
    save(d, "sm_k", np.array([float(args["k"])]))
    save(d, "sm_need_m1", np.array([1.0 if args["need_m1"] else 0.0]))
    (a1, k1, T1) = st["t1"]
    assert a1[0] is M0 and a1[1] is M1, (
        "assemble_t1 did not receive seg_moments' output directly"
    )
    n1 = [
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
    ]
    a1d = dict(zip(n1, a1))
    for n in n1[2:12]:
        save(d, "t1_" + n, a1d[n])
    save(d, "t1_n_path", np.array([float(a1d["n_path"])]))
    save(
        d, "t1_obs_row", np.asarray(k1.get("obs_row", np.empty(0, np.int64)), np.int64)
    )
    (a2, k2, rows) = st["t2"]
    assert a2[1] is T1, "t2_rows did not receive assemble_t1's output directly"
    for n, v in zip(["M0w", "M0k", "plus", "minus"], a2[2:6]):
        save(d, "t2_" + n, v)
    save(d, "t2_q_a", a2[8])
    save(d, "t2_q_b", a2[9])
    save(d, "t2_c_a_imag", np.array([a2[10]]))
    save(d, "t2_c_phi_imag", np.array([a2[11]]))
    np.save(os.path.join(d, "ref_M0.npy"), M0)
    np.save(os.path.join(d, "ref_M1.npy"), M1)
    np.save(os.path.join(d, "ref_t1.npy"), T1)
    np.save(os.path.join(d, "ref_rows.npy"), rows)
    print(
        "captured: M0",
        M0.shape,
        "t1",
        T1.shape,
        "rows",
        rows.shape,
        "rows F-order:",
        rows.flags.f_contiguous,
        "obs_row:",
        "obs_row" in k1,
    )


def compare(d):
    out = {}
    for n in ["M0", "M1", "t1", "rows"]:
        ref = np.load(os.path.join(d, f"ref_{n}.npy"))
        if n == "rows":
            drv = np.fromfile(os.path.join(d, "drv_rows_F.bin"), np.complex128)
            drv = drv.reshape(ref.shape, order="F")
        else:
            drv = np.fromfile(os.path.join(d, f"drv_{n}.bin"), np.complex128).reshape(
                ref.shape
            )
        same = ref.tobytes() == np.ascontiguousarray(drv).tobytes()
        out[n] = (ref.shape, same)
        print(n, ref.shape, "BYTE-IDENTICAL" if same else "DIFFERS")
    return all(v[1] for v in out.values())


if __name__ == "__main__":
    if "--compare" in sys.argv:
        sys.exit(0 if compare(sys.argv[1]) else 1)
    capture(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 108)
