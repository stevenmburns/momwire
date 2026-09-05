"""momwire#895 deliverable 3, on the PRODUCTION route rather than the
prototype: `designed_tables` filling through `six_columns` against the same
work done by the C++ batch twin, on the two BLE decks.

probe3 measured the prototype over the whole solve's unique set in ONE
grouping. This measures what ships: the fill's own calls, its own memo and
its own per-call grouping — which carries singleton columns the global
grouping does not (68 of 138 at N = 4, 0.9 % of the points).

Two clocks, because they answer two different questions:

  * KERNEL — each recorded call's new-unique triples replayed through the
    column grouping and through `near_interface_six_batch`. This is
    probe3's like-for-like comparison, at the production grouping.
  * END-TO-END — the wall inside `designed_tables` itself. It includes the
    two `np.nditer` passes over the FULL asked set (21,140 asks for 7,628
    unique triples at N = 4), which both routes pay identically and which
    is now the largest term the column route does not remove.
"""

import contextlib
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import momwire._near_interface as ni  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
import test_ble_1937_838 as ble  # noqa: E402


def solve(build, route, p=None):
    """Solve on `route`, returning Z, the wall spent inside `designed_tables`,
    the value of every triple it answered, and the per-call lists of newly
    unique triples (the exact work the fill hands the kernel)."""
    seen, calls, wall = {}, [], [0.0]
    real = ni.designed_tables
    was_route, was_p = ni._ROUTE, ni._COLUMN_P
    ni._ROUTE = route
    if p is not None:
        ni._COLUMN_P = p

    def timed(eps_t, k2, rho, z, zp, memo=None, **kw):
        before = set(memo) if memo is not None else set()
        t0 = time.perf_counter()
        out = real(eps_t, k2, rho, z, zp, memo=memo, **kw)
        wall[0] += time.perf_counter() - t0
        r, zz, zpz = np.broadcast_arrays(
            np.asarray(rho, float), np.asarray(z, float), np.asarray(zp, float)
        )
        vals = np.stack([out[kk] for kk in ni.KEYS])
        fresh, it = [], np.nditer(r, flags=["multi_index"])
        for _ in it:
            ix = it.multi_index
            key = (float(r[ix]), float(zz[ix]), float(zpz[ix]))
            if key not in seen:
                seen[key] = vals[(slice(None),) + ix]
                if key not in before:
                    fresh.append(key)
        calls.append(fresh)
        timed.eps_t, timed.k2 = eps_t, k2
        return out

    ni.designed_tables = timed
    try:
        z_in, _ = BSplineSolver(**build).compute_impedance()
    finally:
        ni.designed_tables = real
        ni._ROUTE, ni._COLUMN_P = was_route, was_p
    return z_in, wall[0], seen, calls, timed.eps_t, timed.k2


def kernel_column(calls, eps_t, k2, p=None):
    """The replay runs under the same BLAS pin `designed_tables` applies
    around its column loop (momwire#898), so the kernel clock reads what
    production pays; before #898 there was no pin and this is a no-op."""
    pin = getattr(ni, "_blas_physical_cores", contextlib.nullcontext)
    t0 = time.perf_counter()
    ncol = 0
    for fresh in calls:
        # Production's grouping: by rho alone since momwire#899 (z' rides
        # in the exponential like z); (rho, z') before it.
        cols = {}
        for key in fresh:
            cols.setdefault(key[0], []).append(key)
        ncol += len(cols)
        with pin():
            for r, members in cols.items():
                zs = [m[1] for m in members]
                zps = [m[2] for m in members]
                ni.six_columns(eps_t, k2, r, zs, zps, p=p)
    return time.perf_counter() - t0, ncol


def kernel_twin(calls, eps_t, k2):
    k_p = float(k2)
    k_m = ni.k_medium(complex(eps_t), k_p)
    t0 = time.perf_counter()
    for fresh in calls:
        if not fresh:
            continue
        tri = np.asarray(fresh, dtype=float).reshape(-1, 3)
        ni._nia.near_interface_six_batch(
            k_p,
            k_m,
            np.ascontiguousarray(tri[:, 0]),
            np.ascontiguousarray(tri[:, 1]),
            np.ascontiguousarray(tri[:, 2]),
            1e-10,
            ni._LAM_MULT,
            ni._ADAPT_DEPTH,
            ni._DETOUR,
            ni._GX,
            ni._GW,
        )
    return time.perf_counter() - t0


if __name__ == "__main__":
    ps = [int(x) for x in sys.argv[1:]] or [None]
    threads = os.environ.get("OMP_NUM_THREADS", "default")
    for n in (4, 16):
        build = ble.ble_deck(n)
        z_pt, t_pt, pts, calls, eps_t, k2 = solve(build, "point")
        npts = len(pts)
        t_twin = kernel_twin(calls, eps_t, k2)
        print(f"\n== BLE 45 ft, N = {n}, OMP_NUM_THREADS={threads}")
        print(
            f"   point route : kernel {t_twin:.2f} s ({1e3 * t_twin / npts:.3f} ms/pt), "
            f"end-to-end {t_pt:.2f} s over {npts} unique triples   Z = {z_pt:.6f}"
        )
        for p in ps:
            z_c, t_c, cols, _, _, _ = solve(build, "column", p=p)
            t_col, ncol = kernel_column(calls, eps_t, k2, p=p)
            rel = np.array(
                [
                    np.max(np.abs(cols[k] - v)) / max(float(np.max(np.abs(v))), 1e-300)
                    for k, v in pts.items()
                ]
            )
            worst = list(pts)[int(np.argmax(rel))]
            print(
                f"   column p={ni._COLUMN_P if p is None else p}: kernel {t_col:.2f} s "
                f"({1e6 * t_col / npts:.0f} us/pt, {ncol} columns) -> {t_twin / t_col:.1f}x; "
                f"end-to-end {t_c:.2f} s -> {t_pt / t_c:.1f}x"
            )
            print(
                f"      Z = {z_c:.6f}  |dZ| = {abs(z_c - z_pt):.2e} ohm; kernel max rel "
                f"{rel.max():.2e} at (rho, z, z') = ({worst[0]:.4g}, {worst[1]:.4g}, "
                f"{worst[2]:.4g}); 99th pct {np.percentile(rel, 99):.1e}"
            )
