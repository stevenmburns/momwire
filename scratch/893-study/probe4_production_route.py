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

THREE machines since momwire#899 item 1: the numpy column rule, its C++
twin, and the C++ point twin, all replayed over the same recorded calls at
the same grouping. Run it twice — once at the default thread count and once
under OMP_NUM_THREADS=1 — because only the twin scales: the numpy route's
per-column work is scipy and numpy on one core, which is the whole of #898.
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


def grouped(calls):
    """PRODUCTION's grouping of each call's fresh triples, by calling it: a
    replay that re-implements the grouping can mirror one that exists at no
    commit (the #899 study's E2 block did exactly that). (rho, z') before
    #899, rho since.

    Hoisted out of both kernel clocks below, and so is the marshalling each
    machine needs, because the two machines need DIFFERENT amounts of it —
    the twin turns 7,344 key tuples into arrays, the numpy loop builds two
    small lists per column — and a clock that carried each machine's own
    marshalling would not be comparing the kernels. The marshalling is real
    and production pays it; it is in the END-TO-END clock, where it belongs.
    """
    out = []
    for fresh in calls:
        if fresh:
            out.append(ni.group_columns(fresh))
    return out


def kernel_column(groups, eps_t, k2, p=None):
    """The replay runs under the same BLAS pin `designed_tables` applies
    around its column loop (momwire#898), so the kernel clock reads what
    production pays; before #898 there was no pin and this is a no-op."""
    pin = getattr(ni, "_blas_physical_cores", contextlib.nullcontext)
    packs = [
        [(m[0][0], [x[1] for x in m], [x[2] for x in m]) for m in cols.values()]
        for cols in groups
    ]
    t0 = time.perf_counter()
    for pack in packs:
        with pin():
            for r, zs, zps in pack:
                ni.six_columns(eps_t, k2, r, zs, zps, p=p)
    return time.perf_counter() - t0, sum(len(c) for c in groups)


def kernel_column_twin(groups, eps_t, k2, p=None):
    """The same replay through the C++ column twin (momwire#899 item 1).

    It is handed the CONCATENATION of one call's groups, exactly as
    `designed_tables` hands it over, because the twin's parallel unit is the
    column: a call per column would measure OpenMP starting and stopping.
    No BLAS pin — the twin has no gemm to pin.
    """
    k_p = float(k2)
    k_m = ni.k_medium(complex(eps_t), k_p)
    p = ni._COLUMN_P if p is None else p
    packs = []
    for cols in groups:
        members = [m for g in cols.values() for m in g]
        sizes = [len(g) for g in cols.values()]
        offsets = np.zeros(len(sizes) + 1, dtype=np.intp)
        offsets[1:] = np.cumsum(sizes)
        tri = np.asarray(members, dtype=float).reshape(-1, 3)
        packs.append(
            (
                np.asarray(list(cols), dtype=float),
                offsets,
                np.ascontiguousarray(tri[:, 1]),
                np.ascontiguousarray(tri[:, 2]),
            )
        )
    t0 = time.perf_counter()
    for rho_c, offsets, zs, zps in packs:
        ni._nia.near_interface_six_columns(
            k_p,
            k_m,
            rho_c,
            offsets,
            zs,
            zps,
            ni._LAM_MULT,
            int(p),
            ni._DETOUR,
            ni._physical_cpu_count(),
            ni._GX,
            ni._GW,
        )
    return time.perf_counter() - t0, sum(len(c) for c in groups)


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
            # Both column machines. ORDER MATTERS and this one is deliberate:
            # the twin's replay runs FIRST, before anything has warmed an
            # OpenBLAS pool. OpenBLAS idles its threads by SPINNING (~20 ms),
            # so a twin replay placed after a numpy one reads ~20 ms of
            # somebody else's spin — measured, and it is #898's artifact
            # arriving through the instrument rather than the code.
            groups = grouped(calls)
            t_ctw, ncol_tw = kernel_column_twin(groups, eps_t, k2, p=p)
            z_c, t_c, cols, _, _, _ = solve(build, "column", p=p)
            was = ni._FORCE_NUMPY
            ni._FORCE_NUMPY = True  # the switch that turns the twin off
            try:
                z_n, t_n, _, _, _, _ = solve(build, "column", p=p)
                t_col, ncol = kernel_column(groups, eps_t, k2, p=p)
            finally:
                ni._FORCE_NUMPY = was
            assert ncol_tw == ncol, (ncol_tw, ncol)
            rel = np.array(
                [
                    np.max(np.abs(cols[k] - v)) / max(float(np.max(np.abs(v))), 1e-300)
                    for k, v in pts.items()
                ]
            )
            worst = list(pts)[int(np.argmax(rel))]
            print(
                f"   column p={ni._COLUMN_P if p is None else p}, {ncol} columns:"
                f"\n      numpy  : kernel {t_col:.3f} s ({1e6 * t_col / npts:.0f} us/pt)"
                f" -> {t_twin / t_col:.1f}x the point twin; end-to-end {t_n:.2f} s"
                f" -> {t_pt / t_n:.1f}x"
                f"\n      C++ twin: kernel {t_ctw:.3f} s ({1e6 * t_ctw / npts:.0f} us/pt)"
                f" -> {t_twin / t_ctw:.1f}x the point twin, {t_col / t_ctw:.1f}x numpy;"
                f" end-to-end {t_c:.2f} s -> {t_pt / t_c:.1f}x"
            )
            print(
                f"      Z = {z_c:.6f}  |dZ| = {abs(z_c - z_pt):.2e} ohm"
                f"  |dZ| numpy-column {abs(z_c - z_n):.2e} ohm; kernel max rel "
                f"{rel.max():.2e} at (rho, z, z') = ({worst[0]:.4g}, {worst[1]:.4g}, "
                f"{worst[2]:.4g}); 99th pct {np.percentile(rel, 99):.1e}"
            )
