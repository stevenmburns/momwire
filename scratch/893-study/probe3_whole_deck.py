"""momwire#895: whole-deck projection — every column built and every unique
point evaluated by the fixed column rule, against the C++ batch twin over
the same unique set, both single-threaded. Also the SOLVE-level check: the
column route's values substituted for the twin's must leave the impedance
where it is (a gate on the thing that matters, not on the kernel alone).
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(HERE))

import momwire._near_interface as ni  # noqa: E402
from momwire._sommerfeld_transmitted import k_medium  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402
import test_ble_1937_838 as ble  # noqa: E402
from probe1_census import census  # noqa: E402
from probe2_column_rule import column_rule, column_factors, six_column  # noqa: E402

P = (1, 1, 1)


def column_route(uniq, eps_t, k2):
    """All unique triples by the column rule. Returns dict triple -> (6,)
    and the split of wall into setup vs evaluation."""
    k_p = float(k2)
    k_m = k_medium(complex(eps_t), k_p)
    cols = defaultdict(list)
    for r, z, zp in uniq:
        cols[(r, zp)].append(z)
    out = {}
    t_setup = t_eval = 0.0
    K = []
    for (r, zp), zs in cols.items():
        zs = np.asarray(zs)
        t0 = time.perf_counter()
        lam, w = column_rule(r, k_p, k_m, float(np.min(zs - zp)), *P)
        F, g_p, g_m = column_factors(lam, w, k_p, k_m)
        t1 = time.perf_counter()
        vals = six_column(F, g_p, g_m, zs, zp)
        t2 = time.perf_counter()
        t_setup += t1 - t0
        t_eval += t2 - t1
        K.append(len(lam))
        for z, v in zip(zs, vals):
            out[(r, float(z), zp)] = v
    return out, t_setup, t_eval, len(cols), np.mean(K)


def twin_route(uniq, eps_t, k2):
    k_p = float(k2)
    k_m = k_medium(complex(eps_t), k_p)
    tri = np.asarray(uniq, float)
    t0 = time.perf_counter()
    vals = ni._nia.near_interface_six_batch(
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
    return dict(zip(uniq, vals)), time.perf_counter() - t0


def solve_with(build, table, label):
    """Solve the deck with `designed_tables` answered from `table`."""
    real = ni.designed_tables

    def served(eps_t, k2, rho, z, zp, **kw):
        r, zz, zpz = np.broadcast_arrays(
            np.asarray(rho, float), np.asarray(z, float), np.asarray(zp, float)
        )
        out = np.empty((6,) + r.shape, dtype=np.complex128)
        it = np.nditer(r, flags=["multi_index"])
        for _ in it:
            ix = it.multi_index
            out[(slice(None),) + ix] = table[
                (float(r[ix]), float(zz[ix]), float(zpz[ix]))
            ]
        return dict(zip(ni.KEYS, out))

    ni.designed_tables = served
    try:
        z_in, _ = BSplineSolver(**build).compute_impedance()
    finally:
        ni.designed_tables = real
    print(f"   Z via {label}: {z_in:.6f}")
    return z_in


if __name__ == "__main__":
    for n in (4, 16):
        build = ble.ble_deck(n)
        uniq, eps_t, k2 = census(build, f"BLE 45 ft, N = {n}")
        twin, t_twin = twin_route(uniq, eps_t, k2)
        col, t_setup, t_eval, ncol, kmean = column_route(uniq, eps_t, k2)
        rel = np.array(
            [
                np.max(np.abs(col[k] - twin[k]) / np.maximum(np.abs(twin[k]), 1e-300))
                for k in uniq
            ]
        )
        print(
            f"   twin batch (1 thread): {t_twin:.2f} s = {1e3 * t_twin / len(uniq):.3f} ms/pt"
        )
        print(
            f"   column route: {ncol} columns, mean K {kmean:.0f}; setup {t_setup:.2f} s + eval {t_eval:.2f} s "
            f"= {t_setup + t_eval:.2f} s = {1e6 * (t_setup + t_eval) / len(uniq):.0f} us/pt  "
            f"-> {t_twin / (t_setup + t_eval):.1f}x"
        )
        print(
            f"   kernel agreement: max rel {rel.max():.2e}, 99th pct {np.percentile(rel, 99):.1e}"
        )
        z_t = solve_with(build, twin, "twin table")
        z_c = solve_with(build, col, "column table")
        print(f"   |dZ| = {abs(z_c - z_t):.3e} ohm")
