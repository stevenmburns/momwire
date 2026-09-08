"""momwire#981 open question: would a SQUARE below-remainder evaluation pay?

Measurement only. Four numbers:
  (a) global ACA rank of the square remainder + the #979 probe residual
  (b) ACA cost rank*2n against the dense 7.2 s and the 3.83 s a per-call ACA saves
  (c) the CALLBACK cost, timed -- the #972 trap is that a Python row/column
      callback runs ~30x a bulk fill per entry, and the square table can never
      be densified (3.9 GB at 48 radials)
  (d) whether the union of the per-call observer sets IS the source cloud,
      which is the premise the restructure rests on

One process per arm; the square table is never materialised.
"""

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/home/smburns/stevenmburns/antennaknobs/scripts")

import bench_converge as bnc  # noqa: E402

import momwire._sommerfeld_below as SB  # noqa: E402
from antennaknobs.engines.momwire import MomwireEngine  # noqa: E402
from momwire import BSplineSolver  # noqa: E402
from momwire._aca import aca_partial  # noqa: E402

CALLS = []
BIG = {}
_orig = SB.remainder_field_proj_below


def _spy(obs, t_obs, src, t_src, ground_z, k_p, k_m, grid):
    CALLS.append((np.array(obs), np.array(t_obs), int(np.shape(src)[0])))
    n = int(np.shape(src)[0])
    if n > BIG.get("n_src", -1):
        BIG.update(
            n_src=n,
            src=np.array(src),
            t_src=np.array(t_src),
            ground_z=float(ground_z),
            k_p=k_p,
            k_m=k_m,
            grid=grid,
        )
    return _orig(obs, t_obs, src, t_src, ground_z, k_p, k_m, grid)


def capture(radials, swept):
    SB.remainder_field_proj_below = _spy
    try:
        cls = bnc.load_design("verticals.buried_radial_vertical")
        b = cls()
        b.n_radials = radials
        kw = {"degree": 2}
        if swept:
            kw["swept_mem_mb"] = swept
        MomwireEngine(
            b,
            solver=BSplineSolver,
            solver_kwargs=kw,
            ground=("finite", 13.0, 0.005),
        ).impedance()
    finally:
        SB.remainder_field_proj_below = _orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radials", type=int, required=True)
    ap.add_argument("--swept-mem-mb", type=int, default=None)
    a = ap.parse_args()

    capture(a.radials, a.swept_mem_mb)
    src, t_src = BIG["src"], BIG["t_src"]
    n = src.shape[0]

    # (d) premise: do the per-call observer sets union to the source cloud?
    obs_all = np.concatenate([c[0] for c in CALLS])

    uniq = np.unique(np.round(obs_all, 12), axis=0)
    uniq_src = np.unique(np.round(src, 12), axis=0)
    premise = {
        "obs_rows_total": int(obs_all.shape[0]),
        "obs_unique": int(uniq.shape[0]),
        "src_unique": int(uniq_src.shape[0]),
        "n_src": n,
        "union_is_src": bool(
            uniq.shape[0] == uniq_src.shape[0]
            and np.allclose(np.sort(uniq, axis=0), np.sort(uniq_src, axis=0))
        ),
    }

    def rowf(i):
        return np.asarray(
            _orig(
                src[i : i + 1],
                t_src[i : i + 1],
                src,
                t_src,
                BIG["ground_z"],
                BIG["k_p"],
                BIG["k_m"],
                BIG["grid"],
            )
        ).ravel()

    def colf(j):
        return np.asarray(
            _orig(
                src,
                t_src,
                src[j : j + 1],
                t_src[j : j + 1],
                BIG["ground_z"],
                BIG["k_p"],
                BIG["k_m"],
                BIG["grid"],
            )
        ).ravel()

    # (c) time the callbacks against the bulk rate
    rowf(0)  # warm
    colf(0)
    t = time.perf_counter()
    for i in range(5):
        rowf(i * max(1, n // 7))
    t_row = (time.perf_counter() - t) / 5
    t = time.perf_counter()
    for j in range(5):
        colf(j * max(1, n // 7))
    t_col = (time.perf_counter() - t) / 5

    # bulk rate from a modest square sub-block
    k = min(400, n)
    idx = np.linspace(0, n - 1, k).astype(np.int64)
    t = time.perf_counter()
    _orig(
        src[idx],
        t_src[idx],
        src[idx],
        t_src[idx],
        BIG["ground_z"],
        BIG["k_p"],
        BIG["k_m"],
        BIG["grid"],
    )
    t_bulk = time.perf_counter() - t
    ns_bulk = t_bulk / (k * k) * 1e9
    ns_cb = (t_row + t_col) / 2 / n * 1e9

    # (a) global ACA on the SQUARE matrix, callbacks only
    t = time.perf_counter()
    cap = min(n, 1200)  # bound the runaway; report if it is hit
    U, V, ur, uc = aca_partial(
        rowf, colf, n, n, tol=1e-6, max_rank=cap, return_pivots=True
    )
    t_aca = time.perf_counter() - t
    from momwire.hmatrix import _sampled_residual

    def blk(rows, cols):
        return np.asarray(
            _orig(
                src[rows],
                t_src[rows],
                src[cols],
                t_src[cols],
                BIG["ground_z"],
                BIG["k_p"],
                BIG["k_m"],
                BIG["grid"],
            )
        ).reshape(len(rows), len(cols))

    resid, _s = _sampled_residual(U, V, ur, uc, n, blk)

    print(
        json.dumps(
            {
                "radials": a.radials,
                "n": n,
                "n_calls": len(CALLS),
                "premise": premise,
                "ns_per_entry_bulk": ns_bulk,
                "ns_per_entry_callback": ns_cb,
                "callback_penalty": ns_cb / ns_bulk if ns_bulk else None,
                "row_s": t_row,
                "col_s": t_col,
                "global_rank": int(U.shape[1]),
                "rank_cap": cap,
                "cap_hit": bool(U.shape[1] >= cap),
                "probe_residual": float(resid),
                "aca_wall_s": t_aca,
                "dense_pairs": n * n,
                "dense_s_at_bulk_rate": n * n * ns_bulk / 1e9,
            }
        )
    )


main()
