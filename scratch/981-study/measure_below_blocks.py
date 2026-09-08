"""momwire#981 step 2: is the below-remainder projection block-low-rank?

MEASUREMENT ONLY. No kernel, no production change.

#914's 48-radial profile makes `remainder_field_proj_batch_below` the single
largest term (7.2 of 34.8 s) and calls it "ACA-shaped". #973 showed that the
"globally low rank" premise can be false on ground-coupled geometry, so this
answers the question the umbrella actually asks: block by block, on a radial
screen where every radial is close at the hub and far at the tips, does a
distance-admissible partition give small ranks over a large far fraction?

One process per radial count (`--radials`), tracemalloc for the memory ratio
rather than ru_maxrss, and the global table is NEVER densified: at 48 radials
the node cloud is ~15.7k and a dense projection would be ~3.9 GB.
"""

import argparse
import json
import sys
import tracemalloc

import numpy as np

sys.path.insert(0, "/home/smburns/stevenmburns/antennaknobs/scripts")

import bench_converge as bnc  # noqa: E402

import momwire._sommerfeld_below as SB  # noqa: E402
from antennaknobs.engines.momwire import MomwireEngine  # noqa: E402
from momwire import BSplineSolver  # noqa: E402
from momwire._aca import aca_partial, admissible, build_cluster_tree  # noqa: E402

CAP = {}
_orig = SB.remainder_field_proj_below


CALLS = []


def _spy(obs, t_obs, src, t_src, ground_z, k_p, k_m, grid):
    n = int(np.shape(obs)[0]) * int(np.shape(src)[0])
    CALLS.append((int(np.shape(obs)[0]), int(np.shape(src)[0])))
    if n > CAP.get("pairs", -1):
        CAP.update(
            pairs=n,
            obs=np.array(obs),
            t_obs=np.array(t_obs),
            src=np.array(src),
            t_src=np.array(t_src),
            ground_z=float(ground_z),
            k_p=k_p,
            k_m=k_m,
            grid=grid,
        )
    return _orig(obs, t_obs, src, t_src, ground_z, k_p, k_m, grid)


def capture(radials, nseg, swept_mem_mb):
    SB.remainder_field_proj_below = _spy
    try:
        cls = bnc.load_design("verticals.buried_radial_vertical")
        b = cls()
        b.n_radials = radials
        if nseg:
            b.nominal_nsegs = nseg
        eng = MomwireEngine(
            b,
            solver=BSplineSolver,
            solver_kwargs=(
                {"degree": 2, "swept_mem_mb": swept_mem_mb}
                if swept_mem_mb
                else {"degree": 2}
            ),
            ground=("finite", 13.0, 0.005),
        )
        z = eng.impedance()[0]
    finally:
        SB.remainder_field_proj_below = _orig
    return z


def block_stats(tol_list, eta, max_blocks):
    obs, src = CAP["obs"], CAP["src"]
    n_o, n_s = obs.shape[0], src.shape[0]
    lo = np.minimum(obs, obs)
    tree_o = build_cluster_tree(np.arange(n_o), lo, lo, leaf_size=64)
    tree_s = (
        tree_o
        if n_s == n_o and np.array_equal(obs, src)
        else build_cluster_tree(np.arange(n_s), src, src, leaf_size=64)
    )

    far, near = [], []
    stack = [(tree_o, tree_s)]
    while stack:
        s, t = stack.pop()
        if admissible(s, t, eta):
            far.append((s.indices, t.indices))
        elif s.is_leaf and t.is_leaf:
            near.append((s.indices, t.indices))
        else:
            ss = [s] if s.is_leaf else [s.left, s.right]
            tt = [t] if t.is_leaf else [t.left, t.right]
            if len(ss) == 1 and len(tt) == 1:
                near.append((s.indices, t.indices))
            else:
                for a in ss:
                    for bq in tt:
                        stack.append((a, bq))

    far_entries = sum(len(i) * len(j) for i, j in far)
    near_entries = sum(len(i) * len(j) for i, j in near)
    total = far_entries + near_entries

    def sub(I, J):
        return np.asarray(
            _orig(
                CAP["obs"][I],
                CAP["t_obs"][I],
                CAP["src"][J],
                CAP["t_src"][J],
                CAP["ground_z"],
                CAP["k_p"],
                CAP["k_m"],
                CAP["grid"],
            )
        ).reshape(len(I), len(J))

    far_sorted = sorted(far, key=lambda ij: -len(ij[0]) * len(ij[1]))
    ranks = []
    for I, J in far_sorted[:max_blocks]:
        A = sub(I, J)
        sv = np.linalg.svd(A, compute_uv=False)
        s0 = sv[0] if sv.size else 0.0
        row = {"m": len(I), "n": len(J), "svd": {}}
        for t in tol_list:
            row["svd"][f"{t:.0e}"] = int((sv > t * s0).sum()) if s0 > 0 else 0
        U, V = aca_partial(
            lambda i, I=I, J=J: sub(I[i : i + 1], J).ravel(),
            lambda j, I=I, J=J: sub(I, J[j : j + 1]).ravel(),
            len(I),
            len(J),
            tol=1e-6,
        )
        approx = U @ V if U.shape[1] else np.zeros_like(A)
        row["aca_rank"] = int(U.shape[1])
        row["aca_rel"] = float(np.linalg.norm(A - approx) / np.linalg.norm(A))
        ranks.append(row)

    # The single GLOBAL factorisation of this call, for the #981 comparison:
    # if one ACA over the whole obs x src set is already low rank, a cluster
    # partition buys nothing and the question is moot. Row/column callbacks
    # only -- the table is never densified.
    gU, gV, g_rows, g_cols = aca_partial(
        lambda i: sub(np.arange(n_o)[i : i + 1], np.arange(n_s)).ravel(),
        lambda j: sub(np.arange(n_o), np.arange(n_s)[j : j + 1]).ravel(),
        n_o,
        n_s,
        tol=1e-6,
        return_pivots=True,
    )
    from momwire.hmatrix import _sampled_residual

    g_resid, _scale = _sampled_residual(
        gU,
        gV,
        g_rows,
        g_cols,
        n_o,
        lambda rows, cols: sub(rows, cols),
    )

    return {
        "global_rank": int(gU.shape[1]),
        "global_probe_residual": float(g_resid),
        "n_obs": n_o,
        "n_src": n_s,
        "eta": eta,
        "far_blocks": len(far),
        "near_blocks": len(near),
        "far_entries": far_entries,
        "near_entries": near_entries,
        "far_fraction": far_entries / total if total else 0.0,
        "blocks": ranks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radials", type=int, required=True)
    ap.add_argument("--nseg", type=int, default=None)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--swept-mem-mb", type=int, default=None)
    ap.add_argument("--max-blocks", type=int, default=12)
    a = ap.parse_args()

    tracemalloc.start()
    z = capture(a.radials, a.nseg, a.swept_mem_mb)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    out = {
        "radials": a.radials,
        "z": [z.real, z.imag],
        "solve_peak_mb": peak / 1e6,
        "largest_call_pairs": CAP.get("pairs"),
        "n_calls": len(CALLS),
        "total_pairs": sum(m * n for m, n in CALLS),
    }
    if "obs" in CAP:
        tracemalloc.start()
        out.update(block_stats([1e-4, 1e-6, 1e-8], a.eta, a.max_blocks))
        c2, p2 = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out["analysis_peak_mb"] = p2 / 1e6
    print(json.dumps(out))


main()
