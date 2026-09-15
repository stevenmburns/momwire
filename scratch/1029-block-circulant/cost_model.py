"""momwire#1029 phase 0: a cost model for a block-circulant (sector) route, read
off the fea1ad0 warm profiles of `verticals.buried_radial_vertical` at 48 / 120 /
150 radials. It models; it measures nothing new.

  python scratch/1029-block-circulant/cost_model.py <dir holding warm-N.prof>

Every function's SELF time in the warm cProfile goes into one class:

  pairs      work proportional to the number of basis pairs filled (the
             below-remainder projection, seg-seg moments, the field-Galerkin
             and B-spline assembly, the crossing cross block and its glue). A
             sector fill evaluates sector 0's rows against every sector plus the
             mast's rows against everything: (m + p)(N m + p) of (N m + p)^2
             pairs, i.e. a factor (m + p) / (N m + p).
  solve      the dense LU: (m + p)^3 + (N - 1) m^3 against (N m + p)^3.
  threads    lock waits on the worker pools: bounded between the pair factor
             (they wait on pair work) and unchanged.
  uncertain  generic NumPy kernels (einsum, ufunc reductions, stacking): bounded
             the same way.
  fixed      everything else: unchanged.

m = 54 basis-carrying segments per radial and p = 32 on the mast, from #1067's
segment column (680 at 12 radials, 1,328 at 24; 48 x 54 + 32 = 2,624 and
150 x 54 + 32 = 8,132 reproduce the rest). The route's own added cost (an FFT
over the sector offset and N small solves) is estimated from flop counts.

Predictions are scaled from the profile's warm total onto #1067's momwire warm
column (the two runs differ by a few per cent), and set beside #1067's
3b75639 column.
"""

from __future__ import annotations

import json
import pstats
import sys
from pathlib import Path

M_PER_RADIAL = 54
P_MAST = 32
RUNGS = (48, 120, 150)
LADDER_1067 = {  # radials: (momwire warm s, 3b75639 s), #1067's summary table
    48: (15.28, 2.82),
    120: (84.54, 18.67),
    150: (162.02, 30.14),
}
PAIRS = (
    "remainder_field",
    "seg_seg_full_moments",
    "assemble_field_galerkin",
    "assemble_Z_bsp",
    "_sandwich_dense",
    "_tables",
    "_row_weights",
    "_interp",
    "_real_matvec",
    "_g_of_r",
    "_bnd_and_corner",
    "pair_extents",
    "near_interface",
    "_lagrange4",
    "expm1_neg_jkR",
    "_accumulate_Z_subset",
    "_below_interface",
    "_sommerfeld",
)
SOLVE = ("_batched_linalg._solve", "lu_factor", "getrf", "linalg._basic")
THREADS = ("acquire", "threading.py")
UNCERTAIN = ("einsum", "ufunc", "shape_base.py", "_multiarray_umath", "fromnumeric.py")


def classify(key):
    filename, _line, func = key
    name = f"{filename}:{func}"
    for cls, pats in (
        ("solve", SOLVE),
        ("pairs", PAIRS),
        ("threads", THREADS),
        ("uncertain", UNCERTAIN),
    ):
        if any(p in name for p in pats):
            return cls
    return "fixed"


def main():
    src = Path(sys.argv[1])
    out = {"m_per_radial": M_PER_RADIAL, "p_mast": P_MAST, "rungs": {}}
    for n in RUNGS:
        st = pstats.Stats(str(src / f"warm-{n}.prof"))
        sums = dict.fromkeys(("pairs", "solve", "threads", "uncertain", "fixed"), 0.0)
        top_fixed = []
        for key, (_cc, _nc, tt, _ct, _callers) in st.stats.items():
            cls = classify(key)
            sums[cls] += tt
            if cls == "fixed" and tt >= 0.05:
                top_fixed.append((f"{Path(key[0]).name}:{key[2]}", round(tt, 3)))
        total = st.total_tt
        m, p = M_PER_RADIAL, P_MAST
        nb = n * m + p
        pair_factor = (m + p) / nb
        solve_factor = ((m + p) ** 3 + (n - 1) * m**3) / nb**3
        route_flops = n * m * m * max(1.0, n.bit_length()) * 2 + 2 / 3 * (
            (m + p) ** 3 + (n - 1) * m**3
        )
        route_s = route_flops / 1e9  # ~1 GFlop effective through NumPy, pessimistic
        low = (
            sums["pairs"] * pair_factor
            + sums["solve"] * solve_factor
            + (sums["threads"] + sums["uncertain"]) * pair_factor
            + sums["fixed"]
            + route_s
        )
        high = (
            sums["pairs"] * pair_factor
            + sums["solve"] * solve_factor
            + sums["threads"]
            + sums["uncertain"]
            + sums["fixed"]
            + route_s
        )
        mw_warm, nec5_3b = LADDER_1067[n]
        scale = mw_warm / total
        out["rungs"][n] = {
            "profile_warm_total_s": round(total, 3),
            "class_s": {k: round(v, 3) for k, v in sums.items()},
            "class_pct": {k: round(100 * v / total, 1) for k, v in sums.items()},
            "pair_factor": pair_factor,
            "solve_factor": solve_factor,
            "route_overhead_s": round(route_s, 4),
            "sector_warm_low_s_profile": round(low, 3),
            "sector_warm_high_s_profile": round(high, 3),
            "scale_to_1067": round(scale, 4),
            "sector_warm_low_s": round(low * scale, 2),
            "sector_warm_high_s": round(high * scale, 2),
            "momwire_warm_1067_s": mw_warm,
            "nec5_3b75639_s": nec5_3b,
            "speedup_low": round(mw_warm / (high * scale), 2),
            "speedup_high": round(mw_warm / (low * scale), 2),
            "vs_3b75639_low": round(low * scale / nec5_3b, 3),
            "vs_3b75639_high": round(high * scale / nec5_3b, 3),
            "fixed_terms_ge_50ms": sorted(top_fixed, key=lambda t: -t[1]),
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
