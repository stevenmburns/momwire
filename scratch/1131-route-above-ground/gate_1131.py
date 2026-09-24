"""momwire#1131 gates (a) and (b): the above-ground sector route.

  PYTHONPATH=<tree>/src:scratch/1131-route-above-ground \\
      python gate_1131.py --radials 4 12 --grounds sommerfeld refl-coef pec free \\
          --heights surface elevated --out g.jsonl

Per deck (the momwire-native N6LF screen of `deck.py`):

  (a) route vs dense, both through `compute_impedance`:
        G1a |dZ_in|/|Z_in|        <= 1e-9   (momwire#1029's route bar)
        G1b max|dc| / max|c|      <= 1e-8
  (b) the `rows=` fill against the DENSE CHUNKED fill (the tensor dispatch
      switched off on the instance, so both run the same writers), as uint64:
        square rows=: the requested rows equal, every other row exactly 0;
        compact:      Z[R] equal, R = sector 0 + axial.
      Also recorded, not gated: the dense DEFAULT fill (tensor route where it
      fits) against the chunked one on the same rows -- the roundoff the
      route inherits on a small deck.
  negative controls: sector 1's rows differ from sector 0's bitwise (so the
  bit gate can fail), and dropping one sweep window moves Z_in.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np

from deck import H_SURFACE, deck

warnings.filterwarnings("ignore")

HEIGHTS = {"surface": H_SURFACE, "elevated": 0.5}


def parts(s):
    geom = s._build_geometry()
    supp_seg, polys, _k, _wk, wbg = s._build_basis_polynomials(geom)
    return geom, supp_seg, polys, wbg


def bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def one(n, ground, height, mem=None):
    h = HEIGHTS[height]
    rec = {"radials": n, "ground": ground, "height": height, "swept_mem_mb": mem}
    kw = {} if mem is None else {"swept_mem_mb": mem}
    dense = deck(n, ground=ground, h=h, **kw)
    route = deck(n, ground=ground, h=h, rot=True, **kw)
    t0 = time.perf_counter()
    z_d, c_d = dense.compute_impedance()
    rec["t_dense"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    z_r, c_r = route.compute_impedance()
    rec["t_route"] = time.perf_counter() - t0
    z_d, z_r = complex(np.atleast_1d(z_d)[0]), complex(np.atleast_1d(z_r)[0])
    rec["n_basis"] = int(c_d.shape[0])
    rec["z_dense"] = [z_d.real, z_d.imag]
    rec["z_route"] = [z_r.real, z_r.imag]
    rec["G1a"] = abs(z_r - z_d) / abs(z_d)
    rec["G1b"] = float(np.max(np.abs(c_r - c_d)) / np.max(np.abs(c_d)))

    # (b) the fill, on the route solver (its sector map names the rows)
    from momwire import _rotational_symmetry as RS

    s = route
    geom, supp_seg, polys, wbg = parts(s)
    rows = RS.observer_rows(s, geom)
    sectors, axial = RS.dof_groups(s, wbg, supp_seg.shape[0])
    want = np.union1d(sectors[0], axial)
    Z_default = np.asarray(s._compute_Z_operator(geom, supp_seg, polys))
    s._dense_tensor_fits_budget = lambda n_segs: False
    Z_chunked = np.asarray(s._compute_Z_operator(geom, supp_seg, polys))
    del s._dense_tensor_fits_budget
    Z_sq = s._compute_Z_operator(geom, supp_seg, polys, rows=rows)
    Z_c, R = s._compute_Z_operator(geom, supp_seg, polys, rows=rows, compact=True)
    off = np.ones(Z_sq.shape[0], dtype=bool)
    off[R] = False
    rec["n_rows"] = int(R.size)
    rec["R_is_sector0_axial"] = bool(np.array_equal(R, want))
    rec["square_rows_bitwise"] = bool(np.array_equal(bits(Z_sq[R]), bits(Z_chunked[R])))
    rec["square_other_rows_zero"] = bool(not np.any(Z_sq[off]))
    rec["compact_bitwise"] = bool(np.array_equal(bits(Z_c), bits(Z_chunked[R])))
    rec["compact_differs_entries"] = int(np.sum(bits(Z_c) != bits(Z_chunked[R])))
    scale = float(np.max(np.abs(Z_chunked)))
    rec["default_vs_chunked_rel"] = float(
        np.max(np.abs(Z_default[R] - Z_chunked[R])) / scale
    )
    rec["default_is_chunked"] = bool(np.array_equal(bits(Z_default), bits(Z_chunked)))
    del Z_default, Z_chunked, Z_sq, Z_c

    # negative controls. (1) the bit gate is not vacuous: sector 1's rows are
    # NOT sector 0's, entry for entry, though the route's row sums agree.
    smap = s._rotational_map
    seg_off, per_wire = geom["seg_offsets"], geom["per_wire"]
    s1 = np.sort(
        np.concatenate(
            [
                np.arange(seg_off[w], seg_off[w] + per_wire[w]["n_total"])
                for w in (*smap.sectors[1], *smap.axial)
            ]
        )
    )
    Z1, R1 = s._compute_Z_operator(geom, supp_seg, polys, rows=s1, compact=True)
    Z0, _ = s._compute_Z_operator(geom, supp_seg, polys, rows=rows, compact=True)
    rec["neg_sector1_rows_differ"] = bool(not np.array_equal(bits(Z1), bits(Z0)))
    del Z1, Z0
    # (2) one dropped sweep window (the first requested run of every chunk)
    # moves the route's answer.
    from momwire import bspline

    orig = bspline._ObserverRows.windows
    bspline._ObserverRows.windows = lambda self, i0, i1: orig(self, i0, i1)[1:]
    try:
        z_bad, _ = s.compute_impedance()
    finally:
        bspline._ObserverRows.windows = orig
    z_bad = complex(np.atleast_1d(z_bad)[0])
    rec["neg_dropped_window_moved"] = abs(z_bad - z_d) / abs(z_d)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radials", type=int, nargs="+", default=[4, 12])
    ap.add_argument(
        "--grounds",
        nargs="+",
        default=["sommerfeld", "refl-coef", "pec", "free"],
    )
    ap.add_argument("--heights", nargs="+", default=["surface", "elevated"])
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--swept-mem-mb",
        type=int,
        default=None,
        help="small values force many observer chunks, so bases straddle them",
    )
    a = ap.parse_args()
    for n in a.radials:
        for g in a.grounds:
            for h in a.heights:
                if g == "free" and h == "elevated":
                    continue  # no ground: the height is a translation
                rec = one(n, g, h, a.swept_mem_mb)
                line = json.dumps(rec)
                print(line, flush=True)
                if a.out:
                    with open(a.out, "a") as f:
                        f.write(line + "\n")


if __name__ == "__main__":
    main()
