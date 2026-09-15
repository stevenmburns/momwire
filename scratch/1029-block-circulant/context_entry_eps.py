"""momwire#1029 phase 0, CONTEXT (not registered): block-circulance per entry.

  PYTHONPATH=<momwire src>:<antennaknobs src> \\
      python scratch/1029-block-circulant/context_entry_eps.py --radials 4 12 --out context_entry_eps.json

`feasibility.py`'s ε divides the worst shift difference by the block's LARGEST
entry, and max |Z| is ~1.2e5 ohm from near-self terms, so a small far-coupling
entry could differ by much more than ε relative to itself and still pass. This
grades every entry against its own magnitude:

  rel = |X[s] - X[s+1]| / max(|X[s]|, |X[s+1]|), over entries above a floor of
  1e-9 x max |X|

for the radial-radial offsets Z[s, s+d], the radial rows against the mast
columns, and the mast rows against the radial columns. It assembles Z the same
way (no solve) and reports the worst entry, its magnitude, and the absolute
worst difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feasibility import build_solver, sector_groups


def per_entry(stack):
    x = np.asarray(stack)
    y = np.roll(x, -1, axis=0)
    diff = np.abs(x - y)
    mag = np.maximum(np.abs(x), np.abs(y))
    floor = 1e-9 * float(np.max(mag))
    keep = mag > floor
    rel = np.zeros_like(diff)
    rel[keep] = diff[keep] / mag[keep]
    i = np.unravel_index(int(np.argmax(rel)), rel.shape)
    return {
        "worst_rel": float(rel[i]),
        "worst_rel_entry_abs": float(mag[i]),
        "worst_abs_diff": float(np.max(diff)),
        "max_abs": float(np.max(mag)),
        "entries_below_floor": int(np.size(keep) - np.count_nonzero(keep)),
        "entries": int(np.size(keep)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radials", type=int, nargs="+", default=[4, 12])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = {}
    for n in args.radials:
        s, _z, _t = build_solver(n, structure_only=True)
        geom = s._build_geometry()
        supp_seg, polys, _kcl, _knots, wbg = s._build_basis_polynomials(geom)
        g = sector_groups(s, geom, wbg)
        m = len(g["sectors"][0])
        order = np.concatenate([*g["sectors"], g["mast"]])
        Z = s._compute_Z_operator(geom, supp_seg, polys)
        P = Z[np.ix_(order, order)]
        nr = n * m
        R = P[:nr, :nr].reshape(n, m, n, m)
        diag_shift = np.stack([R[s_, :, (s_ + np.arange(n)) % n, :] for s_ in range(n)])
        out[n] = {
            "radial_blocks": per_entry(diag_shift),
            "radial_rows_mast_cols": per_entry(P[:nr, nr:].reshape(n, m, -1)),
            "mast_rows_radial_cols": per_entry(
                np.moveaxis(P[nr:, :nr].reshape(-1, n, m), 1, 0)
            ),
        }
        print(n, json.dumps(out[n]), flush=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
