"""momwire#1029 phase 2, gate P2-7: the DEFAULT path before and after.

  # before any phase-2 edit (this tree at momwire main 5bfc731):
  PYTHONPATH=<momwire src>:<ak src> python .../p2_default_path.py --dump p2_pre.npz
  # after:
  PYTHONPATH=<patched src>:<ak src> python .../p2_default_path.py --dump p2_post.npz
  python .../p2_default_path.py --compare p2_pre.npz p2_post.npz --out p2_default_path.json

Phase 1's S-A pinned the default path BIT-identical to main. Phase 2 cannot
promise that for the crossing family: the sparse basis-sample products sum the
same nonzero terms in a different order (momwire#914 and #919 made the same
kind of change to the same fill and gated it "to scale, never to the bit").
So this gate records BOTH readings — the sha of every array, so a bit-identical
result is seen as such, and the scaled entrywise difference, which is the bar:

  max |dZ| / max |Z|      <= 1e-12
  max |dc| / max |c|      <= 1e-12
  |dZ_in| / |Z_in|        <= 1e-12

on the dense fill of the 4- and 12-radial decks (phase 0's, `feasibility.py`),
`rows=None`, through `_below_interface.compute_Z_operator_buried` and the
dense KCL solve, exactly as `s_ab_gates.py --dump` reads them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feasibility import build_solver  # noqa: E402
from s_ab_gates import fill  # noqa: E402

RADIALS = (4, 12)
BAR = 1e-12


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


def dump(path):
    out = {}
    for n in RADIALS:
        solver, _z, _s = build_solver(n, structure_only=True)
        Z, _seen, parts = fill(solver)
        geom, supp_seg, polys, kcl_A, wire_knots, wbg = parts
        v, port_vectors, _vpf, all_voltages, kcl_con = solver._feed_drive_and_readout(
            geom, wire_knots, wbg, supp_seg.shape[0], kcl_A
        )
        coeffs = solver._solve_with_kcl(Z.copy(), v, kcl_con)
        z_in = complex(np.atleast_1d(all_voltages)[0] / (port_vectors[0] @ coeffs))
        out[f"Z{n}"] = Z
        out[f"c{n}"] = coeffs
        out[f"zin{n}"] = np.array([z_in])
        print(f"radials {n}: n_basis {Z.shape[0]} Z_in {z_in:.12f} sha_Z {_sha(Z)}")
    np.savez(path, **out)


def compare(pre, post, out):
    a, b = np.load(pre), np.load(post)
    rows = []
    ok = True
    for n in RADIALS:
        Za, Zb = a[f"Z{n}"], b[f"Z{n}"]
        ca, cb = a[f"c{n}"], b[f"c{n}"]
        za, zb = complex(a[f"zin{n}"][0]), complex(b[f"zin{n}"][0])
        rz = float(np.max(np.abs(Zb - Za)) / np.max(np.abs(Za)))
        rc = float(np.max(np.abs(cb - ca)) / np.max(np.abs(ca)))
        rzin = abs(zb - za) / abs(za)
        rec = {
            "radials": n,
            "n_basis": int(Za.shape[0]),
            "sha_Z_pre": _sha(Za),
            "sha_Z_post": _sha(Zb),
            "Z_bit_identical": bool(np.array_equal(Za, Zb)),
            "coeffs_bit_identical": bool(np.array_equal(ca, cb)),
            "rel_dZ_max": rz,
            "rel_dc_max": rc,
            "rel_dzin": rzin,
            "z_in_pre": [za.real, za.imag],
            "z_in_post": [zb.real, zb.imag],
            "P2_7": bool(rz <= BAR and rc <= BAR and rzin <= BAR),
        }
        ok &= rec["P2_7"]
        rows.append(rec)
        print(json.dumps(rec))
    res = {"gate": "P2-7", "bar": BAR, "pass": bool(ok), "decks": rows}
    if out:
        Path(out).write_text(json.dumps(res, indent=1) + "\n")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.dump:
        dump(args.dump)
        return 0
    if args.compare:
        return compare(*args.compare, args.out)
    ap.error("--dump or --compare")


if __name__ == "__main__":
    raise SystemExit(main())
