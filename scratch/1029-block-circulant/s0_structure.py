"""momwire#1029 phase 1, gate S-0: the dof set maps onto itself under a 2 pi / N
rotation — structural, before any fill.

  PYTHONPATH=<momwire src>:<antennaknobs src> \\
      python scratch/1029-block-circulant/s0_structure.py --radials 4 12 48 --out s0.jsonl

No Z is assembled and nothing is solved. A dof whose support straddled two
radials would break the block-circulant structure silently, and no numeric gate
would localise that to the hub, so this gate is dof by dof and includes the
hub's junction dofs (momwire#138: those are physics).

What it checks, per PLAN-phase1.md §5:

  S0a  every dof belongs to exactly ONE wire — no shared support
  S0b  the sector map induces a bijection pi on the whole dof set
  S0c  each dof matches its image's kind, local index, end position and
       junction index; the hub's directional dofs map radial -> radial
  S0d  the axial group's dofs are fixed points of pi
  S0e  the KCL rows are invariant under pi (columns permute onto themselves)
  S0f  the geometry rotates onto itself within tol = 1e-9 x the deck's extent
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feasibility import build_solver, polylines_of, sector_groups

TOL_REL = 1e-9


def _head(path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _kept_key(entry):
    """(local index, kind, end position, junction index) for one kept dof."""
    j, kind, junc, end = entry
    return (
        int(j),
        str(kind),
        None if end is None else str(end),
        None if junc is None else int(junc),
    )


def one(n_radials):
    s, _z, _t = build_solver(n_radials, structure_only=True)
    geom = s._build_geometry()
    supp_seg, _polys, kcl_A, _knots, wbg = s._build_basis_polynomials(geom)
    n_b = int(supp_seg.shape[0])
    g = sector_groups(s, geom, wbg)
    polys = polylines_of(s, geom)
    wires = g["wires"]
    n = len(wires)
    rec = {
        "radials": n_radials,
        "n_basis": n_b,
        "n_sectors": n,
        "sector_sizes": sorted({len(x) for x in g["sectors"]}),
        "axis_dofs": int(len(g["mast"])),
        "n_kcl_rows": int(kcl_A.shape[0]),
    }

    # S0a: the dof -> wire map is a partition.
    owner = {}
    shared = []
    for w, (kept, l2g) in enumerate(wbg):
        for i in range(len(kept)):
            dof = int(l2g[i])
            if dof in owner:
                shared.append(dof)
            owner[dof] = w
    rec["S0a"] = (not shared) and len(owner) == n_b
    rec["shared_dofs"] = len(shared)

    # The sector map sigma, by azimuth order, and the dof permutation pi.
    sigma = {wires[k]: wires[(k + 1) % n] for k in range(n)}
    pi = np.arange(n_b, dtype=np.int64)
    kind_ok = True
    hub_ok = True
    for w, w_img in sigma.items():
        kept, l2g = wbg[w]
        kept_img, l2g_img = wbg[w_img]
        if len(kept) != len(kept_img):
            kind_ok = False
            continue
        for i in range(len(kept)):
            pi[int(l2g[i])] = int(l2g_img[i])
            a, b = _kept_key(kept[i]), _kept_key(kept_img[i])
            if a != b:
                kind_ok = False
            # the hub's directional dofs must map radial -> radial on the SAME
            # junction: a shared hub is one junction index for every sector.
            if a[1] == "dir" and a[3] != b[3]:
                hub_ok = False
    rec["S0b"] = sorted(pi.tolist()) == list(range(n_b))
    rec["S0c"] = bool(kind_ok and hub_ok)
    rec["dir_dofs_per_sector"] = sum(
        1 for kept, _l2g in [wbg[wires[0]]] for e in kept if _kept_key(e)[1] == "dir"
    )

    # S0d: the axial dofs are fixed points.
    rec["S0d"] = all(int(pi[d]) == int(d) for d in g["mast"].tolist())

    # S0e: the KCL rows are invariant under pi.
    if kcl_A.shape[0]:
        permuted = kcl_A[:, pi]
        rec["S0e"] = bool(np.array_equal(permuted, kcl_A))
        rec["kcl_max_abs_change"] = float(np.max(np.abs(permuted - kcl_A)))
    else:
        rec["S0e"] = None
        rec["kcl_max_abs_change"] = None

    # S0f: the geometry rotates onto itself.
    extent = max(float(np.max(np.abs(np.asarray(p)))) for p in polys)
    th = 2 * math.pi / n
    rot = np.array(
        [
            [math.cos(th), -math.sin(th), 0.0],
            [math.sin(th), math.cos(th), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    worst = 0.0
    for w, w_img in sigma.items():
        a, b = np.asarray(polys[w]), np.asarray(polys[w_img])
        if a.shape != b.shape:
            worst = float("inf")
            break
        worst = max(worst, float(np.max(np.linalg.norm(a @ rot.T - b, axis=1))))
    rec["rotation_mismatch_m"] = worst
    rec["tol_m"] = TOL_REL * extent
    rec["S0f"] = worst <= TOL_REL * extent
    rec["verdict"] = all(
        rec[k] for k in ("S0a", "S0b", "S0c", "S0d", "S0f") if rec[k] is not None
    ) and (rec["S0e"] is not False)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radials", type=int, nargs="+", default=[4, 12, 48])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    import antennaknobs

    import momwire

    meta = {
        "_meta": True,
        "gate": "S-0",
        "momwire_head": _head(Path(momwire.__file__).parent),
        "antennaknobs_head": _head(Path(antennaknobs.__file__).parent),
    }
    out = open(args.out, "w") if args.out else None  # noqa: SIM115
    if out:
        out.write(json.dumps(meta) + "\n")
    print(json.dumps(meta))
    ok = True
    for n in args.radials:
        rec = one(n)
        ok = ok and bool(rec["verdict"])
        line = json.dumps(rec)
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()
    if out:
        out.close()
    print("S-0:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
