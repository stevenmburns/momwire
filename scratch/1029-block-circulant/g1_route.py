"""momwire#1029 phase 1, gate G1: the sector route equals the dense solve.

  PYTHONPATH=<momwire src>:<antennaknobs src> \\
      python scratch/1029-block-circulant/g1_route.py --radials 4 12 48 --out g1.jsonl

The deck, the soil and the basis are phase 0's and #1067's:
`verticals.buried_radial_vertical` with `n_radials` set, soil (13.0, 0.005),
bs2 degree 2. Both solves are the SAME deck built twice — once with
`rotational_symmetry=True` and once without — so nothing but the route
differs, and both answers come back through `compute_impedance`'s own return.

Per `PLAN-phase1.md` §6:

  G1a  |dZ_in| / |Z_in|                              <= 1e-9
  G1b  max |dc| / max |c|                            <= 1e-8
  G1c  the far field, through `element_currents` -> `_far_moments` on one
       fixed grid: max |d|m_theta|| and |d|m_phi|| over the grid's own
       scale                                         <= 1e-8
  G1d  the entry-by-entry non-invariance of the DENSE Z at 48 radials,
       each entry graded against its own magnitude above a 1e-9 x max
       floor (phase 0 measured 1.7e-9 worst at 12)   <= 1e-7

G1a's bar is four orders looser than phase 0's 2e-13 on purpose: phase 0
permuted a dense fill, while the route fills only some rows and sums rotated
copies, so its roundoff path is a different one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import box as box_info  # noqa: E402 — after the sys.path insert above
from context_entry_eps import per_entry
from feasibility import GROUND

# One fixed grid, the upper hemisphere a ground-mounted vertical is reported
# on. 19 x 12 is enough that a route-only defect anywhere in the readout
# shows; the readout itself is not under test here, the coefficients are.
THETA = np.linspace(0.0, 0.5 * math.pi, 19)
PHI = np.linspace(0.0, 2 * math.pi, 13)[:-1]


def solvers(n_radials):
    """The same deck twice: dense, and on the route."""
    from antennaknobs.designs.verticals.buried_radial_vertical import Builder
    from antennaknobs.engines.momwire import MomwireEngine

    from momwire import BSplineSolver

    out = []
    for extra in ({}, {"rotational_symmetry": True}):
        b = Builder()
        b.n_radials = n_radials
        eng = MomwireEngine(
            b,
            solver=BSplineSolver,
            solver_kwargs={"degree": 2, **extra},
            ground=GROUND,
        )
        out.append(eng._make_solver(wavelength=eng._wavelength_for(float(b.freq))))
    return out


def far_moments(solver, coeffs):
    from momwire._far_readout import Ground, _far_moments

    mid, moment, _nodes, _delta = solver.element_currents(coeffs, subdiv=1)
    return _far_moments(
        mid,
        moment,
        solver.k,
        THETA,
        PHI,
        Ground("sommerfeld", float(GROUND[1]), float(GROUND[2])),
        solver.ground_z,
        solver.freq,
    )


def rel(a, b, scale=None):
    """max |a - b| over `scale`, or over the pair's own largest entry."""
    if scale is None:
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / max(scale, 1e-300)


def entry_census(route_solver):
    """G1d: the dense Z's block-circulance entry by entry, in the ROUTE's own
    sector grouping — the same grouping the route fills through, so a defect
    in the grouping shows here rather than being averaged away."""
    s = route_solver
    geom = s._build_geometry()
    supp_seg, polys, _kcl, _knots, wbg = s._build_basis_polynomials(geom)
    sectors, axial = s._rotational_dof_groups(wbg, supp_seg.shape[0])
    n = len(sectors)
    m = int(sectors[0].size)
    order = np.concatenate([*sectors, axial])
    Z = s._compute_Z_operator_buried(geom, supp_seg, polys)  # FULL, no rows=
    P = Z[np.ix_(order, order)]
    del Z
    nr = n * m
    R = P[:nr, :nr].reshape(n, m, n, m)
    shift = np.stack([R[t, :, (t + np.arange(n)) % n, :] for t in range(n)])
    out = {
        "radial_blocks": per_entry(shift),
        "radial_rows_mast_cols": per_entry(P[:nr, nr:].reshape(n, m, -1)),
        "mast_rows_radial_cols": per_entry(
            np.moveaxis(P[nr:, :nr].reshape(-1, n, m), 1, 0)
        ),
    }
    out["worst_rel"] = max(v["worst_rel"] for v in out.values())
    return out


def one(n_radials, do_census):
    dense, route = solvers(n_radials)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        z_d, c_d = dense.compute_impedance()
        t_dense = time.perf_counter() - t0
        t0 = time.perf_counter()
        z_r, c_r = route.compute_impedance()
        t_route = time.perf_counter() - t0
        mt_d, mp_d = far_moments(dense, c_d)
        mt_r, mp_r = far_moments(route, c_r)
    z_d = complex(np.atleast_1d(z_d)[0])
    z_r = complex(np.atleast_1d(z_r)[0])
    smap = route._rotational_map
    pattern = float(np.max(np.abs(mt_d)))
    rec = {
        "radials": n_radials,
        "n_basis": int(c_d.shape[0]),
        "n_sectors": smap.n_sectors,
        "wires_per_sector": int(len(smap.sectors[0])),
        "axial_wires": list(smap.axial),
        "z_dense": [z_d.real, z_d.imag],
        "z_route": [z_r.real, z_r.imag],
        "G1a_z_rel": abs(z_r - z_d) / abs(z_d),
        "G1b_currents_rel": rel(c_r, c_d),
        # Two readings of the registered bar, and the magnitudes that make
        # the difference between them legible. Amendment 3: |m_phi| is
        # identically zero for an N-fold symmetric structure, so "each
        # component over its OWN scale" divides roundoff by roundoff.
        "G1c_m_theta_rel": rel(np.abs(mt_r), np.abs(mt_d)),
        "G1c_m_phi_rel_own": rel(np.abs(mp_r), np.abs(mp_d)),
        "G1c_m_phi_rel": rel(np.abs(mp_r), np.abs(mp_d), scale=pattern),
        "m_theta_max": pattern,
        "m_phi_max_dense": float(np.max(np.abs(mp_d))),
        "m_phi_max_route": float(np.max(np.abs(mp_r))),
        "m_phi_over_m_theta": float(np.max(np.abs(mp_d))) / pattern,
        "axis_sector_copy_spread": route._rotational_copy_spread,
        "t_dense_s": round(t_dense, 3),
        "t_route_s": round(t_route, 3),
        "speedup": round(t_dense / max(t_route, 1e-9), 2),
    }
    rec["G1a"] = rec["G1a_z_rel"] <= 1e-9
    rec["G1b"] = rec["G1b_currents_rel"] <= 1e-8
    rec["G1c"] = max(rec["G1c_m_theta_rel"], rec["G1c_m_phi_rel"]) <= 1e-8
    if do_census:
        rec["G1d_census"] = entry_census(route)
        rec["G1d"] = rec["G1d_census"]["worst_rel"] <= 1e-7
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radials", type=int, nargs="+", default=[4, 12, 48])
    ap.add_argument("--census-at", type=int, default=48)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    import antennaknobs

    import momwire

    meta = {
        "_meta": True,
        "gate": "G1",
        "momwire_head": os.environ.get("MW1029_MOMWIRE_HEAD")
        or box_info.git_head(Path(momwire.__file__).parent),
        "antennaknobs_head": os.environ.get("MW1029_AK_HEAD")
        or box_info.git_head(Path(antennaknobs.__file__).parent),
        "box": box_info.provenance(),
        "accel": box_info.accel_variant(),
        "theta_points": len(THETA),
        "phi_points": len(PHI),
    }
    out = open(args.out, "w") if args.out else None  # noqa: SIM115
    if out:
        out.write(json.dumps(meta) + "\n")
    print(json.dumps(meta))
    ok = True
    for n in args.radials:
        rec = one(n, n == args.census_at)
        ok = ok and all(rec[k] for k in ("G1a", "G1b", "G1c"))
        ok = ok and rec.get("G1d", True)
        line = json.dumps(rec)
        print(line, flush=True)
        if out:
            out.write(line + "\n")
            out.flush()
    if out:
        out.close()
    print("G1:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
