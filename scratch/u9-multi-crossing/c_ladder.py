"""U9 (c) (PLAN.md Amendment 2): the two-node separation ladder on the
production path, soil A, bspline, after the source change.

  PYTHONPATH=<src tree>/src python c_ladder.py ladder --d 3 5 8 11 --out F
  PYTHONPATH=<src tree>/src python c_ladder.py pc5 --d 8 --finer L2p --out F

The deck is spelling A: two `crossing_deck(1)` pairs `d` m apart, a feed on
each monopole, Z = inv(Y). The single-node print is `crossing_deck(1)` with its
one feed. PC5 compares the branch's lattice with the next finer candidate, and
with the rung's own mesh step (every edge count x 3). Its metric is the max
over the four Z entries on both sides.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import momwire

sys.path.insert(0, str(Path(momwire.__file__).resolve().parents[2] / "tests"))

from momwire import _sommerfeld, bspline
from momwire import _sommerfeld_below as below
from momwire.bspline import BSplineSolver
from test_crossing_serve_524 import crossing_deck, two_node_deck

FEED = 4.3333333333
FINER = {
    "L2p": (0.05 - 4 * 0.05 / 6, 0.05 / 6),
    "L3": (0.05 - 7 * 0.05 / 9, 0.05 / 9),
    "L3f": (0.05 - 14 * 0.05 / 18, 0.05 / 18),
}
FINER_BUDGET = 48000


def deck(d, scale=1):
    b = two_node_deck(d)
    b["feeds"] = [(1, FEED, 1 + 0j), (3, FEED, 1 + 0j)]
    if scale != 1:
        b["n_per_edge_per_wire"] = [
            [n * scale for n in e] for e in b["n_per_edge_per_wire"]
        ]
    return b


def lattice():
    return dict(
        floor_deg=below._SOMM_BELOW_TH_MIN_DEG,
        dth_lo_deg=below._SOMM_BELOW_DTH_BAND_LO_DEG,
        budget=below._MAX_TAIL_PANELS,
    )


def preflight(build):
    s = BSplineSolver(**build)
    geom = s._build_geometry()
    b_idx = np.nonzero(s._below_segments(geom))[0]
    obs_b = s._buried_nodes(geom, b_idx)[0]
    _r1, th = bspline._pair_extents_below(obs_b[:, 0], obs_b[:, 1], 0.0 - obs_b[:, 2])
    return dict(th_min_deg=math.degrees(th), refusal=s.buried_serve_refusal())


def zmat(build):
    _sommerfeld._GRID_CACHE.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        y = np.asarray(BSplineSolver(**build).compute_port_solution().y, dtype=complex)
        dt = time.time() - t0
    grids = [
        g
        for g in _sommerfeld._GRID_CACHE.values()
        if getattr(g, "regime", None) == "below"
    ]
    lo = [
        float(
            np.degrees(
                g._regions[below.region_index(below._ZONE_INNER, below._BAND_LO)]["dth"]
            )
        )
        for g in grids
        if any(g._regions[i]["filled"] for i in g._band_lo_idx)
    ]
    return np.linalg.inv(np.atleast_2d(y)), round(dt, 2), lo


def enc(z):
    return [[float(v.real), float(v.imag)] for v in np.asarray(z).ravel()]


def ladder(ds):
    rec = dict(mode="ladder", momwire=momwire.__file__, lattice=lattice(), rungs=[])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        z1 = complex(
            np.atleast_1d(BSplineSolver(**crossing_deck(1)).compute_impedance()[0])[0]
        )
    rec["single_node"] = dict(z=[z1.real, z1.imag], seconds=round(time.time() - t0, 2))
    for d in ds:
        build = deck(d)
        row = dict(d=d, preflight=preflight(build))
        if row["preflight"]["refusal"] is None:
            z, dt, lo = zmat(build)
            row.update(
                z=enc(z),
                abs_z11_minus_single=float(abs(z[0, 0] - z1)),
                abs_z22_minus_single=float(abs(z[1, 1] - z1)),
                abs_z12=float(abs(z[0, 1])),
                reciprocity=float(abs(z[0, 1] - z[1, 0]) / abs(z[0, 1])),
                seconds=dt,
                low_band_dth_deg=lo,
                maxrss_mb=round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
                ),
            )
        rec["rungs"].append(row)
        print(json.dumps(row), flush=True)
    return rec


def pc5(d, finer):
    rec = dict(
        mode="pc5", momwire=momwire.__file__, d=d, shipped=lattice(), finer=finer
    )
    z_ship, t_ship, lo_ship = zmat(deck(d))
    saved = (
        below._SOMM_BELOW_TH_MIN_DEG,
        below._SOMM_BELOW_DTH_BAND_LO_DEG,
        below._MAX_TAIL_PANELS,
    )
    try:
        floor, dth = FINER[finer]
        below._SOMM_BELOW_TH_MIN_DEG = floor
        below._SOMM_BELOW_DTH_BAND_LO_DEG = dth
        below._MAX_TAIL_PANELS = max(FINER_BUDGET, saved[2])
        z_fine, t_fine, lo_fine = zmat(deck(d))
    finally:
        (
            below._SOMM_BELOW_TH_MIN_DEG,
            below._SOMM_BELOW_DTH_BAND_LO_DEG,
            below._MAX_TAIL_PANELS,
        ) = saved
    z_mesh, t_mesh, lo_mesh = zmat(deck(d, scale=3))
    d_lat = np.abs(z_fine - z_ship)
    d_mesh = np.abs(z_mesh - z_ship)
    rec.update(
        z_shipped=enc(z_ship),
        z_finer=enc(z_fine),
        z_mesh_x3=enc(z_mesh),
        max_abs_dz_lattice=float(d_lat.max()),
        max_abs_dz_mesh=float(d_mesh.max()),
        ratio=float(d_lat.max() / d_mesh.max()),
        per_entry_lattice=[float(v) for v in d_lat.ravel()],
        per_entry_mesh=[float(v) for v in d_mesh.ravel()],
        guards=dict(
            low_band_read_shipped=bool(lo_ship),
            low_band_read_finer=bool(lo_fine),
            finer_dth_applied=bool(lo_fine)
            and all(abs(x - FINER[finer][1]) < 1e-12 for x in lo_fine),
            lattice_delta_not_bit_zero=bool(d_lat.max() > 0.0),
        ),
        low_band_dth_deg=dict(shipped=lo_ship, finer=lo_fine, mesh_x3=lo_mesh),
        seconds=dict(shipped=t_ship, finer=t_fine, mesh_x3=t_mesh),
    )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("ladder", "pc5"))
    ap.add_argument("--d", type=float, nargs="+", required=True)
    ap.add_argument("--finer", choices=sorted(FINER))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rec = ladder(args.d) if args.what == "ladder" else pc5(args.d[0], args.finer)
    args.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec)[:2000])


if __name__ == "__main__":
    main()
