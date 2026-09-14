"""U9 D1 (PLAN.md): does the below/below LOW band reach Z on the two-node deck
at 8 m, soil A? PC5's re-run read L2 and L2p bitwise equal, with every guard but
`lattice_delta_not_bit_zero` holding, so this asks the table directly before
PC5 is read.

  PYTHONPATH=<src tree>/src python d1_low_band_reach.py --out F

Three solves of the same deck: the tables as filled; the low band's values
times (1 + 1e-6); the mid band's values times (1 + 1e-6), the control that the
scaling itself reaches Z. A spy on the below/below projection counts the pairs
whose theta routes to each band.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import momwire
from momwire import _sommerfeld
from momwire import _sommerfeld_below as below
from c_ladder import deck, zmat

SCALE = 1.0 + 1e-6
REAL_FILL = below.SommerfeldGridBelow._fill_region
REAL_PROJ = below.remainder_field_proj_below
STATE = {"scale_idx": (), "calls": 0, "pairs": 0, "lo": 0, "mid": 0, "lo_min_deg": None}


def fill(self, idx):
    was = self._regions[idx]["filled"]
    REAL_FILL(self, idx)
    if not was and idx in STATE["scale_idx"]:
        self._regions[idx]["vals"] = self._regions[idx]["vals"] * SCALE


def proj(obs, t_obs, src, t_src, ground_z, k_p, k_m, grid):
    obs = np.asarray(obs, dtype=float)
    src = np.asarray(src, dtype=float)
    rho = np.hypot(
        obs[:, 0][:, None] - src[:, 0][None, :], obs[:, 1][:, None] - src[:, 1][None, :]
    )
    hh = (ground_z - obs[:, 2])[:, None] + (ground_z - src[:, 2])[None, :]
    th = np.arctan2(hh, rho)
    lo = th < grid.th_band_lo_hi
    STATE["calls"] += 1
    STATE["pairs"] += int(th.size)
    STATE["lo"] += int(np.count_nonzero(lo))
    STATE["mid"] += int(
        np.count_nonzero((th >= grid.th_band_lo_hi) & (th < grid.th_band_hi))
    )
    if lo.any():
        m = math.degrees(float(th[lo].min()))
        STATE["lo_min_deg"] = (
            m if STATE["lo_min_deg"] is None else min(m, STATE["lo_min_deg"])
        )
    return REAL_PROJ(obs, t_obs, src, t_src, ground_z, k_p, k_m, grid)


def solve(scale_band):
    STATE.update(calls=0, pairs=0, lo=0, mid=0, lo_min_deg=None)
    STATE["scale_idx"] = (
        tuple(
            below.region_index(z, scale_band)
            for z in (below._ZONE_INNER, below._ZONE_NEAR, below._ZONE_FAR)
        )
        if scale_band is not None
        else ()
    )
    z, t, lo_dth = zmat(deck(8.0))
    spy = {k: STATE[k] for k in ("calls", "pairs", "lo", "mid", "lo_min_deg")}
    return z, dict(seconds=t, low_band_dth_deg=lo_dth, spy=spy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    below.SommerfeldGridBelow._fill_region = fill
    below.remainder_field_proj_below = proj
    rec = dict(momwire=momwire.__file__, scale=SCALE)
    z0, rec["base"] = solve(None)
    for name, band in (("low", below._BAND_LO), ("mid", below._BAND_MID)):
        z, info = solve(band)
        d = np.abs(z - z0)
        info.update(
            bit_equal=bool(np.array_equal(z, z0)),
            max_abs_dz=float(d.max()),
            max_rel=float((d / np.abs(z0)).max()),
        )
        rec[name] = info
    _sommerfeld._GRID_CACHE.clear()
    args.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
