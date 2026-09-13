"""U4 Z gate harness: zero-beyond-cap against an extended table, on Z.

Harness-side spellings of the below/below remainder past momwire's
`_SOMM_BELOW_R1_CAP_LAMBDA_M` (no src change):

  shipped   the fill as shipped (a deck past the cap refuses by name)
  extended  the cap raised to `--cap` in-medium wavelengths, so the far
            annulus's existing lattice continues out to the deck's R1
  zeroed    the extended grid, with every below/below pair whose image
            distance R1 exceeds 4 lambda_m zeroed in
            `_sommerfeld_below.remainder_field_proj_below`, so the
            two spellings differ by exactly the beyond-cap remainder

Modes:
  band      G-Z1: the extended grid against direct evaluation over R1 in
            [4, --r1-max] lambda_m and the deck's theta range (bar 2e-4)
  identity  G-Z2/G-Z3: on the in-range catalog buried_radial_vertical,
            shipped, extended and zeroed give Z bit for bit
  ladder    `antennaknobs ladder` in-process under one spelling

Every spelling records what it touched: the grid's resolved r1_max, and the
zeroing wrapper's call and zeroed-pair counts, so a patch the fill never
reaches cannot pass silently.

Run with the antennaknobs venv and this worktree's src first on the path:
  PYTHONPATH=<momwire-wt-u4>/src python scratch/u4-below-range/z_gate.py MODE ...
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import time
from pathlib import Path

import numpy as np

import momwire
from momwire import _ground_refl
from momwire import _sommerfeld_below as below
from momwire._sommerfeld import _SURF_KEYS

C0 = 299792458.0
EPS0 = 8.8541878128e-12
CAP0 = below._SOMM_BELOW_R1_CAP_LAMBDA_M
STATS = {"proj_calls": 0, "pairs": 0, "zeroed_pairs": 0, "grids": []}
_orig_proj = below.remainder_field_proj_below
_orig_get_grid = below.get_grid_below


def apply(spelling, cap):
    """Install a spelling. Returns the original cap for `restore`."""
    if spelling == "shipped":
        below._SOMM_BELOW_R1_CAP_LAMBDA_M = CAP0
        below.remainder_field_proj_below = _orig_proj
    else:
        below._SOMM_BELOW_R1_CAP_LAMBDA_M = float(cap)

    def get_grid(*a, **k):
        g = _orig_get_grid(*a, **k)
        STATS["grids"].append(dict(r1_max=g.r1_max, lam_m=g.lam_m, r1_cap=g.r1_cap))
        return g

    below.get_grid_below = get_grid

    if spelling == "zeroed":

        def proj(obs, t_obs, src, t_src, ground_z, k_p, k_m, grid):
            out = _orig_proj(obs, t_obs, src, t_src, ground_z, k_p, k_m, grid)
            o = np.asarray(obs, dtype=float)
            s = np.asarray(src, dtype=float)
            dx = o[:, 0][:, None] - s[:, 0][None, :]
            dy = o[:, 1][:, None] - s[:, 1][None, :]
            hh = (ground_z - o[:, 2])[:, None] + (ground_z - s[:, 2])[None, :]
            beyond = np.sqrt(dx * dx + dy * dy + hh * hh) > CAP0 * grid.lam_m
            STATS["proj_calls"] += 1
            STATS["pairs"] += int(beyond.size)
            n = int(beyond.sum())
            if n:
                STATS["zeroed_pairs"] += n
                out = np.where(beyond, 0.0, out)
            return out

        below.remainder_field_proj_below = proj
    elif spelling == "extended":
        below.remainder_field_proj_below = _orig_proj


def restore():
    below._SOMM_BELOW_R1_CAP_LAMBDA_M = CAP0
    below.remainder_field_proj_below = _orig_proj
    below.get_grid_below = _orig_get_grid


def mode_band(args):
    """G-Z1: the extended far annulus against direct evaluation."""
    om = 2.0 * math.pi * args.freq_mhz * 1e6
    k2 = om / C0
    eps_t = _ground_refl.eps_tilde((args.eps_r, args.sigma), om, EPS0)
    lam_m = below.lambda_medium(eps_t, k2)
    below._SOMM_BELOW_R1_CAP_LAMBDA_M = float(args.cap)
    try:
        t0 = time.time()
        grid = below.SommerfeldGridBelow(eps_t, k2, args.r1_max * lam_m, omega=om)
        # Sample at fractional offsets so no point can sit on a lattice node
        # along either axis (an on-node sample interpolates exactly and makes
        # the check vacuous; the first steep-band run put every theta on a node).
        span_r = grid.r1_max - CAP0 * lam_m
        r = CAP0 * lam_m + (np.arange(28) + 0.37) * span_r / 28
        span_t = args.th_hi_deg - args.th_lo_deg
        th = np.radians(args.th_lo_deg + (np.arange(25) + 0.61) * span_t / 25)
        rr, tt = np.meshgrid(r, th, indexing="ij")
        grid._ensure_for(float(rr.max()), float(tt.min()), float(tt.max()))
        got_d = grid.eval(rr.ravel(), tt.ravel())
        got = np.stack([np.asarray(got_d[k]).ravel() for k in _SURF_KEYS])
        ref_d = below.iv_surfaces_direct_below(
            eps_t, k2, rr.ravel(), tt.ravel(), rtol=1e-9, omega=om
        )
        ref = np.stack([np.asarray(ref_d[k]).ravel() for k in _SURF_KEYS])
        scale = np.abs(ref).max(axis=0)[None, :]
        err = np.abs(got - ref) / scale
        worst = float(err.max())
        at = np.unravel_index(int(err.max(axis=0).argmax()), rr.shape)
    finally:
        restore()
    out = dict(
        mode="band",
        lam_m=lam_m,
        grid_r1_max_lambda=grid.r1_max / lam_m,
        r1_lambda=[float(r[0] / lam_m), float(r[-1] / lam_m)],
        theta_deg=[args.th_lo_deg, args.th_hi_deg],
        points=int(rr.size),
        worst_rel=worst,
        worst_at=dict(
            r1_lambda=float(rr[at] / lam_m), theta_deg=float(np.degrees(tt[at]))
        ),
        bar=2e-4,
        verdict="HIT" if worst <= 2e-4 else "MISSED",
        seconds=time.time() - t0,
    )
    print(json.dumps(out, indent=1), flush=True)
    return out


def mode_identity(args):
    """G-Z2/G-Z3: an in-range buried design is untouched by both spellings."""
    from antennaknobs.designs.verticals.buried_radial_vertical import Builder
    from antennaknobs.engines.momwire import MomwireEngine

    zs = {}
    for spelling in ("shipped", "extended", "zeroed"):
        STATS.update(proj_calls=0, pairs=0, zeroed_pairs=0, grids=[])
        apply(spelling, args.cap)
        try:
            z = complex(
                MomwireEngine(Builder(), ground=("finite", 13.0, 0.005)).impedance()[0]
            )
        finally:
            restore()
        zs[spelling] = dict(z=repr(z), stats=dict(STATS, grids=list(STATS["grids"])))
        print(spelling, zs[spelling], flush=True)
    same = zs["shipped"]["z"] == zs["extended"]["z"] == zs["zeroed"]["z"]
    plumbed = zs["zeroed"]["stats"]["proj_calls"] > 0
    untouched = zs["zeroed"]["stats"]["zeroed_pairs"] == 0
    out = dict(
        mode="identity",
        zs=zs,
        bit_identical=same,
        zeroing_reached_the_fill=plumbed,
        no_pair_zeroed=untouched,
        verdict="HIT" if (same and plumbed and untouched) else "MISSED",
    )
    print(json.dumps(out, indent=1), flush=True)
    return out


def mode_ladder(args):
    from antennaknobs import cli as entry

    STATS.update(proj_calls=0, pairs=0, zeroed_pairs=0, grids=[])
    apply(args.spelling, args.cap)
    argv = ["antennaknobs", "ladder", "--builder", "@" + str(args.deck)]
    argv += ["--refine", *[str(r) for r in args.refine]]
    argv += ["--engines", args.engine]
    buf = io.StringIO()
    t0 = time.time()
    error = None
    try:
        with contextlib.redirect_stdout(buf):
            try:
                entry(argv[1:])
            except SystemExit as e:
                if e.code not in (0, None):
                    error = f"SystemExit({e.code!r})"
    except Exception as e:  # noqa: BLE001 - a refusal is a result here
        error = f"{type(e).__name__}: {e}"
    finally:
        restore()
    text = buf.getvalue()
    out = dict(
        mode="ladder",
        spelling=args.spelling,
        engine=args.engine,
        cap=args.cap if args.spelling != "shipped" else CAP0,
        deck=str(args.deck),
        refine=list(args.refine),
        stdout=text,
        error=error,
        stats=dict(STATS, grids=list(STATS["grids"])),
        seconds=time.time() - t0,
    )
    print(text, flush=True)
    print(
        json.dumps({k: v for k, v in out.items() if k != "stdout"}, indent=1),
        flush=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("band", "identity", "ladder"))
    ap.add_argument("--cap", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--freq-mhz", type=float, default=3.5)
    ap.add_argument("--eps-r", type=float, default=13.0)
    ap.add_argument("--sigma", type=float, default=0.005)
    ap.add_argument("--r1-max", type=float, default=4.7)
    ap.add_argument("--th-lo-deg", type=float, default=0.1)
    ap.add_argument("--th-hi-deg", type=float, default=2.0)
    ap.add_argument("--spelling", choices=("shipped", "extended", "zeroed"))
    ap.add_argument("--engine", default="momwire")
    ap.add_argument("--deck", type=Path)
    ap.add_argument("--refine", type=int, nargs="+", default=[1])
    args = ap.parse_args()
    meta = dict(momwire_file=momwire.__file__, cap0=CAP0)
    print(meta, flush=True)
    result = {"band": mode_band, "identity": mode_identity, "ladder": mode_ladder}[
        args.mode
    ](args)
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, result=result), indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
