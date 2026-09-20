"""U5 step (d): the S/D map -- where the `a_above` knob is strong and where it
is not (#1140).

    S = |Z(a_above = node member) - Z(a_above = far wire)|
    D = |Z_momwire - Z_engine|
    E = the differential residual of `d_step_rod` (the engine's response to the
        radiator's radius step, which momwire must reproduce)

Reported for every geometry tried, including the failures, because a map of
where the knob is weak is the useful half of a negative result. Step (c)'s rod
is the `a=2.5e-4 h/a=200` row.

Run from anywhere (the script's own directory is on sys.path):
  NEC5_EXE=<nec5cl> python scratch/u5-mixed-radius/d_sd_map.py [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import d_step_rod as rod

# (tag, a_node, NOMINAL node panel / a_node, a_rad/a_node, soil, r_far). The
# printed h/a column is the MEASURED segment touching the node, which runs
# 0.75x the nominal because the fed wire ends up with one segment more than
# the panel count; the measured column is the one to read.
# h/a is the node-adjacent segment length over the node member's radius: the
# fed wire gets 3 segments of h and the rise's node panel 2 of them.
ROWS = [
    ("step-(c) rod", 2.5e-4, 200.0, 1 / 2, "A", 1),
    ("step-(c) rod r_far=4", 2.5e-4, 200.0, 1 / 2, "A", 4),
    ("thin, node mesh 64a", 2.5e-4, 64.0, 1 / 16, "A", 4),
    ("thin, node mesh 16a", 2.5e-4, 16.0, 1 / 16, "A", 4),
    ("thin, node mesh 4a", 2.5e-4, 4.0, 1 / 16, "A", 4),
    ("thin, node mesh 2a", 2.5e-4, 2.0, 1 / 16, "A", 4),
    ("a=1e-3, node mesh 2a", 1.0e-3, 2.0, 1 / 16, "A", 4),
    ("a=2.5e-3, node mesh 2a", 2.5e-3, 2.0, 1 / 16, "A", 4),
    ("a=1e-2, node mesh 8a", 1.0e-2, 8.0, 1 / 16, "A", 4),
    ("a=1e-2, node mesh 4a", 1.0e-2, 4.0, 1 / 16, "A", 4),
    ("a=1e-2, node mesh 2.5a", 1.0e-2, 2.5, 1 / 16, "A", 4),
    ("a=1e-2, node mesh 1.67a", 1.0e-2, 1.67, 1 / 16, "A", 4),
    ("a=1e-2, node mesh 1.2a", 1.0e-2, 1.2, 1 / 16, "A", 4),
    ("a=2.5e-2, node mesh 4a", 2.5e-2, 4.0, 1 / 16, "A", 4),
    ("a=2.5e-2, node mesh 2a", 2.5e-2, 2.0, 1 / 16, "A", 4),
    ("HEADLINE poor 2.5a", 1.0e-2, 2.5, 1 / 16, "poor", 4),
    ("poor 1.67a", 1.0e-2, 1.67, 1 / 16, "poor", 4),
    ("poor 4a", 1.0e-2, 4.0, 1 / 16, "poor", 4),
    ("poor, thin a=2.5e-4", 2.5e-4, 200.0, 1 / 2, "poor", 4),
    ("ratio 1/32", 1.0e-2, 1.67, 1 / 32, "A", 4),
    ("ratio 1/4", 1.0e-2, 1.67, 1 / 4, "A", 4),
    ("ratio 1/2", 1.0e-2, 1.67, 1 / 2, "A", 4),
    ("WA7ARK ratio 1:3.43", 1.0e-2, 1.67, 1 / 3.43, "A", 4),
    ("rich soil (20,.03)", 1.0e-2, 2.5, 1 / 16, "rich", 4),
    ("depth 1.0 m", 1.0e-2, 2.5, 1 / 16, "poor", 4, {"depth": 1.0}),
    ("14.2 MHz", 1.0e-2, 2.5, 1 / 16, "poor", 4, {"freq": 14.2, "design_freq": 14.2}),
    ("3.55 MHz", 1.0e-2, 2.5, 1 / 16, "poor", 4, {"freq": 3.55, "design_freq": 3.55}),
    # The REVERSE spread: the far wire fatter than the node member. The node
    # grading and the radiator's own must then be sized to a_rad, which is
    # exactly what puts h/a at the node high and kills S.
    (
        "far wire 4x FATTER",
        1.0e-2,
        2.5,
        4.0,
        "poor",
        4,
        {"h_node_above": 5 * 4 * 1.0e-2, "rest_h": 0.05},
    ),
    (
        "far wire 16x FATTER",
        1.0e-2,
        2.5,
        16.0,
        "poor",
        4,
        {"h_node_above": 5 * 16 * 1.0e-2, "rest_h": 0.2},
    ),
]
SOILS = {"A": rod.SOIL_A, "poor": rod.SOIL_POOR, "rich": ("finite", 20.0, 0.03)}


def config(a_node, h_over_a, ratio, r_far, **kw):
    h = h_over_a * a_node
    # `kw` OVERRIDES, so a row can move depth or frequency off the default.
    return {
        **dict(
            depth=0.30,
            a_node=a_node,
            a_rad=a_node * ratio,
            # 1e-9 below a_node so the CONTROL is a two-radius deck and takes the
            # same crossing path as the spread deck; see d_step_rod's docstring.
            a_rise=a_node * (1 - 1e-9),
            eps_gap=3 * h,
            z_split=3 * h,
            n_fed=3,
            h_node_below=2 * h,
            r_far=r_far,
        ),
        **kw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    exe = os.environ["NEC5_EXE"]
    meta = dict(
        engine_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest(),
        note="S, D and E in ohm; S/D is the brief's figure of merit, S/E the "
        "differential one",
    )
    print(meta, flush=True)
    print(
        f"{'geometry':<26s} {'a_node':>8s} {'h/a':>6s} {'a_rad/a':>8s} {'soil':>5s} "
        f"{'segs':>5s} {'S':>9s} {'D':>8s} {'S/D':>8s} {'E_node':>8s} "
        f"{'E_far':>8s} {'S/E':>7s}",
        flush=True,
    )
    out = []
    for row in ROWS:
        tag, a_node, hoa, ratio, soil, r_far = row[:6]
        cfg = config(a_node, hoa, ratio, r_far, **(row[6] if len(row) > 6 else {}))
        rec = dict(
            tag=tag, a_node=a_node, h_over_a=hoa, ratio=ratio, soil=soil, r_far=r_far
        )
        try:
            m = rod.measure(cfg, soil=SOILS[soil])
            rec.update(m)
            print(
                f"{tag:<26s} {a_node:8.2e} {m['h_over_a']:6.2f} {ratio:8.4f} "
                f"{soil:>5s} {m['segs']:5d} {m['S']:9.4g} {m['D']:8.4g} "
                f"{m['SD']:8.4g} {m['E_node']:8.4g} {m['E_far']:8.4g} "
                f"{m['SE']:7.4g}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 -- a refused deck is a MAP ROW
            rec["error"] = f"{type(exc).__name__}: {exc}"
            print(
                f"{tag:<26s} {a_node:8.2e} {hoa:6.2f} REFUSED "
                f"{type(exc).__name__}: {str(exc)[:80]}",
                flush=True,
            )
        out.append(rec)
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
