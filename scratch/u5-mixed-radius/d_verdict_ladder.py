"""U5 step (d): the verdict on `a_above`, on the geometry that can see it.

The map (`d_sd_map.py`) says where the knob is strong. This runs the ladder
there, in three parts, each answering a different objection:

  --ladder   Both candidate spellings against the engine, differentially, over
             a far-mesh ladder. Reports S, D, S/D (the brief's figure) and
             E_node, E_far, S/E (the differential one).

  --sweep    `a_above` swept CONTINUOUSLY, not just the two candidates. This is
             what says the differential can see: |E| has to rise away from its
             minimum, and the minimum has to land somewhere in particular.

  --invariance  The sweep repeated with the RADIATOR's radius moved over 16x.
             If `a_above` had any business depending on the far wire, the
             minimum would track it. This is the test the two-candidate ladder
             cannot do, and it rules out every far-dependent rule at once
             (a mean, a max, the far radius itself), not just the one control.

Run from anywhere (the script's own directory is on sys.path):
  NEC5_EXE=<nec5cl> python scratch/u5-mixed-radius/d_verdict_ladder.py \
      [--ladder] [--sweep] [--invariance] [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import d_step_rod as rod

A_NODE = 1.0e-2
SOILS = {"A": rod.SOIL_A, "poor": rod.SOIL_POOR}


def config(h_over_a=2.5, ratio=1 / 16, r_far=4, a_node=A_NODE, **kw):
    h = h_over_a * a_node
    return dict(
        depth=0.30,
        a_node=a_node,
        a_rad=a_node * ratio,
        a_rise=a_node * (1 - 1e-9),
        eps_gap=3 * h,
        z_split=3 * h,
        n_fed=3,
        h_node_below=2 * h,
        r_far=r_far,
        **kw,
    )


def ladder(out):
    print("== far-mesh ladder, node mesh held at the thin-wire floor ==", flush=True)
    for soil in ("A", "poor"):
        for hoa in (1.67, 2.5):
            for r_far in (1, 2, 4, 8):
                cfg = config(h_over_a=hoa, r_far=r_far)
                m = rod.measure(cfg, soil=SOILS[soil])
                verdict = "node" if m["E_node"] < m["E_far"] else "far"
                out.append(
                    dict(
                        part="ladder",
                        soil=soil,
                        hoa=hoa,
                        r_far=r_far,
                        verdict=verdict,
                        **m,
                    )
                )
                print(
                    f"soil={soil:<5s} panel={hoa:<5g}a h/a={m['h_over_a']:5.2f} "
                    f"r_far={r_far} segs={m['segs']:<4d} S={m['S']:8.4g} "
                    f"D={m['D']:7.4g} S/D={m['SD']:8.4g} | E_node={m['E_node']:8.4g} "
                    f"E_far={m['E_far']:8.4g} S/E={m['SE']:7.4g}  -> {verdict}",
                    flush=True,
                )


def sweep_E(cfg, soil, lo=1 / 32, hi=4.0, n=19):
    """|E| as a function of a_above, plus the two named candidates."""
    a_node, a_rad, a_rise = cfg["a_node"], cfg["a_rad"], cfg["a_rise"]
    b, cb = rod.build(**cfg), rod.build(**rod.control_of(cfg))
    z5 = rod.z_nec5(b, soil=soil)
    gap = rod.z_momwire(cb, soil=soil) - rod.z_nec5(cb, soil=soil)
    xs = np.geomspace(a_node * lo, a_node * hi, n)
    es = np.array(
        [abs((rod.z_momwire(b, float(x), a_rise, soil=soil) - z5) - gap) for x in xs]
    )
    i = int(np.argmin(es))
    return dict(
        xs=[float(v) for v in xs],
        es=[float(v) for v in es],
        argmin_over_a_node=float(xs[i] / a_node),
        E_min=float(es[i]),
        E_at_node=abs((rod.z_momwire(b, a_node, a_rise, soil=soil) - z5) - gap),
        E_at_rad=abs((rod.z_momwire(b, a_rad, a_rise, soil=soil) - z5) - gap),
        a_rad_over_a_node=a_rad / a_node,
    )


def sweep(out):
    print("== |E| swept over a_above ==", flush=True)
    for soil, hoa in (("A", 1.67), ("poor", 2.5)):
        cfg = config(h_over_a=hoa)
        rod.assert_rung(cfg, soil=SOILS[soil])
        s = sweep_E(cfg, SOILS[soil])
        out.append(dict(part="sweep", soil=soil, hoa=hoa, **s))
        print(f"-- soil={soil} panel={hoa}a, a_rad = a_node/16 --", flush=True)
        for x, e in zip(s["xs"], s["es"]):
            print(
                f"   a_above/a_node={x / cfg['a_node']:8.4f}  |E|={e:9.5g}", flush=True
            )
        print(
            f"   argmin at {s['argmin_over_a_node']:.4f} x a_node "
            f"(|E|={s['E_min']:.5g});  |E| at the node member "
            f"{s['E_at_node']:.5g}, at the far wire {s['E_at_rad']:.5g}",
            flush=True,
        )


def invariance(out):
    print("== does the optimum TRACK the radiator's radius? ==", flush=True)
    for ratio in (1 / 32, 1 / 16, 1 / 4, 1 / 2):
        cfg = config(h_over_a=1.67, ratio=ratio)
        rod.assert_rung(cfg, soil=rod.SOIL_A)
        s = sweep_E(cfg, rod.SOIL_A, lo=0.4, hi=2.6, n=15)
        out.append(dict(part="invariance", soil="A", ratio=ratio, **s))
        print(
            f"a_rad = {ratio:.4f} x a_node   argmin a_above = "
            f"{s['argmin_over_a_node']:.4f} x a_node  (|E|={s['E_min']:.5g});  "
            f"|E| at the node member {s['E_at_node']:.5g}, at the far wire "
            f"{s['E_at_rad']:.5g}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--invariance", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not (args.ladder or args.sweep or args.invariance):
        args.ladder = args.sweep = args.invariance = True
    exe = os.environ["NEC5_EXE"]
    meta = dict(
        a_node=A_NODE,
        engine_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest(),
    )
    print(meta, flush=True)
    out = []
    if args.ladder:
        ladder(out)
    if args.sweep:
        sweep(out)
    if args.invariance:
        invariance(out)
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
