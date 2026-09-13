"""U5 src PR checks V1 and V2 (SRC-PLAN.md §6).

V1  the src branch against the measured spelling: every check 3' row
    (check3k_check2.json, VC_keep(node_B)) and every step (b') mixed rung
    (bq_*.json, VC_keep(node_B), through antennaknobs' engine). Bar: 1e-9
    relative on every row.
V2  §7 item 1: the catalog buried_radial_vertical through antennaknobs' momwire
    engine at 7.1 (default), 6.8 and 7.4 MHz, soil A, and check 2's rod at
    equal radii, refine 1 and 2. Printed as repr for a bit-for-bit comparison
    between momwire main and the src branch. Also counts crossing-fill calls,
    so a deck that never reaches the crossing serve cannot pass vacuously.

Run with the antennaknobs venv from the momwire scratch directory's parent:
  main: python scratch/u5-mixed-radius/src_verify.py v2 --expect main --out F
  src:  PYTHONPATH=<momwire-wt-u5src>/src python ... --expect src ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def imported_from(expect):
    import momwire

    f = momwire.__file__
    ok = ("momwire-wt-u5src" in f) if expect == "src" else ("antennaknobs/momwire" in f)
    if not ok:
        raise SystemExit(f"momwire imported from {f}, expected {expect}")
    return f


def v1_check2():
    import check2_continuity as c2

    from momwire.bspline import BSplineSolver

    rows = json.loads((HERE / "check3k_check2.json").read_text())["rows"]
    out, worst = [], 0.0
    for row in rows:
        if row["spelling"] != "node_B" or row["a_A"] == row["a_B"]:
            continue
        z_h = complex(row["vc_keep"]["z"])
        s = BSplineSolver(**c2.deck(row["a_A"], row["a_B"], row["refine"]))
        z = complex(s.compute_impedance()[0])
        rel = abs(z - z_h) / abs(z_h)
        worst = max(worst, rel)
        out.append(
            dict(
                a_A=row["a_A"], a_B=row["a_B"], refine=row["refine"], z=repr(z), rel=rel
            )
        )
        print(
            f"check2 a_A={row['a_A']} a_B={row['a_B']} r={row['refine']}: src {z:.10g} harness {z_h:.10g} rel {rel:.2e}",
            flush=True,
        )
    return out, worst


def v1_b():
    import b_rod_ladder as bl

    from antennaknobs.engines.momwire import MomwireEngine

    out, worst = [], 0.0
    for cfg in ("ratio2", "ratio4", "ratio05"):
        a_top, a_rise = bl.CONFIGS[cfg]
        for row in json.loads((HERE / f"bq_{cfg}.json").read_text())["rows"]:
            z_h = complex(row["node_B"]["vc_keep"]["z"])
            b = bl.build(row["L"], row["r"], a_top, a_rise)
            with bl.even_parity():
                eng = MomwireEngine(b, ground=bl.SOIL_A)
                _sim, _coeffs, z = eng._solved_excited(eng._wavelength_for(b.freq))
            z = complex(z)
            rel = abs(z - z_h) / abs(z_h)
            worst = max(worst, rel)
            out.append(dict(config=cfg, L=row["L"], r=row["r"], z=repr(z), rel=rel))
            print(
                f"b {cfg} L={row['L']} r={row['r']}: src {z:.10g} harness {z_h:.10g} rel {rel:.2e}",
                flush=True,
            )
    return out, worst


def v2():
    import check2_continuity as c2

    from antennaknobs.designs.verticals.buried_radial_vertical import Builder
    from antennaknobs.engines.momwire import MomwireEngine
    from momwire import _crossing_fill
    from momwire.bspline import BSplineSolver

    calls = {"n": 0}
    orig = _crossing_fill.cross_complete_block_split

    def counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    _crossing_fill.cross_complete_block_split = counted
    res = {}
    try:
        for f in (7.1, 6.8, 7.4):
            b = Builder()
            b.freq = f
            z = MomwireEngine(b, ground=("finite", 13.0, 0.005)).impedance()
            res[f"brv_{f}"] = [repr(complex(x)) for x in z]
            print(f"brv {f} MHz: {res[f'brv_{f}']}", flush=True)
        res["brv_crossing_calls"] = calls["n"]
        for refine in (1, 2):
            z = complex(
                BSplineSolver(**c2.deck(1e-3, 1e-3, refine)).compute_impedance()[0]
            )
            res[f"rod_equal_r{refine}"] = repr(z)
            print(f"rod equal radii r={refine}: {z!r}", flush=True)
        res["crossing_calls_total"] = calls["n"]
    finally:
        _crossing_fill.cross_complete_block_split = orig
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("v1c2", "v1b", "v2"))
    ap.add_argument("--expect", choices=("main", "src"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    meta = dict(
        mode=args.mode, expect=args.expect, momwire_file=imported_from(args.expect)
    )
    print(meta, flush=True)
    if args.mode == "v2":
        result = v2()
    else:
        rows, worst = v1_check2() if args.mode == "v1c2" else v1_b()
        result = dict(
            rows=rows,
            worst_rel=worst,
            bar=1e-9,
            verdict="HIT" if worst <= 1e-9 else "MISSED",
        )
        print(f"{args.mode}: worst rel {worst:.2e} -> {result['verdict']}", flush=True)
    args.out.write_text(json.dumps(dict(meta=meta, result=result), indent=1))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
