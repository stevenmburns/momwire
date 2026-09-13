"""U5 step (b'): step (b)'s Richardson ladder under ONE node potential.

DERIVATION-MIXED-RADIUS.md §8.8 identified observer-side's error: it evaluates
the crossing node's point tests on two surfaces. This re-runs step (b)'s rod
ladder (b_rod_ladder.py: same builder, configurations, lengths, refinements,
matched ports) under the one-potential spelling of check 3:

  node_B    line tests at their own wire's radius; both node rows' point tests
            at the buried member's radius (check3_node_potential.py)
  VC_keep   the same run with the crossing junction's KCL multiplier added and
            every point term kept (the merged dof)

NEC-5's rungs are the ones step (b) banked in b_<config>.json (the same deck
builder and binary). Before any momwire rung, NEC-5 is re-run at one rung and
the run stops unless it reproduces the banked value exactly. At every rung the
segment count must equal the banked NEC-5 count; on the control every rung must
reproduce step (b)'s banked momwire Z (equal radii: the shipped fill).

Run from the momwire repo root:
  NEC5_EXE=<nec5cl> python scratch/u5-mixed-radius/b_ladder_onepot.py CONFIG [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import b_rod_ladder as bl
import check3_node_potential as c3

from antennaknobs.engines.nec5 import NEC5Engine

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", choices=sorted(bl.CONFIGS))
    ap.add_argument("--lengths", type=float, nargs="+", default=[0.30, 0.60, 1.20])
    ap.add_argument("--refines", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--spot", type=float, nargs=2, default=[0.30, 4])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    a_top, a_rise = bl.CONFIGS[args.config]
    banked = {
        (row["L"], row["r"]): row
        for row in json.loads((HERE / f"b_{args.config}.json").read_text())["rows"]
    }
    exe = os.environ["NEC5_EXE"]
    meta = dict(
        config=args.config,
        a_top=a_top,
        a_rise=a_rise,
        momwire_file=c3.momwire.__file__,
        nec5_exe=exe,
        nec5_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest(),
    )
    print(meta, flush=True)

    spot_L, spot_r = args.spot[0], int(args.spot[1])
    zn = complex(
        NEC5Engine(
            bl.build(spot_L, spot_r, a_top, a_rise), ground=bl.SOIL_A
        ).impedance()[0]
    )
    if str(zn) != banked[(spot_L, spot_r)]["nec5"]:
        raise SystemExit(
            f"NEC-5 at L={spot_L} r={spot_r} gives {zn}, banked "
            f"{banked[(spot_L, spot_r)]['nec5']}: the banked rungs are not this deck; stop"
        )
    meta["nec5_spot"] = dict(L=spot_L, r=spot_r, z=str(zn), reproduces_banked=True)
    print(f"NEC-5 spot L={spot_L} r={spot_r} reproduces the banked {zn}", flush=True)

    out = []
    c3.install()
    try:
        for L in args.lengths:
            for r in args.refines:
                ref = banked[(L, r)]
                rec = c3.run_b_rod(a_top, a_rise, "node_B", L=L, r=r)
                if rec["segs"] != ref["nec5_segs"]:
                    raise RuntimeError(
                        f"momwire {rec['segs']} segments vs NEC-5 {ref['nec5_segs']}"
                    )
                z_banked = complex(ref["momwire"]["z"])
                if args.config == "control":
                    rel = abs(complex(rec["z"]) - z_banked) / abs(z_banked)
                    if rel > 1e-9:
                        raise SystemExit(
                            f"control L={L} r={r}: node_B {rec['z']} vs banked "
                            f"{z_banked} (rel {rel:.1e}); not the shipped fill, stop"
                        )
                out.append(
                    dict(
                        L=L,
                        r=r,
                        nec5=ref["nec5"],
                        nec5_segs=ref["nec5_segs"],
                        banked_observer_z=ref["momwire"]["z"],
                        node_B=rec,
                    )
                )
                zn_r = complex(ref["nec5"])
                z_s, z_k = complex(rec["z"]), complex(rec["vc_keep"]["z"])
                print(
                    f"L={L:4.2f} r={r} segs={rec['segs']}  NEC-5 {zn_r:.6g}  "
                    f"node_B {z_s:.6g} dZ {zn_r - z_s:.4g} kcl {rec['kcl_rel']:.2e}  "
                    f"VC_keep {z_k:.6g} dZ {zn_r - z_k:.4g}  "
                    f"|split-keep| {abs(z_s - z_k):.2e}",
                    flush=True,
                )
    finally:
        c3.uninstall()
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, rows=out), indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
