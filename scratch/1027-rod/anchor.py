"""momwire#1027 step 1: re-run #1027's anchor rows on v0.54.0, both engines.

Geometry, deck and row definition are #1027's own `rod_ladder.py` (banked
here unchanged). Richardson is first order from refine 8 to 16, exactly as
`rod_converged.json` was built: dR_inf = 2*dR16 - dR8, fraction = dR_inf /
R_momwire(16), dR = NEC-5 minus momwire.

A counter on `_crossing_fill.cross_complete_block_split` records whether the
#956 crossing fill is entered at all; a wholly buried rod should never enter it.

Run from the momwire repo root, antennaknobs installed alongside, NEC-5 named:
  NEC5_EXE=<nec5cl> python scratch/1027-rod/anchor.py [--out FILE.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import rod_ladder as rl

import antennaknobs
import momwire
from momwire import _crossing_fill

# #1027's rod_converged.json, verbatim
REF = {
    0.15: dict(
        dR8=-61.679973625099365, dR16=-63.36628952090496, R=1670.866289520905, segs16=36
    ),
    0.60: dict(
        dR8=-6.348372845341601, dR16=-6.4871064522182, R=550.6771064522183, segs16=136
    ),
    2.40: dict(
        dR8=-0.46460893277651394,
        dR16=-0.4741767923253235,
        R=176.80417679232534,
        segs16=524,
    ),
}

CALLS = {"cross": 0}
_orig_cross = _crossing_fill.cross_complete_block_split


def _counting_cross(*args, **kwargs):
    CALLS["cross"] += 1
    return _orig_cross(*args, **kwargs)


_crossing_fill.cross_complete_block_split = _counting_cross


def git_short(path):
    res = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.stdout.strip() or "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    exe = os.environ["NEC5_EXE"]
    meta = dict(
        momwire=git_short(Path(momwire.__file__).parent),
        antennaknobs=git_short(Path(antennaknobs.__file__).parent),
        nec5_exe=exe,
        nec5_sha256=hashlib.sha256(Path(exe).read_bytes()).hexdigest(),
    )
    print(meta, flush=True)

    out = []
    for L, ref in REF.items():
        CALLS["cross"] = 0
        r8 = rl.row("A", L, 0.20, 42, refine=8)
        r16 = rl.row("A", L, 0.20, 42, refine=16)
        d_inf = 2.0 * r16["dR"] - r8["dR"]
        frac = d_inf / r16["momwire_R"]
        ref_inf = 2.0 * ref["dR16"] - ref["dR8"]
        ref_frac = ref_inf / ref["R"]
        rec = dict(
            L=L,
            segs8=r8["segs"],
            segs16=r16["segs"],
            buried=r8["buried"] and r16["buried"],
            crossing_calls=CALLS["cross"],
            rows=[r8, r16],
            dR_inf=d_inf,
            frac_pct=100 * frac,
            ref_frac_pct=100 * ref_frac,
            dfrac_pp=100 * (frac - ref_frac),
            R16_rel=(r16["momwire_R"] - ref["R"]) / ref["R"],
            dR8_vs_ref=r8["dR"] - ref["dR8"],
            dR16_vs_ref=r16["dR"] - ref["dR16"],
        )
        out.append(rec)
        print(
            f"L={L:4.2f} segs {r8['segs']}/{r16['segs']} (ref16 {ref['segs16']}) "
            f"buried={rec['buried']} cross={CALLS['cross']}\n"
            f"   momwire R8/R16 {r8['momwire_R']:.4f}/{r16['momwire_R']:.4f}  "
            f"nec5 R8/R16 {r8['nec5_R']}/{r16['nec5_R']}\n"
            f"   dR8 {r8['dR']:+.4f} (ref {ref['dR8']:+.4f})  "
            f"dR16 {r16['dR']:+.4f} (ref {ref['dR16']:+.4f})  dR_inf {d_inf:+.4f}\n"
            f"   dR/R {100 * frac:+.3f} % (ref {100 * ref_frac:+.3f} %, "
            f"diff {rec['dfrac_pp']:+.4f} pp)  R16 rel {rec['R16_rel']:+.2e}",
            flush=True,
        )
    if args.out is not None:
        args.out.write_text(json.dumps(dict(meta=meta, rungs=out), indent=1))
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
