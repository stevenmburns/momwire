"""momwire#1027 step 2(a): take the interface and the loss out.

Same rod, same row definition and the same refine 8 -> 16 Richardson as
`anchor.py`. What changes is the medium, and each engine gets a whole-space
spelling built here because neither front end has one as antennaknobs drives
it:

* NEC-5: its infinite-medium card (verified against our licensed materials),
  inserted after the GE line of `NEC5Engine.deck(ground="free")`.
* momwire: the wholly buried fill with its two interface terms removed -- the
  image weight a_m set to zero and the below/below Sommerfeld remainder
  returning zeros -- which leaves the direct term at k_m, eps_m.

Modes (each writes only with --out):
  controls    C2a.1-C2a.4, the checks that the two whole-space spellings mean
              what they say
  wholespace  P2a.1: soil A, L = 0.15 and 0.60 m, both engines
  depth       P2a.2: buried, L = 0.60 m, d = 0.2 / 1 / 3 / 10 / 18 m
  sigma       P2a.3: whole space, eps_r = 13, sigma = 5 / 1 / 0.1 / 0 mS/m

Run from the momwire repo root with antennaknobs installed alongside:
  NEC5_EXE=<nec5cl> python scratch/1027-rod/step2a_medium.py MODE [--out F]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rod_ladder as rl

import antennaknobs
import momwire
from antennaknobs.engines.momwire import MomwireEngine
from antennaknobs.engines.nec5 import NEC5Engine, run_deck
from momwire import _sommerfeld_below
from momwire.bspline import BSplineSolver

SOIL_A = (13.0, 0.005)
CALLS = {"remainder": 0, "medium": 0}


@contextmanager
def momwire_whole_space():
    """Zero the two interface terms of momwire's wholly buried fill."""
    orig_medium = BSplineSolver._buried_medium
    orig_rem = _sommerfeld_below.remainder_field_proj_below

    def medium(solver):
        CALLS["medium"] += 1
        eps_t, eps_m, k_p, k_m, c2, a_m = orig_medium(solver)
        return eps_t, eps_m, k_p, k_m, c2, 0.0 * a_m

    def remainder(*args, **kwargs):
        CALLS["remainder"] += 1
        return np.zeros_like(orig_rem(*args, **kwargs))

    BSplineSolver._buried_medium = medium
    _sommerfeld_below.remainder_field_proj_below = remainder
    try:
        yield
    finally:
        BSplineSolver._buried_medium = orig_medium
        _sommerfeld_below.remainder_field_proj_below = orig_rem


def sha12(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def nec5_z(deck):
    text = run_deck(os.environ["NEC5_EXE"], deck, timeout=600.0)
    return complex(NEC5Engine._parse_input_parameters(text)[0][0][2])


def um_deck(b, medium):
    """The free-space deck with the infinite-medium card after its GE line."""
    eps_r, sigma = medium
    lines = NEC5Engine(b, ground="free").deck([b.freq]).splitlines()
    ge = [i for i, ln in enumerate(lines) if ln.startswith("GE ")]
    if len(ge) != 1:
        raise RuntimeError(f"expected one GE line, found {len(ge)}")
    card = f"UM 0 0 0 0 {eps_r:.6E} {sigma:.6E}"
    return "\n".join([*lines[: ge[0] + 1], card, *lines[ge[0] + 1 :]]) + "\n"


def buried_z(soil, L, d, refine, *, whole_space=False):
    """(momwire Z, NEC-5 Z, NEC-5 deck sha, segs) for one rung."""
    b = rl.build(soil, L, d, 42, refine=refine)
    g = ("finite", b.design_eps_r, b.design_sigma)
    if whole_space:
        with momwire_whole_space():
            zm = complex(MomwireEngine(b, ground=g).impedance()[0])
        deck = um_deck(b, (b.design_eps_r, b.design_sigma))
        zn = nec5_z(deck)
    else:
        zm = complex(MomwireEngine(b, ground=g).impedance()[0])
        deck = NEC5Engine(b, ground=g).deck([b.freq])
        zn = complex(NEC5Engine(b, ground=g).impedance()[0])
    segs = sum(int(ln.split()[2]) for ln in deck.splitlines() if ln.startswith("GW "))
    return zm, zn, sha12(deck), segs


def ladder(soil, L, d, *, whole_space):
    rows = {}
    for refine in (8, 16):
        zm, zn, sha, segs = buried_z(soil, L, d, refine, whole_space=whole_space)
        rows[refine] = dict(
            momwire=str(zm), nec5=str(zn), dR=zn.real - zm.real, deck_sha=sha, segs=segs
        )
    d_inf = 2.0 * rows[16]["dR"] - rows[8]["dR"]
    r16 = complex(rows[16]["momwire"]).real
    rec = dict(
        soil=list(soil) if not isinstance(soil, str) else soil,
        L=L,
        d=d,
        whole_space=whole_space,
        rows=rows,
        dR_inf=d_inf,
        R16=r16,
        frac_pct=100.0 * d_inf / r16,
    )
    print(
        f"  soil={rec['soil']} L={L:4.2f} d={d:5.2f} ws={whole_space} "
        f"segs {rows[8]['segs']}/{rows[16]['segs']}  "
        f"R16 {r16:.6g}  NEC-5 R16 {complex(rows[16]['nec5']).real:.6g}  "
        f"dR8 {rows[8]['dR']:+.5g} dR16 {rows[16]['dR']:+.5g}  "
        f"dR/R {rec['frac_pct']:+.4f} %",
        flush=True,
    )
    return rec


def rel(a, b):
    return abs(a - b) / abs(b)


def controls():
    L, ref = 0.60, 8
    out = {}
    CALLS.update(remainder=0, medium=0)
    ws02 = buried_z(SOIL_A, L, 0.2, ref, whole_space=True)
    ws18 = buried_z(SOIL_A, L, 18.0, ref, whole_space=True)
    knob_calls = dict(CALLS)
    full02 = buried_z(SOIL_A, L, 0.2, ref)
    full18 = buried_z(SOIL_A, L, 18.0, ref)
    out["C2a.1"] = dict(
        momwire_ws_d02_vs_d18=rel(ws02[0], ws18[0]), knob_calls=knob_calls
    )
    out["C2a.2"] = dict(
        momwire_full_vs_ws_d02=rel(full02[0], ws02[0]),
        momwire_full_vs_ws_d18=rel(full18[0], ws18[0]),
    )
    b = rl.build(SOIL_A, L, 0.2, 42, refine=ref)
    z_free = nec5_z(NEC5Engine(b, ground="free").deck([b.freq]))
    z_um1 = nec5_z(um_deck(b, (1.0, 0.0)))
    out["C2a.3"] = dict(
        nec5_um_1_0=str(z_um1), nec5_free=str(z_free), rel=rel(z_um1, z_free)
    )
    out["C2a.4"] = dict(
        nec5_um_d02_vs_d18=rel(ws02[1], ws18[1]),
        nec5_buried_d18_vs_um=rel(full18[1], ws18[1]),
    )
    for k, v in out.items():
        print(k, v, flush=True)
    return out


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
    ap.add_argument("mode", choices=["controls", "wholespace", "depth", "sigma"])
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

    if args.mode == "controls":
        result = controls()
    elif args.mode == "wholespace":
        result = [ladder(SOIL_A, L, 0.2, whole_space=True) for L in (0.15, 0.60)]
    elif args.mode == "depth":
        result = [
            ladder(SOIL_A, 0.60, d, whole_space=False)
            for d in (0.2, 1.0, 3.0, 10.0, 18.0)
        ]
    else:
        result = [
            ladder((13.0, sigma), L, 0.2, whole_space=True)
            for L in (0.15, 0.60)
            for sigma in (0.005, 0.001, 0.0001, 0.0)
        ]
    if args.out is not None:
        args.out.write_text(
            json.dumps(dict(meta=meta, mode=args.mode, result=result), indent=1)
        )
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
