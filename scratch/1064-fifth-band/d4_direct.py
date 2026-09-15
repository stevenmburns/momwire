"""momwire#1064 D4a: the direct below/below surfaces at fixed points, with no
grid, on whichever tree PYTHONPATH selects and whichever accelerator variant
MOMWIRE_FORCE_VARIANT forces.

  MOMWIRE_FORCE_VARIANT=avx2|sse2 PYTHONPATH=<momwire src> \\
      python d4_direct.py --label TREE_VARIANT --out d4_TREE_VARIANT.json

Soil B at 7 MHz, R1/lam_m in {0.2, 1, 3} x theta in {0.5, 2, 30, 60} deg: angles
whose nodes are cheap, so the contour's compiled code is exercised without the
grazing tail. `iv_surfaces_direct_below` is unchanged Python on every tree, so a
difference between trees comes from the compiled contour it calls.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import momwire._accelerators as acc
from momwire import _ground_refl
from momwire import _sommerfeld_below as below
from momwire._sommerfeld import _SURF_KEYS

C0 = 299792458.0
EPS0 = 8.8541878128e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    f = 7e6
    k2 = 2.0 * math.pi * f / C0
    om = 2.0 * math.pi * f
    eps_t = _ground_refl.eps_tilde((20.0, 0.03), om, EPS0)
    lam_m = below.lambda_medium(eps_t, k2)
    pts = [(r, t) for r in (0.2, 1.0, 3.0) for t in (0.5, 2.0, 30.0, 60.0)]
    r1 = np.array([r * lam_m for r, _ in pts])
    th = np.radians([t for _, t in pts])
    batch = below.iv_surfaces_direct_below(eps_t, k2, r1, th, rtol=1e-9, omega=om)
    single = [
        below.iv_surfaces_direct_below(
            eps_t, k2, np.array([a]), np.array([b]), rtol=1e-9, omega=om
        )
        for a, b in zip(r1, th, strict=True)
    ]
    rec = dict(
        label=args.label,
        accelerator=acc.__file__,
        points=[[r, t] for r, t in pts],
        batch={
            k: [[complex(v).real, complex(v).imag] for v in batch[k]]
            for k in _SURF_KEYS
        },
        single={
            k: [[complex(s[k][0]).real, complex(s[k][0]).imag] for s in single]
            for k in _SURF_KEYS
        },
    )
    args.out.write_text(json.dumps(rec))
    print(args.label, acc.__file__.rsplit("/", 1)[-1])


if __name__ == "__main__":
    main()
