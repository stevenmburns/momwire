"""U4 src PR guards G-B, G-N and G-X (MEASUREMENTS.md, "The src PR").

Run with the momwire tree under test first on PYTHONPATH:

  PYTHONPATH=<tree>/src python scratch/u4-below-range/src_guards.py gb [--numpy]
  PYTHONPATH=<tree>/src python scratch/u4-below-range/src_guards.py native \\
      --refine 1 --cap 5 [--solver sg]

`gb` solves `fan_rise_deck()` (soil A, inside the cap) from that tree's own
tests; `native` solves the synthetic deck from the branch's
`test_below_past_cap_zero_1053.synthetic_u4` with the cap set to `--cap`.
Prints one JSON line: the momwire file imported, the spelling, and Z in full
precision.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import momwire
from momwire import _sommerfeld_below as below


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("gb", "native"))
    ap.add_argument("--numpy", action="store_true")
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--cap", type=float, default=4.0)
    ap.add_argument("--solver", choices=("bspline", "sg"), default="bspline")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(momwire.__file__).resolve().parents[2] / "tests"))
    below._FORCE_NUMPY = bool(args.numpy)
    from momwire.bspline import BSplineSolver
    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

    cls = SinusoidalGalerkinSolver if args.solver == "sg" else BSplineSolver
    if args.mode == "gb":
        from test_crossing_serve_524 import fan_rise_deck

        build = fan_rise_deck()
    else:
        below._SOMM_BELOW_R1_CAP_LAMBDA_M = args.cap
        from test_below_past_cap_zero_1053 import synthetic_u4

        build = synthetic_u4(args.refine)
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = complex(cls(**build).compute_impedance()[0])
    print(
        json.dumps(
            dict(
                momwire=momwire.__file__,
                mode=args.mode,
                numpy=args.numpy,
                refine=args.refine,
                cap=args.cap,
                solver=args.solver,
                z=repr(z),
                seconds=round(time.time() - t0, 1),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
