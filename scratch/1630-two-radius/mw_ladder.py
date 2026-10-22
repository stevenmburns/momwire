"""momwire over the decks.py NEC-4.2 decks, the same grid as ref_ladder.py,
one JSON line per solve, flushed. Usage:
  python mw_ladder.py OUT.jsonl BASIS[,BASIS] VARIANT[,..] NRAD[,..] r[,..] [CFG,..] [FEED,..]
"""

from __future__ import annotations

import json
import sys
import time
import warnings

import decks
from ref_ladder import RADII

from momwire.deck import build_solver, parse


def solve(basis, variant, r, feed, a_v, a_b, n_rad):
    model = parse(
        decks.nec42(variant, r, feed, a_v=a_v, a_b=a_b, n_rad=n_rad), dialect="nec4"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        s = build_solver(model, basis=basis).solver
        z = s.compute_impedance()
        z = complex(z[0] if isinstance(z, tuple) else z)
        dt = time.perf_counter() - t0
    return dict(
        R=z.real,
        X=z.imag,
        s=round(dt, 2),
        ek=bool(getattr(s, "extended_kernel", False)),
        exact=str(getattr(s, "exact_kernel", None)),
        warn=sorted({type(w.message).__name__ for w in caught}),
    )


def main():
    out = sys.argv[1]
    bases = sys.argv[2].split(",")
    variants = sys.argv[3].split(",")
    nrads = [int(x) for x in sys.argv[4].split(",")]
    rs = [int(x) for x in sys.argv[5].split(",")]
    cfgs = sys.argv[6].split(",") if len(sys.argv) > 6 else list(RADII)
    feeds = sys.argv[7].split(",") if len(sys.argv) > 7 else ["mid"]
    with open(out, "a") as fh:
        for variant in variants:
            for n_rad in nrads:
                for cfg in cfgs:
                    a_v, a_b = RADII[cfg]
                    for feed in feeds:
                        for r in rs:
                            for basis in bases:
                                try:
                                    rec = solve(
                                        basis, variant, r, feed, a_v, a_b, n_rad
                                    )
                                except Exception as e:  # noqa: BLE001 - recorded
                                    rec = dict(error=f"{type(e).__name__}: {e}"[:300])
                                rec.update(
                                    engine=basis,
                                    variant=variant,
                                    n_rad=n_rad,
                                    cfg=cfg,
                                    r=r,
                                    feed=feed,
                                )
                                fh.write(json.dumps(rec) + "\n")
                                fh.flush()


if __name__ == "__main__":
    main()
