"""U9 PR gates G2 and G3 (PLAN.md PR gates; Amendment 2 puts both at soil A on
the 5 m rung), on the src branch.

  PYTHONPATH=<src tree>/src python g23_two_node.py g2 --d 5 --out F
  PYTHONPATH=<src tree>/src python g23_two_node.py g3 --d 5 --out F

G2: the cross block split vs dense-direct, block-relative, the way
`test_g688_4` measures it (gate 1e-6; registered prediction <= 1e-7).
G3: SinusoidalGalerkinSolver against BSplineSolver(degree=1), the arbiter
`test_sg_crossing_980_d3` uses, on the 2x2 Z (gate 1e-2 relative; registered
prediction Z11 and Z12 both <= 1e-2).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import momwire

sys.path.insert(0, str(Path(momwire.__file__).resolve().parents[2] / "tests"))

from momwire import SinusoidalGalerkinSolver
from momwire import _crossing_fill as cf
from momwire.bspline import BSplineSolver
from test_crossing_aca_688 import _deck_axes
from test_crossing_serve_524 import two_node_deck

FEED = 4.3333333333


def deck(d):
    b = two_node_deck(d)
    b["feeds"] = [(1, FEED, 1 + 0j), (3, FEED, 1 + 0j)]
    return b


def g2(d):
    _s, ctx, a_idx, b_idx = _deck_axes(deck(d))
    A = cf.axis_data(ctx, a_idx)
    B = cf.axis_data(ctx, b_idx)
    t0 = time.time()
    dense = cf.cross_complete_block(ctx, A, B)
    t_dense = time.time() - t0
    t0 = time.time()
    split = cf.cross_complete_block_split(ctx, a_idx, b_idx, A, B)
    t_split = time.time() - t0
    rel = float(np.abs(split - dense).max() / np.abs(dense).max())
    return dict(
        mode="g2",
        momwire=momwire.__file__,
        d=d,
        block_shape=list(dense.shape),
        block_rel=rel,
        gate=1e-6,
        prediction=1e-7,
        seconds=dict(dense=round(t_dense, 2), split=round(t_split, 2)),
    )


def _zmat(cls, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        y = np.asarray(cls(**deck(G3_D), **kw).compute_port_solution().y, dtype=complex)
    return np.linalg.inv(np.atleast_2d(y)), round(time.time() - t0, 2)


G3_D = 5.0


def g3(d):
    global G3_D
    G3_D = d
    zs, ts = _zmat(SinusoidalGalerkinSolver)
    zb, tb = _zmat(BSplineSolver, degree=1)
    rel = np.abs(zs - zb) / np.abs(zb)
    return dict(
        mode="g3",
        momwire=momwire.__file__,
        d=d,
        z_sg=[[float(v.real), float(v.imag)] for v in zs.ravel()],
        z_bspline_deg1=[[float(v.real), float(v.imag)] for v in zb.ravel()],
        rel=[float(v) for v in rel.ravel()],
        rel_z11=float(rel[0, 0]),
        rel_z12=float(rel[0, 1]),
        gate=1e-2,
        seconds=dict(sg=ts, bspline=tb),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("g2", "g3"))
    ap.add_argument("--d", type=float, default=5.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rec = g2(args.d) if args.what == "g2" else g3(args.d)
    args.out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
