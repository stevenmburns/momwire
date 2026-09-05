"""momwire#920: what does the per-BLOCK phase guard actually cost?

The cliff: one coarse segment anywhere in a deck disables the pair-order
ladder for every pair in the block, because `_ladder_for_block` keys the
kL test on the block's LONGEST segment. This measures the price on the deck
that exhibits it, before choosing how to fix it.

Arms:
  fine-only        400-segment loop, no strut          ladder ACTIVE
  fine + strut     the same loop plus one 0.3-lambda segment, guard FIRES
  fine + strut,
  strut meshed     the same conductor at lambda/20     ladder ACTIVE again
"""

import statistics
import time

import numpy as np

from momwire.bspline import BSplineSolver

import os

HALF, RAD, LAM = 0.25, 1e-4, 1.0
N_PER = int(os.environ.get("N_PER", "100"))


def deck(strut_segs=None, strut_len=0.3):
    w = np.array(
        [
            [0.0, 0.0, 0.0],
            [HALF, 0.0, 0.0],
            [HALF, HALF, 0.0],
            [0.0, HALF, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    wires, npe = [w], [[N_PER] * 4]
    if strut_segs:
        wires.append(np.array([[0.0, 0.0, -0.4], [0.0, strut_len, -0.4]]))
        npe.append([strut_segs])
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=LAM,
        wire_radius=RAD,
    )


def clock(d, reps=5):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        BSplineSolver(**d).compute_impedance()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


print(f"threads={os.environ.get('OMP_NUM_THREADS', 'default')}  segments={4 * N_PER}")
rows = [
    ("fine only (ladder active)", deck()),
    ("fine + 1 coarse strut seg (guard fires)", deck(strut_segs=1)),
    ("fine + same strut at lambda/20 (active)", deck(strut_segs=6)),
]
base = None
for name, d in rows:
    t = clock(d)
    if base is None:
        base = t
    print(f"{name:44s} {t:7.3f} s   x{t / base:5.2f}")
