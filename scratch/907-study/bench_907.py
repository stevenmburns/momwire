"""#907 benchmark: end-to-end wall, ladder vs no ladder, free space.

Steady state, not "min of N short runs": each arm is looped a fixed number of
times and the FIRST-FIFTH vs LAST-FIFTH drift is reported alongside the
steady-state median, so a thermal/turbo window cannot masquerade as a result.

Threads come from the environment (OMP_NUM_THREADS et al must be set before
numpy loads), so the driver runs this once per thread count.
"""

import os
import statistics
import sys
import time

import numpy as np

from momwire.bspline import BSplineSolver

REPS = int(os.environ.get("BENCH_REPS", "9"))


def square_loop(n_per_edge, side=0.25):
    w = np.array(
        [
            [0.0, 0.0, 0.0],
            [side, 0.0, 0.0],
            [side, side, 0.0],
            [0.0, side, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    return dict(
        wires=[w],
        n_per_edge_per_wire=[[n_per_edge] * 4],
        feeds=[(0, 0.5, 1 + 0j)],
        wavelength=1.0,
        wire_radius=1e-4,
    )


def yagi(n_elem=5, n_seg=40, spacing=0.2, half=0.24):
    wires, nppw = [], []
    for i in range(n_elem):
        x = i * spacing
        wires.append(np.array([[x, -half, 0.0], [x, half, 0.0]]))
        nppw.append([n_seg])
    return dict(
        wires=wires,
        n_per_edge_per_wire=nppw,
        feeds=[(0, half, 1 + 0j)],
        wavelength=1.0,
        wire_radius=1e-4,
    )


DECKS = {
    "bent 4-edge, 400 seg": square_loop(100),
    "bent 4-edge, 1200 seg": square_loop(300),
    "bent 4-edge, 2400 seg": square_loop(600),
    "multi-wire yagi 5x40 (200 seg)": yagi(5, 40),
    "multi-wire yagi 8x150 (1200 seg)": yagi(8, 150),
    "multi-wire yagi 10x240 (2400 seg)": yagi(10, 240),
}

ARMS = {
    "no ladder (shipped)": dict(pair_order_ladder=()),
    "ladder ((16,4),)": dict(pair_order_ladder=((16.0, 4),)),
}

# The n_qp=32 question (#760 asked whether the ladder makes it affordable in
# free space) is 16x the quadrature work per pair, so it is asked only of the
# smaller decks.
ARMS_32 = {
    "flat n_qp=32, no ladder": dict(n_qp_pair=32, pair_order_ladder=()),
    "n_qp=32 + ladder ((2,8),(16,4))": dict(
        n_qp_pair=32, pair_order_ladder=((2.0, 8), (16.0, 4))
    ),
}
SMALL = ("bent 4-edge, 400 seg", "multi-wire yagi 5x40 (200 seg)")


def run(deck, kw, reps):
    ts, z = [], None
    for _ in range(reps):
        t = time.perf_counter()
        z, _ = BSplineSolver(**deck, **kw).compute_impedance()
        ts.append(time.perf_counter() - t)
    return ts, z


def report(name, ts):
    fifth = max(1, len(ts) // 5)
    first, last = statistics.mean(ts[:fifth]), statistics.mean(ts[-fifth:])
    drift = 100 * (last - first) / first
    return (statistics.median(ts), drift)


print(
    f"threads: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')}  "
    f"reps={REPS}"
)
only = sys.argv[1] if len(sys.argv) > 1 else None
for dname, deck in DECKS.items():
    if only and only not in dname:
        continue
    print(f"\n{dname}")
    base = None
    arms = dict(ARMS)
    if dname in SMALL:
        arms.update(ARMS_32)
    for aname, kw in arms.items():
        ts, z = run(deck, kw, REPS)
        med, drift = report(aname, ts)
        if base is None:
            base = med
        print(
            f"   {aname:34s} {med:7.3f}s  x{base / med:5.2f}  "
            f"drift {drift:+5.1f}%   Z={z:.6f}"
        )
