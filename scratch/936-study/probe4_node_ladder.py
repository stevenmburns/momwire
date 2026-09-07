"""momwire#936: does the node treatment CONVERGE for a tilted above member?

Laptop-control's decision rule: if the residual ladder on the tilted deck
tracks the untilted adjudicator's convergence class -- falling with node
grading rather than plateauing -- the direct spelling is safe and no new
kernel is needed. If it plateaus or grows, the ln(a)-class content is showing
and the seventh column is required.

Measured at eps~ = 1, where the crossing fill must reproduce the free-space
junction fill exactly. That is blind to the W move by construction (W = 0
there), which is what makes it a clean test of the NODE TREATMENT rather than
of the kernel: it asks whether the ends/corner machinery converges when the
above member leaves the node at an angle.

The ladder refines the node region on both sides.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from momwire import _crossing_fill as CF  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402

WL = 299792458.0 / 7.1e6
A_WIRE = 1e-3


def graded(p0, p1, finest, growth=2.0):
    """Vertices along p0->p1, geometric from `finest` at p0."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    span = float(np.linalg.norm(p1 - p0))
    us, step = [0.0], finest
    while us[-1] + step < span:
        us.append(us[-1] + step)
        step *= growth
    us = np.array([u for u in us if u < span] + [span])
    return p0[None, :] + (us / span)[:, None] * (p1 - p0)[None, :]


def deck(lean_deg, finest):
    a = np.radians(lean_deg)
    node = np.array([0.0, 0.0, 0.0])
    top = 10.0 * np.array([np.sin(a), 0.0, np.cos(a)])
    below = np.array([(5.0, 0.0, -0.15), (0.0, 0.0, -0.15), (0.0, 0.0, 0.0)])
    above = graded(node, top, finest)
    return dict(
        wires=[below, above],
        n_per_edge_per_wire=[[10, 2], [1] * (len(above) - 1)],
        junctions=[[(0, "end"), (1, "start")]],
        feeds=[(1, 4.3333333333, 1 + 0j)],
        wavelength=WL,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=(1.0, 0.0),
        ground_model="sommerfeld",
    )


def residual(lean_deg, finest):
    build = deck(lean_deg, finest)
    truth = {
        k: v
        for k, v in build.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    zf, _ = BSplineSolver(**truth).compute_impedance()
    zc, _ = BSplineSolver(**build).compute_impedance()
    return abs(complex(zc) - complex(zf)), len(build["wires"][1]) - 1


orig = CF._TILT_TOL
CF._TILT_TOL = 1e9
try:
    print("eps~ = 1 residual |crossing - free space|, node grading refined")
    print(
        f"{'finest m':>9s} {'segs':>5s}   "
        + "".join(f"{f'lean {d}':>13s}" for d in (0, 15, 45))
    )
    for finest in (0.40, 0.20, 0.10, 0.05, 0.025):
        row, segs = [], None
        for lean in (0.0, 15.0, 45.0):
            r, n = residual(lean, finest)
            segs = n
            row.append(r)
        print(f"{finest:9.3f} {segs:5d}   " + "".join(f"{v:13.5f}" for v in row))
finally:
    CF._TILT_TOL = orig
