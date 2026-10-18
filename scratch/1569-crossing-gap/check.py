"""momwire#1569: a source at a crossing node against the same source one
segment up, under uniform refinement, on bspline and razor.

    python check.py DECK.nec M[,M...]   (KERNEL=ek|exact to change kernels)

DECK is a NEC deck of GW cards (tag 1 the rising wire, starting at the node),
an optional FR card and LD 5 conductivity; every wire's segment count is
multiplied by M. The ground is GN 2 13 / 0.005 (Sommerfeld). Prints, per
level, Z with a node gap at the start of wire 0 and with a feed one segment
up that wire, for each engine.
"""

import math
import os
import sys
import time
import warnings

import numpy as np

from momwire import BSplineSolver, RazorSolver

warnings.simplefilter("ignore")


def read(path):
    wires, segs, radii, freq, sigma = [], [], [], 14.0, None
    for line in open(path):
        p = line.split()
        if not p:
            continue
        if p[0] == "GW":
            v = [float(x) for x in p[3:10]]
            wires.append([tuple(v[0:3]), tuple(v[3:6])])
            segs.append(int(p[2]))
            radii.append(v[6])
        elif p[0] == "FR":
            freq = float(p[5])
        elif p[0] == "LD" and p[1] == "5":
            sigma = float(p[5])
    return wires, segs, radii, freq, sigma


def z(cls, wires, segs, radii, freq, sigma, m, port):
    kw = dict(
        wires=wires,
        n_per_edge_per_wire=[[n * m] for n in segs],
        wire_radius=radii,
        wavelength=299.792458 / freq,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
        **port,
    )
    kernel = os.environ.get("KERNEL", "")
    if kernel == "ek":
        kw["extended_kernel"] = True
    elif kernel == "exact":
        kw["exact_kernel"] = True
    if sigma is not None:
        kw["wire_conductivity"] = sigma
    try:
        y = np.asarray(cls(**kw).compute_y_matrix())
    except Exception as exc:  # noqa: BLE001 — a refusal is a row of the table
        return f"{type(exc).__name__}: {str(exc)[:80]}"
    return 1.0 / complex(y.ravel()[0])


def main():
    wires, segs, radii, freq, sigma = read(sys.argv[1])
    height = math.dist(*wires[0])
    t0 = time.perf_counter()
    for m in (int(x) for x in sys.argv[2].split(",")):
        row = [f"segs/wire x{m} (wire 0: {segs[0] * m})"]
        for cls in (BSplineSolver, RazorSolver):
            node = z(
                cls,
                wires,
                segs,
                radii,
                freq,
                sigma,
                m,
                dict(feeds=[], node_gaps=[(0, "start", 1.0)]),
            )
            up = z(
                cls,
                wires,
                segs,
                radii,
                freq,
                sigma,
                m,
                dict(feeds=[(0, height / (segs[0] * m), 1.0)]),
            )
            row.append(f"{cls.__name__} node {node} up1 {up}")
        print(" | ".join(row), f"t={time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
