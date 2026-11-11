"""Per-position cost of `PortSites.y` against a full solve (momwire#1915).

A ~1,000-unknown design: a centre-fed d=2 dipole (wire 0) beside a parallel
knob wire (wire 1) of the same length, 500 cells each. Measures

* the fresh solve the optimiser would otherwise pay per position:
  `compute_y_matrix()` of a solver built with a 0 V point feed at x;
* `port_sites(1)`: the one fill + factorisation + site back-substitutions;
* `y([x])` and `dy_dx([x])` per position, median over many positions.

Run once, single process; memory is one dense N x N complex Z plus the site
columns (reported).

    OMP_NUM_THREADS=2 python scripts/bench_port_sites_1915.py
"""

from __future__ import annotations

import statistics
import time

import numpy as np

import momwire
from momwire import BSplineSolver

WL = 20.0
L = 9.5
N_CELLS = 500


def deck():
    return dict(
        wires=[
            np.array([(0.0, 0.0, 0.0), (0.0, 0.0, L)]),
            np.array([(1.5, 0.0, 0.0), (1.5, 0.0, L)]),
        ],
        n_per_edge_per_wire=[[N_CELLS], [N_CELLS]],
        wavelength=WL,
        wire_radius=0.001,
        degree=2,
    )


def main():
    print(f"momwire {momwire.__file__}  accelerated={momwire.accelerated}")
    feed = [(0, L / 2, 1.0 + 0j)]
    x0 = 0.3 * L

    t0 = time.perf_counter()
    fresh = BSplineSolver(feeds=feed + [(1, x0, 0j)], **deck())
    y_fresh = fresh.compute_y_matrix()
    t_fresh = time.perf_counter() - t0
    n_unknowns = fresh._build_basis_polynomials(fresh._build_geometry())[0].shape[0]

    t0 = time.perf_counter()
    ps = BSplineSolver(feeds=feed, **deck()).port_sites(1)
    t_build = time.perf_counter() - t0

    rel = float(np.max(np.abs(ps.y([x0]) - y_fresh)) / np.max(np.abs(y_fresh)))

    xs = np.linspace(0.01 * L, 0.99 * L, 400)
    ty, td = [], []
    for x in xs:
        t0 = time.perf_counter()
        ps.y([x])
        ty.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        ps.dy_dx([x])
        td.append(time.perf_counter() - t0)
    t_y = statistics.median(ty)
    t_d = statistics.median(td)

    print(f"unknowns                 {n_unknowns}")
    print(f"sites cached             {ps.n_sites}  ({ps.nbytes / 1e6:.1f} MB)")
    print(f"fresh solve (1 position) {t_fresh:.3f} s")
    print(f"port_sites build (once)  {t_build:.3f} s")
    print(f"y([x]) per position      {t_y * 1e6:.1f} us  (median of {xs.size})")
    print(f"dy_dx([x]) per position  {t_d * 1e6:.1f} us")
    print(f"fresh / y per position   {t_fresh / t_y:.0f}x")
    print(f"max rel |Y - fresh| at x0 {rel:.2e}")


if __name__ == "__main__":
    main()
