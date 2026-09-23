"""U3: a lumped load AT the crossing knot (the crossing tent's diagonal).

razor loaded vs razor's own 2-port algebra (exact: one diagonal entry is a
Sherman-Morrison update of the port there), and vs bspline's 2-port algebra
(converging). Red control: the crossing tent's entries omitted."""

import json
import time

import numpy as np
from common import BSplineSolver, RazorSolver, crossing_deck

ZL = 50.0 + 20.0j


def refined(d, m):
    d["n_per_edge_per_wire"] = [[n * m for n in e] for e in d["n_per_edge_per_wire"]]
    return d


def deck(m, **kw):
    d = crossing_deck(1, **kw)
    d["feeds"] = [(1, 4.5, 1 + 0j)]
    return refined(d, m)


def razor(d, **kw):
    return RazorSolver(
        **{k: v for k, v in d.items() if k != "junctions"}, nec5_quadrature=True, **kw
    )


def algebra(Zm):
    return Zm[0, 0] - Zm[0, 1] * Zm[1, 0] / (Zm[1, 1] + ZL)


for m in (1, 2, 4, 8):
    t0 = time.time()
    two = deck(m)
    two["feeds"] = [(1, 4.5, 1 + 0j), (1, 0.0, 1 + 0j)]
    zr_alg = algebra(np.linalg.inv(np.asarray(razor(two).compute_y_matrix())))
    zb_alg = algebra(np.linalg.inv(np.asarray(BSplineSolver(**two).compute_y_matrix())))
    zr0 = complex(razor(deck(m)).compute_impedance()[0])
    zb0 = complex(BSplineSolver(**deck(m)).compute_impedance()[0])
    zr = complex(razor(deck(m), lumped_loads=[(1, 0.0, ZL)]).compute_impedance()[0])
    print(
        json.dumps(
            dict(
                m=m,
                razor_vs_own_algebra=abs(zr - zr_alg),
                shift_gap_vs_bspline=abs((zr - zr0) - (zb_alg - zb0)),
                razor_shift=[(zr - zr0).real, (zr - zr0).imag],
                bspline_shift=[(zb_alg - zb0).real, (zb_alg - zb0).imag],
                t=round(time.time() - t0, 1),
            )
        ),
        flush=True,
    )
