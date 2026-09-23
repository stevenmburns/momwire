"""U3: a jacket on the ABOVE wire of a crossing deck (served): the jacket's
shift razor vs bspline per doubling; and the buried-jacket refusal."""

import json
import time

import numpy as np
from common import BSplineSolver, RazorSolver, crossing_deck

JK = dict(insulation_radius=[np.nan, 0.002], insulation_eps_r=[np.nan, 3.0])


def refined(d, m):
    d["n_per_edge_per_wire"] = [[n * m for n in e] for e in d["n_per_edge_per_wire"]]
    return d


def deck(m, **kw):
    d = crossing_deck(1, **kw)
    d["feeds"] = [(1, 4.5, 1 + 0j)]
    return refined(d, m)


def razor(d):
    return RazorSolver(
        **{k: v for k, v in d.items() if k != "junctions"}, nec5_quadrature=True
    )


for m in (1, 2, 4, 8):
    t0 = time.time()
    zr0 = complex(razor(deck(m)).compute_impedance()[0])
    zrj = complex(razor(deck(m, **JK)).compute_impedance()[0])
    zb0 = complex(BSplineSolver(**deck(m)).compute_impedance()[0])
    zbj = complex(BSplineSolver(**deck(m, **JK)).compute_impedance()[0])
    print(
        json.dumps(
            dict(
                m=m,
                razor_shift=[(zrj - zr0).real, (zrj - zr0).imag],
                bspline_shift=[(zbj - zb0).real, (zbj - zb0).imag],
                gap=abs((zrj - zr0) - (zbj - zb0)),
                t=round(time.time() - t0, 1),
            )
        ),
        flush=True,
    )
s = razor(deck(1, insulation_radius=[0.002, np.nan], insulation_eps_r=[3.0, np.nan]))
print(json.dumps(dict(buried_jacket_preflight=s.buried_serve_refusal())))
