"""U2b-2 supplement: (g1) the radius-response differential against mesh
(does razor - bspline shrink?), (g2) reciprocity under the observer
cross-block rule (is the reciprocity gate a discriminator here?)."""

import json

import numpy as np
from probe2_two_radius import A, Rule, RazorSolver, rod, strip, zb, zr

for m in (2, 4, 8):
    req, beq = zr(rod(m)).real, zb(rod(m)).real
    for name, aa, ab in (("rise/4", A, A / 4), ("top/4", A / 4, A)):
        d = rod(m, aa, ab)
        dr, db = zr(d).real - req, zb(d).real - beq
        print(
            json.dumps(
                dict(
                    gate="g1", deck=name, m=m, dR_razor=dr, dR_bspline=db, diff=dr - db
                )
            ),
            flush=True,
        )

for name, aa, ab in (("rise/2", A, A / 2), ("top/4", A / 4, A)):
    xs = []
    for m in (1, 2, 4):
        d = rod(m, aa, ab)
        d["feeds"] = [(1, 4.5, 1 + 0j), (0, 1.0, 1 + 0j)]
        with Rule(cross=(aa, ab)):
            Y = np.asarray(
                RazorSolver(**strip(d), nec5_quadrature=True).compute_y_matrix()
            )
        xs.append(abs(Y[0, 1] - Y[1, 0]) / abs(Y[0, 1]))
    print(
        json.dumps(
            dict(
                gate="g2",
                deck=name,
                rule="observer cross",
                nonrec=xs,
                ratios=[a / b for a, b in zip(xs, xs[1:])],
            )
        ),
        flush=True,
    )
