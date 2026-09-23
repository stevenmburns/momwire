"""U2b-2: the radius-response differential with the feed ON a knot at every
rung (above arclength 4.5), so razor's feed snap does not move with mesh."""

import json

from probe2_two_radius import A, rod, zb, zr


def fed(d):
    d["feeds"] = [(1, 4.5, 1 + 0j)]
    return d


for m in (1, 2, 4, 8):
    req, beq = zr(fed(rod(m))).real, zb(fed(rod(m))).real
    for name, aa, ab in (
        ("rise/2", A, A / 2),
        ("rise/4", A, A / 4),
        ("top/2", A / 2, A),
        ("top/4", A / 4, A),
    ):
        d = fed(rod(m, aa, ab))
        dr, db = zr(d).real - req, zb(d).real - beq
        print(
            json.dumps(
                dict(gate="h", deck=name, m=m, dR_razor=dr, dR_bspline=db, diff=dr - db)
            ),
            flush=True,
        )
