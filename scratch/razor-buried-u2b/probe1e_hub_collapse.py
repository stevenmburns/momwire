"""Is detached_hub's 9.9e-9 BxA collapse a radius-rule matter or stand-off
quadrature? Uniform radii vs spread, at three gaps; U1's detached deck at the
same gaps as control."""

import functools
import json
import numpy as np
from common import A_WIRE, RazorSolver, detached, detached_hub


def coll(mk):
    s1 = RazorSolver(
        **{k: v for k, v in mk(eps=(1.0, 0.0)).items() if k != "junctions"},
        nec5_quadrature=True,
    )
    sf = RazorSolver(
        **{k: v for k, v in mk(ground=False).items() if k != "junctions"},
        nec5_quadrature=True,
    )
    g1, gf = s1._build_geometry(), sf._build_geometry()
    Z1, Zf = s1._assemble_Z(g1, s1.k), sf._assemble_Z(gf, sf.k)
    media = s1._wire_media()
    off = np.asarray(g1["basis_offsets"])
    B = np.concatenate(
        [np.arange(off[w], off[w + 1]) for w, m in enumerate(media) if m == "below"]
    )
    A = np.concatenate(
        [np.arange(off[w], off[w + 1]) for w, m in enumerate(media) if m == "above"]
    )
    out = {}
    for bn, rr, cc in (("AxB", A, B), ("BxA", B, A)):
        b1, bf = Z1[np.ix_(rr, cc)], Zf[np.ix_(rr, cc)]
        out[bn] = float(np.max(np.abs(b1 - bf)) / np.max(np.abs(bf)))
    return out


SPREAD = (A_WIRE / 2, A_WIRE, A_WIRE / 4, A_WIRE * 2)
for gap in (0.01, 0.05, 0.14):
    for name, radii in (("uniform", (A_WIRE,) * 4), ("spread", SPREAD)):
        print(
            json.dumps(
                dict(
                    deck="hub",
                    gap=gap,
                    radii=name,
                    rel=coll(functools.partial(detached_hub, 1, radii=radii, gap=gap)),
                )
            ),
            flush=True,
        )
    print(
        json.dumps(
            dict(
                deck="u1",
                gap=gap,
                radii="uniform",
                rel=coll(functools.partial(detached, 1, gap=gap)),
            )
        ),
        flush=True,
    )
    print(
        json.dumps(
            dict(
                deck="u1",
                gap=gap,
                radii="mixed",
                rel=coll(
                    functools.partial(
                        detached, 1, gap=gap, wire_radius=[A_WIRE / 4, A_WIRE]
                    )
                ),
            )
        ),
        flush=True,
    )
