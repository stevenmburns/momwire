"""U2b-1 gates on the real constructor: detached decks with mixed radii.

(a) route + partitions counted; (b) reciprocity ladder; (c) eps~=1 collapse
per block against razor's own free-space fill, for the source rule and for
the rules it must discriminate against; (d) gap to bspline per doubling.
"""

import json
import sys
import time

import numpy as np
from common import (
    A_WIRE,
    BSplineSolver,
    RazorSolver,
    detached,
    detached_hub,
    y2,
)

from momwire import _crossing_fill

which = sys.argv[1] if len(sys.argv) > 1 else "all"

AB = (A_WIRE / 4, A_WIRE)  # below, above


def rows(**kw):
    print(json.dumps(kw), flush=True)


# ------------------------------------------------------------------ (a)
calls = []
_f, _r = (
    _crossing_fill.cross_complete_block,
    _crossing_fill.cross_complete_block_reversed,
)


def wrap(fn, tag):
    def g(ctx, *a, **k):
        calls.append((tag, float(ctx.a_wire)))
        return fn(ctx, *a, **k)

    return g


decks = {
    "mixed": lambda m, **k: detached(m, wire_radius=[AB[0], AB[1]], **k),
    "mixed_inv": lambda m, **k: detached(m, wire_radius=[AB[1], AB[0]], **k),
    "hub_spread": lambda m, **k: detached_hub(
        m, radii=(A_WIRE / 2, A_WIRE, A_WIRE / 4, A_WIRE * 2), **k
    ),
}

if which in ("all", "a"):
    _crossing_fill.cross_complete_block = wrap(_f, "fwd")
    _crossing_fill.cross_complete_block_reversed = wrap(_r, "rev")
    for name, mk in decks.items():
        calls.clear()
        s = RazorSolver(
            **{k: v for k, v in mk(1).items() if k != "junctions"}, nec5_quadrature=True
        )
        assert s._detached
        z, _ = s.compute_impedance()
        rows(gate="a", deck=name, calls=calls[:], finite=bool(np.all(np.isfinite(z))))
    _crossing_fill.cross_complete_block = _f
    _crossing_fill.cross_complete_block_reversed = _r

# ------------------------------------------------------------------ (b)
if which in ("all", "b"):
    for name, mk in decks.items():
        nr = []
        for m in (1, 2, 4, 8):
            t = time.time()
            s = RazorSolver(
                **{k: v for k, v in mk(m).items() if k != "junctions"},
                nec5_quadrature=True,
            )
            nr.append(y2(s)[2])
            rows(
                gate="b", deck=name, m=m, nonrec=nr[-1], secs=round(time.time() - t, 2)
            )
        rows(gate="b", deck=name, ratios=[a / b for a, b in zip(nr, nr[1:])])

# ------------------------------------------------------------------ (c)
RULES = ("source", "observer", "wire0", "min", "max")


def forced(rule, radii_of):
    """Wrap the two blocks so each fills at `rule`'s radius. `radii_of` maps
    side -> the radii present there; 'source' is the shipped code path."""

    def f(ctx, A, B, **k):
        if rule == "observer":
            ctx = ctx._replace(a_wire=radii_of["above_obs"])
        elif rule in ("wire0", "min", "max"):
            ctx = ctx._replace(a_wire=radii_of[rule])
        return _f(ctx, A, B, **k)

    def r(ctx, P, Q, **k):
        if rule == "observer":
            ctx = ctx._replace(a_wire=radii_of["below_obs"])
        elif rule in ("wire0", "min", "max"):
            ctx = ctx._replace(a_wire=radii_of[rule])
        return _r(ctx, P, Q, **k)

    return f, r


if which in ("all", "c"):
    for name, mk in decks.items():
        for lane in (True, False):
            s1 = RazorSolver(
                **{k: v for k, v in mk(1, eps=(1.0, 0.0)).items() if k != "junctions"},
                nec5_quadrature=lane,
            )
            sf = RazorSolver(
                **{k: v for k, v in mk(1, ground=False).items() if k != "junctions"},
                nec5_quadrature=lane,
            )
            g1, gf = s1._build_geometry(), sf._build_geometry()
            Zf = sf._assemble_Z(gf, sf.k)
            media = s1._wire_media()
            rad = np.asarray(s1._radius_per_wire)
            bw = [w for w, m in enumerate(media) if m == "below"]
            aw = [w for w, m in enumerate(media) if m == "above"]
            radii_of = {
                "wire0": float(rad[0]),
                "min": float(rad.min()),
                "max": float(rad.max()),
                "above_obs": float(rad[aw[0]]),
                "below_obs": float(rad[bw[0]]),
            }
            off = np.asarray(g1["basis_offsets"])
            B = np.concatenate([np.arange(off[w], off[w + 1]) for w in bw])
            A = np.concatenate([np.arange(off[w], off[w + 1]) for w in aw])
            for rule in RULES:
                f, r = forced(rule, radii_of)
                _crossing_fill.cross_complete_block = f
                _crossing_fill.cross_complete_block_reversed = r
                try:
                    Z1 = s1._assemble_Z(g1, s1.k)
                finally:
                    _crossing_fill.cross_complete_block = _f
                    _crossing_fill.cross_complete_block_reversed = _r
                out = {}
                for bn, rr, cc in (
                    ("AxB", A, B),
                    ("BxA", B, A),
                    ("AxA", A, A),
                    ("BxB", B, B),
                ):
                    b1, bf = Z1[np.ix_(rr, cc)], Zf[np.ix_(rr, cc)]
                    out[bn] = float(np.max(np.abs(b1 - bf)) / np.max(np.abs(bf)))
                rows(gate="c", deck=name, lane=lane, rule=rule, rel=out)

# ------------------------------------------------------------------ (d)
if which in ("all", "d"):
    for name, mk in decks.items():
        gaps = []
        for m in (1, 2, 4, 8):
            t = time.time()
            d = mk(m)
            _, Zr, nr = y2(
                RazorSolver(
                    **{k: v for k, v in d.items() if k != "junctions"},
                    nec5_quadrature=True,
                )
            )
            _, Zb, _ = y2(BSplineSolver(**d))
            g = {
                k: float(abs(Zr[i, j] - Zb[i, j]))
                for k, (i, j) in (("z11", (0, 0)), ("z22", (1, 1)), ("z12", (0, 1)))
            }
            gaps.append(g)
            rows(
                gate="d",
                deck=name,
                m=m,
                gap=g,
                zr12=str(Zr[0, 1]),
                zb12=str(Zb[0, 1]),
                secs=round(time.time() - t, 1),
            )
        for k in ("z11", "z22", "z12"):
            s = [g[k] for g in gaps]
            rows(gate="d", deck=name, key=k, ratios=[b / a for a, b in zip(s, s[1:])])
