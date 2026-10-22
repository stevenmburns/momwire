"""Radius responses on the 1-radial hub deck (momwire#1630): each config's Z
minus the one-radius control's, momwire bases (NEC-4.2-dialect deck, segment
source) against NEC-5 (knot source). The node condition's gap cancels in the
difference. Usage: OUT.jsonl r [patch]
patch (point-matched only, harness-side): none | rows_above | rows_src"""

from __future__ import annotations

import json
import sys
import warnings

import decks
from ref_ladder import NEC5, run

from momwire import _crossing_fill
from momwire.deck import build_solver, parse

CFG = {
    "c0_all1": (0.001, 0.001),
    "c1_v2": (0.002, 0.001),
    "c2_v4": (0.004, 0.001),
    "c3_v19": (0.019, 0.001),
    "c4_b05": (0.001, 0.0005),
    "c5_b025": (0.001, 0.00025),
}
_POB = _crossing_fill.point_observer_block


def patched(mode):
    def pob(ctx, *a, observers_above, **kw):
        if ctx.a_above is not None and mode != "none":
            r = (
                ctx.a_above
                if mode == "rows_above"
                else (ctx.a_below if observers_above else ctx.a_above)
            )
            ctx = ctx._replace(a_wire=float(r))
        return _POB(ctx, *a, observers_above=observers_above, **kw)

    return pob


def mw(basis, text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = build_solver(
            parse(text, dialect="nec4"), basis=basis
        ).solver.compute_impedance()
    return complex(z[0] if isinstance(z, tuple) else z)


out, r = sys.argv[1], int(sys.argv[2])
mode = sys.argv[3] if len(sys.argv) > 3 else "none"
_crossing_fill.point_observer_block = patched(mode)
with open(out, "a") as fh:
    for cfg, (a_v, a_b) in CFG.items():
        kw = dict(a_v=a_v, a_b=a_b, n_rad=1)
        recs = {}
        if mode == "none":
            (R, X), _ = run(NEC5, decks.nec5("hub", r, "mid", **kw))
            recs["nec5"] = complex(R, X)
            for b in ("sinusoidal-galerkin", "bspline"):
                recs[b] = mw(b, decks.nec42("hub", r, "mid", **kw))
        recs["sinusoidal" + ("" if mode == "none" else f"[{mode}]")] = mw(
            "sinusoidal", decks.nec42("hub", r, "mid", **kw)
        )
        for e, z in recs.items():
            fh.write(
                json.dumps(dict(cfg=cfg, r=r, engine=e, R=z.real, X=z.imag)) + "\n"
            )
            fh.flush()
