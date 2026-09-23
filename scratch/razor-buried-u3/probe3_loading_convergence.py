"""U3: the loading SHIFT (loaded - unloaded driving point) razor vs bspline,
every edge refined, feed on a knot at every rung; plus red controls.

Controls (monkeypatched `_loading_stencil`, razor only):
  omit_tent : drop every stencil entry whose row or column is a crossing tent
  above_only: drop every stencil entry on a BELOW segment
"""

import json
import sys
import time

import numpy as np
from common import BSplineSolver, RazorSolver, crossing_deck, hub_deck

A = 0.25e-3
SIGMA = float(sys.argv[2]) if len(sys.argv) > 2 else 3.5e7


def refined(d, m):
    d["n_per_edge_per_wire"] = [[n * m for n in e] for e in d["n_per_edge_per_wire"]]
    return d


def crossing(m, **kw):
    d = crossing_deck(1, **kw)
    d["feeds"] = [(1, 4.5, 1 + 0j)]
    return refined(d, m)


def hub(m, **kw):
    d = hub_deck(**kw)
    d["feeds"] = [(len(d["wires"]) - 1, 4.0, 1 + 0j)]
    return refined(d, m)


def rod(m, **kw):
    d = crossing_deck(2, wire_radius=[A / 2, A], **kw)
    d["feeds"] = [(1, 4.5, 1 + 0j)]
    return refined(d, m)


DECKS = {"crossing": crossing, "hub4": hub, "rod_rise2": rod}

ORIG = RazorSolver._loading_stencil


def control(name):
    def st(self, geom):
        out = ORIG(self, geom)
        if "wing_rise" not in geom or "seg_offsets" not in geom:
            return out
        if name == "omit_tent":
            tents = np.array([m for m, _ in self._crossing_tents(geom)])
            keep = ~(np.isin(out["rows"], tents) | np.isin(out["cols"], tents))
        else:
            media = self._wire_media()
            off = np.asarray(geom["seg_offsets"])
            below = np.concatenate(
                [
                    np.arange(off[w], off[w + 1])
                    for w, m in enumerate(media)
                    if m == "below"
                ]
            )
            keep = ~np.isin(out["seg"], below)
        return {k: v[keep] for k, v in out.items()}

    return st


def z(cls, d, **kw):
    if cls is RazorSolver:
        d = {k: v for k, v in d.items() if k != "junctions"}
        kw = dict(kw, nec5_quadrature=True)
    return complex(cls(**d, **kw).compute_impedance()[0])


deck = sys.argv[1]
mk = DECKS[deck]
L = dict(wire_conductivity=SIGMA)
for m in (1, 2, 4, 8):
    t0 = time.time()
    rec = dict(deck=deck, m=m, sigma=SIGMA)
    zr0, zrl = z(RazorSolver, mk(m)), z(RazorSolver, mk(m, **L))
    zb0, zbl = z(BSplineSolver, mk(m)), z(BSplineSolver, mk(m, **L))
    rec["razor_shift"] = [(zrl - zr0).real, (zrl - zr0).imag]
    rec["bspline_shift"] = [(zbl - zb0).real, (zbl - zb0).imag]
    rec["gap"] = abs((zrl - zr0) - (zbl - zb0))
    for c in ("omit_tent", "above_only"):
        RazorSolver._loading_stencil = control(c)
        try:
            zc = z(RazorSolver, mk(m, **L))
        finally:
            RazorSolver._loading_stencil = ORIG
        rec[f"gap_{c}"] = abs((zc - zr0) - (zbl - zb0))
    rec["zr0"] = [zr0.real, zr0.imag]
    rec["zb0"] = [zb0.real, zb0.imag]
    rec["t"] = round(time.time() - t0, 1)
    print(json.dumps(rec), flush=True)
