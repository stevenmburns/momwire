"""U3 derivation checks on the real constructor:
(a) loaded - unloaded == one full-geometry stencil (crossing, two-radius rod, hub);
(b) the per-medium sub-geometries' stencils, mapped back, sum to the full one
    identically -- so building them is NOT a red control;
(c) the crossing tent's own entries are a nonzero part of the stencil."""

import json

import numpy as np
from common import RazorSolver, crossing_deck, hub_deck

from momwire import _medium_spec, _wire_loading

A = 0.25e-3
DECKS = {
    "crossing": lambda **kw: crossing_deck(1, **kw),
    "rod_rise2": lambda **kw: crossing_deck(2, wire_radius=[A / 2, A], **kw),
    "hub4": lambda **kw: hub_deck(4, **kw),
}
KW = dict(wire_conductivity=3.5e7)


def mk(build, **kw):
    d = {k: v for k, v in build(**kw).items() if k != "junctions"}
    return RazorSolver(**d, nec5_quadrature=True)


for name, build in DECKS.items():
    s0, sl = mk(build), mk(build, **KW)
    g = sl._build_geometry()
    Z0 = s0._assemble_Z(s0._build_geometry(), s0.k)
    ZL = sl._assemble_Z(g, sl.k)
    spec = _wire_loading.loading_for(sl, sl.c * sl.k, g)
    L = np.zeros_like(ZL)
    sl._apply_loading(L, sl._loading_stencil(g), spec)
    err = float(np.max(np.abs((ZL - Z0) - L)))
    # (b) per-medium stencils mapped back through the basis map
    Lm = np.zeros_like(ZL)
    for side in (_medium_spec.ABOVE, _medium_spec.BELOW):
        sub, rows, _chop = sl._medium_geometry(g, side)
        sub = dict(sub, n_segs_total=sub["seg_h"].shape[0])
        st = sl._loading_stencil(sub)
        media = sl._wire_media()
        off = np.asarray(g["seg_offsets"])
        seg_i = np.concatenate(
            [np.arange(off[w], off[w + 1]) for w, m in enumerate(media) if m == side]
        )
        np.add.at(
            Lm,
            (rows[st["rows"]], rows[st["cols"]]),
            spec.z_seg[seg_i[st["seg"]]] * st["vals"],
        )
    tents = [m for m, _ in sl._crossing_tents(g)]
    own = float(np.max(np.abs(L[np.ix_(tents, tents)]))) if tents else 0.0
    print(
        json.dumps(
            dict(
                deck=name,
                tents=tents,
                n=int(ZL.shape[0]),
                max_L=float(np.max(np.abs(L))),
                max_Z0=float(np.max(np.abs(Z0))),
                loaded_minus_unloaded_minus_L=err,
                per_medium_minus_full=float(np.max(np.abs(Lm - L))),
                crossing_tent_diag=own,
            )
        ),
        flush=True,
    )
