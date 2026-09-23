"""#1152: how far razor's driving point moves with the near-plane grading on
(vs forced off, which is the pre-#1152 axis bit for bit), per deck, soil A;
plus the axis node count and fill time."""

import json
import time

import numpy as np
from common import RazorSolver, crossing_deck, detached, detached_hub, hub_deck

from momwire import _crossing_fill

ORIG = _crossing_fill.axis_data
A = 0.25e-3


def off(*a, **kw):
    kw["grade_near_plane"] = False
    return ORIG(*a, **kw)


def nj(d):
    return {k: v for k, v in d.items() if k != "junctions"}


def refined(d, m):
    d["n_per_edge_per_wire"] = [[n * m for n in e] for e in d["n_per_edge_per_wire"]]
    return d


CASES = {
    "crossing_x1": lambda: nj(crossing_deck(1)),
    "crossing_x4": lambda: nj(refined(crossing_deck(1), 4)),
    "hub4": lambda: nj(hub_deck(4)),
    "rod_rise2": lambda: nj(crossing_deck(2, wire_radius=[A / 2, A])),
    "detached_0.3": lambda: detached(),
    "detached_0.01_dx0": lambda: detached(gap=0.01, dx=0.0),
    "detached_hub_0.01": lambda: nj(detached_hub(1, gap=0.01)),
    "detached_hub_0.1": lambda: nj(detached_hub(1, gap=0.1)),
}
for name, mk in CASES.items():
    out = {}
    for tag in ("off", "on"):
        _crossing_fill.axis_data = off if tag == "off" else ORIG
        try:
            t0 = time.time()
            s = RazorSolver(**mk(), nec5_quadrature=True)
            z = np.atleast_1d(np.asarray(s.compute_impedance()[0]))
            out[tag] = (z, time.time() - t0)
        finally:
            _crossing_fill.axis_data = ORIG
    dz = float(np.max(np.abs(out["on"][0] - out["off"][0])))
    print(
        json.dumps(
            dict(
                case=name,
                max_dZ=dz,
                z_on=[[c.real, c.imag] for c in out["on"][0]],
                t_off=round(out["off"][1], 2),
                t_on=round(out["on"][1], 2),
            )
        ),
        flush=True,
    )
