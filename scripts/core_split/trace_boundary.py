"""Trace every native accelerator call during real solves and record which
calls receive (a view of) another native call's output directly.

Usage: python trace_boundary.py <scenario> <out.json>
"""

import json
import sys
import time
import weakref
from collections import defaultdict

import numpy as np

from momwire import _accel

MODS = []
if _accel.acc is not None:
    MODS.append(("acc", _accel.acc))
for comp in ("_near_interface_accel", "_near_interface_point_accel"):
    try:
        m = _accel.import_companion(comp)
    except Exception:  # noqa: BLE001 - a probe: a missing companion is just absent
        m = None
    if m is not None:
        MODS.append((comp, m))

# registry: id(root array) -> (weakref, producer name)
_out = {}
calls = defaultdict(lambda: {"n": 0, "t": 0.0, "in_bytes": 0, "out_bytes": 0})
edges = defaultdict(lambda: {"n": 0, "bytes": 0})
phase = {"name": "?"}


def _root(a):
    while isinstance(a, np.ndarray) and isinstance(a.base, np.ndarray):
        a = a.base
    return a


def _register(name, res):
    items = res if isinstance(res, tuple) else (res,)
    nb = 0
    for r in items:
        if isinstance(r, np.ndarray):
            nb += r.nbytes
            try:
                _out[id(r)] = (weakref.ref(r), name)
            except TypeError:
                pass
    return nb


def _wrap(name, fn):
    def w(*args, **kw):
        key = (phase["name"], name)
        inb = 0
        for a in list(args) + list(kw.values()):
            if isinstance(a, np.ndarray):
                inb += a.nbytes
                rt = _root(a)
                hit = _out.get(id(rt))
                if hit is not None and hit[0]() is rt:
                    e = edges[(phase["name"], hit[1], name)]
                    e["n"] += 1
                    e["bytes"] += a.nbytes
        t0 = time.perf_counter()
        res = fn(*args, **kw)
        dt = time.perf_counter() - t0
        c = calls[key]
        c["n"] += 1
        c["t"] += dt
        c["in_bytes"] += inb
        c["out_bytes"] += _register(name, res)
        return res

    w.__name__ = name
    w.__wrapped__ = fn
    return w


for tag, m in MODS:
    for name in dir(m):
        if name.startswith("_"):
            continue
        f = getattr(m, name)
        if callable(f) and not isinstance(f, type):
            setattr(m, name, _wrap(f"{tag}.{name}" if tag != "acc" else name, f))


def run(scenario):
    from momwire import BSplineSolver, RazorSolver

    WL = 22.0
    base = dict(
        wires=[[(0.0, -5.0, 6.0), (0.0, 5.0, 6.0)]],
        nsegs=int(__import__("os").environ.get("NSEGS", "108")),
        wire_radius=0.005,
        wavelength=WL,
        feed_arclength=5.0,
    )
    grounds = {
        "free": {},
        "somm": dict(ground_z=0.0, ground_eps=(13.0, 5e-3), ground_model="sommerfeld"),
    }
    family, gname = scenario.split("-")
    kw = dict(base, **grounds[gname])
    if family == "razor2p":
        cls, extra = RazorSolver, dict(nec5_quadrature=True, exact_kernel="auto")
    else:
        cls, extra = BSplineSolver, dict(degree=2)
    walls = {}
    for ph, wl in (("cold", WL), ("warm", WL * 1.01), ("warm2", WL * 1.02)):
        phase["name"] = ph
        t0 = time.perf_counter()
        s = cls(**dict(kw, wavelength=wl), **extra)
        sol = s.compute_port_solution()
        walls[ph] = time.perf_counter() - t0
        _ = np.asarray(sol.y)
    return walls


if __name__ == "__main__":
    scen, outp = sys.argv[1], sys.argv[2]
    walls = run(scen)
    json.dump(
        {
            "scenario": scen,
            "walls": walls,
            "calls": [dict(phase=k[0], fn=k[1], **v) for k, v in calls.items()],
            "edges": [
                dict(phase=k[0], src=k[1], dst=k[2], **v) for k, v in edges.items()
            ],
        },
        open(outp, "w"),
        indent=1,
    )
    print(json.dumps(walls))
