"""U3: loading OFF (and every non-crossing route, loaded or not) is
bit-identical to origin/main. Runs main's razor.py as a sibling module
(`momwire._razor_main_u3tmp`, copied in by the caller and deleted after)."""

import importlib
import json
import sys

import numpy as np
from common import crossing_deck, detached, hub_deck

import momwire.razor as branch

main = importlib.import_module(sys.argv[1])

A = 0.25e-3


def below():
    d = detached()
    d["wires"] = [d["wires"][0]]
    d["n_per_edge_per_wire"] = [d["n_per_edge_per_wire"][0]]
    d["feeds"] = [(0, 1.0, 1 + 0j)]
    return d


def free():
    d = detached(ground=False)
    return d


def nj(d):
    return {k: v for k, v in d.items() if k != "junctions"}


CASES = {
    "crossing": lambda: nj(crossing_deck(1)),
    "hub4": lambda: nj(hub_deck(4)),
    "rod": lambda: nj(crossing_deck(2, wire_radius=[A / 2, A])),
    "detached": detached,
    "detached_loaded": lambda: detached(wire_conductivity=3.5e7),
    "detached_lumped": lambda: detached(lumped_loads=[(1, 2.0, 10 + 0j)]),
    "below_loaded": lambda: {**below(), "wire_conductivity": 3.5e7},
    "free_loaded": lambda: {**free(), "wire_conductivity": 3.5e7},
}
for name, mk in CASES.items():
    out = {}
    for tag, mod in (("main", main), ("branch", branch)):
        s = mod.RazorSolver(**mk(), nec5_quadrature=True)
        g = s._build_geometry()
        out[tag] = s._assemble_Z(g, s.k)
    same = bool(np.array_equal(out["main"], out["branch"]))
    print(
        json.dumps(
            dict(
                case=name,
                bit_identical=same,
                max_diff=float(np.max(np.abs(out["main"] - out["branch"]))),
            )
        ),
        flush=True,
    )
