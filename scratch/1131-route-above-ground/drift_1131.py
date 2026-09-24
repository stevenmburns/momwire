"""momwire#1131 gate (c): decks that never pass `rows=` are bit-identical,
base tree vs this one. Dumps a hash of every number, from WHICHEVER momwire
PYTHONPATH resolves.

  PYTHONPATH=<tree>/src:scratch/1131-route-above-ground:<momwire>/tests \\
      python drift_1131.py > <tree>.json

The change is Python-only, so both trees import the SAME compiled extension
(the base checkout's `src/momwire/*.so` are symlinks to this worktree's); a
mismatch is therefore Python drift and nothing else.

Decks: the N6LF screen solved DENSELY over each ground (default dispatch, and
chunked forced by `swept_mem_mb=1`), with loading; the #1029 buried screen
dense AND on the route (the route's call site moved from
`_compute_Z_operator_buried` to `_compute_Z_operator`); a free-space and a
Sommerfeld dipole.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def full_Z(s):
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    return np.asarray(s._compute_Z_operator(geom, supp_seg, polys))


def record(s, with_Z=True):
    z, c = s.compute_impedance()
    out = {"z": [x.hex() for x in np.atleast_1d(z).view(np.float64)], "c": h(c)}
    if with_Z:
        out["Z"] = h(full_Z(s))
    return out


def main():
    from deck import deck
    from test_rotational_symmetry_1029 import solver as buried

    from momwire.bspline import BSplineSolver

    out = {"momwire": __import__("momwire").__file__}
    for g in ("free", "pec", "refl-coef", "sommerfeld"):
        for mem in (None, 1):
            kw = {} if mem is None else {"swept_mem_mb": mem}
            out[f"screen12_{g}_mem{mem}"] = record(deck(12, ground=g, **kw))
        out[f"screen4_{g}_elev"] = record(deck(4, ground=g, h=0.5))
    out["screen4_refl_loaded"] = record(
        deck(4, ground="refl-coef", wire_conductivity=3.5e7)
    )
    out["buried4_dense"] = record(buried(4, rotational_symmetry=False))
    out["buried4_route"] = record(buried(4), with_Z=False)
    out["buried12_route"] = record(buried(12), with_Z=False)
    for g, extra in (
        ("free", {}),
        (
            "somm",
            dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld"),
        ),
    ):
        for mem in (None, 1):
            kw = {} if mem is None else {"swept_mem_mb": mem}
            s = BSplineSolver(
                wires=[np.array([(0.0, -5.0, 3.0), (0.0, 5.0, 3.0)])],
                n_per_edge_per_wire=[[41]],
                feeds=[(0, 5.0, 1 + 0j)],
                wavelength=21.0,
                wire_radius=0.001,
                degree=2,
                **extra,
                **kw,
            )
            out[f"dipole_{g}_mem{mem}"] = record(s)
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
