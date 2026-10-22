"""Does the point-matched BURIED class respond to radius at all (no
crossing)? A wholly buried 6 m dipole at -0.3 m, 14 MHz, soil 13/0.005,
radius a (1 mm, 0.5 mm), and a one-radius-per-half variant whose LEFT half is
0.5 mm (a radius-step junction in the soil). momwire bases vs NEC-5.
Usage: OUT.jsonl"""

from __future__ import annotations

import json
import sys
import warnings

from ref_ladder import NEC5, run

from momwire.deck import build_solver, parse


def deck(a_l, a_r, n, dialect):
    lines = [
        "CM buried radius probe",
        "CE",
        f"GW 1 {n} -3 0 -0.3 0 0 -0.3 {a_l}",
        f"GW 2 {n} 0 0 -0.3 3 0 -0.3 {a_r}",
    ]
    if dialect == "nec4":
        lines += ["GE -1", "GN 2 0 0 0 13 0.005 NOFILE", "EX 0 2 1 0 1 0"]
    else:
        lines += ["GE -1 0", "GN 0 0 0 0 13 0.005 1.0 0.0 NOFILE", f"EX 0 1 {n} 2 1 0"]
    lines += ["FR 0 1 0 0 14 0", "XQ 0", "EN"]
    return "\n".join(lines) + "\n"


with open(sys.argv[1], "a") as fh:
    for name, (a_l, a_r) in {
        "a1": (1e-3, 1e-3),
        "a05": (5e-4, 5e-4),
        "step": (5e-4, 1e-3),
    }.items():
        n = 20
        (R, X), _ = run(NEC5, deck(a_l, a_r, n, "nec5"))
        recs = {"nec5": complex(R, X)}
        for b in ("sinusoidal", "sinusoidal-galerkin", "bspline"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                z = build_solver(
                    parse(deck(a_l, a_r, n, "nec4"), dialect="nec4"), basis=b
                ).solver.compute_impedance()
            recs[b] = complex(z[0] if isinstance(z, tuple) else z)
        for e, z in recs.items():
            fh.write(json.dumps(dict(cfg=name, engine=e, R=z.real, X=z.imag)) + "\n")
            fh.flush()
