"""One momwire solve of a decks.py NEC-4.2 deck: basis variant r feed [a_v a_b]."""

from __future__ import annotations

import sys
import time

import decks

from momwire.deck import build_solver, parse

basis, variant, r, feed = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
a_v, a_b = (
    (float(sys.argv[5]), float(sys.argv[6])) if len(sys.argv) > 6 else (0.019, 0.001)
)
model = parse(decks.nec42(variant, r, feed, a_v=a_v, a_b=a_b), dialect="nec4")
t0 = time.perf_counter()
built = build_solver(model, basis=basis)
s = built.solver
z = s.compute_impedance()
z = z[0] if hasattr(z, "__len__") and not isinstance(z, complex) else z
print(
    basis,
    variant,
    r,
    feed,
    a_v,
    a_b,
    "Z",
    z,
    "ek",
    getattr(s, "extended_kernel", None),
    "t",
    round(time.perf_counter() - t0, 1),
    flush=True,
)
