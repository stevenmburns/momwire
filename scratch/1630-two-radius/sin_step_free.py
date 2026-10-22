"""sinusoidal / SG on nec42_buried_step's decks (free and buried, n=20, 40)."""

from __future__ import annotations

import warnings

from nec42_buried_step import deck

from momwire.deck import build_solver, parse

warnings.simplefilter("ignore")
for free in (False, True):
    for n in (20, 40):
        for b in ("sinusoidal", "sinusoidal-galerkin"):
            zs = {}
            for name, ar in {
                "a1": (1e-3, 1e-3),
                "a05": (5e-4, 5e-4),
                "step": (5e-4, 1e-3),
            }.items():
                z = build_solver(
                    parse(deck(*ar, n, "nec42", free), dialect="nec4"), basis=b
                ).solver.compute_impedance()
                zs[name] = complex(z[0] if isinstance(z, tuple) else z)
            c = zs["a1"]
            print(
                "free" if free else "buried",
                n,
                b,
                f"a1 {c:.2f}",
                " ".join(f"{k} {z - c:+.2f}" for k, z in zs.items() if k != "a1"),
                flush=True,
            )
