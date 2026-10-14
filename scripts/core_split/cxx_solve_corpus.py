"""The C++ solve prototype against the Python route, deck by deck.

    python scripts/core_split/cxx_solve_corpus.py OUT.jsonl DECK_OR_DIR...

Every deck is served as `momwire-nec5-razor-2p` serves it
(`momwire.eznec._shell.render(text, basis="razor-2p")`), twice:

1. with `compute_port_solution` instrumented: each call runs the PYTHON
   route first (Z captured as `_bspline._lu_solve` receives it) and then the
   opted-in route on the same solver, and the two are compared byte for byte
   -- Z before the factorisation, the solved coefficients X, and Y. Which
   route the second call took is read off `razor._CXX_ROUTE`, so a deck the
   prototype does not cover is counted as such and never as an agreement;
2. plain, once with the C++ route off and once on, comparing the two
   printouts byte for byte.

One JSON line per deck: the routes taken, the byte comparisons, and the
largest |dZ|, |dX| when anything differs. Threads are whatever
OMP_NUM_THREADS / OPENBLAS_NUM_THREADS say.
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path


def main():
    out_path = sys.argv[1]
    decks = []
    for arg in sys.argv[2:]:
        p = Path(arg)
        decks += sorted(p.rglob("*.nec")) if p.is_dir() else [p]

    import numpy as np

    from momwire import RazorSolver, razor
    from momwire.eznec import _shell

    warnings.simplefilter("ignore")
    razor._CXX_WANT_Z = True
    orig_cps = RazorSolver.compute_port_solution
    orig_lu = razor._bspline._lu_solve
    cap = {}
    per_deck = {}

    def lu(Z, rhs, **k):
        cap["Z"] = np.array(Z, order="F", copy=True)
        return orig_lu(Z, rhs, **k)

    def cps(self, prepared=None):
        razor._RAZOR_CXX = False
        cap.clear()
        a = orig_cps(self, prepared)
        z_py = cap.get("Z")
        razor._RAZOR_CXX = True
        before = dict(razor._CXX_ROUTE)
        self._cxx_z = None
        b = orig_cps(self, prepared)
        route = [r for r in razor._CXX_ROUTE if razor._CXX_ROUTE[r] != before.get(r, 0)]
        rec = {
            "route": route[0] if route else "none",
            "n_basis": int(a.coeffs.shape[0]),
        }
        rec["X_same"] = a.coeffs.tobytes() == b.coeffs.tobytes()
        rec["Y_same"] = a.y.tobytes() == b.y.tobytes()
        z_cx = self._cxx_z
        if rec["route"] == "cxx":
            rec["Z_same"] = z_py is not None and z_py.tobytes() == z_cx.tobytes()
            if not rec["Z_same"]:
                rec["dZ"] = float(np.abs(z_py - z_cx).max())
        if not rec["X_same"]:
            rec["dX"] = float(np.abs(a.coeffs - b.coeffs).max())
        per_deck.setdefault("solves", []).append(rec)
        return b

    razor._bspline._lu_solve = lu
    with open(out_path, "w") as out:
        for deck in decks:
            text = deck.read_text(encoding="latin-1")
            per_deck.clear()
            row = {"deck": str(deck)}
            t = time.perf_counter()
            RazorSolver.compute_port_solution = cps
            try:
                _shell.render(text, basis="razor-2p")
            except Exception as exc:  # noqa: BLE001 -- recorded per deck
                row["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                RazorSolver.compute_port_solution = orig_cps
            row["solves"] = per_deck.get("solves", [])
            razor._RAZOR_CXX = False
            p_py = _shell.render(text, basis="razor-2p")
            razor._RAZOR_CXX = True
            p_cx = _shell.render(text, basis="razor-2p")
            razor._RAZOR_CXX = None
            row["printout_same"] = p_py == p_cx
            row["refused"] = "NEC ERROR" in p_py or not row["solves"]
            row["seconds"] = round(time.perf_counter() - t, 3)
            out.write(json.dumps(row) + "\n")
            out.flush()
    os.environ.pop("MOMWIRE_RAZOR_CXX", None)


if __name__ == "__main__":
    main()
