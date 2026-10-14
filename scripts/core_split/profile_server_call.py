"""Where one `momwire-nec5-razor-2p` call's time goes, on a tiny deck.

    python scripts/core_split/profile_server_call.py OUT.json DECK [--reps R]

Two measurements:

* **client wall**: the installed thin client (`momwire-nec5-razor-2p DECK
  OUT`) against the resident server it spawns on first use, timed end to
  end after warm-up; beside it, a bare `python -S -c pass` and `python -c
  "import momwire_eznec_client"` (the launcher's own interpreter start).
* **in-server**: the server's body, `momwire.eznec._shell.render(text,
  basis="razor-2p")`, timed in this process with its phases wrapped: deck
  parse (`parse_nec5_shared`), `_serve.serve` (of which solver construction,
  `compute_port_solution` and the C++ call when routed), and the printout
  render (`render_printout`).

IPC + server dispatch = client wall - launcher start - in-server render.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time


def _median_wall(cmd, reps, env=None):
    walls = []
    for _ in range(reps):
        t = time.perf_counter()
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
        walls.append(time.perf_counter() - t)
    return statistics.median(walls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("deck")
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()

    bindir = os.path.dirname(sys.executable)
    client = os.path.join(bindir, "momwire-nec5-razor-2p")
    res = {
        "env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MOMWIRE_RAZOR_CXX")}
    }
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out.txt")
        for _ in range(3):  # spawn + warm the server
            subprocess.run([client, args.deck, out], check=True)
        res["client_wall"] = _median_wall([client, args.deck, out], args.reps)
    res["python_bare"] = _median_wall([sys.executable, "-S", "-c", "pass"], args.reps)
    res["python_client_import"] = _median_wall(
        [sys.executable, "-c", "import momwire_eznec_client"], args.reps
    )

    from momwire import RazorSolver
    from momwire.eznec import _shell

    rec = {}

    def wrap(obj, name, key):
        orig = getattr(obj, name)

        def w(*a, **k):
            t = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                rec[key] = rec.get(key, 0.0) + time.perf_counter() - t

        setattr(obj, name, w)

    wrap(_shell, "parse_nec5_shared", "parse")
    wrap(_shell._serve, "serve", "serve")
    wrap(_shell._printout, "render_printout", "render")
    wrap(RazorSolver, "__init__", "solver_init")
    wrap(RazorSolver, "compute_port_solution", "solve")
    if hasattr(RazorSolver, "_cxx_port_solution"):
        wrap(RazorSolver, "_cxx_port_solution", "cxx_solve")
    with open(args.deck, encoding="latin-1") as fh:
        text = fh.read()
    rows = []
    for r in range(args.reps + 5):
        rec.clear()
        t = time.perf_counter()
        _shell.render(text, basis="razor-2p")
        rec["total"] = time.perf_counter() - t
        if r >= 5:
            rows.append(dict(rec))
    res["in_server"] = {
        k: statistics.median(row.get(k, 0.0) for row in rows) for k in rows[0]
    }
    print(json.dumps(res, indent=1))
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
