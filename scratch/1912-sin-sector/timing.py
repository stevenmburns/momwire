"""momwire#1912: sym4 / hex6 on SinusoidalSolver, route against dense, one
mode per process so the process peak is that mode's.

usage: timing.py DECK {dense,sector} OUT.jsonl [--ground free|sommerfeld]
                 [--nk 13] [--save Y.npy]

Thread counts come from the environment (OMP_NUM_THREADS and
OPENBLAS_NUM_THREADS, set by the caller). Times the 400-412 MHz sweep
(compute_y_matrix_swept, 1 MHz steps) and splits it by timers wrapped
around the three phases: the fill (`_assemble_Z`), the harmonic blocks
(`harmonic_blocks`: gather, DFT accumulation and LU of every block; on the
dense path the LU is inside `_lapack.solve`, so it is counted as solve), and
the solve (`solve_ports` on the route, `_lapack.solve` dense). Appends one
JSON line, flushed."""

import argparse
import json
import os
import resource
import time

import numpy as np

import momwire
from momwire import _lapack
from momwire import _rotational_symmetry as rs
from momwire import sinusoidal as sin_mod
from deck import solver

ap = argparse.ArgumentParser()
ap.add_argument("deck")
ap.add_argument("mode", choices=("dense", "sector"))
ap.add_argument("out")
ap.add_argument("--ground", default="free")
ap.add_argument("--nk", type=int, default=13)
ap.add_argument("--save", default=None)
a = ap.parse_args()

T = {"fill": 0.0, "blocks": 0.0, "solve": 0.0}
CALLS = {"fill": 0, "blocks": 0, "solve": 0, "fill_rows": []}


def timed(name, fn, rows_of=None):
    def w(*args, **kw):
        t = time.perf_counter()
        out = fn(*args, **kw)
        T[name] += time.perf_counter() - t
        CALLS[name] += 1
        if rows_of is not None:
            CALLS["fill_rows"].append(rows_of(out))
        return out

    return w


S = sin_mod.SinusoidalSolver
S._assemble_Z = timed("fill", S._assemble_Z, rows_of=lambda o: int(o[0].shape[0]))
rs.harmonic_blocks = timed("blocks", rs.harmonic_blocks)
rs.solve_ports = timed("solve", rs.solve_ports)
if a.mode == "dense":
    # the dense port solve's one call; the route never reaches it
    sin_mod._lapack.solve = timed("solve", _lapack.solve)

gkw = {}
if a.ground == "sommerfeld":
    gkw = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")
t0 = time.perf_counter()
s, doc = solver(a.deck, a.mode == "sector", **gkw)
t1 = time.perf_counter()
freqs = 400e6 + 1e6 * np.arange(a.nk)
ks = 2 * np.pi * freqs / s.c
Ys = np.array(list(s.compute_y_matrix_swept(ks)))
t2 = time.perf_counter()
if a.save:
    np.save(a.save, Ys)
rec = dict(
    deck=a.deck,
    mode=a.mode,
    ground=a.ground,
    nk=a.nk,
    threads=os.environ.get("OMP_NUM_THREADS"),
    blas_threads=os.environ.get("OPENBLAS_NUM_THREADS"),
    n_segs=int(s._build_geometry()["n_segs"]),
    build_s=t1 - t0,
    sweep_s=t2 - t1,
    fill_s=T["fill"],
    blocks_s=T["blocks"],
    solve_s=T["solve"],
    other_s=(t2 - t1) - sum(T.values()),
    calls={k: v for k, v in CALLS.items() if k != "fill_rows"},
    fill_rows=sorted(set(CALLS["fill_rows"])),
    peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    y00=[complex(Ys[0, 0, 0]).real, complex(Ys[0, 0, 0]).imag],
    momwire=momwire.__file__,
)
with open(a.out, "a") as f:
    f.write(json.dumps(rec) + "\n")
    f.flush()
print(json.dumps(rec))
