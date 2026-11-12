"""momwire#1919: split harmonic_blocks into its parts on a real sweep.
usage: prof.py DECK [--nk 3]   (threads from the environment)"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1912-sin-sector"))
from momwire import _lapack  # noqa: E402
from momwire import _rotational_symmetry as rs  # noqa: E402
from deck import solver  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("deck")
ap.add_argument("--nk", type=int, default=3)
a = ap.parse_args()
T = dict(h0=0.0, gather=0.0, accum=0.0, lu=0.0)


def hb(solver_, Z, sectors, axial, harmonics, row_of=None):
    n = len(sectors)
    out = {}
    t = time.perf_counter()
    K0, _m, _p = rs.harmonic_zero_block(solver_, Z, sectors, axial, row_of=row_of)
    T["h0"] += time.perf_counter() - t
    t = time.perf_counter()
    out[0] = _lapack.lu_factor(K0, overwrite_a=True)
    T["lu"] += time.perf_counter() - t
    s0 = sectors[0]
    r_s0 = s0 if row_of is None else row_of[s0]
    m = int(s0.size)
    hs = [h for h in harmonics if h != 0]
    lams = {h: np.zeros((m, m), dtype=np.complex128, order="F") for h in hs}
    term = np.empty((m, m), dtype=np.complex128, order="F")
    for tt in range(n):
        t = time.perf_counter()
        C_t = Z[np.ix_(r_s0, sectors[tt])]
        T["gather"] += time.perf_counter() - t
        t = time.perf_counter()
        for h in hs:
            np.multiply(np.exp(2j * math.pi * h * tt / n), C_t, out=term)
            lams[h] += term
        T["accum"] += time.perf_counter() - t
    for h in hs:
        t = time.perf_counter()
        out[h] = _lapack.lu_factor(lams.pop(h), overwrite_a=True)
        T["lu"] += time.perf_counter() - t
    return out


rs.harmonic_blocks = hb
s, doc = solver(a.deck, True)
ks = 2 * np.pi * (400e6 + 1e6 * np.arange(a.nk)) / s.c
list(s.compute_y_matrix_swept(ks))
print(a.deck, "nk", a.nk, {k: round(v, 2) for k, v in T.items()}, "total", round(sum(T.values()), 2))
