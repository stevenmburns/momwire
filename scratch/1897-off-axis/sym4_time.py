"""momwire#1897: sym4 (antennaknobs verticals.elt_whip:sym4) at 406 MHz, free
space, dense against the sector route. One mode per process so /usr/bin/time
reads that mode s peak. Appends one JSON line to argv[2], flushed.

usage: sym4_time.py {dense,sector} OUT.jsonl"""

import gzip
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from momwire import BSplineSolver

FIX = Path(__file__).resolve().parents[2] / "tests/fixtures/elt_whip_sym4_1897.json.gz"


def solver(rot):
    doc = json.load(gzip.open(FIX, "rt"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BSplineSolver(
            wires=[np.array(w) for w in doc["wires"]],
            n_per_edge_per_wire=doc["n_per_edge_per_wire"],
            wire_radius=doc["wire_radius"],
            feeds=[(w, a, complex(*v)) for w, a, v in doc["feeds"]],
            junctions=[[(w, e) for w, e in j] for j in doc["junctions"]],
            wavelength=doc["wavelength"],
            ground_z=None,
            exact_kernel="auto",
            rotational_symmetry=rot,
        ), doc


def loaded_zin(Y, wavelength, c_farad):
    omega = 2 * math.pi * 299792458.0 / wavelength
    zl = 1.0 / (1j * omega * c_farad)
    Zm = np.linalg.inv(Y)
    n = Y.shape[0] - 1
    return complex(
        Zm[0, 0] - Zm[0, 1:] @ np.linalg.solve(Zm[1:, 1:] + zl * np.eye(n), Zm[1:, 0])
    )


mode = sys.argv[1]
t0 = time.perf_counter()
s, doc = solver(mode == "sector")
t1 = time.perf_counter()
Y = s.compute_y_matrix()
t2 = time.perf_counter()
z = loaded_zin(Y, doc["wavelength"], doc["post_c_farad"])
rec = dict(
    mode=mode,
    build_s=t1 - t0,
    solve_s=t2 - t1,
    z=[z.real, z.imag],
    n_basis=int(s._build_basis_polynomials(s._build_geometry())[0].shape[0]),
)
if s._rotational_map is not None:
    m = s._rotational_map
    rec.update(
        n_sectors=m.n_sectors,
        axial=list(m.axial),
        sector_wires=len(m.sectors[0]),
        reversed=sum(sum(r) for r in m.reversed),
    )
np.save(Path(sys.argv[2]).with_suffix(f".{mode}.Y.npy"), Y)
with open(sys.argv[2], "a") as f:
    f.write(json.dumps(rec) + "\n")
    f.flush()
print(rec, flush=True)
