"""momwire#1170 stage 0, gates 1 and 2: the bits of everything that must not move.

usage: gates_ab.py <tree> <out.npz>      (run once per tree, then compare)
       gates_ab.py --compare <base.npz> <change.npz>

Every array is the solver's own operator Z (before the KCL solve) or its feed
impedance, on:
  gate 1  decks with NO buried wire, extended_kernel False and True
  gate 2  decks WITH buried wires, extended_kernel False, chunked and dense
"""

import os
import sys
import warnings

import numpy as np

if sys.argv[1] == "--compare":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    assert sorted(a.files) == sorted(b.files), (a.files, b.files)
    same = [k for k in a.files if np.array_equal(a[k], b[k])]
    diff = [k for k in a.files if k not in same]
    for gate in ("g1", "g2"):
        ks = [k for k in a.files if k.startswith(gate)]
        n = sum(a[k].size for k in ks)
        print(
            f"{gate}: {sum(k in same for k in ks)}/{len(ks)} arrays bit-identical, "
            f"{n} complex values"
        )
    for k in diff:
        print("DIFFERS", k, float(np.abs(a[k] - b[k]).max()))
    sys.exit(1 if diff else 0)

tree = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(tree, "src"))
sys.path.insert(0, os.path.join(tree, "tests"))
warnings.simplefilter("ignore")

import momwire  # noqa: E402

assert momwire.__file__.startswith(os.path.join(tree, "src")), momwire.__file__
from momwire import BSplineSolver, HMatrixSolver, RazorSolver  # noqa: E402
from momwire.deck import build_solver, parse  # noqa: E402
from test_buried_serve_553 import SOIL_A, WL7, buried_dipole, served_deck  # noqa: E402
from test_crossing_serve_524 import crossing_deck, hub_deck  # noqa: E402

out = {}


def _first(x):
    return x[0] if isinstance(x, tuple) else x


def record(name, solver):
    geom = solver._build_geometry()
    supp, polys = solver._build_basis_polynomials(geom)[:2]
    out[name + "/Z"] = np.asarray(solver._compute_Z_operator(geom, supp, polys))
    out[name + "/zin"] = np.asarray(solver.compute_impedance()[0])


SOMM = dict(ground_z=0.0, ground_eps=SOIL_A, ground_model="sommerfeld")
REFL = dict(ground_z=0.0, ground_eps=SOIL_A, ground_model="refl-coef")
PEC = dict(ground_z=0.0)
bent = dict(
    wires=[
        np.array([(0.0, 0.0, 6.0), (0.0, 0.0, 11.0)]),
        np.array([(0.0, 0.0, 11.0), (4.0, 0.0, 11.0)]),
    ],
    n_per_edge_per_wire=[[9], [8]],
    junctions=[[(0, "end"), (1, "start")]],
    feeds=[(0, 2.5, 1 + 0j)],
    wavelength=WL7,
)
contact = dict(
    wires=[np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 10.0)])],
    n_per_edge_per_wire=[[15]],
    feeds=[(0, 1.0, 1 + 0j)],
    wavelength=WL7,
    wire_radius=0.02,
)

# ---- gate 1: no buried wire -------------------------------------------------
for ek in (False, True):
    for gname, g in (("free", {}), ("pec", PEC), ("refl", REFL), ("somm", SOMM)):
        record(
            f"g1/bent-{gname}-ek{int(ek)}",
            BSplineSolver(**bent, wire_radius=0.02, extended_kernel=ek, **g),
        )
        record(
            f"g1/bent2r-{gname}-ek{int(ek)}",
            BSplineSolver(**bent, wire_radius=[0.02, 0.005], extended_kernel=ek, **g),
        )
    for gname, g in (("pec", PEC), ("somm", SOMM)):
        record(
            f"g1/contact-{gname}-ek{int(ek)}",
            BSplineSolver(**contact, extended_kernel=ek, **g),
        )
    # the chunked free-space and image fills (a tiny budget forces them)
    record(
        f"g1/bent-somm-chunked-ek{int(ek)}",
        BSplineSolver(
            **bent, wire_radius=0.02, extended_kernel=ek, swept_mem_mb=1, **SOMM
        ),
    )
    h = HMatrixSolver(**bent, wire_radius=0.02, extended_kernel=ek, **PEC)
    out[f"g1/hmatrix-pec-ek{int(ek)}/zin"] = np.asarray(h.compute_impedance()[0])
    r = RazorSolver(**bent, wire_radius=0.02, extended_kernel=ek, **SOMM)
    out[f"g1/razor-somm-ek{int(ek)}/zin"] = np.asarray(r.compute_impedance()[0])

# ---- gate 2: buried wires, reduced kernel ----------------------------------
buried = {
    "served": lambda: served_deck(),
    "served-fat": lambda: BSplineSolver(
        wires=[
            np.array([(0.0, 0.0, 11.0), (0.0, 0.0, 1.0)]),
            np.array([(0.0, 0.0, -0.15), (5.0, 0.0, -0.15)]),
        ],
        n_per_edge_per_wire=[[15], [10]],
        feeds=[(0, 4.0, 1 + 0j)],
        wavelength=WL7,
        wire_radius=[0.02, 0.005],
        **SOMM,
    ),
    "buried-dipole": lambda: _first(buried_dipole()),
    "crossing": lambda: BSplineSolver(**crossing_deck()),
    "hub4": lambda: BSplineSolver(**hub_deck(n_radials=4)),
}
for name, make in buried.items():
    record(f"g2/{name}-chunked", make())
real = BSplineSolver._buried_chunked_serves
BSplineSolver._buried_chunked_serves = property(lambda self: False)
for name, make in buried.items():
    record(f"g2/{name}-dense", make())
BSplineSolver._buried_chunked_serves = real
r = RazorSolver(**hub_deck(n_radials=4))
out["g2/razor-hub4/zin"] = np.asarray(r.compute_impedance()[0])

# the deck front end, kernel explicitly off: a nec4 stake deck
STAKE = (
    "CM stake\nCE\nGW 1 12 0 0 0 0 0 10 0.02\nGW 2 4 0 0 0 0 0 -1 0.02\n"
    "GE -1\nGN 3 0 0 0 13 0.005\nEX 0 1 3 0 1 0\nFR 0 1 0 0 7.0\nXQ\nEN\n"
)
b = build_solver(parse(STAKE, dialect="nec4"), extended_kernel=False)
assert not b.solver.extended_kernel
out["g2/deck-stake-ek-off/y"] = np.asarray(b.solver.compute_port_solution().y)

np.savez(sys.argv[2], **out)
print(len(out), "arrays", sum(v.size for v in out.values()), "values ->", sys.argv[2])
