"""momwire#1029 phase 1, gate G2: the six refusals, as text.

  PYTHONPATH=<momwire src> python scratch/1029-block-circulant/g2_refusals.py --out g2.json

The DECKS are `tests/test_rotational_symmetry_1029.py`'s, imported rather than
copied: the test pins the sentences and this writes down what they say, and a
second copy of the decks would let the two drift. Nothing here is a gate — the
gate is the test file, which runs on every PR.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))

import test_rotational_symmetry_1029 as T  # noqa: E402


def caught(fn):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn()
    except T.RotationalSymmetryRefused as exc:
        return str(exc)
    return None


def terrain_deck():
    from momwire import BSplineSolver

    class _TerrainSolver(BSplineSolver):
        def _rotational_ground_kind(self):
            return "terrain"

    return _TerrainSolver(
        wires=T.screen(4),
        n_per_edge_per_wire=[[6]] * 4 + [[3], [8]],
        degree=2,
        wavelength=T.WL,
        wire_radius=0.001,
        feeds=[(5, 0.5 * T.MAST / 8, 1 + 0j)],
        rotational_symmetry=True,
        **T.GROUND,
    )


DECKS = {
    "1_sector_length": lambda: T.solver(
        wires=T.screen(4, radial=[T.RADIAL, T.RADIAL, T.RADIAL * 1.01, T.RADIAL])
    ),
    "2_sector_azimuth": lambda: T.solver(
        wires=T.screen(4, azimuths=[0.0, 0.5 * math.pi, math.pi, math.radians(272.5)])
    ),
    "3_off_axis_port": lambda: T.solver(feeds=[(0, 3.0, 1 + 0j)]),
    "4_sector_radius": lambda: T.solver(wire_radius=[0.001, 0.0008] + [0.001] * 4),
    "5_ground_not_axisymmetric": terrain_deck,
    "6_axis_not_parallel_to_z": lambda: T.solver(wires=T.screen(4, tilt=0.04)),
    # out of scope rather than asymmetric — recorded beside the six because
    # they are the other things a user will hit
    "scope_one_radial": lambda: T.solver(wires=T.screen(1)),
    "scope_enrichment": lambda: T.solver(use_singular_enrichment=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = {}
    ok = True
    for name, make in DECKS.items():
        msg = caught(make)
        ok = ok and msg is not None
        out[name] = msg
        print(f"{name}: {msg}")
    # the route's own runtime guard, which no deck can reach at construction
    out["runtime_drive_not_invariant"] = _drive_guard()
    print(f"runtime_drive_not_invariant: {out['runtime_drive_not_invariant']}")
    if args.out:
        args.out.write_text(json.dumps(out, indent=1))
    print("G2:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _drive_guard():
    import numpy as np

    s = T.solver(4)
    geom = s._build_geometry()
    supp_seg, polys, kcl_A, _knots, wbg = s._build_basis_polynomials(geom)
    sectors, axial = s._rotational_dof_groups(wbg, supp_seg.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Z = s._compute_Z_operator_buried(
            geom, supp_seg, polys, rows=s._rotational_rows(geom)
        )
    v = np.zeros(supp_seg.shape[0], dtype=np.complex128)
    v[sectors[1][0]] = 1.0
    try:
        s._rotational_solve(Z, v, kcl_A[:0], sectors, axial)
    except T.RotationalSymmetryRefused as exc:
        return str(exc)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
