"""momwire#634: what moves, and what must not.

Records the two fast solvers' and the dense route's impedance on a spread of
ordinary decks (no grazing horizontal run) and on the grazing ones, plus
whether each deck has any qualifying near-image edge at all. Run before and
after the change and compare: the ordinary decks must be BIT-identical,
because `_near_image_edge_blocks` is empty there and the overwrite returns
without touching the fill.

    python scratch/probe634_before_after.py <out.npz>
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))

from momwire import ArrayBlockSolver, BSplineSolver, HMatrixSolver
from test_sommerfeld_ground import _GRAZE_WL, _grazing_wire


def decks():
    """(name, kwargs) — ordinary first, then the grazing ladder."""
    out = [
        (
            "dipole_free",
            dict(
                wires=[[[0.0, 0.0, -5.0], [0.0, 0.0, 5.0]]],
                n_per_edge_per_wire=[[21]],
                feeds=[(0, 5.0, 1.0 + 0j)],
                junctions=None,
                wire_radius=0.005,
                wavelength=20.0,
            ),
        ),
        (
            "monopole_pec",
            dict(
                wires=[[[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]],
                n_per_edge_per_wire=[[17]],
                feeds=[(0, 0.15, 1.0 + 0j)],
                junctions=None,
                wire_radius=0.005,
                wavelength=20.0,
                ground_z=0.0,
            ),
        ),
        (
            "monopole_eps",
            dict(
                wires=[[[0.0, 0.0, 2.0], [0.0, 0.0, 7.0]]],
                n_per_edge_per_wire=[[17]],
                feeds=[(0, 0.15, 1.0 + 0j)],
                junctions=None,
                wire_radius=0.005,
                wavelength=20.0,
                ground_z=0.0,
                ground_eps=(10.0, 0.002),
            ),
        ),
        (
            "horiz_high_pec",
            dict(
                wires=[[[0.0, 0.0, 6.0], [12.0, 0.0, 6.0]]],
                n_per_edge_per_wire=[[21]],
                feeds=[(0, 6.0, 1.0 + 0j)],
                junctions=None,
                wire_radius=0.005,
                wavelength=20.0,
                ground_z=0.0,
            ),
        ),
        (
            "horiz_high_eps",
            dict(
                wires=[[[0.0, 0.0, 6.0], [12.0, 0.0, 6.0]]],
                n_per_edge_per_wire=[[21]],
                feeds=[(0, 6.0, 1.0 + 0j)],
                junctions=None,
                wire_radius=0.005,
                wavelength=20.0,
                ground_z=0.0,
                ground_eps=(10.0, 0.002),
            ),
        ),
    ]
    for hl in (1.09e-4, 1e-3):
        for n in (16, 32):
            out.append(
                (f"graze_pec_{hl:.0e}_{n}", _grazing_wire(hl * _GRAZE_WL, n_seg=n))
            )
            out.append(
                (
                    f"graze_eps_{hl:.0e}_{n}",
                    _grazing_wire(hl * _GRAZE_WL, n_seg=n, ground_eps=(10.0, 0.002)),
                )
            )
    return out


def main(out_path):
    rec = {}
    for name, kw in decks():
        s = BSplineSolver(**kw)
        geom = s._build_geometry()
        # `_near_image_edge_blocks` reads `ground_z`, so it is only asked
        # of a grounded deck — the same condition its callers stand behind.
        n_edges = len(s._near_image_edge_blocks(geom)) if s.ground_z is not None else -1
        rec[f"{name}:qualifying_edges"] = np.array([n_edges])
        for cls in (BSplineSolver, HMatrixSolver, ArrayBlockSolver):
            z, _ = cls(**kw).compute_impedance()
            rec[f"{name}:{cls.__name__}"] = np.array([complex(z)])
        print(
            f"  {name:>22}  near-image edges {n_edges}  "
            f"dense {complex(rec[f'{name}:BSplineSolver'][0]):.6f}"
        )
    np.savez(out_path, **rec)
    print(f"{len(rec)} entries -> {out_path}")


main(sys.argv[1])
