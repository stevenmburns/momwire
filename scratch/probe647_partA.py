"""momwire#647 Part A: what the fast-route remainder keying moves.

`HMatrixSolver._somm_nodes` took `self.n_qp_sommerfeld` raw where the dense
route keys it on the grazing height (`_remainder_qp`, momwire#510 / #631).
This records both halves of the bargain:

  * the ORDER the two routes choose, per deck, so "keyed" is visible rather
    than asserted;
  * the impedances, so a deck with nothing grazing can be shown BIT-identical
    across the change — `remainder_qp` is a max-with-base, so such a deck
    gets `base` exactly and nothing moves.

    python scratch/probe647_partA.py <out.npz>
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))

from momwire import ArrayBlockSolver, BSplineSolver, HMatrixSolver
from test_sommerfeld_ground import _GRAZE_WL, _grazing_wire

SOMM = dict(ground_eps=(10.0, 0.002), ground_model="sommerfeld")


def decks():
    out = [
        # Nothing grazing: a vertical, and a horizontal run well clear.
        (
            "vert_clear",
            dict(
                wires=[[[0.0, 0.0, 2.0], [0.0, 0.0, 7.0]]],
                n_per_edge_per_wire=[[17]],
                feeds=[(0, 0.15, 1.0 + 0j)],
                junctions=None,
                wire_radius=0.005,
                wavelength=20.0,
                ground_z=0.0,
                **SOMM,
            ),
        ),
        (
            "horiz_high",
            dict(
                wires=[[[0.0, 0.0, 6.0], [12.0, 0.0, 6.0]]],
                n_per_edge_per_wire=[[21]],
                feeds=[(0, 6.0, 1.0 + 0j)],
                junctions=None,
                wire_radius=0.005,
                wavelength=20.0,
                ground_z=0.0,
                **SOMM,
            ),
        ),
    ]
    for hl in (1.09e-4, 1e-3, 1e-2, 1e-1):
        out.append((f"graze_{hl:.0e}", _grazing_wire(hl * _GRAZE_WL, **SOMM)))
    return out


def main(out_path):
    rec = {}
    for name, kw in decks():
        s = HMatrixSolver(**kw)
        ctx = s._context()
        # The order each route chooses, read rather than assumed.
        rec[f"{name}:q_fast"] = np.array([int(s._somm_nodes(ctx)["q"])])
        d = BSplineSolver(**kw)
        g = d._build_geometry()
        rec[f"{name}:q_dense"] = np.array(
            [int(d._remainder_qp(g["seg_l"], g["seg_r"], d.ground_z))]
        )
        rec[f"{name}:base"] = np.array([int(s.n_qp_sommerfeld)])
        for cls in (BSplineSolver, HMatrixSolver, ArrayBlockSolver):
            z, _ = cls(**kw).compute_impedance()
            rec[f"{name}:{cls.__name__}"] = np.array([complex(z)])
        print(
            f"  {name:>12}  base {int(s.n_qp_sommerfeld):>3}  "
            f"q_dense {int(rec[f'{name}:q_dense'][0]):>4}  "
            f"q_fast {int(rec[f'{name}:q_fast'][0]):>4}   "
            f"dense {complex(rec[f'{name}:BSplineSolver'][0]):.6f}"
        )
    np.savez(out_path, **rec)
    print(f"{len(rec)} entries -> {out_path}")


main(sys.argv[1])
