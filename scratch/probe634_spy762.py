"""#762 protocol for momwire#634: bspline's own route must not move.

The change adds two cache attributes to `BSplineSolver.__init__` and a
sub-block overwrite to `HMatrixSolver`'s two image fills. Neither is on the
dense/chunked/swept path, and this records that rather than asserting it:
every array `_near_image_edge_blocks`, `_near_image_analytic_block` and
`_build_J_image_blocks` return on the grazing deck and on an ordinary
grounded one, at both grounds, plus the impedances.

    python scratch/probe634_spy762.py <out.npz>
"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))

from momwire import BSplineSolver
from momwire import bspline as _bs
from test_sommerfeld_ground import _GRAZE_WL, _grazing_wire

REC = {}


def wrap(name):
    real = getattr(BSplineSolver, name)

    def spy(self, *a, **kw):
        out = real(self, *a, **kw)
        tag = f"{name}#{len(REC)}"
        if name == "_near_image_edge_blocks":
            for i, (sl, arc, a_eff) in enumerate(out):
                REC[f"{tag}.{i}.span"] = np.array([sl.start, sl.stop])
                REC[f"{tag}.{i}.arc"] = np.asarray(arc)
                REC[f"{tag}.{i}.a_eff"] = np.array([a_eff])
        else:
            REC[tag] = np.asarray(out)
        return out

    setattr(BSplineSolver, name, spy)


def main(out_path):
    for n in (
        "_near_image_edge_blocks",
        "_near_image_analytic_block",
        "_build_J_image_blocks",
    ):
        wrap(n)
    decks = [
        ("graze", _grazing_wire(1.09e-4 * _GRAZE_WL)),
        ("graze32", _grazing_wire(1.09e-4 * _GRAZE_WL, n_seg=32)),
        (
            "high",
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
    ]
    for name, kw in decks:
        for gname, g in (("pec", {}), ("eps", dict(ground_eps=(10.0, 0.002)))):
            deck = dict(kw)
            deck.update(g)
            z, _ = BSplineSolver(**deck).compute_impedance()
            REC[f"Z:{name}:{gname}"] = np.array([complex(z)])
    assert _bs is not None
    np.savez(out_path, **REC)
    print(f"{len(REC)} arrays -> {out_path}")


main(sys.argv[1])
