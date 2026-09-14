"""U9 PR gates G1 / GT2 (PLAN.md): single-node decks' Z, compared bitwise
between two momwire trees.

  PYTHONPATH=<tree>/src python g1_single_node.py run --out F
  python g1_single_node.py compare MAIN.json BRANCH.json

The builders are imported from the tests of the tree under measurement, so each
side is spelled exactly as that tree's own gates spell it. Each record also
says whether the solve filled the below/below grid's low band, which is what
GT2's bit-identical prediction rests on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

import momwire

sys.path.insert(0, str(Path(momwire.__file__).resolve().parents[2] / "tests"))

from momwire import SinusoidalGalerkinSolver, _sommerfeld
from momwire.bspline import BSplineSolver
from test_crossing_serve_524 import (
    _COLLAPSE_N_QP,
    A_WIRE,
    WL7,
    crossing_deck,
    fan_rise_deck,
    hub_deck,
)
from test_crossing_two_radius_u5 import A as U5_A
from test_crossing_two_radius_u5 import _rod
from test_grazing_band_lo_935 import _shape_z
from test_sg_crossing_980_d3 import buried_hub, direct_fan


def rise_eps1():
    """`test_g524_6_rise_deck_eps1_collapse`'s crossing spelling."""
    rise = np.array([(5.0, 0.0, -0.15), (0.0, 0.0, -0.15), (0.0, 0.0, 0.0)])
    mono = np.array([(0.0, 0.0, 10.0), (0.0, 0.0, 0.0)])
    return dict(
        wires=[rise, mono],
        n_per_edge_per_wire=[[10, 2], [15]],
        junctions=[[(0, "end"), (1, "end")]],
        feeds=[(1, 4.3333333333, 1 + 0j)],
        wavelength=WL7,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=(1.0, 0.0),
        ground_model="sommerfeld",
        n_qp_pair=_COLLAPSE_N_QP,
    )


def _z_of(solver):
    return np.atleast_1d(np.asarray(solver.compute_impedance()[0], dtype=complex))


DECKS = {
    "rod_soilA": lambda: _z_of(BSplineSolver(**crossing_deck(1))),
    "rod_eps1": lambda: _z_of(
        BSplineSolver(
            **crossing_deck(1, ground_eps=(1.0, 0.0), n_qp_pair=_COLLAPSE_N_QP)
        )
    ),
    "rise_eps1": lambda: _z_of(BSplineSolver(**rise_eps1())),
    "fan_soilA": lambda: _z_of(BSplineSolver(**fan_rise_deck())),
    "hub_soilA": lambda: _z_of(BSplineSolver(**hub_deck())),
    "u5_rod": lambda: _z_of(_rod(U5_A, U5_A / 2)),
    "sg_direct_fan4": lambda: _z_of(SinusoidalGalerkinSolver(**direct_fan(4))),
    "sg_buried_hub2": lambda: _z_of(SinusoidalGalerkinSolver(**buried_hub(2))),
    "dipole_935_3mm_n41": lambda: np.atleast_1d(complex(_shape_z(0.003, 41))),
}


def _below_grids():
    return [
        g
        for g in _sommerfeld._GRID_CACHE.values()
        if getattr(g, "regime", None) == "below"
    ]


def run():
    rec = dict(mode="run", momwire=momwire.__file__, decks={})
    for name, solve in DECKS.items():
        _sommerfeld._GRID_CACHE.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.time()
            z = solve()
            dt = time.time() - t0
        rec["decks"][name] = dict(
            z=[repr(complex(v)) for v in z],
            seconds=round(dt, 2),
            low_band_filled=any(
                g._regions[i]["filled"] for g in _below_grids() for i in g._band_lo_idx
            ),
        )
        print(name, rec["decks"][name], flush=True)
    return rec


def compare(a_path, b_path):
    a = json.loads(Path(a_path).read_text())["decks"]
    b = json.loads(Path(b_path).read_text())["decks"]
    rows = {}
    for name in a:
        za = np.array([complex(v) for v in a[name]["z"]])
        zb = np.array([complex(v) for v in b[name]["z"]])
        rows[name] = dict(
            bit_identical=a[name]["z"] == b[name]["z"],
            rel=float(np.max(np.abs(za - zb) / np.abs(za))),
            low_band_filled=[a[name]["low_band_filled"], b[name]["low_band_filled"]],
        )
        print(name, rows[name])
    return dict(mode="compare", a=str(a_path), b=str(b_path), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "compare"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rec = run() if args.what == "run" else compare(*args.files)
    args.out.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
