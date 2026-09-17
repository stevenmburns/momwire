"""momwire#1029 phase 1, gate S-C's NUMERIC half: sinusoidal-Galerkin's buried
path is bit-identical across the `rows=` patch.

  # in BOTH trees, then diff the two dumps
  PYTHONPATH=<tree>/src python scratch/1029-block-circulant/s_c_gate.py --out x.json

S-C's static half is an argument about call sites (only `bspline.py` reaches
`compute_Z_operator_buried`, and nothing anywhere passes `rows=`). That
argument is only as good as the reading, so this half measures it instead:
three SG decks that exercise every buried pair class, hashed byte for byte.

  * D1 vertical and D1 horizontal — fully buried, ONE pair class
    (`tests/test_sg_buried_980_d1.py`'s `sg_dipole`);
  * D2 mixed — above x above, below x below and the transmitted pair, none of
    them ending in the plane (`tests/test_sg_mixed_980_d2.py`'s decks).

The dump carries a sha256 of the assembled operator's bytes, of the solved
coefficients' bytes, and Z_in's exact hex. No antennaknobs deck is involved:
SG's own test fixtures are the decks, so this runs on a bare momwire tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

SOIL_A = (13.0, 0.005)
C0 = 299792458.0
WL7 = C0 / 7e6


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


def _sg(wires, npe, feeds):
    from momwire import SinusoidalGalerkinSolver

    return SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=npe,
        feeds=feeds,
        wavelength=WL7,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def buried_dipole(n=21, length=1.0, depth=0.15, vertical=True):
    pts = (
        np.array([(0.0, 0.0, -(depth + length)), (0.0, 0.0, -depth)])
        if vertical
        else np.array([(-0.5 * length, 0.0, -depth), (0.5 * length, 0.0, -depth)])
    )
    fed = (n + 1) // 2
    return _sg([pts], [[n]], [(0, (fed - 0.5) / n * length, 1 + 0j)])


def mixed_deck(n=9):
    above = np.array([(0.0, 0.0, 0.25), (0.0, 0.0, 1.25)])
    below = np.array([(0.0, 0.0, -1.15), (0.0, 0.0, -0.15)])
    return _sg([above, below], [[n], [n]], [(0, 0.5, 1 + 0j)])


DECKS = {
    "d1_vertical": lambda: buried_dipole(vertical=True),
    "d1_horizontal": lambda: buried_dipole(vertical=False),
    "d2_mixed": mixed_deck,
}


def operator_of(s):
    """The assembled operator, through SG's own medium dispatch."""
    geom = s._build_geometry()
    if s._is_mixed(geom):
        return s._assemble_Z(geom, s.k)[0]
    with s._operating_medium(geom):
        return s._assemble_Z(geom, s.k)[0]


def one(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = DECKS[name]()
        G = operator_of(s)
        z_in, coeffs = s.compute_impedance()
    z_in = complex(np.atleast_1d(z_in)[0])
    return {
        "deck": name,
        "n_unknowns": int(np.shape(G)[0]),
        "sha_operator": _sha(G),
        "sha_coeffs": _sha(coeffs),
        "z_in_hex": [z_in.real.hex(), z_in.imag.hex()],
        "z_in": [z_in.real, z_in.imag],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    import momwire

    recs = {
        "_meta": {"gate": "S-C", "momwire": str(Path(momwire.__file__).parent)},
        "decks": [],
    }
    for name in DECKS:
        rec = one(name)
        recs["decks"].append(rec)
        print(json.dumps(rec), flush=True)
    if args.out:
        args.out.write_text(json.dumps(recs, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
