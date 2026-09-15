"""momwire#1064 step 0, geometry only (no fill, no Z): the two decks the re-timing
adds, on whichever momwire PYTHONPATH selects.

  PYTHONPATH=<momwire src> python s0_geometry.py --tree NAME --out F.jsonl

Reads each deck's below/below theta_min on the solver's own nodes and asks the
solver's serve verdict, so the registered expectation (main and the branch serve
both decks; 0.55.0 refuses both) is read before any timing run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import momwire
from momwire import bspline
from momwire.bspline import BSplineSolver

TESTS = Path(momwire.__file__).resolve().parents[2] / "tests"
sys.path.insert(0, str(TESTS))

C0 = 299792458.0
HALF, RAD, FREQ = 2.9557, 5.0e-4, 7.1e6  # `test_grazing_band_lo_935._shape_z`
SOIL_A = (13.0, 0.005)


def dipole_kwargs(depth_m, n=41):
    return dict(
        wires=[np.array([[-HALF, 0.0, -depth_m], [HALF, 0.0, -depth_m]])],
        n_per_edge_per_wire=[[n]],
        wire_radius=RAD,
        wavelength=C0 / FREQ,
        degree=2,
        feed_model="segment",
        feed_wire_index=0,
        feed_arclength=HALF,
        ground_z=0.0,
        ground_eps=SOIL_A,
        ground_model="sommerfeld",
    )


def twonode_kwargs(separation):
    """Main's `two_node_deck(separation)`, spelled from `crossing_deck` so that
    0.55.0 (whose `two_node_deck` takes no separation) builds the same deck."""
    from test_crossing_serve_524 import crossing_deck

    one = crossing_deck()
    shift = np.array([separation, 0.0, 0.0])
    build = dict(one)
    build["wires"] = one["wires"] + [w + shift for w in one["wires"]]
    build["n_per_edge_per_wire"] = one["n_per_edge_per_wire"] + [
        list(e) for e in one["n_per_edge_per_wire"]
    ]
    build["junctions"] = [[(0, "end"), (1, "start")], [(2, "end"), (3, "start")]]
    return build


def same_as_main_two_node_deck(separation):
    """True when this tree's `two_node_deck(separation)` is the deck built above
    (None where that signature does not exist)."""
    import test_crossing_serve_524 as t524

    try:
        theirs = t524.two_node_deck(separation)
    except TypeError:
        return None
    ours = twonode_kwargs(separation)
    if set(theirs) != set(ours):
        return False
    for k in ours:
        a, b = ours[k], theirs[k]
        if k == "wires":
            if len(a) != len(b) or not all(
                np.array_equal(x, y) for x, y in zip(a, b, strict=True)
            ):
                return False
        elif a != b:
            return False
    return True


DECKS = {
    "dipole1mm": lambda: dipole_kwargs(0.001),
    "twonode11": lambda: twonode_kwargs(11.0),
}


def facts(kw):
    s = BSplineSolver(**kw)
    geom = s._build_geometry()
    b_idx = np.nonzero(s._below_segments(geom))[0]
    obs_b = s._buried_nodes(geom, b_idx)[0]
    d_b = float(kw.get("ground_z", 0.0)) - obs_b[:, 2]
    _r1, th = bspline._pair_extents_below(obs_b[:, 0], obs_b[:, 1], d_b)
    return dict(th_min_deg=math.degrees(th), refusal=s.buried_serve_refusal())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from momwire import _sommerfeld_below as below

    with args.out.open("a") as fh:
        for name, make in DECKS.items():
            rec = dict(
                tree=args.tree,
                deck=name,
                momwire=str(Path(momwire.__file__).resolve().parent),
                floor_deg=below._SOMM_BELOW_TH_MIN_DEG,
            )
            try:
                rec.update(facts(make()))
                if name == "twonode11":
                    rec["same_as_two_node_deck"] = same_as_main_two_node_deck(11.0)
            except Exception as err:  # noqa: BLE001 - a record, not a handler
                rec["error"] = f"{type(err).__name__}: {err!s:.300}"
            fh.write(json.dumps(rec) + "\n")
            print(json.dumps(rec)[:400])


if __name__ == "__main__":
    main()
