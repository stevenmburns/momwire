"""momwire#1919: BSplineSolver route decks, Y saved for a base-vs-change
compare. usage: dy_bspline.py OUT_DIR   (run in each clone, same decks).
Decks: the #1897 cage with 1 + 4 ports (every harmonic), free and over
Sommerfeld; the #1029 screen (harmonic 0); the sym4 whip (5 ports)."""

import gzip
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
import test_rotational_off_axis_1897 as t97  # noqa: E402
import test_rotational_symmetry_1029 as t29  # noqa: E402
from momwire import BSplineSolver  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
res = {}
for g in (None, "sommerfeld"):
    s = t97.build(ground=g, feeds=[t97.FEED, *t97.posts()])
    res[f"cage-{g}"] = s.compute_port_solution().y
for nr in (4, 6):
    res[f"screen{nr}"] = t29.solver(nr).compute_y_matrix()
doc = json.load(gzip.open(t97.SYM4, "rt"))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    s = BSplineSolver(
        wires=[np.array(w) for w in doc["wires"]],
        n_per_edge_per_wire=doc["n_per_edge_per_wire"],
        wire_radius=doc["wire_radius"],
        feeds=[(w, a, complex(*v)) for w, a, v in doc["feeds"]],
        junctions=[[(w, e) for w, e in j] for j in doc["junctions"]],
        wavelength=doc["wavelength"],
        ground_z=None,
        exact_kernel="auto",
        rotational_symmetry=True,
    )
res["sym4-bspline"] = s.compute_y_matrix()
for k, v in res.items():
    np.save(out / f"{k}.npy", v)
    print(k, v.shape)
