"""Load the sym4 (tests/fixtures) and hex6 (here) decks as SinusoidalSolver
kwargs: the geometry antennaknobs' MomwireEngine builds, free space."""

import gzip
import json
import warnings
from pathlib import Path

import numpy as np

from momwire import SinusoidalSolver

HERE = Path(__file__).resolve().parent
FILES = {
    "sym4": HERE.parents[1] / "tests/fixtures/elt_whip_sym4_1897.json.gz",
    "hex6": HERE / "hex6.json.gz",
}


def load(name):
    return json.load(gzip.open(FILES[name], "rt"))


def solver(name, rot, **kw):
    doc = load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SinusoidalSolver(
            wires=[np.array(w) for w in doc["wires"]],
            n_per_edge_per_wire=doc["n_per_edge_per_wire"],
            wire_radius=doc["wire_radius"],
            feeds=[(w, a, complex(*v)) for w, a, v in doc["feeds"]],
            junctions=[[(w, e) for w, e in j] for j in doc["junctions"]],
            wavelength=doc["wavelength"],
            rotational_symmetry=rot,
            **kw,
        ), doc
