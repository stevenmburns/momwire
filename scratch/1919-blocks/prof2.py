"""momwire#1919: cProfile of harmonic_blocks (new code) on a short sweep."""

import cProfile
import pstats
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "1912-sin-sector"))
from momwire import _rotational_symmetry as rs  # noqa: E402
from deck import solver  # noqa: E402

s, doc = solver(sys.argv[1], True)
ks = 2 * np.pi * (400e6 + 1e6 * np.arange(3)) / s.c
orig = rs.harmonic_blocks
pr = cProfile.Profile()


def hb(*a, **k):
    pr.enable()
    try:
        return orig(*a, **k)
    finally:
        pr.disable()


rs.harmonic_blocks = hb
list(s.compute_y_matrix_swept(ks))
pstats.Stats(pr).sort_stats("tottime").print_stats(12)
