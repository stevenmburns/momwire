"""momwire#1688: the sheet census's distinct (rho, z) count in one sort.

`_near_interface._distinct_pairs` used to walk every group in Python with one
`np.unique` per shared height (106 censuses of 5 920 groups on a 4-radial
buried screen at 500 segs/wl: 1.9 s of a 5.3 s razor-2p solve). It now counts
the distinct pairs of the expanded (z, rho) set; the walk stays as the
fallback for a census whose expansion is large, and as the reference here.
"""

import math

import numpy as np

from momwire import _near_interface as ni
from momwire.eznec._shell import render

LAM = 299.8 / 14.0


def _shallow_screen(segs_per_wl):
    """The #1672 shallow geometry (a 5 m vertical, a = 30 mm, over four 5 m
    radials of a = 2 mm sloping from its base to 5 cm down) at a density
    whose crossing fill takes sheet censuses (from about 300 segs/wl)."""

    def n(length):
        return max(1, round(length / LAM * segs_per_wl))

    run = math.hypot(5.0, 0.05)
    lines = ["CM 1688", "CE", f"GW 1,{n(5.0)},0.,0.,0.,0.,0.,5.,.03"]
    for k, (x, y) in enumerate([(5.0, 0.0), (-5.0, 0.0), (0.0, 5.0), (0.0, -5.0)]):
        lines.append(f"GW {k + 2},{n(run)},0.,0.,0.,{x},{y},-0.05,.002")
    lines += ["GE -1", "FR 0,1,0,0,14.", "GN 2,0,0,0,13.,.005"]
    lines += ["EX 0,1,-1,0,1.,0.", "XQ", "EN"]
    return "\n".join(lines) + "\n"


def _random_parts(rng):
    parts = []
    for _ in range(int(rng.integers(1, 4))):
        n_g, k = int(rng.integers(1, 30)), int(rng.integers(1, 5))
        # A small value pool so rows and heights collide, signed zeros included.
        pool = np.array([0.0, -0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        r = rng.choice(pool, size=(n_g, k))
        zg = [rng.choice(pool, size=int(rng.integers(0, 4))) for _ in range(n_g)]
        parts.append((r, zg, None, None))
    return parts


def test_the_sort_counts_what_the_walk_counts():
    rng = np.random.default_rng(1688)
    for _ in range(500):
        parts = _random_parts(rng)
        assert ni._distinct_pairs(parts) == ni._distinct_pairs_loop(parts)


def test_a_large_census_takes_the_walk(monkeypatch):
    rng = np.random.default_rng(7)
    parts = _random_parts(rng)
    monkeypatch.setattr(ni, "_DISTINCT_PAIRS_FLAT_MAX", 0)
    assert ni._distinct_pairs(parts) == ni._distinct_pairs_loop(parts)


def test_the_census_of_a_shallow_screen_is_unchanged(monkeypatch):
    """Every census a razor-2p solve of the #1672 shallow geometry takes
    (46 at 300 segs/wl), counted both ways."""
    seen = []
    orig = ni._distinct_pairs

    def spy(parts):
        n = orig(parts)
        seen.append((n, ni._distinct_pairs_loop(parts)))
        return n

    monkeypatch.setattr(ni, "_distinct_pairs", spy)
    render(_shallow_screen(300), basis="razor-2p")
    assert seen, "the census never ran"
    assert all(a == b for a, b in seen), seen
