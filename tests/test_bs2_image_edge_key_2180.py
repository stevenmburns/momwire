"""bs2's image edge key (momwire#2180): a level edge's own image pairs key
on the edge's h, the integer offset and the edge's distance to its image,
as the direct block's same-edge pairs key on h and the offset.

- A uniform horizontal wire over ground has one image class per offset |s - o|
  (reciprocity folds the sign), where the general key split them by the
  segments' lengths' last bits.
- Edges at different heights, and sloped edges, never take one another's
  classes: the probes that drop the height or take a sloped edge as level
  move Z far past the drift.
- The folded wire's Zin stays inside bs2's registered drift against reuse
  off, on every ground.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _accel, _features

from _reuse_seam import bs2

acc = _accel.acc
pytestmark = [
    pytest.mark.skipif(
        acc is None or not hasattr(acc, "bspline_reuse_set_image_key_probe"),
        reason="needs the C++ accelerator",
    ),
    pytest.mark.filterwarnings("ignore"),
]

DRIFT = _features.family("pair_reuse", "bspline").drift
PEC = {"ground_z": 0.0}
REFL = {"ground_z": 0.0, "ground_eps": (13.0, 0.005)}
SOMMERFELD = {**REFL, "ground_model": "sommerfeld"}
GROUNDS = {"pec": PEC, "refl": REFL, "sommerfeld": SOMMERFELD}
N = 240


def deck_of(wires, n_seg):
    return {
        "wires": [np.asarray(w, dtype=float) for w in wires],
        "n_per_edge_per_wire": [[n] for n in n_seg],
        "wavelength": 21.0,
        "wire_radius": 0.002,
        "feed_wire_index": 0,
    }


def long_wire(height=5.3, n=N):
    """One horizontal wire, its ends not binary fractions, so its segments'
    lengths and centres differ in their last bits."""
    return deck_of([[[0.1, -26.17, height], [0.1, 26.31, height]]], [n])


def two_heights(n=N // 2):
    """Two equal parallel horizontal wires at different heights."""
    return deck_of(
        [
            [[0.0, -13.07, 5.3], [0.0, 13.11, 5.3]],
            [[4.1, -13.07, 7.9], [4.1, 13.11, 7.9]],
        ],
        [n, n],
    )


def sloped(n=N):
    """The long wire with one end raised 2 m."""
    return deck_of([[[0.1, -26.17, 4.3], [0.1, 26.31, 6.3]]], [n])


@pytest.fixture(autouse=True)
def _fresh():
    acc.bspline_reuse_cache_clear()
    yield
    acc.bspline_reuse_set_image_key_probe(0)
    acc.bspline_reuse_cache_clear()


def plan_of(deck, ground=PEC):
    s = bs2(**deck, **ground, pair_reuse=True)
    plan = s._reuse_plan(s._build_geometry())
    assert plan is not None and plan.image, s.pair_reuse_declined
    return plan.stats()


def zin(deck, ground, **kw):
    s = bs2(**deck, **ground, **kw)
    return complex(np.ravel(s.compute_impedance()[0])[0])


def z_of(deck, ground, **kw):
    s = bs2(**deck, **ground, **kw)
    geom = s._build_geometry()
    supp, polys, *_ = s._build_basis_polynomials(geom)
    return np.array(s._compute_Z_operator(geom, supp, polys))


def rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def test_a_level_wire_has_one_image_class_per_offset():
    st = plan_of(long_wire())
    assert st["image_edge_pairs"] == N * N
    assert st["classes_block"][1] == N, st
    # The general key (0.99.0's plan) splits the offsets by the lengths' bits.
    acc.bspline_reuse_cache_clear()
    acc.bspline_reuse_set_image_key_probe(4)
    old = plan_of(long_wire())
    assert old["image_edge_pairs"] == 0
    assert old["classes_block"][1] > 4 * N, old


def test_two_heights_keep_their_own_classes():
    """The two wires' own image pairs are congruent but for the height:
    each wire keeps its own offsets' classes, and a key without the height
    (probe 2) merges them, moving Z far past the drift."""
    n = N // 2
    deck = two_heights()
    st = plan_of(deck)
    assert st["image_edge_pairs"] == 2 * n * n
    # Each wire's own offsets, and its pairs with the other wire's image by
    # the general key.
    acc.bspline_reuse_set_image_key_probe(2)
    acc.bspline_reuse_cache_clear()
    merged = plan_of(deck)
    assert st["classes_block"][1] - merged["classes_block"][1] == n, (st, merged)
    z_bad = z_of(deck, PEC, pair_reuse=True)
    acc.bspline_reuse_set_image_key_probe(0)
    acc.bspline_reuse_cache_clear()
    z_on = z_of(deck, PEC, pair_reuse=True)
    z_off = z_of(deck, PEC, pair_reuse=False)
    assert rel(z_on, z_off) <= 1e-12
    assert rel(z_bad, z_off) > 1e-6


def test_a_sloped_wire_is_not_level():
    """A sloped wire's image pairs of one offset lie at different heights:
    it keeps the general key, and a key that took it as level (probe 3)
    moves Z far past the drift."""
    deck = sloped()
    assert plan_of(deck)["image_edge_pairs"] == 0
    z_off = z_of(deck, PEC, pair_reuse=False)
    z_on = z_of(deck, PEC, pair_reuse=True)
    assert rel(z_on, z_off) <= 1e-12
    acc.bspline_reuse_set_image_key_probe(3)
    acc.bspline_reuse_cache_clear()
    st = plan_of(deck)
    assert st["image_edge_pairs"] == N * N and st["classes_block"][1] == N
    assert rel(z_of(deck, PEC, pair_reuse=True), z_off) > 1e-6


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_the_folded_wire_stays_inside_the_drift(ground):
    deck = long_wire()
    z_on = z_of(deck, GROUNDS[ground], pair_reuse=True)
    z_off = z_of(deck, GROUNDS[ground], pair_reuse=False)
    assert not np.array_equal(z_on, z_off)
    assert rel(z_on, z_off) <= 1e-12
    a = zin(deck, GROUNDS[ground], pair_reuse=True)
    b = zin(deck, GROUNDS[ground], pair_reuse=False)
    assert abs(a - b) / abs(b) <= DRIFT
