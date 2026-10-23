"""momwire#1595: the crossing far block's dense-vs-ACA cost rule.

`_dense_beats_aca` counts the column twin's rule sharing. Two properties keep
it honest: with every rho distinct it IS the old guard (no block moves), and
over a vertical on the axis, where rho does not depend on the above node, it
routes the block dense."""

from __future__ import annotations

import numpy as np

from momwire import _crossing_fill as cf


def _axis(xyz):
    return {"nodes": np.asarray(xyz, dtype=float)}


def _radial_nodes(n, length=5.0, angle=0.3):
    t = np.linspace(0.05, length, n)
    return np.stack([t * np.cos(angle), t * np.sin(angle), np.full(n, -0.0254)], 1)


def test_a_vertical_on_the_axis_is_dense():
    m, n = 60, 840
    vert = _axis(np.stack([np.zeros(m), np.zeros(m), np.linspace(0.1, 5.0, m)], 1))
    rad = _axis(_radial_nodes(n))
    iA, iB = np.arange(m), np.arange(n)
    assert m * n > cf._ACA_COST_GUARD * (m + n)  # the old guard sent it to ACA
    assert cf._dense_beats_aca(vert, rad, iA, iB, 0.001)


def test_distinct_rho_is_the_old_guard():
    # an above line off the axis and skewed: every pair its own rho
    rng = np.random.default_rng(1595)
    for m, n in ((60, 840), (40, 70), (97, 97), (200, 31)):
        above = _axis(np.c_[rng.uniform(1, 9, (m, 2)), rng.uniform(0.1, 5, m)])
        below = _axis(np.c_[rng.uniform(-9, -1, (n, 2)), np.full(n, -0.03)])
        old_dense = m * n <= cf._ACA_COST_GUARD * (m + n)
        got = cf._dense_beats_aca(above, below, np.arange(m), np.arange(n), 0.001)
        assert got == old_dense, (m, n)


def test_a_block_past_the_pair_budget_stays_aca():
    m, n = 700, 900
    assert m * n > cf._DENSE_FAR_MAX_PAIRS
    vert = _axis(np.stack([np.zeros(m), np.zeros(m), np.linspace(0.1, 5.0, m)], 1))
    rad = _axis(_radial_nodes(n))
    assert not cf._dense_beats_aca(vert, rad, np.arange(m), np.arange(n), 0.001)
