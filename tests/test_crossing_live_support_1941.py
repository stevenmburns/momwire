"""momwire#1941 (buried memory arc U2): bs2's crossing block is held on its
live support and folded into Z there, never as an (n, n) array.

The unrestricted serve used to answer the whole (n_basis, n_basis) block and
the caller composed `Z -= t; Z -= t.T`; `_main_split` held that block and
returned its `c1 *` copy, two (n, n) arrays beside Z. `into=Z` folds the same
entries from a block on (above live rows x below live rows) alone. What is
pinned here is the BITS: every entry sees the same writes in the same order
(`_Support`, `_fold_pair`), so the folded Z is the old composition exactly —
including through several fold slabs (`_Z_SLAB_BYTES` shrunk so the slab
loop runs more than once) and through the ACA branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _crossing_fill as cf
from momwire.bspline import BSplineSolver

from test_crossing_serve_524 import crossing_deck

pytestmark = pytest.mark.filterwarnings(
    "ignore::momwire._crossing_fill.CoarseCrossingNode"
)


def _axes(**override):
    s = BSplineSolver(**crossing_deck(1, **override))
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    ctx = s._crossing_context(geom, supp_seg, polys)
    below = np.asarray(s._below_segments(geom))
    a_idx, b_idx = np.flatnonzero(~below), np.flatnonzero(below)
    return ctx, a_idx, b_idx, cf.axis_data(ctx, a_idx), cf.axis_data(ctx, b_idx)


@pytest.fixture(scope="module")
def axes():
    return _axes()


def _z0(n):
    """A dense, nonzero starting Z, so a fold that touched the wrong entry,
    or the right one twice, cannot hide behind a zero."""
    rng = np.random.default_rng(1941)
    return rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))


def _slabs(monkeypatch, sup):
    """Shrink the fold's slab to three columns of the support's rows, and
    count the slabs, so the test shows the multi-slab path ran."""
    monkeypatch.setattr(cf, "_Z_SLAB_BYTES", 16 * 3 * sup.rows.size)
    seen = []
    orig = cf._fold_sub

    def spy(Z, zr, zc, block):
        w = max(1, cf._Z_SLAB_BYTES // (16 * max(1, zr.size)))
        seen.append(-(-zc.size // w))
        return orig(Z, zr, zc, block)

    monkeypatch.setattr(cf, "_fold_sub", spy)
    return seen


def test_the_live_support_carries_every_nonzero_of_the_block(axes):
    ctx, a_idx, b_idx, ax_a, ax_b = axes
    full = cf.cross_complete_block_split(ctx, a_idx, b_idx, ax_a, ax_b)
    sup = cf._live_support(ax_a, ax_b)
    n = full.shape[0]
    # not vacuous: the support leaves rows and columns out
    assert 0 < sup.rows.size < n and 0 < sup.cols.size < n
    off = full.copy()
    off[np.ix_(sup.rows, sup.cols)] = 0.0
    assert not np.any(off), "the block writes outside its live support"


@pytest.mark.parametrize("aca_guard", [None, 0.0], ids=["shipped", "aca-forced"])
def test_the_fold_is_the_whole_block_composition_to_the_bit(
    axes, monkeypatch, aca_guard
):
    ctx, a_idx, b_idx, ax_a, ax_b = axes
    if aca_guard is not None:
        # the far blocks' `_put` as well as the direct ones' `_sandwich_dense`
        monkeypatch.setattr(cf, "_ACA_COST_GUARD", aca_guard)
    full = cf.cross_complete_block_split(ctx, a_idx, b_idx, ax_a, ax_b)
    want = _z0(full.shape[0])
    got = want.copy()
    want -= full
    want -= full.T
    slabs = _slabs(monkeypatch, cf._live_support(ax_a, ax_b))
    out = cf.cross_complete_block_split(ctx, a_idx, b_idx, ax_a, ax_b, into=got)
    assert out is None
    assert len(slabs) == 2 and min(slabs) > 1, slabs
    assert np.array_equal(got.view(np.uint8), want.view(np.uint8))


def test_the_whole_axis_switch_folds_the_same_way(axes, monkeypatch):
    ctx, a_idx, b_idx, ax_a, ax_b = axes
    monkeypatch.setattr(cf, "_WHOLE_AXIS_NO_ACA", True)
    full = cf.cross_complete_block_split(ctx, a_idx, b_idx, ax_a, ax_b)
    want = _z0(full.shape[0])
    got = want.copy()
    want -= full
    want -= full.T
    cf.cross_complete_block_split(ctx, a_idx, b_idx, ax_a, ax_b, into=got)
    assert np.array_equal(got.view(np.uint8), want.view(np.uint8))


def test_two_radius_folds_its_pair_to_the_bit(monkeypatch):
    a = 0.25e-3
    ctx, a_idx, b_idx, ax_a, ax_b = _axes(wire_radius=[a / 2, a])
    assert ctx.a_below is not None
    t_above, t_below = cf.cross_complete_blocks_two_radius(
        ctx, a_idx, b_idx, ax_a, ax_b
    )
    want = _z0(t_above.shape[0])
    got = want.copy()
    want -= t_above
    want -= t_below.T
    slabs = _slabs(monkeypatch, cf._live_support(ax_a, ax_b))
    out = cf.cross_complete_blocks_two_radius(ctx, a_idx, b_idx, ax_a, ax_b, into=got)
    assert out is None
    assert len(slabs) == 2 and min(slabs) > 1, slabs
    assert np.array_equal(got.view(np.uint8), want.view(np.uint8))


def test_one_answer_at_a_time(axes):
    ctx, a_idx, b_idx, ax_a, ax_b = axes
    Z = np.zeros((ax_a["n_basis"],) * 2, dtype=np.complex128)
    rows = np.arange(2)
    with pytest.raises(ValueError, match="ask for one"):
        cf.cross_complete_block_split(ctx, a_idx, b_idx, ax_a, ax_b, rows=rows, into=Z)
    with pytest.raises(ValueError, match="rows= answers a slice"):
        cf.cross_complete_blocks_two_radius(
            ctx, a_idx, b_idx, ax_a, ax_b, rows=rows, into=Z
        )
