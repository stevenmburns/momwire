"""G-688 — the crossing fill's admissibility split (momwire#688).

The cross block's (above x below) segment product is partitioned by the
standard box rule: inadmissible near blocks keep the dense graded axes and
the designed direct evaluation; admissible far blocks get the coarser axes
(the banked 2026-08-27 density-ladder combo) and, past the sampling cost
guard, the low-rank ACA pass. The corner term and the by-parts ends ride
the dense axes always.

What these rows pin, and why:

  * the PARTITION is total and corner-adjacent pairs can only land near
    (touching boxes have distance 0, which the admissibility rule
    refuses) — the structural half of "the corner never routes low-rank";
  * the shared designed-tables MEMO is bit-identical to fresh evaluation
    and actually skips re-evaluation — the split's whole economy on
    symmetric screens rides on it (mirrored blocks are the same triples);
  * split vs dense-direct PARITY on the pinned adjudication decks at a
    measured envelope (the #568 lesson: parity before integrated gates —
    an integrated gate cannot see a conditioning defect). Measured
    2026-08-27: g1/fan block-relative 1e-8..1e-9 class; gated 100x wide;
  * the ACA lane ENGAGES on the dense-mesh class (the momwire#674
    convergence decks) and holds parity — measured rank 6, 2.7e-7
    relative, gated at 1e-5;
  * the corner V(a) is evaluated by six_point at _CORNER_RTOL from the
    split path — the routing half of "never low-rank, never coarse".
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _aca, _near_interface
from momwire import _crossing_fill as cf
from momwire.bspline import BSplineSolver

from test_buried_serve_553 import WL7
from test_crossing_serve_524 import A_WIRE, crossing_deck, fan_rise_deck


def _deck_axes(build):
    s = BSplineSolver(**build)
    geom = s._build_geometry()
    below = s._below_segments(geom)
    b_idx = np.nonzero(below)[0]
    a_idx = np.nonzero(~below)[0]
    # The fill takes a context, not the solver (momwire#801); build it the
    # way `_compute_Z_operator_buried` does.
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    ctx = s._crossing_context(geom, supp_seg, polys)
    return s, ctx, a_idx, b_idx


def test_g688_1_partition_is_total_and_corner_adjacent_stays_near():
    """Every (above-seg, below-seg) pair lands in exactly one block, and
    every far block keeps a strictly positive box gap — so the segments
    meeting at the crossing node (gap 0) can only land near/dense. Also
    pins that the split has teeth: most of the fan deck's area is far."""
    _s, ctx, a_idx, b_idx = _deck_axes(fan_rise_deck())
    tree_a, seg_a = cf._axis_segment_tree(ctx.geom, a_idx, cf._CLUSTER_LEAF_SEGS)
    tree_b, seg_b = cf._axis_segment_tree(ctx.geom, b_idx, cf._CLUSTER_LEAF_SEGS)
    far, near = _aca.build_block_tree(tree_a, tree_b, cf._ADM_ETA)
    seen = np.zeros((len(seg_a), len(seg_b)), dtype=int)
    for cs, ct in far + near:
        seen[np.ix_(cs.indices, ct.indices)] += 1
    assert (seen == 1).all()
    for cs, ct in far:
        # member segment boxes sit inside the cluster boxes, so every
        # member pair's gap is >= the cluster gap
        assert _aca.box_distance(cs, ct) > 0.0
    far_area = sum(cs.size * ct.size for cs, ct in far)
    assert far_area / seen.size > 0.5


def test_g688_2_shared_memo_is_bit_identical_and_skips_reevaluation(monkeypatch):
    """The caller-owned memo (designed_tables' cross-call dedup): values
    through the memo are the SAME floats as fresh evaluation, and a
    repeated call evaluates nothing new. Forced onto the numpy walk so
    the evaluation count is a statement about the reference path."""
    monkeypatch.setattr(_near_interface, "_FORCE_NUMPY", True)
    k = 2.0 * np.pi / WL7
    calls = []
    real = _near_interface.six_point

    def counting(eps_t, k2, r, zz, zzp, **kw):
        calls.append((r, zz, zzp))
        return real(eps_t, k2, r, zz, zzp, **kw)

    monkeypatch.setattr(_near_interface, "six_point", counting)
    rho = np.array([0.3, 0.5])
    fresh = _near_interface.designed_tables(4.0 - 0.5j, k, rho, 0.2, -0.15, rtol=1e-8)
    assert len(calls) == 2
    memo = {}
    first = _near_interface.designed_tables(
        4.0 - 0.5j, k, rho, 0.2, -0.15, rtol=1e-8, memo=memo
    )
    assert len(calls) == 4
    again = _near_interface.designed_tables(
        4.0 - 0.5j, k, rho, 0.2, -0.15, rtol=1e-8, memo=memo
    )
    assert len(calls) == 4  # the second memo call evaluated NOTHING
    sub = _near_interface.designed_tables(
        4.0 - 0.5j, k, rho[:1], 0.2, -0.15, rtol=1e-8, memo=memo
    )
    assert len(calls) == 4  # subsets of the memo evaluate nothing either
    for kk in _near_interface.KEYS:
        assert np.array_equal(fresh[kk], first[kk])
        assert np.array_equal(first[kk], again[kk])
        assert np.array_equal(first[kk][:1], sub[kk])


def test_g688_3_corner_routes_direct_at_corner_rtol(monkeypatch):
    """Structural: the split path evaluates the corner V(a) through
    six_point at (a, 0, 0) with _CORNER_RTOL — never through the memo,
    the coarse axes, or a low-rank factor. Kernels are stubbed to zeros
    so this costs nothing and asserts pure routing."""
    _s, ctx, a_idx, b_idx = _deck_axes(fan_rise_deck())
    A = cf.axis_data(ctx, a_idx)
    B = cf.axis_data(ctx, b_idx)

    def zero_tables(eps_t, k2, rho, z, zp, rtol=1e-10, lam_mult=0.0, memo=None):
        shape = np.broadcast(
            np.asarray(rho, float), np.asarray(z, float), np.asarray(zp, float)
        ).shape
        return {kk: np.zeros(shape, dtype=np.complex128) for kk in _near_interface.KEYS}

    monkeypatch.setattr(_near_interface, "designed_tables", zero_tables)
    corner_calls = []

    def spy_six(eps_t, k2, rho, z, zp, rtol=1e-10, **kw):
        corner_calls.append((float(rho), float(z), float(zp), rtol))
        return np.ones(6, dtype=np.complex128)

    monkeypatch.setattr(_near_interface, "six_point", spy_six)
    cf.cross_complete_block_split(ctx, a_idx, b_idx, A, B)
    assert corner_calls == [(A_WIRE, 0.0, 0.0, cf._CORNER_RTOL)]


# ----------------------------------------------------------------------
# The parity gates (the #568 lesson: parity BEFORE integrated)
# ----------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.crossgate
def test_g688_4_split_matches_dense_on_the_adjudication_decks(record_property):
    """Block-level parity, split vs dense-direct, on the pinned g1 and
    fan decks. Measured 2026-08-27 (leaf=3, eta=1): fan 1.4e-8 relative,
    AK 2.5e-11 — the gate holds 1e-6 with ~100x headroom. A miss here
    with the integrated gates green is exactly the conditioning-defect
    signature this row exists to catch."""
    for name, build in (("g1", crossing_deck(1)), ("fan", fan_rise_deck())):
        _s, ctx, a_idx, b_idx = _deck_axes(build)
        A = cf.axis_data(ctx, a_idx)
        B = cf.axis_data(ctx, b_idx)
        dense = cf.cross_complete_block(ctx, A, B)
        split = cf.cross_complete_block_split(ctx, a_idx, b_idx, A, B)
        rel = float(np.abs(split - dense).max() / np.abs(dense).max())
        record_property(f"{name}_block_rel", f"{rel:.3e}")
        assert rel <= 1e-6, (
            f"the split cross block diverges from dense-direct on the "
            f"{name} deck: {rel:.3e} relative (measured 1e-8 class, "
            f"gate 1e-6)"
        )


@pytest.mark.slow
@pytest.mark.crossgate
def test_g688_5_dense_mesh_engages_low_rank_and_holds_parity(
    record_property, monkeypatch
):
    """The dense-mesh fan (the momwire#674 convergence class, the g2-rung
    mesh of test_g524_2) must actually ENGAGE the ACA lane — on the
    catalog-class decks every far block is cheaper direct and the lane
    correctly stays idle, so without this row a broken sampler would sit
    unexercised until someone's convergence sweep. Measured 2026-08-27:
    one engaged block, rank 6, 2.7e-7 relative; gated 1e-5."""
    ranks = []
    real = _aca.aca_partial

    def spy(get_row, get_col, m, n, tol=1e-3, max_rank=None):
        U, V = real(get_row, get_col, m, n, tol=tol, max_rank=max_rank)
        ranks.append(U.shape[1])
        return U, V

    monkeypatch.setattr(_aca, "aca_partial", spy)
    build = fan_rise_deck()
    build["n_per_edge_per_wire"] = [[20, 6] for _ in range(4)] + [[30]]
    _s, ctx, a_idx, b_idx = _deck_axes(build)
    A = cf.axis_data(ctx, a_idx)
    B = cf.axis_data(ctx, b_idx)
    dense = cf.cross_complete_block(ctx, A, B)
    split = cf.cross_complete_block_split(ctx, a_idx, b_idx, A, B)
    rel = float(np.abs(split - dense).max() / np.abs(dense).max())
    record_property("aca_blocks", str(len(ranks) // len(cf._CROSS_KEYS)))
    record_property("rank_max", str(max(ranks) if ranks else 0))
    record_property("block_rel", f"{rel:.3e}")
    assert ranks, "the dense-mesh deck no longer engages the ACA lane"
    assert max(ranks) <= 24, f"ACA rank blew up: {max(ranks)}"
    assert rel <= 1e-5, (
        f"the ACA-engaged split diverges from dense-direct: {rel:.3e} "
        "relative (measured 2.7e-7, gate 1e-5)"
    )
