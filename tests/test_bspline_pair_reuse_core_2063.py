"""bs2's adapter on the common pair-reuse core (momwire#2063 stage 3a):
`_bspline_pair_reuse_core.h` through its binding, below any fill route.

- THE ExternalClasses SUB-GATE: a plan of given classes (the front end's hint
  as segment classes: each director pair at one spacing a copy of the first)
  gathers exactly the fill's moments copied block for block, so the gather
  and the representatives' grouped calls are wiring, to the bit;
- the census keys what the moments read: congruent pairs share a class, the
  same-edge pairs one sentinel, near pairs their own, and a transpose shares
  its pair's class only at equal radii;
- the declines are the core's sentences, and a class costs its layout;
- the remainder's plan (momwire#2063 step 2b) shares a pair's class with
  its reverse and with its moves along the plane, and reads the height sum.
"""

from __future__ import annotations

import numpy as np
import pytest

import momwire.bspline as bs
from momwire.bspline import BSplineSolver

pytestmark = pytest.mark.skipif(
    bs._acc is None or not hasattr(bs._acc, "bspline_reuse_plan"),
    reason="bs2's pair-reuse census is not built",
)

LAM = 21.1


@pytest.fixture(autouse=True)
def _fresh_plans():
    bs._acc.bspline_reuse_cache_clear()
    yield
    bs._acc.bspline_reuse_cache_clear()


def _yagi(D=8, h=0.0, radius=0.006):
    """Reflector, driver and D identical directors, 0.2 lambda apart."""
    Ld = 0.47 * LAM
    elems = [(1.05 * Ld, 22), (Ld, 21)] + [(0.95 * Ld, 20)] * D
    wires = [
        np.array([[0.2 * LAM * i, -L / 2, h], [0.2 * LAM * i, L / 2, h]])
        for i, (L, _n) in enumerate(elems)
    ]
    return BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[n] for _L, n in elems],
        feeds=[(1, None, 1.0 + 0j)],
        wavelength=LAM,
        wire_radius=radius,
        extended_kernel=True,
    )


def _inputs(s):
    geom = s._build_geometry()
    n = int(geom["n_segs_total"])
    edge = np.empty(n, dtype=np.int32)
    for e, sl in enumerate(s._same_edge_slices(geom)):
        edge[sl] = e
    ek = s._ek_spec(geom)
    return geom, edge, ek


def _plan(s, thresholds=()):
    geom, edge, ek = _inputs(s)
    return bs._acc.bspline_reuse_plan(
        np.ascontiguousarray(geom["seg_l"]),
        np.ascontiguousarray(geom["seg_r"]),
        np.ascontiguousarray(s._seg_radius(geom), dtype=float),
        edge,
        np.asarray(ek.group_i, dtype=np.int64),
        np.asarray(ek.group_j, dtype=np.int64),
        None if ek.b_j is None else np.asarray(ek.b_j, dtype=float),
        list(thresholds),
        2,
    )


def _values(s, plan):
    """Each class's moments from the shipped off-edge fill over the plan's
    groups."""
    geom, _edge, ek = _inputs(s)
    seg_l, seg_r = geom["seg_l"], geom["seg_r"]
    a = s._seg_radius(geom)
    ladder = s._fill_ladder(s.k, seg_l, seg_r, ek)
    out = np.empty((plan.classes, 3, 3), dtype=complex)
    for obs, src, cls, at in plan.groups():
        block = bs._seg_seg_full_moments_offedge(
            seg_l[obs],
            seg_r[obs],
            seg_l[src],
            seg_r[src],
            a[obs],
            s.k,
            2,
            s.n_qp_pair,
            ek=bs._ek_slice(ek, rows=obs, cols=src),
            ladder=ladder,
        )
        out[cls] = np.moveaxis(block.reshape(3, 3, -1)[:, :, at], 2, 0)
    return out


def test_a_plan_of_given_classes_is_the_copy_to_the_bit():
    """The ExternalClasses sub-gate (survey section 5)."""
    s = _yagi()
    geom = s._build_geometry()
    J0 = s._build_J_blocks(geom, s.k)
    sl = s._same_edge_slices(geom)
    N = int(geom["n_segs_total"])
    cls = np.arange(N * N, dtype=np.int64).reshape(N, N)
    ref = J0.copy()
    for gap in range(1, 8):
        for w in range(3, 2 + 8 - gap):
            for a, b, sa, sb in ((w, w + gap, 2, 2 + gap), (w + gap, w, 2 + gap, 2)):
                cls[sl[a], sl[b]] = cls[sl[sa], sl[sb]]
                ref[:, :, sl[a], sl[b]] = J0[:, :, sl[sa], sl[sb]]
    _, inv = np.unique(cls, return_inverse=True)
    plan = bs._acc.bspline_reuse_plan_of_classes(inv.reshape(N, N).astype(np.int32))
    J = np.empty_like(J0)
    plan.gather(_values(s, plan), 0, N, J)
    for e in sl:  # the same-edge blocks are the fill's own, not the core's
        J[:, :, e, e] = J0[:, :, e, e]
    assert plan.classes < N * N
    assert J.tobytes() == ref.tobytes()
    # a window of rows is the same rows of the whole
    W = np.empty((3, 3, 30, N), dtype=complex)
    plan.gather(_values(s, plan), 50, 80, W)
    J2 = np.empty_like(J0)
    plan.gather(_values(s, plan), 0, N, J2)
    assert W.tobytes() == np.ascontiguousarray(J2[:, :, 50:80]).tobytes()


def test_the_census_keys_what_the_moments_read():
    s = _yagi()
    plan = _plan(s)
    st = plan.stats()
    N = st["n_seg"]
    assert st["declined"] is None
    assert st["same_edge_pairs"] == sum(n * n for n in (22, 21) + (20,) * 8)
    assert st["near_pairs"] == 0  # 0.2 lambda apart: nothing near
    # Every off-edge class is evaluated in a group; most pairs are copies.
    far = st["classes"] - 1
    assert far < (N * N - st["same_edge_pairs"]) // 3
    assert st["rep_evals"] >= st["classes"]
    # The values the census shares agree to the key's rounding.
    vals = _values(s, plan)
    geom = s._build_geometry()
    J = np.empty((3, 3, N, N), dtype=complex)
    plan.gather(vals, 0, N, J)
    J0 = s._build_J_blocks(geom, s.k)
    off = np.ones((N, N), dtype=bool)
    for e in s._same_edge_slices(geom):
        off[e, e] = False
    scale = np.max(np.abs(J0[:, :, off]))
    assert np.max(np.abs(J[:, :, off] - J0[:, :, off])) <= 1e-12 * scale


def test_a_transpose_shares_a_class_only_at_equal_radii():
    one = _plan(_yagi(D=2)).stats()
    Ld = 0.47 * LAM
    elems = [(1.05 * Ld, 22), (Ld, 21), (0.95 * Ld, 20), (0.95 * Ld, 20)]
    wires = [
        np.array([[0.2 * LAM * i, -L / 2, 0.0], [0.2 * LAM * i, L / 2, 0.0]])
        for i, (L, _n) in enumerate(elems)
    ]
    s = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[n] for _L, n in elems],
        feeds=[(1, None, 1.0 + 0j)],
        wavelength=LAM,
        wire_radius=[0.006, 0.006, 0.006, 0.007],
        extended_kernel=True,
    )
    two = _plan(s).stats()
    assert two["classes"] > one["classes"]


def test_a_tier_tie_is_its_own_class():
    """A pair whose centre distance sits on a ladder threshold could take the
    other tier from its representative, so it is not shared."""
    s = _yagi(D=2)
    plain = _plan(s).stats()
    bs._acc.bspline_reuse_cache_clear()
    ratio = (0.2 * LAM) / (0.95 * 0.47 * LAM / 20)  # a director pair, aligned
    tied = _plan(s, thresholds=[ratio]).stats()
    assert tied["near_pairs"] > plain["near_pairs"]


def test_the_declines_are_the_cores_sentences():
    s = _yagi(D=2)
    acc = bs._acc
    acc.bspline_reuse_plan_allowance(1 << 16)
    try:
        assert _plan(s).declined == (
            "the kept plan would exceed its allowance of 0.0625 MB"
        )
    finally:
        acc.bspline_reuse_plan_allowance(0)
    acc.bspline_reuse_cache_clear()
    acc.pair_reuse_cap_compression(1000)
    try:
        plan = _plan(s)
        N = plan.stats()["n_seg"]
        assert plan.declined == (
            f"more than {N * N // 1000} classes for {N * N} pairs "
            "(the cap is 1000x compression)"
        )
    finally:
        acc.pair_reuse_cap_compression(-1)
    c = acc.bspline_reuse_cache()
    assert c["declined"] >= 2


def test_a_class_costs_its_layout():
    assert bs._acc.bspline_reuse_kept_bytes_per_class(2) == 9 * 16 + 8 + 1 + 8


def _remainder(s, wires_dz=None):
    """The remainder's plan by plane invariants (momwire#2063 step 2b), one
    edge's segments raised `dz` first when `wires_dz` = (edge, dz)."""
    geom = s._build_geometry()
    sl = s._same_edge_slices(geom)
    seg_l, seg_r = geom["seg_l"].copy(), geom["seg_r"].copy()
    if wires_dz is not None:
        e, dz = wires_dz
        seg_l[sl[e], 2] += dz
        seg_r[sl[e], 2] += dz
    plan = bs._acc.bspline_reuse_remainder_plan(
        seg_l,
        seg_r,
        np.ascontiguousarray(geom["h_per_seg"], dtype=float),
        0.0,
        np.zeros(0, dtype=np.int64),
        2,
    )
    return plan, sl


def test_the_remainder_classes_are_reciprocal_and_read_the_height_sum():
    """Directors 2, 3, 4 at one height: (3, 4) is (2, 3) moved along the
    boom, and each director against itself is the first's; a pair and its
    reverse share their class. One director 1 m higher: its own pairs keep
    no height sum of the others, and (3, 4) is no longer (2, 3)."""
    s = _yagi(D=3, h=15.0)
    plan, sl = _remainder(s)
    e = [np.arange(x.start, x.stop) for x in sl]
    n = 20
    far = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) >= 5
    c = plan.classes_of
    assert np.array_equal(c(0, e[3], e[4]), c(0, e[2], e[3]))
    assert np.array_equal(c(0, e[3], e[3])[far], c(0, e[2], e[2])[far])
    assert np.array_equal(c(0, e[4], e[3]), c(0, e[3], e[4]).T)
    raised, sl = _remainder(s, wires_dz=(4, 1.0))
    r = raised.classes_of
    assert np.array_equal(r(0, e[3], e[3])[far], r(0, e[2], e[2])[far])
    assert not np.any(r(0, e[4], e[4])[far] == r(0, e[2], e[2])[far])
    assert not np.any(r(0, e[3], e[4]) == r(0, e[2], e[3]))


def test_the_wire_map_counts_every_pair_once():
    s = _yagi(D=3)
    plan = _plan(s)
    st = plan.stats()
    N = st["n_seg"]
    sizes = [22, 21, 20, 20, 20]
    wires = np.repeat(np.arange(5, dtype=np.int32), sizes)
    m = plan.ref().wire_map(wires, 5)
    assert m.sum() == N * N
    assert m[:, :, 2].sum() == st["same_edge_pairs"] + st["near_pairs"]
    assert m[:, :, 0].sum() == st["classes"] - 1 - st["near_pairs"]
