"""bs2's `pair_reuse` on the common core (momwire#2063 stage 3; first the
front end's edge-pair hint, momwire#1809).

The census finds which (observer segment, source segment) pairs of the moment
fill share their moments (`_bspline_pair_reuse_core.h`), each class is
evaluated once by the shipped off-edge kernel, and every fill route gathers
its pairs from those values. So:

- off, or a plan that declines, is the fill bit for bit;
- the gather is wiring only (the ExternalClasses sub-gate,
  tests/test_bspline_pair_reuse_core_2063.py);
- the census moves Zin inside the feature's declared drift, and a pair that
  is not a repeat (a director slid 1 um) keys apart;
- a front end's hint is read as on: the census does not read it;
- over Sommerfeld ground the remainder copies the blocks the direct classes
  make congruent through the plane, and none across heights.

These gates are of the census alone, so each passes `offset_reuse=False`,
the part's own keyword, which wins (tests/test_offset_reuse_1809.py holds
the part). Every reuse gate checks the route ran: a counter of plans built
and not declined, and what the fill says it served.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import momwire.bspline as bs
from momwire import _features
from momwire.bspline import BSplineSolver

# An experimental feature (`momwire._features`, reach 4: through the drop-in
# on Linux and Windows).
pytestmark = pytest.mark.reach(4)

LAM = 21.1
DRIFT = _features.FEATURES["pair_reuse"].drift
SOMM = {"ground_z": 0.0, "ground_eps": (13.0, 0.005), "ground_model": "sommerfeld"}


@pytest.fixture(autouse=True)
def _small_decks():
    """These decks are below the small-deck threshold; the census is what is
    under test, so it runs on them."""
    acc = bs._acc
    if acc is None or not hasattr(acc, "bspline_reuse_plan"):
        pytest.skip("bs2's pair-reuse census is not built")
    acc.bspline_reuse_set_small_deck_segs(0)
    acc.bspline_reuse_cache_clear()
    yield
    acc.bspline_reuse_set_small_deck_segs(-1)
    acc.bspline_reuse_cache_clear()


def _served_plans():
    c = bs._acc.bspline_reuse_cache()
    return c["built"] - c["declined"]


def _yagi(D, h=None):
    """Reflector, driver (fed) and D identical directors, 0.2 lambda apart."""
    Ld = 0.47 * LAM
    elems = [(1.05 * Ld, 22), (Ld, 21)] + [(0.95 * Ld, 20)] * D
    wires, n = [], []
    z = 0.0 if h is None else h
    for i, (L, ns) in enumerate(elems):
        x = 0.2 * LAM * i
        wires.append(np.array([[x, -L / 2, z], [x, L / 2, z]]))
        n.append([ns])
    return wires, n


def _director_hint(D):
    """What antennaknobs sends (AK#2165): every director pair at one spacing
    is one pose, and every director's own block is one pose."""
    hint = [[((w, 0), (w, 0)) for w in range(2, 2 + D)]]
    for s in range(1, D):
        hint.append([((w, 0), (w + s, 0)) for w in range(2, 2 + D - s)])
    return hint


def _solver(D, reuse, extended_kernel=False, ground=None, deck=None, **kw):
    wires, n = (
        deck if deck is not None else _yagi(D, h=None if ground is None else 15.0)
    )
    g = {}
    if ground == "refl":
        g = {
            "ground_z": 0.0,
            "ground_eps": complex(13.0, -60 * LAM * 0.005),
            "ground_model": "refl-coef",
        }
    elif ground == "pec":
        g = {"ground_z": 0.0}
    elif ground == "somm":
        g = SOMM
    kw.setdefault("wire_radius", 0.006)
    kw.setdefault("feeds", [(1, None, 1.0 + 0j)])
    return BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=n,
        wavelength=LAM,
        extended_kernel=extended_kernel,
        pair_reuse=reuse,
        offset_reuse=False,
        **g,
        **kw,
    )


def _z(s):
    return np.atleast_1d(np.asarray(s.compute_impedance()[0], dtype=complex))


def _bits(z):
    return np.ascontiguousarray(z).view(np.uint64)


def _rel(z1, z0):
    return float(np.max(np.abs(z1 - z0) / np.abs(z0)))


@pytest.mark.parametrize("ek", [False, True])
def test_off_and_a_declined_plan_are_the_fill_bit_for_bit(ek):
    z0 = _z(_solver(6, None, ek))
    assert np.array_equal(_bits(z0), _bits(_z(_solver(6, False, ek))))
    bs._acc.bspline_reuse_set_small_deck_segs(10_000)
    s = _solver(6, True, ek)
    z1 = _z(s)
    assert s.pair_reuse_declined == "too few segments to repay the plan (N < 10000)"
    assert s.pair_reuse_served is None
    assert np.array_equal(_bits(z0), _bits(z1))


@pytest.mark.parametrize("ek", [False, True])
def test_the_census_moves_zin_inside_the_declared_drift(ek):
    z0 = _z(_solver(8, False, ek))
    before = _served_plans()
    s = _solver(8, True, ek)
    z1 = _z(s)
    assert _served_plans() == before + 1
    served = s.pair_reuse_served
    assert served["route"] == "tensor"
    assert served["copied_pairs"] > served["classes"] > 0
    assert _rel(z1, z0) <= DRIFT


def test_a_hint_is_read_as_on():
    """The census does not read the hint (momwire#2063): a hint, True and
    "verify" fill the same bits."""
    z1 = _z(_solver(8, True))
    for reuse in (_director_hint(8), "verify", [[((0, 0), (1, 0))]]):
        assert np.array_equal(_bits(z1), _bits(_z(_solver(8, reuse))))


def test_each_class_is_evaluated_once(monkeypatch):
    """The off-edge kernel is asked for the plan's groups and nothing more on
    the tensor route: the representatives' calls, not the copies."""
    asked = []
    real = bs._seg_seg_full_moments_offedge

    def spy(seg_l_i, seg_r_i, seg_l_j, seg_r_j, *a, **kw):
        asked.append(len(seg_l_i) * len(seg_l_j))
        return real(seg_l_i, seg_r_i, seg_l_j, seg_r_j, *a, **kw)

    monkeypatch.setattr(bs, "_seg_seg_full_moments_offedge", spy)
    s = _solver(8, True)
    _z(s)
    n = int(s._build_geometry()["n_segs_total"])
    assert sum(asked) == s.pair_reuse_served["kernel_evals"]
    assert sum(asked) < (n * n) // 2
    asked.clear()
    _z(_solver(8, False))
    same = sum(k * k for k in (22, 21) + (20,) * 8)
    assert sum(asked) == n * n - same


def test_a_slid_director_keys_apart():
    """The last director slid 1 um along itself changes a distance at first
    order: its pairs with the others no longer share the unslid classes, and
    the answer still moves only inside the drift."""
    deck = _yagi(4)
    plain = _solver(4, True, deck=deck)
    _z(plain)
    wires, n = deck
    wires = [w.copy() for w in wires]
    wires[5] = wires[5] + np.array([0.0, 1e-6, 0.0])
    slid = _solver(4, True, deck=(wires, n))
    z1 = _z(slid)
    assert slid.pair_reuse_served["classes"] > plain.pair_reuse_served["classes"]
    assert _rel(z1, _z(_solver(4, False, deck=(wires, n)))) <= DRIFT


@pytest.mark.parametrize("ground", ["refl", "pec"])
def test_over_ground_the_image_term_is_filled_as_before(ground):
    z0 = _z(_solver(6, False, ground=ground))
    s = _solver(6, True, ground=ground)
    z1 = _z(s)
    assert s.pair_reuse_served["copied_pairs"] > 0
    assert _rel(z1, z0) <= DRIFT


def test_the_plan_is_the_processes():
    """A second solver of the same mesh finds the plan the first built."""
    c0 = bs._acc.bspline_reuse_cache()
    _z(_solver(6, True))
    _z(_solver(6, True))
    c1 = bs._acc.bspline_reuse_cache()
    assert c1["built"] - c0["built"] == 1
    assert c1["found"] - c0["found"] >= 1


def _random_wires(count=12, n=11):
    rng = np.random.default_rng(2063)
    wires = []
    for _ in range(count):
        a = rng.uniform(-5, 5, 3) + np.array([0, 0, 10.0])
        wires.append(np.array([a, a + rng.uniform(-3, 3, 3)]))
    return wires, [[n]] * count


def test_reciprocity_needs_equal_radii():
    """Random wires repeat nothing but each pair's transpose: at one radius
    every pair between two wires shares a class with its reverse (2x); with a
    radius per wire it shares none, and the census passes the 2x cap."""
    deck = _random_wires()
    N = 12 * 11
    same = 12 * 11 * 11
    s = _solver(0, True, deck=deck, feeds=None, feed_wire_index=0)
    z1 = _z(s)
    served = s.pair_reuse_served
    assert served["copied_pairs"] == served["classes"]
    assert 2 * served["classes"] + served["near_pairs"] == N * N - same
    assert (
        _rel(z1, _z(_solver(0, False, deck=deck, feeds=None, feed_wire_index=0)))
        <= DRIFT
    )
    radii = [0.004 + 1e-4 * i for i in range(12)]
    kw = dict(deck=deck, feeds=None, feed_wire_index=0, wire_radius=radii)
    s = _solver(0, True, **kw)
    z1 = _z(s)
    assert s.pair_reuse_declined == (
        f"more than {N * N // 2} classes for {N * N} pairs (the cap is 2x compression)"
    )
    assert np.array_equal(_bits(_z(_solver(0, False, **kw))), _bits(z1))


def test_the_allowance_declines_in_the_cores_words():
    acc = bs._acc
    acc.bspline_reuse_plan_allowance(1 << 16)
    try:
        s = _solver(6, True)
        z1 = _z(s)
    finally:
        acc.bspline_reuse_plan_allowance(0)
    assert (
        s.pair_reuse_declined == "the kept plan would exceed its allowance of 0.0625 MB"
    )
    assert np.array_equal(_bits(_z(_solver(6, False))), _bits(z1))


def test_the_kept_plan_costs_its_layout():
    """161 B a class at degree 2: 9 complex moments, the group entry and the
    representative's code, a row and a source in its group's lists."""
    assert bs._acc.bspline_reuse_kept_bytes_per_class(2) == 9 * 16 + 8 + 1 + 8
    assert bs._acc.bspline_reuse_kept_bytes_per_class(1) == 4 * 16 + 17


# ---- The Sommerfeld remainder (the fused kernel's symmetric route) ----------


def _acc_or_skip():
    acc = bs._acc
    if acc is None or not hasattr(acc, "remainder_q_reused_pairs"):
        pytest.skip("fused sommerfeld kernel with pair reuse unavailable")
    return acc


def _z_somm(wires, n, reuse):
    s = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=n,
        feeds=[(1, None, 1.0 + 0j)],
        wavelength=LAM,
        wire_radius=0.006,
        pair_reuse=reuse,
        offset_reuse=False,
        **SOMM,
    )
    return s, _z(s)


def test_sommerfeld_off_is_the_fill_bit_for_bit():
    _acc_or_skip()
    wires, n = _yagi(3, h=15.0)
    _, z0 = _z_somm(wires, n, None)
    _, z1 = _z_somm(wires, n, False)
    assert np.array_equal(_bits(z0), _bits(z1))


def test_sommerfeld_copies_the_remainder_blocks():
    """Directors 2, 3, 4 of 20 segments: (3, 3) and (4, 4) are the block of
    (2, 2), and (3, 4) and (4, 3) the blocks of (2, 3) and (3, 2), moved along
    the boom, heights kept: four ordered blocks of 400 copied at least once
    each (a segment under two basis tiles is visited by both), and Zin inside
    the drift."""
    acc = _acc_or_skip()
    wires, n = _yagi(3, h=15.0)
    _, z0 = _z_somm(wires, n, False)
    c0 = acc.remainder_q_reused_pairs()
    s, z1 = _z_somm(wires, n, True)
    copied = acc.remainder_q_reused_pairs() - c0
    src = s._remainder_reuse_args(s._build_geometry(), 0.0)["reuse_src"]
    assert sorted(zip(*np.nonzero(src >= 0), strict=True)) == [
        (3, 3),
        (3, 4),
        (4, 3),
        (4, 4),
    ]
    assert copied >= 4 * 400
    assert _rel(z1, z0) <= DRIFT


def test_sommerfeld_does_not_copy_across_heights():
    """Three identical dipoles stacked vertically repeat in free space (a
    translation in z), so the direct moments are shared, but the remainder
    reads heights, so none of its blocks are."""
    acc = _acc_or_skip()
    L = 0.47 * LAM
    wires = [np.array([[0.0, -L / 2, h], [0.0, L / 2, h]]) for h in (8.0, 16.0, 24.0)]
    n = [[21]] * 3
    _, z0 = _z_somm(wires, n, False)
    c0 = acc.remainder_q_reused_pairs()
    k0 = acc.remainder_q_symmetric_calls()
    s, z1 = _z_somm(wires, n, True)
    assert acc.remainder_q_symmetric_calls() > k0  # the route ran
    assert acc.remainder_q_reused_pairs() == c0
    assert s.pair_reuse_served["copied_pairs"] > 0
    assert _rel(z1, z0) <= DRIFT


def test_sommerfeld_refuses_a_malformed_map(monkeypatch):
    """The kernel checks the map's shape: a source that is itself a copy is
    an error, not a silent wrong block."""
    _acc_or_skip()
    wires, n = _yagi(3, h=15.0)
    E = 5
    src = np.full((E, E), -1, dtype=np.int32)
    src[3, 3] = 2 * E + 2
    src[4, 4] = 3 * E + 3  # (3,3) is a copy, so it cannot be a source
    bad = {
        "reuse_edge_seg0": np.array([0, 22, 43, 63, 83, 103], dtype=np.int64),
        "reuse_seg_edge": np.repeat(np.arange(E, dtype=np.int32), [22, 21, 20, 20, 20]),
        "reuse_src": src,
    }
    monkeypatch.setattr(BSplineSolver, "_remainder_reuse_args", lambda self, g, gz: bad)
    with pytest.raises(ValueError, match="pair reuse"):
        _z_somm(wires, n, False)


def test_sommerfeld_fold_copies_as_the_dense_route_does():
    """The fill folds the remainder into Z with no Q held (momwire#1798),
    and that route takes the copy map too: the same copies, and the same
    bits, as the dense block `Z -= Q` with the same plan."""
    acc = _acc_or_skip()
    if not hasattr(acc, "sommerfeld_remainder_bspline_Q_into"):
        pytest.skip("fused sommerfeld fold kernel unavailable")
    from momwire import _ground_refl

    wires, n = _yagi(3, h=15.0)
    s = _solver(3, True, ground="somm", deck=(wires, n))
    geom = s._build_geometry()
    supp_seg, polys, *_rest = s._build_basis_polynomials(geom)
    eps_t = _ground_refl.eps_tilde(s.ground_eps, s.omega, s.eps)
    c0 = acc.remainder_q_reused_pairs()
    Q = s._Z_sommerfeld_remainder(geom, supp_seg, polys, eps_t)
    dense = acc.remainder_q_reused_pairs() - c0
    nb = Q.shape[0]
    rng = np.random.default_rng(1809)
    Z0 = rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
    T = np.array(Z0, order="F")
    c1 = acc.remainder_q_reused_pairs()
    s._Z_sommerfeld_remainder_into(T, -1, geom, supp_seg, polys, eps_t)
    assert acc.remainder_q_reused_pairs() - c1 == dense >= 4 * 400
    ref = np.ascontiguousarray(Z0 - Q)
    got = np.ascontiguousarray(T)
    if sys.platform != "win32":
        assert got.tobytes() == ref.tobytes()
    else:
        assert np.abs(got - ref).max() <= 1e-13 * np.abs(ref).max()


def test_sommerfeld_cuts_the_map_to_fit_rather_than_dropping_it(monkeypatch):
    """The remainder kernel fills everything when its cache of source blocks
    would pass its budget. The map is cut to fit instead, keeping the source
    pairs with the most copies per byte: with room for one director pair's
    two orders (115,200 bytes) but not two, the cut copies are counted and
    the rest still served."""
    acc = _acc_or_skip()
    wires, n = _yagi(4, h=15.0)
    _, z0 = _z_somm(wires, n, False)
    monkeypatch.setattr(bs, "_REMAINDER_REUSE_CACHE_BYTES", 120_000)
    c0 = acc.remainder_q_reused_pairs()
    s, z1 = _z_somm(wires, n, True)
    copied = acc.remainder_q_reused_pairs() - c0
    assert s.pair_reuse_remainder_cut > 0
    assert copied >= 2 * 400
    assert _rel(z1, z0) <= DRIFT
