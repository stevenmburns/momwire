"""bs2's Sommerfeld remainder on a plan of segment pairs keyed by plane
invariants (momwire#2063 step 2b).

The remainder's moments of a segment pair read the grid at each node pair's
horizontal separation and height sum, and project on the two tangents in the
pair's horizontal frame. So a pair's complete invariants under the motions
that keep the plane are the horizontal Gram words of the centre difference
and the two segment vectors, their vertical parts, the height sum and the
two lengths (`remainder_plan_for`, `_bspline_pair_reuse_core.h`); a class is
evaluated once at its representative and every member takes that value,
transposed under reciprocity.

The gates, and why each is not green by construction:

* every agreement proves the class route RAN first (the kernel's process
  counter of member pairs served and the fill's note of the plan), and
  that Z moved: a fill that fell back to the walk would agree to the bit;
* a vertical stack shares classes through the height sum, which the old
  edge-block key (one height per end point) could not see;
* three wrong keys, each in the new route only: no height sum, the tangents
  as they lie instead of in the pair's frame, the height DIFFERENCE (a
  vertical shift keyed alike). Each moves Z by far more than the drift, so
  the agreement gates can see the errors they guard against;
* a class whose representative sits at a grid region break is left to the
  walk (the tie rule), and with the rule off the same deck moves: the rule
  is what keeps a member in its representative's region;
* the raised pairs are their own classes, a deck whose only compression is
  reciprocity declines, a malformed map is refused, and reuse off reads no
  plan.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _accel, _features, _ground_refl
from momwire.bspline import BSplineSolver

from _reuse_seam import bs2

acc = _accel.acc
pytestmark = [
    pytest.mark.skipif(
        acc is None or not hasattr(acc, "bspline_reuse_remainder_plan"),
        reason="needs the C++ accelerator",
    ),
    pytest.mark.filterwarnings("ignore"),
]

DRIFT = _features.family("pair_reuse", "bspline").drift
LAM = 21.0
RADIUS = 0.002
SOMMERFELD = {
    "ground_z": 0.0,
    "ground_eps": (13.0, 0.005),
    "ground_model": "sommerfeld",
}
REMAINDER_NOTE = "the remainder term copied no block between edges"


def deck_of(wires, n_seg):
    return {
        "wires": [np.asarray(w, dtype=float) for w in wires],
        "n_per_edge_per_wire": [[n] for n in n_seg],
        "wavelength": LAM,
        "wire_radius": RADIUS,
        "feed_wire_index": 0,
    }


def row(n_el=5, n_seg=21, height=6.0, length=10.0, step=(3.7, 0.3, 0.0)):
    """Equal horizontal elements along y, element i offset by i * `step`.
    The steps are not binary fractions, so congruent pairs differ in their
    last bits."""
    step = np.asarray(step, dtype=float)
    wires = [
        np.array([[0.0, -length / 2, height], [0.0, length / 2, height]]) + i * step
        for i in range(n_el)
    ]
    return deck_of(wires, [n_seg] * n_el)


def tee():
    """A along x, B beside it (offset 2 m in y), D collinear with A beyond
    its end, all 2 m up, h = 0.25 m: the pairs (A_k, B_k) and
    (A_k, D_{k-14}) have one centre distance (2 m) and one fixed-frame
    tangent, but the separation lies across the tangents for the first and
    along them for the second."""
    L, z = 5.0, 2.0
    return deck_of(
        [
            [[-L / 2, 0.0, z], [L / 2, 0.0, z]],
            [[-L / 2, 2.0, z], [L / 2, 2.0, z]],
            [[L / 2 + 0.5, 0.0, z], [L / 2 + 0.5 + L, 0.0, z]],
        ],
        [20, 20, 20],
    )


@pytest.fixture(autouse=True)
def _small_decks():
    """Serve the reuse on these small decks, from no plan left by another
    test."""
    acc.bspline_reuse_set_small_deck_segs(0)
    acc.bspline_reuse_cache_clear()
    yield
    acc.bspline_reuse_set_small_deck_segs(-1)
    acc.bspline_reuse_set_remainder_key_probe(0)
    acc.bspline_reuse_class_limit(0)
    acc.remainder_set_tie_rel(-1.0)
    acc.bspline_reuse_cache_clear()


def z_of(deck, reuse, **kw):
    s = bs2(**deck, **SOMMERFELD, pair_reuse=reuse, toeplitz=False, **kw)
    geom = s._build_geometry()
    supp, polys, *_ = s._build_basis_polynomials(geom)
    return s, np.array(s._compute_Z_operator(geom, supp, polys))


def zin(deck, reuse):
    s = bs2(**deck, **SOMMERFELD, pair_reuse=reuse, toeplitz=False)
    return complex(np.ravel(s.compute_impedance()[0])[0])


def rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def ran(fn):
    """fn()'s result and the member pairs the kernel's class route served."""
    c0 = acc.remainder_q_reused_pairs()
    out = fn()
    return out, acc.remainder_q_reused_pairs() - c0


def plan_of(s):
    """The remainder plan the solver's last fill read."""
    return s._remainder_reuse_cache[3]


def note_of(s):
    return s._fill_ground_notes["remainder"]


@pytest.mark.parametrize("layout", ["row", "stack"])
def test_the_class_route_ran_and_agrees_with_reuse_off(layout):
    deck = row(step=(0.0, 0.0, 2.3)) if layout == "stack" else row()
    (s_on, z_on), served = ran(lambda: z_of(deck, True))
    assert served > 0
    note = note_of(s_on)
    assert (
        len(note["maps"]) == 1 and note["pairs"] - note["computed"] > note["pairs"] // 2
    )
    _s, z_off = z_of(deck, False)
    assert not np.array_equal(z_on, z_off)
    assert rel(z_on, z_off) <= 1e-10
    a, b = zin(deck, True), zin(deck, False)
    assert abs(a - b) / abs(b) <= DRIFT


def test_a_vertical_stack_shares_classes_through_the_height_sum():
    """Elements 0, 1, 2 of an evenly spaced stack: the pair (0_k, 2_l) has
    the height sum of (1_k, 1_l) and the same horizontal pose, so the two
    share a class, which no key on each height could find. Every pair
    between elements is then served, and the report names no remainder term
    left out."""
    deck = row(n_el=6, step=(0.0, 0.0, 2.0))
    s = bs2(**deck, **SOMMERFELD, pair_reuse=True, toeplitz=False)
    s.compute_impedance()
    plan = plan_of(s)
    n = 21
    e = [np.arange(i * n, (i + 1) * n) for i in range(6)]
    # an element against itself is near within four segments: own classes
    far = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) >= 5
    for (a, b), (c, d) in (((0, 2), (1, 1)), ((1, 5), (3, 3)), ((0, 4), (1, 3))):
        assert np.array_equal(
            plan.classes_of(0, e[a], e[b])[far], plan.classes_of(0, e[c], e[d])[far]
        )
    reuse = s.fill_report()["reuse"]
    rem = reuse["families"]["remainder"]
    assert rem["pairs"] >= 10 * rem["computed"], rem
    assert REMAINDER_NOTE not in (reuse["declined"] or "")
    assert reuse["state"] == "served", reuse["declined"]
    wires = reuse["wires"]
    assert np.array(wires["reused"]).sum() == reuse["reused"]


@pytest.mark.parametrize(
    ("probe", "layout"),
    [(1, "stack"), (2, "tee"), (3, "stack")],
    ids=["no-height-sum", "fixed-frame-tangent", "height-difference"],
)
def test_a_wrong_key_in_the_new_route_moves_z(probe, layout):
    """Each wrong key merges pairs the plane does not carry onto each other:
    without the height sum or on the height difference, the stack's
    elements against themselves at different heights; on the fixed-frame
    tangents, the tee's side-by-side and end-to-end pairs. Z moves by far
    more than the drift, and the reuse-off fill does not move at all."""
    # heights 9, 12.1, 15.2: the elements' own height sums share D's binade,
    # so a key blind to them merges the elements' pairs with themselves
    deck = tee() if layout == "tee" else row(n_el=3, height=9.0, step=(0.0, 0.0, 3.1))
    s_good, z_on = z_of(deck, True)
    _s, z_off = z_of(deck, False)
    assert rel(z_on, z_off) <= 1e-10
    acc.bspline_reuse_set_remainder_key_probe(probe)
    (s_bad, z_bad), served = ran(lambda: z_of(deck, True))
    assert served > 0
    if probe != 2:
        # elements 0 and 1 against themselves, at different heights, merged
        n = 21
        e0, e1 = np.arange(n), np.arange(n, 2 * n)
        far = np.abs(np.subtract.outer(np.arange(n), np.arange(n))) >= 5
        bad = plan_of(s_bad)
        assert np.array_equal(
            bad.classes_of(0, e0, e0)[far], bad.classes_of(0, e1, e1)[far]
        )
        assert not np.any(
            plan_of(s_good).classes_of(0, e0, e0)[far]
            == plan_of(s_good).classes_of(0, e1, e1)[far]
        )
    assert rel(z_bad, z_on) > 1e-8
    _s, z_off_probed = z_of(deck, False)
    assert np.array_equal(z_off_probed, z_off)


def test_the_fixed_frame_probe_merges_the_tee():
    acc.bspline_reuse_set_remainder_key_probe(2)
    s, _z = z_of(tee(), True)
    plan = plan_of(s)
    a, b, d = (np.arange(20), np.arange(20, 40), np.arange(40, 60))
    k = np.arange(14, 20)
    assert np.all(
        plan.classes_of(0, a[k], b[k]).diagonal()
        == plan.classes_of(0, a[k], d[k - 14]).diagonal()
    )
    acc.bspline_reuse_set_remainder_key_probe(0)
    acc.bspline_reuse_cache_clear()
    s, _z = z_of(tee(), True)
    plan = plan_of(s)
    assert np.all(
        plan.classes_of(0, a[k], b[k]).diagonal()
        != plan.classes_of(0, a[k], d[k - 14]).diagonal()
    )


def tie_deck(delta):
    """Three wires along y at height 1.5 m, at x = 0, d0 (1 + delta) and
    that plus d0 (1 - delta), d0 = 3 / tan(30 deg): each segment's middle
    Gauss node faces its neighbour's at rho = d, hh = 3, so the
    representative pair (wire 0, wire 1) reads the grid at theta just under
    the 30-degree split and its member (wire 1, wire 2) just over, a hair of
    the key's quantum apart. The two outer tables meet there at their own
    radial spacings, so their interpolants differ at the split."""
    d0 = 3.0 / np.tan(np.radians(30.0))
    x1 = d0 * (1.0 + delta)
    x2 = x1 + d0 * (1.0 - delta)
    return deck_of(
        [[[x, -2.5, 1.5], [x, 2.5, 1.5]] for x in (0.0, x1, x2)],
        [21, 21, 21],
    )


def test_the_tie_rule_leaves_a_class_at_a_region_break_to_the_walk():
    deck = tie_deck(1e-14)
    _s, z_off = z_of(deck, False)
    (s_on, z_on), served = ran(lambda: z_of(deck, True))
    plan = plan_of(s_on)
    n = 21
    w0, w1, w2 = (np.arange(i * n, (i + 1) * n) for i in range(3))
    # the representative and its member, across the split, in one class
    assert np.array_equal(
        plan.classes_of(0, w0, w1).diagonal(), plan.classes_of(0, w1, w2).diagonal()
    )
    assert note_of(s_on)["maps"][0]["tied"].sum() >= 1
    assert served > 0
    good = rel(z_on, z_off)
    assert good <= 1e-14
    # The rule off: the members across the split take the representative's
    # table's interpolant (measured 4.7e-10 here, against 8.8e-17 with it).
    acc.remainder_set_tie_rel(0.0)
    (s_bad, z_bad), _served = ran(lambda: z_of(deck, True))
    assert note_of(s_bad)["maps"][0]["tied"].sum() == 0
    assert rel(z_bad, z_off) > 1e-11


def test_the_tie_rule_moves_nothing_off_the_breaks():
    """On a deck with no node pair near a break the rule ties no class."""
    s, _z = z_of(row(), True)
    assert note_of(s)["maps"][0]["tied"].sum() == 0


def test_the_raised_pairs_are_their_own_classes():
    """A wire grazing the ground raises its pairs' remainder order
    (momwire#1189): each such pair is a single, so its base-order value is
    the walk's own, which the correction subtracts."""
    deck = row(n_el=3, height=0.05, step=(3.7, 0.0, 0.0))
    s, z_on = z_of(deck, True)
    geom = s._build_geometry()
    I, J, _q = s._remainder_qp_pairs(geom["seg_l"], geom["seg_r"], 0.0)
    assert I.size > 0
    cls, _rep, _code, single = plan_of(s).remainder_arrays()
    c = cls[I, J] >> 1
    assert np.all((single[c // 64] >> (c % 64).astype(np.uint64)) & np.uint64(1))
    _s, z_off = z_of(deck, False)
    assert rel(z_on, z_off) <= 1e-10


def test_a_deck_compressed_only_by_reciprocity_declines():
    """Random wires: the pairs repeat only as their own reverses, which the
    kernel's walk shares already, so the remainder's plan declines at the
    cap on unordered pairs and the report says so; the direct term still
    serves."""
    rng = np.random.default_rng(2063)
    wires = []
    for _ in range(8):
        c = rng.uniform(-10.0, 10.0, 3)
        c[2] = rng.uniform(3.0, 9.0)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        wires.append(np.array([c - 2.0 * d, c + 2.0 * d]))
    deck = deck_of(wires, [15] * 8)
    s = bs2(**deck, **SOMMERFELD, pair_reuse=True, toeplitz=False)
    s.compute_impedance()
    why = s._reuse_remainder_declined
    assert why is not None and "unordered pairs" in why, why
    reuse = s.fill_report()["reuse"]
    assert reuse["families"]["remainder"]["reused"] == 0
    assert f"{REMAINDER_NOTE} ({why})" in reuse["declined"]


def test_a_declined_remainder_census_keeps_the_direct_plan():
    deck = row()
    s, _z = z_of(deck, True)
    classes = plan_of(s).classes
    acc.bspline_reuse_cache_clear()
    acc.bspline_reuse_class_limit(classes - 1)
    s = bs2(**deck, **SOMMERFELD, pair_reuse=True, toeplitz=False)
    s.compute_impedance()
    reuse = s.fill_report()["reuse"]
    # the direct plan has more classes than the remainder's: still served
    assert reuse["families"]["direct"]["reused"] > 0
    assert f"{REMAINDER_NOTE} (too many classes to code)" in reuse["declined"]


def test_a_malformed_class_map_is_refused(monkeypatch):
    """The kernel checks the map's shape: a representative outside its own
    class is an error, not a silent wrong value."""
    deck = row()
    s, _z = z_of(deck, True)
    args = dict(s._remainder_reuse_cache[2])
    rep = np.array(args["reuse_rep"])
    rep[[0, 1]] = rep[[1, 0]]
    args["reuse_rep"] = rep
    monkeypatch.setattr(
        BSplineSolver, "_remainder_reuse_args", lambda self, g, gz: args
    )
    with pytest.raises(ValueError, match="pair reuse"):
        z_of(deck, True)


def test_the_fold_and_the_dense_block_serve_the_same_bits():
    """The fill folds the remainder into Z with no Q held (momwire#1798);
    with the class map it forms the same entries, to the bit, as the dense
    block, and serves the same member pairs."""
    deck = row()
    s = bs2(**deck, **SOMMERFELD, pair_reuse=True, toeplitz=False)
    geom = s._build_geometry()
    supp, polys, *_ = s._build_basis_polynomials(geom)
    eps_t = _ground_refl.eps_tilde(s.ground_eps, s.omega, s.eps)
    Q, dense = ran(lambda: s._Z_sommerfeld_remainder(geom, supp, polys, eps_t))
    nb = Q.shape[0]
    rng = np.random.default_rng(2063)
    Z0 = rng.standard_normal((nb, nb)) + 1j * rng.standard_normal((nb, nb))
    T = np.array(Z0, order="F")
    _r, folded = ran(
        lambda: s._Z_sommerfeld_remainder_into(T, -1, geom, supp, polys, eps_t)
    )
    assert folded == dense > 0
    assert np.ascontiguousarray(T).tobytes() == np.ascontiguousarray(Z0 - Q).tobytes()


def test_reuse_off_reads_no_remainder_plan():
    c0 = acc.bspline_reuse_cache()
    (s, _z), served = ran(lambda: z_of(row(), False))
    assert served == 0
    assert s._remainder_reuse_cache is None or s._remainder_reuse_cache[3] is None
    assert acc.bspline_reuse_cache() == c0
