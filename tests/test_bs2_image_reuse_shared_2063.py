"""bs2's ground image term on the shared reuse core (momwire#2063 step 2a).

bs2's image moment of observer o against source s is the off-edge kernel on
o against the MIRRORED segment M(s), on the direct fill's ladder and EK
labels. So the direct key's rigid invariants of the pair (o, M(s)) are its
complete ones, and the image is block 1 of the direct family's class space
(razor2p's n_blocks = 2): an image pair may take a direct pair's value. The
value is the plane-blind moments; every plane-dependent factor (PEC's -Mt
tangent dot and the -1, refl-coef's Fresnel dyad, the Sommerfeld image's C2)
is the pair's own weight, applied at assembly from the class values by the
weighted tiled assembler (`assemble_Z_bspline_weighted_windowed_tiles`).

The gates, and why each is not green by construction:

* every agreement proves the plan's image route RAN first (its fill counter
  and the weighted tiled assembler's), and that Z moved: a fill that fell
  back to the sweep would agree to the bit and measure nothing;
* a vertical stack shares image classes between edges, the case the old
  edge-block key (momwire#2074's branch) could not find;
* three wrong-sign probes, each in the new route only: the image added at +1,
  weighted with the unmirrored tangent, or keyed on the unmirrored
  direction. Each moves Z by far more than the drift, so the agreement
  gates can see the errors they guard against;
* over refl-coef, a tilted source and a horizontal one whose image pairs are
  congruent share their classes, yet their weights differ at O(1): the
  agreement there shows the weights stayed per pair;
* the direct family's bits do not move with the image block beside it, and
  every route that reads no plan declines as before.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import _accel, _features, _potential_ground
from momwire import bspline as bspline_mod
from momwire.bspline import BSplineSolver

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
LAM = 21.0
RADIUS = 0.002
PEC = {"ground_z": 0.0}
REFL = {"ground_z": 0.0, "ground_eps": (13.0, 0.005)}
SOMMERFELD = {**REFL, "ground_model": "sommerfeld"}
GROUNDS = {"pec": PEC, "refl": REFL, "sommerfeld": SOMMERFELD}
IMAGE_NOTE = "the image term copied no block between edges"


def deck_of(wires, n_seg):
    return {
        "wires": [np.asarray(w, dtype=float) for w in wires],
        "n_per_edge_per_wire": [[n] for n in n_seg],
        "wavelength": LAM,
        "wire_radius": RADIUS,
        "feed_wire_index": 0,
    }


def row(n_el=5, n_seg=21, height=6.0, length=10.0, step=(3.7, 0.3, 0.0)):
    """Equal horizontal elements, element i offset by i * `step`: side by
    side for a horizontal step, a vertical stack for (0, 0, dz). The steps
    are not binary fractions, so congruent pairs differ in their last bits."""
    step = np.asarray(step, dtype=float)
    wires = [
        np.array([[0.0, -length / 2, height], [0.0, length / 2, height]]) + i * step
        for i in range(n_el)
    ]
    return deck_of(wires, [n_seg] * n_el)


STACK = (0.0, 0.0, 2.3)


def seg(center, direction, half):
    c, d = np.asarray(center, dtype=float), np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return np.array([c - half * d, c + half * d])


H_A = 4.0
C45 = np.sqrt(0.5)


def tilted_twin():
    """Observer A along x at height H_A; B horizontal along y; B' the tilted
    twin whose image pairs with A are congruent to B's: M(B) rotated by 45
    degrees about A's axis, mirrored back above the plane. A is carried onto
    itself segment by segment, so (A_i, M(B_j)) and (A_i, M(B'_j)) have the
    same invariants, while B' leans 45 degrees to the plane and B does not."""
    hb, xb = 6.0, 8.0
    r = hb + H_A  # M(B)'s centre from A's axis
    b_prime = seg((xb, r * C45, r * C45 - H_A), (0.0, C45, -C45), 2.0)
    return deck_of(
        [
            seg((0.0, 0.0, H_A), (1.0, 0.0, 0.0), 5.0),
            seg((xb, 0.0, hb), (0.0, 1.0, 0.0), 2.0),
            b_prime,
        ],
        [21, 9, 9],
    )


def key_bait():
    """A along x at height H_A, S vertical (+z), and S2: M(S) with each
    segment reversed (the segment the key probe describes: M(S)'s centres
    with S's own, unmirrored direction), rotated half a turn about A's axis
    back above the plane, so drawn top down. A key that read S's direction
    for M(S) (the key probe) would file (A, M(S_j)) with the direct pair
    (A, S2_{n-1-j}), whose moments are not the image's."""
    xs, hs, half = 8.0, 6.0, 2.0
    return deck_of(
        [
            seg((0.0, 0.0, H_A), (1.0, 0.0, 0.0), 5.0),
            seg((xs, 0.0, hs), (0.0, 0.0, 1.0), half),
            seg((xs, 0.0, 2 * H_A + hs), (0.0, 0.0, -1.0), half),
        ],
        [21, 9, 9],
    )


@pytest.fixture(autouse=True)
def _small_decks():
    """Serve the reuse on these small decks, from no plan left by another
    test."""
    acc.bspline_reuse_set_small_deck_segs(0)
    acc.bspline_reuse_cache_clear()
    yield
    acc.bspline_reuse_set_small_deck_segs(-1)
    acc.bspline_reuse_set_image_key_probe(0)
    acc.bspline_reuse_class_limit(0)
    acc.bspline_reuse_cache_clear()


def z_of(deck, ground, **kw):
    s = bs2(**deck, **ground, **kw)
    geom = s._build_geometry()
    supp, polys, *_ = s._build_basis_polynomials(geom)
    return s, np.array(s._compute_Z_operator(geom, supp, polys))


def zin(deck, ground, **kw):
    s = bs2(**deck, **ground, **kw)
    return s, complex(np.ravel(s.compute_impedance()[0])[0])


def rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def ran(fn):
    """fn()'s result, with the plan's image fills and weighted tiled calls
    it made."""
    fills = bspline_mod.IMAGE_REUSE_FILLS
    c0 = acc.bspline_reuse_window_counts()
    out = fn()
    c1 = acc.bspline_reuse_window_counts()
    calls = sum(c1[k] - c0[k] for k in ("weighted_tiled", "weighted_whole"))
    return out, bspline_mod.IMAGE_REUSE_FILLS - fills, calls


@pytest.mark.parametrize("ground", sorted(GROUNDS))
@pytest.mark.parametrize("layout", ["row", "stack"])
def test_the_image_route_ran_and_agrees_with_reuse_off(layout, ground):
    deck = row(step=STACK) if layout == "stack" else row()
    (s_on, z_on), fills, calls = ran(
        lambda: z_of(deck, GROUNDS[ground], pair_reuse=True, toeplitz=False)
    )
    assert fills == 1 and calls >= 1, (fills, calls)
    note = s_on._fill_ground_notes["image"]
    assert note["plans"] and note["pairs"] - note["computed"] > note["pairs"] // 2
    _s, z_off = z_of(deck, GROUNDS[ground], pair_reuse=False)
    assert not np.array_equal(z_on, z_off)
    assert rel(z_on, z_off) <= 1e-10
    _s, a = zin(deck, GROUNDS[ground], pair_reuse=True, toeplitz=False)
    _s, b = zin(deck, GROUNDS[ground], pair_reuse=False)
    assert abs(a - b) / abs(b) <= DRIFT


def test_a_tensor_image_fill_plans_no_image():
    """Past the chunked image fill's degree (`_BSPLINE_ASSEMBLE_ACCEL_MAX_D`)
    the image is the tensor route's, which reads no plan: none is built for
    it, and the report says why."""
    s = bs2(**row(), **PEC, pair_reuse=True, toeplitz=False, degree=3)
    (_r, fills, _c) = ran(s.compute_impedance)
    assert fills == 0
    assert not s._reuse_plan(s._build_geometry()).image
    assert s._reuse_image_declined == "the image fill is not the chunked one"


@pytest.mark.parametrize("degree", [1])
def test_the_weighted_tiles_switch_on_the_degree(degree):
    """The weighted tiled assembler's degree switch, run off d = 2 (d = 3
    reaches it only through the binding: no image fill serves d = 3)."""
    deck = row()
    (_s, z_on), fills, calls = ran(
        lambda: z_of(deck, PEC, pair_reuse=True, toeplitz=False, degree=degree)
    )
    assert fills == 1 and calls >= 1
    _s, z_off = z_of(deck, PEC, pair_reuse=False, degree=degree)
    assert not np.array_equal(z_on, z_off)
    assert rel(z_on, z_off) <= 1e-10


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_a_vertical_stack_shares_image_classes_between_edges(ground):
    """The old branch's negative test, inverted. Pairs (i, M(j)) and
    (i + 1, M(j - 1)) of an evenly spaced stack are congruent (the same
    height sum and offset), which the edge-block map, keyed on each height,
    could not see. Without the Toeplitz form every reused image pair is a copy
    by the plan, and the report names no image term left out."""
    deck = row(n_el=6, step=STACK)
    s = bs2(**deck, **GROUNDS[ground], pair_reuse=True, toeplitz=False)
    s.compute_impedance()
    reuse = s.fill_report()["reuse"]
    img = reuse["families"]["image"]
    n = deck["n_per_edge_per_wire"][0][0]
    between = img["pairs"] - len(deck["wires"]) * n * n
    assert img["reused"] > between // 2, img
    assert img["pairs"] >= 20 * img["computed"], img
    assert IMAGE_NOTE not in (reuse["declined"] or "")
    if ground != "sommerfeld":  # the remainder's map is step 2b's
        assert reuse["state"] == "served", reuse["declined"]
    # The image family's wire map: copies between the stacked elements.
    wires = reuse["wires"]
    assert np.array(wires["reused"]).sum() == reuse["reused"]


def test_an_edges_own_image_block_is_the_plans():
    """An edge's own image block is keyed like any other image pair
    (momwire#2063 step 3 retired the image offset block): the image tile
    serves it from the class values, and the fill agrees with the one
    without reuse."""
    deck = row()
    (s, z_on), fills, _ = ran(lambda: z_of(deck, PEC, pair_reuse=True))
    assert fills == 1
    plan = s._reuse_plan(s._build_geometry())
    n = deck["n_per_edge_per_wire"][0][0]
    idx = np.arange(n, dtype=np.int64)
    own = plan.classes_of(1, idx, idx)
    # one element's own image pairs share their classes along the edge
    assert np.unique(own).size < n * n // 4
    _s, z_off = z_of(deck, PEC, pair_reuse=False, toeplitz=False)
    assert rel(z_on, z_off) <= 1e-12


@pytest.mark.parametrize("probe", ["scale", "tangent"])
def test_a_wrong_sign_in_the_new_route_moves_z(probe, monkeypatch):
    """The image added at +1, or weighted with the source's unmirrored
    tangent, in the plan's route only: Z moves by far more than the drift,
    and the reuse-off fill does not move at all."""
    # The tangent's z-flip reads only where both tangents lean (t_z t_z' != 0).
    deck = key_bait() if probe == "tangent" else row()
    _s, z_on = z_of(deck, PEC, pair_reuse=True, toeplitz=False)
    _s, z_off = z_of(deck, PEC, pair_reuse=False)
    monkeypatch.setattr(bspline_mod, "_IMAGE_REUSE_PROBE", probe)
    (_s, z_bad), fills, _ = ran(
        lambda: z_of(deck, PEC, pair_reuse=True, toeplitz=False)
    )
    assert fills == 1
    assert rel(z_bad, z_on) > 1e-8
    _s, z_off_probed = z_of(deck, PEC, pair_reuse=False)
    assert np.array_equal(z_off_probed, z_off)
    assert rel(z_on, z_off) <= 1e-10


def test_a_key_that_drops_the_mirror_moves_z():
    """The key probe files (A, M(S)) on S's unmirrored direction, which is
    congruent to the direct pairs (A, S2): the image pairs then take the
    direct pairs' moments, and Z moves at O(1) relative to the drift."""
    deck = key_bait()
    _s, z_off = z_of(deck, PEC, pair_reuse=False)
    _s, z_on = z_of(deck, PEC, pair_reuse=True, toeplitz=False)
    assert rel(z_on, z_off) <= 1e-10
    acc.bspline_reuse_set_image_key_probe(1)
    s, z_bad = z_of(deck, PEC, pair_reuse=True, toeplitz=False)
    plan = s._reuse_plan(s._build_geometry())
    a, w_s, w_s2 = (np.arange(21), np.arange(21, 30), np.arange(30, 39))
    # The probe's merge: the image pairs (A, M(S)) are the direct (A, S2)'s.
    merged = plan.classes_of(1, a, w_s) == plan.classes_of(0, a, w_s2[::-1])
    assert merged.mean() >= 0.9, merged.mean()
    assert rel(z_bad, z_off) > 1e-8


def test_refl_tilted_and_horizontal_share_classes_not_weights():
    """B' leans 45 degrees to the plane and B lies flat, and their image
    pairs with A are congruent: they share block-1 classes. The reflection
    coefficient reads the pair's own angle to the plane, so their weights
    differ at O(1); a copy of weighted entries by this key would be wrong by
    that much, and the class values (moments) copied with each pair's own
    weights agree with the fill without reuse."""
    deck = tilted_twin()
    s, z_on = z_of(deck, REFL, pair_reuse=True, toeplitz=False)
    geom = s._build_geometry()
    plan = s._reuse_plan(geom)
    a, b, b2 = (np.arange(21), np.arange(21, 30), np.arange(30, 39))
    same = plan.classes_of(1, a, b) == plan.classes_of(1, a, b2)
    assert same.mean() >= 0.9, same.mean()
    ground = _potential_ground.potential_ground_for(s, geom, s.k, s.omega)
    w_A, _w_Phi = ground.weight_windows()(0, 21)
    # The vector potential's dyad reads the tilt (the charge's weight, under
    # the default phi mode, does not).
    d = np.abs(w_A[:, b] - w_A[:, b2])[same]
    assert np.min(d / np.abs(w_A[:, b][same])) > 1e-2
    _s, z_off = z_of(deck, REFL, pair_reuse=False)
    assert rel(z_on, z_off) <= 1e-10
    _s, za = zin(deck, REFL, pair_reuse=True, toeplitz=False)
    _s, zb = zin(deck, REFL, pair_reuse=False)
    assert abs(za - zb) / abs(zb) <= DRIFT


@pytest.mark.parametrize("ground", ["pec", "sommerfeld"])
def test_the_direct_familys_bits_do_not_move_with_the_image(ground, monkeypatch):
    """The image block's classes come after the direct block's in the census,
    so every direct pair keeps its representative and its value's bits: the
    direct fill reads the same Z with the image block beside it as without."""
    deck = row()

    def direct():
        s = bs2(**deck, **GROUNDS[ground], pair_reuse=True)
        geom = s._build_geometry()
        supp, polys, *_ = s._build_basis_polynomials(geom)
        plan = s._reuse_plan(geom)
        return plan.image, np.array(s._compute_Z_dense_chunked(geom, s.k, supp, polys))

    with_image, z_img = direct()
    assert with_image
    acc.bspline_reuse_cache_clear()
    monkeypatch.setattr(BSplineSolver, "_reuse_image_args", lambda *a: None)
    with_image, z_plain = direct()
    assert not with_image
    assert np.array_equal(z_img, z_plain)


def test_reuse_off_reads_no_image_plan():
    for ground in GROUNDS.values():
        for off in (False, True):
            (_s, _z), fills, calls = ran(
                lambda g=ground, o=off: z_of(row(), g, pair_reuse=False, toeplitz=o)
            )
            assert fills == 0 and calls == 0


def test_a_fill_of_chosen_rows_declines_as_before():
    """A restricted fill reads no plan, the image's included: its rows are
    reuse-off's to the bit, and the decline is the direct fill's."""
    deck = row()
    rows = np.arange(42, dtype=np.int64)  # wires 0 and 1

    def restricted(reuse):
        # The Toeplitz form (which follows the switch) is a route of its own
        # here.
        s = bs2(**deck, **PEC, pair_reuse=reuse, toeplitz=False)
        geom = s._build_geometry()
        supp, polys, *_ = s._build_basis_polynomials(geom)
        return s, np.array(s._compute_Z_operator(geom, supp, polys, rows=rows))

    (s, z_on), fills, calls = ran(lambda: restricted(True))
    assert fills == 0 and calls == 0
    assert s.pair_reuse_declined == "a fill of chosen rows does not read the hint"
    _s, z_off = restricted(False)
    assert np.array_equal(z_on, z_off)


def test_an_image_ladder_of_its_own_is_not_planned_and_says_why():
    """The image key is the direct key only on the direct fill's ladder."""
    s = bs2(**row(), **PEC, pair_reuse=True, image_pair_order_ladder=())
    (_r, fills, _c) = ran(s.compute_impedance)
    assert fills == 0
    reuse = s.fill_report()["reuse"]
    assert reuse["families"]["direct"]["reused"] > 0
    assert reuse["state"] == "partial"
    assert (
        f"{IMAGE_NOTE} (the image fill's pair-order ladder is not the direct "
        "fill's)" in reuse["declined"]
    )


def test_a_declined_image_census_keeps_the_direct_plan():
    """Past the class limit with its image block but not without it, the
    plan is the direct block alone: the direct term still serves, and the
    image term is named with its census's reason."""
    deck = row()
    s = bs2(**deck, **PEC, pair_reuse=True, toeplitz=False)
    s.compute_impedance()
    st = s._reuse_plan(s._build_geometry()).stats()
    acc.bspline_reuse_cache_clear()
    acc.bspline_reuse_class_limit(int(st["classes_block"][0]) + 1)
    s = bs2(**deck, **PEC, pair_reuse=True, toeplitz=False)
    (_r, fills, _c) = ran(s.compute_impedance)
    assert fills == 0
    reuse = s.fill_report()["reuse"]
    assert reuse["families"]["direct"]["reused"] > 0
    assert f"{IMAGE_NOTE} (too many classes to code)" in reuse["declined"]


def test_a_buried_deck_plans_no_image():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = bs2(
            **row(n_el=2, height=-0.5),
            **SOMMERFELD,
            pair_reuse=True,
        )
        assert s._reuse_image_args(s._build_geometry(), None, ()) is None
