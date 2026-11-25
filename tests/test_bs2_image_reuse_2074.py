"""bs2 copies its ground image term between edges under `pair_reuse`
(momwire#2074).

The image moment of a pair is the free-space moment of the observer segment
against the MIRROR of its source, so it depends on the pair's horizontal
offsets and on the height sums, |P - M(Q)|^2 = rho^2 + (h_P + h_Q)^2. A motion
that keeps the plane and every height keeps it. The Sommerfeld remainder's
copy map (`_remainder_reuse_map`, `remainder_blocks`) finds exactly the edge
blocks alike under such a motion, so the image reads the same map
(`_image_copy_map`). Only the moments are copied; each pair's weights (PEC's
mirrored tangent dot, the reflection coefficient's Fresnel dyad and cos theta,
the Sommerfeld image's C2) are its own, so the map does not depend on the
image model, and each model is held here on against off.

The gates, and why each is not green by construction:

* every agreement test first proves the copy RAN (the image family's reused
  pairs, with `offset_reuse` off so no other reuse is counted there); a fill
  that fell back would agree to the bit and measure nothing;
* on against off within the registry's drift, on PEC, reflection-coefficient
  and Sommerfeld grounds, the side-by-side array translated by a step that is
  not a binary fraction, so source and copy differ in their last bits;
* every copied block's moments, recomputed from its own geometry, against the
  cached source's: a wrong map moves them at O(1), not O(1e-12);
* the physics: a vertical stack of the same elements copies no image block
  (the map keys on each height, not their sum), side by side they do;
* razor2p's image family on the same deck: what bs2 now reaches against it.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire import _accel, _features
from momwire import _sin_pair_reuse as pr
from momwire import bspline as bspline_mod
from momwire._bspline_kernels import _seg_seg_full_moments_offedge
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver

acc = _accel.acc
pytestmark = [
    pytest.mark.skipif(
        acc is None or not hasattr(acc, "bspline_reuse_remainder_blocks"),
        reason="needs the C++ accelerator",
    ),
    pytest.mark.filterwarnings("ignore"),
]

DRIFT = _features.FEATURES["pair_reuse"].drift
LAM = 21.0
RADIUS = 0.002
PEC = {"ground_z": 0.0}
REFL = {"ground_z": 0.0, "ground_eps": (13.0, 0.005)}
SOMMERFELD = {**REFL, "ground_model": "sommerfeld"}
GROUNDS = {"pec": PEC, "refl": REFL, "sommerfeld": SOMMERFELD}


def array(n_el=5, n_seg=21, height=6.0, length=10.0, step=(3.7, 0.3, 0.0)):
    """Equal horizontal elements, element i offset by i * `step`: side by
    side at one height for a horizontal step, a vertical stack for
    (0, 0, dz)."""
    step = np.asarray(step, dtype=float)
    wires = [
        np.array([[0.0, -length / 2, height], [0.0, length / 2, height]]) + i * step
        for i in range(n_el)
    ]
    return {
        "wires": wires,
        "n_per_edge_per_wire": [[n_seg]] * n_el,
        "wavelength": LAM,
        "wire_radius": RADIUS,
        "feed_wire_index": 1,
    }


STACK = (0.0, 0.0, 2.3)


@pytest.fixture(autouse=True)
def _small_decks(monkeypatch):
    """Serve the reuse on these small decks, from no plan left by another
    test."""
    acc.bspline_reuse_set_small_deck_segs(0)
    acc.razor_reuse_set_small_deck_segs(0, 0)
    monkeypatch.setattr(pr, "SMALL_DECK_SEGS_FREE", 0)
    monkeypatch.setattr(pr, "SMALL_DECK_SEGS_GROUND", 0)
    acc.bspline_reuse_cache_clear()
    acc.razor_reuse_cache_clear()
    pr.clear_shared_plans()
    yield
    acc.bspline_reuse_set_small_deck_segs(-1)
    acc.razor_reuse_set_small_deck_segs(-1, -1)
    acc.bspline_reuse_cache_clear()
    acc.razor_reuse_cache_clear()
    pr.clear_shared_plans()


def z_matrix(deck, ground, **kw):
    s = BSplineSolver(**deck, **ground, **kw)
    geom = s._build_geometry()
    supp, polys, *_ = s._build_basis_polynomials(geom)
    return s, np.array(s._compute_Z_operator(geom, supp, polys))


def image_family(s):
    s.compute_impedance()
    return s.fill_report()["reuse"]["families"]["image"]


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_on_against_off_within_the_drift(ground):
    deck = array()
    s_on, z_on = z_matrix(deck, GROUNDS[ground], pair_reuse=True, offset_reuse=False)
    note = s_on._fill_ground_notes["image"]
    assert note["maps"], "the image fill read no copy map"
    assert note["pairs"] - note["computed"] > note["pairs"] // 4
    _s, z_off = z_matrix(deck, GROUNDS[ground], pair_reuse=False)
    # Not equal: the copies are of pairs a translation away, which rounds.
    assert not np.array_equal(z_on, z_off)
    rel = np.max(np.abs(z_on - z_off)) / np.max(np.abs(z_off))
    assert rel <= DRIFT, rel


def test_every_copied_block_is_its_own_moments():
    """Each block the map copies, recomputed from its own geometry, against
    the source block it reads; and node for node the same horizontal
    distances and height sums."""
    deck = array(n_el=4, n_seg=9)
    s = BSplineSolver(**deck, **PEC, pair_reuse=True, offset_reuse=False)
    s.compute_impedance()
    geom = s._build_geometry()
    copy = s._image_copy_map(geom, set())
    seg0, src = copy["reuse_edge_seg0"], copy["reuse_src"]
    E = src.shape[0]
    L, R = geom["seg_l"], geom["seg_r"]
    Li, Ri = s._image_positions(L), s._image_positions(R)
    a = s._seg_radius(geom)
    ek = s._ek_spec(geom, mirror=True) if s.extended_kernel else None
    ladder = s._image_fill_ladder(s.k, L, R, ek)

    def block(x, y):
        r, c = slice(seg0[x], seg0[x + 1]), slice(seg0[y], seg0[y + 1])
        return _seg_seg_full_moments_offedge(
            L[r], R[r], Li[c], Ri[c], a[r], s.k, s.degree, s.n_qp_pair, ladder=ladder
        )

    copied = 0
    for ta, tb in zip(*np.nonzero(src >= 0), strict=True):
        sa, sb = divmod(int(src[ta, tb]), E)
        J_t, J_s = block(ta, tb), block(sa, sb)
        assert np.max(np.abs(J_t - J_s)) <= 1e-12 * np.max(np.abs(J_s))
        for P, Q in ((L, Li), (L, Ri), (R, Li), (R, Ri)):
            for x in range(seg0[ta + 1] - seg0[ta]):
                d_t = np.linalg.norm(
                    P[seg0[ta] + x] - Q[seg0[tb] : seg0[tb + 1]], axis=1
                )
                d_s = np.linalg.norm(
                    P[seg0[sa] + x] - Q[seg0[sb] : seg0[sb + 1]], axis=1
                )
                assert np.allclose(d_t, d_s, rtol=0.0, atol=1e-9)
        copied += 1
    assert copied > 0


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_a_vertical_stack_copies_no_image_block(ground):
    """The physics check. Side by side (one height) the image copies; stacked
    (heights differ) the map copies nothing, and the fill report says the
    image term went without. The map keys on each height; the image itself
    depends on their sum, so a pair-level key on the sum (razor2p's) does
    find repeats on a stack (`test_bs2_against_razor2p`)."""
    side = BSplineSolver(
        **array(), **GROUNDS[ground], pair_reuse=True, offset_reuse=False
    )
    assert image_family(side)["reused"] > 0
    assert side.fill_report()["reuse"]["state"] == "served"
    stack = BSplineSolver(
        **array(step=STACK), **GROUNDS[ground], pair_reuse=True, offset_reuse=False
    )
    img = image_family(stack)
    reuse = stack.fill_report()["reuse"]
    assert img["reused"] == 0 and img["computed"] == img["pairs"]
    assert reuse["families"]["direct"]["reused"] > 0
    assert reuse["state"] == "partial"
    assert "the image term copied no block between edges" in reuse["declined"]


def test_a_stack_keeps_only_the_offset_spread():
    """Under `offset_reuse` a stack's image reuse is each edge's own spread,
    n^2 - (2n - 1) an edge, and nothing between edges."""
    deck = array(step=STACK)
    s = BSplineSolver(**deck, **PEC, pair_reuse=True)
    img = image_family(s)
    n = deck["n_per_edge_per_wire"][0][0]
    assert img["reused"] == len(deck["wires"]) * (n * n - (2 * n - 1))
    assert s.fill_report()["reuse"]["state"] == "partial"


def test_a_cut_map_reads_partial_and_names_both_terms(monkeypatch):
    """The map cut to the cache budget is the image's and the remainder's."""
    deck = array()
    full = BSplineSolver(**deck, **SOMMERFELD, pair_reuse=True, offset_reuse=False)
    whole = image_family(full)["reused"]
    acc.bspline_reuse_cache_clear()
    # Two source blocks of 21 x 21 pairs, both orders: 254 KB.
    monkeypatch.setattr(bspline_mod, "_REMAINDER_REUSE_CACHE_BYTES", 260_000)
    s = BSplineSolver(**deck, **SOMMERFELD, pair_reuse=True, offset_reuse=False)
    cut = image_family(s)["reused"]
    reuse = s.fill_report()["reuse"]
    assert s.pair_reuse_remainder_cut > 0
    assert 0 < cut < whole
    assert reuse["state"] == "partial"
    assert "left the image and remainder copy map" in reuse["declined"]


def test_off_reads_no_copy_map(monkeypatch):
    """Reuse off never takes a copy map (the fill is the shipped one)."""
    seen = []
    orig = BSplineSolver._image_copy_map

    def spy(self, geom, skip):
        seen.append(orig(self, geom, skip))
        return seen[-1]

    monkeypatch.setattr(BSplineSolver, "_image_copy_map", spy)
    for ground in GROUNDS.values():
        z_matrix(array(), ground, pair_reuse=False)
        z_matrix(array(), ground, pair_reuse=False, offset_reuse=True)
    assert len(seen) == 2 * len(GROUNDS) and not any(seen)
    z_matrix(array(), PEC, pair_reuse=True)
    assert seen[-1] is not None


# Measured on `array(n_el=6, n_seg=21, step=(3.7, 0, 0))` over Sommerfeld,
# pairs / computed of each image family: razor2p 126; bs2 1.18 before
# momwire#2074 (the offset spread alone), 3.41 after. The gap that is left is
# the copy's unit: bs2 copies whole edge blocks, which on E equal elements
# leaves about 2E - 1 distinct blocks, where razor2p's pair key also finds the
# repeats inside a block (a pair of parallel equal edges is Toeplitz in
# j - i). So the bound below is 50: it holds after (37x) and fails before
# (107x).
RAZOR_FACTOR = 50.0


def test_bs2_against_razor2p():
    deck = array(n_el=6, n_seg=21, step=(3.7, 0.0, 0.0))

    def compression(s, family):
        s.compute_impedance()
        f = s.fill_report()["reuse"]["families"][family]
        return f["pairs"] / f["computed"]

    bs2 = compression(BSplineSolver(**deck, **SOMMERFELD, pair_reuse=True), "image")
    razor = compression(
        RazorSolver(nec5_quadrature=True, **deck, **SOMMERFELD, pair_reuse=True),
        "image",
    )
    sin = compression(
        SinusoidalSolver(**deck, **SOMMERFELD, pair_reuse=True), "reflected"
    )
    assert bs2 > 1.5
    assert bs2 * RAZOR_FACTOR >= razor, (bs2, razor, sin)
    assert bs2 * RAZOR_FACTOR >= sin, (bs2, razor, sin)
