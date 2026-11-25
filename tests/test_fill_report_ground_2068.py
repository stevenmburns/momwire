"""bspline's fill report counts its ground terms (momwire#2068).

Over a ground a matrix entry is the direct term plus the image, plus on a
Sommerfeld ground the remainder. Each is its own family in `fill_report()`,
counted in the direct family's unit (ordered segment pairs), and `uncounted`
is empty when every term was counted.

The counts are tied to the physics rather than to the code that keeps them:
the remainder copies a pair only where an isometry keeps every height, so a
stack of horizontal elements (heights differ) copies nothing there while the
same elements side by side (one height) do; and on a deck the kernel walks in
one tile, the kernel's own copy counter must equal the reported reuse.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from momwire import _accel
from momwire import bspline as bspline_mod
from momwire.bspline import BSplineSolver

acc = _accel.acc
pytestmark = pytest.mark.skipif(
    acc is None or not hasattr(acc, "sommerfeld_remainder_bspline_Q"),
    reason="needs the C++ accelerator",
)

LAM = 21.0
RADIUS = 0.002
SOMMERFELD = {
    "ground_z": 0.0,
    "ground_eps": (13.0, 0.005),
    "ground_model": "sommerfeld",
}


def row(n_el=4, n_seg=21, height=6.0, length=10.0, step=(4.0, 0.0, 0.0)):
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


@pytest.fixture(autouse=True)
def _small_decks():
    """Serve the reuse on these small decks, from no plan left by another
    test."""
    acc.bspline_reuse_set_small_deck_segs(0)
    acc.bspline_reuse_cache_clear()
    yield
    acc.bspline_reuse_set_small_deck_segs(-1)
    acc.bspline_reuse_cache_clear()


def solve(deck, ground, **kw):
    s = BSplineSolver(**deck, pair_reuse=True, **ground, **kw)
    s.compute_impedance()
    return s


def reconciles(reuse):
    for name, f in reuse["families"].items():
        assert f["computed"] + f["reused"] + f["near"] == f["pairs"], name
        assert f["kernel_evals"] >= f["computed"], name
    assert reuse["computed"] + reuse["reused"] + reuse["near"] == reuse["pairs"]
    wires = reuse["wires"]
    p, c, r, n = (np.array(wires[k]) for k in ("pairs", "computed", "reused", "near"))
    assert np.array_equal(c + r + n, p)
    assert (int(p.sum()), int(c.sum()), int(r.sum()), int(n.sum())) == (
        reuse["pairs"],
        reuse["computed"],
        reuse["reused"],
        reuse["near"],
    )


def n_segs(deck):
    return sum(sum(e) for e in deck["n_per_edge_per_wire"])


def test_sommerfeld_reports_three_families_and_nothing_uncounted():
    deck = row()
    s = solve(deck, SOMMERFELD, offset_reuse=False)
    reuse = s.fill_report()["reuse"]
    json.dumps(reuse)
    assert set(reuse["families"]) == {"direct", "image", "remainder"}
    assert reuse["uncounted"] == []
    n2 = n_segs(deck) ** 2
    image, rem = reuse["families"]["image"], reuse["families"]["remainder"]
    # Same pair convention in every family: ordered segment pairs.
    assert reuse["families"]["direct"]["pairs"] == image["pairs"] == rem["pairs"] == n2
    # The image copies the remainder's blocks (momwire#2074): one map.
    assert image["reused"] == rem["reused"] > 0
    assert rem["computed"] + rem["reused"] + rem["near"] == rem["pairs"]
    assert rem["reused"] > 0
    reconciles(reuse)


def test_pec_reports_the_image_and_no_remainder():
    s = solve(row(), {"ground_z": 0.0}, offset_reuse=False)
    reuse = s.fill_report()["reuse"]
    assert set(reuse["families"]) == {"direct", "image"}
    assert reuse["uncounted"] == []
    reconciles(reuse)


def test_a_reflection_coefficient_ground_has_no_remainder_to_count():
    ground = {"ground_z": 0.0, "ground_eps": (13.0, 0.005)}
    s = solve(row(), ground, offset_reuse=False)
    reuse = s.fill_report()["reuse"]
    assert set(reuse["families"]) == {"direct", "image"}
    assert reuse["uncounted"] == []


def test_free_space_reports_only_the_direct_family():
    s = solve(row(height=0.0), {})
    reuse = s.fill_report()["reuse"]
    assert set(reuse["families"]) == {"direct"}
    assert reuse["uncounted"] == []


def test_the_remainder_copies_nothing_where_the_heights_differ():
    """A vertical stack: the direct term repeats (translation), the
    remainder cannot (it reads the height sum)."""
    stack = solve(row(step=(0.0, 0.0, 2.0)), SOMMERFELD, offset_reuse=False)
    reuse = stack.fill_report()["reuse"]
    assert reuse["families"]["direct"]["reused"] > 0
    assert reuse["families"]["remainder"]["reused"] == 0
    reconciles(reuse)
    side = solve(row(), SOMMERFELD, offset_reuse=False)
    assert side.fill_report()["reuse"]["families"]["remainder"]["reused"] > 0


def test_the_remainder_reuse_matches_the_kernels_own_counter_and_the_physics():
    """Brute force on a one-tile deck: the kernel's process counter of copied
    pairs moves by exactly the reported reuse, and every ordered segment pair
    the map copies has the same horizontal offsets and heights, node for
    node, as the pair it is copied from."""
    deck = row(n_el=3, n_seg=13)
    s = BSplineSolver(**deck, pair_reuse=True, **SOMMERFELD, offset_reuse=False)
    before = acc.remainder_q_reused_pairs()
    s.compute_impedance()
    copied = acc.remainder_q_reused_pairs() - before
    rem = s.fill_report()["reuse"]["families"]["remainder"]
    n2 = n_segs(deck) ** 2
    assert rem["pairs"] == n2
    assert rem["reused"] == copied > 0
    # One tile: every ordered pair walked once, and the walk reads the source
    # pairs back from the cache that filled them (momwire#2073), so each pair
    # the kernel did not copy is evaluated exactly once.
    assert rem["kernel_evals"] == rem["computed"]

    geom = s._build_geometry()
    cmap = s._remainder_reuse_args(geom, 0.0)
    seg0 = cmap["reuse_edge_seg0"]
    src = cmap["reuse_src"]
    E = src.shape[0]
    L, R = geom["seg_l"], geom["seg_r"]
    scale = float(np.max(np.abs(np.r_[L, R])))
    count = 0
    for ta, tb in zip(*np.nonzero(src >= 0), strict=True):
        sa, sb = divmod(int(src[ta, tb]), E)
        for x in range(seg0[ta + 1] - seg0[ta]):
            for y in range(seg0[tb + 1] - seg0[tb]):
                i, j = seg0[ta] + x, seg0[tb] + y
                i0, j0 = seg0[sa] + x, seg0[sb] + y
                for P, Q in ((L, L), (L, R), (R, L), (R, R)):
                    d_t, d_s = P[i] - Q[j], P[i0] - Q[j0]
                    assert abs(np.hypot(*d_t[:2]) - np.hypot(*d_s[:2])) <= 1e-9 * scale
                    assert abs(P[i, 2] - P[i0, 2]) <= 1e-9 * scale
                    assert abs(Q[j, 2] - Q[j0, 2]) <= 1e-9 * scale
                count += 1
    assert count == rem["reused"]


def test_remainder_reuse_saves_kernel_work():
    """momwire#2073: the walk reads the copy map's source pairs back from the
    cache that filled them, so every reused pair is kernel work saved. Before,
    the walk formed each source pair a second time, and on a deck of mostly
    reused pairs the kernel evaluated more entries than there are pairs.

    Eight elements of 31 segments span two basis tiles, so the halo repeats
    are in play: the saving falls short of the reuse by those alone (under
    1 % here; base 7be69268 saved 72 % of the reuse)."""
    deck = row(n_el=8, n_seg=31, step=(3.0, 0.0, 0.0))
    s = solve(deck, SOMMERFELD, offset_reuse=False)
    rem = s.fill_report()["reuse"]["families"]["remainder"]
    assert rem["reused"] > rem["pairs"] // 2
    assert rem["kernel_evals"] < rem["pairs"]
    assert rem["pairs"] - rem["kernel_evals"] >= 0.99 * rem["reused"]
    reconciles(s.fill_report()["reuse"])


def test_offset_edges_count_as_reuse_in_the_ground_families():
    """Under `offset_reuse` an edge's own image block is spread from 2n - 1
    distinct values: the image family says so."""
    deck = row()
    before = bspline_mod.IMAGE_OFFSET_EDGES
    s = solve(deck, SOMMERFELD)
    assert bspline_mod.IMAGE_OFFSET_EDGES - before == len(deck["wires"])
    reuse = s.fill_report()["reuse"]
    image = reuse["families"]["image"]
    n = deck["n_per_edge_per_wire"][0][0]
    # Beside the spread, the blocks between edges the copy map serves
    # (momwire#2074), which leaves the offset edges' own blocks to the spread.
    (cmap,) = s._fill_ground_notes["image"]["maps"]
    _ta, _tb, na, nb = s._copy_map_blocks(cmap)
    copied = int(np.sum(na * nb))
    assert copied > 0
    assert image["reused"] == len(deck["wires"]) * (n * n - (2 * n - 1)) + copied
    reconciles(reuse)


def test_a_ground_fill_of_chosen_rows_is_not_counted_as_the_mesh():
    """A term no whole-mesh fill noted is listed in `uncounted`."""
    s = solve(row(), SOMMERFELD, offset_reuse=False)
    s._fill_ground_notes = {}
    ledger = bspline_mod._fill_report.ledger_of(s)
    s._fill_report_end(ledger)
    assert ledger.reuse["uncounted"] == ["image", "remainder"]
