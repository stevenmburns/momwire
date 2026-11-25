"""bs2's ground image term is not reused between edges yet (momwire#2159),
and the fill report says so rather than reading "served".

Under `pair_reuse` bs2 copies its direct term by the census and its
Sommerfeld remainder by its copy map; the image term keeps only the offset
spread within an edge. Over any ground with an image term, the headline is
therefore "partial" and names the image term; in free space, which has none,
it stays "served".

The cross-engine bound is the gap #2159 closes: on equal elements side by
side, razor2p's and Sinusoidal's image families compress about 125x, bs2's
1.2x. It is held as a strict expected failure, so the day bs2 reaches it the
test fails until the marker comes off.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import _accel
from momwire import _sin_pair_reuse as pr
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver

acc = _accel.acc
pytestmark = pytest.mark.skipif(
    acc is None or not hasattr(acc, "sommerfeld_remainder_bspline_Q"),
    reason="needs the C++ accelerator",
)

LAM = 21.0
RADIUS = 0.002
PEC = {"ground_z": 0.0}
REFL = {"ground_z": 0.0, "ground_eps": (13.0, 0.005)}
SOMMERFELD = {**REFL, "ground_model": "sommerfeld"}
GROUNDS = {"pec": PEC, "refl": REFL, "sommerfeld": SOMMERFELD}
IMAGE_NOTE = "the image term copied no block between edges"


def row(n_el=5, n_seg=21, height=6.0, length=10.0, step=(3.7, 0.0, 0.0)):
    """Equal horizontal elements side by side at one height."""
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


def reuse_of(solver):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.compute_impedance()
    return solver.fill_report()["reuse"]


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_a_ground_with_an_image_reads_partial_and_names_it(ground):
    reuse = reuse_of(BSplineSolver(**row(), **GROUNDS[ground], pair_reuse=True))
    # The direct term did serve: the partial is the image's alone.
    assert reuse["families"]["direct"]["reused"] > 0
    assert reuse["state"] == "partial", reuse["state"]
    assert IMAGE_NOTE in reuse["declined"], reuse["declined"]


def test_free_space_has_no_image_and_reads_served():
    reuse = reuse_of(BSplineSolver(**row(), pair_reuse=True))
    assert "image" not in reuse["families"]
    assert reuse["state"] == "served" and reuse["declined"] is None


# The bound #2159 is to meet: bs2's image family within this factor of
# razor2p's and Sinusoidal's on the same deck. Measured today (Sommerfeld):
# razor2p 126x, Sinusoidal 124x, bs2 1.18x.
FACTOR = 4.0


@pytest.mark.xfail(
    strict=True,
    reason="momwire#2159: bs2 does not copy its image term between edges yet",
)
def test_bs2_image_compresses_like_the_other_engines():
    deck = row(n_el=6)

    def compression(solver, family):
        f = reuse_of(solver)["families"][family]
        return f["pairs"] / f["computed"]

    bs2 = compression(BSplineSolver(**deck, **SOMMERFELD, pair_reuse=True), "image")
    razor = compression(
        RazorSolver(nec5_quadrature=True, **deck, **SOMMERFELD, pair_reuse=True),
        "image",
    )
    sin = compression(
        SinusoidalSolver(**deck, **SOMMERFELD, pair_reuse=True), "reflected"
    )
    # The other two do compress here, so the bound is not vacuous.
    assert razor > 50 and sin > 50, (razor, sin)
    assert bs2 * FACTOR >= razor, (bs2, razor, sin)
    assert bs2 * FACTOR >= sin, (bs2, razor, sin)
