"""bs2's `offset_reuse`: a uniform edge's same-edge reg moments filled by
segment offset (momwire#1809), inside the same-edge kernel (R).

On a uniform edge the smooth kernel's moment between segments i and j depends
on j - i alone, so the (N, N) block has 2N - 1 distinct entries. With
`offset_reuse=True` they come from one row of a 2N - 1 segment edge and the
block is a strided view of it. Off -- the default -- the dense table is the
fill as it was, bit for bit.

Its step-2 mechanisms (the free-space offset block, the image offset block,
the Sommerfeld remainder's offset fold, each spreading one row of an edge's
own moments by index) are gone since momwire#2063 step 3; `pair_reuse`'s
census serves those blocks, by the same-edge routine
(`test_bs2_census_same_edge_2063.py`). Here:

- off is the dense fill (the existing suite holds that), and on a graded edge
  `offset_reuse=True` changes nothing, because nothing qualifies;
- on, the route runs (a counter, since an untaken route agrees trivially) and
  Z moves at the evaluation-order roundoff the same moments summed in another
  order give, in free space, with the extended kernel, over Sommerfeld ground,
  on the chunked route and in a frequency sweep;
- the decks the offset blocks served are the census's now: an edge's own
  free-space, image and remainder blocks are classes of its plan, and agree.
"""

from __future__ import annotations

import numpy as np
import pytest

import momwire._bspline_kernels as K
from momwire.bspline import BSplineSolver

# An experimental feature (`momwire._features`, reach 4: through the drop-in
# on Linux and Windows).
pytestmark = pytest.mark.reach(4)

LAM = 21.1


def _wire(n, *, graded=False):
    L = 0.5 * LAM * n / 21
    if not graded:
        return [np.array([[0.0, -L / 2, 10.0], [0.0, L / 2, 10.0]])], [[n]]
    # Two collinear pieces at different segment lengths: neither is the
    # whole-wire uniform edge, and the shorter edge's mesh is finer.
    return [np.array([[0.0, -L / 2, 10.0], [0.0, 0.1, 10.0], [0.0, L / 2, 10.0]])], [
        [n // 3, n - n // 3]
    ]


def _solver(n, reuse, *, graded=False, ground=False, **kw):
    wires, segs = _wire(n, graded=graded)
    if ground:
        kw.update(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")
    return BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=segs,
        feeds=[(0, None, 1.0 + 0j)],
        wavelength=LAM,
        wire_radius=0.001,
        offset_reuse=reuse,
        **kw,
    )


def _Z(s):
    geom = s._build_geometry()
    supp, polys, *_ = s._build_basis_polynomials(geom)
    return s._compute_Z_operator(geom, supp, polys)


def _rel(Z1, Z0):
    return float(np.max(np.abs(Z1 - Z0)) / np.max(np.abs(Z0)))


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"extended_kernel": True},
        {"ground": True},
        {"swept_mem_mb": 1},  # the chunked route, windowed same-edge correction
    ],
    ids=["free", "ek", "sommerfeld", "chunked"],
)
def test_on_runs_and_moves_z_by_roundoff(kw):
    Z0 = _Z(_solver(81, False, **kw))
    c0 = K.REG_TOEPLITZ_EDGES
    Z1 = _Z(_solver(81, True, **kw))
    assert K.REG_TOEPLITZ_EDGES > c0  # the Toeplitz reg form was built
    assert _rel(Z1, Z0) <= 1e-13


def test_off_and_on_agree_on_the_impedance():
    z0 = _solver(121, False).compute_impedance()[0]
    z1 = _solver(121, True).compute_impedance()[0]
    assert abs(z1 - z0) / abs(z0) <= 1e-12


def test_a_sweep_takes_it_too():
    """The swept path builds its own per-k moments (one Toeplitz row per k)."""
    ks = 2 * np.pi / np.array([20.0, 21.1, 22.5])
    z0 = np.asarray(_solver(61, False).compute_impedance_swept(ks)[0])
    c0 = K.REG_TOEPLITZ_EDGES
    z1 = np.asarray(_solver(61, True).compute_impedance_swept(ks)[0])
    assert K.REG_TOEPLITZ_EDGES > c0
    assert np.max(np.abs(z1 - z0) / np.abs(z0)) <= 1e-12


def test_a_short_edge_keeps_the_dense_table():
    """Below `_REG_TOEPLITZ_MIN_N` segments the dense table is cheaper, and
    is what runs: Z is the off fill bit for bit."""
    n = K._REG_TOEPLITZ_MIN_N - 1
    c0 = K.REG_TOEPLITZ_EDGES
    Z1 = _Z(_solver(n, True))
    assert K.REG_TOEPLITZ_EDGES == c0
    Z0 = _Z(_solver(n, False))
    assert np.array_equal(Z1.view(np.uint64), Z0.view(np.uint64))


def test_the_scope_ends_with_the_fill():
    """The switch is the solver's, for its own fills: after one, a fresh
    default solver fills densely."""
    _Z(_solver(61, True))
    assert K._REG_TOEPLITZ.get() is False
    c0 = K.REG_TOEPLITZ_EDGES
    _Z(_solver(61, False))
    assert K.REG_TOEPLITZ_EDGES == c0


# -- the step-2 decks, served by the census since momwire#2063 step 3 ---------

import momwire.bspline as _B  # noqa: E402
from momwire import _features  # noqa: E402

DRIFT = _features.FEATURES["pair_reuse"].drift


@pytest.fixture
def _small_decks():
    """These decks are below the census's small-deck threshold; the census is
    what is under test, so it runs on them."""
    acc = _B._acc
    if acc is None or not hasattr(acc, "bspline_reuse_set_routine_probe"):
        pytest.skip("bs2's pair-reuse census is not built")
    acc.bspline_reuse_set_small_deck_segs(0)
    acc.pair_reuse_cap_compression(1)
    yield
    acc.bspline_reuse_set_small_deck_segs(-1)
    acc.pair_reuse_cap_compression(0)


_GROUND_DECKS = {
    # horizontal: the remainder's pair (i, j) depends on j - i
    "horizontal": ([np.array([[0.0, -5.0, 10.0], [0.0, 5.0, 10.0]])], [[41]]),
    # low enough that #1189 raises some pairs' order
    "horizontal_low": ([np.array([[0.0, -5.0, 0.05], [0.0, 5.0, 0.05]])], [[41]]),
    # vertical: on i + j
    "vertical": ([np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 10.5]])], [[41]]),
    # one of each, joined: a basis straddles the corner
    "inverted_l": (
        [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0], [20.0, 0.0, 6.0]])],
        [[24, 41]],
    ),
    # two separate edges, and the cross block between them
    "two_wires": (
        [
            np.array([[0.0, -5.0, 10.0], [0.0, 5.0, 10.0]]),
            np.array([[3.0, -5.0, 10.0], [3.0, 5.0, 10.0]]),
        ],
        [[31], [31]],
    ),
    # free space has no preferred direction: a sloped edge
    "sloped": ([np.array([[0.0, 0.0, 1.0], [8.0, 0.0, 7.0]])], [[41]]),
}
_GROUNDS = {
    "free": {},
    "pec": {"ground_z": 0.0},
    "sommerfeld": {
        "ground_z": 0.0,
        "ground_eps": (13.0, 0.005),
        "ground_model": "sommerfeld",
    },
}


@pytest.mark.usefixtures("_small_decks")
@pytest.mark.parametrize("ground", sorted(_GROUNDS))
@pytest.mark.parametrize("deck", sorted(_GROUND_DECKS))
def test_the_census_serves_an_edges_own_blocks(deck, ground):
    """An edge's own blocks, which the offset routes spread by index, are
    the census's classes now (the same-edge routine's in the direct block,
    the image's), and on against off holds the drift, `offset_reuse` on or
    off beside it."""
    wires, segs = _GROUND_DECKS[deck]

    def solver(**kw):
        return BSplineSolver(
            wires=wires,
            n_per_edge_per_wire=segs,
            feeds=[(0, None, 1.0 + 0j)],
            wavelength=LAM,
            wire_radius=0.001,
            **_GROUNDS[ground],
            **kw,
        )

    z0 = solver().compute_impedance()[0]
    s = solver(pair_reuse=True)
    z1 = s.compute_impedance()[0]
    st = s._reuse_plan(s._build_geometry()).stats()
    assert 0 < st["same_edge_classes"] < st["same_edge_pairs"] // 2, st
    assert abs(z1 - z0) / abs(z0) <= DRIFT
    z2 = solver(pair_reuse=True, offset_reuse=True).compute_impedance()[0]
    assert abs(z2 - z0) / abs(z0) <= DRIFT + _features.FEATURES["offset_reuse"].drift


def test_no_offset_block_is_left():
    """The offset blocks' routes and their counters are gone."""
    for name in (
        "FREE_OFFSET_EDGES",
        "IMAGE_OFFSET_EDGES",
        "REMAINDER_OFFSET_EDGES",
    ):
        assert not hasattr(_B, name)
    for name in (
        "_free_offset_edges",
        "_offset_edges",
        "_remainder_offset_fold",
        "_image_offset_block",
    ):
        assert not hasattr(BSplineSolver, name)
