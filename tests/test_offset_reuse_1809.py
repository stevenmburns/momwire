"""bs2's `offset_reuse`: a uniform straight edge's same-edge reg moments filled
by segment offset (momwire#1809).

On a uniform edge the smooth kernel's moment between segments i and j depends
on j - i alone, so the (N, N) block has 2N - 1 distinct entries. With
`offset_reuse=True` they come from one row of a 2N - 1 segment edge and the
block is a strided view of it. Off -- the default, and the standalone
executable's only fill -- the dense table is the fill as it was, bit for bit.

So:

- off is the dense fill (the existing suite holds that), and on a graded edge
  `offset_reuse=True` changes nothing, because nothing qualifies;
- on, the route runs (a counter, since an untaken route agrees trivially) and
  Z moves at the evaluation-order roundoff the same moments summed in another
  order give, in free space, with the extended kernel, over Sommerfeld ground,
  on the chunked route and in a frequency sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

import momwire._bspline_kernels as K
from momwire.bspline import BSplineSolver

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
    assert K.REG_TOEPLITZ_EDGES > c0  # the Toeplitz form was built
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


# -- step 2: the Sommerfeld remainder's own block of an offset edge ----------

import momwire.bspline as _B  # noqa: E402


def _ground_solver(wires, segs, reuse, **kw):
    return BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=segs,
        feeds=[(0, None, 1.0 + 0j)],
        wavelength=LAM,
        wire_radius=0.001,
        offset_reuse=reuse,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
        **kw,
    )


_GROUND_DECKS = {
    # horizontal: the remainder's pair (i, j) depends on j - i
    "horizontal": ([np.array([[0.0, -5.0, 10.0], [0.0, 5.0, 10.0]])], [[41]]),
    # low enough that #1189 raises some pairs' order: those are corrected
    # after the offset fill, as after the dense one
    "horizontal_low": ([np.array([[0.0, -5.0, 0.05], [0.0, 5.0, 0.05]])], [[41]]),
    # vertical: on i + j
    "vertical": ([np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 10.5]])], [[41]]),
    # one of each, joined: a basis straddles the corner
    "inverted_l": (
        [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0], [20.0, 0.0, 6.0]])],
        [[24, 41]],
    ),
    # two separate offset edges, and the cross block between them
    "two_wires": (
        [
            np.array([[0.0, -5.0, 10.0], [0.0, 5.0, 10.0]]),
            np.array([[3.0, -5.0, 10.0], [3.0, 5.0, 10.0]]),
        ],
        [[31], [31]],
    ),
}


@pytest.mark.parametrize("deck", sorted(_GROUND_DECKS))
def test_the_remainder_by_offset_runs_and_moves_z_by_roundoff(deck):
    wires, segs = _GROUND_DECKS[deck]
    Z0 = _Z(_ground_solver(wires, segs, False))
    c0 = _B.REMAINDER_OFFSET_EDGES
    s1 = _ground_solver(wires, segs, True)
    Z1 = _Z(s1)
    assert _B.REMAINDER_OFFSET_EDGES > c0  # the offset route ran
    assert _rel(Z1, Z0) <= 1e-13
    z0 = _ground_solver(wires, segs, False).compute_impedance()[0]
    z1 = _ground_solver(wires, segs, True).compute_impedance()[0]
    assert abs(z1 - z0) / abs(z0) <= 1e-12
    if deck == "horizontal_low":
        orders = s1._last_remainder_orders
        assert len(orders) > 1  # raised pairs were present, and corrected


def test_a_sloped_edge_keeps_the_dense_remainder():
    """Height sum and separation both vary along a sloped edge: nothing
    repeats, and the remainder is the dense fill."""
    wires = [np.array([[0.0, 0.0, 1.0], [8.0, 0.0, 7.0]])]
    c0 = _B.REMAINDER_OFFSET_EDGES
    _Z(_ground_solver(wires, [[41]], True))
    assert _B.REMAINDER_OFFSET_EDGES == c0


def test_the_full_block_and_the_fold_agree():
    """`_Z_sommerfeld_remainder` (the block) and `_into` (folded into T) take
    the same offset route and land the same numbers."""
    wires, segs = _GROUND_DECKS["inverted_l"]
    s = _ground_solver(wires, segs, True)
    cap = {}
    orig = BSplineSolver._Z_sommerfeld_remainder_into

    def spy(self, T, sign, geom, supp, polys, eps_t):
        cap.update(geom=geom, supp=supp, polys=polys, eps_t=eps_t)
        return orig(self, T, sign, geom, supp, polys, eps_t)

    try:
        BSplineSolver._Z_sommerfeld_remainder_into = spy
        _Z(s)
    finally:
        BSplineSolver._Z_sommerfeld_remainder_into = orig
    g, supp, polys, eps_t = cap["geom"], cap["supp"], cap["polys"], cap["eps_t"]
    with K.offset_reuse_scope(True):
        Q = s._Z_sommerfeld_remainder(g, supp, polys, eps_t)
        T = np.zeros_like(Q)
        s._Z_sommerfeld_remainder_into(T, 1, g, supp, polys, eps_t)
    assert np.array_equal(T, Q)
