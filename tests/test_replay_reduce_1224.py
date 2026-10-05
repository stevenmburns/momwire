"""The sinusoidal replay's source-node reduction in C++ (momwire#1224).

`_replay_remainder` reduces each projected chunk over its q source nodes,
`block[s, i, n] = sum_k shp_w[s, n, k] * proj[i, n*q + k]`. numpy's einsum
spent 0.43 s on it in the SG buried x8 fill, single-threaded, against the
1.15 s kernel feeding it; `remainder_shape_reduce` is the same sum in
numpy's own arithmetic -- the unfused complex product per node, added k in
order to an accumulator that starts at +0.0 -- threaded over observer rows.
einsum stays reachable (`_REPLAY_REDUCE_ACCEL = False`) and is the
reference.

On x86-64 wheels that is einsum's bits (measured on numpy 2.5: SG and
point-matched buried Z and currents equal main's as uint64), and that is
load-bearing -- a fused spelling of the same sum agreed with einsum to 1e-16
per entry and still moved SG buried x8 Z by 1.7e-11, because SG subtracts
this block from the scaled image. It is NOT gated to the bit here: einsum's
arithmetic is numpy's build's (a build that contracts, e.g. on arm64, fuses
it), so kernel against einsum is a derived tolerance on every platform, and
only the kernel against itself is held as uint64.

Gates:

- G-1224-R1  the kernel equals einsum on random data (q = 1 and a single
             observer row included) and on a live SG chunk, entry by entry
             within the q-term dot product's rounding bound.
- G-1224-R2  an entry does not depend on the rows beside it: a row slice's
             reduction is the same rows of the whole, as uint64 -- the
             chunk-independence `_replay_remainder` promises its consumers.
- G-1224-R3  SG and point-matched buried Z and currents through the kernel
             against the einsum route, and the kernel was the one called.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import momwire.sinusoidal as _sin
from momwire import SinusoidalGalerkinSolver, SinusoidalSolver

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_crossing_serve_524 import hub_deck  # noqa: E402


def _direct_sg(*args, **kw):
    """This module's subject is the DIRECT-field fill's own machinery, so
    every solver here names it (`SinusoidalGalerkinSolver`'s default fill is
    the mixed-potential one since momwire#1354, gated in
    `test_sinusoidal_mp_1354.py`)."""
    from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver as _cls

    kw.setdefault("fill", "direct")
    return _cls(*args, **kw)


_acc = _sin._acc
pytestmark = [
    pytest.mark.filterwarnings("ignore:crossing node"),
    pytest.mark.skipif(
        not _sin._REPLAY_REDUCE_ACCEL,
        reason="the .so predates the replay reduction: rebuild",
    ),
]


def _einsum(shp_w, table):
    m = table.shape[0]
    n, q = shp_w.shape[1:]
    return np.einsum("snq,mnq->smn", shp_w, table.reshape(m, n, q))


def _assert_within_rounding(got, shp_w, table):
    """|kernel - einsum| per entry within twice the q-term complex dot
    product's rounding bound, 2 sqrt(2) (q + 2) u sum_k |w_k| |p_k| (u the
    unit roundoff), since each side carries at most one such error."""
    ref = _einsum(shp_w, table)
    q = shp_w.shape[2]
    u = np.finfo(float).eps / 2
    mag = _einsum(np.abs(shp_w).astype(complex), np.abs(table).astype(complex)).real
    bound = 2.0 * np.sqrt(2.0) * (q + 2) * u * mag
    excess = np.abs(got - ref) - bound
    assert np.all(excess <= 0.0), float(np.max(excess))


# A rounding-level change in this block moves SG's buried Z by 1.6e-13 on
# hub_deck(4) (a pairwise-summed, fused spelling against the kernel, Haswell)
# and 1.7e-11 on hub x8, because SG subtracts it from the scaled image; the
# point-matched lane moved 0. 1e-10 sits ~600x above the deck gated here.
_Z_ROUNDING_AMPLIFIED = 1e-10


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


@pytest.mark.parametrize("m,n,q", [(1, 5, 1), (7, 13, 6), (40, 211, 8)])
def test_g1224_r1_the_kernel_is_einsum_on_random_data(m, n, q):
    rng = np.random.default_rng(1224 + m + n + q)
    shp_w = rng.standard_normal((3, n, q)) + 1j * rng.standard_normal((3, n, q))
    table = rng.standard_normal((m, n * q)) + 1j * rng.standard_normal((m, n * q))
    got = _acc.remainder_shape_reduce(shp_w, table)
    assert got.shape == (3, m, n)
    _assert_within_rounding(got, shp_w, table)


@pytest.fixture(scope="module")
def live_chunk():
    """The first (shp_w, table) pair the SG buried replay reduces."""
    seen = []
    real = _acc.remainder_shape_reduce

    def spy(shp_w, table):
        if not seen:
            seen.append((shp_w.copy(), table.copy()))
        return real(shp_w, table)

    mp = pytest.MonkeyPatch()
    mp.setattr(_acc, "remainder_shape_reduce", spy)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _direct_sg(**hub_deck(n_radials=4)).compute_impedance()
    finally:
        mp.undo()
    assert seen, "the replay never reached the reduction"
    return seen[0]


def test_g1224_r1_the_kernel_is_einsum_on_a_live_chunk(live_chunk):
    shp_w, table = live_chunk
    got = _acc.remainder_shape_reduce(shp_w, table)
    _assert_within_rounding(got, shp_w, table)


def test_g1224_r2_an_entry_does_not_see_its_neighbours(live_chunk):
    shp_w, table = live_chunk
    whole = _acc.remainder_shape_reduce(shp_w, table)
    m = table.shape[0]
    for lo, hi in ((0, 1), (1, m // 2), (m // 2, m)):
        if hi <= lo:
            continue
        part = _acc.remainder_shape_reduce(shp_w, np.ascontiguousarray(table[lo:hi]))
        assert np.array_equal(_bits(part), _bits(whole[:, lo:hi]))


@pytest.mark.parametrize("cls", [SinusoidalGalerkinSolver, SinusoidalSolver])
def test_g1224_r3_buried_z_through_the_kernel_is_the_einsum_routes(cls, monkeypatch):
    calls = {"n": 0}
    real = _acc.remainder_shape_reduce

    def spy(*a):
        calls["n"] += 1
        return real(*a)

    def solve():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z, cur = cls(**hub_deck(n_radials=4)).compute_impedance()
        return np.atleast_1d(np.asarray(z)), np.asarray(cur)

    monkeypatch.setattr(_acc, "remainder_shape_reduce", spy)
    z_k, c_k = solve()
    assert calls["n"] > 0, "the kernel never ran"
    monkeypatch.setattr(_sin, "_REPLAY_REDUCE_ACCEL", False)
    calls["n"] = 0
    z_e, c_e = solve()
    assert calls["n"] == 0, "the reference route reached the kernel"
    assert _rel(z_k, z_e) <= _Z_ROUNDING_AMPLIFIED, _rel(z_k, z_e)
    assert _rel(c_k, c_e) <= _Z_ROUNDING_AMPLIFIED, _rel(c_k, c_e)
