"""The remainder windows' arc-moment reduction in C++ (momwire#1290).

`Remainder.field_windows` and `RemainderBelow.field_windows` reduce each
window's projected table over its q source nodes against the real moment
weights, `mom[o, j, p] = sum_k proj[o, j*q + k] * W[p, j, k]`, which numpy's
einsum did single-threaded (0.2 s of razor buried x16, 0.8 s of invl x32 on
Haswell). `remainder_moment_reduce` is the same sum in einsum's arithmetic
-- W cast to complex, the unfused complex product per node, added k in order
from +0.0 -- threaded over observer rows. einsum stays reachable
(`_potential_ground._MOMENT_REDUCE = False`) and is the reference.

On x86-64 that is einsum's bits: razor buried/inverted-L x4, x8 and x16 and
invl x32 Z and currents equal main's as uint64 (measured by script at the
PR). As with `remainder_shape_reduce` (test_replay_reduce_1224), it is NOT
gated to the bit against einsum here, because einsum's arithmetic is the
numpy build's (a contracting build fuses it): kernel against einsum is a
derived tolerance on every platform, the kernel against itself is uint64,
and whole razor fills through the kernel are held to the einsum route's at
the bit on x86-64 Linux/macOS only (MSVC's /fp:fast contracts).
"""

from __future__ import annotations

import platform
import sys

import numpy as np
import pytest

from momwire import _potential_ground as pg
from momwire.razor import RazorSolver

from test_crossing_serve_524 import hub_deck, invl_deck

_acc = pg._acc
pytestmark = pytest.mark.skipif(
    _acc is None or not hasattr(_acc, "remainder_moment_reduce"),
    reason="the .so predates the moment reduction: rebuild",
)


def _einsum(proj, W):
    n, q = W.shape[1:]
    return np.einsum("ojq,pjq->ojp", proj.reshape(proj.shape[0], n, q), W)


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _assert_within_rounding(got, proj, W):
    """|kernel - einsum| per entry within twice the q-term dot product's
    rounding bound, 2 sqrt(2) (q + 2) u sum_k |w_k| |p_k|."""
    ref = _einsum(proj, W)
    q = W.shape[2]
    u = np.finfo(float).eps / 2
    mag = _einsum(np.abs(proj).astype(complex), np.abs(W)).real
    bound = 2.0 * np.sqrt(2.0) * (q + 2) * u * mag
    excess = np.abs(got - ref) - bound
    assert np.all(excess <= 0.0), float(np.max(excess))


@pytest.mark.parametrize("m,n,q,p", [(1, 5, 1, 1), (7, 13, 6, 2), (40, 211, 8, 3)])
def test_the_kernel_is_einsum_on_random_data(m, n, q, p):
    rng = np.random.default_rng(1290 + m + n + q)
    W = rng.standard_normal((p, n, q))
    proj = rng.standard_normal((m, n * q)) + 1j * rng.standard_normal((m, n * q))
    proj[0, :3] = [0.0, -0.0 + 0j, complex(-0.0, -0.0)]
    got = _acc.remainder_moment_reduce(proj, W)
    assert got.shape == (m, n, p)
    _assert_within_rounding(got, proj, W)


def test_an_entry_does_not_see_its_neighbours():
    rng = np.random.default_rng(7)
    W = rng.standard_normal((2, 50, 4))
    proj = rng.standard_normal((23, 200)) + 1j * rng.standard_normal((23, 200))
    whole = _acc.remainder_moment_reduce(proj, W)
    for lo, hi in ((0, 1), (1, 11), (11, 23)):
        part = _acc.remainder_moment_reduce(np.ascontiguousarray(proj[lo:hi]), W)
        assert np.array_equal(_bits(part), _bits(whole[lo:hi]))


DECKS = {
    "buried": lambda: hub_deck(n_radials=4),
    "invl": lambda: invl_deck(n_radials=4),
}


@pytest.mark.parametrize("deck", sorted(DECKS))
def test_razor_fills_through_the_kernel_are_the_einsum_routes(deck, monkeypatch):
    calls = {"n": 0}
    real = _acc.remainder_moment_reduce

    def spy(*a):
        calls["n"] += 1
        return real(*a)

    def fill():
        s = RazorSolver(**DECKS[deck](), nec5_quadrature=True)
        return s._assemble_Z(s._build_geometry(), s.k)

    monkeypatch.setattr(_acc, "remainder_moment_reduce", spy)
    z_k = fill()
    assert calls["n"] > 0, "the kernel never ran"
    monkeypatch.setattr(pg, "_MOMENT_REDUCE", False)
    calls["n"] = 0
    z_e = fill()
    assert calls["n"] == 0, "the reference route reached the kernel"
    # x86-64 GCC/clang builds do not contract (-ffp-contract=off); MSVC's
    # /fp:fast does, and the Windows lane carries a derived tolerance.
    if platform.machine().lower() in ("x86_64", "amd64") and sys.platform != "win32":
        assert np.array_equal(_bits(z_k), _bits(z_e))
    else:
        rel = np.max(np.abs(z_k - z_e)) / np.max(np.abs(z_e))
        assert rel <= 1e-12, rel
