"""The sinusoidal tiered pair-moment kernel's AVX2 lane path.

`seg_seg_full_moments_sinusoidal_tiered` builds the pair moments SG's
Galerkin fill reads (free space, the image term, the buried in-medium term).
The AVX2 build runs four column pairs (i, j0..j0+3) at a time, one per vector
lane, wherever the four share a tier of even order that fits one qr chunk --
the B-spline off-edge kernel's lanes (momwire#1290) on this basis's
contraction -- and walks everything else. Each lane is the walk's arithmetic
operation for operation, and `reference=True` runs the walk alone, so the
gates are bit equality against it:

* kernel level: whole lane groups, a ragged N_j % 4 tail, groups straddling a
  tier boundary, odd orders (walked), a multi-chunk order (walked), real and
  complex k, one rule and the buried ladder;
* NEGATIVE CONTROL: one ulp on an end point moves the moments;
* end to end: SG solves (free space and buried) equal the reference's Z_in
  and currents bit for bit, with the kernel calls COUNTED.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

from momwire import _bspline_kernels as BK
from momwire import _sinusoidal_mp as SMP
from momwire import bspline as B
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

acc = BK._acc

pytestmark = pytest.mark.skipif(
    acc is None or not hasattr(acc, "seg_seg_full_moments_sinusoidal_tiered"),
    reason="needs the C++ sinusoidal tiered kernel",
)

K = 2 * np.pi / (299792458.0 / 7e6)
K_SOIL = complex(K * 3.6, -K * 0.9)


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _segments(seed, n_wires, n_per, *, graded):
    rng = np.random.default_rng(seed)
    left, right = [], []
    for _ in range(n_wires):
        a = rng.uniform(-6.0, 6.0, size=3)
        b = a + rng.normal(size=3) * 4.0
        u = np.linspace(0.0, 1.0, n_per + 1)
        if graded:
            u = u**1.3
        pts = a[None, :] + u[:, None] * (b - a)[None, :]
        left.append(pts[:-1])
        right.append(pts[1:])
    return np.concatenate(left), np.concatenate(right)


def _block(seed, n_i, n_j, graded=False):
    sl, sr = _segments(seed, 4, 23, graded=graded)
    return sl[:n_i], sr[:n_i], sl[3 : 3 + n_j], sr[3 : 3 + n_j]


def _tiered(blk, k, n_qp, ladder, reference, a2=1e-6):
    return acc.seg_seg_full_moments_sinusoidal_tiered(
        *blk, a2, complex(k), *BK._ladder_arrays(n_qp, ladder), reference=reference
    )


@pytest.mark.parametrize("n_qp", [1, 2, 3, 4, 5, 6, 7, 8, 10])
@pytest.mark.parametrize("k", [K, K_SOIL], ids=["real_k", "complex_k"])
def test_one_rule_is_the_walk_to_the_bit(n_qp, k):
    blk = _block(0, 37, 61)
    want = _tiered(blk, k, n_qp, (), True)
    got = _tiered(blk, k, n_qp, (), False)
    assert np.array_equal(_bits(got), _bits(want))


@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize(
    "k, n_qp, ladder",
    [
        (K, 8, B.DEFAULT_PAIR_ORDER_LADDER),
        (K, 8, ((1.5, 6), (3.0, 3), (6.0, 2))),  # an odd tier among even ones
        (K_SOIL, 8, B.BURIED_PAIR_ORDER_LADDER),
        (K_SOIL, 32, B.BURIED_PAIR_ORDER_LADDER),  # a multi-chunk base order
    ],
    ids=["free_ladder", "mixed_parity", "buried_ladder", "buried_ladder_q32"],
)
def test_a_ladder_is_the_walk_to_the_bit(graded, k, n_qp, ladder):
    blk = _block(1, 41, 70, graded=graded)
    want = _tiered(blk, k, n_qp, ladder, True)
    got = _tiered(blk, k, n_qp, ladder, False)
    assert np.array_equal(_bits(got), _bits(want))


def test_short_and_empty_column_blocks():
    for n_j in (0, 1, 3, 4, 5):
        blk = _block(2, 9, n_j)
        want = _tiered(blk, K_SOIL, 8, B.BURIED_PAIR_ORDER_LADDER, True)
        got = _tiered(blk, K_SOIL, 8, B.BURIED_PAIR_ORDER_LADDER, False)
        assert got.shape == (3, 3, 9, n_j)
        assert np.array_equal(_bits(got), _bits(want))


def test_the_gate_sees_one_ulp():
    blk = _block(3, 20, 40)
    want = _tiered(blk, K_SOIL, 8, (), True)
    nudged = (*blk[:3], np.nextafter(blk[3], np.inf))
    got = _tiered(nudged, K_SOIL, 8, (), False)
    assert (_bits(got) != _bits(want)).mean() > 0.5


def _solve(make, monkeypatch, reference):
    calls = {"n": 0}
    f = functools.partial(
        acc.seg_seg_full_moments_sinusoidal_tiered, reference=reference
    )

    def counted(*a, **k):
        calls["n"] += 1
        return f(*a, **k)

    with monkeypatch.context() as mp:
        mp.setattr(acc, "seg_seg_full_moments_sinusoidal_tiered", counted)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z, cur = make().compute_impedance()
    return np.atleast_1d(np.asarray(z)), np.asarray(cur), calls["n"]


def _free():
    wires = [np.array([(-5.0, 4.0 * j, 10.0), (5.0, 4.0 * j, 10.0)]) for j in range(3)]
    return SinusoidalGalerkinSolver(
        wires=wires,
        n_per_edge_per_wire=[[41], [41], [41]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=299792458.0 / 14e6,
        wire_radius=0.001,
    )


def _buried():
    """A buried dipole in soil A: complex k, the buried ladder."""
    return SinusoidalGalerkinSolver(
        wires=[np.array([(-2.0, 0.0, -0.3), (2.0, 0.0, -0.3)])],
        n_per_edge_per_wire=[[41]],
        feeds=[(0, 2.0, 1 + 0j)],
        wavelength=299792458.0 / 7e6,
        wire_radius=0.001,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )


@pytest.mark.parametrize("make", [_free, _buried], ids=["free", "buried"])
def test_a_solve_is_the_reference_to_the_bit(monkeypatch, make):
    assert SMP._HAVE_SIN_TIERED
    z_ref, c_ref, n_ref = _solve(make, monkeypatch, True)
    z, c, n = _solve(make, monkeypatch, False)
    assert n_ref == n > 0
    assert np.array_equal(_bits(z), _bits(z_ref))
    assert np.array_equal(_bits(c), _bits(c_ref))
