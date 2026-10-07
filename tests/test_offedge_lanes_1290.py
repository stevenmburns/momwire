"""The off-edge moment kernel's AVX2 lane path -- momwire#1290.

`seg_seg_full_moments_bspline` (and its complex-k and tiered twins) builds the
off-edge moment tensor every bs2 fill reads: free space, the above-ground
image term and the buried subset fill. The AVX2 build now runs four column
pairs (i, j0..j0+3) at a time, one per vector lane, wherever the four share
a tier of even order that fits one qr chunk; everything else is the per-pair
walk. Each lane's arithmetic is the walk's operation for operation, and
`reference=True` still runs the walk alone, so the gates here are bit
equality against it, not tolerance:

* kernel level, on synthetic blocks built to reach every routing case: whole
  lane groups, a ragged N_j % 4 tail, groups whose four pairs straddle a tier
  boundary, segment lengths that differ in the last bits (the lane table is
  refilled) and ones that repeat (it is held), every order 1..8 (odd orders
  stay on the walk, even ones take lanes), a multi-chunk order, d = 1, 2 and
  3, real and complex k, one rule and a ladder;
* NEGATIVE CONTROL: one ulp on the radius moves the moments, so agreement is
  evidence about the arithmetic, not an accident of the data;
* end to end, a free-space deck (one-tier ladder) and a buried deck (complex
  k, three tiers): Z_in and currents equal the reference's bit for bit, with
  the kernel calls COUNTED, so neither gate passes without the kernel having
  run;
* the lane path is compiled into the x86-64 AVX2 variant (elsewhere both
  spellings are the walk, and the bit gates hold trivially).

On Windows (MSVC, /fp:fast) the lanes run too since momwire#1371, and every
lanes-vs-walk comparison here takes the derived win32 tolerance instead of bit
equality (`assert_lanes_match`, tests/_lane_gate.py). Linux and macOS keep the
bit gates.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

from momwire import _accel
from momwire import _bspline_kernels as BK
from momwire import bspline as B
from _lane_gate import assert_lanes_match

acc = BK._acc

pytestmark = pytest.mark.skipif(
    not (BK._HAVE_BSPLINE_ACCEL and BK._HAVE_BSPLINE_OFFEDGE_CPLX_TIERED_ACCEL),
    reason="needs the C++ off-edge moment kernels",
)

K = 2 * np.pi / (299792458.0 / 7e6)
K_SOIL = complex(K * 3.6, -K * 0.9)  # an in-medium wavenumber, Im k < 0
ENTRIES = (
    "seg_seg_full_moments_bspline",
    "seg_seg_full_moments_bspline_cplx",
    "seg_seg_full_moments_bspline_tiered",
    "seg_seg_full_moments_bspline_cplx_tiered",
)


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def test_the_lane_path_is_compiled_into_the_avx2_variant():
    assert hasattr(acc, "offedge_lanes_1290"), (
        "the .so predates momwire#1290: rebuild (`make build`)"
    )
    if _accel.VARIANT == "avx2":
        assert acc.offedge_lanes_1290


def _segments(seed, n_wires, n_per, *, graded):
    """Straight wires in assorted orientations, meshed uniformly (lengths
    then agree on paper and differ in the last bits) or graded (lengths
    differ outright)."""
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
    # Rows and columns from the same mesh, so near pairs (low tiers) and far
    # pairs (high tiers) both occur, and offset so groups straddle tier
    # boundaries; N_j is chosen off a multiple of four by the callers.
    return sl[:n_i], sr[:n_i], sl[3 : 3 + n_j], sr[3 : 3 + n_j]


def _plain(blk, k, d, n_qp, reference, a2=1e-6):
    t, w = BK._gl01(n_qp)
    fn = (
        acc.seg_seg_full_moments_bspline_cplx
        if isinstance(k, complex)
        else (acc.seg_seg_full_moments_bspline)
    )
    return fn(*blk, a2, k, d, t, w, reference=reference)


def _tiered(blk, k, d, n_qp, ladder, reference, a2=1e-6):
    fn = (
        acc.seg_seg_full_moments_bspline_cplx_tiered
        if isinstance(k, complex)
        else acc.seg_seg_full_moments_bspline_tiered
    )
    return fn(*blk, a2, k, d, *BK._ladder_arrays(n_qp, ladder), reference=reference)


@pytest.mark.parametrize("n_qp", [1, 2, 3, 4, 5, 6, 7, 8, 10])
@pytest.mark.parametrize("k", [K, K_SOIL], ids=["real_k", "complex_k"])
def test_one_rule_is_the_walk_to_the_bit(n_qp, k):
    blk = _block(0, 37, 61)
    want = _plain(blk, k, 2, n_qp, True)
    got = _plain(blk, k, 2, n_qp, False)
    assert_lanes_match(got, want)


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("graded", [False, True])
@pytest.mark.parametrize(
    "k, n_qp, ladder",
    [
        (K, 8, B.DEFAULT_PAIR_ORDER_LADDER),
        (K, 8, ((1.5, 6), (3.0, 3), (6.0, 2))),  # an odd tier among even ones
        (K_SOIL, 32, B.BURIED_PAIR_ORDER_LADDER),
    ],
    ids=["free_ladder", "mixed_parity", "buried_ladder"],
)
def test_a_ladder_is_the_walk_to_the_bit(d, graded, k, n_qp, ladder):
    blk = _block(1, 41, 70, graded=graded)
    want = _tiered(blk, k, d, n_qp, ladder, True)
    got = _tiered(blk, k, d, n_qp, ladder, False)
    assert_lanes_match(got, want)


def test_short_and_empty_column_blocks():
    for n_j in (0, 1, 3, 4, 5):
        blk = _block(2, 9, n_j)
        want = _tiered(blk, K, 2, 8, B.DEFAULT_PAIR_ORDER_LADDER, True)
        got = _tiered(blk, K, 2, 8, B.DEFAULT_PAIR_ORDER_LADDER, False)
        assert got.shape == (3, 3, 9, n_j)
        assert_lanes_match(got, want)


def test_the_gate_sees_one_ulp():
    blk = _block(3, 20, 40)
    want = _plain(blk, K, 2, 4, True)
    # One ulp up on every source end point: most moments must move.
    nudged = (*blk[:3], np.nextafter(blk[3], np.inf))
    got = _plain(nudged, K, 2, 4, False)
    moved = _bits(got) != _bits(want)
    assert moved.mean() > 0.5


def _counting(monkeypatch, reference):
    """Route the four entries through `reference` and count calls."""
    calls = {"n": 0}
    for name in ENTRIES:
        f = functools.partial(getattr(acc, name), reference=reference)

        def counted(*a, _f=f, **k):
            calls["n"] += 1
            return _f(*a, **k)

        monkeypatch.setattr(acc, name, counted)
    return calls


def _solve(make, monkeypatch, reference):
    with monkeypatch.context() as mp:
        calls = _counting(mp, reference)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z, cur = make().compute_impedance()
    return np.atleast_1d(np.asarray(z)), np.asarray(cur), calls["n"]


def _free_array():
    """Three parallel dipoles and a bent wire: far pairs take the order-4
    tier, near ones the base order 8."""
    wires = [np.array([(-5.0, 4.0 * j, 10.0), (5.0, 4.0 * j, 10.0)]) for j in range(3)]
    wires.append(np.array([(-5.0, -4.0, 10.0), (0.0, -4.0, 12.0), (5.0, -4.0, 10.0)]))
    return B.BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[41], [41], [41], [20, 21]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=299792458.0 / 14e6,
        wire_radius=0.001,
    )


def test_a_free_space_solve_is_the_reference_to_the_bit(monkeypatch):
    z_ref, c_ref, n_ref = _solve(_free_array, monkeypatch, True)
    z, c, n = _solve(_free_array, monkeypatch, False)
    assert n_ref == n > 0
    assert_lanes_match(z, z_ref)
    assert_lanes_match(c, c_ref)


def test_a_buried_solve_is_the_reference_to_the_bit(monkeypatch):
    from test_rotational_symmetry_1029 import solver

    def make():
        return solver(4, rotational_symmetry=False)

    z_ref, c_ref, n_ref = _solve(make, monkeypatch, True)
    z, c, n = _solve(make, monkeypatch, False)
    assert n_ref == n > 0
    assert_lanes_match(z, z_ref)
    assert_lanes_match(c, c_ref)
