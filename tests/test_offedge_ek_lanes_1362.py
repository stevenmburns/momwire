"""The extended-kernel off-edge moments on the reduced kernel's lanes -- momwire#1362.

0.72.0 made the extended kernel (EK) the default for NEC-4/5 decks, and bs2's
free and above-ground fills went 2-3x slower: `seg_seg_full_moments_bspline_ek`
was a separate per-pair loop without the t-outer stage 2 (8c311afb) or the
AVX2 four-column lanes (momwire#1290) the reduced kernel had grown. It is now
that kernel with `EK = true`: the coaxial factor multiplies G between stage 1
and stage 2 on the pairs whose group labels match. Each pair's arithmetic is
the old EK kernel's, so the solver's Z is bit-identical to the pre-#1362 build
(measured on the issue's free/above decks and a junction deck; that gate needs
two builds and lives in the PR, not here). What a single build can pin, and
this file does, is the lane path against the walk (`reference=True`) to the
bit:

* kernel level, on synthetic blocks whose lane groups straddle eligibility
  boundaries (labels per wire, some -1, offset so a group of four columns
  covers two wires), every order 1..8 and 10, d = 1, 2, 3;
* an all-ineligible EK block IS the reduced kernel's, bit for bit (the
  momwire#781 equality, now structural);
* NEGATIVE CONTROLS: the factor moves the eligible pairs and only them, and
  one part per million on the EK radius moves the eligible moments (one ulp
  does not: fac - 1 is ~1e-6 here, so an ulp of a lands below fac's own
  ulp), so agreement is evidence about the arithmetic, not an accident of
  the data;
* end to end, EK-on solves of a free-space array and a junction deck: Z_in
  and currents equal the reference's bit for bit, with the EK entry COUNTED.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

from momwire import _bspline_kernels as BK
from momwire import bspline as B

acc = BK._acc

pytestmark = pytest.mark.skipif(
    not (BK._HAVE_BSPLINE_ACCEL and BK._HAVE_BSPLINE_OFFEDGE_EK_ACCEL),
    reason="needs the C++ off-edge moment kernels",
)

K = 2 * np.pi / (299792458.0 / 7e6)
A = 0.002


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _block(seed, n_i, n_j, n_per=23):
    """Four straight wires; each wire is one coaxial group, every third wire's
    segments in rows carry -1 (never eligible). Columns are offset by three so
    no lane group of four lines up with a wire's start."""
    rng = np.random.default_rng(seed)
    left, right, lab = [], [], []
    for w in range(4):
        a = rng.uniform(-6.0, 6.0, size=3)
        b = a + rng.normal(size=3) * 4.0
        u = np.linspace(0.0, 1.0, n_per + 1)
        pts = a[None, :] + u[:, None] * (b - a)[None, :]
        left.append(pts[:-1])
        right.append(pts[1:])
        lab.append(np.full(n_per, w, dtype=np.int64))
    sl, sr, g = np.concatenate(left), np.concatenate(right), np.concatenate(lab)
    gi = g[:n_i].copy()
    gi[::3] = -1
    return (
        (sl[:n_i], sr[:n_i], sl[3 : 3 + n_j], sr[3 : 3 + n_j]),
        gi,
        g[3 : 3 + n_j].copy(),
    )


def _ek(blk, gi, gj, d, n_qp, reference, a_ek=A):
    t, w = BK._gl01(n_qp)
    return acc.seg_seg_full_moments_bspline_ek(
        *blk, A * A, K, d, t, w, gi, gj, a_ek, reference=reference
    )


def _reduced(blk, d, n_qp):
    t, w = BK._gl01(n_qp)
    return acc.seg_seg_full_moments_bspline(*blk, A * A, K, d, t, w)


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("n_qp", [1, 2, 3, 4, 5, 6, 7, 8, 10])
def test_ek_lanes_are_the_walk_to_the_bit(n_qp, d):
    blk, gi, gj = _block(0, 47, 61)
    want = _ek(blk, gi, gj, d, n_qp, True)
    got = _ek(blk, gi, gj, d, n_qp, False)
    assert np.array_equal(_bits(got), _bits(want))


def test_short_and_empty_column_blocks():
    for n_j in (0, 1, 3, 4, 5):
        blk, gi, gj = _block(2, 9, n_j)
        want = _ek(blk, gi, gj, 2, 8, True)
        got = _ek(blk, gi, gj, 2, 8, False)
        assert got.shape == (3, 3, 9, n_j)
        assert np.array_equal(_bits(got), _bits(want))


@pytest.mark.parametrize("reference", [False, True])
def test_an_all_ineligible_block_is_the_reduced_kernel(reference):
    blk, gi, gj = _block(1, 40, 57)
    none = np.full_like(gi, -1)
    got = _ek(blk, none, gj, 2, 8, reference)
    assert np.array_equal(_bits(got), _bits(_reduced(blk, 2, 8)))


def test_the_factor_moves_the_eligible_pairs_and_only_them():
    blk, gi, gj = _block(3, 40, 57)
    ek = _ek(blk, gi, gj, 2, 8, False)
    red = _reduced(blk, 2, 8)
    eligible = (gi[:, None] == gj[None, :]) & (gi[:, None] >= 0)
    assert eligible.any() and (~eligible).any()
    moved = np.any(ek != red, axis=(0, 1))
    assert np.array_equal(moved, eligible)
    # Lane groups straddling eligibility are present (a group of four columns
    # with both kinds of pair), so the per-lane blend is exercised.
    n4 = (eligible.shape[1] // 4) * 4
    g = eligible[:, :n4].reshape(eligible.shape[0], -1, 4)
    assert np.any(g.any(axis=2) & ~g.all(axis=2))


def test_the_gate_sees_a_one_ppm_ek_radius():
    blk, gi, gj = _block(4, 30, 44)
    want = _ek(blk, gi, gj, 2, 8, True)
    got = _ek(blk, gi, gj, 2, 8, False, a_ek=A * (1 + 1e-6))
    eligible = (gi[:, None] == gj[None, :]) & (gi[:, None] >= 0)
    moved = np.any(got != want, axis=(0, 1))
    assert moved[eligible].mean() > 0.5
    assert not moved[~eligible].any()


def _counting(monkeypatch, reference):
    """Route the EK entries (one rule and, since the EK fill takes the
    pair-order ladder, tiered) through `reference` and count calls."""
    calls = {"n": 0}
    for name in (
        "seg_seg_full_moments_bspline_ek",
        "seg_seg_full_moments_bspline_ek_tiered",
    ):
        if not hasattr(acc, name):
            continue
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
            s = make()
            assert s.extended_kernel
            z, cur = s.compute_impedance()
    return np.atleast_1d(np.asarray(z)), np.asarray(cur), calls["n"]


def _free_array():
    """Three parallel dipoles and a bent wire, EK on."""
    wires = [np.array([(-5.0, 4.0 * j, 10.0), (5.0, 4.0 * j, 10.0)]) for j in range(3)]
    wires.append(np.array([(-5.0, -4.0, 10.0), (0.0, -4.0, 12.0), (5.0, -4.0, 10.0)]))
    return B.BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[41], [41], [41], [20, 21]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=299792458.0 / 14e6,
        wire_radius=0.001,
        extended_kernel=True,
    )


def _junction():
    """A vertical with four drooping radials at one junction and a bent hat:
    coaxial groups split at the bend and the junction."""
    base = (0.0, 0.0, 6.0)
    wires = [np.array([base, (0.0, 0.0, 15.8)])]
    for c, s in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        wires.append(np.array([base, (9.0 * c, 9.0 * s, 3.0)]))
    wires.append(np.array([(0.0, 0.0, 15.8), (2.0, 0.0, 15.8), (2.0, 2.0, 15.8)]))
    return B.BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[19], [17], [17], [17], [17], [5, 6]],
        junctions=[
            [(0, "start")] + [(i, "start") for i in range(1, 5)],
            [(0, "end"), (5, "start")],
        ],
        feeds=[(0, 0.3, 1 + 0j)],
        wavelength=299792458.0 / 7e6,
        wire_radius=0.0015,
        extended_kernel=True,
    )


@pytest.mark.parametrize(
    "make", [_free_array, _junction], ids=["free_array", "junction"]
)
def test_an_ek_solve_is_the_reference_to_the_bit(monkeypatch, make):
    z_ref, c_ref, n_ref = _solve(make, monkeypatch, True)
    z, c, n = _solve(make, monkeypatch, False)
    assert n_ref == n > 0
    assert np.array_equal(_bits(z), _bits(z_ref))
    assert np.array_equal(_bits(c), _bits(c_ref))
