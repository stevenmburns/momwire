"""The extended kernel on the pair-order ladder -- momwire#1362.

The ladder (momwire#906/#907) runs a far pair at a lower Gauss-Legendre order:
in free space, order 4 instead of 8 beyond sixteen segment lengths. Until
#1362 the extended kernel (EK) refused it, so an EK deck -- every NEC-4/5 deck
since 0.72.0 -- paid four times the points on its far pairs, and bs2 free x16
ran 2.6x threaded NEC-5. The EK fill now takes the deck's ladder, with the
coaxial factor applied at whatever order a pair runs (C++
`seg_seg_full_moments_bspline_ek_tiered`; the numpy twin applies it per tier).
This gives up EK bit-identity by design; EK-off fills are untouched.

The tolerance is DERIVED, not chosen. Measured on Skylake (EK on, ladder vs
the flat order-8 fill, |dZ_in| / |Z_in|):

    deck                              EK, ladder   EK off, ladder
    free x16   (N 2816)               2.8e-14      2.6e-13
    free x32   (N 5632)               3.9e-13      2.3e-13
    above x16  (N 2816, Sommerfeld)   1.8e-13      1.5e-13
    above x32  (N 5632, Sommerfeld)   1.0e-13      4.5e-13
    junction/bend, free  (N 233)      1.6e-15      1.8e-15
    junction/bend, Sommerfeld         9.4e-16      2.7e-15

so the EK ladder moves Z as much as the ladder already moves it with EK
off -- at most 4.5e-13 -- against 4e-5 to 2.4e-4 between EK on and off on
the same decks. Z_TOL = 1e-11 sits 25x over the worst measured EK spread
and two orders under #906's per-pair bar (1e-9, G-906-3, which the per-pair
gate below holds the EK kernel to as well). The solver gates here run the
junction decks and a smaller free array; the x16/x32 decks are too slow for
the suite, and the table above is their record.

Negative controls, so a pass is evidence:
* the EK tiered entry is counted, and its Z is NOT bit-equal to the flat
  fill's -- a ladder that was silently ignored would be;
* per pair, the EK tiered moments differ from the REDUCED tiered moments on
  exactly the eligible pairs, including pairs served by the lower tiers -- a
  kernel that dropped the factor on tiered pairs fails it, and so does the
  numpy-twin gate (the factor there is ~1e-7 against a 1e-13 gate);
* the EK-on and EK-off ladder Z differ by far more than Z_TOL.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

import momwire._bspline_kernels as _bk
from momwire import bspline as B
from momwire._bspline_kernels import (
    _EK,
    _gl01,
    _ladder_arrays,
    _pair_ratio,
    _seg_seg_full_moments_offedge,
)

acc = _bk._acc

pytestmark = pytest.mark.skipif(
    not (_bk._HAVE_BSPLINE_OFFEDGE_EK_TIERED_ACCEL and _bk._HAVE_BSPLINE_ACCEL),
    reason="needs the C++ EK tiered off-edge kernel",
)

Z_TOL = 1e-11
K_REAL = 0.5
A = 1e-3
LADDER_BURIED_SHAPE = ((2.0, 8), (16.0, 4))  # base 32, the #906 study's
LADDER_FREE = B.DEFAULT_PAIR_ORDER_LADDER  # base 8: ((16.0, 4),)
EK_PLAIN = "seg_seg_full_moments_bspline_ek"
EK_TIERED = "seg_seg_full_moments_bspline_ek_tiered"


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _chain(n=40, h=0.1, offset=0.15, tilt=False):
    """Two parallel straight lines of n segments (#906's chain deck): each
    line is one coaxial group, so every ratio bin holds eligible pairs and
    ineligible ones. `tilt` turns the second line so it is not parallel."""
    x = np.arange(n) * h
    sl = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    sr = sl + np.array([h, 0.0, 0.0])
    d2 = np.array([h, 0.3 * h, 0.0]) if tilt else np.array([h, 0.0, 0.0])
    sl2 = np.array([0.0, offset, 0.0]) + np.arange(n)[:, None] * d2
    sr2 = sl2 + d2
    g = np.repeat(np.array([0, 1], dtype=np.int64), n)
    return np.vstack([sl, sl2]), np.vstack([sr, sr2]), g


def _ek(g_i, g_j):
    return _EK(a=None, group_i=g_i, group_j=g_j)


def _numpy(monkeypatch, *args, **kwargs):
    with monkeypatch.context() as mp:
        mp.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_EK_TIERED_ACCEL", False)
        mp.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_EK_ACCEL", False)
        mp.setattr(_bk, "_HAVE_BSPLINE_OFFEDGE_TIERED_ACCEL", False)
        return _seg_seg_full_moments_offedge(*args, **kwargs)


class _Count:
    def __init__(self, monkeypatch, reference=None):
        self.n = dict.fromkeys((EK_PLAIN, EK_TIERED), 0)
        for name in self.n:
            f = getattr(acc, name)
            if reference is not None:
                f = functools.partial(f, reference=reference)

            def counted(*a, _f=f, _name=name, **k):
                self.n[_name] += 1
                return _f(*a, **k)

            monkeypatch.setattr(acc, name, counted)


# --- kernel level ----------------------------------------------------------


@pytest.mark.parametrize(
    "n_qp, ladder", [(32, LADDER_BURIED_SHAPE), (8, LADDER_FREE)], ids=["b32", "f8"]
)
@pytest.mark.parametrize("mixed", [False, True], ids=["scalar-a", "mixed-a"])
def test_the_ek_tiered_accelerator_matches_its_numpy_twin(
    monkeypatch, n_qp, ladder, mixed
):
    sl, sr, g = _chain(tilt=True)
    a = np.where(np.arange(len(sl)) < len(sl) // 2, A, 2 * A) if mixed else A
    with monkeypatch.context() as mp:
        c = _Count(mp)
        got = _seg_seg_full_moments_offedge(
            sl, sr, sl, sr, a, K_REAL, 2, n_qp, ek=_ek(g, g), ladder=ladder
        )
    assert c.n[EK_TIERED] == (2 if mixed else 1) and c.n[EK_PLAIN] == 0, c.n
    ref = _numpy(
        monkeypatch, sl, sr, sl, sr, a, K_REAL, 2, n_qp, ek=_ek(g, g), ladder=ladder
    )
    rel = np.abs(got - ref).max() / np.abs(ref).max()
    assert rel < 1e-13, f"C++ EK tiered vs numpy EK tiered: {rel:.3e}"


def test_one_tier_is_the_plain_ek_entry_bit_for_bit():
    sl, sr, g = _chain()
    t, w = _gl01(8)
    plain = acc.seg_seg_full_moments_bspline_ek(
        sl, sr, sl, sr, A * A, K_REAL, 2, t, w, g, g, A
    )
    one = acc.seg_seg_full_moments_bspline_ek_tiered(
        sl, sr, sl, sr, A * A, K_REAL, 2, *_ladder_arrays(8, ()), g, g, A
    )
    assert np.array_equal(_bits(one), _bits(plain))


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize(
    "n_qp, ladder",
    [(32, LADDER_BURIED_SHAPE), (8, LADDER_FREE), (8, ((1.5, 6), (3.0, 3), (6.0, 2)))],
    ids=["b32", "f8", "mixed_parity"],
)
def test_ek_tiered_lanes_are_the_walk_to_the_bit(d, n_qp, ladder):
    sl, sr, g = _chain(n=37, tilt=True)
    gi = g.copy()
    gi[::5] = -1
    args = (sl, sr, sl[3:], sr[3:], A * A, K_REAL, d, *_ladder_arrays(n_qp, ladder))
    want = acc.seg_seg_full_moments_bspline_ek_tiered(
        *args, gi, g[3:], A, reference=True
    )
    got = acc.seg_seg_full_moments_bspline_ek_tiered(
        *args, gi, g[3:], A, reference=False
    )
    assert np.array_equal(_bits(got), _bits(want))


def test_every_served_ek_pair_is_within_1e_9_of_the_base_order():
    """G-906-3 for the extended kernel."""
    sl, sr, g = _chain()
    ek = _ek(g, g)
    base = _seg_seg_full_moments_offedge(sl, sr, sl, sr, A, K_REAL, 2, 32, ek=ek)
    got = _seg_seg_full_moments_offedge(
        sl, sr, sl, sr, A, K_REAL, 2, 32, ek=ek, ladder=LADDER_BURIED_SHAPE
    )
    scale = np.abs(base).reshape(9, -1).max(axis=1)[:, None, None]
    err = (np.abs(got - base).reshape(9, *base.shape[2:]) / scale).max(axis=0)
    ratio = _pair_ratio(sl, sr, sl, sr)
    assert err.max() < 1e-9, f"worst pair {err.max():.3e}"
    served_8 = (ratio >= 2.0) & (ratio < 16.0)
    assert err[served_8].max() < 1e-11, f"order-8 tier worst {err[served_8].max():.3e}"
    assert np.array_equal(err[ratio < 2.0], np.zeros(int((ratio < 2.0).sum())))


def test_the_factor_is_applied_on_the_tiered_pairs_and_only_the_eligible_ones():
    sl, sr, g = _chain(tilt=True)
    args = (sl, sr, sl, sr, A * A, K_REAL, 2, *_ladder_arrays(32, LADDER_BURIED_SHAPE))
    ek = acc.seg_seg_full_moments_bspline_ek_tiered(*args, g, g, A)
    red = acc.seg_seg_full_moments_bspline_tiered(*args)
    eligible = g[:, None] == g[None, :]
    moved = np.any(ek != red, axis=(0, 1))
    assert np.array_equal(moved, eligible)
    ratio = _pair_ratio(sl, sr, sl, sr)
    for lo, hi in ((2.0, 16.0), (16.0, np.inf)):  # the order-8 and order-4 tiers
        tier = (ratio >= lo) & (ratio < hi)
        assert (tier & eligible).any() and (tier & ~eligible).any()
    # An all-ineligible EK block is the reduced tiered kernel, bit for bit.
    none = np.full_like(g, -1)
    assert np.array_equal(
        _bits(acc.seg_seg_full_moments_bspline_ek_tiered(*args, none, g, A)), _bits(red)
    )


# --- solver level ----------------------------------------------------------

LAM7 = 299792458.0 / 7e6


def _free_array():
    """Three parallel dipoles and a bent wire (#1290's deck), EK on: far
    pairs take the order-4 tier."""
    wires = [np.array([(-5.0, 4.0 * j, 10.0), (5.0, 4.0 * j, 10.0)]) for j in range(3)]
    wires.append(np.array([(-5.0, -4.0, 10.0), (0.0, -4.0, 12.0), (5.0, -4.0, 10.0)]))
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[41], [41], [41], [20, 21]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=299792458.0 / 14e6,
        wire_radius=0.001,
    )


def _junction(ground):
    """An elevated vertical with four drooping radials at one junction, a bent
    hat and a parasitic: coaxial groups split at the bend and the junction."""
    h = 6.0
    base = (0.0, 0.0, h)
    wires = [np.array([base, (0.0, 0.0, h + 9.8)])]
    for c, s in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        wires.append(np.array([base, (9.0 * c, 9.0 * s, h - 3.0)]))
    wires.append(
        np.array([(0.0, 0.0, h + 9.8), (2.0, 0.0, h + 9.8), (2.0, 2.0, h + 9.8)])
    )
    wires.append(np.array([(4.0, 3.0, h), (4.0, 3.0, h + 10.0)]))
    d = dict(
        wires=wires,
        n_per_edge_per_wire=[[37], [33], [33], [33], [33], [9, 10], [38]],
        junctions=[
            [(0, "start")] + [(i, "start") for i in range(1, 5)],
            [(0, "end"), (5, "start")],
        ],
        feeds=[(0, 0.3, 1 + 0j)],
        wavelength=LAM7,
        wire_radius=0.0015,
    )
    if ground:
        d.update(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")
    return d


DECKS = {
    "free_array": _free_array,
    "junction_free": lambda: _junction(False),
    "junction_sommerfeld": lambda: _junction(True),
}


def _z(monkeypatch, deck, **kw):
    with monkeypatch.context() as mp:
        c = _Count(mp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z, _ = B.BSplineSolver(**deck, **kw).compute_impedance()
    return complex(np.atleast_1d(z)[0]), c.n


@pytest.mark.parametrize("name", list(DECKS))
def test_an_ek_solve_on_the_ladder_is_the_flat_fill_within_the_derived_tolerance(
    monkeypatch, name
):
    deck = DECKS[name]()
    z_lad, n_lad = _z(monkeypatch, deck, extended_kernel=True)
    z_flat, n_flat = _z(monkeypatch, deck, extended_kernel=True, pair_order_ladder=())
    rel = abs(z_lad - z_flat) / abs(z_flat)
    assert rel < Z_TOL, f"{name}: EK ladder vs flat {rel:.3e}"
    # Negative controls: the ladder route ran and moved the bits (an ignored
    # ladder would be the flat fill exactly) ...
    assert n_lad[EK_TIERED] >= 1 and n_flat[EK_TIERED] == 0, (n_lad, n_flat)
    assert z_lad != z_flat
    # ... and EK itself is still on: the ladder fill with EK off sits far away.
    z_off, _ = _z(monkeypatch, deck, extended_kernel=False)
    assert abs(z_lad - z_off) / abs(z_off) > 1e3 * Z_TOL


def test_an_ek_solve_on_the_ladder_is_the_reference_walk_to_the_bit(monkeypatch):
    deck = _free_array()

    def run(reference):
        with monkeypatch.context() as mp:
            c = _Count(mp, reference=reference)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                z, cur = B.BSplineSolver(
                    **deck, extended_kernel=True
                ).compute_impedance()
        return np.atleast_1d(np.asarray(z)), np.asarray(cur), c.n[EK_TIERED]

    z_ref, c_ref, n_ref = run(True)
    z, c, n = run(False)
    assert n_ref == n > 0
    assert np.array_equal(_bits(z), _bits(z_ref))
    assert np.array_equal(_bits(c), _bits(c_ref))
