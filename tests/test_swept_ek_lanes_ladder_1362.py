"""The frequency-swept extended-kernel off-edge fill -- momwire#1362.

`seg_seg_full_moments_bspline_swept_ek` was the last EK off-edge fill on the
pre-#1290 per-pair loop: no t-outer stage 2, no AVX2 four-column lanes, no
pair-order ladder. Measured on Haswell (free x16, N 2816, 11 frequencies, 4
threads) it was 72 % of `compute_impedance_swept`, and the sweep took 2.3x
as long as eleven separate per-k solves. It now has all three:

* the walk (`reference=True`) is the pre-#1362 kernel's arithmetic with
  stage 2 t-outer, which reorders work only ACROSS moment chains -- gated
  against the pre-#1362 build to the bit in the PR (two builds; not here);
* the lanes are the walk to the bit (kernel level, every order 1..8 and 10,
  d = 1, 2, 3, on blocks whose lane groups straddle eligibility);
* at one k, a swept moment IS the single-k EK entry's at that k, bit for bit
  (one qr chunk; a multi-chunk order keeps the swept kernel's own
  chunk-by-chunk sums, as it always has);
* the ladder (`seg_seg_full_moments_bspline_swept_ek_tiered`): one tier is
  the flat entry bit for bit, and each k's moments are the single-k
  `seg_seg_full_moments_bspline_ek_tiered`'s -- including across the
  per-pair phase guard (momwire#920), which a sweep answers per k.

The reduced (EK-off) sweep takes no ladder and is untouched; it is pinned
here only as the all-ineligible identity and as "the ladder does not reach
it".

Solver-level tolerance, DERIVED as #1365 derived it (EK swept, ladder vs flat,
max over 11 frequencies of |dZ_in| / |Z_in|, Haswell):

    deck                         EK swept ladder vs flat   EK on vs off
    free_array   (N 164)         5.4e-14                   5.7e-5
    junction, free (N 226)       4.8e-15                   3.1e-5
    junction, PEC ground         2.1e-14                   3.1e-5
    free x4      (N 704)         1.1e-13                   2.7e-5
    free x16     (N 2816)        7.9e-13                   4.0e-5

-- the size of the per-k ladder's own movement (#1365's table: up to 4.5e-13
with EK off), against 3e-5 or more between EK on and off; x4 over a PEC
ground measured 1.0e-12. Z_TOL = 1e-11, the #1365 bar, sits an order over
the worst of these and two over the test decks below.

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

from momwire import _bspline_kernels as BK
from momwire import bspline as B
from _lane_gate import WIN32, assert_lanes_match, moved_pairs

acc = BK._acc

pytestmark = pytest.mark.skipif(
    not (
        BK._HAVE_BSPLINE_OFFEDGE_SWEPT_EK_ACCEL
        and BK._HAVE_BSPLINE_OFFEDGE_SWEPT_EK_TIERED_ACCEL
        and BK._HAVE_BSPLINE_OFFEDGE_EK_TIERED_ACCEL
    ),
    reason="needs the C++ swept EK off-edge kernels",
)

Z_TOL = 1e-11
A = 0.002
KS = 2 * np.pi / (299792458.0 / np.array([6.5e6, 7e6, 7.4e6]))
SWEPT = "seg_seg_full_moments_bspline_swept_ek"
SWEPT_TIERED = "seg_seg_full_moments_bspline_swept_ek_tiered"


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _block(seed, n_i, n_j, n_per=23):
    """Four straight wires, one coaxial group each, every third row -1;
    columns offset by three so lane groups straddle wire boundaries."""
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


def _swept(blk, gi, gj, d, n_qp, reference=False, a_ek=A, ks=KS):
    t, w = BK._gl01(n_qp)
    return acc.seg_seg_full_moments_bspline_swept_ek(
        *blk, A * A, ks, d, t, w, gi, gj, a_ek, reference=reference
    )


# --- kernel level: lanes and walk ------------------------------------------


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("n_qp", [1, 2, 3, 4, 5, 6, 7, 8, 10])
def test_swept_ek_lanes_are_the_walk_to_the_bit(n_qp, d):
    blk, gi, gj = _block(0, 47, 61)
    want = _swept(blk, gi, gj, d, n_qp, reference=True)
    got = _swept(blk, gi, gj, d, n_qp)
    assert_lanes_match(got, want)


def test_short_and_empty_column_blocks():
    for n_j in (0, 1, 3, 4, 5):
        blk, gi, gj = _block(2, 9, n_j)
        want = _swept(blk, gi, gj, 2, 8, reference=True)
        got = _swept(blk, gi, gj, 2, 8)
        assert got.shape == (len(KS), 3, 3, 9, n_j)
        assert_lanes_match(got, want)


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("n_qp", [2, 5, 8])
def test_each_k_is_the_single_k_ek_entry_to_the_bit(n_qp, d):
    """One qr chunk (n_qp <= 8): the sweep's moments at k are what the
    single-k EK entry returns at k -- the same R, G, factor and chains."""
    blk, gi, gj = _block(5, 31, 42)
    got = _swept(blk, gi, gj, d, n_qp)
    t, w = BK._gl01(n_qp)
    for kk, k in enumerate(KS):
        one = acc.seg_seg_full_moments_bspline_ek(*blk, A * A, k, d, t, w, gi, gj, A)
        assert_lanes_match(got[kk], one, kk)


@pytest.mark.parametrize("reference", [False, True])
def test_an_all_ineligible_block_is_the_reduced_swept_kernel(reference):
    blk, gi, gj = _block(1, 40, 57)
    none = np.full_like(gi, -1)
    got = _swept(blk, none, gj, 2, 8, reference=reference)
    t, w = BK._gl01(8)
    red = acc.seg_seg_full_moments_bspline_swept(*blk, A * A, KS, 2, t, w)
    assert_lanes_match(got, red)


def test_the_factor_moves_the_eligible_pairs_and_only_them():
    blk, gi, gj = _block(3, 40, 57)
    ek = _swept(blk, gi, gj, 2, 8)
    t, w = BK._gl01(8)
    red = acc.seg_seg_full_moments_bspline_swept(*blk, A * A, KS, 2, t, w)
    eligible = (gi[:, None] == gj[None, :]) & (gi[:, None] >= 0)
    assert eligible.any() and (~eligible).any()
    moved = moved_pairs(ek, red, (0, 1, 2), eligible)
    assert np.array_equal(moved, eligible)
    n4 = (eligible.shape[1] // 4) * 4
    g = eligible[:, :n4].reshape(eligible.shape[0], -1, 4)
    assert np.any(g.any(axis=2) & ~g.all(axis=2))


def test_the_gate_sees_a_one_ppm_ek_radius():
    blk, gi, gj = _block(4, 30, 44)
    # The walk is the reference off Windows. On win32 the lanes already differ
    # from it at the rounding level (/fp:fast), which would drown a one-ppm
    # change; there the control holds the path fixed and compares lanes
    # with lanes, so a moved bit is the radius reaching the lanes.
    want = _swept(blk, gi, gj, 2, 8, reference=not WIN32)
    got = _swept(blk, gi, gj, 2, 8, a_ek=A * (1 + 1e-6))
    eligible = (gi[:, None] == gj[None, :]) & (gi[:, None] >= 0)
    moved = np.any(got != want, axis=(0, 1, 2))
    assert moved[eligible].mean() > 0.5
    assert not moved[~eligible].any()


# --- kernel level: the ladder -----------------------------------------------

LADDERS = [
    (8, ((16.0, 4),)),
    (8, ((1.5, 6), (3.0, 3), (6.0, 2))),
]
K_LAD = np.array([0.4, 0.5, 0.55])
A_LAD = 1e-3


def _chain(n=37, h=0.1, offset=0.15):
    """Two lines, the second tilted: every ratio bin holds eligible and
    ineligible pairs (#906's chain deck)."""
    x = np.arange(n) * h
    sl = np.stack([x, np.zeros(n), np.zeros(n)], axis=1)
    sr = sl + np.array([h, 0.0, 0.0])
    d2 = np.array([h, 0.3 * h, 0.0])
    sl2 = np.array([0.0, offset, 0.0]) + np.arange(n)[:, None] * d2
    sr2 = sl2 + d2
    g = np.repeat(np.array([0, 1], dtype=np.int64), n)
    return np.vstack([sl, sl2]), np.vstack([sr, sr2]), g


def _tiered_args(n_qp, ladder):
    sl, sr, g = _chain()
    gi = g.copy()
    gi[::5] = -1
    return (
        (sl, sr, sl[3:], sr[3:], A_LAD * A_LAD),
        BK._ladder_arrays(n_qp, ladder),
        gi,
        g[3:],
    )


@pytest.mark.parametrize("d", [1, 2, 3])
@pytest.mark.parametrize("n_qp, ladder", LADDERS, ids=["f8", "mixed_parity"])
def test_tiered_lanes_walk_and_single_k(d, n_qp, ladder):
    geo, lad, gi, gj = _tiered_args(n_qp, ladder)
    want = acc.seg_seg_full_moments_bspline_swept_ek_tiered(
        *geo, K_LAD, d, *lad, gi, gj, A_LAD, reference=True
    )
    got = acc.seg_seg_full_moments_bspline_swept_ek_tiered(
        *geo, K_LAD, d, *lad, gi, gj, A_LAD
    )
    assert_lanes_match(got, want)
    for kk, k in enumerate(K_LAD):
        one = acc.seg_seg_full_moments_bspline_ek_tiered(
            *geo, k, d, *lad, gi, gj, A_LAD
        )
        assert_lanes_match(got[kk], one, kk)
    # Negative control: the ladder moved the moments (an ignored ladder
    # would be the flat entry exactly).
    t, w = BK._gl01(n_qp)
    flat = acc.seg_seg_full_moments_bspline_swept_ek(
        *geo, K_LAD, d, t, w, gi, gj, A_LAD
    )
    assert not np.array_equal(got, flat)


def test_one_tier_is_the_flat_swept_entry_bit_for_bit():
    geo, lad, gi, gj = _tiered_args(8, ())
    one = acc.seg_seg_full_moments_bspline_swept_ek_tiered(
        *geo, K_LAD, 2, *lad, gi, gj, A_LAD
    )
    t, w = BK._gl01(8)
    flat = acc.seg_seg_full_moments_bspline_swept_ek(
        *geo, K_LAD, 2, t, w, gi, gj, A_LAD
    )
    assert_lanes_match(one, flat)


def test_the_factor_is_applied_on_the_tiered_pairs():
    """A kernel that dropped the factor on the lower tiers fails this: per
    pair, EK tiered differs from the reduced single-k tiered kernel exactly
    on the eligible pairs, in every tier."""
    sl, sr, g = _chain()
    lad = BK._ladder_arrays(8, ((1.5, 6), (3.0, 4)))
    geo = (sl, sr, sl, sr, A_LAD * A_LAD)
    ek = acc.seg_seg_full_moments_bspline_swept_ek_tiered(
        *geo, K_LAD, 2, *lad, g, g, A_LAD
    )
    eligible = g[:, None] == g[None, :]
    ratio = BK._pair_ratio(sl, sr, sl, sr)
    for kk, k in enumerate(K_LAD):
        red = acc.seg_seg_full_moments_bspline_tiered(*geo, k, 2, *lad)
        moved = moved_pairs(ek[kk], red, (0, 1), eligible)
        assert np.array_equal(moved, eligible)
    for lo, hi in ((1.5, 3.0), (3.0, np.inf)):
        tier = (ratio >= lo) & (ratio < hi)
        assert (tier & eligible).any() and (tier & ~eligible).any()


def test_a_sweep_across_the_phase_guard_is_the_per_k_fill():
    """A fine line and a coarse one: at the low k every segment passes the
    kL guard, at the high k the coarse segments fail it, and in between the
    block straddles it. Each k's moments are the single-k fill's at that k --
    which also exercises the single-k split, whose EK labels must be cut
    with the segments (it raised on main before #1362's follow-up)."""
    fine = np.linspace(0.0, 20.0, 42)
    sl1 = np.stack([fine[:-1] - 10.0, np.zeros(41), np.full(41, 10.0)], axis=1)
    sr1 = np.stack([fine[1:] - 10.0, np.zeros(41), np.full(41, 10.0)], axis=1)
    coarse = np.linspace(0.0, 24.0, 6)
    sl2 = np.stack([coarse[:-1] - 12.0, np.full(5, 60.0), np.full(5, 10.0)], axis=1)
    sr2 = np.stack([coarse[1:] - 12.0, np.full(5, 60.0), np.full(5, 10.0)], axis=1)
    sl, sr = np.vstack([sl1, sl2]), np.vstack([sr1, sr2])
    g = np.concatenate([np.zeros(41, np.int64), np.ones(5, np.int64)])
    ek = BK._EK(a=None, group_i=g, group_j=g)
    ladder = ((16.0, 4),)
    # coarse L = 4.8 m: kL = 0.5 at k ~ 0.104; fine L ~ 0.49 m passes to k ~ 1.
    ks = np.array([0.05, 0.1, 0.15, 0.6])
    masks = {BK._phase_row_mask(k, sl, sr).tobytes() for k in ks}
    assert len(masks) == 2  # the guard's answer changes inside the sweep
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = BK._seg_seg_full_moments_offedge_swept(
            sl, sr, sl, sr, 0.001, ks, 2, 8, ek=ek, ladder=ladder
        )
        for kk, k in enumerate(ks):
            one = BK._seg_seg_full_moments_offedge(
                sl, sr, sl, sr, 0.001, float(k), 2, 8, ek=ek, ladder=ladder
            )
            assert_lanes_match(got[kk], one, kk)


# --- solver level ------------------------------------------------------------

C0 = 299792458.0


def _free_array():
    wires = [np.array([(-5.0, 4.0 * j, 10.0), (5.0, 4.0 * j, 10.0)]) for j in range(3)]
    wires.append(np.array([(-5.0, -4.0, 10.0), (0.0, -4.0, 12.0), (5.0, -4.0, 10.0)]))
    return (
        dict(
            wires=wires,
            n_per_edge_per_wire=[[41], [41], [41], [20, 21]],
            feeds=[(0, 5.0, 1 + 0j)],
            wavelength=C0 / 14e6,
            wire_radius=0.001,
        ),
        14e6,
    )


def _junction(pec):
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
        wavelength=C0 / 7e6,
        wire_radius=0.0015,
    )
    if pec:
        d.update(ground_z=0.0)
    return d, 7e6


DECKS = {
    "free_array": _free_array,
    "junction_free": lambda: _junction(False),
    "junction_pec": lambda: _junction(True),
}


class _Count:
    def __init__(self, monkeypatch, reference=None):
        self.n = dict.fromkeys((SWEPT, SWEPT_TIERED), 0)
        for name in self.n:
            f = getattr(acc, name)
            if reference is not None:
                f = functools.partial(f, reference=reference)

            def counted(*a, _f=f, _name=name, **k):
                self.n[_name] += 1
                return _f(*a, **k)

            monkeypatch.setattr(acc, name, counted)


def _sweep(monkeypatch, deck, f0, reference=None, **kw):
    ks = 2 * np.pi * np.linspace(0.93 * f0, 1.07 * f0, 5) / C0
    with monkeypatch.context() as mp:
        c = _Count(mp, reference=reference)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = B.BSplineSolver(**deck, **kw)
            assert s._swept_batched_available()
            z = np.asarray(s.compute_impedance_swept(ks))
    return z.reshape(len(ks), -1)[:, 0], c.n


@pytest.mark.parametrize("name", list(DECKS))
def test_an_ek_sweep_on_the_ladder_is_the_flat_sweep_within_the_derived_tolerance(
    monkeypatch, name
):
    deck, f0 = DECKS[name]()
    z_lad, n_lad = _sweep(monkeypatch, deck, f0, extended_kernel=True)
    z_flat, n_flat = _sweep(
        monkeypatch, deck, f0, extended_kernel=True, pair_order_ladder=()
    )
    rel = np.max(np.abs(z_lad - z_flat) / np.abs(z_flat))
    assert rel < Z_TOL, f"{name}: EK swept ladder vs flat {rel:.3e}"
    # Negative controls: the tiered entry served the ladder and moved bits ...
    assert n_lad[SWEPT_TIERED] >= 1 and n_flat[SWEPT_TIERED] == 0, (n_lad, n_flat)
    assert not np.array_equal(z_lad, z_flat)
    # ... and EK is on: the EK-off sweep sits far away, and it never reaches
    # either EK entry nor moves with the ladder.
    z_off, n_off = _sweep(monkeypatch, deck, f0, extended_kernel=False)
    assert np.min(np.abs(z_lad - z_off) / np.abs(z_off)) > 1e3 * Z_TOL
    assert n_off[SWEPT] == n_off[SWEPT_TIERED] == 0, n_off
    z_off_flat, _ = _sweep(
        monkeypatch, deck, f0, extended_kernel=False, pair_order_ladder=()
    )
    assert_lanes_match(z_off, z_off_flat)


@pytest.mark.parametrize("ladder", [None, ()], ids=["ladder", "flat"])
def test_an_ek_sweep_is_the_reference_walk_to_the_bit(monkeypatch, ladder):
    deck, f0 = _free_array()
    kw = dict(extended_kernel=True)
    if ladder is not None:
        kw["pair_order_ladder"] = ladder
    z_ref, n_ref = _sweep(monkeypatch, deck, f0, reference=True, **kw)
    z, n = _sweep(monkeypatch, deck, f0, reference=False, **kw)
    assert n_ref == n and sum(n.values()) > 0
    assert_lanes_match(z, z_ref)
