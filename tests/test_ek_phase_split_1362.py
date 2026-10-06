"""An extended-kernel block that straddles the ladder's phase guard -- momwire#1362.

The pair-order ladder's phase-limited tiers serve a pair only when both its
segments have |k| L <= 0.5 (momwire#920), and a block whose segments answer
that differently is split by index and recursed. Until the EK fill took the
ladder (#1365) no EK block reached that split; since then one does on any
deck mixing segments either side of L = lambda / (4 pi) -- a fine element
beside a coarse one -- and the recursion handed the WHOLE block's EK labels
to the sub-block, which the C++ entry refused ("group_i/group_j must match
N_i/N_j"). The labels are per segment and are now cut with the segments.

Measured on main before the fix: the deck below raised under EK and solved
with EK off.
"""

from __future__ import annotations

import warnings

import numpy as np

from momwire import _bspline_kernels as BK
from momwire import bspline as B

C0 = 299792458.0
Z_TOL = 1e-11  # test_ek_pair_order_ladder_1362's derived bar


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _deck():
    """A 41-segment dipole (L ~ 0.49 m) and a 5-segment parasitic (L = 4.8 m)
    at 7 MHz, where lambda / (4 pi) ~ 3.4 m: the dipole's segments pass the
    guard, the parasitic's fail it."""
    wires = [
        np.array([(-10.0, 0.0, 10.0), (10.0, 0.0, 10.0)]),
        np.array([(-12.0, 60.0, 10.0), (12.0, 60.0, 10.0)]),
    ]
    return dict(
        wires=wires,
        n_per_edge_per_wire=[[41], [5]],
        feeds=[(0, 10.0, 1 + 0j)],
        wavelength=C0 / 7e6,
        wire_radius=0.001,
    )


def _segments():
    s = B.BSplineSolver(**_deck(), extended_kernel=True)
    geom = s._build_geometry()
    return s, geom["seg_l"], geom["seg_r"], s._ek_spec(geom)


def test_the_deck_straddles_the_guard():
    s, sl, sr, ek = _segments()
    ladder = s._fill_ladder(s.k, sl, sr, ek)
    assert ladder, "the free-space default ladder is the tier under test"
    assert BK._phase_split_needed(ladder, s.k, sl, sr, sl, sr) is not None


def test_a_straddling_ek_block_is_its_sub_blocks():
    """The split fill equals each quadrant filled on its own with the labels
    cut alongside -- what the recursion is meant to compute."""
    s, sl, sr, ek = _segments()
    ladder = s._fill_ladder(s.k, sl, sr, ek)
    rows_ok, cols_ok = BK._phase_split_needed(ladder, s.k, sl, sr, sl, sr)
    a = float(s.wire_radius)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = BK._seg_seg_full_moments_offedge(
            sl, sr, sl, sr, a, s.k, s.degree, s.n_qp_pair, ek=ek, ladder=ladder
        )
        for ri in (np.flatnonzero(rows_ok), np.flatnonzero(~rows_ok)):
            for cj in (np.flatnonzero(cols_ok), np.flatnonzero(~cols_ok)):
                sub = BK._seg_seg_full_moments_offedge(
                    sl[ri],
                    sr[ri],
                    sl[cj],
                    sr[cj],
                    a,
                    s.k,
                    s.degree,
                    s.n_qp_pair,
                    ek=BK._EK(a=ek.a, group_i=ek.group_i[ri], group_j=ek.group_j[cj]),
                    ladder=ladder,
                )
                assert np.array_equal(
                    _bits(got[:, :, ri[:, None], cj[None, :]]), _bits(sub)
                )


def test_the_deck_solves_under_ek_within_the_ladder_tolerance():
    cls = B.BSplineSolver
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z, _ = cls(**_deck(), extended_kernel=True).compute_impedance()
        z_flat, _ = cls(
            **_deck(), extended_kernel=True, pair_order_ladder=()
        ).compute_impedance()
        z_off, _ = cls(**_deck(), extended_kernel=False).compute_impedance()
    assert np.isfinite(z)
    assert abs(z - z_flat) / abs(z_flat) < Z_TOL
    assert abs(z - z_off) / abs(z_off) > 1e3 * Z_TOL
