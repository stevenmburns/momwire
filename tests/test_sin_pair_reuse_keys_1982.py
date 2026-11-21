"""razor2p's two key faults, measured on the sinusoidal fill (momwire#1982).

Razor's pair reuse (momwire#1971) found two faults in a key the sinusoidal
one shares the shape of. Each was measured here (synthetic decks):

- The near rule in source lengths alone. Where a segment is shorter than
  its radius the kernel varies over a, not h, and pairs a few h apart but
  inside a radius were shared. It REPRODUCES here: on a hairpin whose
  bridge segments are 0.095 a long, Zin moved 1.4e-3 against the flag-off
  fill. The rule is now in max(2H, a), razor's (`near_scale`, in the C++ census), and the
  same deck moves 1.6e-9.
- The source length quantised against the pair distance. It does NOT
  reproduce: this key quantises the half-length against itself at KEY_BITS
  relative bits (`_quantise_length`), so lengths that differ by more than a
  few ulp never share a class. Held below by a census, as razor's is.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import _sin_pair_reuse as pr
from momwire._accel import acc
from momwire.sinusoidal import SinusoidalSolver

# An experimental feature (`momwire._features`, reach 4: through the drop-in
# on Linux and Windows).
pytestmark = pytest.mark.reach(4)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_PAIR_REUSE", raising=False)
    # Cheap decks, under the small-deck thresholds on purpose (the reuse
    # itself is what is tested; test_sin_pair_reuse_small_deck_1969 holds
    # the thresholds).
    monkeypatch.setattr(pr, "SMALL_DECK_SEGS_FREE", 0)
    monkeypatch.setattr(pr, "SMALL_DECK_SEGS_GROUND", 0)
    pr.clear_shared_plans()
    yield
    pr.clear_shared_plans()


def _hairpin(a=0.06, nb=21, nw=21, wing=10.0):
    """Two `wing` m wires 2a apart joined by a 2a bridge of `nb` segments,
    so the bridge's h = 2a/nb is under the radius (h/a 0.095 at 21). At
    lambda 20 m and a = 6 cm, k h = 1.8e-3: inside the served range (below
    `_WELL_SCALED_KD` the fill takes the folded shapes and the reuse
    declines, which is why razor's 1 mm hairpin is not this deck)."""
    g = 2 * a
    ws = [
        [[0.0, -wing / 2, 8.0], [0.0, wing / 2, 8.0]],
        [[0.0, wing / 2, 8.0], [g, wing / 2, 8.0]],
        [[g, wing / 2, 8.0], [g, -wing / 2, 8.0]],
    ]
    return dict(
        wires=[np.array(w, dtype=float) for w in ws],
        n_per_edge_per_wire=[[nw], [nb], [nw]],
        feeds=[(0, wing * 0.5238095238095238, 1 + 0j)],
        wire_radius=a,
        wavelength=20.0,
    )


def _zin(deck, mode, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SinusoidalSolver(**deck, **kw, pair_reuse=mode)
        z = complex(1.0 / s.compute_port_solution().y[0, 0])
    return z, s


def test_segments_shorter_than_their_radius_keep_their_near_pairs(monkeypatch):
    deck = _hairpin()
    off, _ = _zin(deck, False)
    on, s = _zin(deck, True)
    assert s.pair_reuse_declined is None
    n_pairs, n_classes, n_near = s.pair_reuse_census["direct"]
    assert n_classes < 0.3 * n_pairs  # it ran, and shared
    good = abs(on - off) / abs(off)
    assert good < 1e-7, good  # measured 1.6e-9

    # The rule in source lengths alone: fewer near pairs, and the answer
    # moves by orders more (measured 1.4e-3).
    acc.sin_pair_reuse_configure(probe=1)  # the C++ census's PROBE_NEAR_H
    pr.clear_shared_plans()
    try:
        wrong, s = _zin(deck, True)
    finally:
        acc.sin_pair_reuse_configure(probe=0, clear=True)
    assert s.pair_reuse_census["direct"][2] < n_near
    bad = abs(wrong - off) / abs(off)
    assert bad > 1e-5, bad


def test_the_near_rule_is_unchanged_where_segments_outrun_their_radius():
    # 2H >= a everywhere: max(2H, a) is 2H, the rule as it was.
    # A pair is near where its observer lies within 4 kernel scales of the
    # source's centre: each observer here sits on the source's axis just
    # inside and just outside 4 max(2H, a), as the C++ census keys it.
    def near(H, a, d):
        zero = np.zeros(1, np.int64)
        _, _, nr = acc.sin_pair_reuse_keys(
            np.array([[0.0, 0.0, d]]), np.array([[1.0, 0.0, 0.0]]),
            np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]),
            np.array([2 * H]), np.array([a]), zero, zero,
        )  # fmt: skip
        return bool(nr[0])

    for H, a in [(0.05, 0.001), (0.1, 0.004), (0.002, 0.001), (0.01, 0.05)]:
        scale = max(2 * H, a)
        assert near(H, a, 0.99 * 4 * scale) and not near(H, a, 1.01 * 4 * scale)
    # where 2H >= a the scale is 2H, the rule as it was
    assert not near(0.05, 0.001, 4 * 0.1 * 1.01) and near(0.05, 0.001, 0.39)


def test_segments_of_different_lengths_never_share_a_class():
    """Two parallel dipoles 3 m apart, the second 1e-12 m longer (1e-13
    relative, far over the half-length's 2^-48 step): every pair of the
    second's sources differs from the first's only in that length, and a
    key quantising it against the pair distance would merge them (razor's
    fault). Here they keep classes of their own: at least twice the twin
    deck's shared classes."""

    def census(dy):
        deck = dict(
            wires=[
                np.array([[0.0, -5.0, 8.0], [0.0, 5.0, 8.0]]),
                np.array([[3.0, -5.0, 8.0], [3.0, 5.0 + dy, 8.0]]),
            ],
            n_per_edge_per_wire=[[41], [41]],
            feed_wire_index=0,
            wavelength=20.0,
            wire_radius=1e-3,
        )
        on, s = _zin(deck, True)
        off, _ = _zin(deck, False)
        # (pairs, shared classes, near pairs: evaluated directly, no class)
        _n_pairs, n_classes, _n_near = s.pair_reuse_census["direct"]
        return n_classes, abs(on - off) / abs(off)

    (twin, d_twin), (apart, d_apart) = census(0.0), census(1e-12)
    assert twin > 0
    assert apart >= 2 * twin, (twin, apart)
    assert max(d_twin, d_apart) < 1e-11, (d_twin, d_apart)
