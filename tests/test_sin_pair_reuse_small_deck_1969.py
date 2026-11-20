"""`pair_reuse`'s small-deck threshold on the sinusoidal fill (momwire#1969).

Below `SMALL_DECK_SEGS_FREE` segments (free space, PEC) or
`SMALL_DECK_SEGS_GROUND` (a finite ground) the census costs more than the
fill saves, so the route steps aside, says so in `pair_reuse_declined`, and
the fill is the shipped one bit for bit. An explicit `True` follows it;
`"verify"` ignores it. An earlier reason keeps its own words.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import _sin_pair_reuse as pr
from momwire.sinusoidal import SinusoidalSolver

# An experimental feature (`momwire._features`, reach 4: through the drop-in
# on Linux and Windows).
pytestmark = pytest.mark.reach(4)

LAM = 10.7
GROUNDS = {
    "free": {},
    "pec": dict(ground_z=0.0),
    "refl-coef": dict(ground_z=0.0, ground_eps=(13.0, 0.005)),
    "sommerfeld": dict(
        ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld"
    ),
}
FINITE = {"refl-coef", "sommerfeld"}


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("MOMWIRE_X_PAIR_REUSE", raising=False)
    pr.clear_shared_plans()
    yield
    pr.clear_shared_plans()


def _dipole(segs):
    """A horizontal half-wave dipole at 8 m, centre fed (odd `segs`) or fed
    at the segment just short of centre."""
    return dict(
        wires=[np.array([[0.0, -2.6, 8.0], [0.0, 2.6, 8.0]])],
        n_per_edge_per_wire=[[segs]],
        feed_wire_index=0,
        wavelength=LAM,
        wire_radius=1e-3,
    )


def _fill(deck, pair_reuse=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = SinusoidalSolver(**deck, **kw, pair_reuse=pair_reuse)
        geom = s._build_geometry()
        with s._operating_medium(geom) as medium:
            G, _ = s._assemble_Z(geom, s.k, s._medium_eta(medium))
    return G, s


def _threshold(ground):
    return pr.SMALL_DECK_SEGS_GROUND if ground in FINITE else pr.SMALL_DECK_SEGS_FREE


def test_the_thresholds():
    assert (pr.SMALL_DECK_SEGS_FREE, pr.SMALL_DECK_SEGS_GROUND) == (400, 200)


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_a_small_deck_declines_and_fills_as_off(ground):
    deck = _dipole(41)
    base, _ = _fill(deck, **GROUNDS[ground])
    on, s = _fill(deck, pair_reuse=True, **GROUNDS[ground])
    assert s.pair_reuse_declined == (
        f"too few segments to repay the plan (N < {_threshold(ground)})"
    )
    assert s.pair_reuse_census == {}
    assert on.tobytes() == base.tobytes()


@pytest.mark.parametrize("ground", sorted(GROUNDS))
def test_verify_runs_the_reuse_on_a_small_deck(ground):
    deck = _dipole(41)
    base, _ = _fill(deck, **GROUNDS[ground])
    on, s = _fill(deck, pair_reuse="verify", **GROUNDS[ground])
    assert s.pair_reuse_declined is None
    assert s.pair_reuse_census and "direct" in s.pair_reuse_census
    assert 0.0 < s.pair_reuse_verify_max <= pr.VERIFY_TOL
    assert float(np.max(np.abs(on - base)) / np.max(np.abs(base))) <= 3e-12


@pytest.mark.parametrize("ground", ["free", "sommerfeld"])
def test_the_threshold_is_the_first_segment_count_served(ground):
    """One segment short of the threshold declines; at it the reuse runs."""
    n = _threshold(ground)
    _, s = _fill(_dipole(n - 1), pair_reuse=True, **GROUNDS[ground])
    assert s.pair_reuse_declined.startswith("too few segments")
    _, s = _fill(_dipole(n), pair_reuse=True, **GROUNDS[ground])
    assert s.pair_reuse_declined is None and s.pair_reuse_census


def test_an_earlier_reason_keeps_its_words():
    """A deck the reuse could not serve anyway says why, not that it is
    small: the 41-segment dipole at a wavelength of 1 km, k*delta 8e-4,
    takes the folded cos-1 shapes."""
    deck = dict(_dipole(41), wavelength=1000.0)
    _, s = _fill(deck, pair_reuse=True)
    assert s.pair_reuse_declined.startswith("the folded cos-1 shapes")
