"""The tie detector, and what it found (stevenmburns/momwire#623 step 1).

`_feed_snap.snap` reports; it never moves a pick. These gates pin the three
things step 2 will reason from: that it fires on a tie, that it stays quiet
on the two things that look like one and are not, and that the bar it uses
still sits in the void the suite's own snaps leave.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from momwire import _feed_snap
from momwire.pulse import PulseSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

WIRE = np.array([[-5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
WL = 20.0


@pytest.mark.parametrize(
    "cls,n,arc,expect",
    [
        # A centre-snapper ties on a KNOT, which an even segment count puts at
        # the wire midpoint — so the DEFAULT feed is the tie.
        (SinusoidalGalerkinSolver, 20, None, True),
        (SinusoidalGalerkinSolver, 21, None, False),
        (SinusoidalGalerkinSolver, 20, 5.0, True),
        (PulseSolver, 20, None, True),
        (PulseSolver, 21, None, False),
        # A knot-snapper is the mirror image: it ties on a segment CENTRE,
        # which is where an ODD count puts the midpoint.
        (RazorSolver, 21, None, True),
        (RazorSolver, 20, None, False),
        # And the thing that must NOT fire: a request that is simply
        # off-grid. Snapping to the nearest site is the feature.
        (SinusoidalGalerkinSolver, 20, 3.7, False),
        (RazorSolver, 20, 3.7, False),
    ],
)
def test_the_detector_fires_on_a_tie_and_not_on_an_off_grid_request(
    monkeypatch, cls, n, arc, expect
):
    monkeypatch.setenv(_feed_snap._TAP, "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls(
            wires=[WIRE],
            n_per_edge_per_wire=[[n]],
            wavelength=WL,
            feed_arclength=arc,
        ).compute_impedance()
    hits = [w for w in caught if w.category is _feed_snap.AmbiguousSite]
    assert bool(hits) is expect, [str(w.message) for w in hits]


def test_the_detector_is_silent_without_its_tap(monkeypatch):
    """It is a diagnostic, not a contract. 530 of the quick suite's 3341 snaps
    are ties — almost all on symmetric decks where the two candidates are
    mirror images — so a default-on warning would cry on harmless work and
    make step 2's decision quietly. Flipping it on is step 2's call."""
    monkeypatch.delenv(_feed_snap._TAP, raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SinusoidalGalerkinSolver(
            wires=[WIRE], n_per_edge_per_wire=[[20]], wavelength=WL
        ).compute_impedance()
    assert not [w for w in caught if w.category is _feed_snap.AmbiguousSite]


def test_the_pick_is_the_argmin_whatever_the_tap_says(monkeypatch):
    """Away from a tie the pick is `argmin`, and the tap never moves it —
    turning the diagnostic on must not change an answer.

    AT a tie it is the smaller arclength instead (momwire#672 made that
    sayable by putting razor's knots in arc order). On an ascending grid the
    two agree, because `argmin` already returns the first occurrence — which
    is why every target below still matches `argmin`, ties included. The
    rule earns its keep on the NEAR-ties, where the distances differ by
    rounding and `argmin` follows the noise instead of the geometry;
    `test_a_near_tie_goes_to_the_smaller_arclength` covers that."""
    grid = np.array([0.25, 0.75, 1.25, 1.75])
    for tap in ("1", None):
        if tap:
            monkeypatch.setenv(_feed_snap._TAP, tap)
        else:
            monkeypatch.delenv(_feed_snap._TAP, raising=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", _feed_snap.AmbiguousSite)
            for target in (0.0, 0.5, 0.9, 1.5, 2.0):
                pick, _margin = _feed_snap.snap(grid, target, total_arc=2.0, family="T")
                assert pick == int(np.argmin(np.abs(grid - target)))


def test_the_tap_writes_one_line_per_snap(monkeypatch, tmp_path):
    """The instrument step 2 needs: every snap, not only the ties, because a
    void is only visible when both of its sides are in the data."""
    monkeypatch.setenv(_feed_snap._TAP, str(tmp_path / "tally"))
    monkeypatch.setattr(_feed_snap, "_TALLY", [])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _feed_snap.AmbiguousSite)
        _feed_snap.snap([0.25, 0.75], 0.5, total_arc=1.0, family="T", wire=3)
        _feed_snap.snap([0.25, 0.75], 0.3, total_arc=1.0, family="T", wire=3)
    _feed_snap._flush_tally()
    rows = [
        json.loads(line)
        for path in Path(tmp_path).glob("tally.*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert len(rows) == 2, rows
    tie, clean = sorted(rows, key=lambda r: r["frac"])
    assert tie["frac"] <= _feed_snap.AMBIGUITY_TOL_FRAC
    assert clean["frac"] > _feed_snap.AMBIGUITY_TOL_FRAC
    assert {r["family"] for r in rows} == {"T"}
    assert {r["wire"] for r in rows} == {3}


def test_the_bar_still_sits_in_the_void_the_suite_leaves():
    """The derivation, kept as an assertion rather than only as prose.

    Measured over the 3341 snaps the quick suite makes: the largest margin
    that is rounding is 1.301e-14 of the arc, and the smallest that is a real
    separation is 1.248e-03 — eleven empty decades. The bar has to stay
    inside that, with room either side, and a retune that walks it out should
    fail here rather than silently start calling requests ambiguous or stop
    catching ties.
    """
    assert 1.301e-14 * 100 < _feed_snap.AMBIGUITY_TOL_FRAC, (
        "the bar has dropped onto the rounding it must clear"
    )
    assert _feed_snap.AMBIGUITY_TOL_FRAC * 100 < 1.248e-3, (
        "the bar has risen into the separations it must spare"
    )
    # And under the deliberate straddle in tests/test_feed_snap_623.py, which
    # asks for both sides of a tie on purpose and is not confused.
    assert _feed_snap.AMBIGUITY_TOL_FRAC < 2.0e-10


def test_a_near_tie_goes_to_the_smaller_arclength(monkeypatch):
    """The case the rule exists for, and the one `argmin` gets wrong.

    Two sites the same distance away to within rounding, arranged so the
    FARTHER-by-a-crumb one is the smaller arclength. `argmin` follows the
    crumb and takes the larger; the rule takes the smaller, and keeps taking
    it whichever way the crumb falls — which is what lets two spellings of
    one antenna resolve the same tie the same way (momwire#672).
    """
    monkeypatch.delenv(_feed_snap._TAP, raising=False)
    crumb = 1e-15
    grid = np.array([1.0, 2.0])
    target = 1.5 + crumb  # nearer to 2.0 by a crumb; 1.0 is the smaller arc
    assert int(np.argmin(np.abs(grid - target))) == 1
    pick, margin = _feed_snap.snap(grid, target, total_arc=2.0, family="T")
    assert margin <= _feed_snap.AMBIGUITY_TOL_FRAC * 2.0
    assert pick == 0, "the rule must not follow the crumb"
