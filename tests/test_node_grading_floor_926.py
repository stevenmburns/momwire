"""momwire#926: the crossing advisory and the stand-off floor now agree.

Two rules used to give opposite instructions for the same wire.
`CoarseCrossingNode` (#674, #696) asked for "~6 mm at the node" unconditionally;
the `h/a >= 2` stand-off floor (#865, #875) refuses a conductor closer than two
radii to the plane. Grading toward a node ON the interface puts vertices
arbitrarily close to it, so following the first instruction walked a sloping
deck into the second, with nothing in either message saying they were related.

Neither rule moves here. `_crossing_fill.node_panel_floor` is the one place the
connection is written down --

    L_min = h_floor / slope

-- and both messages now read it: the advisory asks for panels from L_min
instead of ~6 mm, and the refusal says when the offending vertex came from
grading. Physics untouched; this file gates TEXT and the arithmetic behind it.

WHICH DECK. The issue's own deck -- four radials sloping up from a node with a
vertical mast -- raises the refusal but NOT the advisory: every wire is ABOVE,
so the junction is grounded but not CROSSING, and `_crossing_junctions()`
returns () on it (asserted below, because it is the thing that made the first
version of this gate measure nothing). The geometry where both rules apply
needs a crossing junction whose single ABOVE member slopes: a sloping wire off
a grounded node over a buried screen. Both decks are here.
"""

import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from momwire import _crossing_fill  # noqa: E402
from momwire.bspline import BSplineSolver  # noqa: E402

C0 = 299792458.0
WL, A_WIRE, SOIL = C0 / 7e6, 1e-3, (13.0, 0.005)
DEPTH, REACH, RISE = 0.15, 10.0, 0.5

# The issue's arithmetic, restated: a bare 1 mm conductor must clear
# h = 2a = 2 mm, and on a 5 % slope it reaches that at 40 mm from the node.
#
# "5 % slope" colloquially means rise/RUN, and #926 quotes 40 mm from
# h = 0.05 * s. momwire measures rise per ARCLENGTH, because that is the
# quantity that makes h = slope * l true along the wire, and the two differ by
# 1/sqrt(1 + s^2): 0.049969 against 0.05 here, so the exact L_min is 40.05 mm.
# Invisible at the printed precision (both render "40.0 mm") and worth
# recording rather than rounding away -- a gate written to the colloquial
# figure fails by 0.12 %, which is what happened when this one was.
SLOPE = RISE / float(np.hypot(REACH, RISE))
SLOPE_PCT = 100.0 * SLOPE
H_FLOOR_M = 2.0 * A_WIRE
L_MIN_M = H_FLOOR_M / SLOPE
L_MIN_MM = 40.0  # what the message PRINTS, at one decimal


def _graded(finest_mm, span, ratio=1.35):
    s, pts, r = finest_mm / 1000.0, [0.0], 1.0
    while pts[-1] < span:
        pts.append(pts[-1] + s * r)
        r *= ratio
    return np.array([p for p in pts if p < span] + [span])


def crossing_deck(above="slope", finest_mm=None, n_radials=4):
    """N buried radials rising to a node at z = 0, plus ONE above member.

    `above="slope"` makes that member a 5 % sloper -- the geometry in which
    the floor binds the grading. `finest_mm=None` leaves it ungraded, which is
    the state the advisory is for.
    """
    dirs = [
        (np.cos(2 * np.pi * i / n_radials), np.sin(2 * np.pi * i / n_radials))
        for i in range(n_radials)
    ]
    wires = [
        np.array([(5.0 * dx, 5.0 * dy, -DEPTH), (0.0, 0.0, -DEPTH), (0.0, 0.0, 0.0)])
        for dx, dy in dirs
    ]
    npe = [[10, 2] for _ in dirs]
    mono_i = len(wires)
    tip = (
        np.array([0.0, 0.0, 10.0]) if above == "plumb" else np.array([REACH, 0.0, RISE])
    )
    if finest_mm is None:
        wires.append(np.array([(0.0, 0.0, 0.0), tip]))
        npe.append([15])
    else:
        span = float(np.linalg.norm(tip))
        wires.append(np.outer(_graded(finest_mm, span) / span, tip))
        npe.append([1] * (len(wires[-1]) - 1))
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        junctions=[[(i, "end") for i in range(n_radials)] + [(mono_i, "start")]],
        feeds=[(mono_i, 0.25, 1 + 0j)],
        wavelength=WL,
        wire_radius=A_WIRE,
        ground_z=0.0,
        ground_eps=SOIL,
        ground_model="sommerfeld",
    )


def issue_deck(finest_mm, rise=RISE):
    """The deck as filed on #926: four ABOVE radials sloping up from a node,
    plus a vertical mast. No below member, so no crossing junction."""
    pts = _graded(finest_mm, REACH)
    wires, npe = [], []
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        wires.append(np.stack([pts * dx, pts * dy, rise * (pts / REACH)], axis=1))
        npe.append([1] * (len(pts) - 1))
    wires.append(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]]))
    npe.append([20])
    return dict(
        wires=wires,
        n_per_edge_per_wire=npe,
        wire_radius=A_WIRE,
        wavelength=WL,
        degree=2,
        feed_model="segment",
        feed_wire_index=4,
        feed_arclength=0.25,
        ground_z=0.0,
        ground_eps=SOIL,
        ground_model="sommerfeld",
    )


def _advisory(build):
    s = BSplineSolver(**build)
    arms = s._crossing_node_members(s._crossing_junctions(), s._wire_media())
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _crossing_fill.warn_coarse_node(arms)
    msgs = [
        str(w.message) for w in rec if w.category is _crossing_fill.CoarseCrossingNode
    ]
    return (msgs[0] if msgs else None), arms


# ---------------------------------------------------------------------------
# the formula
# ---------------------------------------------------------------------------


def test_g926_1_the_floor_formula_is_the_issues_arithmetic():
    """L_min = h_floor / slope, and both None cases are real geometries."""
    assert _crossing_fill.node_panel_floor(0.002, 0.05) == pytest.approx(0.040)
    # a LEVEL arm never approaches the interface by nearing the node
    assert _crossing_fill.node_panel_floor(0.002, 0.0) is None
    # a BELOW-side arm has no floor to clear
    assert _crossing_fill.node_panel_floor(None, 0.5) is None
    # and it grows without limit as the slope flattens -- #926's closing point
    assert _crossing_fill.node_panel_floor(0.002, 0.01) == pytest.approx(0.200)


# ---------------------------------------------------------------------------
# (a) the advisory states a floor-aware minimum
# ---------------------------------------------------------------------------


def test_g926_2_the_advisory_names_the_floor_on_a_sloping_arm():
    msg, arms = _advisory(crossing_deck("slope"))
    above = [a for a in arms if a.side == "above"]
    assert len(above) == 1, "the fan widening allows exactly one above member"
    assert above[0].slope == pytest.approx(SLOPE, rel=1e-12)
    assert above[0].h_floor == pytest.approx(H_FLOOR_M, rel=1e-12)
    assert _crossing_fill.node_panel_floor(
        above[0].h_floor, above[0].slope
    ) == pytest.approx(L_MIN_M, rel=1e-12)
    # and that IS the issue's 40 mm, to the precision it quotes
    assert L_MIN_M * 1000.0 == pytest.approx(40.0, abs=0.06)
    assert msg is not None
    assert f"L_min = {L_MIN_MM:.1f} mm" in msg, msg
    assert "stand-off floor forbids anything shorter" in msg
    assert "momwire#865/#926" in msg
    # and it says the node cannot be resolved to the bar on this slope
    assert "cannot be graded to the bar" in msg


def test_g926_3_the_floor_is_read_across_arms_not_off_the_worst_one():
    """The binding floor belongs to whichever arm slopes, which is usually not
    the worst-meshed arm. On this deck the worst arm by mesh is a BELOW radial
    at 75 mm, whose floor is None -- reading L_min off it printed the old text
    on the one deck the issue is about."""
    msg, arms = _advisory(crossing_deck("slope", finest_mm=6.0))
    worst = max(arms, key=lambda a: a.h_resolved)
    assert worst.side == "below" and worst.h_floor is None
    assert f"L_min = {L_MIN_MM:.1f} mm" in msg, msg


@pytest.mark.parametrize(
    "build,why",
    [
        (crossing_deck("plumb"), "a plumb arm's L_min is 2 mm, under the ~6 mm"),
        (crossing_deck("plumb", finest_mm=6.0), "the same, graded"),
    ],
)
def test_g926_4_a_steep_arm_keeps_the_old_text(build, why):
    """Where the floor cannot bind, the message must be what it always was --
    including the level BELOW radials, whose arms carry no floor at all."""
    msg, arms = _advisory(build)
    assert msg is not None, why
    assert "~6 mm at the node growing to the design's own segment length" in msg
    assert "L_min" not in msg, why
    assert "momwire#926" not in msg, why
    assert all(a.h_floor is None for a in arms if a.side == "below")


# ---------------------------------------------------------------------------
# (b) the refusal names the advisory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build,why",
    [
        (crossing_deck("slope", finest_mm=6.0), "the crossing deck"),
        (issue_deck(6.0), "the deck as filed on #926"),
    ],
)
def test_g926_5_the_refusal_names_the_advisory_for_a_graded_vertex(build, why):
    with pytest.raises(ValueError) as exc:
        BSplineSolver(**build).compute_impedance()
    msg = str(exc.value)
    # the refusal it always was
    assert "validity floor for a BARE conductor" in msg
    assert "h = 0.300 mm" in msg
    # plus the connection
    assert "came from node grading" in msg, why
    assert "CoarseCrossingNode" in msg
    assert f"L_min = {L_MIN_MM:.1f} mm" in msg, msg
    assert "momwire#926" in msg
    # and the sentence does not run into the trailing citation
    assert "momwire#926. See" in msg


def test_g926_6_the_deck_as_filed_has_no_crossing_junction():
    """Why this file carries two decks, asserted rather than explained.

    #926's deck is all ABOVE wires, so its junction is grounded but not
    crossing and the advisory never walks it. The refusal half of the fix is
    therefore the half that reaches that deck -- and a gate built only on it
    would measure the advisory not at all.
    """
    s = BSplineSolver(**issue_deck(6.0))
    assert s._crossing_junctions() == ()
    assert all(m == "above" for m in s._wire_media())


def test_g926_7_an_ungraded_refusal_does_not_blame_grading():
    """A wire the USER laid along the interface must not be told it came from
    node grading. Two vertices low, no interior vertex, no junctioned end at
    the plane -- a plain low horizontal wire.
    """
    build = issue_deck(6.0)
    build["wires"] = [np.array([[1.0, 0.0, 0.0005], [6.0, 0.0, 0.0005]])]
    build["n_per_edge_per_wire"] = [[10]]
    build["feed_wire_index"] = 0
    build["feed_arclength"] = 2.5
    with pytest.raises(ValueError) as exc:
        BSplineSolver(**build).compute_impedance()
    msg = str(exc.value)
    assert "validity floor for a BARE conductor" in msg
    assert "came from node grading" not in msg, msg
    assert "momwire#926" not in msg, msg
