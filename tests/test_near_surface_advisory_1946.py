"""momwire#1946: the advisory for bare wire within ~2 radii of the ground
surface, below it (buried, or a crossing radial's shallow first stretch).

The shapes are the crossing-node study's (momwire#1149, comment of 11-14):
`slant14` is #1687's deck, a 5 m, 2 mm vertical on a z = 0 crossing node over
4 radials of 5.2 m sloping to -5 cm at 14 MHz; `kinked` is its -1 cm control,
each radial dropping to -1 cm within 5 cm of the node.

Geometry only: every solver here is constructed and asked for its medium
labels, which is where the advisory is read. Nothing is filled or solved.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from momwire import _near_surface
from momwire._near_surface import NearSurfaceWire
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver

LAM14 = 299.792458 / 14.0
SOIL = complex(13.0, -6.42)  # eps_r 13, sigma 5 mS/m at 14 MHz
SOLVERS = (RazorSolver, BSplineSolver, SinusoidalSolver)


def _crossing_deck(*, drop=0.05, kink=None, grade=0, n_rad=4, length=5.2):
    """A 5 m vertical on a z = 0 crossing node, `n_rad` buried radials.

    `kink=(x, z)`: each radial first runs to (x, depth z), then on to the tip.
    `grade`: each radial's first `grade` pieces split off as separate wires,
    as a deck grading its node spells them (one GW card per piece).
    """
    wires = [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.25]]),
        np.array([[0.0, 0.0, 0.25], [0.0, 0.0, 5.0]]),
    ]
    for i in range(n_rad):
        ang = 2 * math.pi * i / n_rad
        u = np.array([math.cos(ang), math.sin(ang), 0.0])
        tip = length * u + np.array([0.0, 0.0, -drop])
        if kink is not None:
            kp = kink[0] * u + np.array([0.0, 0.0, -kink[1]])
            wires += [np.array([[0.0, 0.0, 0.0], kp]), np.array([kp, tip])]
            continue
        cuts = [0.0] + [0.02 * 2**k for k in range(grade)] + [1.0]
        cuts = sorted(set(min(c, 1.0) for c in cuts))
        pts = [c * tip for c in cuts]
        wires += [np.array([p0, p1]) for p0, p1 in zip(pts[:-1], pts[1:])]
    return wires


def _advisories(cls, wires, radius=0.002, wavelength=LAM14, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = cls(
            wires=wires,
            nsegs=5,
            wire_radius=radius,
            ground_z=0.0,
            ground_eps=SOIL,
            ground_model="sommerfeld",
            wavelength=wavelength,
            feed_wire_index=0,
            **kw,
        )
        solver._wire_media()
    return [str(w.message) for w in caught if w.category is NearSurfaceWire]


# --- the two firing cases ---------------------------------------------------


@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_fires_on_buried_wire_within_two_radii(cls):
    # A detached level radial 1.5 radii under the surface, fed in its middle.
    wires = [np.array([[0.0, 0.0, -0.003], [5.0, 0.0, -0.003]])]
    got = _advisories(cls, wires)
    assert len(got) == 1
    text = got[0]
    assert text.startswith("buried wire runs within 2 radii of the ground surface")
    assert "from (0, 0, -0.003) to (5, 0, -0.003) m" in text
    assert "stays that close for 5.00 m" in text
    assert "closest approach is 1.5 radii" in text
    assert "reactance X" in text and "0.5 ohm" in text
    assert "Advisory only" in text


@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_fires_on_shallow_radial_start(cls):
    got = _advisories(cls, _crossing_deck())
    assert len(got) == 1
    text = got[0]
    # 2a / slope = 4 mm / (0.05 / 5.2) = 0.416 m on each radial.
    assert "stays that close for 0.42 m" in text
    assert "it touches the surface" in text
    assert "3 other buried wires do the same" in text


def test_graded_node_is_one_stretch_not_several():
    """A radial spelled as several cards (node grading) is still ONE stretch,
    so refining the node cannot switch the advisory off."""
    for grade in (0, 3, 5):
        got = _advisories(RazorSolver, _crossing_deck(grade=grade))
        assert len(got) == 1, grade
        assert "stays that close for 0.42 m" in got[0], grade


# --- the silent cases ---------------------------------------------------------


@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
def test_silent_on_the_minus_1cm_kinked_control(cls):
    # 4 mm / 0.2 slope = 2 cm within the zone, under lambda/500 = 4.3 cm.
    assert _advisories(cls, _crossing_deck(kink=(0.05, 0.01))) == []


@pytest.mark.parametrize("cls", SOLVERS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("radius", [0.002, 0.03])
def test_silent_on_a_vertical_through_the_interface(cls, radius):
    """A ground rod crossing the surface spends 2a in the zone whatever its
    radius. At 30 mm 2a = 60 mm passes lambda/500 on its own; the alongside
    condition (>= 10a) is what keeps it silent."""
    wires = [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -2.0]]),
    ]
    assert _advisories(cls, wires, radius=radius) == []


@pytest.mark.parametrize(
    "cls", (RazorSolver, SinusoidalSolver), ids=lambda c: c.__name__
)
def test_silent_on_above_ground_wire(cls):
    """Above the interface is momwire#865's (the B-spline family refuses an
    edge under h = 2a; `SurfaceRadialHeight` advises up to 20a): no second
    message from this one. razor-2p and the sinusoidal family serve it."""
    wires = [np.array([[0.0, 0.0, 0.003], [5.0, 0.0, 0.003]])]
    assert _advisories(cls, wires) == []


def test_bspline_above_ground_is_865s_refusal_alone():
    wires = [np.array([[0.0, 0.0, 0.003], [5.0, 0.0, 0.003]])]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = BSplineSolver(
            wires=wires,
            nsegs=5,
            wire_radius=0.002,
            ground_z=0.0,
            ground_eps=SOIL,
            ground_model="sommerfeld",
            wavelength=LAM14,
        )
        with pytest.raises(ValueError, match="momwire#865"):
            solver._wire_endpoint_status()
    assert not [w for w in caught if w.category is NearSurfaceWire]


def test_silent_on_an_ordinary_buried_screen():
    # Level radials 5 cm deep under a 5 cm link: nothing near the surface but
    # the link, which is vertical.
    wires = [
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.05]]),
    ]
    for i in range(8):
        ang = 2 * math.pi * i / 8
        wires.append(
            np.array([[0.0, 0.0, -0.05], [5 * math.cos(ang), 5 * math.sin(ang), -0.05]])
        )
    assert _advisories(RazorSolver, wires, radius=0.001) == []


def test_silent_on_a_jacketed_buried_wire():
    wires = [np.array([[0.0, 0.0, -0.003], [5.0, 0.0, -0.003]])]
    got = _advisories(
        RazorSolver, wires, insulation_radius=0.0025, insulation_eps_r=3.0
    )
    assert got == []


def test_silent_over_free_space_and_without_a_buried_wire():
    wires = [np.array([[0.0, 0.0, -0.001], [5.0, 0.0, -0.001]])]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RazorSolver(
            wires=wires, nsegs=5, wire_radius=0.002, wavelength=LAM14
        )._wire_media()
    assert not [w for w in caught if w.category is NearSurfaceWire]


# --- the rule's thresholds, on the pure function -------------------------------


def _fire(wires, media, a, lam):
    st = _near_surface.near_surface_stretches(wires, media, [a] * len(wires), 0.0)
    return st, _near_surface.stretches_that_fire(st, lam)


def test_the_study_decks_split_at_lambda_over_500():
    """The module docstring's table, re-derived: x5 slope fires (0.08 m,
    -0.09 ohm in the study), 0.2 mm radials do not (0.04 m, -0.04 ohm)."""
    below = ("below",)
    steep = [np.array([[0.0, 0.0, 0.0], [5.2, 0.0, -0.25]])]
    st, fired = _fire(steep, below, 0.002, LAM14)
    assert st[0].length_m == pytest.approx(0.004 * math.hypot(5.2, 0.25) / 0.25)
    assert fired
    thin = [np.array([[0.0, 0.0, 0.0], [5.2, 0.0, -0.05]])]
    st, fired = _fire(thin, below, 0.0002, LAM14)
    assert st[0].length_m == pytest.approx(0.0004 * math.hypot(5.2, 0.05) / 0.05)
    assert not fired


def test_once_per_solver():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = RazorSolver(
            wires=_crossing_deck(),
            nsegs=5,
            wire_radius=0.002,
            ground_z=0.0,
            ground_eps=SOIL,
            ground_model="sommerfeld",
            wavelength=LAM14,
        )
        solver._wire_media()
        solver._wire_media()
    assert len([w for w in caught if w.category is NearSurfaceWire]) == 1
