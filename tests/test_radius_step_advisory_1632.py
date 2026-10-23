"""A radius-step junction on the point-matched lane warns (momwire#1632).

`SinusoidalSolver` follows NEC-2 where two wire radii meet at a junction,
including NEC-2's known weakness there, and stays that way by decision: the
NEC-2 slot is meant to match nec2c. So it says so, once per solver, with
`RadiusStepJunction`, and changes nothing else. Sin-Galerkin and B-spline
give the converged answer and stay silent. Every check is construction-only: the advisory is raised before any fill.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire.bspline import BSplineSolver
from momwire.deck import build_solver, parse
from momwire.sinusoidal import (
    RadiusStepJunction,
    SinusoidalSolver,
    radius_step_junctions,
    radius_step_message,
)
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

WL = 299792458.0 / 14e6
LEFT = np.array([(-3.0, 0.0, 0.0), (0.0, 0.0, 0.0)])
RIGHT = np.array([(0.0, 0.0, 0.0), (3.0, 0.0, 0.0)])


def _advisories(cls, radii, wires=(LEFT, RIGHT), **kw):
    """Construct (not solve) #1632's two-wire dipole and return the
    RadiusStepJunction messages raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls(
            wavelength=WL,
            wire_radius=list(radii),
            wires=list(wires),
            n_per_edge_per_wire=[[20]] * len(wires),
            feeds=[(1, 0.075, 1 + 0j)],
            **kw,
        )
    return [
        str(w.message) for w in caught if issubclass(w.category, RadiusStepJunction)
    ]


def test_a_radius_step_warns_once_naming_the_wires_and_radii():
    (msg,) = _advisories(SinusoidalSolver, (0.0005, 0.001))
    assert msg.startswith("SinusoidalSolver: wires of different radius meet")
    assert "wires 0 and 1 (0.5 mm to 1 mm)" in msg
    assert "follows NEC-2" in msg
    assert "sinusoidal-galerkin or bspline" in msg
    assert "momwire#1632" in msg and "momwire#1644" in msg


def test_the_extended_kernel_does_not_silence_it():
    assert (
        len(_advisories(SinusoidalSolver, (0.0005, 0.001), extended_kernel=True)) == 1
    )


def test_uniform_radius_is_silent():
    assert _advisories(SinusoidalSolver, (0.001, 0.001)) == []
    assert _advisories(SinusoidalSolver, (0.0005, 0.0005)) == []


def test_two_radii_that_never_meet_are_silent():
    """Mixed radii alone are not a step: two parallel, unjoined wires."""
    apart = [LEFT, LEFT + np.array([0.0, 1.0, 0.0])]
    assert _advisories(SinusoidalSolver, (0.0005, 0.001), wires=apart) == []


@pytest.mark.parametrize(
    "cls", [SinusoidalGalerkinSolver, BSplineSolver], ids=["sin-galerkin", "bspline"]
)
def test_the_converged_bases_are_silent(cls):
    assert _advisories(cls, (0.0005, 0.001)) == []


def test_several_junctions_are_counted_and_capped():
    found = [(j, [j, j + 1], 0.0005, 0.001) for j in range(6)]
    msg = radius_step_message("SinusoidalSolver", found)
    assert "meet at 6 junctions" in msg
    assert "and 2 more" in msg
    assert "wires 3 and 4" in msg and "wires 4 and 5" not in msg


def test_the_scan_reads_each_junction_once():
    junctions = [[(0, "end"), (1, "start")], [(1, "end"), (2, "start")]]
    assert radius_step_junctions(junctions, [0.001, 0.001, 0.002]) == [
        (1, [1, 2], 0.001, 0.002)
    ]


_DECK = """CM momwire#1632 stepped dipole
CE
GW 1 20 -3 0 0 0 0 0 {a1}
GW 2 20 0 0 0 3 0 0 {a2}
GE 0
FR 0 1 0 0 14.
EX 0 2 1 0 1.0 0
XQ
EN
"""


@pytest.mark.parametrize(
    "basis, fires",
    [("sinusoidal", True), ("sinusoidal-galerkin", False), ("bspline", False)],
)
def test_the_deck_front_door_carries_it(basis, fires):
    """Through `momwire.deck` (the nec2 portal's and the NEC-4.2 slot's
    route to the solver): the two GW cards keep their radii as two wires
    joined at a junction, so the advisory reaches every deck-fed caller."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_solver(parse(_DECK.format(a1=0.0005, a2=0.001)), basis=basis)
    hits = [w for w in caught if issubclass(w.category, RadiusStepJunction)]
    assert len(hits) == (1 if fires else 0)
