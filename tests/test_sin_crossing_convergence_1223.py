"""The point-matched crossing node closes on the Galerkin lanes — momwire#1223 U6.

Stage 2's decision 7: convergence onto SG / B-spline shrinks monotonically
over three graded rungs and sits under a fixed bar at the finest. R and X are
gated SEPARATELY: U5's first gate read one complex gap and passed over an R
plateau that X's convergence hid (the lesson recorded on #1223).

The reference is B-spline at n = 161, checked against SG at the same mesh on
every deck over soil A. The two share no fill code below the interface, so their
agreement is what makes the reference a reference. The bars are 1.5x the
values measured on Skylake (2026-09-28, #1223 U6 comment), R relative to R
and X relative to |Z|: the rod's X is -2500 ohm, so R relative to |Z| would
be blind.

Decks (7 MHz):
  * rod: a vertical 2 m above and 1.5 m below, collinear (tilt sum 0);
  * lean: the same, the buried member 25 deg off the normal (in scope);
  * hub4: the BRV class. A quarter-wave mast, a 15 cm rise to a buried hub,
    and four 5 m radials. The node is two-member and vertical; the hub is an
    ordinary wholly-below junction.

Plus the record of what option A refuses: an unscoped 45 deg lean sits more
than 1 % off the rod in R at both rungs, so the refusal is not caution about
nothing.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from momwire.bspline import BSplineSolver
from momwire.sinusoidal import SinusoidalSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

pytestmark = pytest.mark.slow

C0 = 299792458.0
WL7 = C0 / 7e6
RUNGS = (21, 41, 81)
N_REF = 161


def _eps(er, sigma):
    return complex(er, -sigma / (2 * np.pi * 7e6 * 8.8541878128e-12))


SOILS = {"soilA": (13.0, 0.005), "lossy": (30.0, 0.05)}


def _lean(deg):
    t = math.radians(deg)
    return (
        [
            [[0, 0, 0.0], [0, 0, 2.0]],
            [[0, 0, 0.0], [1.5 * math.sin(t), 0, -1.5 * math.cos(t)]],
        ],
        [[(0, "start"), (1, "start")]],
        (0, 1.0),
    )


def _hub4():
    h, d, arm = 10.0, 0.15, 5.0
    wires = [[[0, 0, 0.0], [0, 0, h]], [[0, 0, 0.0], [0, 0, -d]]]
    for i in range(4):
        a = i * math.pi / 2
        wires.append([[0, 0, -d], [arm * math.cos(a), arm * math.sin(a), -d]])
    junctions = [
        [(0, "start"), (1, "start")],
        [(1, "end")] + [(2 + i, "start") for i in range(4)],
    ]
    return wires, junctions, (0, 0.5)


DECKS = {"rod": _lean(0.0), "lean": _lean(25.0), "hub4": _hub4()}

# (bar on |dR|/R, bar on |dX|/|Z|) at the finest rung, 1.5x the measured
# worst of the two soils: rod 1.28 % / 0.67 %, lean 0.70 % / 0.67 %,
# hub4 2.37 % / 2.20 %.
BARS = {"rod": (0.020, 0.010), "lean": (0.012, 0.010), "hub4": (0.036, 0.033)}


class _Unscoped(SinusoidalSolver):
    """Option A's scope switched off, to measure what it refuses."""

    def _refuse_oblique_crossing(self, crossing):
        pass


def _z(cls, deck, n, soil):
    wires, junctions, (fw, fp) = deck
    lens = [float(np.linalg.norm(np.subtract(w[1], w[0]))) for w in wires]
    longest = max(lens)
    npw = [[max(3, int(round(n * length / longest)))] for length in lens]
    s = cls(
        wires=wires,
        n_per_edge_per_wire=npw,
        feeds=[(fw, fp, 1 + 0j)],
        wavelength=WL7,
        wire_radius=0.001,
        junctions=junctions,
        ground_z=0.0,
        ground_eps=_eps(*soil),
        ground_model="sommerfeld",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return complex(s.compute_impedance()[0])


@pytest.mark.parametrize("soil", sorted(SOILS))
@pytest.mark.parametrize("deck", sorted(DECKS))
def test_the_point_matched_node_closes_on_the_galerkin_lanes(deck, soil):
    d, s = DECKS[deck], SOILS[soil]
    ref = _z(BSplineSolver, d, N_REF, s)
    if soil == "soilA":
        # The reference is two independent lanes agreeing, not one lane's
        # word (measured <= 0.004 % in R on all three decks). Checked on one
        # soil: SG at n = 161 is the costliest solve in this file.
        sg = _z(SinusoidalGalerkinSolver, d, N_REF, s)
        assert abs(sg.real - ref.real) <= 1e-3 * ref.real
        assert abs(sg.imag - ref.imag) <= 1e-3 * abs(ref)

    er, ex = [], []
    for n in RUNGS:
        z = _z(SinusoidalSolver, d, n, s)
        er.append(abs(z.real - ref.real) / ref.real)
        ex.append(abs(z.imag - ref.imag) / abs(ref))
    assert er[0] > er[1] > er[2], er
    assert ex[0] > ex[1] > ex[2], ex
    bar_r, bar_x = BARS[deck]
    assert er[-1] <= bar_r, er
    assert ex[-1] <= bar_x, ex


def test_the_refused_scope_is_biased():
    """What option A refuses: a 45 deg lean's R sits more than 1 % off the
    rod's at the same mesh, and it does not close as the rod does (measured
    -1.9 % at n = 161 against B-spline and SG at 321)."""
    soil = SOILS["soilA"]
    ref_rod = _z(BSplineSolver, DECKS["rod"], N_REF, soil)
    ref_45 = _z(BSplineSolver, _lean(45.0), N_REF, soil)
    for n in (41, 81):
        e_rod = (
            _z(SinusoidalSolver, DECKS["rod"], n, soil).real - ref_rod.real
        ) / ref_rod.real
        e_45 = (_z(_Unscoped, _lean(45.0), n, soil).real - ref_45.real) / ref_45.real
        assert e_rod - e_45 > 0.01, (n, e_rod, e_45)
