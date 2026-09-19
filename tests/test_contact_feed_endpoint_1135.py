"""A `PortOnWire` may sit at an endpoint, because a grounded end has a current
path (momwire#1135).

`PortOnWire` refused `at` outside the OPEN interval (0, 1), on the rule its
docstring states: *"A port must interrupt a current path, so it lives in a
wire's interior, never at an endpoint."*

That holds at a FREE end, where the current is zero and there is nothing to
interrupt. It does not hold at an end standing IN the ground plane: there the
current path is the ground contact, the image supplies the other side of the
gap, and it is exactly the drive `test_contact_nec5_lane.py` certifies against
NEC-5 over five grounds and five densities at `feed_arclength=0.0`.
`momwire.eznec.serve` places a deck's contact feed there too.

So the solver served it and the certified lane used it, while the port type
forbade saying it — which cost antennaknobs a real defect (AK#1598 settled for
the first mesh cell's CENTRE instead, a different drive under the default
`feed_model="point"`, 29 % in X on a base-fed vertical).

`PortOnWire` cannot tell a grounded end from a free one, because ground is not
in its vocabulary. So it admits the position, and what stops a free end being
asked for silently is that the answer is an OPEN, which is correct rather than
plausible-but-wrong — `test_a_free_end_feed_reads_as_the_open_it_is` measures
that.
"""

import math

import numpy as np
import pytest

from momwire.bspline import BSplineSolver
from momwire.networks import PortOnWire

WL = 40.0
MONO_H = WL / 4.0
RAD = 1.0e-3
N = 21


def _monopole(arclength, *, grounded, model="segment"):
    return BSplineSolver(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, MONO_H]])],
        n_per_edge_per_wire=[[N]],
        wire_radius=RAD,
        wavelength=WL,
        degree=2,
        feed_model=model,
        feed_wire_index=0,
        feed_arclength=arclength,
        **({"ground_z": 0.0} if grounded else {}),
    )


def test_the_port_type_admits_an_endpoint():
    """The one-line change, and the reason the rest of this file exists."""
    assert PortOnWire("feed", wire="w", at=0.0).at == 0.0
    assert PortOnWire("feed", wire="w", at=1.0).at == 1.0


def test_a_position_off_the_wire_is_still_refused():
    """Relaxing to the CLOSED interval is not relaxing the check."""
    for bad in (-1e-9, 1.0 + 1e-9, math.nan, math.inf):
        with pytest.raises(ValueError, match=r"fraction in \[0, 1\]"):
            PortOnWire("feed", at=bad)


def test_the_grounded_endpoint_is_the_certified_contact_drive():
    """`test_contact_nec5_lane.py` drives its base-fed monopoles at exactly
    this arclength and model. Pinned against that lane's own configuration, so
    the position this type now admits is the one already certified."""
    z, _ = _monopole(0.0, grounded=True).compute_impedance()
    assert complex(z) == pytest.approx(
        complex(39.201378865973574, 22.566288939925254), rel=1e-9
    )


def test_a_point_gap_lands_exactly_where_it_was_asked_to():
    """`feed_placements()` reports requested vs placed, which is the direct
    statement that an endpoint is not snapped inward. Under `point` the drive
    IS the named arclength, so placed == requested == 0."""
    (placement,) = _monopole(0.0, grounded=True, model="point").feed_placements()
    assert placement.requested == 0.0
    assert placement.placed == 0.0


def test_the_segment_model_reads_the_cell_and_the_point_model_does_not():
    """Why the endpoint matters at all, stated as a measurement.

    Under `segment` the gap is the mesh CELL holding the arclength, so the
    contact and that cell's centre are ONE drive and the position inside the
    cell is a non-question. Under `point` — the production default on this
    solver and on `SinusoidalGalerkinSolver` since momwire#654 — they are two
    different drives. That difference is what makes admitting the endpoint
    worth doing rather than cosmetic."""
    seg = MONO_H / N
    same = [
        complex(_monopole(a, grounded=True, model="segment").compute_impedance()[0])
        for a in (0.0, 0.5 * seg)
    ]
    assert same[0] == pytest.approx(same[1], rel=0, abs=0)  # one cell, one drive

    differ = [
        complex(_monopole(a, grounded=True, model="point").compute_impedance()[0])
        for a in (0.0, 0.5 * seg)
    ]
    assert abs(differ[0] - differ[1]) / abs(differ[0]) > 1e-3


def test_a_free_end_feed_reads_as_the_open_it_is():
    """The risk admitting endpoints could have created, measured and closed.

    A gap at a FREE end has no current path to interrupt, so the question is
    whether it returns a plausible WRONG driving point or an obvious one. It
    returns the obvious one: an essentially OPEN port, which is the correct
    answer for a port with nothing to drive — the same wire reads
    31.19 - 13017j fed at its free end against 39.20 + 22.57j when that end
    stands in the ground plane, a reactance three orders larger and of the
    opposite sign.

    Measured at two heights to show it is the END and not the geometry: an
    elevated free end gives the identical number, so nothing about proximity
    to the plane is doing the work.

    (An earlier draft of this test asserted the result was non-finite. That is
    true under `feed_model="point"` on some geometries and false here — the
    honest invariant is the open, not the NaN.)

    A refusal naming the cause would still beat a 13 kOhm reactance, and is
    worth filing; this test exists so that if the answer ever becomes an
    ordinary-looking impedance, someone has to come back and think about it.
    """
    grounded = complex(_monopole(0.0, grounded=True).compute_impedance()[0])
    free = complex(_monopole(0.0, grounded=False).compute_impedance()[0])
    assert abs(free.imag) > 100.0 * abs(grounded.imag)
    assert free.imag < 0.0  # capacitive: an open, not a resonance

    elevated = BSplineSolver(
        wires=[np.array([[0.0, 0.0, 5.0], [0.0, 0.0, 5.0 + MONO_H]])],
        n_per_edge_per_wire=[[N]],
        wire_radius=RAD,
        wavelength=WL,
        degree=2,
        feed_model="segment",
        feed_wire_index=0,
        feed_arclength=0.0,
    )
    assert complex(elevated.compute_impedance()[0]) == pytest.approx(free, rel=1e-9)
