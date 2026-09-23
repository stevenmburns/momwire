"""SinusoidalGalerkinSolver.compute_impedance never returns a NaN — momwire#1162.

#1159's probe 8 read Z = NaN off an L with `junction_ports=[0]` and no feeds.
The cause was not the junction-port serve. A plain-int junction port means
0 V (the #172 convention every family shares), so that deck drove nothing:
the RHS was zero, alpha identically zero, every port current exactly 0, and
Z = V/I came out 0/0. Driven at 1 V the same port agrees with BSplineSolver
to ~1e-5 at every mesh from 10+8 to 80+64 segments, free space and PEC alike.

What this module pins:

  * an undriven deck REFUSES, with a sentence naming the drive-independent
    route (Y), on both the single-k and the swept entry;
  * that route answers on the same deck, and is the same Y a driven solve
    reads back (Z = 1/Y for one port);
  * the driven junction port agrees with BSplineSolver, so the refusal is
    not standing in for a serve SG lacks;
  * a backstop: a port current of exactly 0 on a DRIVEN deck refuses too,
    rather than returning NaN or inf as a Z.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire.bspline import BSplineSolver
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver

WL = 299792458.0 / 7e6
GROUNDS = {"free": {}, "pec": {"ground_z": 0.0}}


def _l_deck(**kw):
    """probe 8's L: a 5 m vertical and a 4 m horizontal meeting at one
    junction, 1 m above the plane when there is one."""
    return dict(
        wires=[
            np.array([(0.0, 0.0, 1.0), (0.0, 0.0, 6.0)]),
            np.array([(0.0, 0.0, 6.0), (4.0, 0.0, 6.0)]),
        ],
        n_per_edge_per_wire=[[10], [8]],
        junctions=[[(0, "end"), (1, "start")]],
        wavelength=WL,
        wire_radius=1e-3,
        **kw,
    )


def _solver(cls=SinusoidalGalerkinSolver, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(**_l_deck(**kw))


@pytest.mark.parametrize("ground", GROUNDS)
def test_the_probe8_deck_refuses_instead_of_nan(ground):
    s = _solver(feeds=[], junction_ports=[0], **GROUNDS[ground])
    with pytest.raises(ValueError, match="no port is driven"):
        s.compute_impedance()
    with pytest.raises(ValueError, match="no port is driven"):
        s.compute_impedance_swept(np.array([]))
    with pytest.raises(ValueError, match="no port is driven"):
        s.compute_impedance_swept(np.array([s.k]))


def test_a_zero_volt_gap_feed_refuses_too():
    """Not junction-port specific: any deck whose every port is at 0 V."""
    s = _solver(feeds=[(0, 2.5, 0j)])
    with pytest.raises(ValueError, match="no port is driven"):
        s.compute_impedance()


@pytest.mark.parametrize("ground", GROUNDS)
def test_the_named_route_answers_and_is_the_driven_solves_y(ground):
    y0 = _solver(feeds=[], junction_ports=[0], **GROUNDS[ground])
    y0 = y0.compute_port_solution().y
    assert np.all(np.isfinite(y0))
    z1, _ = _solver(
        feeds=[], junction_ports=[(0, 1 + 0j)], **GROUNDS[ground]
    ).compute_impedance()
    assert np.isfinite(z1)
    assert abs(z1 * y0[0, 0] - 1.0) < 1e-12


@pytest.mark.parametrize("ground", GROUNDS)
def test_the_driven_junction_port_agrees_with_bspline(ground):
    """Measured 1.1e-5 (free) and 1.1e-5 (PEC) at this mesh; the two stay
    within 2e-5 of each other to 8x refinement while both drift ~0.2 ohm."""
    kw = dict(feeds=[], junction_ports=[(0, 1 + 0j)], **GROUNDS[ground])
    z_sg, _ = _solver(**kw).compute_impedance()
    z_bs, _ = _solver(cls=BSplineSolver, **kw).compute_impedance()
    assert abs(z_sg - z_bs) / abs(z_bs) < 1e-4


def test_a_zero_port_current_on_a_driven_deck_refuses(monkeypatch):
    """The backstop behind the up-front check: V/I at I == 0 is not a Z."""
    s = _solver(feeds=[(0, 2.5, 1 + 0j)], junction_ports=[0])

    def _zero_currents(self, alpha, geom, seg_view, U):
        return np.zeros(self.n_ports, dtype=np.complex128)

    monkeypatch.setattr(SinusoidalGalerkinSolver, "_port_currents", _zero_currents)
    with pytest.raises(FloatingPointError, match=r"port\(s\) \[0, 1\]"):
        s.compute_impedance()


def test_a_driven_deck_with_an_undriven_port_still_answers():
    """A 0 V port beside a driven one reads Z = 0/I = 0, a number — the
    guard must not refuse it."""
    z, _ = _solver(feeds=[(0, 2.5, 1 + 0j)], junction_ports=[0]).compute_impedance()
    assert np.all(np.isfinite(z))
    assert z[1] == 0
