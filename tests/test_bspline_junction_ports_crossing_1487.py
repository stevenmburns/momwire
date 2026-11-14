"""bspline junction ports (an end drive) on a deck with an interface-crossing
wire -- momwire#1487.

The end drive antennaknobs builds from `PortAtEnd`: a conductor split at
the feed so that two wire ends face each other across a 1 cm gap, each end
a one-member junction port, and Z the differential `Z11 + Z22 - Z12 - Z21`
of `inv(Y)`.

The crossing fill adds a by-parts end term for every wire end where a basis
is nonzero. At a junction port's end that term double-counts the port (the
boundary term there IS the port's impressed voltage, and the end current
flows into the source, not into a point charge), so with ANY crossing wire
in the deck the two ported faces read as their own end-to-end capacitance
(Y11 ~ j5 uS, Y12 ~ 0) and Z came back near -j357 kOhm, silently. The
fix leaves a ported end out of the crossing end table
(`CrossingContext.port_ends`), which is what every non-crossing fill
already does.

The oracle is the gap feed, which the defect never reached: the same
conductor UNSPLIT, driven by a point gap at the split's centre. Two gates
derive from it (measured on Haswell, x1 mesh, 7.2 MHz, Sommerfeld 13/0.005):

* A rod 5 m away is mutual coupling only, so it may move the end drive no
  more than it moves the gap feed. Measured |dZ_end| / |dZ_gap| = 0.23 at
  both feed points; the defect gave ~1e7.
* The end-vs-gap feed-model difference, in admittance (~ the gap's own
  capacitance at the high-impedance 0.25 m feed), is a local property and
  must not move with the rod. Measured: 2.3 % at the centre, which is
  EXACTLY what an unconnected wholly-buried wire (never on the crossing
  path) moves it by, and 0.1 % at 0.25 m. Gated at 5 %.

The G8HUL-like stake has no rod-free twin of the same structure, so its
gate is that feed-model admittance against the raised 0.25 m deck's: 10 %
apart (the stake puts a ground contact 0.25 m below the gap), gated at
25 %; the defect put it 17x off.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire.bspline import BSplineSolver

L, A, F = 20.1, 1e-3, 7.2
H0 = L / 40
GAP = 0.01
GROUND = dict(ground_z=0.0, ground_eps=(13.0, 0.005), ground_model="sommerfeld")
KW = dict(exact_kernel=False)


def _deck(config, feed, drive):
    """`drive` 'end': split at `feed` with a GAP, two one-member junction
    ports. 'gap': the conductor unsplit, a vertex at `feed` carrying a point
    gap. `config` 'raised*' lifts it 2 m; '_buried' adds an unconnected
    buried wire 5 m away; '_crossing' a separate rod crossing the interface
    5 m away; 'stake' starts it at z = 0 on a 2 m buried stake."""
    z0 = 0.0 if config == "stake" else 2.0
    if drive == "end":
        lo_top, up_bot = feed - GAP / 2, feed + GAP / 2
        wires = [
            np.array([[0, 0, z0], [0, 0, z0 + lo_top]]),
            np.array([[0, 0, z0 + up_bot], [0, 0, z0 + L]]),
        ]
        nper = [[max(1, round(lo_top / H0))], [max(1, round((L - up_bot) / H0))]]
        junctions = [[(0, "end")], [(1, "start")]]
        feeds, ports = [], [(0, 1.0 + 0j), (1, 1.0 + 0j)]
    else:
        wires = [np.array([[0, 0, z0], [0, 0, z0 + feed], [0, 0, z0 + L]])]
        nper = [[max(1, round(feed / H0)), max(1, round((L - feed) / H0))]]
        junctions, feeds, ports = [], [(0, feed, 1.0 + 0j)], []
    if config.endswith("buried"):
        wires.append(np.array([[5.0, 0, -0.5], [5.0, 0, -2.0]]))
        nper.append([3])
    if config.endswith("crossing"):
        b = len(wires)
        wires += [np.array([[5.0, 0, 0.0], [5.0, 0, 1.0]])]
        wires += [np.array([[5.0, 0, 0.0], [5.0, 0, -1.0]])]
        nper += [[2], [2]]
        junctions = junctions + [[(b, "start"), (b + 1, "start")]]
    if config == "stake":
        b = len(wires)
        wires.append(np.array([[0, 0, 0.0], [0, 0, -2.0]]))
        nper.append([10])
        junctions = junctions + [[(0, "start"), (b, "start")]]
    return dict(
        wires=wires,
        n_per_edge_per_wire=nper,
        feeds=feeds,
        junctions=junctions,
        junction_ports=ports,
    )


def _z(config, feed, drive):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = BSplineSolver(
            wavelength=299_792_458.0 / (F * 1e6),
            wire_radius=A,
            **GROUND,
            **_deck(config, feed, drive),
            **KW,
        )
        y = np.asarray(s.compute_port_solution().y)
    if drive == "gap":
        return complex(1.0 / y[0, 0])
    zp = np.linalg.inv(y)
    return complex(zp[0, 0] + zp[1, 1] - zp[0, 1] - zp[1, 0])


def _feed_model_y(config, feed):
    """Y_end - Y_gap: what the end drive's model adds over a point gap."""
    return 1.0 / _z(config, feed, "end") - 1.0 / _z(config, feed, "gap")


@pytest.mark.parametrize("feed", [9.80, 0.25], ids=["centre", "split-0.25m"])
def test_a_distant_crossing_rod_moves_the_end_drive_by_coupling_only(feed):
    z_end, z_end_rod = _z("raised", feed, "end"), _z("raised_crossing", feed, "end")
    z_gap, z_gap_rod = _z("raised", feed, "gap"), _z("raised_crossing", feed, "gap")
    # The defect's signature, named so a regression reads as itself.
    assert abs(z_end_rod.imag) < 1e4, z_end_rod
    assert abs(z_end_rod - z_end) <= abs(z_gap_rod - z_gap), (
        z_end,
        z_end_rod,
        z_gap,
        z_gap_rod,
    )


@pytest.mark.parametrize("feed", [9.80, 0.25], ids=["centre", "split-0.25m"])
def test_the_feed_model_difference_does_not_move_with_the_rod(feed):
    base = _feed_model_y("raised", feed)
    rod = _feed_model_y("raised_crossing", feed)
    assert abs(rod - base) <= 0.05 * abs(base), (base, rod)


def test_the_g8hul_stake_end_drive_matches_the_gap_feed_model():
    base = _feed_model_y("raised", 0.25)
    stake = _feed_model_y("stake", 0.25)
    z_stake = _z("stake", 0.25, "end")
    assert abs(z_stake.imag) < 1e4, z_stake
    assert abs(stake - base) <= 0.25 * abs(base), (base, stake)


# The decks the defect never reached, pinned at their pre-fix values
# (momwire main 84fee248, Haswell): no crossing wire, so no crossing fill.
_UNCHANGED = {
    ("raised", 9.80): 82.15360800285062 - 19.691367550729918j,
    ("raised_buried", 9.80): 82.15363784563948 - 19.68768234416326j,
    ("raised", 0.25): 1891.0719281232205 - 7749.056828418276j,
}


@pytest.mark.parametrize(
    "config, feed",
    [
        # The buried wire's cold below/below grid is ~26 s on Haswell.
        pytest.param(c, f, marks=pytest.mark.slow) if c.endswith("buried") else (c, f)
        for c, f in _UNCHANGED
    ],
)
def test_decks_without_a_crossing_wire_are_unchanged(config, feed):
    z = _z(config, feed, "end")
    want = _UNCHANGED[(config, feed)]
    assert abs(z - want) <= 1e-9 * abs(want), (z, want)


def test_a_crossing_deck_without_junction_ports_carries_no_port_ends():
    """The new context field is empty unless a junction port exists, which
    is what keeps every other crossing deck's end table -- and bytes --
    as they were."""
    d = _deck("raised_crossing", 9.80, "gap")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = BSplineSolver(
            wavelength=299_792_458.0 / (F * 1e6), wire_radius=A, **GROUND, **d, **KW
        )
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    assert s._crossing_context(geom, supp_seg, polys).port_ends == frozenset()
    d = _deck("raised_crossing", 9.80, "end")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = BSplineSolver(
            wavelength=299_792_458.0 / (F * 1e6), wire_radius=A, **GROUND, **d, **KW
        )
    geom = s._build_geometry()
    supp_seg, polys, *_ = s._build_basis_polynomials(geom)
    assert s._crossing_context(geom, supp_seg, polys).port_ends == frozenset(
        {(0, "end"), (1, "start")}
    )
