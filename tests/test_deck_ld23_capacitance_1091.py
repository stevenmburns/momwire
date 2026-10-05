"""``LD 2`` / ``LD 3``'s capacitance field, folded with the segment length
(momwire#1091).

NEC scales that field BY the segment length (momwire#1088's measurement):
``LD 2`` with C' alone is ``LD 0`` with a lumped C'·d on every segment, so the
card's per-metre impedance is 1/(jωC'd²).  The ``nec2`` dialect serves it by
folding the field into a per-metre series capacitance C'·d² with each wire's
own d; ``nec4`` (NEC-4.2 unmeasured) and ``nec5`` still refuse it.

The oracle numbers below are nec2c 1.3.1's printed feed-point impedance
(2026-10-04, Skylake ``/usr/bin/nec2c``) for the deck in :func:`_deck`, read
off its ANTENNA INPUT PARAMETERS row.  ``sinusoidal`` is NEC-2's own basis
(a same-basis comparison, the only kind momwire#510 allows gating).
"""

from __future__ import annotations

import pytest

from momwire.deck import DeckError, build_solver
from momwire.deck import parse as parse_nec2
from momwire.deck._nec5 import parse_nec5

# (segments, C' field, LD type, R' field) -> nec2c's Z, ohms.
NEC2C = {
    (5, 1e-9, 2, 0.0): 78.268 + 30.688j,
    (9, 1e-9, 2, 0.0): 76.640 + 0.582j,
    (17, 1e-9, 2, 0.0): 69.677 - 107.780j,
    (33, 1e-9, 2, 0.0): 49.133 - 449.590j,
    (9, 1e-11, 3, 100.0): 333.970 - 6.632j,
    (17, 1e-11, 3, 100.0): 337.530 - 2.725j,
}
# The same decks without the card, for the size of the card's own effect.
NEC2C_BARE = {5: 78.883 + 44.520j, 9: 79.240 + 45.364j, 17: 79.699 + 45.933j}


def _deck(n: int, c: float, kind: int = 2, r: float = 0.0) -> str:
    """A 5 m centre-fed dipole of 1 mm wire at 30 MHz, ``n`` segments."""
    return (
        f"CE ld{kind} c\nGW 1 {n} 0 0 -2.5 0 0 2.5 0.001\nGE 0\n"
        f"LD {kind} 1 0 0 {r} 0. {c}\nEX 0 1 {n // 2 + 1} 0 1. 0.\n"
        f"FR 0 1 0 0 30. 0\nXQ\nEN\n"
    )


def _z(text: str, basis: str) -> complex:
    built = build_solver(parse_nec2(text), basis=basis)
    port = built.ports.feed_ports[0]
    return complex(1.0 / built.solver.compute_port_solution().y[port, port])


def test_the_field_folds_with_each_wires_own_segment_length():
    """Two wires of different segment length under one whole-structure card:
    each gets C'·d² with ITS d, so each segment's integral is NEC's C'·d."""
    model = parse_nec2(
        "GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3\n"
        "GW 2 8 0. 1. 0. 1. 1. 0. 1.E-3\n"
        "GE 0\nEX 0 1 3 0 1.\nFR 0 1 0 0 14.\n"
        "LD 2 0 0 0 0. 0. 2.E-9\nXQ\nNX\n"
    )
    first, second = (w.material.distributed_rlc for w in model.wires)
    assert first.c == pytest.approx(2e-9 * 0.25**2, rel=1e-15)
    assert second.c == pytest.approx(2e-9 * 0.125**2, rel=1e-15)


@pytest.mark.parametrize(("n", "c", "kind", "r"), sorted(NEC2C))
def test_sinusoidal_reproduces_nec2c(n, c, kind, r):
    """NEC-2's basis lands on nec2c's printed Z to 0.25 ohm, where the card
    itself moves X by 14 to 496 ohm (`NEC2C_BARE`)."""
    assert abs(_z(_deck(n, c, kind, r), "sinusoidal") - NEC2C[n, c, kind, r]) < 0.25


@pytest.mark.parametrize("n", [5, 9, 17])
def test_bspline_carries_the_cards_effect(n):
    """bspline is a different basis, so its gate is the card's own effect:
    the change in X the card makes agrees with nec2c's to 2 % (measured
    0.9 %, 0.4 %, 0.1 % at 5, 9, 17 segments)."""
    loaded = _z(_deck(n, 1e-9), "bspline")
    bare = _z(_deck(n, 0.0), "bspline")
    oracle = (NEC2C[n, 1e-9, 2, 0.0] - NEC2C_BARE[n]).imag
    assert abs((loaded - bare).imag / oracle - 1.0) < 0.02


def test_a_negative_capacitance_still_refuses():
    with pytest.raises(DeckError, match="negative per-unit-length capacitance"):
        parse_nec2(_deck(5, -1e-9))


def test_the_nec4_dialect_keeps_the_refusal():
    with pytest.raises(DeckError, match="no NEC-4.2 run shows"):
        parse_nec2(_deck(5, 1e-9), dialect="nec4")


def test_the_nec5_dialect_keeps_the_refusal():
    deck = (
        "CM w\nCE\nGW 1,9,0.,-.25,0.,0.,.25,0.,.0005\nGE 0,-1\nGN -1\n"
        "FR 0,1,0,0,299.7925\nLD 3,1,0,0,0.,0.,1.E-11\nEX 4,1,5,0,1.,0.\nXQ\nEN\n"
    )
    with pytest.raises(DeckError, match="asks for a capacitance"):
        parse_nec5(deck)
