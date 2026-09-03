"""The per-deck cross-edge quadrature default (momwire#760).

#760 recorded the buried/crossing class as losing its convergence RATE --
"the cross-edge quadrature error falls only as C/q, first order". Walked
against a converged (q=256) reference it does not: the error is
superalgebraic everywhere, and the local slope RISES rather than sitting at
a constant. The class carries a large quadrature CONSTANT instead, so the
remedy is order, not a singularity subtraction.

Measured on main @ f729cb5, soil A, degree 2, |Z(q) - Z(256)| in ohms:

    deck                              q=4     q=8    q=16    q=32     fit
    antennaknobs bundle (retired)   7.004   3.121   1.136   0.303  q^-1.78
    fan_rise_deck_graded('n2')      6.808   2.556   0.698   0.105  q^-2.57
    antennaknobs hub (shipped)      0.675   0.172  0.0367  0.0047  q^-3.11
    crossing_deck(1)                0.433   0.093  0.0087  0.0003  q^-5.19
    hub_deck()                      0.193   0.051  0.0092  0.0001  q^-4.12

First order would be q^-1.00. The "C/q" came from the bottom of the BUNDLE
ladder (slope 1.17 at q=4->8, climbing to 3.09 by q=64) measured against an
unconverged q=32 reference -- the origin deck is antennaknobs'
`buried_radial_vertical:bundle`, a coincident-rise spelling that
antennaknobs#1108 has since retired and razor refuses by name (#846/#856).

Why the order is not raised everywhere: on a buried deck the Sommerfeld
evaluation dominates and the order is free (+1-2% steady state), but in free
space the pair quadrature IS the cost and it is O(n_qp^2) -- on a bent
400-segment free-space deck, 8 -> 32 measured **4.8x**. A single-edge deck
is unaffected either way, since momwire#759 it never enters an off-edge
kernel at all -- which is exactly why timing this on a straight wire would
have made a global change look free.
"""

import sys

import numpy as np
import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from momwire.bspline import BURIED_N_QP_PAIR, DEFAULT_N_QP_PAIR, BSplineSolver
from test_crossing_serve_524 import crossing_deck

_BENT_FREE = np.array([[0.0, -5.0, 0.0], [0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])


def _free_space(**kw):
    return BSplineSolver(
        wires=[_BENT_FREE],
        nsegs=12,
        wavelength=22.0,
        wire_radius=5e-4,
        degree=2,
        feed_wire_index=0,
        feed_arclength=5.0,
        **kw,
    )


def test_a_buried_deck_resolves_to_the_buried_order():
    assert BSplineSolver(**crossing_deck(1)).n_qp_pair == BURIED_N_QP_PAIR


def test_free_space_keeps_the_shipped_order():
    """The scoping, pinned. A global raise costs 4.8x on this deck class at
    400 segments (see the module docstring), so free space must not move."""
    assert _free_space().n_qp_pair == DEFAULT_N_QP_PAIR


def test_an_explicit_order_is_never_second_guessed():
    """Including one that matches neither default, and on both deck classes:
    a caller who names an order owns it."""
    assert BSplineSolver(**dict(crossing_deck(1), n_qp_pair=5)).n_qp_pair == 5
    assert _free_space(n_qp_pair=5).n_qp_pair == 5
    # And the shipped default, passed explicitly on a buried deck, must stay 8
    # rather than being "helpfully" promoted -- that is what lets a caller
    # reproduce a pre-#760 number.
    assert BSplineSolver(**dict(crossing_deck(1), n_qp_pair=8)).n_qp_pair == 8


# The three gates below SOLVE, and a `crossing_deck(1)` Sommerfeld fill is
# ~1 s alone but ~12 s under the xdist workers the suite runs, so they are
# `slow` (push lane) rather than PR-lane. The behaviour this PR changes -- WHICH
# order a deck resolves to -- is pinned by the free gates above and stays in the
# PR lane; what moves to the push lane is the physics justifying the number.

_Z_CACHE: dict[int, complex] = {}


def _z(q):
    """One solve per order for the whole module.

    Each `crossing_deck(1)` solve is ~1 s alone and more under xdist, and the
    q=128 reference is shared by every gate below -- recomputing it per test
    put three of them over the suite's 20 s hard ceiling.
    """
    if q not in _Z_CACHE:
        _Z_CACHE[q] = BSplineSolver(
            **dict(crossing_deck(1), n_qp_pair=q)
        ).compute_impedance()[0]
    return _Z_CACHE[q]


REF_Q = 128


@pytest.mark.slow
def test_the_resolved_default_lands_inside_a_stated_bar():
    """The substantive gate: at the resolved order the deck is converged to
    0.01 ohm against a q=128 reference."""
    assert abs(_z(BURIED_N_QP_PAIR) - _z(REF_Q)) < 0.01


@pytest.mark.slow
def test_the_shipped_default_would_not_have():
    """...and this is why the change is worth making rather than a no-op.

    The old default misses that bar by ~9x on this deck. Without this test the
    one above passes just as well at `BURIED_N_QP_PAIR = 8` and the whole
    change is unobservable -- which is exactly the state the suite was in
    before this file: setting the constant to 3 broke NOTHING in the buried
    and crossing suites, because every banked deck there passes an explicit
    order. This default had no coverage at all.
    """
    assert abs(_z(DEFAULT_N_QP_PAIR) - _z(REF_Q)) > 0.05


def test_the_two_constants_are_ordered_and_the_buried_one_is_the_larger():
    """Cheap, but it is the invariant the whole file rests on: if these ever
    invert, every gate above still passes while the class silently gets
    COARSER quadrature than free space."""
    assert BURIED_N_QP_PAIR > DEFAULT_N_QP_PAIR


@pytest.mark.slow
def test_the_error_shrinks_monotonically_with_order():
    """The rate claim at the cheap end: whatever the exponent, more order is
    never worse. A first-order-in-disguise fit would satisfy this too; what it
    rules out is the ladder being non-monotone, which is how a "converged"
    reference at too low an order misleads -- #760's origin used q=32 as its
    reference while it was still 0.30 ohm from converged.
    """
    errs = [(q, abs(_z(q) - _z(REF_Q))) for q in (4, 8, 16, 32, 64)]
    for (q0, e0), (q1, e1) in zip(errs, errs[1:]):
        assert e1 < e0, (q0, e0, q1, e1)


def test_stripping_the_ground_keys_changes_the_resolved_order():
    """The hazard a per-deck default creates, pinned where it will be read.

    momwire#813's ε̃ = 1 collapse adjudicators build their free-space TRUTH by
    stripping the ground keys off the same build dict. Before #760 both sides
    inherited one global default and matched automatically; now the buried
    side resolves higher and the stripped side does not, so an unpinned
    adjudicator compares two quadratures and fails by the gap rather than by
    anything about the composition. Three of the four did exactly that.

    This is the cost of resolving a default from geometry, and it is worth a
    gate rather than a comment: anything that compares a buried result against
    a free-space reference must pin the order on BOTH sides.
    """
    build = crossing_deck(1, ground_eps=(1.0, 0.0))
    truth = {
        k: v
        for k, v in build.items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    assert BSplineSolver(**build).n_qp_pair == BURIED_N_QP_PAIR
    assert BSplineSolver(**truth).n_qp_pair == DEFAULT_N_QP_PAIR

    # ...and pinning the order is what makes the two comparable again.
    pinned = DEFAULT_N_QP_PAIR
    assert (
        BSplineSolver(**dict(build, n_qp_pair=pinned)).n_qp_pair
        == BSplineSolver(**dict(truth, n_qp_pair=pinned)).n_qp_pair
        == pinned
    )
