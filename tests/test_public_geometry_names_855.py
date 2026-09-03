"""momwire#855 — the two interface-side geometry answers, promoted.

A consumer that refuses a buried deck the way momwire refuses it has to
answer two questions IDENTICALLY, or its refusal fires on a deck momwire
would have served:

  * "is this wire end on the plane" — `ground_touch_tol`, a per-wire
    RELATIVE tolerance (1e-6 of the wire's own length). A locally invented
    absolute one disagrees at the margin, which is the whole point of
    exporting this rather than documenting the number.
  * "does this in-plane junction earn the crossing exemption" —
    `grounded_crossing_exemption`.

THE IRONY IS THE ARGUMENT, and it is why these are exports rather than a
documented recipe. momwire#848 exists because two copies of the exemption
test — `BSplineSolver`'s and `RazorSolver`'s — drifted apart and answered
differently on the same deck; it moved the geometry into one place so they
could not. A consumer with no public path was then obliged to import
privately or write the THIRD copy, which is the failure #848 had just
finished repairing one layer down.

So these are RE-EXPORTS, never reimplementations: `momwire.ground_touch_tol`
IS `_ground_spec.ground_touch_tol`, the object the solvers call. The first
test pins that, because a well-meaning later edit that gave the public name
its own body would restore exactly the problem this closes.

On `wire_to_element`'s precedent (momwire#932), including keeping the
private spelling working so nothing in this tree or in a consumer moves.
Costs nothing at import: both modules were already loaded by `import
momwire` before the export existed (measured — `razor` and `bspline` pull
them in), so this adds a name, not a module.
"""

import numpy as np

import momwire
from momwire import _ground_spec, _medium_spec

# One above member, one below, meeting in the plane: the crossing junction
# the exemption exists for.
_ABOVE = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
_BELOW = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.15]])
_BELOW_2 = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, -0.15]])
_JOINED = [[(0, "start"), (1, "start")]]


def test_the_public_names_are_the_private_objects():
    """The load-bearing test of this unit: same objects, not same answers.

    Equality of answers would pass for a second implementation that happens
    to agree today, which is precisely how the two copies momwire#848 merged
    got away with disagreeing for as long as they did.
    """
    assert momwire.ground_touch_tol is _ground_spec.ground_touch_tol
    assert (
        momwire.grounded_crossing_exemption is _medium_spec.grounded_crossing_exemption
    )
    assert "ground_touch_tol" in momwire.__all__
    assert "grounded_crossing_exemption" in momwire.__all__


def test_the_private_spellings_still_resolve():
    """Nothing moves. Both trunks call these by their private names and
    antennaknobs pins momwire exactly, so the private path has to keep
    working until a pin carries the public one (momwire#855's sequencing).
    Delete this with the private names, not before."""
    from momwire._ground_spec import ground_touch_tol
    from momwire._medium_spec import grounded_crossing_exemption

    assert ground_touch_tol is momwire.ground_touch_tol
    assert grounded_crossing_exemption is momwire.grounded_crossing_exemption


def test_the_tolerance_is_relative_to_the_wire():
    """Why a consumer cannot just hard-code a number: the answer scales with
    each wire's own length, so one absolute tolerance is wrong for some wire
    on any deck with a range of lengths."""
    short = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])
    assert momwire.ground_touch_tol(_ABOVE) == 100.0 * momwire.ground_touch_tol(short)
    assert momwire.ground_touch_tol(_ABOVE) < 1e-4  # far tighter than a 1 mm stand-off


def test_the_exemption_answers_from_geometry_alone():
    """Bare polylines, no solver and no labels — which is what lets a
    consumer ask it at deck-construction time, before any solve exists.

    Both of the necessary conditions are exercised, because a consumer that
    got either wrong would refuse the wrong decks: a junction with no member
    reaching above the plane has no above side to cross TO (momwire#700's
    wholly-below finding), and a lone grounded end cannot join two media
    (momwire#698).
    """
    granted = momwire.grounded_crossing_exemption([_ABOVE, _BELOW], 0.0, _JOINED)
    assert granted == frozenset({(0, "start"), (1, "start")})

    wholly_below = momwire.grounded_crossing_exemption([_BELOW, _BELOW_2], 0.0, _JOINED)
    assert wholly_below == frozenset()

    lone = momwire.grounded_crossing_exemption([_ABOVE, _BELOW], 0.0, [[(0, "start")]])
    assert lone == frozenset()
