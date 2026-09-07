"""momwire#884 and #876 — the capability axes and the coated-wire pair, promoted.

Two consumers' questions that had no public answer, so antennaknobs reached
through three private modules to ask them:

  * "what do these solvers span" — `_capabilities.axes_for`, which is the
    SINGLE derivation point for the derived axes (`ground_model` from
    `grounds`, `wire_position` from `buried`/`contact`). A consumer
    re-deriving those on its own side is exactly the drift `_capabilities`
    refuses to allow, so calling it was right and reaching privately to do so
    was not.
  * "does this build model a coated wire as the equivalent-radius pair" —
    `_wire_loading.equivalent_radius` and `_surface_height.SURFACE_HEIGHT_CLASS`.

A VERSION CHECK CANNOT REPLACE EITHER PROBE, and that is structural rather
than incidental. momwire's submodule pointer runs ahead of its PyPI release by
convention — that convention is what lets a consumer's pointer bump land
without touching its version triple — so a build WITH these names and a build
WITHOUT them declare the SAME version. Measured on the coated-wire pair: the
older build does not refuse the surface-radial deck, it answers it, 221.7 -
144.3j against 159.2 - 35.3j, 62 ohm apart in R and 109 in X, in silence.
Promoting the names does not remove the need to probe; it removes the need to
probe a private module.

RE-EXPORTS, NEVER REIMPLEMENTATIONS, on #855's and #932's precedent: the first
test pins that these are the same OBJECTS, because a later edit that gave a
public name its own body would restore the drift the exports exist to prevent.
Equality of answers would pass for a second implementation that happens to
agree today, which is how the two copies momwire#848 merged got away with
disagreeing for as long as they did.

`SURFACE_HEIGHT_CLASS` is exported as the object rather than behind a
`models_coated_wire()` predicate, for three reasons: identity is the point
here and a predicate is a new thing to keep true; the consumer probes for it
and `equivalent_radius` separately so it can refuse by naming which half is
absent; and the tuple carries the numbers (`floor_h_over_a`,
`advisory_h_over_a`) a consumer's own advisory quotes.
"""

import momwire
from momwire import _capabilities, _surface_height, _wire_loading

# The five names this unit promotes, paired with the module each came from.
_PROMOTED = [
    ("axes_for", _capabilities),
    ("AXIS_VALUES", _capabilities),
    ("DERIVED_AXES", _capabilities),
    ("equivalent_radius", _wire_loading),
    ("SURFACE_HEIGHT_CLASS", _surface_height),
]


def test_the_public_names_are_the_private_objects():
    """The load-bearing test of this unit: same objects, not same answers."""
    for name, mod in _PROMOTED:
        assert getattr(momwire, name) is getattr(mod, name), name


def test_every_promoted_name_is_in_dunder_all():
    for name, _mod in _PROMOTED:
        assert name in momwire.__all__, name


def test_the_private_spellings_still_resolve():
    """Nothing moves. This tree calls these by their private names and a
    consumer pins momwire exactly, so the private path keeps working until a
    pin carries the public one."""
    from momwire._capabilities import AXIS_VALUES, DERIVED_AXES, axes_for
    from momwire._surface_height import SURFACE_HEIGHT_CLASS
    from momwire._wire_loading import equivalent_radius

    assert axes_for is momwire.axes_for
    assert AXIS_VALUES is momwire.AXIS_VALUES
    assert DERIVED_AXES is momwire.DERIVED_AXES
    assert equivalent_radius is momwire.equivalent_radius
    assert SURFACE_HEIGHT_CLASS is momwire.SURFACE_HEIGHT_CLASS


def test_a_getattr_probe_answers_on_the_top_level_module():
    """The shape the consumer actually uses. A build without these names has to
    fail this probe rather than raise ImportError from a private module, which
    is the whole point of promoting them."""
    for name, _mod in _PROMOTED:
        assert getattr(momwire, name, None) is not None, name


def test_the_three_capability_names_describe_one_thing():
    """The reason #884 promotes all three rather than only the one antennaknobs
    calls today: they are the declared vocabulary, the computed pair, and the
    function that returns their union. A consumer given only `axes_for` and
    rendering a panel over the axes reaches straight back into the private
    module for the other two.

    `DERIVED_AXES` is deliberately NOT a subset of `AXIS_VALUES` -- the derived
    pair is computed from `grounds` / `buried` / `contact` rather than declared
    on a row, and restating it as a row would be the second source of truth
    `_capabilities` exists to prevent.
    """
    from momwire import BSplineSolver

    got = momwire.axes_for(BSplineSolver.capabilities)
    assert set(momwire.AXIS_VALUES) <= set(got), "declared axes missing"
    assert set(momwire.DERIVED_AXES) <= set(got), "derived axes missing"
    assert set(momwire.DERIVED_AXES).isdisjoint(momwire.AXIS_VALUES)


def test_the_surface_height_class_carries_the_floor_a_consumer_quotes():
    """The reason this is the object and not a boolean: a consumer refusing a
    low-stand-off deck states the floor it is refusing against."""
    cls = momwire.SURFACE_HEIGHT_CLASS
    assert cls.floor_h_over_a > 0
    assert cls.advisory_h_over_a > cls.floor_h_over_a
