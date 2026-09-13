"""antennaknobs#1464: `BSplineSolver.buried_serve_refusal` is the fill's own verdict.

The pre-flight runs `_below_interface.plan_buried`, the sequence the buried
fill itself runs before it fills a grid, so its answer and the solve's raise
cannot part. Each refusing case here is cheap because the fill raises at the
plan, before any grid.
"""

import pytest

from momwire import _sommerfeld_below
from momwire.bspline import BSplineSolver

from test_crossing_serve_524 import crossing_deck, fan_rise_deck


def _raised(build):
    with pytest.raises((ValueError, NotImplementedError)) as exc:
        BSplineSolver(**build).compute_impedance()
    return str(exc.value)


def test_the_served_fan_has_no_refusal():
    assert BSplineSolver(**fan_rise_deck()).buried_serve_refusal() is None


def test_a_free_space_deck_has_no_refusal():
    build = crossing_deck()
    for key in ("ground_eps", "ground_model"):
        build.pop(key)
    build["ground_z"] = None
    assert BSplineSolver(**build).buried_serve_refusal() is None


def test_the_grazing_refusal_is_the_fills_own_sentence(monkeypatch):
    """Raise the floor past every pair on the fan, so the plan refuses. The
    pre-flight must return exactly what the solve raises."""
    monkeypatch.setattr(_sommerfeld_below, "_SOMM_BELOW_TH_MIN_DEG", 60.0)
    build = fan_rise_deck()
    got = BSplineSolver(**build).buried_serve_refusal()
    assert got is not None and "grazing floor" in got
    assert got == _raised(build)


def test_a_solver_configuration_refusal_is_the_fills_own_sentence():
    build = fan_rise_deck(use_singular_enrichment=True)
    got = BSplineSolver(**build).buried_serve_refusal()
    assert got is not None
    assert got == _raised(build)
