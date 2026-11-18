"""Driven by test_xdist_warning_import_race.py, never collected by the suite
(the name does not match ``test_*.py``): each test raises one momwire
advisory from a different module, so the xdist controller's first decodes
import momwire from several receiver threads at once."""

import importlib
import warnings

import pytest

CATEGORIES = [
    ("momwire._wire_spec", "ShortSegments"),
    ("momwire._near_surface", "NearSurfaceWire"),
    ("momwire._crossing_fill", "CoarseCrossingNode"),
    ("momwire._feed_snap", "AmbiguousSite"),
    ("momwire.array_block", "ArrayBlockNoRepeats"),
    ("momwire.hmatrix", "HMatrixFragmented"),
    ("momwire.sinusoidal", "RadiusStepJunction"),
    ("momwire._surface_height", "SurfaceRadialHeight"),
]


@pytest.mark.parametrize(("module", "name"), CATEGORIES)
def test_raise_one_advisory(module, name):
    category = getattr(importlib.import_module(module), name)
    warnings.warn(category(f"probe from {module}"), stacklevel=1)
