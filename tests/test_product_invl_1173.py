"""The product route on a second buried family: the inverted-L over buried
radials (momwire#1173 design B).

Every other deck `test_product_grid_1173` gates has an above side that is one
vertical line or a leaning line. The inverted-L's is a mast plus a horizontal
top wire, whose nodes each have their own (x, y): the above side is one mast
group plus one group per top-wire node, so the product is a MULTI-group merge
there or not taken at all. `invl_deck(lean=True)` also moves the mast top
off the axis, so no two above nodes share an (x, y) and a product can only
group the below side (the z′ slot).

The gate is the one `test_product_grid_1173` states: Z of the real fill
with design B's switches on, against the same fill with them off (the grid
dedup, the lookup path and the two exact walks), at `np.array_equal`, with
the route counters proving which route ran. By default these decks take the
multi-group merge in both cross blocks (the two caps that used to send them
to the grid route are gone: they guarded no cost the grid route does not
pay, and lifted the inverted-L x8 peaked at 477 MB instead of 822 MB on
Skylake, Z to the bit). Setting the old 0.25 caps still sends them to the
grid route, which is that FALLBACK's gate; a negative control on the merge
shows the comparison can fail.

Measured 2026-09-25 by script, Z to the bit against the switches-off fill
and against origin/main: invl_deck(16) x2 / x4 / x8 and the leaning variant
x2 / x4 (bench decks, Skylake).
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from momwire import _near_interface as ni

from test_crossing_serve_524 import invl_deck
from test_product_grid_1173 import _fill, _hub, _razor


# Every gate here runs at the shipped cost rule and with every plane taken
# (conftest's `sheet_modes`, momwire#1173 Design E phase 2).
pytestmark = pytest.mark.usefixtures("sheet_modes")


# The caps as they were before they were dropped: the grid route's gate.
_CAPPED = {"cf._PRODUCT_MAX_CAND_FRAC": 0.25, "cf._PRODUCT_MAX_GROUP_FRAC": 0.25}


def _make(lean):
    # One radial suffices for the leaning deck: nothing above is shared, so
    # both blocks decline on groups whatever the radial count, and its grid
    # route (every triple distinct) is the costly reference here.
    deck = invl_deck(n_radials=1 if lean else 2, lean=lean)
    return lambda: _razor(deck)


def _reference(lean):
    """The switches-off Z, shared by this module's tests (read-only), for the
    plane-sheet settings in force (`sheet_modes` runs each test under two)."""
    return _reference_at(
        lean,
        ni._SHEET,
        ni._SHEET_BUILD_WEIGHT,
        ni._SHEET_HEIGHT_WEIGHT,
        ni._SHEET_MIN_ROWS,
    )


@functools.cache
def _reference_at(lean, *_sheet_settings):
    Z, r = _fill(_make(lean), off=True)
    assert r["cf.main_product"] == 0 and r["cf.ends_fast"] == 0, r
    Z.setflags(write=False)
    return Z


def test_hub_x1_takes_the_product_to_the_bit():
    """The headline deck at x1 on the PR lane (x2 / x4 ride the push lane in
    `test_product_grid_1173`): product on both blocks, fast ends, both
    bounds."""
    make = lambda: _razor(_hub(1))  # noqa: E731
    ref, r_off = _fill(make, off=True)
    got, r = _fill(make)
    assert r_off["cf.main_product"] == 0 and r_off["cf.ends_fast"] == 0
    assert r["cf.main_product"] == 2 and r["cf.main_generic"] == 0, r
    assert r["cf.ends_fast"] > 300, r
    assert r["rz.bound"] >= 1 and r["pg.bracket"] >= 1, r
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("lean", [False, True], ids=["invl", "lean"])
def test_the_inverted_l_falls_back_to_the_bit(lean):
    """Capped as before, neither cross block takes the product: the reason
    each declined is counted, and Z is the grid route's to the bit."""
    ref = _reference(lean)
    got, r = _fill(_make(lean), **_CAPPED)
    assert r["cf.main_product"] == 0 and r["cf.ends_fast"] == 0, r
    declined = r["cf.main_generic_groups"] + r["cf.main_generic_candidates"]
    assert declined == r["cf.main_generic"] == 2, r
    if lean:
        # No above group is shared at all: both blocks decline on groups.
        assert r["cf.main_generic_groups"] == 2, r
    assert r["rz.bound"] >= 1 and r["pg.bracket"] >= 1, r
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("lean", [False, True], ids=["invl", "lean"])
def test_the_inverted_l_merges_many_groups_to_the_bit(lean):
    """By default both blocks take the product over one group per top-wire
    node (and, leaning, one block groups the below side instead): the
    multi-group merge and its grouped fast ends give the grid route's Z."""
    ref = _reference(lean)
    got, r = _fill(_make(lean))
    assert r["cf.main_product"] == 2 and r["cf.main_generic"] == 0, r
    assert r["cf.main_product_groups"] > 8, r
    assert r["cf.ends_fast"] > 0, r
    if lean:
        assert r["cf.main_product_zp"] == 1, r
    assert np.array_equal(got, ref)


@pytest.mark.parametrize("control", ["split", "row"])
def test_the_inverted_l_merge_can_fail(control):
    """The gate above is not blind on this family: evaluating the product's
    rows in two calls ("split": columns lose members, so their rule witness
    s_min moves) or serving every fast end one row off ("row") moves Z."""
    ref = _reference(False)
    got, r = _fill(_make(False), **{"cf._PRODUCT_NEG_CONTROL": control})
    assert r["cf.main_product"] == 2, r
    assert np.count_nonzero(got != ref) > 0, f"{control}: the gate is blind to it"
