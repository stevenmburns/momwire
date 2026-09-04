"""Every row's declared `feed_model` matches what the class actually does.

FOUND DOWNSTREAM, WHICH IS THE POINT. antennaknobs#1006 G2-7 renders a
composition line per solver tab from these rows, and it read "segment gap" on
the B-spline family — while their constructors have always defaulted to
`feed_model="point"` and accepted either. The row described a source the class
does not use. Nothing here noticed, because nothing here compared the row to
the constructor.

A capability row is a PROMISE ABOUT THE CLASS. The `axes` cells say what a
solver can be configured to, and a consumer reading one to answer "what does
this do by default" is doing the obvious thing — so the row now leads with the
default, and this file makes both halves true by construction:

  * every declared value is actually constructible;
  * the FIRST declared value is what the constructor picks when nothing is
    passed.

CONSTRUCTED, NEVER INTROSPECTED. `inspect.signature` reports no `feed_model`
at all for `HMatrixSolver`, `ArrayBlockSolver`, `RazorSolver` and
`HarringtonSolver` — they take it through `**kwargs` or not at all — so a
signature-based check would have called four of the nine rows unanswerable and
missed the bug on two of them.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire.deck._solver import BASES

WL = 42.83

# `feed_model=` is a string kwarg; the axis vocabulary spells the same thing
# with a "-gap" suffix. One mapping, here, so the comparison below is between
# the row and the RUNTIME rather than between two spellings.
AXIS_VALUE = {"point": "point-gap", "segment": "segment-gap"}


def _deck():
    return dict(
        wires=[np.array([(0.0, 0.0, 5.0), (0.0, 0.0, -5.0)])],
        n_per_edge_per_wire=[[9]],
        feeds=[(0, 5.0, 1 + 0j)],
        wavelength=WL,
        wire_radius=1e-3,
    )


def _build(cls, bound, **extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls(**_deck(), **dict(bound or {}), **extra)


def _effective_feed_model(solver):
    """What the built solver is actually using, or None if it has no such
    notion. None is a real answer — `RazorSolver` and `HarringtonSolver` do
    not take the keyword, and their rows describe their feed differently."""
    for attr in ("feed_model", "_feed_model"):
        if hasattr(solver, attr):
            return getattr(solver, attr)
    return None


NAMES = sorted(BASES)


@pytest.mark.parametrize("name", NAMES)
def test_the_first_declared_value_is_the_constructor_default(name):
    """The invariant the composition line depends on, and the one that would
    have caught this bug on the day it was written."""
    cls, bound = BASES[name]
    declared = tuple(cls.capabilities.axes.get("feed_model", ()))
    assert declared, f"{name} declares no feed_model"
    got = _effective_feed_model(_build(cls, bound))
    if got is None:
        # The class does not take the keyword. Its row still describes its
        # feed (`node-port`, `segment-gap`), and there is no default to
        # compare — asserted rather than skipped, so "no attribute" cannot
        # quietly become the answer for a class that DOES take it.
        assert "gap" in declared[0] or declared[0] == "node-port", declared
        return
    assert AXIS_VALUE[got] == declared[0], (
        f"{name}: constructor defaults to {got!r} ({AXIS_VALUE[got]}) but the "
        f"row leads with {declared[0]!r} — a consumer reading this row for "
        "the default gets the wrong answer"
    )


@pytest.mark.parametrize("name", NAMES)
def test_every_declared_value_is_actually_constructible(name):
    """The other half. A row may not offer a value the class rejects."""
    cls, bound = BASES[name]
    declared = tuple(cls.capabilities.axes.get("feed_model", ()))
    if _effective_feed_model(_build(cls, bound)) is None:
        return  # no keyword to pass
    for value in declared:
        kwarg = {v: k for k, v in AXIS_VALUE.items()}[value]
        got = _effective_feed_model(_build(cls, bound, feed_model=kwarg))
        assert AXIS_VALUE[got] == value, f"{name}: asked {kwarg!r}, got {got!r}"


def test_the_bspline_family_declares_both_and_leads_with_point():
    """The specific regression, pinned by name.

    These four rows said ("segment-gap",) while defaulting to the point gap —
    so a panel reading the row described the wrong source on every stock
    solve. `bspline-d1` is the same class under a bound kwarg, and the two
    accelerators inherit via `_replace`, which is why one wrong literal
    reached four rows.
    """
    for name in ("bspline", "bspline-d1", "hmatrix", "arrayblock"):
        assert BASES[name][0].capabilities.axes["feed_model"] == (
            "point-gap",
            "segment-gap",
        ), name


def test_the_point_matched_solver_still_offers_only_the_segment_gap():
    """The negative case, so "declare both everywhere" cannot satisfy the
    tests above. `SinusoidalSolver` refuses the point gap — a zero-width gap
    has no collocation RHS (momwire#212) — and that refusal is the first row
    in the COUPLINGS table."""
    assert BASES["sinusoidal"][0].capabilities.axes["feed_model"] == ("segment-gap",)
    with pytest.raises(NotImplementedError):
        _build(*BASES["sinusoidal"], feed_model="point")
