"""`_wire_media` / `_grounded_junction_ends` / `_crossing_junctions` on
RazorSolver — momwire#813, under `BSplineSolver`'s names.

Which side of the interface each wire is on, which junction ends sit in the
plane, and which junctions cross it are questions about the DECK, not about a
formulation.  Both trunks now answer them through the same three method names
and the same `_medium_spec` call, so a consumer asks once — antennaknobs#1103's
buried catalog gate is the first, and this module's own crossing assembly
(momwire#813's remaining steps) is the second.

The one difference from the B-spline twins is where the junction groups come
from: razor has no `junctions=` spec to read, so the scan runs over the
groups it DETECTS (`_find_junctions`).  Which is what these gates check —
that the two trunks agree on decks where one reads a spec and the other reads
the geometry.

**A behaviour change rides here, and it is the point.**  Razor used to pass
no `crossing_ends` to `_medium_spec.wire_media`, on that function's own rule
that "a caller with no crossing basis (razor) passes nothing and keeps the
refusals verbatim".  What that cost was a refusal naming a workaround the
reader had already applied: a deck SPLIT at the interface into a below wire
and an above wire with the junction declared — which is precisely the shape
the crossing refusal instructs you to build — was answered with the crossing
refusal, because without the exemption the below wire's plane-touching anchor
reads as a mid-span crossing.  The deck still refuses.  It refuses for the
reason that is true now, and `test_a_split_crossing_deck_is_not_told_to_split`
is what holds that.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from momwire import _medium_spec
from momwire import razor as _razor
from momwire.bspline import BSplineSolver
from momwire.razor import RazorSolver

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

from test_crossing_serve_524 import crossing_deck, fan_rise_deck  # noqa: E402


def _razor_reading(deck, monkeypatch):
    """The three labels off a RazorSolver built on a deck it REFUSES.

    The refusal is the point of momwire#813 — razor cannot fill a mixed
    above/below deck until this arc's assembly lands — so the labels are
    reached by stubbing the constructor's buried gate, the way
    `tests/test_razor_below_plane_812.py` reaches unit 1's fill through
    `_SERVE_BELOW_PLANE`.  Nothing here solves; these are geometry reads.
    """
    monkeypatch.setattr(RazorSolver, "_refuse_buried_geometry", lambda self: None)
    rd = {k: v for k, v in deck.items() if k != "junctions"}
    rs = RazorSolver(**rd, nec5_quadrature=True)
    return rs._wire_media(), rs._grounded_junction_ends(), rs._crossing_junctions()


DECKS = {
    "crossing": lambda: crossing_deck(1),
    "fan (4 radials)": fan_rise_deck,
}


@pytest.mark.parametrize("name", sorted(DECKS))
def test_the_three_labels_are_bsplines_own(name, monkeypatch):
    """Razor detects its junctions where bspline is handed a spec, and the
    two still agree cell for cell."""
    deck = DECKS[name]()
    b = BSplineSolver(**deck)
    want = (b._wire_media(), b._grounded_junction_ends(), b._crossing_junctions())
    got = _razor_reading(deck, monkeypatch)
    assert got[0] == want[0], f"media: {got[0]} != {want[0]}"
    assert got[1] == want[1], f"grounded junction ends: {got[1]} != {want[1]}"
    assert got[2] == want[2], f"crossing junctions: {got[2]} != {want[2]}"


def test_the_crossing_deck_actually_exercises_all_three(monkeypatch):
    """The gate above would pass vacuously on a deck with nothing to label."""
    media, ends, crossing = _razor_reading(crossing_deck(1), monkeypatch)
    assert _medium_spec.BELOW in media and _medium_spec.ABOVE in media
    assert ends, "the split deck's junction sits in the plane"
    assert crossing == (0,)


def test_free_space_labels_every_wire_above():
    """`ground_z is None` invents no interface, and the junction scan is
    empty rather than absent."""
    lam = 299792458.0 / 7.0e6
    rs = RazorSolver(
        wires=[np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])],
        n_per_edge_per_wire=[[12]],
        wire_radius=0.005,
        wavelength=lam,
        feeds=[(0, 1.0, 1 + 0j)],
    )
    assert set(rs._wire_media()) == {_medium_spec.ABOVE}
    assert rs._grounded_junction_ends() == frozenset()
    assert rs._crossing_junctions() == ()


def test_wire_media_is_computed_once():
    lam = 299792458.0 / 7.0e6
    rs = RazorSolver(
        wires=[np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]])],
        n_per_edge_per_wire=[[12]],
        wire_radius=0.005,
        wavelength=lam,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
        feeds=[(0, 0.0, 1 + 0j)],
    )
    assert rs._wire_media() is rs._wire_media()


def test_a_split_crossing_deck_is_not_told_to_split():
    """The refusal a crossing deck gets must not instruct the reader to build
    the deck they already built.

    This is momwire#604's class 2 — a refusal naming a workaround is making a
    claim about a code path it does not test — caught in razor's constructor
    by momwire#813's own reading.
    """
    rd = {k: v for k, v in crossing_deck(1).items() if k != "junctions"}
    with pytest.raises(ValueError) as exc:
        RazorSolver(**rd, nec5_quadrature=True)
    msg = str(exc.value)
    assert "split the wire AT the interface" not in msg
    assert "crosses the ground interface mid-span" not in msg
    # ... and it names the arc that will serve it.
    assert "momwire#813" in msg


def test_a_wholly_below_deck_has_no_crossing_junction(monkeypatch):
    monkeypatch.setattr(_razor, "_SERVE_BELOW_PLANE", True)
    lam = 299792458.0 / 7.0e6
    rs = RazorSolver(
        wires=[np.array([[-2.0, 0.0, -1.0], [2.0, 0.0, -1.0]])],
        n_per_edge_per_wire=[[12]],
        wire_radius=0.005,
        wavelength=lam,
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
        feeds=[(0, 2.0, 1 + 0j)],
    )
    assert set(rs._wire_media()) == {_medium_spec.BELOW}
    assert rs._crossing_junctions() == ()
