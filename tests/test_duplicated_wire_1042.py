"""A deck that lists the same wire twice solves it ONCE, and says so —
momwire#1042.

qantenna/airplane.nec carries GW 116 and GW 117 with identical endpoints in
opposite directions. The two copies have no answer of their own: every basis
on one has an identical twin on the other, so the operator is singular and
the split of current between them is undetermined. NEC-5 reports `Singular
matrix`; nec2c answers 68.5-91.06j, and 68.457-91.179j with GW 117 deleted —
the conductor present once, which is what momwire now solves, warning
`DuplicateWire`.

The copy is DROPPED before anything is built, and every wire-indexed input is
remapped; the per-wire readouts answer in the caller's numbering, with the
copy at zero current. Copies that differ in anything the physics reads, or
that carry a site, are refused by name instead.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from momwire._wire_spec import (  # noqa: E402
    DuplicateWire,
    drop_duplicated_wires,
    find_duplicated_wires,
)
from momwire.bspline import BSplineSolver  # noqa: E402
from momwire.harrington import HarringtonSolver  # noqa: E402
from momwire.razor import RazorSolver  # noqa: E402
from momwire.sinusoidal import SinusoidalSolver  # noqa: E402
from momwire.sinusoidal_galerkin import SinusoidalGalerkinSolver  # noqa: E402
from test_crossing_serve_524 import fan_rise_deck  # noqa: E402

FAMILIES = [
    HarringtonSolver,
    SinusoidalSolver,
    SinusoidalGalerkinSolver,
    RazorSolver,
    BSplineSolver,
]
IDS = [c.__name__ for c in FAMILIES]

# airplane.nec's GW 115-119, a corner of the wire grid with GW 116 and
# GW 117 the same span run both ways (wires 2 and 3 here). Free space, as the
# deck is. The feed sits on wire 5, AFTER the copy, so its index must be
# remapped for the answer to be right.
A = (5.42580, -0.844875, 2.80628)
B = (5.41486, -3.08670, 2.80627)
C = (6.32782, -3.08619, 2.80627)
D = (6.28484, -4.66354, 2.80627)
E = (6.30227, -0.883203, 2.80563)
WIRES = [(E, C), (A, B), (C, B), (B, C), (D, B), (C, D)]


def _corner(dup=True, **kw):
    wires = list(WIRES)
    feed_wire = 5
    if not dup:
        del wires[3]
        feed_wire = 4
    deck = dict(
        wires=[np.array(w) for w in wires],
        n_per_edge_per_wire=[[3] for _ in wires],
        wavelength=299792458.0 / 7.5e6,
        wire_radius=0.01,
        feeds=[(feed_wire, None, 1 + 0j)],
    )
    deck.update(kw)
    return deck


def _build(cls, **deck):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = cls(**deck)
    return s, [w for w in caught if issubclass(w.category, DuplicateWire)]


@pytest.mark.parametrize("cls", FAMILIES, ids=IDS)
def test_the_copy_is_solved_once_and_said_so(cls):
    s, found = _build(cls, **_corner())
    assert len(found) == 1
    msg = str(found[0].message)
    assert "1 wire(s) listed twice" in msg and "wires 2 and 3." in msg
    assert "solved ONCE" in msg and "momwire#1042" in msg
    ref, none = _build(cls, **_corner(dup=False))
    assert none == []
    # The solve IS the deck without the copy, input for input.
    z, coeffs = s.compute_impedance()
    z_ref, coeffs_ref = ref.compute_impedance()
    assert z == z_ref
    assert np.array_equal(coeffs, coeffs_ref)


@pytest.mark.parametrize("cls", FAMILIES, ids=IDS)
def test_the_readouts_speak_the_callers_numbering(cls):
    s, _ = _build(cls, **_corner())
    ref, _ = _build(cls, **_corner(dup=False))
    _z, coeffs = s.compute_impedance()
    got = s.currents_at_knots(coeffs)
    want = ref.currents_at_knots(coeffs)
    assert len(got) == len(WIRES)
    for w_in, w_ref in ((0, 0), (1, 1), (2, 2), (4, 3), (5, 4)):
        assert np.array_equal(got[w_in], want[w_ref])
    assert len(got[3]) == len(got[2]) and not np.any(got[3])
    # Walked over the caller's six wires, the copy contributing nothing.
    mid, moment, nodes, delta = s.element_currents(coeffs)
    rmid, rmoment, _n, _d = ref.element_currents(coeffs)
    assert mid.shape[0] == rmid.shape[0] + 3
    assert np.allclose(moment.sum(axis=0), rmoment.sum(axis=0), rtol=1e-13, atol=0)
    placed = s.feed_placements()
    assert [p.wire for p in placed] == [5]


@pytest.mark.parametrize("cls", [SinusoidalSolver, RazorSolver, BSplineSolver])
def test_the_copy_dissipates_nothing(cls):
    s, _ = _build(cls, **_corner(wire_conductivity=5.8e7))
    _z, coeffs = s.compute_impedance()
    total, per_wire = s.wire_loss_power(coeffs)
    assert per_wire.shape == (len(WIRES),)
    assert per_wire[3] == 0.0 and per_wire[2] > 0.0
    assert total == pytest.approx(per_wire.sum())


@pytest.mark.parametrize(
    "change, why",
    [
        (dict(wire_radius=[0.01, 0.01, 0.01, 0.02, 0.01, 0.01]), "wire_radius"),
        (dict(n_per_edge_per_wire=[[3], [3], [3], [4], [3], [3]]), "segment counts"),
        (dict(feeds=[(3, None, 1 + 0j)]), "places a site on wire 3"),
        (dict(feeds=[(2, None, 1 + 0j)]), "places a site on wire 2"),
        (
            dict(wire_conductivity=[5.8e7, 5.8e7, 5.8e7, 3.5e7, 5.8e7, 5.8e7]),
            "wire_conductivity",
        ),
    ],
    ids=["radius", "segments", "feed-on-copy", "feed-on-twin", "conductivity"],
)
def test_copies_that_differ_are_refused_by_name(change, why):
    with pytest.raises(ValueError) as exc:
        _build(BSplineSolver, **_corner(**change))
    msg = str(exc.value)
    assert "wires 2 and 3 are the same wire listed twice" in msg
    assert why in msg and "cannot be merged" in msg


def test_a_lumped_load_on_the_copy_is_refused():
    with pytest.raises(ValueError, match="lumped_loads places a site on wire 3"):
        _build(RazorSolver, **_corner(lumped_loads=[(3, None, 50.0)]))


def test_an_explicit_junction_is_remapped_or_kept_as_written():
    """The copy (3) runs B -> C, its twin (2) C -> B, so the copy's START
    stands beside the twin's END. Naming both drops the copy's member; naming
    the copy alone joins it to different wires than its twin, so it is not a
    repeat and stays as written (momwire#1333; #1042 refused it)."""
    groups = [[(0, "end"), (2, "start"), (3, "end"), (5, "start")]]
    *_rest, junctions, dedup = drop_duplicated_wires(
        "X", [np.array(w) for w in WIRES], None, junctions=groups
    )
    assert junctions == [[(0, "end"), (2, "start"), (4, "start")]]
    assert dedup.removed == ((3, 2, True),)
    alone = [[(0, "end"), (3, "end"), (5, "start")]]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DuplicateWire)
        wires, *_rest, junctions, dedup = drop_duplicated_wires(
            "X", [np.array(w) for w in WIRES], None, junctions=alone
        )
    assert dedup is None and junctions == alone and len(wires) == len(WIRES)


def test_many_copies_raise_one_advisory():
    """A GM-built grid repeats hundreds of wires: one sentence, not hundreds."""
    wires = [np.array(w) for w in WIRES] * 4
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = drop_duplicated_wires("X", wires, None)
    found = [w for w in caught if issubclass(w.category, DuplicateWire)]
    assert len(found) == 1
    assert len(out[0]) == 5  # wires 2 and 3 are one conductor
    assert "19 wire(s) listed twice" in str(found[0].message)
    assert "and 13 more pairs" in str(found[0].message)


def test_close_is_not_coincident():
    """Equal to rounding, not close: 1 um apart, and 0.6 nm apart (inside a
    nanometre rounding bin, and Harrington's near-coincident window, which
    must still refuse it), are two conductors here."""
    for shift in (1e-6, 6e-10):
        moved = np.array([A, B]) + np.array([shift, 0.0, 0.0])
        assert find_duplicated_wires([np.array([A, B]), moved]) == []
    assert find_duplicated_wires([np.array([A, B, C]), np.array([A, B, D])]) == []


@pytest.mark.parametrize("cls", [BSplineSolver, RazorSolver], ids=lambda c: c.__name__)
def test_the_bundle_is_untouched(cls):
    """momwire#524's fan: four radials whose RISES coincide. Whole wires
    differ, so nothing is dropped (razor refuses it at the solve, by its own
    bundle sentence, #846)."""
    deck = {
        k: v
        for k, v in fan_rise_deck().items()
        if k not in ("ground_z", "ground_eps", "ground_model")
    }
    s, found = _build(cls, **deck)
    assert found == [] and s._wire_dedup is None
