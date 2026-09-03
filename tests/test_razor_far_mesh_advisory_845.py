"""momwire#845: razor-2p states its far-mesh accuracy class at construction.

The advisory is UNCONDITIONAL, which makes it easy to gate badly. A test that
merely asserts "a warning came out" passes just as happily if the text is
wrong, if it fires on the wrong lane, or if the numbers drift away from the
sweep that produced them — the failure mode this repo keeps rediscovering
(momwire#852's two green gates that measured nothing). So each gate here is
paired with the thing it guards:

  * that it fires — and that the OTHER lanes stay silent, which is what makes
    "it fires" mean something rather than being true of every construction;
  * that the text is pinned to `FAR_MESH_CLASS`, checked by MUTATING the row
    and requiring the message to follow, so replacing the composed message
    with a hard-coded sentence fails here;
  * that it is advisory in fact and not just in wording — same impedance, same
    mesh, warning or no warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from momwire import BSplineSolver, RazorSolver
from momwire._razor_class import (
    FAR_MESH_CLASS,
    FarMeshClass,
    RazorFarMeshClass,
    far_mesh_class_message,
    warn_far_mesh_class,
)

WL = 42.83  # metres, ~7 MHz


def _deck(n=21):
    """A 20 m centre-fed free-space dipole — #845's own free-space row, and
    deliberately a deck with no crossing junction and no ground, so nothing
    else in the stack has a reason to warn and a stray warning here is this
    advisory or a real regression."""
    return {
        "wires": [np.array([[0.0, 0.0, -10.0], [0.0, 0.0, 10.0]])],
        "n_per_edge_per_wire": [n],
        "wire_radius": 1.0e-3,
        "wavelength": WL,
        "feed_arclength": 10.0,
    }


def _warnings_from(**kwargs):
    """Every warning raised by ONE construction, as a list of instances."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        RazorSolver(**_deck(), **kwargs)
    return [w.message for w in rec]


def test_razor_2p_states_its_class_once():
    """The razor-2p lane warns exactly once per construction."""
    got = [
        m
        for m in _warnings_from(nec5_quadrature=True)
        if isinstance(m, RazorFarMeshClass)
    ]
    assert len(got) == 1, f"want exactly one advisory, got {len(got)}"


def test_the_other_lanes_stay_silent():
    """What makes the gate above mean anything.

    If the advisory fired on every solver, "razor-2p warns" would be true and
    uninformative. bspline is the converged engine the advisory POINTS AT, so
    it warning here would be self-contradicting; the Gauss-Legendre razor lane
    is deliberately silent (see the scoping comment in `RazorSolver.__init__`).
    """
    assert not [m for m in _warnings_from() if isinstance(m, RazorFarMeshClass)]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        BSplineSolver(**_deck(), degree=2)
    assert not [w for w in rec if isinstance(w.message, RazorFarMeshClass)]


def test_the_message_is_pinned_to_the_measured_row_not_to_a_literal():
    """Mutate the row; the message must follow.

    This is the gate that would catch someone "simplifying" the composed
    message into a hard-coded sentence, at which point a re-sweep would update
    `FAR_MESH_CLASS` and the text would go on quoting last month's numbers.
    A test asserting a fixed substring cannot catch that -- it is exactly as
    green either way.
    """
    moved = FarMeshClass(
        median_rel_pct=9.9,
        tail_rel_pct=88.0,
        median_abs_ohm=7.77,
        order=1,
        ground_spread_pct=5.0,
        n_decks=123,
        issue="stevenmburns/momwire#845",
        provenance="test row",
    )
    text = far_mesh_class_message(moved)
    for token in ("9.9 %", "88 %", "7.77 ohm", "123 antennaknobs catalog decks", "5 %"):
        assert token in text, f"{token!r} missing -- message is not built from the row"

    # ...and the shipped row's own figures really are the ones a caller sees.
    shipped = far_mesh_class_message()
    assert f"{FAR_MESH_CLASS.median_rel_pct:.1f} %" in shipped
    assert f"{FAR_MESH_CLASS.tail_rel_pct:.0f} %" in shipped
    assert FAR_MESH_CLASS.issue in shipped
    assert FAR_MESH_CLASS.provenance in shipped


def test_the_message_says_the_three_things_a_caller_needs():
    """Not a spell-check: each clause is a decision the caller can act on --
    which direction is wrong, what to do about it, and what it costs."""
    text = far_mesh_class_message()
    assert "first order" in text  # the rate, so a mesh ladder is predictable
    assert "BSplineSolver" in text  # the converged alternative, named
    assert "HALVES" in text  # what doubling the mesh buys
    assert "nothing is remeshed" in text  # that it did not act


def test_warn_returns_exactly_what_it_emitted():
    """The seam a report generator or a diagnostic uses without catching a
    warning. If these two ever diverge, a caller reading the return value is
    quoting something the user never saw."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        returned = warn_far_mesh_class()
    assert len(rec) == 1
    assert str(rec[0].message) == returned == far_mesh_class_message()


def test_it_is_advisory_in_FACT_not_just_in_wording():
    """The claim "nothing is remeshed and nothing is refused", tested.

    Wording is cheap. This solves the same deck with the advisory raised and
    with it silenced, and requires bit-identical impedance and an unchanged
    mesh -- so an "advisory" that quietly bumped the segment count, or that
    refused a deck it disapproved of, fails here rather than in a user's
    convergence study.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        loud = RazorSolver(**_deck(), nec5_quadrature=True)
        z_loud, _ = loud.compute_impedance()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RazorFarMeshClass)
        quiet = RazorSolver(**_deck(), nec5_quadrature=True)
        z_quiet, _ = quiet.compute_impedance()

    assert complex(z_loud) == complex(z_quiet)
    # The stored mesh spec, which is what a covert remesh would have to move.
    # (`nsegs` is NOT the mesh on this path: it keeps its 101 default whenever
    # `n_per_edge_per_wire` is given, so asserting on it would pass for any
    # deck and prove nothing.)
    assert np.array_equal(
        np.asarray(loud.n_per_edge_per_wire), np.asarray(quiet.n_per_edge_per_wire)
    )
    assert np.array_equal(np.asarray(loud.n_per_edge_per_wire), np.asarray([[21]]))


def test_a_refusal_still_beats_the_advisory_to_the_door():
    """A deck that is going to be rejected is rejected, not advised about.

    The advisory sits after argument validation for this reason: telling a
    caller about razor's convergence class and THEN raising on their bad
    argument buries the actionable error under a paragraph of prose.
    """
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            RazorSolver(**_deck(), nec5_quadrature=True, n_qp_source=0)
    assert not [w for w in rec if isinstance(w.message, RazorFarMeshClass)]


def test_the_advisory_can_be_silenced_by_category():
    """A user who has read it once must be able to turn it off without
    silencing every UserWarning momwire raises -- which is the whole reason it
    has its own class rather than being a bare UserWarning."""
    assert issubclass(RazorFarMeshClass, UserWarning)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", RazorFarMeshClass)
        RazorSolver(**_deck(), nec5_quadrature=True)
    assert not [w for w in rec if isinstance(w.message, RazorFarMeshClass)]
