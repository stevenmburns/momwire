"""Loads are per execute group, as NEC's are (momwire#1069).

An ``LD`` arms (spec ``#arming``) and ``LD -1`` clears, and the oracle answers
each execute card with the loads in force AT it: a deck that runs bare and
then loaded prints THIS STRUCTURE IS NOT LOADED and the bare impedance for its
first group.  The model used to carry one deck-level load list, applied to
every group.

The oracle side of this is two captured fixtures, ``dipole_ld_after_xq`` and
``dipole_ld_cleared_between_runs`` under ``tests/fixtures/nec_portal/``, which
``test_portal_differential.py`` runs through the portal and compares with the
oracle's printout like every other fixture.  This module pins the model and
the port algebra underneath.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire.deck import DeckError, LoadSpec, build_solver, parse
from momwire.portal._portal import DeckSolver, parse_deck, run_deck

DIPOLE = "GW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\nGE 0\n"
DRIVE = "EX 0 1 5 0 1.\nFR 0 1 0 0 30. 0\n"
SERIES = LoadSpec("series", r=50.0, l=1e-6, c=0.0)


def _group_impedances(text: str) -> list[complex]:
    deck = parse_deck(text)
    solver = DeckSolver(deck, basis="bspline")
    out = []
    for index, group in enumerate(deck.groups):
        if group is None:
            continue
        result = solver.solve_group(group, group.freqs_mhz[0], index)
        port, _segment, volts = result.driven[0]
        out.append(complex(volts / result.i_source[port]))
    return out


def test_an_ld_between_two_runs_loads_only_the_second():
    model = parse(DIPOLE + DRIVE + "XQ\nLD 0 1 3 3 50. 1.e-6 0.\nXQ\nNX\n")
    assert [spec for _w, _a, spec in model.loads] == [SERIES]
    first, second = model.groups
    assert first.loads == () and second.loads == (0,)
    # The LD between the runs reprints LOADING / ENVIRONMENT / MATRIX TIMING
    # and no FREQUENCY block, which is the operator-card shape.
    assert second.refilled_partial and not second.refilled


def test_the_first_run_answers_the_bare_antenna_exactly():
    """Not "close to": the first group stamps nothing, so its impedance IS the
    bare deck's, and the second is the loaded deck's."""
    scoped = _group_impedances(DIPOLE + DRIVE + "XQ\nLD 0 1 3 3 50. 1.e-6 0.\nXQ\n")
    # The bare and loaded single-run decks, with the same port set (a zero
    # load at segment 3 is the shorted gap a group without its load sees).
    bare = _group_impedances(DIPOLE + "LD 4 1 3 3 0. 1.E-300\n" + DRIVE + "XQ\n")
    loaded = _group_impedances(DIPOLE + "LD 0 1 3 3 50. 1.e-6 0.\n" + DRIVE + "XQ\n")
    assert scoped[0] == pytest.approx(bare[0], rel=1e-12)
    assert scoped[1] == pytest.approx(loaded[0], rel=1e-12)
    assert abs(scoped[1] - scoped[0]) > 50.0


def test_ld_minus_one_between_runs_unloads_the_next_one():
    model = parse(
        DIPOLE
        + "LD 0 1 3 3 50. 1.e-6 0.\n"
        + DRIVE
        + "XQ\nLD -1\nXQ\nLD 4 1 7 7 20. 30.\nXQ\nNX\n"
    )
    assert [spec for _w, _a, spec in model.loads] == [
        SERIES,
        LoadSpec("fixed", r=20.0, x=30.0),
    ]
    assert [g.loads for g in model.groups] == [(0,), (), (1,)]


def test_a_load_cleared_before_any_run_is_not_in_the_model():
    model = parse(DIPOLE + "LD 0 1 3 3 50. 0. 0.\nLD -1\n" + DRIVE + "XQ\nNX\n")
    assert model.loads == ()
    assert model.groups[0].loads == ()


def test_a_load_restated_after_a_clear_is_one_entry():
    model = parse(
        DIPOLE
        + "LD 0 1 3 3 50. 1.e-6 0.\n"
        + DRIVE
        + "XQ\nLD -1\nLD 0 1 3 3 50. 1.e-6 0.\nXQ\nNX\n"
    )
    assert len(model.loads) == 1
    assert [g.loads for g in model.groups] == [(0,), (0,)]


def test_one_segment_loaded_differently_in_two_runs_is_one_gap():
    """Two loads at one position that are never in force together are one
    gap stamped per group, not the two-loads-on-a-segment refusal."""
    text = (
        DIPOLE
        + "LD 0 1 3 3 50. 0. 0.\n"
        + DRIVE
        + "XQ\nLD -1\nLD 4 1 3 3 10. -40.\nXQ\n"
    )
    model = parse(text + "NX\n")
    assert len(model.loads) == 2
    built = build_solver(model)
    assert built.ports.load_ports[0] == built.ports.load_ports[1]
    scoped = _group_impedances(text)
    alone = _group_impedances(DIPOLE + "LD 4 1 3 3 10. -40.\n" + DRIVE + "XQ\n")
    assert scoped[1] == pytest.approx(alone[0], rel=1e-12)


def test_two_loads_in_force_together_at_one_position_still_refuse():
    with pytest.raises(DeckError, match="already carries a load"):
        parse(DIPOLE + "LD 0 1 3 3 50. 0. 0.\nLD 4 1 3 3 10. 5.\n" + DRIVE + "XQ\n")


def test_wire_loading_that_changes_between_runs_refuses():
    """A material enters the fill, and the fill is shared by every group at
    one frequency, so a conductivity in force for one run and not another
    cannot be served by stamping."""
    with pytest.raises(DeckError, match="changes between execute cards"):
        parse(DIPOLE + DRIVE + "XQ\nLD 5 0 0 0 5.8E7\nXQ\nNX\n")


def test_wire_loading_after_the_last_run_reaches_no_run():
    """As in NEC: the card reads, and no execute card follows it."""
    model = parse(DIPOLE + DRIVE + "XQ\nLD 5 0 0 0 5.8E7\nNX\n")
    assert model.wires[0].material is None


def test_the_printout_says_not_loaded_for_the_bare_run():
    stdout, _err = run_deck(DIPOLE + DRIVE + "XQ\nLD 0 1 3 3 50. 1.e-6 0.\nXQ\nNX\n")
    blocks = stdout.split("STRUCTURE IMPEDANCE LOADING")
    assert len(blocks) == 3
    assert "THIS STRUCTURE IS NOT LOADED" in blocks[1].split("ANTENNA ENVIRONMENT")[0]
    assert "SERIES" in blocks[2].split("ANTENNA ENVIRONMENT")[0]
    assert stdout.count("--------- FREQUENCY --------") == 1


def test_a_hand_built_model_without_scoping_still_loads_every_group():
    """`ExecuteGroup.loads` defaults to None, which is every load: a model
    built without per-group scoping keeps meaning what it meant."""
    from dataclasses import replace

    model = parse(DIPOLE + "LD 0 1 3 3 50. 1.e-6 0.\n" + DRIVE + "XQ\nNX\n")
    unscoped = replace(
        model, groups=tuple(replace(g, loads=None) for g in model.groups)
    )
    a = build_solver(model).solver.compute_port_solution().y
    b = build_solver(unscoped).solver.compute_port_solution().y
    assert np.array_equal(a, b)
