"""``LD 5`` (wire conductivity) on the ``nec5`` (EZNEC) dialect — momwire#1082.

The refusal this issue lifts was reasoning from silence: the 80-deck corpus
has 75 ``LD`` cards and every one is type 4 (a lumped impedance EZNEC
computed before writing the deck), so the corpus never exercised type 5 at
all, and "every load is LD 4" got generalised to a card nothing had tried.
A field report's deck (a licensed EZNEC Pro/2+ 7.0.3 / AutoEZ launch) carries
one: ``LD 5,0,1,402,5.7471E+7,1.``, whole-structure copper conductivity on a
78 m wire at 1.8 MHz.

Six gates, matching the issue's requirements:

1. parsing — whole-structure, per-wire and tag-0 forms, each with its own
   refusal boundary;
2. the conductivity reaches the SOLVE — R_in strictly higher with the card
   present, bounded by the skin-depth formula this fix wires in rather than
   a solved number lifted from one run;
3. a SAME-BASIS cross-check against ``momwire.deck._nec2``'s own ``LD 5``
   (momwire#510's standing rule: never gate cross-BASIS agreement — this is
   the same solver and the same basis, fed the value each dialect reads);
4. an adversarial probe — the conductivity lands on the WIRE it names and
   nothing else, so a parse-and-discard stub cannot pass;
5. a relative permeability other than 1 refuses, because
   ``momwire._wire_loading.wire_internal_impedance`` has no permeability
   parameter;
6. the printout's loading table gets a WIRE row, byte-compared against our
   licensed materials on a small deck of our own
   (``tests/fixtures/eznec_ld5_1082/``, a SEPARATE directory from
   ``tests/fixtures/eznec/`` so the 80-capture corpus and its manifest are
   untouched).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from momwire.bspline import BSplineSolver
from momwire.deck import DeckError
from momwire.deck import parse as parse_nec2
from momwire.deck._nec5 import Nec5Conductivity, parse_nec5
from momwire.eznec import _serve
from momwire.eznec._shell import render
from momwire._wire_loading import wire_internal_impedance

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_ld5_1082"

# A minimal deck in the shape EZNEC writes (the corpus's own "bare-wires
# control", `tests/test_deck_nec5.py`'s `DIPOLE`, at the same geometry):
# an 11-segment half-wave dipole at 299.7925 MHz, fed at its centre node.
DIPOLE = """CM ld5 gate test dipole
CE
GW 1,11,0.,-.25,0.,0.,.25,0.,.0005
GE 0,-1
GN -1
FR 0,1,0,0,299.7925
EX 4,1,6,0,1.414214,0.
PQ 0
RP 0,1,361,1000,90.,0.,0.,1.,0.
EN
"""

TWO_WIRES = """CM two wires
CE
GW 1,9,0.,-.25,0.,0.,.25,0.,.0005
GW 2,9,1.,-.25,0.,1.,.25,0.,.0005
GE 0,-1
GN -1
FR 0,1,0,0,299.7925
{ld}
EX 4,1,5,0,1.414214,0.
PQ 0
RP 0,1,361,1000,90.,0.,0.,1.,0.
EN
"""

SIGMA = 5.8e7  # S/m, copper order of magnitude (the field report's is 5.7471E+7)


def _with_card(card: str) -> str:
    """``DIPOLE`` with one extra card inserted just before ``EN``."""
    lines = DIPOLE.splitlines()
    at = lines.index("EN")
    return "\n".join([*lines[:at], card, "EN"]) + "\n"


# --------------------------------------------------------------------------
# gate 1: parsing
# --------------------------------------------------------------------------


def test_whole_structure_form_sets_the_declared_wire():
    deck = parse_nec5(_with_card(f"LD 5,0,1,11,{SIGMA:g},1."))
    assert deck.conductivities == (
        Nec5Conductivity(tag=0, segment_from=1, segment_thru=11, sigma=SIGMA),
    )
    assert deck.wire_conductivity == {1: SIGMA}


def test_per_wire_form_sets_only_that_wire():
    text = TWO_WIRES.format(ld=f"LD 5,1,1,9,{SIGMA:g},1.")
    deck = parse_nec5(text)
    assert deck.wire_conductivity == {1: SIGMA}


def test_tag_zero_spans_every_declared_wire():
    """No capture addresses tag 0 (the issue's own "no precedent" note), so
    this is the dedicated test the whole-structure spelling needs."""
    text = TWO_WIRES.format(ld=f"LD 5,0,1,18,{SIGMA:g},1.")
    deck = parse_nec5(text)
    assert deck.wire_conductivity == {1: SIGMA, 2: SIGMA}


def test_tag_zero_partial_range_refuses():
    with pytest.raises(DeckError, match="whole-structure"):
        parse_nec5(_with_card(f"LD 5,0,1,10,{SIGMA:g},1."))


def test_per_wire_partial_range_refuses():
    with pytest.raises(DeckError, match="whole wires only"):
        parse_nec5(_with_card(f"LD 5,1,1,5,{SIGMA:g},1."))


def test_undeclared_tag_refuses():
    with pytest.raises(DeckError, match="no GW card"):
        parse_nec5(_with_card(f"LD 5,9,1,11,{SIGMA:g},1."))


def test_ld_4_still_refuses_a_nonzero_fourth_field():
    """The LD 4 branch's own refusal survives the LD 5 split unchanged."""
    with pytest.raises(DeckError, match="single-point"):
        parse_nec5(_with_card("LD 4,1,1,2,50.,0."))


# --------------------------------------------------------------------------
# gate 5: relative permeability
# --------------------------------------------------------------------------


def test_mu_other_than_one_is_refused_by_name():
    with pytest.raises(DeckError, match="permeability"):
        parse_nec5(_with_card(f"LD 5,0,1,11,{SIGMA:g},2."))


def test_mu_of_one_passes():
    deck = parse_nec5(_with_card(f"LD 5,0,1,11,{SIGMA:g},1."))
    assert deck.wire_conductivity == {1: SIGMA}


def test_mu_field_absent_passes_and_defaults_to_one():
    deck = parse_nec5(_with_card(f"LD 5,0,1,11,{SIGMA:g}"))
    assert deck.wire_conductivity == {1: SIGMA}


# --------------------------------------------------------------------------
# gate 2: the conductivity reaches the solve
# --------------------------------------------------------------------------


def test_conductivity_raises_r_in_by_the_skin_depth_order_of_magnitude():
    lossy = _serve.serve(parse_nec5(_with_card(f"LD 5,0,1,11,{SIGMA:g},1.")))
    lossless = _serve.serve(parse_nec5(DIPOLE))
    r_lossy = lossy.sources[0].impedance.real
    r_lossless = lossless.sources[0].impedance.real
    delta_r = r_lossy - r_lossless

    # Direction: the metal now dissipates, so R_in can only go up.
    assert delta_r > 0.0

    # Order of magnitude: the per-length internal impedance this fix wires
    # in (`wire_internal_impedance`, the same function the refusal's message
    # names), times the dipole's own 0.5 m length -- not a number lifted
    # from a solve, and not tied to momwire's own basis error against any
    # other engine (momwire#510's standing rule).
    frequency_hz = 299.7925e6
    omega = 2.0 * math.pi * frequency_hz
    r_per_metre = wire_internal_impedance(omega, 0.0005, SIGMA).real
    expected = r_per_metre * 0.5
    assert expected / 10.0 < delta_r < expected * 10.0


# --------------------------------------------------------------------------
# gate 4: adversarial probe -- lands on the named wire and nowhere else
# --------------------------------------------------------------------------


def test_conductivity_lands_on_the_named_wire_and_not_the_other():
    """A no-op implementation that parses ``LD 5`` and never threads it into
    the solver passes every gate-1 test above and fails only here."""
    text = TWO_WIRES.format(ld=f"LD 5,1,1,9,{SIGMA:g},1.")
    deck = parse_nec5(text)
    structure = _serve.structure_of(deck)
    mesh = _serve.build_mesh(deck, structure, solver_class=BSplineSolver)
    wavelength = _serve.SPEED_OF_LIGHT_MHZ_M / 299.7925
    medium = _serve._medium(deck.ground, wavelength)
    solver = _serve._solver_for(
        deck, mesh, wavelength, medium, BSplineSolver, basis_kwargs={}
    )
    tags = [piece.tag for piece in mesh.pieces]
    conductivity = solver.wire_conductivity
    assert conductivity is not None
    seen = {1: False, 2: False}
    for tag, sigma in zip(tags, conductivity, strict=True):
        if tag == 1:
            assert sigma == SIGMA
            seen[1] = True
        else:
            assert tag == 2
            assert math.isnan(sigma)
            seen[2] = True
    assert seen == {1: True, 2: True}


# --------------------------------------------------------------------------
# gate 3: same-basis cross-check against the nec2 dialect's own LD 5
# --------------------------------------------------------------------------

# The nec2 dialect's own equivalent: same geometry, same centre feed (nec2
# addresses the segment CENTRE directly rather than a node), same frequency,
# `LD 5 0 0 0 sigma` being nec2's OWN whole-structure spelling (§ld--loading,
# `test_deck_nec2.py::test_ld5_whole_structure_form`) -- a different literal
# card than nec5's `LD 5,0,1,11,...` for the same semantic, which the two
# dialects are free to spell differently.
NEC2_DIPOLE = (
    "GW 1 11 0. -.25 0. 0. .25 0. .0005\nGE 0\nEX 0 1 6 0 1. 0.\nFR 0 1 0 0 299.7925\n"
)


def _solve_dipole_directly(sigma: float) -> np.ndarray:
    """The bare-solver Y-matrix for the DIPOLE geometry at one conductivity.

    Bypasses BOTH dialect readers' mesh-building on purpose: the point of
    this gate is that the CONDUCTIVITY VALUE each dialect resolves reaches
    the SAME solver construction and the SAME basis, not that the two
    dialects agree on addressing or feed placement (they do not -- nec2
    addresses segment centres, nec5 addresses nodes, and
    `test_contact_nec5_lane.py` already measures the resulting Ω-class
    offset between the two seams' own full pipelines).  So this is the
    decisive same-basis check the issue asks for, not the cross-basis
    agreement momwire#510 forbids gating.
    """
    wires = [np.array([[0.0, -0.25, 0.0], [0.0, 0.25, 0.0]])]
    solver = BSplineSolver(
        wires=wires,
        n_per_edge_per_wire=[[11]],
        feeds=[(0, 0.25, 1 + 0j)],
        wire_radius=0.0005,
        wavelength=_serve.SPEED_OF_LIGHT_MHZ_M / 299.7925,
        wire_conductivity=np.array([sigma]),
    )
    return solver.compute_port_solution().y


def test_same_basis_cross_check_agrees_bit_exact():
    nec5_sigma = parse_nec5(_with_card(f"LD 5,0,1,11,{SIGMA:g},1.")).wire_conductivity[
        1
    ]
    nec2_sigma = (
        parse_nec2(NEC2_DIPOLE + f"LD 5 0 0 0 {SIGMA:g}\nXQ\nNX\n")
        .wires[0]
        .material.conductivity
    )
    assert nec5_sigma == nec2_sigma == SIGMA

    y_nec5 = _solve_dipole_directly(nec5_sigma)
    y_nec2 = _solve_dipole_directly(nec2_sigma)
    # Bit-exact: same solver, same basis, same float -- there is nothing left
    # to differ by.
    assert np.array_equal(y_nec5, y_nec2)


# --------------------------------------------------------------------------
# gate 6: the printout's loading table
# --------------------------------------------------------------------------


def _loading_table_block(lines: list[str]) -> list[str]:
    """The heading through the last data row -- NOT ``ALLOCATE CM:``, which
    is the matrix size and basis-dependent (BSplineSolver's unknown count is
    not NEC-5's), so it is not part of what this FORMAT gate is answering
    for."""
    start = next(
        i for i, line in enumerate(lines) if "STRUCTURE IMPEDANCE LOADING" in line
    )
    end = next(i for i, line in enumerate(lines) if line.startswith("ALLOCATE CM:"))
    return lines[start:end]


def test_the_printout_emits_a_wire_row_byte_for_byte():
    """gate 6: a FORMAT comparison, so byte-exactness is the right bar here
    (unlike gate 2's physics comparison) -- verified against our licensed
    materials on `tests/fixtures/eznec_ld5_1082/synthetic_ld5.{nec,out}`, a
    small deck of our own, kept out of `tests/fixtures/eznec/` so the
    80-capture corpus and its manifest (`test_eznec_reproducibility.py`)
    stay untouched."""
    deck_text = (FIXTURES / "synthetic_ld5.nec").read_text()
    oracle_lines = (FIXTURES / "synthetic_ld5.out").read_text().splitlines()
    rendered_lines = render(deck_text).splitlines()
    assert _loading_table_block(rendered_lines) == _loading_table_block(oracle_lines)


def test_the_printout_names_no_load_when_only_the_wire_row_would_print():
    """The pre-#1082 ``THIS STRUCTURE IS NOT LOADED`` line must not survive
    a deck whose only ``LD`` card is a conductivity."""
    text = (FIXTURES / "synthetic_ld5.nec").read_text()
    rendered = render(text)
    assert "THIS STRUCTURE IS NOT LOADED" not in rendered
    assert "WIRE" in rendered
