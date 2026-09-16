"""``LD 2`` / ``LD 3`` (per-unit-length RLC) on BOTH dialects — momwire#1088.

Both readers refused these two types by name, so this was a momwire-wide gap
rather than the nec5-behind-nec2 parity gap #1085 tracks.  The consumer is
antennaknobs: its NEC-5 writer spells a jacketed wire's equivalent-radius
pair (its issue #1523) as an ``LD 2 tag 0 0 0. L' 0.`` — a per-metre
inductance — beside an ``LD 5`` for the conductivity, and three catalog
designs (``dipoles.invvee_catenary``, ``dipoles.pota_invvee``,
``wire.efhw_sloper``) refused to parse over that one card.

The fix reads the card as a DISTRIBUTED series impedance Z'(w) [Ohm/m] and
folds it into the same per-wire sum ``LD 5``'s internal impedance lands in
(:func:`momwire._wire_loading.series_impedance_per_wire`), so one seam change
covers every family whose ``capabilities.wire_loading`` is True and the terms
ADD rather than exclude each other.

WHAT THE CAPACITANCE FIELD TURNED OUT TO BE
-------------------------------------------
Measured 2026-09-16 on nec2c 1.3.1 and on our licensed NEC-5 materials, on a
1 m centre-fed wire at 30 MHz at 4, 8 and 16 segments:

* the RESISTANCE and INDUCTANCE fields are genuinely per unit length — a
  segment of length d carries R'd and jw L' d, ``LD 2`` with R' alone
  reproduces ``LD 0`` with a lumped R'd to every printed digit on nec2c, and
  the answer CONVERGES as the mesh refines (nec2c dZ at 4/8/16 segments for
  ``LD 3,1,1,n,300.,1.E-6,0.``: 35.8+56.7j, 31.4+49.6j, 29.1+45.9j);
* the CAPACITANCE field is not.  NEC multiplies it BY the segment length
  instead of dividing: ``LD 2`` with C' alone reproduces ``LD 0`` with a
  LUMPED C = C'd, exactly, so the card's contribution to the wire's PER-METRE
  impedance is 1/(jw C' d^2) and moves with the mesh (nec2c dX at 4/8/16:
  -3110, -7851, -16723 ohms — the reactance grows without bound as the mesh
  refines, where a genuine per-metre term would converge).

A number that only means something multiplied by exactly the deck's segment
length is not a material property, and ``z_wire`` is read by four different
testing schemes — only a point-matched one would reproduce NEC's
chain-of-lumped-capacitors from a per-metre value.  So both readers REFUSE a
nonzero capacitance field by name, and serve R' and L'.  That costs nothing
observed: every ``LD 2`` antennaknobs writes has ``0.`` there, and no deck in
either repo's corpus carries a nonzero one.

The gates, matching the issue's requirements:

1. the SEAM's own closed form, at two frequencies — that one lives in
   ``tests/test_wire_loading.py`` beside the rest of the loading physics;
2. the nec2 dialect against nec2c AND PyNEC on ``basis="sinusoidal"``, which
   is NEC-2's own basis (a SAME-BASIS comparison, the only kind momwire#510
   allows gating), plus a power BOUND on the loading's own share of R_in;
3. the nec5 dialect against our licensed NEC-5 materials on
   ``basis="razor-nec5"``, the formulation twin;
4. an adversarial probe — the card lands on the wire it names and no other;
5. the range rules: whole wire and whole structure, explicit and wildcard,
   a partial range refusing, ``LD 2`` + ``LD 5`` composing, and a basis
   without wire loading refusing by the card's own name;
6. the printout's loading table, byte-compared against our licensed
   materials on a small deck of our own
   (``tests/fixtures/eznec_ld23_1088/``, a SEPARATE directory so the
   80-capture corpus and its manifest are untouched).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from momwire._wire_loading import DistributedRLC, loading_for
from momwire.bspline import BSplineSolver
from momwire.deck import DeckError, build_solver
from momwire.deck import parse as parse_nec2
from momwire.deck._nec5 import Nec5DistributedRLC, parse_nec5
from momwire.eznec import _serve
from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_ld23_1088"

# The gate deck, in both dialects: a 5 m centre-fed dipole at 30 MHz, 11
# segments, 2 mm radius.  Free space, so nothing but the wire and its loading
# is in the answer.
N_SEG, HALF, RADIUS, FREQ_MHZ = 11, 2.5, 0.002, 30.0
OMEGA = 2.0 * math.pi * FREQ_MHZ * 1e6

NEC2_BODY = (
    f"CM momwire#1088 gate\nCE\n"
    f"GW 1 {N_SEG} 0. -{HALF} 0. 0. {HALF} 0. {RADIUS}\nGE 0\n"
    f"EX 0 1 6 0 1. 0.\nFR 0 1 0 0 {FREQ_MHZ}\n"
)

NEC2_TWO_WIRES = (
    "GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3\n"
    "GW 2 4 0. 1. 0. 1. 1. 0. 1.E-3\n"
    "GE 0\nEX 0 1 3 0 1.\nFR 0 1 0 0 14.\n"
)

# The nec5 twin uses 21 segments: `razor-nec5` is first order in the mesh
# (momwire#845), and the twin relationship is what is being pinned, so the
# rung is chosen where the two agree closely enough that the LOADING is what
# the comparison is about.
N_SEG_5 = 21
NEC5_BODY = (
    f"CM momwire#1088 gate\nCE\n"
    f"GW 1,{N_SEG_5},0.,-{HALF},0.,0.,{HALF},0.,{RADIUS}\nGE 0,-1\nGN -1\n"
    f"FR 0,1,0,0,{FREQ_MHZ}\n"
)

NEC5_TWO_WIRES = """CM two wires
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

# The two loaded cards the physics gates use, spelled in each dialect.
LD2_FIELDS = (4.0, 5.0e-7)  # R' [ohm/m], L' [H/m]
LD3_FIELDS = (300.0, 2.0e-6)


def nec2_deck(ld: str = "") -> str:
    return NEC2_BODY + (ld + "\n" if ld else "") + "XQ\nEN\n"


def nec5_deck(ld: str = "") -> str:
    return (
        NEC5_BODY
        + (ld + "\n" if ld else "")
        + f"EX 0,1,{(N_SEG_5 + 1) // 2},0,1.,0.\nXQ\nEN\n"
    )


def _z_sinusoidal(ld: str) -> complex:
    """The nec2 dialect's own driving-point Z on NEC-2's basis."""
    built = build_solver(parse_nec2(nec2_deck(ld)), basis="sinusoidal")
    return complex(1.0 / built.solver.compute_port_solution().y[0, 0])


# --------------------------------------------------------------------------
# gate 5 (part 1): the nec2 dialect's range rules and refusals
# --------------------------------------------------------------------------


def test_nec2_whole_structure_form_sets_every_wire():
    """``I2 = 0, I3 = 0``, the same whole-structure sentinel ``LD 5`` uses —
    and the shape antennaknobs' writer emits."""
    model = parse_nec2(NEC2_TWO_WIRES + "LD 2 0 0 0 0. 2.5E-7 0.\nXQ\nNX\n")
    spec = DistributedRLC("series", l=2.5e-7)
    assert [w.material.distributed_rlc for w in model.wires] == [spec, spec]


def test_nec2_whole_wire_form_sets_only_that_wire():
    model = parse_nec2(NEC2_TWO_WIRES + "LD 2 1 0 0 1.5 2.E-6 0.\nXQ\nNX\n")
    assert model.wires[0].material.distributed_rlc == DistributedRLC(
        "series", r=1.5, l=2.0e-6
    )
    assert model.wires[1].material is None


def test_nec2_explicit_whole_wire_range_is_the_wildcard_range():
    """``1`` to the wire's own segment count, the spelling with no sentinel
    in it, resolves to the same loading as ``0,0``."""
    explicit = parse_nec2(NEC2_TWO_WIRES + "LD 2 2 1 4 1.5 0. 0.\nXQ\nNX\n")
    wildcard = parse_nec2(NEC2_TWO_WIRES + "LD 2 2 0 0 1.5 0. 0.\nXQ\nNX\n")
    assert [w.material for w in explicit.wires] == [w.material for w in wildcard.wires]
    assert explicit.wires[1].material.distributed_rlc == DistributedRLC("series", r=1.5)


def test_nec2_parallel_type_is_read_as_parallel():
    model = parse_nec2(NEC2_TWO_WIRES + "LD 3 1 0 0 300. 1.E-6 0.\nXQ\nNX\n")
    assert model.wires[0].material.distributed_rlc == DistributedRLC(
        "parallel", r=300.0, l=1.0e-6
    )


def test_nec2_partial_wire_range_refuses_by_name():
    with pytest.raises(DeckError) as exc:
        parse_nec2(NEC2_TWO_WIRES + "LD 2 1 1 2 1.5 0. 0.\nXQ\nNX\n")
    assert str(exc.value) == (
        "LD 2 per-unit-length loading on a partial-wire segment range is "
        "not supported by this engine — a distributed RLC is a wire "
        "property and covers whole wires only"
    )


def test_nec2_capacitance_field_refuses_by_name():
    """The measurement in this module's docstring, at the card."""
    with pytest.raises(DeckError) as exc:
        parse_nec2(NEC2_TWO_WIRES + "LD 2 1 0 0 0. 0. 1.E-11\nXQ\nNX\n")
    message = str(exc.value)
    assert message.startswith("LD 2 asks for a capacitance of 1e-11")
    assert "segment length" in message and "momwire#1088" in message


def test_nec2_negative_field_refuses_by_name():
    with pytest.raises(DeckError, match="negative per-unit-length resistance"):
        parse_nec2(NEC2_TWO_WIRES + "LD 2 1 0 0 -1. 0. 0.\nXQ\nNX\n")
    with pytest.raises(DeckError, match="negative per-unit-length inductance"):
        parse_nec2(NEC2_TWO_WIRES + "LD 3 1 0 0 1. -1.E-6 0.\nXQ\nNX\n")


def test_nec2_zero_valued_card_is_a_no_op():
    """The same rule a zero-valued lumped ``LD 0`` gets: a card
    byte-identical to omitting it is dropped, not refused."""
    model = parse_nec2(NEC2_TWO_WIRES + "LD 2 1 1 2 0. 0. 0.\nXQ\nNX\n")
    assert [w.material for w in model.wires] == [None, None]


def test_nec2_ld_minus_one_clears_the_per_metre_loading():
    """``LD -1`` nullifies every load read so far, and the per-metre RLC is
    read under the same mnemonic."""
    model = parse_nec2(NEC2_TWO_WIRES + "LD 2 0 0 0 1.5 0. 0.\nLD -1\nXQ\nNX\n")
    assert [w.material for w in model.wires] == [None, None]


def test_nec2_still_refuses_the_types_it_never_served():
    """Serving 2 and 3 must not open the door to the 4nec2 extensions."""
    for ldtyp in (6, 7):
        with pytest.raises(DeckError) as exc:
            parse_nec2(NEC2_TWO_WIRES + f"LD {ldtyp} 1 0 0 1. 1. 1.\nXQ\nNX\n")
        assert str(exc.value) == (f"LD type {ldtyp} is not supported by this engine")


# --------------------------------------------------------------------------
# gate 5 (part 2): the nec5 dialect's range rules and refusals
# --------------------------------------------------------------------------


def test_nec5_whole_structure_wildcard_fans_out_to_every_wire():
    """antennaknobs' own writer's spelling: ``LD 2 0 0 0 0. L' 0.``."""
    deck = parse_nec5(NEC5_TWO_WIRES.format(ld="LD 2,0,0,0,0.,2.5E-07,0."))
    spec = DistributedRLC("series", l=2.5e-7)
    assert dict(deck.wire_distributed_rlc) == {1: spec, 2: spec}
    assert deck.distributed_rlc == (
        Nec5DistributedRLC(tag=0, segment_from=1, segment_thru=18, spec=spec),
    )


def test_nec5_whole_structure_explicit_range_is_the_same():
    explicit = parse_nec5(NEC5_TWO_WIRES.format(ld="LD 2,0,1,18,0.,2.5E-07,0."))
    wildcard = parse_nec5(NEC5_TWO_WIRES.format(ld="LD 2,0,0,0,0.,2.5E-07,0."))
    assert explicit.distributed_rlc == wildcard.distributed_rlc


def test_nec5_per_wire_form_sets_only_that_wire():
    deck = parse_nec5(NEC5_TWO_WIRES.format(ld="LD 3,2,0,0,300.,1.E-6,0."))
    assert dict(deck.wire_distributed_rlc) == {
        2: DistributedRLC("parallel", r=300.0, l=1.0e-6)
    }
    assert deck.distributed_rlc[0].segment_from == 1
    assert deck.distributed_rlc[0].segment_thru == 9


def test_nec5_partial_range_refuses_by_name():
    with pytest.raises(DeckError, match="covers whole wires only"):
        parse_nec5(NEC5_TWO_WIRES.format(ld="LD 2,1,1,5,1.5,0.,0."))
    with pytest.raises(DeckError, match="whole-structure"):
        parse_nec5(NEC5_TWO_WIRES.format(ld="LD 2,0,1,10,1.5,0.,0."))


def test_nec5_undeclared_tag_refuses():
    with pytest.raises(DeckError, match="no GW card"):
        parse_nec5(NEC5_TWO_WIRES.format(ld="LD 2,9,0,0,1.5,0.,0."))


def test_nec5_short_card_refuses_rather_than_defaulting_to_zeros():
    """``Card.f`` reads a missing field as 0.0, which would otherwise make a
    truncated card look like a legal zero-valued one."""
    with pytest.raises(DeckError, match="needs at least 7"):
        parse_nec5(NEC5_TWO_WIRES.format(ld="LD 2,1,0,0,1.5"))


def test_nec5_capacitance_field_refuses_by_name():
    with pytest.raises(DeckError) as exc:
        parse_nec5(NEC5_TWO_WIRES.format(ld="LD 3,1,0,0,0.,0.,1.E-11"))
    message = str(exc.value)
    assert message.startswith("LD 3 asks for a capacitance of 1e-11")
    assert "segment length" in message and "momwire#1088" in message


def test_nec5_ld5_range_refusals_survive_the_shared_helper():
    """The range rule is now one method both cards call; ``LD 5``'s own
    messages must still name ``LD 5``."""
    with pytest.raises(DeckError, match=r"^LD 5 addresses tag 1 segments"):
        parse_nec5(NEC5_TWO_WIRES.format(ld="LD 5,1,1,5,5.8E+7,1."))
    with pytest.raises(DeckError, match=r"^LD 5 names tag 9"):
        parse_nec5(NEC5_TWO_WIRES.format(ld="LD 5,9,0,0,5.8E+7,1."))


# --------------------------------------------------------------------------
# gate 5 (part 3): composition, and a basis that cannot host the card
# --------------------------------------------------------------------------


def test_ld2_and_ld5_on_one_wire_compose_in_z_wire():
    """Both terms, added — not one overwriting the other.  This is the
    antennaknobs shape (its #1523 a'+L' pair writes exactly this pair of
    cards), so it is the case with a downstream consumer."""
    both = build_solver(
        parse_nec2(
            NEC2_TWO_WIRES + "LD 2 1 0 0 1.5 0. 0.\nLD 5 1 0 0 5.8E+7\nXQ\nNX\n"
        ),
        basis="sinusoidal",
    ).solver
    metal = build_solver(
        parse_nec2(NEC2_TWO_WIRES + "LD 5 1 0 0 5.8E+7\nXQ\nNX\n"),
        basis="sinusoidal",
    ).solver
    omega = 2.0 * math.pi * 14e6
    z_both = loading_for(both, omega).z_wire[0]
    z_metal = loading_for(metal, omega).z_wire[0]
    assert z_both == pytest.approx(z_metal + 1.5, rel=1e-12)
    assert z_metal.real > 0.0  # the skin term is really there to add to


@pytest.mark.parametrize(
    ("card", "name"),
    [("LD 2 0 0 0 1.5 0. 0.", "LD 2"), ("LD 3 0 0 0 300. 0. 0.", "LD 3")],
)
def test_a_basis_without_wire_loading_refuses_by_the_cards_own_name(card, name):
    """momwire#1087's check keys on the loading kwargs being non-empty, so a
    deck whose ONLY loading is an ``LD 2`` has to reach it too — and hear its
    own card named back, not ``LD 5``."""
    with pytest.raises(ValueError) as exc:
        build_solver(parse_nec2(nec2_deck(card)), basis="pulse")
    assert str(exc.value).startswith(f"{name} sets wire loading on this deck")
    assert "does not serve it" in str(exc.value)


def test_the_nec5_seam_refuses_by_name_too():
    deck = parse_nec5(NEC5_TWO_WIRES.format(ld="LD 2,1,0,0,1.5,0.,0."))
    with pytest.raises(_serve.ServeRefusal) as exc:
        _serve.serve(deck, basis="pulse")
    assert "LD 2 sets a per-unit-length series RLC" in str(exc.value)


# --------------------------------------------------------------------------
# gate 4: adversarial — the card lands on the wire it names and no other
# --------------------------------------------------------------------------


def test_nec2_the_card_lands_on_the_named_wire_only():
    """A reader that parses ``LD 2`` and never threads it into the solver
    passes every parse gate above and fails only here."""
    solver = build_solver(
        parse_nec2(NEC2_TWO_WIRES + "LD 2 2 0 0 1.5 2.E-6 0.\nXQ\nNX\n"),
        basis="sinusoidal",
    ).solver
    assert solver.distributed_rlc[0] is None
    assert solver.distributed_rlc[1] == DistributedRLC("series", r=1.5, l=2.0e-6)
    omega = 2.0 * math.pi * 14e6
    z_wire = loading_for(solver, omega).z_wire
    assert z_wire[0] == 0j
    assert z_wire[1] == pytest.approx(1.5 + 1j * omega * 2.0e-6, rel=1e-12)


def test_nec2_the_two_placements_give_different_impedances():
    """Same card, other wire: the answer must move.  (Wire 2 is the
    undriven one, so its loading reaches the feed only through coupling —
    which is exactly why a stub that loads every wire would be caught.)"""
    on_1 = _two_wire_impedance("LD 2 1 0 0 1.5 2.E-6 0.")
    on_2 = _two_wire_impedance("LD 2 2 0 0 1.5 2.E-6 0.")
    bare = _two_wire_impedance("")
    assert abs(on_1 - bare) > 1.0
    assert abs(on_2 - bare) > 1e-4
    assert abs(on_1 - on_2) > 1.0


def _two_wire_impedance(ld: str) -> complex:
    built = build_solver(
        parse_nec2(NEC2_TWO_WIRES + (ld + "\n" if ld else "") + "XQ\nNX\n"),
        basis="sinusoidal",
    )
    return complex(1.0 / built.solver.compute_port_solution().y[0, 0])


def test_nec5_the_card_lands_on_the_named_wire_only():
    deck = parse_nec5(NEC5_TWO_WIRES.format(ld="LD 2,2,0,0,1.5,2.E-6,0."))
    structure = _serve.structure_of(deck)
    mesh = _serve.build_mesh(deck, structure, solver_class=BSplineSolver)
    wavelength = _serve.SPEED_OF_LIGHT_MHZ_M / 299.7925
    solver = _serve._solver_for(
        deck,
        mesh,
        wavelength,
        _serve._medium(deck.ground, wavelength),
        BSplineSolver,
        basis_kwargs={},
    )
    seen = {1: False, 2: False}
    for piece, spec in zip(mesh.pieces, solver.distributed_rlc, strict=True):
        if piece.tag == 1:
            assert spec is None
            seen[1] = True
        else:
            assert piece.tag == 2
            assert spec == DistributedRLC("series", r=1.5, l=2.0e-6)
            seen[2] = True
    assert seen == {1: True, 2: True}


# --------------------------------------------------------------------------
# gate 2: the nec2 dialect against nec2c and PyNEC, on NEC-2's own basis
# --------------------------------------------------------------------------

# nec2c 1.3.1 (/usr/bin/nec2c), run 2026-09-16 on the decks `nec2_deck()`
# builds, free space, 30 MHz.  Transcribed rather than shelled out to so the
# gate needs no binary on the box; PyNEC is checked LIVE beside it, which is
# what keeps these three numbers honest.
NEC2C = {
    "": 80.263 + 45.934j,
    "LD 2 0 0 0 4. 5.E-7 0.": 115.88 + 325.52j,
    "LD 3 0 0 0 300. 2.E-6 0.": 740.29 + 281.27j,
}

# The measured agreement floor, same run: momwire's sinusoidal row against
# nec2c on the UNLOADED deck is 0.45 % relative, which is the basis-level
# difference between two implementations of NEC-2's basis and has nothing to
# do with this issue.  The LOADED decks come in at 0.19 % (LD 2) and 0.09 %
# (LD 3) — BETTER than the unloaded deck, because the loading dominates the
# impedance — so a gate at 3x the unloaded floor has real headroom and would
# still catch a loading term that was wrong by a factor near one.
UNLOADED_FLOOR = 0.0045
LOADED_GATE = 3.0 * UNLOADED_FLOOR


def test_the_unloaded_agreement_floor_is_what_this_module_says_it_is():
    """Measure it first — the loaded gate is a multiple of this number, so
    a drift here must fail here rather than silently loosen gate 2."""
    z_mw = _z_sinusoidal("")
    z_oracle = NEC2C[""]
    rel = abs(z_mw - z_oracle) / abs(z_oracle)
    assert rel == pytest.approx(UNLOADED_FLOOR, abs=5e-4), rel


@pytest.mark.parametrize("card", ["LD 2 0 0 0 4. 5.E-7 0.", "LD 3 0 0 0 300. 2.E-6 0."])
def test_loaded_impedance_matches_nec2c_within_three_times_the_floor(card):
    z_mw = _z_sinusoidal(card)
    z_oracle = NEC2C[card]
    rel = abs(z_mw - z_oracle) / abs(z_oracle)
    assert rel < LOADED_GATE, f"{card}: momwire={z_mw}, nec2c={z_oracle}, {rel=}"


@pytest.mark.parametrize(
    ("ldtyp", "card"),
    [
        (2, "LD 2 0 0 0 4. 5.E-7 0."),
        (3, "LD 3 0 0 0 300. 2.E-6 0."),
    ],
)
def test_loaded_impedance_matches_pynec_within_three_times_the_floor(ldtyp, card):
    """PyNEC (nec2++) is the second NEC-2 oracle, driven LIVE — so the
    transcribed nec2c numbers above cannot quietly rot."""
    pytest.importorskip("PyNEC")
    import PyNEC as nec

    fields = LD2_FIELDS if ldtyp == 2 else LD3_FIELDS
    ctx = nec.nec_context()
    geo = ctx.get_geometry()
    geo.wire(1, N_SEG, 0.0, -HALF, 0.0, 0.0, HALF, 0.0, RADIUS, 1.0, 1.0)
    ctx.geometry_complete(0)
    ctx.ld_card(ldtyp, 0, 0, 0, fields[0], fields[1], 0.0)
    ctx.ex_card(0, 1, 6, 0, 1.0, 0.0, 0, 0, 0, 0)
    ctx.fr_card(0, 1, FREQ_MHZ, 0)
    ctx.xq_card(0)
    z_oracle = complex(ctx.get_input_parameters(0).get_impedance()[0])
    z_mw = _z_sinusoidal(card)
    rel = abs(z_mw - z_oracle) / abs(z_oracle)
    assert rel < LOADED_GATE, f"{card}: momwire={z_mw}, pynec={z_oracle}, {rel=}"
    # And the two oracles agree with each other far more closely than either
    # agrees with momwire, which is what makes the transcription safe.
    assert abs(z_oracle - NEC2C[card]) / abs(z_oracle) < 1e-3


# --------------------------------------------------------------------------
# gate 2 (part 2): the loading's own share of R_in, from the solved current
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reactive_only", "loaded", "r_prime"),
    [
        (
            "LD 2 0 0 0 0. 1.E-7 0.",
            "LD 2 0 0 0 4. 1.E-7 0.",
            4.0,
        ),
        (
            "LD 3 0 0 0 0. 1.E-7 0.",
            "LD 3 0 0 0 50. 1.E-7 0.",
            None,  # Re(Z') of the parallel combination, computed below
        ),
    ],
)
def test_delta_r_in_is_the_loss_integral_over_the_solved_current(
    reactive_only, loaded, r_prime
):
    """dR_in = Re(Z') * integral |I(s)|^2 ds / |I_feed|^2, within 5 %.

    The two decks differ ONLY in R': both carry the same L', so the current
    distribution barely moves between them and the first-order identity is
    the right one to assert.  (Against the BARE deck it is not: adding L'
    shifts X_in by ~50 ohms, the current collapses, and dR_in then mixes in
    the change in radiation resistance — measured 165 % away from the loss
    integral, which is a fact about perturbation theory and not about this
    fix.)

    The integral is a trapezoid over 4001 samples of the SOLVED current from
    `currents_at_knots`, deliberately NOT `wire_loss_power`'s own closed-form
    integral — gating that against itself would prove nothing.
    """
    if r_prime is None:
        r_prime = (1.0 / (1.0 / 50.0 + 1.0 / (1j * OMEGA * 1.0e-7))).real

    built_base = build_solver(parse_nec2(nec2_deck(reactive_only)), basis="sinusoidal")
    sol_base = built_base.solver.compute_port_solution()
    built = build_solver(parse_nec2(nec2_deck(loaded)), basis="sinusoidal")
    sol = built.solver.compute_port_solution()

    delta_r = (1.0 / sol.y[0, 0] - 1.0 / sol_base.y[0, 0]).real
    arc = [np.linspace(0.0, 2.0 * HALF, 4001)]
    current = built.solver.currents_at_knots(sol.coeffs[:, 0], arc)[0]
    integral = float(np.trapezoid(np.abs(current) ** 2, arc[0]))
    bound = r_prime * integral / abs(sol.y[0, 0]) ** 2

    assert delta_r > 0.0
    assert delta_r == pytest.approx(bound, rel=0.05), (
        f"{loaded}: dR={delta_r}, loss integral={bound}"
    )


# --------------------------------------------------------------------------
# gate 3: the nec5 dialect against our licensed NEC-5 materials
# --------------------------------------------------------------------------

# NEC-5 (LLNL-CODE-746721), run 2026-09-16 on the decks `nec5_deck()` builds:
# the 5 m dipole at 21 segments, 30 MHz, free space (`GE 0,-1` / `GN -1`),
# centre-fed at node 11.  `razor-nec5` is the formulation twin
# (`tests/test_razor_nec5_twin.py`), so this is a same-basis comparison in
# the sense momwire#510 requires.
NEC5 = {
    "": 80.085 + 39.074j,
    "LD 2,1,0,0,4.,5.E-7,0.": 124.62 + 329.36j,
    "LD 3,1,0,0,300.,2.E-6,0.": 779.49 + 204.46j,
}

# Measured on that run: the twin agrees to 2.4e-5 relative UNLOADED at this
# rung, and to 9.7e-4 / 9.2e-4 with the two cards present.  The gate is 0.5 %
# — five times the measured worst case, and still an order of magnitude
# tighter than razor-nec5's own mesh error at this density (momwire#845).
NEC5_GATE = 0.005


@pytest.mark.parametrize("card", list(NEC5))
def test_razor_nec5_matches_the_licensed_binary_with_the_card_present(card):
    z_mw = _serve.serve(parse_nec5(nec5_deck(card)), basis="razor-nec5")
    z_mw = z_mw.sources[0].impedance
    z_oracle = NEC5[card]
    rel = abs(z_mw - z_oracle) / abs(z_oracle)
    assert rel < NEC5_GATE, f"{card!r}: momwire={z_mw}, NEC-5={z_oracle}, {rel=}"


@pytest.mark.parametrize("card", ["LD 2,1,0,0,4.,5.E-7,0.", "LD 3,1,0,0,300.,2.E-6,0."])
def test_the_loading_delta_itself_matches_the_licensed_binary(card):
    """The sharper form: the twin's own basis gap is common to both runs, so
    the DELTA the card makes is what this issue actually changed.  Measured
    1.2e-3 (LD 2) and 1.0e-3 (LD 3) relative."""
    bare = _serve.serve(parse_nec5(nec5_deck()), basis="razor-nec5")
    loaded = _serve.serve(parse_nec5(nec5_deck(card)), basis="razor-nec5")
    delta_mw = loaded.sources[0].impedance - bare.sources[0].impedance
    delta_oracle = NEC5[card] - NEC5[""]
    rel = abs(delta_mw - delta_oracle) / abs(delta_oracle)
    assert rel < NEC5_GATE, f"{card!r}: momwire={delta_mw}, NEC-5={delta_oracle}"


# --------------------------------------------------------------------------
# gate 6: the printout's loading table
# --------------------------------------------------------------------------


def _loading_table_block(lines: list[str]) -> list[str]:
    """The heading through the last data row — NOT ``ALLOCATE CM:``, which is
    the matrix size and basis-dependent, so it is not part of what this
    FORMAT gate answers for."""
    start = next(
        i for i, line in enumerate(lines) if "STRUCTURE IMPEDANCE LOADING" in line
    )
    end = next(i for i, line in enumerate(lines) if line.startswith("ALLOCATE CM:"))
    return lines[start:end]


def test_the_printout_emits_per_metre_rows_byte_for_byte():
    """gate 6: a FORMAT comparison, so byte-exactness is the right bar —
    verified against our licensed materials on
    ``tests/fixtures/eznec_ld23_1088/synthetic_ld23.{nec,out}``, a small deck
    of our own carrying one ``LD 2`` row and one ``LD 3`` row."""
    deck_text = (FIXTURES / "synthetic_ld23.nec").read_text()
    oracle = (FIXTURES / "synthetic_ld23.out").read_text().splitlines()
    rendered = render(deck_text).splitlines()
    assert _loading_table_block(rendered) == _loading_table_block(oracle)


def test_the_two_type_texts_are_the_engines_own():
    """The TYPE column distinguishes the two cards, and the 20-column field
    reproduces both spellings exactly — ``PARALLEL (PER METER)`` fills it and
    ``SERIES (PER METER)`` lands one column in with one blank to spare."""
    rows = _loading_table_block(
        render((FIXTURES / "synthetic_ld23.nec").read_text()).splitlines()
    )
    body = [line for line in rows if "E+" in line or "E-" in line]
    assert any(line.endswith("SERIES (PER METER) ") for line in body)
    assert any(line.endswith("PARALLEL (PER METER)") for line in body)


def test_a_zero_field_prints_blank_rather_than_a_zero():
    """Measured: an ``LD 2`` whose only nonzero field is the inductance
    prints its RESISTANCE and CAPACITANCE cells EMPTY.  This is the
    antennaknobs shape, so it is the row a catalog design produces."""
    text = NEC5_TWO_WIRES.format(ld="LD 2,0,0,0,0.,2.5E-07,0.")
    rows = [
        line
        for line in _loading_table_block(render(text).splitlines())
        if "2.5000E-07" in line
    ]
    (row,) = rows
    # Exactly one number in the whole row: the RESISTANCE and CAPACITANCE
    # cells the card wrote as `0.` are BLANK, not `0.0000E+00`.
    assert row.count("E-") + row.count("E+") == 1, row
    assert row.endswith("SERIES (PER METER) ")
    # ITAG blank for the whole-structure spelling; the range RESOLVED (the
    # card spelled it `0,0`), FROM ending in column 13 and THRU in 18 — the
    # same three integer cells every other row shape uses.
    assert row[:18] == " " * 12 + "1" + " " * 3 + "18"


def test_a_deck_whose_only_load_is_per_metre_is_not_called_unloaded():
    text = NEC5_TWO_WIRES.format(ld="LD 3,1,0,0,300.,1.E-6,0.")
    rendered = render(text)
    assert "THIS STRUCTURE IS NOT LOADED" not in rendered
    assert "PARALLEL (PER METER)" in rendered


def test_the_ld5_row_shape_is_untouched():
    """The per-metre shape is a THIRD write statement; the ``LD 5`` WIRE row
    and the ``LD 4`` fixed-impedance row must render exactly as before."""
    fixture = Path(__file__).parent / "fixtures" / "eznec_ld5_1082"
    oracle = (fixture / "synthetic_ld5.out").read_text().splitlines()
    rendered = render((fixture / "synthetic_ld5.nec").read_text()).splitlines()
    assert _loading_table_block(rendered) == _loading_table_block(oracle)
