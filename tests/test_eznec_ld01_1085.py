"""``LD 0`` / ``LD 1`` (series / parallel RLC) and antennaknobs' explicit-end
``LD`` addressing on the ``nec5`` (EZNEC) dialect — momwire#1085.

The nec2 dialect has read ``LD`` types 0, 1, 4, 5 and ranges since before
this repo's nec5 front-end existed; the nec5 reader lagged behind on all of
it, each refusal a capture-census absence hardened into a rule (EZNEC never
writes ``LD 0``/``LD 1`` — it reduces a lumped load to a frequency-evaluated
``LD 4`` before writing the deck — so the 80-capture corpus never exercised
either type).  antennaknobs' own NEC-5 writer does write them, and it
addresses every discrete load — ``LD 0``, ``LD 1`` and ``LD 4`` alike —
through an EXPLICIT end code (the manual's LDTAGT field, the same
(segment, end) pair its ``EX`` cards use), not EZNEC's sign-of-LDTAGF
spelling.

The issue named this "a ranged LD 4"; it is not.  The NEC-5 Users Manual's
LD section defines a discrete load as a single point (a range is one LD
card per element), and the momwire#1085 probe
(this file's gate 2) confirms it against our licensed materials: an
explicit end code addresses exactly the one node the manual's arithmetic
gives it (end 1 of segment *s* is node *s-1*, end 2 is node *s*), never a
range.

Five gates, matching the issue's requirements:

1. LD 0 / LD 1 reach the SOLVE as an impedance evaluated at the deck's own
   frequency — proven against the SAME solve done through an equivalent
   ``LD 4`` whose R+jX was computed at that frequency independently, in
   this test, from the same :class:`~momwire.deck.model.LoadSpec` class the
   reader itself reuses (never a second evaluator).
2. antennaknobs' explicit end-code addressing (LDTAGT 1 or 2) decodes to
   the node the momwire#1085 probe measured, proven as a SAME-PARSE
   identity against EZNEC's own sign-of-LDTAGF spelling of the identical
   node (the same pattern the GN NOFILE-trailer test uses for two spellings
   of one ground).
3. LD 2 / LD 3 still refuse by name (momwire#1088, not this issue).
4. The printout's loading table gets a SERIES and a PARALLEL row, and an
   explicit-end LD 4 row, byte-compared against our licensed materials on a
   deck of our own that ALSO gates the rows' DECK ORDER against an LD 5 row
   in the same table (``tests/fixtures/eznec_ld01_1085/``, a directory of
   its own so the 80-capture corpus and its manifest stay untouched).
5. A zero-valued LD 0/1/4 card is a no-op (dropped), the same rule the
   nec2 dialect's own ``_load_spec`` applies and for the same reason: a
   zero impedance stamped in series with an already-perfect conductor
   changes nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from momwire.deck import DeckError
from momwire.deck._nec5 import Nec5Node, parse_nec5
from momwire.deck.model import LoadSpec
from momwire.eznec import _serve
from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_ld01_1085"

FREQUENCY_MHZ = 29.97925

DIPOLE = """CM ld01 gate test dipole
CE
GW 1,9,0.,-.5,0.,0.,.5,0.,.0005
GE 0,-1
GN -1
FR 0,1,0,0,{freq}
EX 4,1,5,0,1.414214,0.
PQ 0
RP 0,1,361,1000,90.,0.,0.,1.,0.
EN
""".format(freq=FREQUENCY_MHZ)


def _with_card(card: str) -> str:
    """``DIPOLE`` with one extra card inserted just before ``EN``."""
    lines = DIPOLE.splitlines()
    at = lines.index("EN")
    return "\n".join([*lines[:at], card, "EN"]) + "\n"


def _with_cards(*cards: str) -> str:
    lines = DIPOLE.splitlines()
    at = lines.index("EN")
    return "\n".join([*lines[:at], *cards, "EN"]) + "\n"


# --------------------------------------------------------------------------
# gate 1: parsing -- LD 0 / LD 1 read to a symbolic LoadSpec
# --------------------------------------------------------------------------


def test_ld_0_reads_a_series_spec():
    (load,) = parse_nec5(_with_card("LD 0,1,4,0,50.,1.E-6,1.E-10")).loads
    assert load.spec == LoadSpec("series", r=50.0, l=1e-6, c=1e-10)
    assert load.at == Nec5Node(tag=1, node=4)


def test_ld_1_reads_a_parallel_spec():
    (load,) = parse_nec5(_with_card("LD 1,1,4,0,75.,2.E-6,5.E-11")).loads
    assert load.spec == LoadSpec("parallel", r=75.0, l=2e-6, c=5e-11)


def test_ld_0_needs_all_three_rlc_fields():
    with pytest.raises(DeckError, match="needs at least 7"):
        parse_nec5(_with_card("LD 0,1,4,0,50.,1.E-6"))


def test_ld_2_still_refuses_by_name():
    with pytest.raises(DeckError, match="momwire#1088"):
        parse_nec5(_with_card("LD 2,1,1,9,1.,0.,0."))


def test_ld_3_still_refuses_by_name():
    with pytest.raises(DeckError, match="momwire#1088"):
        parse_nec5(_with_card("LD 3,1,1,9,1.,0.,0."))


# --------------------------------------------------------------------------
# gate 5: a zero-valued load is a no-op, dropped -- the nec2 sibling's rule
# --------------------------------------------------------------------------


def test_a_zero_series_load_is_a_no_op():
    deck = parse_nec5(_with_card("LD 0,1,4,0,0.,0.,0."))
    assert deck.loads == ()


def test_a_zero_fixed_load_is_a_no_op():
    deck = parse_nec5(_with_card("LD 4,1,4,0,0.,0."))
    assert deck.loads == ()


def test_a_partly_zero_series_load_is_not_a_no_op():
    """Only ALL-zero drops; R alone (L, C absent/zero) is a real load."""
    deck = parse_nec5(_with_card("LD 0,1,4,0,50.,0.,0."))
    assert deck.loads != ()


# --------------------------------------------------------------------------
# gate 2: antennaknobs' explicit end-code addressing
# --------------------------------------------------------------------------


def test_explicit_end_2_is_the_same_node_as_the_bare_positive_spelling():
    explicit = parse_nec5(_with_card("LD 4,1,5,2,10.,20.")).loads[0].at
    implicit = parse_nec5(_with_card("LD 4,1,5,0,10.,20.")).loads[0].at
    assert explicit == implicit == Nec5Node(tag=1, node=5)


def test_explicit_end_1_is_one_node_short_of_the_segment():
    """Segment 5 end 1 is node 4 -- verified against our licensed materials
    (momwire#1085 probe): `LD 4,1,5,1,R,X` solves bit-identical to
    `LD 4,1,4,0,R,X` on the same 9-segment geometry."""
    node = parse_nec5(_with_card("LD 4,1,5,1,10.,20.")).loads[0].at
    assert node == Nec5Node(tag=1, node=4)
    explicit = _serve.serve(parse_nec5(_with_card("LD 4,1,5,1,10.,20.")))
    implicit = _serve.serve(parse_nec5(_with_card("LD 4,1,4,0,10.,20.")))
    assert explicit.sources[0].impedance == implicit.sources[0].impedance


def test_explicit_end_1_of_segment_1_is_node_0():
    """The vertex case: segment 1 end 1 is node 0 -- EZNEC's own `-1`
    spelling of the identical node.  A PARSE-only check: node 0 of this
    fixture's single free-standing wire is a dangling end with nothing else
    attached, and momwire's own solver refuses a load THERE regardless of
    how it is addressed (no current path past a lone conductor end) -- a
    pre-existing, address-independent constraint, not something this
    fixture's geometry is built to exercise.  The momwire#1085 probe
    (this module's docstring) is what settles the SOLVE-level bit-identity
    on a deck where node 0 is NOT a free end."""
    explicit = parse_nec5(_with_card("LD 4,1,1,1,10.,20.")).loads[0].at
    eznec_spelling = parse_nec5(_with_card("LD 4,1,-1,0,10.,20.")).loads[0].at
    assert explicit == eznec_spelling == Nec5Node(tag=1, node=0)


def test_explicit_end_applies_to_ld_0_and_ld_1_too():
    """antennaknobs' writer uses the SAME (segment, end) pair for every
    discrete load type -- `engines/nec5.py`'s `_source_address` feeds both
    the `LD 4` and the `LD 0`/`LD 1` branch through one `where` string."""
    explicit = _serve.serve(parse_nec5(_with_card("LD 0,1,5,1,50.,1.E-6,1.E-10")))
    implicit = _serve.serve(parse_nec5(_with_card("LD 0,1,4,0,50.,1.E-6,1.E-10")))
    assert explicit.sources[0].impedance == implicit.sources[0].impedance


def test_an_end_code_outside_zero_one_two_refuses_by_name():
    """Not a range: the manual's LD section defines a discrete load as a
    single point (a range is one LD card per element), and this is the
    refusal that holds that line -- gate 2's negative case."""
    with pytest.raises(DeckError, match="LDTAGT"):
        parse_nec5(_with_card("LD 4,1,2,5,10.,20."))


def test_an_explicit_end_code_needs_a_positive_segment():
    with pytest.raises(DeckError, match="explicit"):
        parse_nec5(_with_card("LD 4,1,-1,1,10.,20."))


def test_an_explicit_end_code_names_its_undeclared_tag():
    with pytest.raises(DeckError, match="no GW card"):
        parse_nec5(_with_card("LD 4,9,1,1,10.,20."))


def test_an_explicit_end_code_bounds_its_segment():
    with pytest.raises(DeckError, match="9 segments"):
        parse_nec5(_with_card("LD 4,1,10,1,10.,20."))


# --------------------------------------------------------------------------
# gate 1: the solve -- LD 0 / LD 1 evaluate at the deck's own frequency
# --------------------------------------------------------------------------


def test_series_load_agrees_with_the_equivalent_fixed_impedance():
    spec = LoadSpec("series", r=50.0, l=1e-6, c=1e-10)
    z = spec.impedance(FREQUENCY_MHZ * 1e6)

    series = _serve.serve(parse_nec5(_with_card("LD 0,1,4,0,50.,1.E-6,1.E-10")))
    fixed = _serve.serve(
        parse_nec5(_with_card(f"LD 4,1,4,0,{z.real:.12E},{z.imag:.12E}"))
    )
    z_series = series.sources[0].impedance
    z_fixed = fixed.sources[0].impedance
    assert z_series == pytest.approx(z_fixed, rel=1e-12)


def test_parallel_load_agrees_with_the_equivalent_fixed_impedance():
    spec = LoadSpec("parallel", r=75.0, l=2e-6, c=5e-11)
    z = spec.impedance(FREQUENCY_MHZ * 1e6)

    parallel = _serve.serve(parse_nec5(_with_card("LD 1,1,4,0,75.,2.E-6,5.E-11")))
    fixed = _serve.serve(
        parse_nec5(_with_card(f"LD 4,1,4,0,{z.real:.12E},{z.imag:.12E}"))
    )
    z_parallel = parallel.sources[0].impedance
    z_fixed = fixed.sources[0].impedance
    assert z_parallel == pytest.approx(z_fixed, rel=1e-12)


def test_series_load_impedance_is_frequency_dependent():
    """Loaded R_in must differ between two frequencies -- proof the value
    reaching the solve is not a single frozen-at-parse-time number."""
    low = _with_cards("LD 0,1,4,0,50.,1.E-6,1.E-10").replace(
        f"FR 0,1,0,0,{FREQUENCY_MHZ}", "FR 0,1,0,0,14.15"
    )
    high = _with_cards("LD 0,1,4,0,50.,1.E-6,1.E-10").replace(
        f"FR 0,1,0,0,{FREQUENCY_MHZ}", "FR 0,1,0,0,29.7"
    )
    z_low = _serve.serve(parse_nec5(low)).sources[0].impedance
    z_high = _serve.serve(parse_nec5(high)).sources[0].impedance
    assert z_low != z_high


# --------------------------------------------------------------------------
# gate 4: the printout's loading table
# --------------------------------------------------------------------------


def _loading_table_block(lines: list[str]) -> list[str]:
    """The heading through the last data row -- NOT ``ALLOCATE CM:``, which
    is the matrix size and basis-dependent, so it is not part of what this
    FORMAT gate answers for."""
    start = next(
        i for i, line in enumerate(lines) if "STRUCTURE IMPEDANCE LOADING" in line
    )
    end = next(i for i, line in enumerate(lines) if line.startswith("ALLOCATE CM:"))
    return lines[start:end]


def test_the_printout_emits_series_parallel_and_wire_rows_in_deck_order():
    """gate 4: a FORMAT comparison, byte-exact -- verified against our
    licensed materials on `tests/fixtures/eznec_ld01_1085/synthetic_ld01.
    {nec,out}`, a small deck of our own mixing LD 4 (explicit end 1), LD 0,
    LD 1 (explicit end 2) and LD 5 in that order, kept out of
    `tests/fixtures/eznec/` so the 80-capture corpus and its manifest
    (`test_eznec_reproducibility.py`) stay untouched.  The row ORDER is the
    other half of the gate: momwire#1085's probe found the licensed engine
    prints its rows in DECK order across every LD type, not grouped by
    type, which this fixture is the only deck in this tree that exercises."""
    deck_text = (FIXTURES / "synthetic_ld01.nec").read_text()
    oracle_lines = (FIXTURES / "synthetic_ld01.out").read_text().splitlines()
    rendered_lines = render(deck_text).splitlines()
    assert _loading_table_block(rendered_lines) == _loading_table_block(oracle_lines)


def test_the_printout_still_says_not_loaded_with_no_load_cards():
    text = DIPOLE
    rendered = render(text)
    assert "THIS STRUCTURE IS NOT LOADED" in rendered


def test_a_zero_component_prints_blank_not_zero():
    """Measured against our licensed materials (momwire#1085 probe): a zero
    R, L or C prints blank, the same convention LD 4's IMAGINARY column
    already follows for a zero reactance."""
    rendered = render(_with_card("LD 0,1,4,0,50.,0.,1.E-10"))
    lines = rendered.splitlines()
    (row,) = [line for line in _loading_table_block(lines) if "SERIES" in line]
    assert "0.0000E+00" not in row
    assert "5.0000E+01" in row  # R, ending at column 34
    assert "1.0000E-10" in row  # C, ending at column 60
    assert row[23:34].strip() == "5.0000E+01"  # R cell
    assert row[36:47].strip() == ""  # L's cell: blank, not a zeroed E-field
    assert row[49:60].strip() == "1.0000E-10"  # C cell
