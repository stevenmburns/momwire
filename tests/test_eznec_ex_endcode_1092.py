"""``EX`` honours antennaknobs' own explicit-end connection addressing on
the ``nec5`` (EZNEC) dialect, the same way #1085 gave it to ``LD`` --
momwire#1092.

``TL`` and ``NT`` do NOT get this fix, and cannot: both cards address TWO
nodes and their field layout matches NEC-2's exactly (two two-field
addresses back to back, then the card's own data), so the field right
after the FIRST address's node is the SECOND address's tag, not a spare
slot. There is no field on either card antennaknobs' writer could spell
an end code into, and (as of this issue) antennaknobs' own NEC-5 writer
never emits a ``TL`` or ``NT`` card at all -- momwire#1092's own
investigation found no consumer to gate on that side.

Gates, matching the issue:

1. ``EX 0,tag,1,1,...`` (segment 1, end 1) decodes to node 0, identically
   to EZNEC's own ``EX 0,tag,-1,0,...``; ``EX 0,tag,k,2,...`` decodes to
   node k, identically to ``EX 0,tag,k,0,...``; an end code outside
   {0, 1, 2} refuses by name -- the same-parse identity pattern
   test_eznec_ld01_1085.py already uses for ``LD``.
2. Every capture in tests/fixtures/eznec/decks/ still decodes its ``EX``
   cards with end code 0 (unaffected) -- the corpus is the record of what
   this fix must NOT move.
3. The AK gate: a vertex-fed catalog design (dipoles.invvee_apex, AK#898),
   its geometry banked in tests/fixtures/eznec_endcode_1092/invvee_apex.nec
   (antennaknobs' own ``NEC5Engine`` export -- captured once, not imported
   at test time: momwire#988 refuses a test gated behind
   ``importorskip("antennaknobs")``, since no CI lane anywhere installs
   both repos). Served through momwire on the ``razor-nec5`` basis (the
   NEC-5 formulation twin), it agrees with the licensed engine's own
   printout on that export, and with the SAME geometry solved through
   momwire's own ``node_gaps`` port (#305's series node gap, the primitive
   ``PortAtVertex`` compiles to) on the same basis -- both to the
   tolerances this file measures and states.
4. The printout: the ANTENNA INPUT PARAMETERS table's TAG / SEG. / end-
   digit columns for an explicit-end EX card, byte-compared against our
   licensed materials on a deck of our own
   (tests/fixtures/eznec_endcode_1092/), kept out of
   tests/fixtures/eznec/ so the 80-capture corpus and its manifest
   (test_eznec_reproducibility.py) stay untouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from momwire.deck import DeckError
from momwire.deck._nec5 import Nec5Node, parse_nec5
from momwire.eznec import _serve
from momwire.eznec._shell import render

DECKS = Path(__file__).parent / "fixtures" / "eznec" / "decks"
CORPUS = sorted(DECKS.glob("*.nec"))
FIXTURES = Path(__file__).parent / "fixtures" / "eznec_endcode_1092"

FREQUENCY_MHZ = 29.97925

DIPOLE = """CM a dipole
CE
GW 1,9,0.,-.5,0.,0.,.5,0.,.0005
GE 0,-1
GN -1
FR 0,1,0,0,{freq}
EX {ex}
PQ 0
EN
""".format(freq=FREQUENCY_MHZ, ex="{ex}")


def _with_ex(ex: str) -> str:
    return DIPOLE.format(ex=ex)


# --------------------------------------------------------------------------
# gate 1: the explicit end code, same-parse identity against EZNEC's own
# --------------------------------------------------------------------------


def test_explicit_end_2_is_the_same_node_as_the_bare_positive_spelling():
    explicit = parse_nec5(_with_ex("4,1,5,2,10.,20.")).sources[0].at
    implicit = parse_nec5(_with_ex("4,1,5,0,10.,20.")).sources[0].at
    assert explicit == implicit == Nec5Node(tag=1, node=5)


def test_explicit_end_1_of_segment_1_is_node_0_like_eznecs_minus_1():
    """The vertex case (AK#898's apex feed): segment 1 end 1 is node 0 --
    EZNEC's own ``-1`` spelling of the identical node. A PARSE-only check
    here (this fixture's node 0 is a free wire end with nothing else
    attached, and momwire's solver refuses a source there regardless of
    how it is addressed, the same pre-existing constraint
    test_eznec_ld01_1085.py's sibling test notes for ``LD``); the AK gate
    below settles the SOLVE-level agreement on a design where node 0 is a
    real junction."""
    explicit = parse_nec5(_with_ex("0,1,1,1,1.,0.")).sources[0].at
    eznec_spelling = parse_nec5(_with_ex("0,1,-1,0,1.,0.")).sources[0].at
    assert explicit == eznec_spelling == Nec5Node(tag=1, node=0)


def test_explicit_end_1_is_one_node_short_of_the_segment():
    """Segment 6 end 1 is node 5 on this 9-segment wire."""
    node = parse_nec5(_with_ex("4,1,6,1,10.,20.")).sources[0].at
    assert node == Nec5Node(tag=1, node=5)


def test_end_code_defaults_to_the_deck_field_and_reads_zero_for_every_capture_shape():
    src = parse_nec5(_with_ex("4,1,6,0,10.,20.")).sources[0]
    assert src.end_code == 0
    explicit = parse_nec5(_with_ex("4,1,6,2,10.,20.")).sources[0]
    assert explicit.end_code == 2


def test_an_end_code_outside_zero_one_two_refuses_by_name():
    with pytest.raises(DeckError, match="end-code field"):
        parse_nec5(_with_ex("4,1,2,5,10.,20."))


def test_an_explicit_end_code_needs_a_positive_segment():
    with pytest.raises(DeckError, match="explicit"):
        parse_nec5(_with_ex("4,1,-1,1,10.,20."))


def test_an_explicit_end_code_names_its_undeclared_tag():
    with pytest.raises(DeckError, match="no GW card"):
        parse_nec5(_with_ex("4,9,1,1,10.,20."))


def test_an_explicit_end_code_bounds_its_segment():
    with pytest.raises(DeckError, match="9 segments"):
        parse_nec5(_with_ex("4,1,10,1,10.,20."))


def test_the_old_negative_and_zero_refusals_are_unmoved_by_end_code_zero():
    """end_code 0 delegates to `_address` unchanged -- the same refusals
    `test_deck_nec5.py`'s corpus-vocabulary parametrize already pins,
    reasserted here beside the new explicit-end gates."""
    with pytest.raises(DeckError, match="-1 is the only negative"):
        parse_nec5(_with_ex("4,1,-2,0,1.,0."))
    with pytest.raises(DeckError, match="this dialect spells node 0"):
        parse_nec5(_with_ex("4,1,0,0,1.,0."))


# --------------------------------------------------------------------------
# gate 2: the corpus is unmoved -- every captured EX still reads end_code 0
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.stem)
def test_every_captured_ex_card_still_reads_end_code_zero(path: Path):
    deck = parse_nec5(path.read_text())
    for source in deck.sources:
        assert source.end_code == 0


# --------------------------------------------------------------------------
# requirement 6: TL / NT have no end-code field -- a regression, not a fix
# --------------------------------------------------------------------------

TWO_WIRE = """CM two wires for TL/NT addressing
CE
GW 1,9,0.,-.5,0.,0.,.5,0.,.0005
GW 2,9,10.,-.5,0.,10.,.5,0.,.0005
GE 0,-1
GN -1
FR 0,1,0,0,{freq}
{{card}}
EX 4,1,5,0,1.414214,0.
PQ 0
EN
""".format(freq=FREQUENCY_MHZ)


def _with_network_card(card: str) -> str:
    return TWO_WIRE.format(card=card)


def test_tl_has_no_end_code_field_the_second_address_owns_it():
    """``TL``'s layout is tag_a, node_a, tag_b, node_b, Z0, length, ... --
    the field right after the FIRST address's node is the SECOND
    address's own tag, not a spare end-code slot. A tag of 1 or 2 there
    (exactly the values `_end_coded_address` treats specially on ``EX``
    and ``LD``) reads as a tag, never as an end code -- confirming
    momwire#1092's finding that ``TL`` has no field to carry
    antennaknobs' explicit-end spelling into."""
    (line,) = parse_nec5(
        _with_network_card("TL 1,5,2,5,50.,3.048,0.,0.,0.,0.")
    ).transmission_lines
    assert line.end_a == Nec5Node(tag=1, node=5)
    assert line.end_b == Nec5Node(tag=2, node=5)


def test_nt_has_no_end_code_field_either():
    (net,) = parse_nec5(_with_network_card("NT 1,5,2,5,.01,0.,0.,0.,0.,0.")).networks
    assert net.end_a == Nec5Node(tag=1, node=5)
    assert net.end_b == Nec5Node(tag=2, node=5)


# --------------------------------------------------------------------------
# gate 4: the printout's address columns, byte-gated
# --------------------------------------------------------------------------


def _antenna_input_rows(lines: list[str]) -> list[str]:
    start = (
        next(i for i, line in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in line)
        + 4
    )  # heading, blank, the two-line column header
    rows = []
    for line in lines[start:]:
        if not line.strip():
            break
        rows.append(line)
    return rows


def test_the_printout_prints_the_written_segment_and_the_inverse_end_digit():
    """gate 4: byte-exact on the TAG / SEG. / end-digit columns (the first
    12 characters of each row -- `_port_row`'s own field widths) against
    our licensed materials on
    tests/fixtures/eznec_endcode_1092/synthetic_ex_endcode.{nec,out}, a
    small deck of our own mixing EZNEC's sign-of-the-node-field spelling
    with both antennaknobs explicit end codes. The voltage/current/
    impedance cells are NOT gated here -- they are a solve, and momwire's
    own solver reads these decks to different numbers than the licensed
    engine's by design (see this fixture's README)."""
    deck_text = (FIXTURES / "synthetic_ex_endcode.nec").read_text()
    oracle_lines = (FIXTURES / "synthetic_ex_endcode.out").read_text().splitlines()
    rendered_lines = render(deck_text).splitlines()
    oracle_rows = [row[:12] for row in _antenna_input_rows(oracle_lines)]
    rendered_rows = [row[:12] for row in _antenna_input_rows(rendered_lines)]
    assert (
        rendered_rows == oracle_rows == ["   1     6 1", "   2    16 2", "   3    23 1"]
    )


# --------------------------------------------------------------------------
# gate 3: the AK cross-engine gate
# --------------------------------------------------------------------------


def test_the_vertex_feed_decodes_to_the_shared_apex_node():
    """`EX 0 1 1 1 ...` -- segment 1 end 1 of the fed arm -- is node 0, the
    apex both wires share. Before momwire#1092 this decoded to node 1, one
    knot in from the apex (measured: served Z moved from
    54.036-14.823j to 54.353-14.916j on the razor-nec5 basis, a change an
    order of magnitude bigger than this design's own solver-to-solver
    agreement -- see the cross-engine gate below)."""
    deck = parse_nec5((FIXTURES / "invvee_apex.nec").read_text())
    (source,) = deck.sources
    assert source.at == Nec5Node(tag=1, node=0)
    assert source.end_code == 1
    assert source.printed_location == 1


def test_served_agrees_with_the_licensed_engine_on_the_export():
    """The licensed engine's own ANTENNA INPUT PARAMETERS row for this
    export (tests/fixtures/eznec_endcode_1092/invvee_apex.out) is
    5.4038E+01 -1.4823E+01 (54.038-14.823j). momwire's `razor-nec5` solve
    of the SAME export -- the NEC-5 formulation twin, chosen because it is
    the basis measured against the licensed engine's own tent-basis
    arithmetic -- lands at 54.036-14.823j, within 0.01 ohm."""
    deck = parse_nec5((FIXTURES / "invvee_apex.nec").read_text())
    served = _serve.serve(deck, basis="razor-nec5")
    z = served.sources[0].impedance
    licensed = complex(54.038, -14.823)
    assert abs(z - licensed) < 0.01


def test_served_agrees_with_the_native_ak_solve_on_the_same_basis():
    """Gate 3's other leg: the SAME apex geometry solved through momwire's
    OWN ``node_gaps`` port (#305's series node gap, "the apex feed" --
    ``test_node_gaps.py``) instead of an ``EX`` card, on the same
    ``razor-nec5`` basis, compared against the served ``NEC5Engine``
    export.

    This reconstructs antennaknobs' ``dipoles.invvee_apex`` (AK#898) by
    hand from the geometry already banked in ``invvee_apex.nec`` --
    ``momwire#1092`` deliberately does NOT import antennaknobs here
    (momwire#988: a test gated behind ``importorskip("antennaknobs")``
    runs in no CI lane anywhere, since neither repo's CI installs the
    other -- momwire's CLAUDE.md, "Two repos, one working tree"). Wire 0
    is tag 1's ``GW`` (its own ``start`` is node 0, the fed vertex), wire
    1 is tag 2's, and the two share that point as a junction -- the same
    "both arms meet at one point" shape ``invvee_apex.py``'s own docstring
    describes for ``PortAtVertex``.

    The two feed models (a series node gap vs. an ``EX`` card reading the
    same exported deck) are different circuit topologies that are
    supposed to agree physically, not two readings of one deck, so the
    tolerance here is wider than the byte-level export/native-parser
    agreement a same-basis round trip usually gets -- measured at
    0.05 ohm, an order of magnitude tighter than the ~0.3 ohm the
    pre-#1092 wrong-node reading was off by.
    """
    from momwire.razor import RazorSolver

    deck = parse_nec5((FIXTURES / "invvee_apex.nec").read_text())
    served = _serve.serve(deck, basis="razor-nec5")
    z_served = served.sources[0].impedance

    wires = deck.wires
    wavelength = 299.792458 / deck.frequency_mhz
    solver = RazorSolver(
        wires=[
            np.array([wires[0].end1, wires[0].end2]),
            np.array([wires[1].end1, wires[1].end2]),
        ],
        feeds=[],
        node_gaps=[(0, "start", 1.0 + 0j)],
        junctions=[[(0, "start"), (1, "end")]],
        n_per_edge_per_wire=[[wires[0].segment_count], [wires[1].segment_count]],
        wavelength=wavelength,
        wire_radius=wires[0].radius,
        nec5_quadrature=True,
    )
    z_native, _ = solver.compute_impedance()
    z_native = complex(np.atleast_1d(z_native)[0])
    assert abs(z_served - z_native) < 0.05
