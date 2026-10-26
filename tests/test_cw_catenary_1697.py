"""CW, the catenary wire, on the nec5 and nec4 dialects — momwire#1697.

Every probe deck under ``tests/fixtures/nec_cw_catenary_1697`` (README there)
was run once on each licensed engine as a black box, and these tests hold
momwire to both printouts:

* the NEC-5 STRUCTURE SPECIFICATION block, the CW echo and its "Catenary
  length" line included, byte for byte;
* the NEC-4.2 wire table, its catenary entry included, byte for byte;
* every NEC-4.2 segment centre and length, to that printout's precision;
* the feed impedance, razor-2p against NEC-5 and the sinusoidal basis
  against NEC-4.2, within tolerances measured on these decks;
* and the forms both dialects refuse, each by name.

The curve itself is ``deck._nec4.catenary_vertices``, the construction
antennaknobs' importer uses (`nec_import._cw`).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from momwire.deck import DeckError, parse
from momwire.deck._cards import parse_card
from momwire.deck._nec4 import catenary_length, catenary_vertices
from momwire.deck._nec5 import parse_nec5
from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures" / "nec_cw_catenary_1697"
PROBES = sorted(p.stem for p in FIXTURES.glob("*.nec"))
CW_PROBES = [p for p in PROBES if p.startswith("c")]


def deck(name: str) -> str:
    return (FIXTURES / f"{name}.nec").read_text()


def oracle(name: str, engine: str) -> str:
    return (FIXTURES / f"{name}.{engine}.out").read_text()


def _block(text: str, start: str, end: str) -> list[str]:
    lines = text.splitlines()
    i = next(k for k, line in enumerate(lines) if start in line)
    j = next(k for k, line in enumerate(lines) if k > i and end in line)
    return lines[i:j]


def _nec5_specification(text: str) -> list[str]:
    return _block(text, "Model definition commands", "***** INPUT LINE")


def _nec42_wire_table(text: str) -> list[str]:
    return _block(text, "STRUCTURE SPECIFICATION", "TOTAL SEGMENTS USED")


def _nec42_segments(text: str) -> list[list[float]]:
    """The SEGMENTATION DATA rows' centre and length."""
    lines = text.splitlines()
    start = next(
        k for k, line in enumerate(lines) if "COORDINATES OF SEG. CENTER" in line
    )
    rows = []
    for line in lines[start + 2 :]:
        cells = line.split()
        if len(cells) < 8 or not cells[0].isdigit():
            if rows:
                break
            continue
        rows.append([float(c) for c in cells[1:5]])
    return rows


def _impedance(text: str) -> complex:
    lines = text.splitlines()
    start = next(
        k for k, line in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in line
    )
    for line in lines[start + 1 :]:
        values = re.findall(r"-?\d\.\d+E[+-]\d+", line)
        if len(values) >= 6:
            return complex(float(values[4]), float(values[5]))
    raise AssertionError("no input parameters row")


def test_the_probes_are_there():
    assert len(PROBES) == 8, PROBES


@pytest.mark.parametrize("name", PROBES)
def test_the_nec5_structure_specification_is_the_engines(name):
    """The CW echo (seven reals, ICAT, RHM, ZM) and its "Catenary length ...
    Total segment length = 0" line, the other cards' echoes and the counts."""
    assert _nec5_specification(render(deck(name))) == _nec5_specification(
        oracle(name, "nec5")
    )


@pytest.mark.parametrize("name", PROBES)
def test_the_nec42_wire_table_is_the_engines(name):
    """THE FOLLOWING WIRE IS A CATENARY, its lengths line (the chords' total
    as the second number), then the card's row as a GW's."""
    assert _nec42_wire_table(render(deck(name), dialect="nec4")) == _nec42_wire_table(
        oracle(name, "nec42")
    )


@pytest.mark.parametrize("name", PROBES)
def test_every_nec42_segment_centre_and_length_is_the_engines(name):
    """F10.5 cells, so a row agrees to half a unit in its last place, plus
    the engine's own rounding of the curve (one unit: a centre the engine
    prints as -0.00000 or -0.00001 sits at 0 here, c6 and c7)."""
    ours = _nec42_segments(render(deck(name), dialect="nec4"))
    theirs = _nec42_segments(oracle(name, "nec42"))
    assert len(ours) == len(theirs)
    for mine, engine in zip(ours, theirs, strict=True):
        for a, b in zip(mine, engine, strict=True):
            assert abs(a - b) <= 1.5e-5, (name, mine, engine)


@pytest.mark.parametrize("name", CW_PROBES)
def test_the_curve_length_is_what_both_engines_echo(name):
    card = next(
        c
        for c in (parse_card(line) for line in deck(name).splitlines())
        if c is not None and c.mnemonic == "CW"
    )
    for engine in ("nec5", "nec42"):
        echoed = re.search(r"Catenary length =\s*(\S+)", oracle(name, engine))
        assert f"{catenary_length(card):.5E}" == f"{float(echoed.group(1)):.5E}"


# razor-2p is NEC-5's own basis. Measured against the licensed engine on these
# 8 decks (2026-10-25), its worst row is 2.08e-4 of |Z| (c2, fed at a wire end).
RAZOR_RTOL = 3e-4


@pytest.mark.parametrize("name", PROBES)
def test_the_feed_impedance_agrees_with_nec5_on_razor_2p(name):
    z = _impedance(render(deck(name), basis="razor-2p"))
    ref = _impedance(oracle(name, "nec5"))
    assert abs(z - ref) <= RAZOR_RTOL * abs(ref), (name, z, ref)


# The sinusoidal basis against NEC-4.2, measured the same day: 1.07e-3 of |Z|
# on the straight control (g6), 1.1e-3 on the centre-fed catenaries, and
# 4.47e-3 at worst (c1, fed at a wire end).
SINUSOIDAL_RTOL = 6e-3


@pytest.mark.parametrize("name", PROBES)
def test_the_feed_impedance_agrees_with_nec42_on_the_sinusoidal_basis(name):
    z = _impedance(render(deck(name), basis="sinusoidal", dialect="nec4"))
    ref = _impedance(oracle(name, "nec42"))
    assert abs(z - ref) <= SINUSOIDAL_RTOL * abs(ref), (name, z, ref)


def test_a_cw_is_one_wire_of_ns_elements_in_the_nec5_dialect():
    (wire,) = parse_nec5(deck("c1_icat1_sag")).wires
    assert (wire.tag, wire.segment_count, len(wire.path)) == (1, 10, 11)
    assert wire.path[0] == wire.end1 == (-20.0, 0.0, 15.0)
    assert wire.path[-1] == wire.end2 == (20.0, 0.0, 15.0)


def test_a_sag_is_cut_by_arc_length_and_an_arch_by_horizontal_step():
    """What both engines' segment tables show (c1, c2)."""
    import math

    sag = catenary_vertices(parse_card("CW 1 10 -20 0 15 20 0 15 0.001 1 20 8"))
    chords = [math.dist(a, b) for a, b in zip(sag, sag[1:], strict=False)]
    assert max(chords) - min(chords) < 1e-3 * min(chords)
    arch = catenary_vertices(parse_card("CW 1 10 -20 0 15 20 0 15 0.001 2 10 -2"))
    steps = [b[0] - a[0] for a, b in zip(arch, arch[1:], strict=False)]
    assert max(steps) - min(steps) < 1e-12


def test_a_length_no_longer_than_the_line_is_the_line():
    line = catenary_vertices(parse_card("CW 1 4 0 0 0 3 0 4 0.001 3 4 0"))
    assert line == [(0.75 * i, 0.0, 1.0 * i) for i in range(5)]


BODY = "CM cw\nCE\n{cw}\nGE 0\nEX 0 1 1 0 1 0\nFR 0 1 0 0 14 0\nXQ\nEN\n"


@pytest.mark.parametrize(
    "cw, match",
    [
        ("CW 1 10 -20 0 15 20 0 15 0.001 4 20 8", "CW ICAT must be 1"),
        ("CW 1 10 -20 0 15 20 0 15 0.001 1 40 8", "at an end of the wire"),
        ("CW 1 10 -20 0 15 20 0 15 0.001 1 20 15", "on the straight line"),
        ("CW 1 10 0 0 0 0 0 15 0.001 3 20 0", "one above the other"),
        ("CW 1 10 -20 0 15 20 0 15 0 1 20 8", "positive wire radius"),
        ("CW 1 0 -20 0 15 20 0 15 0.001 1 20 8", "segment count must be >= 1"),
    ],
    ids=["icat", "rhm-at-end", "point-on-line", "vertical", "radius", "no-segments"],
)
@pytest.mark.parametrize("dialect", ["nec4", "nec5"])
def test_what_both_dialects_refuse(dialect, cw, match):
    with pytest.raises(DeckError, match=match):
        if dialect == "nec5":
            parse_nec5(BODY.format(cw=cw))
        else:
            parse(BODY.format(cw=cw), dialect="nec4")


def test_the_nec2_dialect_has_no_cw():
    with pytest.raises(DeckError, match="CW"):
        parse(BODY.format(cw="CW 1 10 -20 0 15 20 0 15 0.001 1 20 8"), dialect="nec2")


def test_the_nec42_slot_refuses_a_cw_after_a_gs():
    text = BODY.format(
        cw="GW 2 3 0 0 1 0 0 2 .001\nGS 0 0 .5\nCW 1 10 -20 0 15 20 0 15 0.001 1 20 8"
    )
    printout = render(text, dialect="nec4")
    assert "a CW card after a GS card" in printout
