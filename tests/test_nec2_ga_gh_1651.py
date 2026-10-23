"""GA and GH on the nec2 dialect, against nec2c (momwire#1651).

The nec2 dialect reads ``GA`` as NEC-2's arc and ``GH`` as NEC-2's helix
(``deck._nec2_geometry.arc_vertices`` and ``nec2_helix_vertices``), where the
nec4 dialect reads ``GH`` as NEC-4.2's different card. The reference is
nec2c, the public code: seventeen decks and nec2c's printouts of them under
``tests/fixtures/nec2_ga_gh_1651/`` (README there). Each is served through
the ``momwire-nec2c`` printout in process and compared:

* the STRUCTURE SPECIFICATION block, byte for byte;
* SEGMENTATION DATA, every integer exactly and every real to nec2c's printed
  digits (README: one digit on a rounding half, and the sign of a zero).

The solve is not gated here; the printout's numbers are the portal's, as
for every other card.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from momwire.deck import DeckError, parse
from momwire.portal._portal import file_printout

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "nec2_ga_gh_1651"
DECKS = tuple(sorted(p.stem for p in FIXTURES.glob("*.nec")))

_STRUCTURE = "-------- STRUCTURE SPECIFICATION --------"
_SEGMENTATION = "---------- SEGMENTATION DATA ----------"


def _lines(text: str) -> list[str]:
    return text.replace("\r\n", "\n").split("\n")


def _structure_block(lines: list[str]) -> list[str]:
    start = next(i for i, ln in enumerate(lines) if _STRUCTURE in ln)
    end = next(i for i, ln in enumerate(lines) if _SEGMENTATION in ln)
    return lines[start:end]


def _segment_rows(lines: list[str]) -> list[list[str]]:
    at = next(i for i, ln in enumerate(lines) if _SEGMENTATION in ln)
    head = next(i for i in range(at, len(lines)) if lines[i].startswith("   No:"))
    rows = []
    for line in lines[head + 1 :]:
        if not line.strip():
            break
        rows.append(line.split())
    return rows


@pytest.fixture(scope="module")
def served() -> dict[str, list[str]]:
    return {
        name: _lines(file_printout((FIXTURES / f"{name}.nec").read_text()))
        for name in DECKS
    }


def nec2c(name: str) -> list[str]:
    return _lines((FIXTURES / f"{name}.out").read_text())


def test_the_fixtures_are_the_readme_s():
    assert len(DECKS) == 17
    assert all((FIXTURES / f"{name}.out").exists() for name in DECKS)
    assert sum(name.startswith(("n01_ga", "n02_ga", "n03_ga")) for name in DECKS) == 3


@pytest.mark.parametrize("name", DECKS)
def test_the_structure_specification_is_nec2c_s_bytes(served, name):
    """GA's row, GH's two rows and its pitch or cone line, among the rest of
    the block (the GM lines and segment count of n01, the GW of n10)."""
    assert "ERROR" not in "\n".join(served[name]), served[name][-5:]
    assert _structure_block(served[name]) == _structure_block(nec2c(name))


@pytest.mark.parametrize("name", DECKS)
def test_the_segments_are_nec2c_s(served, name):
    ours, theirs = _segment_rows(served[name]), _segment_rows(nec2c(name))
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs, strict=True):
        number = int(b[0])
        # SEG, I-, I, I+, TAG: the numbering and the connections.
        assert [int(v) for v in (a[0], *a[8:])] == [int(v) for v in (b[0], *b[8:])]
        # X, Y, Z, LENGTH, ALPHA, BETA, RADIUS to the printed 1e-4; BETA has
        # no value on a vertical segment.
        reals = range(7) if abs(float(b[5])) < 89.9999 else (0, 1, 2, 3, 4, 6)
        for k in reals:
            assert float(a[1 + k]) == pytest.approx(float(b[1 + k]), abs=1.01e-4), (
                number,
                k,
                a,
                b,
            )


@pytest.mark.parametrize(
    "geometry, match",
    [
        ("GA 1 12 1 0 360.00001 .002", "which NEC-2 stops on"),
        ("GA 1 0 1 0 90 .002", "GA segment count"),
        ("GA 1 12 0 0 90 .002", "positive arc radius"),
        ("GA 1 12 1 30 30 .002", "has no length"),
        ("GA 1 12 1 0 90 0", "nec2 dialect has no tapered arc"),
        ("GH 1 8 0 .5 .05 .05 .05 .05 .001", "nonzero spacing"),
        ("GH 1 8 .1 0 .05 .05 .05 .05 .001", "nonzero total length"),
        ("GH 1 8 .1 .5 .05 -.05 .05 .05 .001", "non-negative helix radii"),
        ("GH 1 8 .1 .5 0 0 0 0 .001", "non-negative helix radii"),
        # NEC-4.2's layout (turns, length, two helix radii, two wire radii)
        # leaves F7 empty, so it is not silently read as a NEC-2 helix.
        ("GH 1 8 1 .3 .1 .1 .001 .001", "positive wire radius"),
        ("GC 0 0 1 .001 .002", "GC (tapered wire continuation)"),
        ("GF 0", "GF (numerical Green's function)"),
    ],
)
def test_what_the_nec2_dialect_refuses(geometry, match):
    deck = f"CM\nCE\n{geometry}\nGE 0\nFR 0 1 0 0 30 0\nEX 0 1 1 0 1\nXQ\nEN\n"
    with pytest.raises(DeckError, match=re.escape(match)):
        parse(deck, dialect="nec2")


def test_the_two_dialects_read_gh_differently():
    """One card, two meanings: NEC-2's turn spacing is NEC-4.2's number of
    turns. The issue's deck (n11, and probe p83 on the NEC-4.2 slot) builds
    three turns in the nec2 dialect and half a turn in the nec4 one."""
    import math

    card = "GH 1 12 0.5 1.5 0.2 0.2 0.2 0.2 0.002"
    deck = f"CM\nCE\n{card}\nGE 0\nFR 0 1 0 0 100 0\nEX 0 1 6 0 1\nXQ\nEN\n"

    def sweep(dialect: str) -> float:
        from momwire.portal._portal import parse_deck

        wires = parse_deck(deck, dialect=dialect).structure.wires
        angles = [
            math.atan2(w.p2[1], w.p2[0]) - math.atan2(w.p1[1], w.p1[0]) for w in wires
        ]
        return sum((a + math.pi) % (2 * math.pi) - math.pi for a in angles) / (
            2 * math.pi
        )

    assert sweep("nec2") == pytest.approx(3.0)
    assert sweep("nec4") == pytest.approx(0.5)
