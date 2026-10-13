"""The NEC-4.2 slot prints every ``RP``/``NE``/``NH`` card of a run, and serves
SimNEC's ``RP`` (momwire#1463).

Reported by Dan AC6LA (2026-10-13): SimNEC with the ``momwire-nec42``
launcher on a GN 3 deck carrying ``NE 1`` showed a bad run and a 1 KB
printout.  Two faults, both reproduced on 0.77.0 with the decks below:

1. Only the FIRST execute card of a run printed its table.  A further
   ``RP``, ``NE`` or ``NH`` was echoed as an input line and nothing followed
   it: no table, no refusal, exit 0.
2. SimNEC writes its pattern card with a range, ``RP 0 19 73 1001 0 0 5 5
   1000``, and the slot refused every such card.

The layout these decks must print is gated byte for byte against the
licensed NEC-4.2's own printouts in ``test_eznec_nec42_printout_1295.py``
(probes p16-p32, ``tests/fixtures/eznec_nec42/README.txt``).  This module
holds the issue's own decks through the slot's process entry, the way SimNEC
and EZNEC launch it, and the shapes that must refuse by name instead.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from momwire.eznec import _nec4, _shell

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "eznec_nec42"
BASIS = "bspline"

# The issue's base deck: Dan's shape, a vertical over GN 3 with SimNEC's
# NOFILE on the ground card.
BASE = (
    "CM vertical over GN 3, NEC-4.2 slot\n"
    "CE\n"
    "GW 1 20 0. 0. 0. 0. 0. 10. .001\n"
    "GE 1\n"
    "GN 3 0 0 0 13. .005 0 0 NOFILE\n"
    "FR 0 1 0 0 7.1 0\n"
    "EX 0 1 1 0 1. 0.\n"
)
RP = "RP 0 19 2 1001 0. 0. 5. 90."
RP_RANGE = "RP 0 19 2 1001 0. 0. 5. 90. 1000."
NE = "NE 1 1 10 1 100. 0. 0. 0. 10. 0."
SIMNEC_RP = "RP 0 19 73 1001 0 0 5 5 1000"

_PATTERNS = "- - - RADIATION PATTERNS - - -"
_NEAR = "- - - NEAR ELECTRIC FIELDS - - -"
_INPUT = "- - - ANTENNA INPUT PARAMETERS - - -"


def deck(*cards: str, base: str = BASE) -> str:
    return base + "".join(card + "\n" for card in cards) + "EN\n"


def serve(text: str, tmp_path: Path) -> str:
    """One deck through the slot's process entry: two paths, a printout."""
    source, out = tmp_path / "EZ.NEC", tmp_path / "NEC.OUT"
    source.write_bytes(text.encode("latin-1"))
    assert _shell.main([str(source), str(out)], basis=BASIS, dialect="nec4") == 0
    return out.read_bytes().decode("latin-1").replace("\r\n", "\n")


def headings(text: str) -> list[str]:
    """The table headings and the execute cards' echoes, in printed order."""
    found = []
    for line in text.split("\n"):
        if line.strip() in (_PATTERNS, _NEAR, _INPUT):
            found.append(line.strip())
        elif line.startswith(" ***** INPUT LINE") and line[22:24] in (
            "RP",
            "NE",
            "NH",
            "XQ",
        ):
            found.append(line[22:24])
    return found


_ROW = re.compile(r"^ *-?\d+\.\d\d +-?\d+\.\d\d +-?\d+\.\d\d ")


def pattern_rows(text: str) -> list[list[float]]:
    """Every pattern row's numbers (the SENSE word dropped)."""
    return [
        [float(x) for x in (line[:66] + line[72:]).split()]
        for line in text.split("\n")
        if _ROW.match(line) and len(line) > 100
    ]


# ==========================================================================
# fault 1: every execute card of a run prints
# ==========================================================================


@pytest.mark.integration
@pytest.mark.parametrize(
    "cards, want",
    [
        pytest.param((RP, NE), ["RP", _INPUT, _PATTERNS, "NE", _NEAR], id="rp-ne"),
        pytest.param((NE, RP), ["NE", _INPUT, _NEAR, "RP", _PATTERNS], id="ne-rp"),
        pytest.param(
            ("XQ", RP, NE),
            ["XQ", _INPUT, "RP", _PATTERNS, "NE", _NEAR],
            id="xq-rp-ne",
        ),
        pytest.param(
            (RP, RP, NE, "XQ"),
            ["RP", _INPUT, _PATTERNS, "RP", _PATTERNS, "NE", _NEAR, "XQ"],
            id="rp-rp-ne-xq",
        ),
    ],
)
def test_every_execute_card_of_a_run_prints_its_table(cards, want, tmp_path):
    """The issue's three rows and one more: each table once, in card order,
    under its own card's echo, with one set of input parameters.  A further
    ``XQ`` prints nothing, as on the licensed engine (p27)."""
    text = serve(deck(*cards), tmp_path)
    assert "NEC ERROR" not in text, text[-600:]
    assert headings(text) == want
    rows = pattern_rows(text)
    assert len(rows) == 38 * sum(c == RP for c in cards)
    assert all(math.isfinite(v) for row in rows for v in row)


@pytest.mark.integration
def test_a_further_table_reads_the_run_it_follows(tmp_path):
    """The pattern printed after an ``NE`` is the pattern the same card
    prints when it runs the deck itself, row for row."""
    alone = pattern_rows(serve(deck(RP), tmp_path))
    after = pattern_rows(serve(deck(NE, RP), tmp_path))
    assert len(alone) == 38 and after == alone


# ==========================================================================
# fault 2: SimNEC's RP, at a range
# ==========================================================================


@pytest.mark.integration
def test_simnecs_rp_card_is_served(tmp_path):
    """The card SimNEC writes on every call that asks for a pattern.  Before
    momwire#1463 the whole printout was ``RP at a nonzero range is not
    served``."""
    text = serve(deck(SIMNEC_RP), tmp_path)
    assert "NEC ERROR" not in text, text[-600:]
    assert "RANGE= 1.000000E+03 METERS" in text
    assert "EXP(-JKR)/R= 1.00000E-03 AT PHASE 114.32 DEGREES" in text
    assert "LOWER MEDIUM - EXP(-JKR)/R= 2.23389-107 AT PHASE-164.12 DEGREES" in text
    assert text.count("VOLTS/M    DEGREES      VOLTS/M    DEGREES") == 1
    assert len(pattern_rows(text)) == 19 * 73
    assert "AVERAGE POWER GAIN=" in text


@pytest.mark.integration
def test_dans_shape_serves_through_the_slot(tmp_path):
    """Dan's run: GN 3 with NOFILE, SimNEC's pattern card and an ``NE 1`` in
    the same run.  Both tables print, and the near field is finite at every
    point from the zenith to the horizon."""
    text = serve(deck(SIMNEC_RP, NE), tmp_path)
    assert "NEC ERROR" not in text, text[-600:]
    assert "FINITE GROUND.  SOMMERFELD SOLUTION GN3" in text
    assert headings(text) == ["RP", _INPUT, _PATTERNS, "NE", _NEAR]
    assert len(pattern_rows(text)) == 19 * 73
    lines = text.split("\n")
    at = lines.index(next(ln for ln in lines if ln.strip() == _NEAR))
    near = []
    for line in lines[at + 5 :]:
        if not line.strip():
            break
        near.append([float(x) for x in line.split()])
    assert len(near) == 10
    assert near[0][:3] == [0.0, 0.0, 100.0]
    assert all(math.isfinite(v) for row in near for v in row)
    assert all(max(row[3], row[5], row[7]) > 0.0 for row in near)


@pytest.mark.integration
def test_the_range_form_is_the_range_free_pattern_scaled(tmp_path):
    """What the licensed printouts show of one deck at range 0 and at 1000 m
    (p17 against p16): the gains, the axial ratio and the tilt do not move,
    each E magnitude is divided by the range, and each E phase gains the
    heading's phase, 114.32 degrees at 7.1 MHz, without being wrapped."""
    plain = pattern_rows(serve(deck(RP), tmp_path))
    ranged = pattern_rows(serve(deck(RP_RANGE), tmp_path))
    assert len(plain) == len(ranged) == 38
    for a, b in zip(plain, ranged, strict=True):
        assert a[:7] == b[:7]
        for magnitude, phase in ((7, 8), (9, 10)):
            assert b[magnitude] == pytest.approx(a[magnitude] / 1000.0, rel=2e-5)
            assert b[phase] == pytest.approx(a[phase] + 114.32, abs=0.011)


# ==========================================================================
# what is refused, by name
# ==========================================================================

_SWEEP = BASE.replace("FR 0 1 0 0 7.1 0", "FR 0 2 0 0 7.1 .1")
_GD_AFTER_A_RUN = (FIXTURES / "refused" / "r1_gd_after_a_run.nec").read_text()


@pytest.mark.parametrize(
    "text, match",
    [
        pytest.param(
            deck(RP, NE, base=_SWEEP),
            "a further NE card after a frequency sweep's execute card",
            id="further-table-after-a-sweep",
        ),
        pytest.param(
            _GD_AFTER_A_RUN,
            "a further RP card after a GD card read since the run it follows",
            id="gd-between",
        ),
        pytest.param(
            deck(NE, RP.replace("1001", "0")),
            "RP with XNDA 0",
            id="further-rp-xnda",
        ),
        pytest.param(
            deck(RP, "RP 1 19 2 1001 0. 0. 5. 90."),
            "RP mode 1",
            id="further-rp-mode",
        ),
        pytest.param(
            deck(RP, NE.replace("NE 1", "NE 2")),
            "NE coordinate system 2",
            id="further-ne-coordinates",
        ),
        pytest.param(
            deck("RP 0 19 2 1001 0. 0. 5. 90. -1000."),
            "RP at a negative range",
            id="negative-range",
        ),
    ],
)
def test_a_shape_no_printout_shows_refuses_by_name(text, match):
    """Before a matrix is filled.  The further cards are read as strictly as
    the card that ran: before momwire#1463 each of these was skipped with
    nothing said, the refusal lost with the table."""
    with pytest.raises(_nec4.Nec4Refusal, match=match):
        _nec4.solve(text, basis=BASIS)


def test_the_refused_gd_shape_is_what_the_licensed_engine_prints_differently():
    """Why ``GD`` between a run and a further ``RP`` refuses: the licensed
    engine's printout of that deck (kept beside it) puts a FREQUENCY and an
    ANTENNA ENVIRONMENT block in front of the second pattern and no input
    parameters, which is neither a run's layout nor a further table's."""
    printout = (FIXTURES / "refused" / "r1_gd_after_a_run.out").read_text()
    lines = [ln.strip() for ln in printout.split("\n")]
    assert lines.count(_PATTERNS) == 2 and lines.count(_INPUT) == 1
    assert lines.count("- - - - - - FREQUENCY - - - - - -") == 2
    assert lines.count("- - - ANTENNA ENVIRONMENT - - -") == 2
