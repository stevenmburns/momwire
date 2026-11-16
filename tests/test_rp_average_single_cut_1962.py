"""``RP 0`` with one theta or one phi averages nothing (momwire#1962).

The NEC-5 engine reads XNDA's A digit (the average gain) only when the card's
pattern has more than one theta AND more than one phi: on a single cut it
prints the card as at A = 0, with no average-gain trailer at A = 1 or 3 and
the pattern rows at A = 2. Read black-box off licensed x13 printouts of
synthetic decks: ``rp_normalized_gain/small*`` (one phi, DPH = 0, A = 1,
where the issue was found) and ``rp_average_single_cut_1962/`` (one phi at
DPH = 10 and at A = 2 and 3, one theta at A = 1 and 2, and two phis as the
control). momwire printed the trailer on every one of them.

Each deck is held to the engine's printout twice: the engine's own capture
pins the rule, and momwire's printout of the same deck is held to the
capture's pattern block line for line (the angles, the column headers, the
trailer or its absence, the blank lines before the next block), each line's
text with its numbers masked. The C++ printout core prints the
Python bytes (test_razor_core_standalone's
`test_a_single_cut_prints_no_average_as_python`). The NEC-4.2 slot, which
shares the row builder, keeps its trailer: no NEC-4.2 capture measures the
case.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from momwire.dropin.nec5 import _nec4 as _nec4_slot
from momwire.dropin.nec5 import _shell

FIXTURES = Path(__file__).parent / "fixtures"
SINGLE_CUT = FIXTURES / "rp_average_single_cut_1962"
NORMALIZED = FIXTURES / "rp_normalized_gain"
TRAILER = "AVERAGE POWER GAIN="

# (directory, stem) -> (trailer printed, pattern rows printed)
DECKS = {
    (NORMALIZED, "small1"): (False, 1),
    (NORMALIZED, "small2"): (False, 2),
    (NORMALIZED, "small4"): (False, 4),
    (NORMALIZED, "small5"): (False, 5),
    (SINGLE_CUT, "cut_phi_dphi10"): (False, 3),
    (SINGLE_CUT, "cut_phi_a2"): (False, 3),
    (SINGLE_CUT, "cut_phi_a3"): (False, 3),
    (SINGLE_CUT, "cut_theta"): (False, 3),
    (SINGLE_CUT, "cut_theta_a2"): (False, 3),
    (SINGLE_CUT, "cut_two_phi"): (True, 6),
}
IDS = [stem for _d, stem in DECKS]
# Numbers are not compared: momwire's fields differ from x13's in their last
# digits (another solver), and its average gain over a partial phi range is
# not x13's yet (a separate gap; the control's trailer is compared by shape).
_NUMBER = re.compile(r"-?\d+\.\d+(?:E[-+]\d+)?|NaN")


def _engine(directory: Path, stem: str) -> str:
    return (directory / f"{stem}.nec5.out").read_bytes().decode("latin-1")


def _deck(directory: Path, stem: str) -> str:
    return (directory / f"{stem}.nec").read_bytes().decode("latin-1")


def _pattern_block(text: str) -> list[str]:
    """The RADIATION PATTERNS block: its heading through the line before the
    NORMALIZED GAIN table (every deck here asks for one, N = 5)."""
    lines = text.replace("\r\n", "\n").split("\n")
    start = next(k for k, line in enumerate(lines) if "RADIATION PATTERNS" in line)
    stop = next(k for k in range(start, len(lines)) if "NORMALIZED GAIN" in lines[k])
    return lines[start:stop]


def _rows(block: list[str]) -> list[str]:
    return [
        line for line in block if "LINEAR" in line or "RIGHT" in line or "LEFT" in line
    ]


def test_the_fixtures_are_there():
    stems = sorted(p.stem for p in SINGLE_CUT.glob("*.nec"))
    assert stems == sorted(stem for d, stem in DECKS if d == SINGLE_CUT)
    for directory, stem in DECKS:
        assert (directory / f"{stem}.nec5.out").exists(), stem


@pytest.mark.parametrize(("directory", "stem"), list(DECKS), ids=IDS)
def test_the_engine_averages_only_a_grid(directory, stem):
    """The evidence: x13 prints the trailer only where both axes have more
    than one angle, and A = 2 drops the rows only there."""
    trailer, rows = DECKS[(directory, stem)]
    block = _pattern_block(_engine(directory, stem))
    assert any(TRAILER in line for line in block) is trailer
    assert len(_rows(block)) == rows


def _shape(line: str) -> str:
    return _NUMBER.sub("#", line.rstrip())


@pytest.mark.parametrize("basis", ["razor-2p", "bspline"])
@pytest.mark.parametrize(("directory", "stem"), list(DECKS), ids=IDS)
def test_momwire_prints_the_engines_pattern_block(directory, stem, basis):
    engine = _pattern_block(_engine(directory, stem))
    mine = _pattern_block(_shell.render(_deck(directory, stem), basis=basis))
    assert len(mine) == len(engine), "\n".join(mine)
    for e, m in zip(engine, mine, strict=True):
        assert _shape(e) == _shape(m), (e, m)
    # the directions, as the engine walks them
    assert [r.split()[:2] for r in _rows(mine)] == [
        r.split()[:2] for r in _rows(engine)
    ]
    assert any(TRAILER in line for line in mine) is DECKS[(directory, stem)][0]


def test_the_nec42_slot_keeps_its_trailer():
    """Not measured on NEC-4.2, so the NEC-4.2 slot prints what it printed."""
    # N = 0: the NEC-4.2 printout has no NORMALIZED GAIN table
    text = _deck(SINGLE_CUT, "cut_phi_dphi10").replace(" 1501 ", " 1001 ", 1)
    assert " 1001 " in text
    out = _nec4_slot.render(text, basis="sinusoidal")
    assert TRAILER in out, out[-2000:]
