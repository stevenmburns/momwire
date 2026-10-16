"""4nec2's far-field request (momwire#1503): ``RP 0`` with XNDA 1000 to 1003.

4nec2's Generate > Far Field writes XNDA 1003 with its default Expert
settings, and this seam used to refuse everything but EZNEC's 1000 and 1001.
The A (averaging) digit was measured black-box on the NEC-5 engine
(``tests/fixtures/eznec_xnda_1503/``): A = 3 prints exactly what A = 1
prints, and A = 2 prints the heading and the average-gain trailer with no
rows. Each rule is pinned twice: on the engine's own captures, so the
evidence is in the tree, and on momwire's printouts of the same decks.
"""

import math
from pathlib import Path

import pytest

from momwire.deck._nec5 import DeckError, parse_nec5
from momwire.eznec import _shell

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_xnda_1503"
ROW_PREFIX_WIDTH = 17  # "%8.2f%9.2f": a pattern row starts with two angles


def _lines(text: str) -> list[str]:
    """The printout's lines with the per-deck and per-run lines dropped: the
    comment echo (it names the XNDA), the RP card's echo, the engine stamp
    (line 2) and the timings."""
    out = []
    for k, line in enumerate(text.replace("\r\n", "\n").split("\n")):
        if k == 1 or "XNDA probe" in line or "RP   0" in line:
            continue
        if "FILL=" in line or "RUN TIME" in line:
            continue
        out.append(line)
    return out


def _is_pattern_row(line: str) -> bool:
    head = line[:ROW_PREFIX_WIDTH].split()
    if len(head) != 2:
        return False
    try:
        [float(v) for v in head]
    except ValueError:
        return False
    return len(line) > 100 and len(line.split()) >= 11


def _engine(xnda: int, stem: str = "rp") -> list[str]:
    path = FIXTURES / f"{stem}{xnda}.nec5.out"
    return _lines(path.read_bytes().decode("latin-1"))


def _momwire(xnda: int, basis: str, stem: str = "rp") -> list[str]:
    text = (FIXTURES / f"{stem}{xnda}.nec").read_bytes().decode("latin-1")
    return _lines(_shell.render(text, basis=basis))


@pytest.mark.parametrize("source", ["engine", "razor-2p", "bspline"])
def test_a3_prints_exactly_what_a1_prints(source):
    get = _engine if source == "engine" else lambda x: _momwire(x, source)
    assert get(1003) == get(1001)


@pytest.mark.parametrize("source", ["engine", "razor-2p", "bspline"])
def test_a2_prints_the_trailer_without_the_rows(source):
    get = _engine if source == "engine" else lambda x: _momwire(x, source)
    a1, a2 = get(1001), get(1002)
    rows = [line for line in a1 if _is_pattern_row(line)]
    assert len(rows) == 7 * 4
    assert a2 == [line for line in a1 if not _is_pattern_row(line)]
    assert any("AVERAGE POWER GAIN=" in line for line in a2)


@pytest.mark.parametrize("source", ["engine", "razor-2p"])
def test_a0_prints_the_rows_without_the_trailer(source):
    get = _engine if source == "engine" else lambda x: _momwire(x, source)
    a0, a1 = get(1000), get(1001)
    assert not any("AVERAGE POWER GAIN=" in line for line in a0)
    assert [r for r in a0 if _is_pattern_row(r)] == [
        r for r in a1 if _is_pattern_row(r)
    ]


def test_the_engine_and_momwire_lay_the_a2_block_out_alike():
    """The A = 2 block's heading, columns and trailer, line for line: only
    the trailer's two numbers differ (the solvers differ in their last
    digits, as everywhere)."""

    def block(lines):
        start = next(k for k, l in enumerate(lines) if "RADIATION PATTERNS" in l)
        end = next(k for k, l in enumerate(lines) if "POWER RADIATED ASSUMING" in l)
        return lines[start : end + 1]

    engine, mine = block(_engine(1002)), block(_momwire(1002, "razor-2p"))
    assert len(engine) == len(mine)
    for e, m in zip(engine, mine, strict=True):
        if "AVERAGE POWER GAIN=" in e or "POWER RADIATED" in e:
            assert e.split("=")[0] == m.split("=")[0]
        else:
            assert e == m


@pytest.mark.parametrize(
    ("xnda", "words"),
    [
        (3, "major / minor axis gains (X = 0)"),
        (1103, "a normalized-gain table (N = 1)"),
        (1023, "gain type D = 2"),
        (1004, "averaging A = 4"),
        (2003, "output format X = 2"),
        (20003, "not an XNDA this engine reads"),
    ],
)
def test_the_other_forms_refuse_by_digit_and_4nec2_setting(xnda, words):
    text = (FIXTURES / "rp1003.nec").read_text(encoding="latin-1")
    text = text.replace("RP 0 7 4 1003 ", f"RP 0 7 4 {xnda} ")
    with pytest.raises(DeckError) as excinfo:
        parse_nec5(text)
    message = str(excinfo.value)
    assert words in message
    assert "XNDA 1000 to 1003 and 1010 to 1013" in message
    if xnda <= 9999:
        assert "In 4nec2, set Generate > Far Field's Expert settings" in message
    assert "EZNEC" not in message


def _gain_heading(lines):
    return next(line for line in lines if "- - ANGLES - -" in line)


def test_d1_heads_the_rows_directive_gains_as_the_engine_does():
    for xnda in (1010, 1011):
        engine = _gain_heading(_engine(xnda, "lossy"))
        assert "- DIRECTIVE GAINS -" in engine
        assert _gain_heading(_momwire(xnda, "razor-2p", "lossy")) == engine
    assert "- POWER GAINS -" in _gain_heading(_engine(1001, "lossy"))


@pytest.mark.parametrize("source", ["engine", "razor-2p"])
def test_d1_moves_the_rows_by_the_efficiency_and_not_the_trailer(source):
    """On a 40-ohm-loaded dipole (76.9 % efficient) every row's gain moves by
    10 log10(Pin / Prad) and the average-gain trailer stays a power gain."""
    if source == "engine":
        d0, d1 = _engine(1001, "lossy"), _engine(1011, "lossy")
    else:
        d0, d1 = _momwire(1001, source, "lossy"), _momwire(1011, source, "lossy")
    budget = {
        line.split("=")[0].strip(): float(line.split("=")[1].split()[0])
        for line in d0
        if "INPUT POWER" in line or "RADIATED POWER" in line
    }
    shift = 10.0 * math.log10(budget["INPUT POWER"] / budget["RADIATED POWER"])
    rows0 = [r for r in d0 if _is_pattern_row(r)]
    rows1 = [r for r in d1 if _is_pattern_row(r)]
    assert len(rows0) == len(rows1) == 28
    for a, b in zip(rows0, rows1, strict=True):
        g0, g1 = float(a.split()[4]), float(b.split()[4])
        if g0 > -999:
            assert abs((g1 - g0) - shift) <= 0.011
    assert [r for r in d0 if "AVERAGE POWER GAIN" in r] == [
        r for r in d1 if "AVERAGE POWER GAIN" in r
    ]


def test_d1_is_the_power_gain_over_the_radiated_power_bit_for_bit():
    """momwire's directive rows are the power-gain rows scaled by Pin / Prad
    before the dB, so 1011 and 1001 differ in the gain cells alone: the
    angles, the fields and the ellipse are the same numbers."""
    from momwire.eznec import _serve
    from momwire.deck._nec5 import parse_nec5 as parse

    text = (FIXTURES / "lossy1011.nec").read_text(encoding="latin-1")
    data = _serve.serve(parse(text), basis="razor-2p")
    (block,) = data.patterns
    assert block.directive
    power = _serve.serve(parse(text.replace(" 1011 ", " 1001 ")), basis="razor-2p")
    (pblock,) = power.patterns
    assert not pblock.directive
    budget = data.power
    scale = budget.input_power / budget.radiated_power
    for drow, prow in zip(block.rows, pblock.rows, strict=True):
        assert (drow.theta_deg, drow.phi_deg) == (prow.theta_deg, prow.phi_deg)
        assert drow.e_theta_magnitude == prow.e_theta_magnitude
        assert drow.axial_ratio == prow.axial_ratio
    assert block.average_power_gain == pblock.average_power_gain
    assert scale > 1.0
