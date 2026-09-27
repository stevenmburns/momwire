"""SimNEC's NEC-5 decks serve — momwire#1230.

SimNEC 5.3 takes ``momwire-eznec-client-razor-nec5`` as a NEC-5 engine
(the path carries ``nec5``; ``-version`` answers since #1229) and writes two
things EZNEC never does: an ``EX`` with no imaginary drive, and an ``RP`` with
a range. Both were refused; both are measured on the licensed NEC-5 here.

``tests/fixtures/eznec_simnec_1230/`` (README there) holds the decks and the
licensed printouts.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from momwire.deck._nec5 import parse_nec5
from momwire.eznec._serve import SPEED_OF_LIGHT_MHZ_M, refusal
from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_simnec_1230"
RANGE_M = 1000.0
# NEC-5's own formulation twin, the one the SimNEC link selects by name; the
# licensed comparisons below are like for like only on it (the default basis
# is B-spline, several ohms of X from NEC-5 at this mesh).
TWIN = "razor-nec5"
FREQ_MHZ = 7.331794


def deck(name: str) -> str:
    return (FIXTURES / f"{name}.nec").read_text()


def oracle(name: str) -> list[str]:
    return (FIXTURES / f"{name}.out").read_text(encoding="latin-1").splitlines()


def _no_timing(text: str) -> list[str]:
    return [
        line
        for line in text.splitlines()
        if "TIME" not in line.upper() and "FILL" not in line.upper()
    ]


def _pattern_section(lines: list[str]) -> list[str]:
    start = next(i for i, line in enumerate(lines) if "RADIATION PATTERNS" in line)
    return lines[start:]


_ROW = re.compile(r"^\s+-?\d+\.\d\d\s+-?\d+\.\d\d\s")
# Fixed columns (``_printout._pattern``): the SENSE cell is BLANK on a null
# row, so splitting on whitespace would shift every field after it.
_COLUMNS = {
    "angles": (0, 17),
    "gains": (17, 44),
    "polarisation": (44, 72),
    "e_theta": (72, 87),
    "e_theta_phase": (87, 96),
    "e_phi": (96, 111),
    "e_phi_phase": (111, 120),
}


def _rows(lines: list[str]) -> list[dict[str, str]]:
    return [
        {name: line[a:b].strip() for name, (a, b) in _COLUMNS.items()}
        for line in _pattern_section(lines)
        if _ROW.match(line)
    ]


def _total_db(row: dict[str, str]) -> float:
    return float(row["gains"].split()[2])


def _feed(lines: list[str]) -> tuple[float, float]:
    """R and X off the ANTENNA INPUT PARAMETERS row."""
    start = next(
        i for i, line in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in line
    )
    row = next(line for line in lines[start + 1 :] if re.match(r"^\s+1\s+11\s", line))
    cells = row.split()
    return float(cells[7]), float(cells[8])


# --------------------------------------------------------------------------
# EX with no imaginary drive
# --------------------------------------------------------------------------


def test_a_five_field_ex_parses_and_drives_at_one_plus_j_zero():
    five = parse_nec5(deck("dipole-ex5"))
    six = parse_nec5(deck("dipole-ex5").replace("1.000000e+00\n", "1.000000e+00 0.0\n"))
    assert five.sources == six.sources
    assert five.sources[0].drive == 1 + 0j


def test_a_five_field_ex_serves_the_same_printout_as_the_six_field_deck():
    """The licensed engine prints the two decks identically, timing aside
    (README); the seam must too."""
    text = deck("dipole-ex5")
    five = render(text)
    six = render(text.replace("1.000000e+00\n", "1.000000e+00 0.0\n"))
    assert "NEC ERROR" not in five
    assert _no_timing(five) == _no_timing(six)


def test_the_five_field_deck_lands_on_the_licensed_impedance():
    """razor-nec5 is NEC-5's formulation twin: well under 0.1 % apart."""
    r_ours, x_ours = _feed(render(deck("dipole-ex5"), basis=TWIN).splitlines())
    r_lic, x_lic = _feed(oracle("dipole-ex5"))
    assert r_ours == pytest.approx(r_lic, rel=1e-3)
    assert x_ours == pytest.approx(x_lic, rel=1e-3)


# --------------------------------------------------------------------------
# RP with a range
# --------------------------------------------------------------------------


def test_the_range_form_is_no_longer_refused_and_a_negative_range_is():
    assert refusal(parse_nec5(deck("dipole-rp-range"))) is None
    negative = deck("dipole-rp-range").replace(" 90 1000\n", " 90 -1000\n")
    assert "negative range" in refusal(parse_nec5(negative))


def test_the_range_heading_is_the_licensed_one_byte_for_byte():
    """RANGE= / EXP(-JKR)/R= / two blanks / the columns with VOLTS/M."""
    ours = _pattern_section(render(deck("dipole-rp-range")).splitlines())
    theirs = _pattern_section(oracle("dipole-rp-range"))
    assert ours[:8] == theirs[:8]


def test_the_range_shift_is_nec5s_own_k():
    """-164.02, not the SI c's -164.24: k at 299.8 MHz*m."""
    kr = 2.0 * math.pi * FREQ_MHZ / SPEED_OF_LIGHT_MHZ_M * RANGE_M
    shift = math.degrees(math.atan2(-math.sin(kr), math.cos(kr)))
    assert f"{shift:7.2f}" == "-164.02"
    heading = _pattern_section(render(deck("dipole-rp-range")).splitlines())
    assert heading[2].endswith("AT PHASE-164.02 DEGREES")


def test_each_range_row_is_the_norange_row_over_r_with_the_phase_added():
    """Measured on the licensed capture: gains unchanged, |E| / R, and the
    shift ADDED in degrees without re-wrapping (a zero component prints the
    bare shift). The seam's rows must follow the same rule off its own
    RFLD = 0 rows."""
    shift = -164.02
    for rows_range, rows_zero in (
        (_rows(oracle("dipole-rp-range")), _rows(oracle("dipole-rp-norange"))),
        (
            _rows(render(deck("dipole-rp-range")).splitlines()),
            _rows(render(deck("dipole-rp-norange")).splitlines()),
        ),
    ):
        assert len(rows_range) == len(rows_zero) > 0
        for a, b in zip(rows_range, rows_zero, strict=True):
            for same in ("angles", "gains", "polarisation"):
                assert a[same] == b[same]
            for mag, phase in (("e_theta", "e_theta_phase"), ("e_phi", "e_phi_phase")):
                assert float(a[mag]) == pytest.approx(
                    float(b[mag]) / RANGE_M, rel=1e-4, abs=1e-20
                )
                assert float(a[phase]) == pytest.approx(
                    float(b[phase]) + shift, abs=0.011
                )


def test_the_range_rows_land_on_the_licensed_rows():
    ours = _rows(render(deck("dipole-rp-range"), basis=TWIN).splitlines())
    theirs = _rows(oracle("dipole-rp-range"))
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs, strict=True):
        assert a["angles"] == b["angles"]
        if _total_db(b) > -100:
            assert _total_db(a) == pytest.approx(_total_db(b), abs=0.02)
        if float(b["e_theta"]) > 1e-8:
            assert float(a["e_theta"]) == pytest.approx(float(b["e_theta"]), rel=2e-3)
            assert float(a["e_theta_phase"]) == pytest.approx(
                float(b["e_theta_phase"]), abs=0.05
            )


def test_rfld_zero_takes_no_range_arithmetic():
    """RFLD = 0 is every EZNEC capture: no header lines, VOLTS, unshifted."""
    lines = _pattern_section(render(deck("dipole-rp-norange")).splitlines())
    assert lines[1] == ""
    assert "RANGE=" not in "\n".join(lines[:6])
    assert lines[4].rstrip().endswith("VOLTS     DEGREES")
