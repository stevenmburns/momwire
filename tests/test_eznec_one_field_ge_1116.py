"""A one-field ``GE`` card parses — momwire#1116.

Two of Mike WA7ARK's OCF decks (QRZ 1003328 #101/#104, "Created from
AutoEZ", 2026-09-17) carry a bare ``GE 0`` where every one of the 87
captured EZNEC decks carries the two-field ``GE n,-1``.  The second field is
NEC-5's own flag for whether it runs its geometry segment check (illegal
intersections, thin-wire violations); it has no bearing on the solve, and
NEC-5 itself takes 0 as its default when the field is absent — the same
value ``Card.i`` already reads a missing field as, so no separate "absent"
sentinel is needed downstream.

``tests/fixtures/eznec_one_field_ge_1116/`` (README there) holds the two
WA7ARK decks and their licensed-NEC-5 printouts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from momwire.deck import DeckError
from momwire.deck._nec5 import parse_nec5
from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures" / "eznec_one_field_ge_1116"

# The same minimal EZNEC-shaped dipole test_eznec_ld5_conductivity_1082.py
# uses, with the GE card left as a `{ge}` slot.
DIPOLE = """CM ge gate test dipole
CE
GW 1,11,0.,-.25,0.,0.,.25,0.,.0005
{ge}
GN -1
FR 0,1,0,0,299.7925
EX 4,1,6,0,1.414214,0.
PQ 0
RP 0,1,361,1000,90.,0.,0.,1.,0.
EN
"""

DECKS = ("WA7ARK-OCF-LoadOnly", "WA7ARK-OCF-Load-Xfmr-TL")
TOLERANCE = 0.01  # under 1 % on R and X, the session's measured bound


def deck(name: str) -> str:
    return (FIXTURES / f"{name}.nec").read_text()


def oracle(name: str) -> list[str]:
    return (FIXTURES / f"{name}.out").read_text().splitlines()


def _rows(lines: list[str]) -> list[str]:
    start = (
        next(i for i, line in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in line)
        + 4
    )
    out = []
    for line in lines[start:]:
        if not line.strip():
            break
        out.append(line)
    return out


def _cells(row: str) -> list[float]:
    return [float(x) for x in row[12:].split()]


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_one_field_ge_parses_and_defaults_the_second_field_to_zero():
    """NEC-5's own default for the missing flag is 0 (measured: the
    licensed printout echoes a bare ``GE 0`` back as ``GE    0    0``,
    see test_the_printout_echoes_the_default_second_field below)."""
    parsed = parse_nec5(DIPOLE.format(ge="GE 0"))
    assert parsed.ge_flag == 0
    assert parsed.ge_second == 0


def test_two_field_ge_is_unaffected():
    parsed = parse_nec5(DIPOLE.format(ge="GE 0,-1"))
    assert parsed.ge_flag == 0
    assert parsed.ge_second == -1


def test_a_nonzero_one_field_ground_flag_also_parses():
    parsed = parse_nec5(DIPOLE.format(ge="GE 1"))
    assert parsed.ge_flag == 1
    assert parsed.ge_second == 0


def test_zero_field_ge_still_refuses_by_name():
    """The field count is 1 now, not 0 -- a bare ``GE`` with no fields at
    all is still short and must still refuse, distinctly from the one-field
    form this issue accepts."""
    with pytest.raises(DeckError, match="GE carries 0 fields and needs at least 1"):
        parse_nec5(DIPOLE.format(ge="GE"))


def test_the_printout_echoes_the_default_second_field():
    """Matches the licensed NEC-5 printout's own echo of a bare ``GE 0``
    (``tests/fixtures/eznec_one_field_ge_1116/README.txt``): NEC-5 prints
    the defaulted second field, not a blank."""
    rendered = render(DIPOLE.format(ge="GE 0"))
    assert " GE    0    0\n" in rendered


def test_the_printout_still_echoes_a_written_second_field():
    rendered = render(DIPOLE.format(ge="GE 0,-1"))
    assert " GE    0   -1\n" in rendered


# --------------------------------------------------------------------------
# the two WA7ARK decks: served, no NEC ERROR, within tolerance of NEC-5
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", DECKS)
def test_the_deck_is_served_with_no_nec_error(name):
    rendered = render(deck(name))
    assert "NEC ERROR" not in rendered


@pytest.mark.parametrize("name", DECKS)
def test_the_drive_row_tag_and_segment_match_the_oracle(name):
    rows = _rows(render(deck(name)).splitlines())
    theirs = _rows(oracle(name))
    assert len(rows) == 1
    # TAG, SEG. and the trailing end digit, byte for byte
    assert rows[0][:12] == theirs[0][:12]


@pytest.mark.parametrize("name", DECKS)
def test_the_drive_row_impedance_agrees_with_the_oracle_within_one_percent(name):
    """Session-measured: 274.72-1431.8j vs NEC-5's 275.53-1441.0j (0.63 % of
    |Z|) on the load-only deck, 49.98+104.96j vs 49.31+104.23j (0.86 % of
    |Z|) on the transformer+line deck -- both under the 1110 seam's own
    complex-distance-over-|Z| measure (``test_eznec_load_probes_1110.py``'s
    ``test_the_drive_row_impedance_agrees_with_the_oracle``)."""
    ours = _cells(_rows(render(deck(name)).splitlines())[0])
    theirs = _cells(_rows(oracle(name))[0])
    z_ours, z_theirs = complex(ours[4], ours[5]), complex(theirs[4], theirs[5])
    assert abs(z_ours - z_theirs) <= TOLERANCE * abs(z_theirs), (z_ours, z_theirs)
