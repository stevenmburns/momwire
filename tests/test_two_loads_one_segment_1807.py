"""Two lumped loads on one segment, served as NEC adds them (momwire#1807).

The NEC-4.2 slot refused a deck with two ``LD`` cards on one segment, the
shape EZNEC writes for stacked traps in NEC-4.2 format (the NEC-4.2 form of
momwire#1787).  Licensed NEC-4.2 (the release build, black box) and nec2c
both answer it the same way, measured 2026-11-04 on an 11-segment dipole at
299.7925 MHz:

* the impedances ADD.  10 + 20 ohm series R prints the 30 ohm deck's drive
  row to every digit (104.729 + j42.4681 on NEC-4.2), the two parallel-LC
  traps of the repro print the deck with one ``LD 4`` of their summed
  reactance, and every pairing of types 0, 1 and 4 tried did the same;
* the loading table lists each card as written and closes with a "LOADED
  TWICE - IMPEDANCES ADDED" note, NEC-4.2 and nec2c each in its own layout;
* only CONSECUTIVE ``LD`` cards stack.  Any other card between two of them
  (FR, GN, EX, XQ) makes both engines DISCARD the earlier card's loads: with
  an EX between 10 and 20 ohm the drive row is the 20 ohm deck's, 97.1142 +
  j43.8649.  This reader discards them too since momwire#1812
  (``tests/test_ld_runs_1812.py``).

Fixtures: ``tests/fixtures/eznec_load_probes_1110/N42_*`` (licensed NEC-4.2
printouts) and ``N2_*`` (nec2c 1.3), beside ``M_0195_two_loads_one_node``;
the README there says how they were run.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from momwire.deck import DeckError, parse
from momwire.eznec._nec4 import render
from momwire.portal._portal import run_deck

FIX = Path(__file__).resolve().parent / "fixtures" / "eznec_load_probes_1110"
_NUM = re.compile(r"-?\d\.\d+E[+-]\d\d")
TWICE = "SOME OF THE ABOVE SEGMENTS HAVE BEEN LOADED TWICE - IMPEDANCES ADDED"


def deck(name: str) -> str:
    return (FIX / f"{name}.nec").read_text()


def oracle(name: str) -> list[str]:
    return (FIX / f"{name}.out").read_text().splitlines()


def ours42(text: str) -> list[str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return render(text, basis="sinusoidal").splitlines()


def ours2(text: str) -> list[str]:
    stdout, _err = run_deck(text)
    return stdout.splitlines()


def drive_row(lines: list[str]) -> str:
    """The first row of ANTENNA INPUT PARAMETERS (both layouts)."""
    at = next(i for i, line in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in line)
    return next(line for line in lines[at + 1 :] if len(_NUM.findall(line)) >= 6)


def drive_z(lines: list[str]) -> complex:
    v = _NUM.findall(drive_row(lines))
    return complex(float(v[4]), float(v[5]))


def loading(lines: list[str], until: str) -> list[str]:
    """STRUCTURE IMPEDANCE LOADING up to the next section heading."""
    start = next(i for i, line in enumerate(lines) if "IMPEDANCE LOADING" in line)
    stop = next(i for i in range(start, len(lines)) if until in lines[i])
    return lines[start:stop]


N42_SERVED = ["N42_0195_two_loads_one_segment", "N42_series_r_pair", "N42_mixed_three"]
N2_SERVED = ["N2_series_r_pair", "N2_mixed_three"]


@pytest.mark.parametrize("name", N42_SERVED)
def test_nec42_loading_table_matches_the_oracle_byte_for_byte(name):
    theirs = loading(oracle(name), "MATRIX TIMING")
    assert any(TWICE in line for line in theirs)
    assert loading(ours42(deck(name)), "MATRIX TIMING") == theirs


@pytest.mark.parametrize("name", N42_SERVED)
def test_nec42_drive_impedance_tracks_the_oracle(name):
    """Within the slot's gap on this 11-segment dipole: the bare dipole is
    itself 0.97 % of |Z| from NEC-4.2 (81.55 + j45.12 against 81.75 +
    j46.00), and the loads do not widen it."""
    theirs = drive_z(oracle(name))
    assert abs(drive_z(ours42(deck(name))) - theirs) <= 0.012 * abs(theirs)


def test_a_stacked_pair_answers_as_its_sum_on_both_sides():
    """NEC-4.2 prints 10 + 20 ohm and 30 ohm identically, and so do we."""
    pair, single = "N42_series_r_pair", "N42_series_r_single"
    assert drive_row(oracle(pair)) == drive_row(oracle(single))
    assert drive_row(ours42(deck(pair))) == drive_row(ours42(deck(single)))
    assert not any(TWICE in line for line in ours42(deck(single)))


def test_the_order_of_the_two_cards_does_not_matter():
    text = deck("N42_0195_two_loads_one_segment")
    first, second = "LD 1,1,8,0,0.,.0000001,5.E-12", "LD 1,1,8,0,0.,.0000002,3.E-12"
    swapped = text.replace(first, "@").replace(second, first).replace("@", second)
    assert swapped != text
    assert drive_row(ours42(swapped)) == drive_row(ours42(text))


def test_two_loads_separated_by_another_card_answer_as_the_second():
    """The oracle DISCARDS the first card's load here: the drive row is the
    second card's alone, with no note.  Since momwire#1812 this reader does
    the same, rather than refuse."""
    theirs = oracle("N42_separated_by_ex")
    assert drive_row(theirs) == drive_row(oracle("N42_separated_single"))
    assert not any(TWICE in line for line in theirs)
    ours = ours42(deck("N42_separated_by_ex"))
    assert drive_row(ours) == drive_row(ours42(deck("N42_separated_single")))
    assert not any(TWICE in line for line in ours)


@pytest.mark.parametrize("name", N2_SERVED)
def test_nec2_loading_table_matches_nec2c_byte_for_byte(name):
    theirs = loading(oracle(name), "ANTENNA ENVIRONMENT")
    assert any(TWICE in line for line in theirs)
    assert loading(ours2(deck(name)), "ANTENNA ENVIRONMENT") == theirs


def test_nec2_drive_impedance_tracks_nec2c():
    """The series pair only: on ``N2_mixed_three`` the portal's B-spline
    basis sits 5 % from nec2c, which is that basis's gap under 500 ohm
    parallel loads on three adjacent segments (one such load alone, no
    stacking, sits 4.0 % away), not anything this fix decides."""
    name = "N2_series_r_pair"
    theirs = drive_z(oracle(name))
    assert abs(drive_z(ours2(deck(name))) - theirs) <= 0.012 * abs(theirs)


def test_nec2_stacked_pair_answers_as_its_sum():
    pair, single = "N2_series_r_pair", "N2_series_r_single"
    assert drive_row(oracle(pair)) == drive_row(oracle(single))
    assert drive_row(ours2(deck(pair))) == drive_row(ours2(deck(single)))


def test_a_natively_loading_fill_is_handed_the_sum():
    """`_lumped_loads` (the kwarg a fill that carries its loads takes, razor's
    family) has one entry per site, the sum of every load in force there, a
    card written twice counted twice.  No nec2/nec4 deck reaches a razor
    basis today (centre feeds), so this pins the helper itself."""
    from momwire.deck._solver import _lumped_loads, prepare_mesh

    def loads(text: str) -> list[tuple[int, float, complex]]:
        model = parse(text)
        prepared = prepare_mesh(model)
        return _lumped_loads(model, prepared.ports, prepared.mesh.ports, 299.7925e6)

    body = deck("N2_series_r_single")
    single = loads(body)
    assert len(single) == 1 and single[0][2] == 30 + 0j
    assert loads(deck("N2_series_r_pair")) == single
    twice = body.replace("LD 0,1,8,8,30.,0.,0.", "LD 0,1,8,8,15.,0.,0.\n" * 2)
    assert loads(twice) == single


def test_one_card_reaching_one_segment_twice_still_refuses():
    """Not a stack.  ``LD 4,1,-1,0`` names segments -1 and 0, which this
    reader resolves onto one element; licensed NEC-4.2 prints such a card as
    ONE load row, no note, and a different antenna (the momwire#1300
    junction decks: 221.9 + j61.3 where adding would read 256.7 - j153.2).
    So the repeat inside one card keeps the refusal it always had."""
    text = deck("N42_series_r_single").replace(
        "LD 0,1,8,8,30.,0.,0.", "LD 4,1,-1,0,50.,0."
    )
    with pytest.raises(DeckError, match="a second load on one segment is not merged"):
        parse(text, dialect="nec4")
