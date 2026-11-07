"""A run of LD cards, and a load on a current-driven segment (momwire#1812,
momwire#1813).

Measured 2026-11-05 on licensed NEC-4.2 (the release build, black box) and
nec2c 1.3, on the 11-segment free-space dipole of momwire#1807 at 299.7925 MHz.

**momwire#1812.**  NEC reads a run of consecutive ``LD`` cards as the deck's
loading, and the first ``LD`` after any other card starts a NEW run that
discards every load read before it: 10 ohm on segment 8, FR, 20 ohm on segment
3 answers as 20 ohm alone (91.639 + j44.187 on both engines, the second
card's deck to every digit).  The discard takes every LD-borne load, types 0,
1, 4, the distributed 2/3 and type 5's conductivity alike, and a zero-valued
card does it too.  The cards that break a run are FR, GN, GD, EX, XQ, RP, NE,
NH, TL, NT (and IS on NEC-4.2); PT and PQ do not, on either engine.  A
comment between two LD cards keeps the run on NEC-4.2 (nec2c refuses the
deck).  EK splits the engines: NEC-4.2 reads past it and keeps the run,
nec2c's EK ends it.  momwire used to keep every run's loads, a different
antenna with no warning.

**momwire#1813.**  ``EX 6`` (the NEC-4.2 slot's current source) on a segment
that also carries an ``LD``: NEC-4.2's impedance includes the load, so every
LD type gives the same drive impedance under EX 6 as under EX 0 (30 ohm: 111.75
+ j46.003 both ways).  The solve was right and the readout was not: the port
algebra solved for the gap voltage behind the load, and a current source read
its voltage, and so its impedance and input power, from there.

Fixtures: ``tests/fixtures/eznec_load_probes_1110/N42_run_*``,
``N42_ex*_fed_load_*``, ``N42_ex6_phased_*`` (NEC-4.2 printouts) and
``N2_run_*`` (nec2c); the README there says how they were run.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

from momwire.deck import parse
from momwire.dropin.nec5._nec4 import render
from momwire.dropin.nec2c._portal import run_deck

FIX = Path(__file__).resolve().parent / "fixtures" / "eznec_load_probes_1110"
_NUM = re.compile(r"-?\d\.\d+E[+-]\d\d")
TWICE = "SOME OF THE ABOVE SEGMENTS HAVE BEEN LOADED TWICE - IMPEDANCES ADDED"
# The slot's gap on this 11-segment dipole: the bare dipole alone is 0.97 % of
# |Z| from NEC-4.2 (81.55 + j45.12 against 81.75 + j46.00), and nothing here
# may widen it.
GAP = 0.012


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


def drive_rows(lines: list[str]) -> list[str]:
    """Every row of the first ANTENNA INPUT PARAMETERS table (both layouts)."""
    at = next(i for i, line in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in line)
    rows = []
    for line in lines[at + 1 :]:
        if len(_NUM.findall(line)) >= 6:
            rows.append(line)
        elif rows:
            break
    return rows


def drive_z(lines: list[str], row: int = 0) -> complex:
    v = _NUM.findall(drive_rows(lines)[row])
    return complex(float(v[4]), float(v[5]))


def loading(lines: list[str], until: str) -> list[str]:
    """STRUCTURE IMPEDANCE LOADING up to the next section heading."""
    start = next(i for i, line in enumerate(lines) if "IMPEDANCE LOADING" in line)
    stop = next(i for i in range(start, len(lines)) if until in lines[i])
    return lines[start:stop]


def budget(lines: list[str]) -> dict[str, float]:
    out = {}
    for line in lines:
        for key in ("INPUT POWER", "RADIATED POWER", "WIRE LOSS"):
            if key in line and "=" in line:
                out[key] = float(line.split("=")[1].split()[0])
    return out


# -- momwire#1812 ------------------------------------------------------------

# (deck, the deck NEC answers it as, whether the pair stacked)
N42_RUNS = [
    ("N42_run_fr_other_segment", "N42_run_single_segment3", False),
    ("N42_run_ld5_dropped", "N42_run_single_segment3", False),
    ("N42_run_zero_drops", "N42_run_bare", False),
    ("N42_separated_by_ex", "N42_separated_single", False),
    ("N42_run_ek_keeps", "N42_series_r_single", True),
    ("N42_run_cm_keeps", "N42_series_r_single", True),
]


@pytest.mark.parametrize("name, same_as, stacked", N42_RUNS)
def test_nec42_answers_a_broken_run_as_its_last_run(name, same_as, stacked):
    """The oracle's own evidence first (the deck prints its equivalent's drive
    row to every digit), then ours, which must do the same."""
    theirs, ours = oracle(name), ours42(deck(name))
    assert drive_rows(theirs) == drive_rows(oracle(same_as))
    assert drive_rows(ours) == drive_rows(ours42(deck(same_as)))
    assert any(TWICE in line for line in theirs) == stacked
    assert any(TWICE in line for line in ours) == stacked


N42_TABLES = [
    "N42_run_fr_other_segment",
    "N42_run_ek_keeps",
    "N42_run_cm_keeps",
    "N42_run_pq_keeps",
    "N42_separated_by_ex",
]


@pytest.mark.parametrize("name", N42_TABLES)
def test_nec42_loading_table_is_the_last_runs_byte_for_byte(name):
    theirs = loading(oracle(name), "MATRIX TIMING")
    assert loading(ours42(deck(name)), "MATRIX TIMING") == theirs


@pytest.mark.parametrize(
    "name", [name for name, _same, _stacked in N42_RUNS] + ["N42_run_pq_keeps"]
)
def test_nec42_drive_impedance_tracks_the_oracle(name):
    theirs = drive_z(oracle(name))
    assert abs(drive_z(ours42(deck(name))) - theirs) <= GAP * abs(theirs)


N2_RUNS = [
    ("N2_run_fr_other_segment", "N2_run_single_segment3", False),
    ("N2_run_ld5_dropped", "N2_run_single_segment3", False),
    ("N2_run_ek_breaks", "N2_run_ek_single", False),
    ("N2_run_pt_keeps", "N2_series_r_single", True),
]


@pytest.mark.parametrize("name, same_as, stacked", N2_RUNS)
def test_nec2c_answers_a_broken_run_as_its_last_run(name, same_as, stacked):
    theirs, ours = oracle(name), ours2(deck(name))
    assert drive_rows(theirs) == drive_rows(oracle(same_as))
    assert drive_rows(ours) == drive_rows(ours2(deck(same_as)))
    assert any(TWICE in line for line in theirs) == stacked
    assert any(TWICE in line for line in ours) == stacked
    assert loading(ours, "ANTENNA ENVIRONMENT") == loading(
        theirs, "ANTENNA ENVIRONMENT"
    )
    assert abs(drive_z(ours) - drive_z(theirs)) <= GAP * abs(drive_z(theirs))


def test_ek_ends_a_run_in_the_nec2_dialect_only():
    """The same deck, two dialects, two antennas: nec2c's EK ends the run,
    NEC-4.2's does not (both oracles above)."""
    text = deck("N2_run_ek_breaks")
    assert len(parse(text, dialect="nec2").loads) == 1
    assert len(parse(text, dialect="nec4").loads) == 2


def test_each_execute_group_carries_its_own_run():
    """10 ohm, XQ, then 20 ohm and 5 ohm: the second group solves with the
    second run alone (nec2c 89.451 + j45.044 then 107.21 + j42.071), so a
    group's loads are its run's."""
    model = parse(
        deck("N2_run_single_segment3")
        .replace("LD 0,1,3,3,20.,0.,0.\n", "")
        .replace(
            "XQ\n",
            "LD 0,1,8,8,10.,0.,0.\nXQ\nLD 0,1,8,8,30.,0.,0.\nLD 0,1,3,3,5.,0.,0.\nXQ\n",
        ),
        dialect="nec2",
    )
    ran = [group for group in model.groups if group is not None]
    ohms = [sorted(model.loads[i][2].r for i in group.loads) for group in ran]
    assert ohms == [[10.0], [5.0, 30.0]]


# -- momwire#1813 ------------------------------------------------------------


@pytest.mark.parametrize("load", ["r30", "par500", "zx"])
def test_ex6_on_a_loaded_segment_includes_the_load(load):
    """NEC-4.2 prints one drive impedance under EX 0 and EX 6, the load
    included; so do we, and both track the oracle."""
    ex6, ex0 = f"N42_ex6_fed_load_{load}", f"N42_ex0_fed_load_{load}"
    assert drive_z(oracle(ex6)) == drive_z(oracle(ex0))
    ours6, ours0 = ours42(deck(ex6)), ours42(deck(ex0))
    assert abs(drive_z(ours6) - drive_z(ours0)) <= 1e-5 * abs(drive_z(ours0))
    theirs = drive_z(oracle(ex6))
    assert abs(drive_z(ours6) - theirs) <= GAP * abs(theirs)


def test_ex6_power_budget_counts_the_load():
    """30 ohm on the fed segment at 1.414 A is 30 W of load loss on top of the
    dipole's own radiated power, in the INPUT POWER too (NEC-4.2 111.75 W in,
    81.75 radiated, 30.00 lost)."""
    theirs = budget(oracle("N42_ex6_fed_load_r30"))
    ours = budget(ours42(deck("N42_ex6_fed_load_r30")))
    assert ours["WIRE LOSS"] == pytest.approx(theirs["WIRE LOSS"], rel=1e-4)
    assert ours["INPUT POWER"] == pytest.approx(
        ours["RADIATED POWER"] + ours["WIRE LOSS"], rel=1e-4
    )
    assert ours["INPUT POWER"] == pytest.approx(theirs["INPUT POWER"], rel=GAP)


def test_a_phased_drive_reads_the_load_too():
    """An EX 6 on the loaded segment beside an EX 0 elsewhere: the current
    source's impedance moves by the load exactly, on NEC-4.2 (81.25 to 111.25
    ohm) and here, and the EX 0 row does not move at all."""
    loaded, bare = "N42_ex6_phased_fed_load", "N42_ex6_phased_unloaded"
    for side in (oracle, lambda name: ours42(deck(name))):
        shift = drive_z(side(loaded), 0) - drive_z(side(bare), 0)
        assert abs(shift - 30.0) <= 0.01
        assert drive_rows(side(loaded))[1] == drive_rows(side(bare))[1]
    theirs = drive_z(oracle(loaded))
    assert abs(drive_z(ours42(deck(loaded))) - theirs) <= GAP * abs(theirs)
