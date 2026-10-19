"""momwire 0.79.2: a deck read as the licensed NEC-5 reads it, a program top
to bottom -- each RP / XQ runs with the FR in force at that point.

Dan AC6LA's 4nec2 cards put an FR after the XQ (`FR 0 0 0 0 3.68 0 / XQ /
FR 0 6 0 0 3.5 0.1 / RP ...`): NEC-5 x13 prints the run at 3.68 MHz with no
pattern, then six frequencies 3.5-4.0 MHz, each a whole block with its
pattern. The layout was probed black-box on x13 for that deck and for six
more orderings (an FR per run, a source change after a sweep, an RP after a
sweep, an FR repeating the frequency in force); the probe printouts stay in
scratch. These tests pin the shape: which frequencies print, in which run,
with which sections.
"""

import re
from pathlib import Path

import pytest

from momwire.eznec._shell import render

FIXTURE = Path(__file__).parent / "fixtures" / "nec5_sequence" / "fr_after_xq.nec"
HEAD = "\n".join(
    [
        "CM probe",
        "CE",
        "GW 1 21 0 -20 10 0 20 10 0.001",
        "GW 2 21 5 -20 10 5 20 10 0.001",
        "GE 0",
        "GN 1",
    ]
)


def _out(cards, basis="razor-2p"):
    return render(HEAD + "\n" + "\n".join(cards) + "\n", basis=basis)


def _frequencies(out):
    return [float(m) for m in re.findall(r"FREQUENCY= ([0-9.E+-]+) MHZ", out)]


@pytest.mark.parametrize("basis", ["razor-2p", "bspline"])
def test_dans_fr_after_xq_runs_each_loop_at_its_own_fr(basis):
    out = render(FIXTURE.read_text(), basis=basis)
    assert "NEC ERROR" not in out
    assert _frequencies(out) == pytest.approx([3.68, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0])
    # the run at 3.68 MHz prints no pattern; each of the six after it does
    first, rest = out.split("INPUT LINE  5", 1)
    assert "RADIATION PATTERNS" not in first
    assert rest.count("RADIATION PATTERNS") == 6
    assert out.count("ANTENNA INPUT PARAMETERS") == 7


def test_a_source_change_after_a_sweep_runs_at_its_last_frequency():
    out = _out(
        ["EX 0 1 11 0 1 0", "FR 0 3 0 0 3.5 0.1", "XQ", "EX 0 2 11 0 1 0", "XQ", "EN"]
    )
    assert _frequencies(out) == pytest.approx([3.5, 3.6, 3.7])
    assert out.count("ANTENNA INPUT PARAMETERS") == 4
    tail = out.split("INPUT LINE  5", 1)[1]
    assert "FREQUENCY=" not in tail  # results only, at 3.7 MHz


def test_an_rp_after_a_sweep_prints_its_pattern_alone():
    out = _out(
        ["EX 0 1 11 0 1 0", "FR 0 2 0 0 3.5 0.1", "XQ", "RP 0 1 3 1000 90 0 0 45", "EN"]
    )
    tail = out.split("INPUT LINE  5", 1)[1]
    assert tail.count("RADIATION PATTERNS") == 1
    assert "ANTENNA INPUT PARAMETERS" not in tail


def test_an_fr_repeating_the_frequency_in_force_is_a_whole_new_block():
    out = _out(
        ["EX 0 1 11 0 1 0", "FR 0 1 0 0 3.6 0", "XQ", "FR 0 1 0 0 3.6 0", "XQ", "EN"]
    )
    assert _frequencies(out) == pytest.approx([3.6, 3.6])


def test_a_later_run_that_changes_a_load_is_refused_by_name():
    out = _out(
        [
            "EX 0 1 11 0 1 0",
            "FR 0 1 0 0 3.6 0",
            "XQ",
            "FR 0 2 0 0 3.5 0.2",
            "LD 4 1 5 0 50 0",
            "XQ",
            "EN",
        ]
    )
    assert "NEC ERROR" in out and "run 2 carries LD" in out
