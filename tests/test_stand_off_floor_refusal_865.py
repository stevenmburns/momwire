"""The stand-off floor is a refusal of the deck on every deck seam (momwire#865).

A bare conductor running alongside the interface below h = 2a is refused by
`BSplineSolver._wire_endpoint_status` with a sentence that names the wire, its
height and the floor. The NEC-4.2 slot has always printed that sentence as a
NEC ERROR (`_nec4.solve` refuses every ValueError); the NEC-5 seam let it
escape to the INTERNAL ERROR frame, because the basis is built inside the
solve rather than the constructor. `StandOffFloorError` names it, still a
ValueError for every other caller, and both seams now refuse it by name.

The C++ cores print the same bytes (`test_razor_core_standalone`'s
`test_the_stand_off_floor_refuses_as_python_does`); razor-2p has no floor and
serves the deck.
"""

from __future__ import annotations

from pathlib import Path

from momwire._surface_height import StandOffFloorError
from momwire.dropin.nec5 import _nec4, _shell

DECKS = Path(__file__).resolve().parent / "fixtures" / "stand_off_floor_865"
NEC5 = DECKS / "bare_radials_below_floor_nec5.nec"
NEC4 = DECKS / "bare_radials_below_floor_nec4.nec"
SENTENCE = (
    " ***** NEC ERROR - wire 0 runs at h = 0.900 mm above the interface, "
    "h/a = 1.50, below the h/a = 2 validity floor for a BARE conductor"
)


def _error_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if "NEC ERROR" in line]


def test_the_floor_is_still_a_value_error():
    assert issubclass(StandOffFloorError, ValueError)


def test_the_nec5_seam_refuses_the_floor_by_name():
    (line,) = _error_lines(_shell.render(NEC5.read_text(), basis="bspline"))
    assert line.startswith(SENTENCE), line
    assert "INTERNAL ERROR" not in line
    assert line.endswith("See stevenmburns/momwire#865"), line


def test_the_nec42_slot_refuses_the_floor_in_the_same_words():
    (line,) = _error_lines(_nec4.render(NEC4.read_text(), basis="bspline"))
    assert line.startswith(SENTENCE), line
