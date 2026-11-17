"""Two bs2 refusals the NEC-5 seam printed as INTERNAL ERROR (census 11-17).

The 11-17 exe census ran every deck at hand through each standalone exe and
its Python launcher. On the NEC-5 Python reference two refusals of bs2
(`BSplineSolver`) escaped the seam as exceptions, so the printout carried
"INTERNAL ERROR IN MOMWIRE ENGINE - NotImplementedError" and "...
ValueError" where the executable printed a sentence:

* a crossing node whose BELOW wires differ in radius, which the two-radius
  rule does not serve (`_below_interface.crossing_side_radii`); bs2 builds
  its junctions in the solve, so the NotImplementedError arrived past the
  constructor, where `_serve._prepare` refused only the constructor's;
* a buried deck whose grazing pairs lie past the Sommerfeld range cap, with
  a pair INSIDE the cap still under the grazing floor
  (`_below_interface.BURIED_GRAZING_REFUSAL`, a ValueError from the fill).

`_serve._prepare` now refuses both by name wherever the solve raises them,
and the C++ seams and run half print the same sentence: bs2's front refuses
its NotImplementedError, its buried plan takes Python's capped walk and
raises the grazing sentence, razor's solve hands its fill's exception on as
Python's, and the run half prints either as the refusal (the driver tests
in `tests/test_razor_core_standalone.py` hold the bytes equal). The decks
here are synthetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from momwire import _below_interface
from momwire.deck._nec5 import parse_nec5
from momwire.dropin.nec5 import _serve
from momwire.dropin.nec5._shell import render

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "nec5_refusal_sentences_1117"
SPREAD = FIXTURES / "crossing_radius_spread.nec"
GRAZE = FIXTURES / "buried_grazing_inside_cap.nec"

_SPREAD_SENTENCE = (
    "crossing serve with per-wire radii that differ within the below wires: "
    "the two-radius rule serves one radius per side of the interface (the "
    "node's point tests at the buried radius, antennaknobs plan U5); a spread "
    "within one side has no measured node radius"
)
_GRAZE_HEAD = "this deck's buried wires reach a below/below pair elevation of theta = "


def _text(path: Path) -> str:
    return path.read_bytes().decode("latin-1")


def _error_line(printout: str) -> str:
    (line,) = [ln for ln in printout.splitlines() if "NEC ERROR" in ln]
    return line


@pytest.mark.parametrize("basis", ["bspline", "razor-2p"])
def test_a_radius_spread_below_a_crossing_node_is_refused_by_name(basis):
    line = _error_line(render(_text(SPREAD), basis=basis))
    assert line == " ***** NEC ERROR - " + _SPREAD_SENTENCE
    with pytest.raises(_serve.ServeRefusal, match="differ within the below wires"):
        _serve.serve(parse_nec5(_text(SPREAD)), basis=basis)


def test_bs2_raises_the_spread_in_the_solve_not_the_constructor():
    """Why the constructor's catch did not reach it: bs2 builds its
    junctions in `compute_port_solution`, razor in its constructor."""
    from momwire.bspline import BSplineSolver

    seen = []
    real = BSplineSolver.compute_port_solution

    def spy(self, *a, **k):
        seen.append("solve")
        return real(self, *a, **k)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(BSplineSolver, "compute_port_solution", spy)
        with pytest.raises(_serve.ServeRefusal):
            _serve.serve(parse_nec5(_text(SPREAD)), basis="bspline")
    assert seen == ["solve"]


@pytest.mark.parametrize(
    "basis, theta", [("bspline", "0.01531"), ("razor-2p", "0.01528")]
)
def test_a_grazing_pair_inside_the_cap_is_refused_by_name(basis, theta):
    line = _error_line(render(_text(GRAZE), basis=basis))
    assert line.startswith(" ***** NEC ERROR - " + _GRAZE_HEAD + theta + " deg ")
    assert "among the pairs inside the cap" in line
    assert "(the pair's two depths add to 0.006 m," in line
    assert "INTERNAL ERROR" not in line


def test_only_the_grazing_value_error_is_a_refusal():
    """`_refuse_if_grazing` turns the two plan refusals into a ServeRefusal
    and passes every other ValueError through (an INTERNAL ERROR frame as
    before)."""
    for template in (
        _below_interface.BURIED_GRAZING_REFUSAL,
        _below_interface.BURIED_CROSS_GRAZING_REFUSAL,
    ):
        exc = ValueError(template.format(th=0.01, floor=0.0166667, depth=0.006, r=9.0))
        with pytest.raises(_serve.ServeRefusal):
            _serve._refuse_if_grazing(exc)
    _serve._refuse_if_grazing(ValueError("wires must be non-empty"))
