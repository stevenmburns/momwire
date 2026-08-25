"""The basis is a choice at this seam, and every refusal is a printout (#603 U3).

`_solver_for` used to name ``BSplineSolver`` and pass ``degree=_DEGREE``, so
the NEC-5 drop-in had exactly one formulation and #593 had nothing to ship a
second executable of.  It now takes any name in ``momwire.deck._solver.BASES``
— the same names antennaknobs' ``--basis`` takes and the nec2 front end
resolves — and the per-family kwarg rules are ONE function both front ends
call rather than two lists that have to stay equal.

The gates here are about the choice, not about any one basis's physics:

**Nothing moves for the default.**  Gated where it can be seen — the 122
captures round-tripping byte for byte in ``test_eznec_printout.py``.  Two of
this module's asserts are about the mechanism of that: the roster's
``bspline`` entry carries no ``degree``, and it does not need to.

**Every family answers or refuses; none of them raises.**  A basis that
cannot host a deck says so in the printout, naming the basis and the reason,
because the process EZNEC launches has no other channel — it discards the
exit code and never reads stderr.

**What each family actually serves is recorded**, so a change is a diff.
"""

from __future__ import annotations

import pytest

from momwire import BSplineSolver
from momwire.array_block import ArrayBlockSolver
from momwire.deck._nec5 import parse_nec5
from momwire.deck._solver import BASES, basis_entry, port_kwargs
from momwire.eznec import _serve
from momwire.eznec._shell import render
from momwire.razor import RazorSolver
from momwire.sinusoidal import SinusoidalSolver
from test_eznec_printout import MANIFEST, deck_text

CAPTURE_IDS = tuple(entry["id"] for entry in MANIFEST["captures"])


# --------------------------------------------------------------------------
# the shared rule, which is the half of this unit that is not the seam's


def test_port_kwargs_withholds_from_razor_what_razor_refuses():
    """``junctions`` is withheld and ``node_gaps`` is never handed over empty.

    Both are refusals by PRESENCE: ``RazorSolver`` raises on a ``node_gaps=``
    it was given even when the value is ``None``, so a caller that writes
    ``node_gaps=gaps or None`` refuses every deck rather than the ones with a
    node gap in them.  That mistake cost a whole "razor serves 0 of 62"
    measurement in #603, which is why the rule has one owner now.
    """
    assert port_kwargs(RazorSolver) == {}
    assert port_kwargs(RazorSolver, junctions=[[(0, "end"), (1, "start")]]) == {}
    assert port_kwargs(RazorSolver, node_gaps=[]) == {}
    assert port_kwargs(RazorSolver, node_gaps=[(1, "start", 0j)]) == {
        "node_gaps": [(1, "start", 0j)]
    }
    assert port_kwargs(RazorSolver, lumped_loads=[(0, 1.0, 5 + 0j)]) == {
        "lumped_loads": [(0, 1.0, 5 + 0j)]
    }


def test_port_kwargs_hands_a_spline_family_its_junctions():
    """Everything not natively loading gets the key, ``None`` when empty.

    ``junctions=None`` and no ``junctions`` at all are the same thing to a
    family that declares the parameter, and the empty spelling is the one
    that has been going in since before there was a rule; keeping it is what
    makes the extraction a pure move.
    """
    assert port_kwargs(BSplineSolver) == {"junctions": None}
    assert port_kwargs(BSplineSolver, junctions=[]) == {"junctions": None}
    assert port_kwargs(ArrayBlockSolver, junctions=[[(0, "end")]]) == {
        "junctions": [[(0, "end")]]
    }


def test_an_unknown_basis_names_the_ones_that_are_known():
    with pytest.raises(ValueError, match="unknown basis 'bspline-d2'"):
        basis_entry("bspline-d2")
    message = str(pytest.raises(ValueError, basis_entry, "nope").value)
    for name in BASES:
        assert repr(name) in message


# --------------------------------------------------------------------------
# the seam


def test_the_seam_takes_a_basis_and_answers_in_it():
    """0010 is a free-space dipole every family below can host.

    Three DIFFERENT numbers from three formulations of one antenna, which is
    the whole of what "selectable" has to mean.  razor-nec5 is the twin, so
    it is the one that lands on the licensed engine's 79.948 + 29.919j.
    """
    deck = parse_nec5(deck_text("0010"))
    z = {
        basis: _serve.serve(deck, basis=basis).sources[0].impedance
        for basis in ("bspline", "bspline-d1", "razor-nec5")
    }
    assert len({(round(v.real, 6), round(v.imag, 6)) for v in z.values()}) == 3
    assert z["razor-nec5"] == pytest.approx(79.948 + 29.919j, abs=5e-3)
    # The default and the name of the default are one answer.
    assert _serve.serve(deck).sources[0].impedance == z[_serve.BASIS]


def test_an_unknown_basis_refuses_in_the_printout_rather_than_raising():
    text = render(deck_text("0010"), basis="rzaor")
    assert "NEC ERROR" in text
    assert "unknown basis 'rzaor'" in text


def test_a_family_with_no_current_slopes_refuses_by_name():
    """The dialect prints a CHARGE DENSITY table and reads it off the basis.

    The three sinusoidal entries have no ``current_slopes``; razor had none
    either until #603 U2 gave it one.  Asked of the class rather than kept as
    a list of which families qualify — a list is the second thing to update.
    """
    assert not hasattr(SinusoidalSolver, "current_slopes")
    text = render(deck_text("0010"), basis="sinusoidal")
    assert "NEC ERROR" in text
    assert "current_slopes" in text


@pytest.mark.parametrize(
    "cid, fragment",
    [
        ("0021", "FINITE ground plane"),  # ground contact — #603 U5
        ("0011", "one segment long"),  # a one-segment GW — momwire#608
    ],
)
def test_what_razor_cannot_host_is_named_in_the_printout(cid, fragment):
    """Each of the three is a refusal momwire would raise anyway.

    Moved one step forward so it arrives as a printout: the shell's last line
    of defence would otherwise write ``INTERNAL ERROR IN MOMWIRE ENGINE`` and
    a Python exception name, which tells a ham nothing about their model.
    The prose is the solver's own ``capabilities`` row where there is a cell
    for it, not a second copy kept here.
    """
    text = render(deck_text(cid), basis="razor-nec5")
    assert fragment in text
    assert "INTERNAL ERROR" not in text
    assert "razor-nec5" in text


# --------------------------------------------------------------------------
# what the roster serves, recorded

# Measured over all 62 committed captures.  Three of them refuse under EVERY
# basis (a near-field point at a contact, `test_eznec_serve.py`), so 59 is
# what "serves the corpus" means here.
#
#   arrayblock  57 + 2 RAISED — momwire#609, a broadcast bug on a deck that
#               carries junctions and node gaps at once.  Recorded as it is
#               rather than papered over; when #609 lands this becomes 59.
SERVE_MATRIX = {
    "bspline": 59,
    "bspline-d1": 59,
    "hmatrix": 59,
    "arrayblock": 57,
    "sinusoidal": 0,
    "sinusoidal-galerkin": 0,
    "sinusoidal-galerkin-converged": 0,
    "razor": 49,
    "razor-nec5": 49,
}


def test_the_roster_is_the_one_the_matrix_was_measured_over():
    assert set(SERVE_MATRIX) == set(BASES)


@pytest.mark.slow
@pytest.mark.parametrize("basis", sorted(SERVE_MATRIX))
def test_each_basis_serves_what_it_was_measured_to_serve(basis):
    """The serve matrix, one basis at a time (#603 U6's seed).

    ``momwire#609`` is the one basis that still RAISES rather than refusing,
    on two decks; it is allowed for here by name so that a new raise anywhere
    else is a failure rather than a silently absorbed one.
    """
    served = raised = 0
    for cid in CAPTURE_IDS:
        try:
            served += "NEC ERROR" not in render(deck_text(cid), basis=basis)
        except ValueError:
            raised += 1
    assert served == SERVE_MATRIX[basis]
    assert raised == (2 if basis == "arrayblock" else 0)
