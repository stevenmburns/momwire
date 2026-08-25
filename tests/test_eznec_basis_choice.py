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

**What each family actually serves is recorded, deck by deck**, so a change is
a diff.  momwire#635 made that record per DECK and gave it the REASON: a count
per basis is cancelled out by a gain on one deck and a loss on another, and
``capabilities.refusals`` only says what a solver CLAIMS it will not do.
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
    it is the one that REPRODUCES the licensed engine's 79.948 + 29.919j —
    a statement about which reference it tracks, not about which of the
    three is nearest the converged answer.
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
# what the roster serves, recorded deck by deck (momwire#635)

# This used to be a COUNT per basis, which a gain on one deck and a loss on
# another cancel out of.  #510 and #631 both showed how long a divergence
# between the two served bases can sit unnoticed, so the record is per deck
# now and it carries the REASON.  That is the missing half of the refusal
# machinery: ``capabilities.refusals`` says what a solver CLAIMS it will not
# do, and nothing checked what it actually does, deck by deck.
#
# Measured over all 62 committed captures at momwire 0.39.0, and re-measured
# on top of momwire#609.  Three causes account for every refusal in the corpus
# and NOTHING RAISES AT ALL, which is a smaller thing to say than the first
# measurement could: ``arrayblock`` was 57 + 2 RAISED, and #609 makes it 59
# and silent, which is parity with ``bspline``.  The two decks were never a
# refusal — ``_shape_classes`` called four verticals one shape on segment
# geometry alone, two of them cut mid-wire by an ``LD`` and so carrying 9
# bases against the others' 7, and the 9x9 block went at a 7-basis element.
#
# That re-measure is this gate's first firing, and it is the shape the gate
# was built for: #609 landed while this was in review, the walk named 0116 and
# 0117 as decks arrayblock now accepts, and the table had to be restated
# rather than a count quietly agreeing.

SERVED = None

# The field AT a contact-fed wire's base over a finite ground is singular, so
# 0022, 0107 and 0112 refuse under EVERY basis: a property of what their ``NE``
# grid asks for, not of any formulation, which is why 59 rather than 62 is what
# "serves the corpus" means.  The sentence itself is owned by
# ``test_eznec_serve.py`` — a fragment is enough here to tell the causes apart.
NEAR_FIELD_AT_A_CONTACT = "asks for the field at (0, 0, 0) metres"

# momwire#624, and the whole of razor's asymmetry: five decks stand a wire end
# in a finite ground plane.  Not a COPY of the prose — this IS the solver's own
# capabilities row, so a refusal that keeps firing on the same decks for a
# different reason cannot pass by matching a fragment kept here.
CONTACT_OVER_A_FINITE_GROUND = RazorSolver.capabilities.refusals[
    "contact+finite_ground"
]

# The dialect prints a CHARGE DENSITY table and reads it off the basis; the
# three sinusoidal entries have no ``current_slopes`` to read it from, so they
# refuse the whole corpus.  Named above by
# ``test_a_family_with_no_current_slopes_refuses_by_name``.
NO_CURRENT_SLOPES = "has no such method to read it from"

# The cells that RAISE rather than refuse — EMPTY since momwire#609, and kept
# as its own statement rather than deleted with its last tenant.  The
# difference is the whole of what an operator sees: the shell's last line of
# defence writes ``INTERNAL ERROR IN MOMWIRE ENGINE`` and a Python exception
# name, where a refusal writes a sentence about their model.  Empty is the
# claim "no basis raises on any deck in this corpus", and the walk below still
# checks it, so a new raise anywhere fails here rather than being absorbed.
RAISED = frozenset()

_SINUSOIDAL = ("sinusoidal", "sinusoidal-galerkin", "sinusoidal-galerkin-converged")


def _row(refusals=()):
    """One row over the WHOLE roster: every basis to its cause, or to SERVED.

    The sinusoidal three refuse every deck in the corpus, so they are the
    row's floor rather than three entries each shape below repeats.
    """
    row = dict.fromkeys(BASES, SERVED)
    row.update(dict.fromkeys(_SINUSOIDAL, NO_CURRENT_SLOPES))
    row.update(refusals)
    return row


# Three shapes cover all 62 — it was four before #609 — and naming them is what
# makes the table's intent survive a re-baseline: a deck that MOVES between
# shapes is a one-line diff, and a shape that changes membership is a one-line
# diff.  A shape whose last deck leaves goes with it, as ``NOT_ARRAYBLOCK`` did.
EVERY_BASIS = _row()
NOT_RAZOR = _row(dict.fromkeys(("razor", "razor-nec5"), CONTACT_OVER_A_FINITE_GROUND))
# Including the sinusoidal three, which is an ORDERING fact worth having
# written down: these decks are refused for what they ASK FOR before any basis
# is asked whether it can host them, so their cells say the near field rather
# than ``current_slopes``.
NO_BASIS = _row(dict.fromkeys(BASES, NEAR_FIELD_AT_A_CONTACT))

# 59/59 for every basis outside the razor family, 54/59 for razor, and the
# asymmetry is exactly the five NOT_RAZOR decks with exactly one cause.
# razor's 49 -> 54 was momwire#608, which narrowed a refusal that used to read
# "one segment" to what it should always have read, "one segment and junctioned
# at neither end".
ACCEPTS = {
    "0000": EVERY_BASIS,
    "0001": EVERY_BASIS,
    "0002": EVERY_BASIS,
    "0003": EVERY_BASIS,
    "0004": EVERY_BASIS,
    "0005": EVERY_BASIS,
    "0006": EVERY_BASIS,
    "0007": EVERY_BASIS,
    "0008": EVERY_BASIS,
    "0009": EVERY_BASIS,
    "0010": EVERY_BASIS,
    "0011": EVERY_BASIS,
    "0012": EVERY_BASIS,
    "0013": EVERY_BASIS,
    "0014": EVERY_BASIS,
    "0015": EVERY_BASIS,
    "0016": EVERY_BASIS,
    "0017": EVERY_BASIS,
    "0018": EVERY_BASIS,
    "0019": EVERY_BASIS,
    "0020": EVERY_BASIS,
    "0021": NOT_RAZOR,
    "0022": NO_BASIS,
    "0023": EVERY_BASIS,
    "0024": EVERY_BASIS,
    "0025": EVERY_BASIS,
    "0026": EVERY_BASIS,
    "0027": EVERY_BASIS,
    "0028": EVERY_BASIS,
    "0029": EVERY_BASIS,
    "0030": EVERY_BASIS,
    "0031": EVERY_BASIS,
    "0032": EVERY_BASIS,
    "0033": EVERY_BASIS,
    "0034": EVERY_BASIS,
    "0035": EVERY_BASIS,
    "0036": EVERY_BASIS,
    "0037": EVERY_BASIS,
    "0038": EVERY_BASIS,
    "0039": EVERY_BASIS,
    "0040": EVERY_BASIS,
    "0041": EVERY_BASIS,
    "0042": EVERY_BASIS,
    "0043": EVERY_BASIS,
    "0044": EVERY_BASIS,
    "0045": EVERY_BASIS,
    "0046": EVERY_BASIS,
    "0047": NOT_RAZOR,
    "0048": NOT_RAZOR,
    "0107": NO_BASIS,
    "0108": EVERY_BASIS,
    "0109": EVERY_BASIS,
    "0110": NOT_RAZOR,
    "0111": NOT_RAZOR,
    "0112": NO_BASIS,
    "0113": EVERY_BASIS,
    "0114": EVERY_BASIS,
    "0115": EVERY_BASIS,
    "0116": EVERY_BASIS,
    "0117": EVERY_BASIS,
    "0120": EVERY_BASIS,
    "0121": EVERY_BASIS,
}


def _accepting(cid):
    return frozenset(b for b, cause in ACCEPTS[cid].items() if cause is SERVED)


def _accepted_by(basis):
    return frozenset(cid for cid in ACCEPTS if ACCEPTS[cid][basis] is SERVED)


def test_the_table_covers_every_capture_and_every_basis():
    """Both axes, and every named shape earning its tenancy.

    A shape whose last deck moves away is dead prose that still reads as a
    statement about the roster, so it is a failure rather than a leftover.
    """
    assert set(ACCEPTS) == set(CAPTURE_IDS)
    for cid, row in ACCEPTS.items():
        assert set(row) == set(BASES), cid
    shapes = (
        ("EVERY_BASIS", EVERY_BASIS, 54),
        ("NOT_RAZOR", NOT_RAZOR, 5),
        ("NO_BASIS", NO_BASIS, 3),
    )
    for name, shape, count in shapes:
        assert sum(row is shape for row in ACCEPTS.values()) == count, name
    assert sum(count for _, _, count in shapes) == len(CAPTURE_IDS)
    # Vacuous while nothing raises, and a constraint on the next edit rather
    # than a leftover: a cell added to RAISED has to be one the table already
    # knows is not SERVED, or the walk below would read it as an accepted deck
    # gone missing rather than as a bug.
    for cid, basis in RAISED:
        assert ACCEPTS[cid][basis] is not SERVED


def test_razor_accepts_a_strict_subset_of_what_bspline_accepts():
    """The asymmetry is one-directional, and this says so about the TABLE.

    Stated against the expectation rather than against a walk of the corpus,
    deliberately: a re-baseline that INVERTS the relationship has to delete
    this test rather than absorb it into a new set of numbers.  Anything razor
    serves, bspline serves; there are zero decks the other way; and every deck
    in the gap carries the same one reason, which is #624.  Closing that takes
    the difference to the empty set, at which point the two lines below stop
    holding and the table wants re-baselining to "both accept all 59" — a
    stronger and simpler statement, and one worth arriving at on purpose.
    """
    razor, bspline = _accepted_by("razor"), _accepted_by("bspline")
    assert razor < bspline
    assert len(razor) == 54 and len(bspline) == 59
    assert bspline - razor == {"0021", "0047", "0048", "0110", "0111"}
    for cid in bspline - razor:
        assert ACCEPTS[cid]["razor"] == CONTACT_OVER_A_FINITE_GROUND
    # The two entries of a family are one formulation and differ only in a
    # quadrature rule or a degree, so they accept the same decks.  A split
    # inside a family is a finding of its own.
    assert razor == _accepted_by("razor-nec5")
    # And since momwire#609 the corpus divides in three: the razor family at
    # 54, the sinusoidal family at 0, and EVERYTHING ELSE at the same 59 —
    # ``hmatrix`` and ``arrayblock`` are their own solvers rather than bspline
    # variants, so their agreeing here is a measurement and not a tautology.
    for basis in ("bspline-d1", "hmatrix", "arrayblock"):
        assert _accepted_by(basis) == bspline, basis
    assert all(not _accepted_by(basis) for basis in _SINUSOIDAL)


@pytest.mark.slow
@pytest.mark.parametrize("basis", sorted(BASES))
def test_each_basis_accepts_exactly_the_decks_the_table_names(basis):
    """The walk itself: 62 decks through the shell, in one basis.

    Both directions of drift fail here.  A basis that newly accepts a deck and
    one that newly refuses one both change the first assert; a refusal that
    survives for a DIFFERENT reason changes the second, so a matching count
    can no longer absorb a change of cause.  momwire#609 exercised the first
    of those while this gate was in review: arrayblock stopped raising on 0116
    and 0117, the walk named both, and the table was restated.

    No basis raises now, and the last assert is what says so — the allowance
    that used to sit here by name is an empty ``RAISED`` instead, so a raise
    anywhere is a failure rather than a silently absorbed one.
    """
    accepted, causes = set(), {}
    for cid in CAPTURE_IDS:
        try:
            text = render(deck_text(cid), basis=basis)
        except ValueError as exc:
            causes[cid] = ("raised", str(exc))
            continue
        if "NEC ERROR" in text:
            causes[cid] = ("refused", text.split(" ***** NEC ERROR - ")[1])
        else:
            accepted.add(cid)

    assert accepted == _accepted_by(basis)
    for cid, (how, reason) in sorted(causes.items()):
        expected = ACCEPTS[cid][basis]
        assert expected in reason, f"{basis} refuses {cid} for a new reason: {reason}"
        assert how == ("raised" if (cid, basis) in RAISED else "refused"), cid
