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
from momwire.deck._solver import (
    BASES,
    basis_entry,
    basis_from_program_name,
    port_kwargs,
)
from momwire.eznec import _serve
from momwire.eznec._shell import render
from momwire.pulse import PulseSolver
from momwire.razor import RazorSolver
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


def test_the_charge_table_gate_is_asked_of_the_class():
    """The dialect prints a CHARGE DENSITY table and reads it off the basis.

    Razor had no ``current_slopes`` until #603 U2 and the sinusoidal family
    had none until #611; both gates were this one.  The check survives its
    last tenant on purpose — it is asked of the CLASS rather than kept as a
    list of which families qualify, so a family that arrives without the
    method refuses instead of dying on an attribute error.  ``PulseSolver``
    is the standing example: no ``current_slopes``, and the issue that would
    give it one (#611 step 4) has to decide what the column MEANS first,
    because a pulse's charge is two point charges at the segment ends and
    there is no density at a centre to print.
    """
    for cls, _kwargs in BASES.values():
        assert hasattr(cls, "current_slopes"), cls
    assert not hasattr(PulseSolver, "current_slopes")


def test_the_sinusoidal_family_refuses_the_KNOT_DRIVE_by_name():
    """#611 moved this refusal one layer down, and the move is the point.

    It used to read "no ``current_slopes``", and that sentence went stale the
    day the method landed.  What outlives it is the DRIVE: every deck in this
    dialect names a node, and this family resolves a ``feeds`` arclength to
    the nearest segment CENTRE — half a segment away, silently.  Were the
    charge-table gate the only one, all three entries would have started
    serving well-formed printouts about a source in the wrong place.

    The two reasons are different and both are asserted, because a caller who
    reads "not supported" and a caller who reads "not yet wired up" do
    different things next.  The point-matched entry's is permanent (there is
    no collocation pairing for a gap at a knot, #212); the Galerkin pair's is
    plumbing (#648).
    """
    point, galerkin = "does not place a gap there", "not yet served"
    for basis, expect in (
        ("sinusoidal", "Point matching admits no gap at a knot"),
        ("sinusoidal-galerkin", "not yet served"),
        ("sinusoidal-galerkin-converged", "not yet served"),
    ):
        text = render(deck_text("0010"), basis=basis)
        assert "NEC ERROR" in text, basis
        assert point in text, basis
        assert expect in text, basis
        # The refusal that USED to fire must not still be the one talking.
        assert "no such method to read it from" not in text, basis
    assert galerkin not in render(deck_text("0010"), basis="sinusoidal")


def test_razor_now_hosts_the_deck_it_used_to_name_a_refusal_for():
    """0021 stands a wire end in a finite ground plane, and is SERVED.

    This test used to assert the opposite, and the inversion is momwire#624:
    the refusal it pinned — "ground CONTACT over a finite ground" — was
    lifted after §5.5's experiment measured what it protected. The reason to
    keep a test here rather than delete one is that a refusal arriving as a
    PRINTOUT is the seam's own property, independent of which refusals exist:
    the shell's last line of defence would otherwise write ``INTERNAL ERROR
    IN MOMWIRE ENGINE`` and a Python exception name, which tells a ham
    nothing about their model. So the surviving claim is the seam's, on the
    deck that used to exercise it.

    ``test_eznec_serve.py`` owns the printout's numbers; what is checked here
    is only that the deck comes back as an ANSWER, with a real impedance in
    it, and with neither the old refusal nor a leaked traceback.
    """
    text = render(deck_text("0021"), basis="razor-nec5")
    assert "NEC ERROR" not in text
    assert "INTERNAL ERROR" not in text
    assert "FINITE ground plane" not in text
    assert "ANTENNA INPUT PARAMETERS" in text


# --------------------------------------------------------------------------
# the executable NAME is the choice (momwire#593)

# EZNEC owns the command line — two positional paths, nothing else — so the
# basis cannot be a flag and rides on the filename instead.  That makes
# `basis_from_program_name` a piece of the SHIPPED contract rather than a
# helper, and until now nothing in the PR lane touched it through the
# ``eznec-`` marker at all: `test_portal.py` covers the ``nec2c-`` one, and
# the frozen bundle's smoke gate runs only post-merge, on Windows.
#
# The two edges here are the ones that fail SILENTLY, which is why they are
# tests and not a comment.  A Windows filename is case-insensitive, so
# `Momwire-Eznec-Razor-Nec5.exe` is the SAME FILE the README tells the user
# to make; matching the marker case-sensitively gave it ``None`` and served
# the default under a name asking for the twin.  And a name ending at the
# marker asked for a basis and spelt none, which is a typo to refuse, not a
# request for the default — ``suffix or None`` turned it into one.


def _entry_module():
    """`scripts/eznec_freeze/entry.py`, which is not in a package.

    Loaded by path, as `test_portal_shared.py` loads the capture script: it
    is the frozen exe's ``__main__`` and PyInstaller wants a file, so there
    is nothing to import it as.
    """
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parent.parent / "scripts" / "eznec_freeze" / "entry.py"
    )
    spec = importlib.util.spec_from_file_location("eznec_freeze_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "prog,expected",
    [
        # No marker: the plain spelling, a module run, a pytest runner.
        ("momwire-eznec", None),
        ("momwire-eznec.exe", None),
        ("python", None),
        ("/usr/bin/python3.12", None),
        # The shipped pair, and the paths they arrive as.
        ("momwire-eznec-razor-nec5", "razor-nec5"),
        ("momwire-eznec-razor-nec5.exe", "razor-nec5"),
        ("C:\\Users\\ham\\momwire-eznec\\momwire-eznec-razor-nec5.exe", "razor-nec5"),
        ("/opt/momwire-eznec/momwire-eznec-hmatrix", "hmatrix"),
        # Windows filenames are case-insensitive and the user renames by hand.
        ("Momwire-Eznec-Razor-Nec5.exe", "razor-nec5"),
        ("MOMWIRE-EZNEC-RAZOR-NEC5.EXE", "razor-nec5"),
        ("C:\\Program Files\\Momwire-EZNEC-Bspline.Exe", "bspline"),
        # A typo is a name, not the default: both of these must reach a
        # refusal, so both must come back as a suffix.
        ("momwire-eznec-", ""),
        ("momwire-eznec-.exe", ""),
        ("momwire-eznec-rzaor", "rzaor"),
    ],
)
def test_the_eznec_marker_reads_the_basis_off_the_filename(prog, expected):
    assert basis_from_program_name(prog, "eznec-") == expected


@pytest.mark.parametrize(
    "prog,expected",
    [
        ("momwire-nec2c", None),
        ("momwire-nec2c-razor", "razor"),
        ("C:\\SimNEC\\Momwire-Nec2c-Sinusoidal-Galerkin.EXE", "sinusoidal-galerkin"),
        ("momwire-nec2c-", ""),
    ],
)
def test_one_owner_means_the_nec2c_marker_obeys_the_same_rule(prog, expected):
    """The rule is about executable names, not about which front end asks."""
    assert basis_from_program_name(prog, "nec2c-") == expected


@pytest.mark.parametrize(
    "prog,expected",
    [
        ("momwire-eznec.exe", _serve.BASIS),
        ("python", _serve.BASIS),
        ("Momwire-Eznec-Razor-Nec5.exe", "razor-nec5"),
        ("momwire-eznec-", ""),
        ("momwire-eznec-rzaor.exe", "rzaor"),
    ],
)
def test_the_frozen_entry_point_defaults_only_when_no_basis_was_named(prog, expected):
    assert _entry_module().basis_for(prog) == expected


@pytest.mark.parametrize("prog", ["momwire-eznec-", "momwire-eznec-rzaor.exe"])
def test_a_name_that_matches_no_basis_refuses_in_the_printout(prog):
    """The README's promise, end to end: not a silent fallback, a refusal.

    Through `render` because that is the frozen exe's whole body — the basis
    `basis_for` returns is handed straight to it — and because the printout
    is the only channel EZNEC reads.
    """
    basis = _entry_module().basis_for(prog)
    text = render(deck_text("0010"), basis=basis)
    assert "NEC ERROR" in text
    assert f"unknown basis {basis!r}" in text
    assert "ANTENNA INPUT PARAMETERS" not in text


# --------------------------------------------------------------------------
# what the roster serves, recorded deck by deck (momwire#635)

# This used to be a COUNT per basis, which a gain on one deck and a loss on
# another cancel out of.  #510 and #631 both showed how long a divergence
# between the two served bases can sit unnoticed, so the record is per deck
# now and it carries the REASON.  That is the missing half of the refusal
# machinery: ``capabilities.refusals`` says what a solver CLAIMS it will not
# do, and nothing checked what it actually does, deck by deck.
#
# Measured over all 62 committed captures at momwire 0.39.0 and re-measured
# TWICE since, once per capability that landed.  **Every basis outside the
# sinusoidal family now accepts all 59, and nothing raises at all** — the
# strongest and simplest form this table has had, and it was reached in two
# deliberate steps rather than by drift:
#
#   momwire#609 — ``arrayblock`` was 57 + 2 RAISED.  ``_shape_classes`` called
#     four verticals one shape on segment geometry alone, two of them cut
#     mid-wire by an ``LD`` and so carrying 9 bases against the others' 7, and
#     the 9x9 block went at a 7-basis element.  The two decks were never a
#     refusal, and the walk named 0116 and 0117 the day the fix landed.
#
#   momwire#624 — ``razor`` and ``razor-nec5`` were 54, refusing ground
#     CONTACT over a finite ground on 0021/0047/0048/0110/0111.  That refusal
#     was lifted after §5.5's experiment measured what it was protecting: the
#     residual is bounded and saturating, inside the very envelope pins
#     ``BSplineSolver``'s contact row already ships under, and on sea water
#     razor is twenty times closer to the binary than bspline is.
#
# Both were caught by this gate rather than absorbed, which is what it is for.
# #624 fired it HARDER than #609 did: ``CONTACT_OVER_A_FINITE_GROUND`` read
# its prose out of the capabilities roster, so removing that row broke this
# module's IMPORT rather than one assert.  A blunt failure, but it arrives
# before any measurement runs and it cannot be mistaken for a flake.

SERVED = None

# The field AT a contact-fed wire's base over a finite ground is singular, so
# 0022, 0107 and 0112 refuse under EVERY basis: a property of what their ``NE``
# grid asks for, not of any formulation, which is why 59 rather than 62 is what
# "serves the corpus" means.  The sentence itself is owned by
# ``test_eznec_serve.py`` — a fragment is enough here to tell the causes apart.
NEAR_FIELD_AT_A_CONTACT = "asks for the field at (0, 0, 0) metres"

# Every deck in this dialect drives a NODE, and the sinusoidal family puts a
# delta gap at the nearest segment CENTRE instead — so all three entries still
# refuse the whole corpus, and the reason is now the DRIVE rather than the
# charge table (momwire#611).  The prose differs between the point-matched
# entry and the Galerkin pair; the fragment below is the part they share, and
# ``test_the_sinusoidal_family_refuses_the_KNOT_DRIVE_by_name`` above is what
# tells the two apart.
#
# This is the second sentence in this cell, not the first: it used to read
# "no ``current_slopes``", which stopped being true when #611 added the
# method.  A refusal that outlives its own reason is the failure mode this
# per-deck record exists to make visible.
NO_KNOT_FEEDS = "does not place a gap there"

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
    row.update(dict.fromkeys(_SINUSOIDAL, NO_KNOT_FEEDS))
    row.update(refusals)
    return row


# TWO shapes cover all 62 now — four before #609, three before #624 — and
# naming them is what makes the table's intent survive a re-baseline: a deck
# that MOVES between shapes is a one-line diff, and a shape that changes
# membership is a one-line diff.  A shape whose last deck leaves goes with it,
# as ``NOT_ARRAYBLOCK`` and then ``NOT_RAZOR`` both did.
EVERY_BASIS = _row()
# Including the sinusoidal three, which is an ORDERING fact worth having
# written down: these decks are refused for what they ASK FOR before any basis
# is asked whether it can host them, so their cells say the near field rather
# than ``current_slopes``.
NO_BASIS = _row(dict.fromkeys(BASES, NEAR_FIELD_AT_A_CONTACT))

# 59/59 for every basis outside the sinusoidal family, with no asymmetry left
# to record.  razor's road here was 49 -> 54 (momwire#608, which narrowed a
# refusal that used to read "one segment" to what it should always have read,
# "one segment and junctioned at neither end") -> 59 (momwire#624, ground
# contact over a finite ground).
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
    "0021": EVERY_BASIS,
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
    "0047": EVERY_BASIS,
    "0048": EVERY_BASIS,
    # The seven models promoted out of the antennaknobs capture area.
    # Eighteen decks, every one accepted by every basis that answers at
    # all — which is what a coverage batch SHOULD look like arriving, and
    # is why they are worth having: a corpus that only grows by decks the
    # roster already handles stops being able to surprise it.
    "0059": EVERY_BASIS,
    "0060": EVERY_BASIS,
    "0063": EVERY_BASIS,
    "0064": EVERY_BASIS,
    "0067": EVERY_BASIS,
    "0068": EVERY_BASIS,
    "0069": EVERY_BASIS,
    "0070": EVERY_BASIS,
    "0075": EVERY_BASIS,
    "0076": EVERY_BASIS,
    "0077": EVERY_BASIS,
    "0078": EVERY_BASIS,
    "0079": EVERY_BASIS,
    "0080": EVERY_BASIS,
    "0081": EVERY_BASIS,
    "0082": EVERY_BASIS,
    "0083": EVERY_BASIS,
    "0084": EVERY_BASIS,
    "0107": NO_BASIS,
    "0108": EVERY_BASIS,
    "0109": EVERY_BASIS,
    "0110": EVERY_BASIS,
    "0111": EVERY_BASIS,
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
        ("EVERY_BASIS", EVERY_BASIS, 77),
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


def test_every_basis_that_answers_at_all_accepts_the_same_59():
    """Serving parity, as an equality — which is what #635 was asking for.

    The predecessor of this test asserted ``razor < bspline`` and said in its
    own docstring what would happen when momwire#624 closed: "the two lines
    below stop holding and the table wants re-baselining to *both accept all
    59* — a stronger and simpler statement, and one worth arriving at on
    purpose".  #624 landed and that is this test.  The subset form is GONE
    rather than weakened to ``<=``, because a subset claim that is really an
    equality invites the gap to reopen unnoticed on the slack side.

    So the corpus now divides in two, not three: the sinusoidal family, which
    cannot read a CHARGE DENSITY table off its basis and refuses everything,
    and every other basis, which accepts the same 59.  ``hmatrix``,
    ``arrayblock`` and the razor family are their own solvers rather than
    bspline variants — three different bases and two different testing
    schemes — so their agreeing on the accept SET is a measurement, not a
    tautology, and it is the property "we accept the same decks" names.
    """
    bspline = _accepted_by("bspline")
    assert len(bspline) == 77
    for basis in ("bspline-d1", "hmatrix", "arrayblock", "razor", "razor-nec5"):
        assert _accepted_by(basis) == bspline, basis
    assert all(not _accepted_by(basis) for basis in _SINUSOIDAL)
    # The three the whole roster refuses are the corpus's own, not any
    # formulation's: 62 - 59, and they are the near-field cell.
    refused_by_all = set(CAPTURE_IDS) - bspline
    assert refused_by_all == {"0022", "0107", "0112"}
    for cid in refused_by_all:
        assert set(ACCEPTS[cid].values()) == {NEAR_FIELD_AT_A_CONTACT}


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
