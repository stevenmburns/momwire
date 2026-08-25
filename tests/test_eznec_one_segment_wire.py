"""The seam's one-segment rule is razor's own rule (momwire#608).

``_refuse_by_capability`` refuses ahead of the solver so that a deck razor
cannot host comes back as a PRINTOUT rather than as an exception EZNEC
discards.  That is worth the duplication only while the two agree, and the
rule it duplicates just moved: it used to be "any one-segment ``GW``" and is
now "a one-segment ``GW`` junctioned at neither end".  Two copies of a rule
that has to stay equal is the shape of momwire#588's defect, so the equality
is gated here rather than assumed:

**the seam refuses a deck exactly when the constructor would.**  Every gate
below asks both sides and compares, over the corpus and over four decks
written to put a one-segment ``GW`` in each topological position.

The corpus measurement is recorded too, because it is what made the narrowing
worth doing: 74 one-segment polylines across the five decks that used to
refuse, every one of them junctioned at one end or both, and not one inert.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _serve
from momwire.eznec._shell import render
from momwire.razor import RazorSolver
from test_eznec_printout import MANIFEST, deck_text

CAPTURE_IDS = tuple(entry["id"] for entry in MANIFEST["captures"])

# The five that used to refuse, and what each declares.  (a) is junctioned at
# both ends, (b) at one; (c) — junctioned at neither — is the row the refusal
# survives for, and the corpus has none of it.
ONE_SEGMENT_CENSUS = {
    #        pieces  1-seg  (a)  (b)  (c)
    "0011": (5, 2, 1, 1, 0),
    "0029": (5, 2, 1, 1, 0),
    "0030": (5, 2, 1, 1, 0),
    "0034": (44, 38, 38, 0, 0),
    "0035": (55, 30, 30, 0, 0),
}


def _mesh_and_ground(text: str):
    deck = parse_nec5(text)
    structure = _serve.structure_of(deck)
    mesh = _serve.build_mesh(deck, structure, solver_class=RazorSolver)
    ground = _serve._ground_kwargs(deck, _serve._medium(deck.ground, 1.0))
    return mesh, ground


def _razor_refuses(mesh, ground) -> bool:
    """Does ``RazorSolver`` itself refuse this mesh for the #608 reason?

    Only that reason.  A capture can be refused for another — five of them are
    ground contact over a finite ground — and those are not this gate's
    business, so they read as False rather than as a refusal.  What keeps the
    False from being vacuous is the caller: the four decks written for this
    module are asserted to CONSTRUCT, so a spurious refusal there is a
    failure rather than an agreement.
    """
    try:
        RazorSolver(
            wires=[piece.points for piece in mesh.pieces],
            n_per_edge_per_wire=[[piece.n_elements] for piece in mesh.pieces],
            feeds=[(0, None, 1.0 + 0j)],
            wire_radius=[piece.radius for piece in mesh.pieces],
            wavelength=10.0,
            **ground,
        )
    except (ValueError, NotImplementedError) as exc:
        return "junctioned at neither end" in str(exc)
    return False


def _razor_constructs(mesh, ground) -> bool:
    """Does it build at all?  The other half of the four-deck gate."""
    RazorSolver(
        wires=[piece.points for piece in mesh.pieces],
        n_per_edge_per_wire=[[piece.n_elements] for piece in mesh.pieces],
        feeds=[(0, None, 1.0 + 0j)],
        wire_radius=[piece.radius for piece in mesh.pieces],
        wavelength=10.0,
        **ground,
    )
    return True


# --------------------------------------------------------------------------
# 1. the two rules agree, deck by deck
# --------------------------------------------------------------------------
# A one-segment GW in each of the four positions it can occupy. The dipole
# they hang off is the same in all four so the only difference is the topology
# of wire 2.
_HEAD = "CM one-segment GW, {}\nCE\n"
_TAIL = "GE 0,-1\nFR 0,1,0,0,30.\nEX 0,1,2,0,1.,0.\nXQ\nEN\n"

BOTH_ENDS = (
    _HEAD.format("junctioned at both ends")
    + "GW 1,4,0.,-2.,0.,0.,-0.5,0.,.001\n"
    + "GW 2,1,0.,-0.5,0.,0.,0.,0.,.001\n"  # the short middle piece
    + "GW 3,4,0.,0.,0.,0.,2.,0.,.001\n"
    + _TAIL
)
ONE_END = (
    _HEAD.format("junctioned at one end")
    + "GW 1,8,0.,-2.,0.,0.,1.5,0.,.001\n"
    + "GW 2,1,0.,1.5,0.,0.,2.,0.,.001\n"  # a one-segment tip
    + _TAIL
)
NEITHER_END = (
    _HEAD.format("junctioned at neither end")
    + "GW 1,8,0.,-2.,0.,0.,2.,0.,.001\n"
    + "GW 2,1,0.5,-0.25,0.,0.5,0.25,0.,.001\n"  # a floater
    + _TAIL
)
# A one-segment GW standing in a PEC plane and meeting no WIRE at all: its
# grounded end carries a tent whose lower wing is its own image, so it is
# junctioned in the sense that matters, and `mesh.junctions` — which lists
# groups of two or more wire ends — does not know that.
GROUNDED_END = (
    _HEAD.format("one end in a PEC plane and no wire at either")
    + "GW 1,8,-2.,0.,3.,2.,0.,3.,.001\n"
    + "GW 2,1,0.5,0.,0.,0.5,0.,0.3,.001\n"
    + "GE 1,-1\nGN 1\nFR 0,1,0,0,30.\nEX 0,1,4,0,1.,0.\nXQ\nEN\n"
)


@pytest.mark.parametrize(
    "name, text, inert",
    [
        ("both ends", BOTH_ENDS, False),
        ("one end", ONE_END, False),
        ("neither end", NEITHER_END, True),
        ("grounded end", GROUNDED_END, False),
    ],
)
def test_the_seam_and_the_constructor_refuse_the_same_decks(name, text, inert):
    """The gate this module exists for.  Both sides asked, both compared."""
    mesh, ground = _mesh_and_ground(text)
    assert any(piece.n_elements == 1 for piece in mesh.pieces), name
    assert bool(_serve._inert_pieces(mesh, ground)) is inert, name
    assert _razor_refuses(mesh, ground) is inert, name
    if not inert:
        # Not merely "did not refuse for #608": it builds.
        assert _razor_constructs(mesh, ground), name


@pytest.mark.parametrize("cid", CAPTURE_IDS)
def test_no_capture_declares_an_inert_wire(cid):
    """And so the narrowing costs the corpus nothing.  Asked of the seam and
    of the constructor separately — a capture that grew one would have to
    fail both, not slip through the half that was updated."""
    mesh, ground = _mesh_and_ground(deck_text(cid))
    assert _serve._inert_pieces(mesh, ground) == []
    assert _razor_refuses(mesh, ground) is False


# --------------------------------------------------------------------------
# 2. what the corpus actually declares
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cid", sorted(ONE_SEGMENT_CENSUS))
def test_the_one_segment_census_is_the_one_the_narrowing_was_measured_over(cid):
    """Recorded so that a deck changing shape is a diff and not a silent
    re-measurement.  Classification is by razor's OWN junction grouping —
    ``_find_junctions``, a 1e-9 first-match — and not by the deck's node
    grid, which fuses at 1e-6 and can therefore call two ends one node that
    this grouping keeps apart."""
    mesh, ground = _mesh_and_ground(deck_text(cid))
    probe = RazorSolver.__new__(RazorSolver)
    probe.wires_polylines = [piece.points for piece in mesh.pieces]
    probe.ground_z = ground.get("ground_z")
    probe._declared_junctions = None
    joined = {end for group in probe._find_junctions() for end in group["ends"]}

    counts = [0, 0, 0]  # (c), (b), (a) by number of joined ends
    for i, piece in enumerate(mesh.pieces):
        if piece.n_elements == 1:
            counts[sum((i, k) in joined for k in ("start", "end"))] += 1

    n_pieces, n_one_seg, n_a, n_b, n_c = ONE_SEGMENT_CENSUS[cid]
    assert len(mesh.pieces) == n_pieces
    assert sum(counts) == n_one_seg
    assert (counts[2], counts[1], counts[0]) == (n_a, n_b, n_c)


def test_the_five_decks_serve_and_name_no_refusal():
    """The headline, end to end through the shell: all five come back as
    printouts, none of them carrying a ``NEC ERROR``."""
    for cid in sorted(ONE_SEGMENT_CENSUS):
        text = render(deck_text(cid), basis="razor-nec5")
        assert "NEC ERROR" not in text, cid
        assert "INTERNAL ERROR" not in text, cid


def test_an_inert_wire_still_comes_back_as_a_printout():
    """The refusal that survives goes through the seam's channel, naming the
    wire and the basis — never as an exception into EZNEC's face."""
    text = render(NEITHER_END, basis="razor-nec5")
    assert "NEC ERROR" in text
    assert "INTERNAL ERROR" not in text
    assert "razor-nec5" in text
    assert "meets no other wire at either end" in text


def test_the_default_basis_hosts_the_inert_wire_it_refuses_for_razor():
    """The refusal is razor's basis and not the seam's opinion of the model:
    the same deck serves under the default."""
    assert "NEC ERROR" not in render(NEITHER_END, basis="bspline")


# --------------------------------------------------------------------------
# 3. the seam's ground question is the solver's ground question
# --------------------------------------------------------------------------
def test_a_grounded_end_counts_as_joined_at_the_seam_too():
    """The one place the seam could disagree by construction: ``mesh.junctions``
    lists groups of TWO OR MORE ends, so a lone grounded end is not in it, and
    reading only that list would refuse a wire razor serves.  ``_inert_pieces``
    asks the ground question with ``_ground_spec.ground_touch_tol`` — the same
    function ``RazorSolver._ground_ends`` asks it with."""
    mesh, ground = _mesh_and_ground(GROUNDED_END)
    (one_seg,) = [i for i, p in enumerate(mesh.pieces) if p.n_elements == 1]
    joined = {end for group in mesh.junctions for end in group}
    assert (one_seg, "start") not in joined and (one_seg, "end") not in joined
    assert _serve._inert_pieces(mesh, ground) == []


def test_free_space_leaves_the_ground_question_unasked():
    """``ground_z`` of None must not read as a plane at z = 0 — the floater in
    :data:`NEITHER_END` has both anchors at z = 0 and would look grounded."""
    mesh, ground = _mesh_and_ground(NEITHER_END)
    assert ground.get("ground_z") is None
    assert np.allclose([p.points[:, 2] for p in mesh.pieces][-1], 0.0)
    assert _serve._inert_pieces(mesh, ground) != []
