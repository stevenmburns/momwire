"""The drive spelling belongs to the BASIS, not to the deck (momwire#603 U1).

This seam used to have one basis and one spelling of a series EMF at a node:
a ``node_gaps`` port, and a CUT through any ``GW`` a card addressed inside,
because that port needs the node to be a wire end on both sides.  The cut is
what kept the NEC-5 formulation twin off its own dialect — not the physics.
``RazorSolver`` has no node gaps, and a cut that lands one element from a
wire's end leaves a one-segment polyline it refuses outright.

So :func:`~momwire.eznec._serve.build_mesh` now takes the solver class and
reads ``capabilities.node_gaps`` off it.  Three gates carry the unit:

**Nothing moves for bspline.**  Not tested here — it is tested by the 122
captures still round-tripping byte for byte in ``test_eznec_printout.py``,
which is the only gate that could see a change and the reason the spelling is
per-basis rather than replaced.

**The two spellings are one source.**  Measured in-basis below, both ways
round: degree-2 B-splines answer within 2.03 % of themselves across the cut,
and razor answers a junction-knot gap and an interior-knot gap to the last
ulp.

**The corpus reaches razor.**  62 committed captures, 11 served under the cut
and 47 without it, with every one of the remaining 15 named — and on the 42
carrying a printout, razor-nec5's input impedance lands a median 0.00 % from
the licensed engine's own.
"""

from __future__ import annotations

import cmath
import math
import re

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _serve
from momwire.eznec._shell import render
from momwire.razor import RazorSolver
from test_eznec_printout import MANIFEST, deck_text, printout_text

CAPTURE_IDS = tuple(entry["id"] for entry in MANIFEST["captures"])


def mesh_for(cid: str, solver_class: type):
    deck = parse_nec5(deck_text(cid))
    return deck, _serve.build_mesh(
        deck, _serve.structure_of(deck), solver_class=solver_class
    )


# --------------------------------------------------------------------------
# the cut, and its absence


def test_a_node_gap_basis_cuts_the_wire_and_a_delta_gap_basis_does_not():
    """0010 drives ``EX 4,1,6`` — six segments into an eleven-segment dipole.

    Under bspline that address cuts the ``GW`` in two and the source is a
    node gap across the join.  Under razor the wire stays whole and the
    source is a delta gap at that node's own arc length, which on a 0.5 m
    dipole of eleven equal segments is 6/11 of the way along.
    """
    _deck, cut = mesh_for("0010", BSplineSolver)
    assert [(p.first_node, p.last_node) for p in cut.pieces] == [(0, 6), (6, 11)]
    assert len(cut.gaps) == 1 and not cut.feeds

    _deck, whole = mesh_for("0010", RazorSolver)
    assert [(p.first_node, p.last_node) for p in whole.pieces] == [(0, 11)]
    assert not whole.gaps
    (site,) = whole.feeds
    assert site.spelling == "gap"
    assert site.end == "interior"
    assert site.arclength == pytest.approx(0.5 * 6 / 11)
    # The port is the wire's own direction either way: the cut hands the
    # address to the LATER piece's `start`, whose sigma is +1, and an uncut
    # wire's delta gap drives with increasing arc length.
    assert site.weight == 1.0
    assert cut.gaps[0].weight == 1.0


@pytest.mark.parametrize("cid", CAPTURE_IDS)
def test_no_capture_hands_razor_a_piece_the_cut_invented(cid):
    """One polyline per ``GW``, for every deck in the corpus.

    The delta-gap spelling cuts nothing, so the mesh's pieces are the deck's
    own wires — which is also what keeps a one-segment piece from being
    manufactured where a card addressed the node next to a wire's end.
    """
    deck, mesh = mesh_for(cid, RazorSolver)
    assert [p.tag for p in mesh.pieces] == [w.tag for w in deck.wires]
    assert [p.n_elements for p in mesh.pieces] == [w.segment_count for w in deck.wires]


def test_only_the_two_apexes_still_ask_razor_for_a_node_gap():
    """K = 2 is a through-current path and becomes a delta gap; K >= 3 is not.

    At a K >= 3 apex an ARCLENGTH cannot say which of the node's K-1
    through-current tents the gap sits in, so ``feeds`` cannot spell it and
    this seam keeps the node port — which leaves exactly the two decks
    momwire#603 U4 is about, and no others.  Not a statement that the source
    has no meaning there: NEC-5 serves 0013's ``EX 4,5,-1``, and the address
    names the branch through its favored tag.
    """
    needing = [cid for cid in CAPTURE_IDS if mesh_for(cid, RazorSolver)[1].gaps]
    assert needing == ["0013", "0033"]


def test_an_addressed_interior_node_another_wire_lands_on_refuses():
    """A delta gap drives ONE wire's knot; a T-junction there needs the cut.

    No capture writes one — all 78 addressed interior nodes in the corpus are
    nodes their own wire alone passes through — so this is the case that must
    refuse rather than quietly model the stem as unconnected.
    """
    tee = (
        "CM T junction addressed at the stem's landing node\n"
        "CE\n"
        "GW 1,4,0.,-2.,0.,0.,2.,0.,.001\n"
        "GW 2,2,0.,0.,0.,0.,0.,2.,.001\n"
        "GE 0,-1\n"
        "FR 0,1,0,0,30.\n"
        "GN -1\n"
        "EX 4,1,2,0,1.,0.\n"
        "EN\n"
    )
    deck = parse_nec5(tee)
    structure = _serve.structure_of(deck)
    # bspline cuts wire 1 and welds all three ends into one junction.
    assert len(_serve.build_mesh(deck, structure, solver_class=BSplineSolver).gaps) == 1
    with pytest.raises(_serve.ServeRefusal, match="element ends meet"):
        _serve.build_mesh(deck, structure, solver_class=RazorSolver)


# --------------------------------------------------------------------------
# the two spellings are one source


def test_the_cut_and_the_whole_wire_are_one_source_in_one_basis():
    """5 m dipole, 30 MHz, 10 elements, fed at the centre, degree-2 B-splines.

    The same physics twice: cut into two polylines with a node gap across the
    join, and whole with a delta gap at the same arc length.  They differ by
    2.03 % — the expansion is clamped on each side of the node in one and
    runs through it in the other — which is the same class as the 2–6 %
    envelope this seam already reports against the licensed engine, and is
    what makes the spelling a per-basis choice rather than a correction.
    """
    wavelength, half = 299.792458 / 30.0, 2.5
    common = dict(wire_radius=0.001, wavelength=wavelength, degree=2)
    cut = BSplineSolver(
        wires=[
            np.array([[0.0, -half, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, half, 0.0]]),
        ],
        n_per_edge_per_wire=[[5], [5]],
        # An empty `feeds` and not a missing one: leave it out and the solver
        # adds its default gap at wire 0's midpoint, which becomes port 0 and
        # answers 165.87 + 77.186j for a quarter-point drive.
        feeds=[],
        junctions=[[(0, "end"), (1, "start")]],
        node_gaps=[(1, "start", 1.0 + 0j)],
        **common,
    )
    whole = BSplineSolver(
        wires=[np.array([[0.0, -half, 0.0], [0.0, half, 0.0]])],
        n_per_edge_per_wire=[[10]],
        feeds=[(0, half, 1.0 + 0j)],
        **common,
    )
    z_cut = 1.0 / cut.compute_port_solution().y[0, 0]
    z_whole = 1.0 / whole.compute_port_solution().y[0, 0]
    assert z_cut == pytest.approx(80.320 + 44.899j, abs=1e-3)
    assert z_whole == pytest.approx(79.117 + 46.321j, abs=1e-3)
    assert abs(z_cut - z_whole) / abs(z_whole) == pytest.approx(0.0203, abs=5e-4)


def test_razor_reads_a_k2_junction_knot_and_an_interior_knot_alike():
    """The K = 2 half of the spelling, and the reason it needs no cut either.

    A 12 m dipole at 30 MHz split into two 6-element polylines and driven at
    the joint IS the uncut 12-element wire driven at its middle knot — the
    junction's through-current tent and the interior tent are the same
    function.  To the last ulp: 2e-16 in the impedance and 7e-16 across all
    thirteen knot currents, which is the same arithmetic reached through a
    different mesh rather than a second answer that happens to be close.

    And the two ways to NAME that joint are bit for bit equal to each other,
    because the port does not know which address reached it — which is why
    :func:`~momwire.eznec._serve._assign_columns` signs a junction gap off
    the junction's own side A rather than off arrival order.
    """
    wavelength, half = 299.792458 / 30.0, 6.0
    west = np.array([[0.0, -half, 0.0], [0.0, 0.0, 0.0]])
    east = np.array([[0.0, 0.0, 0.0], [0.0, half, 0.0]])
    common = dict(wire_radius=0.001, wavelength=wavelength, nec5_quadrature=True)

    def solved(solver):
        sol = solver.compute_port_solution()
        knots = solver.currents_at_knots(sol.coeffs[:, 0])
        chained = (
            np.concatenate([np.asarray(knots[0]), np.asarray(knots[1])[1:]])
            if len(knots) == 2
            else np.asarray(knots[0])
        )
        return 1.0 / sol.y[0, 0], chained

    named_start = solved(
        RazorSolver(
            wires=[west, east],
            n_per_edge_per_wire=[[6], [6]],
            feeds=[(1, 0.0, 1.0 + 0j)],
            **common,
        )
    )
    named_end = solved(
        RazorSolver(
            wires=[west, east],
            n_per_edge_per_wire=[[6], [6]],
            feeds=[(0, half, 1.0 + 0j)],
            **common,
        )
    )
    interior = solved(
        RazorSolver(
            wires=[np.array([[0.0, -half, 0.0], [0.0, half, 0.0]])],
            n_per_edge_per_wire=[[12]],
            feeds=[(0, half, 1.0 + 0j)],
            **common,
        )
    )
    assert named_start[0] == named_end[0]
    assert np.array_equal(named_start[1], named_end[1])
    for other in (named_start, named_end):
        assert abs(other[0] - interior[0]) / abs(interior[0]) < 1e-14
        assert (
            np.max(np.abs(other[1] - interior[1])) / np.max(np.abs(interior[1])) < 1e-14
        )


def test_the_junction_gap_is_signed_off_side_a_and_not_arrival_order():
    """0011 addresses both ends of one head-to-head join — ``4,1`` and ``5,20``.

    Two wires meeting END to END carry one through current, and the deck's
    two addresses read it along OPPOSITE wire directions, so their weights
    are opposite.  Which one is ``+1`` is momwire's, not this seam's: ``+1 A``
    of a junction tent flows IN along the group's first end
    (:meth:`RazorSolver._junction_wings`), and first means
    :func:`~momwire.eznec._serve._canonical_end`'s order — wire, then start
    before end — which is what ``canonical_groups`` promises a declared spec
    is normalized to.
    """
    _deck, mesh = mesh_for("0011", RazorSolver)
    joint = [s for s in mesh.sites if (s.at.tag, s.at.node) == (4, 1)]
    (site,) = joint
    junction = next(
        members for members in mesh.junctions if (site.piece, site.end) in members
    )
    assert len(junction) == 2
    assert junction == sorted(junction, key=_serve._canonical_end)
    side_a = junction[0] == (site.piece, site.end)
    assert site.weight == (-site.sign if side_a else site.sign)


# --------------------------------------------------------------------------
# the corpus reaches razor

# What each capture razor-nec5 still cannot serve is refused BY, all four of
# them named and none of them a drive spelling.  Written out rather than
# counted so that a deck moving between rows is a diff.
ONE_SEGMENT_WIRE = ("0011", "0029", "0030", "0034", "0035")
APEX_NODE_GAP = ("0013", "0033")
CONTACT_OVER_FINITE_GROUND = ("0021", "0047", "0048", "0110", "0111")
# The near-field point at the base of a contact-fed monopole, which the seam
# refuses under EVERY basis (`test_eznec_serve.py`'s module docstring).
REFUSED_UNDER_BSPLINE_TOO = ("0022", "0107", "0112")
UNSERVED = (
    ONE_SEGMENT_WIRE
    + APEX_NODE_GAP
    + CONTACT_OVER_FINITE_GROUND
    + REFUSED_UNDER_BSPLINE_TOO
)


@pytest.mark.parametrize("cid", CAPTURE_IDS)
def test_razor_nec5_serves_every_capture_but_the_fifteen_named(cid):
    """11 of 62 before U1, 47 after, and the other 15 refuse by name.

    The unit's headline gate, driven through the real seam since U3 made the
    basis selectable.  A deck that starts serving belongs in neither list and
    should be taken OUT of :data:`UNSERVED`; a deck that stops is a
    regression this catches by name rather than by count.

    Every unserved deck comes back as a PRINTOUT carrying a ``NEC ERROR``
    line, never as an exception — which is the whole point of routing the
    capability refusals through :func:`~momwire.eznec._serve.serve` rather
    than letting a constructor raise into EZNEC's face.
    """
    text = render(deck_text(cid), basis="razor-nec5")
    assert ("NEC ERROR" in text) == (cid in UNSERVED)


FLOAT = re.compile(r"-?\d\.\d+E[-+]\d+")


def input_impedance(text: str) -> complex | None:
    """The ``ANTENNA INPUT PARAMETERS`` row's Z, third of its four pairs."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "ANTENNA INPUT PARAMETERS" not in line:
            continue
        for row in lines[i + 1 : i + 8]:
            cells = FLOAT.findall(row)
            if len(cells) >= 6:
                return complex(float(cells[4]), float(cells[5]))
    return None


# One per drive spelling the unit touches, plus the sign gate.  0010 is the
# interior knot (``EX 4,1,6``), 0012 the K = 2 junction knot (``EX 4,2,-1``),
# 0019 the ground contact that never moved (``EX 4,1,-1``), and 0017 is the
# W7EL triple's config C — the one deck whose answer depends on the RELATIVE
# sign of two addresses at one junction (manifest ``w7el_gate``).
ACCURACY_IDS = ("0010", "0012", "0017", "0019")


@pytest.mark.parametrize("cid", ACCURACY_IDS)
def test_razor_nec5_lands_on_the_licensed_engine_it_is_a_twin_of(cid):
    """0.01 %, not the 2–6 % a different formulation costs.

    This is the payoff and it is also the sharpest check on the spelling: a
    delta gap at the wrong arc length, or on the wrong knot, or with the
    wrong sign, does not land on the reference to four figures.  Measured
    across the whole corpus the served decks sit a median 0.00 % and a worst
    0.03 % away; these four are pinned so the claim has a gate.
    """
    served = input_impedance(render(deck_text(cid), basis="razor-nec5"))
    reference = input_impedance(printout_text(cid))
    assert reference is not None and served is not None
    assert abs(served - reference) / abs(reference) < 1e-3


def test_the_w7el_triple_still_answers_series_where_it_must():
    """Config C is 70 % from config A, and a lost junction sign hides that.

    ``2,3`` and ``3,-1`` name the two sides of ONE port, so their EMFs add in
    series; a seam that folds them into one node — or that signs one of them
    wrong — answers A's number for C.  The licensed engine prints
    114.470 + 21.096j for A and B and 195.340 − 57.458j for C, and razor-nec5
    reproduces the pair through the delta-gap spelling.
    """
    z = {
        cid: input_impedance(render(deck_text(cid), basis="razor-nec5"))
        for cid in ("0012", "0014", "0016", "0017")
    }
    assert z["0012"] == z["0014"] == z["0016"]  # one point, two tags, one answer
    assert z["0012"] == pytest.approx(114.470 + 21.096j, abs=5e-3)
    assert z["0017"] == pytest.approx(195.340 - 57.458j, abs=5e-3)


@pytest.mark.parametrize("cid", ACCURACY_IDS)
def test_the_current_table_is_not_printed_180_degrees_out(cid):
    """The one error an impedance cannot see.

    A globally flipped port cancels between the drive and the readout, so Z
    never moves; the current TABLE is where it shows, as a systematic half
    turn.  Read on the largest currents in the table, where a phase is
    meaningful.
    """
    served = _wire_currents(render(deck_text(cid), basis="razor-nec5"))
    reference = _wire_currents(printout_text(cid))
    assert len(served) == len(reference) and served
    pairs = sorted(zip(reference, served), key=lambda p: -abs(p[0]))[:5]
    turn = [abs(cmath.phase(b / a)) for a, b in pairs if abs(a) and abs(b)]
    assert max(turn) < math.radians(5.0)


def _wire_currents(text: str) -> list[complex]:
    """Every element's current from the ``Wire Currents`` table, in order."""
    lines = text.splitlines()
    out: list[complex] = []
    for i, line in enumerate(lines):
        if "Wire Currents" not in line:
            continue
        for row in lines[i + 1 :]:
            cells = FLOAT.findall(row)
            if len(cells) >= 7:
                # X Y Z length, then real and imag; the phase is not E-notation
                out.append(complex(float(cells[4]), float(cells[5])))
            elif out and not row.strip():
                break
        break
    return out
