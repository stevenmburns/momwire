"""The wire ``EX`` drive on a REACTIVE deck — where the knot feed already is
(momwire#703).

momwire#703 read a ~15 Ω gap between this seam and the licensed engine on the
insulated-base class (10 m vertical, base 0.5 m up, ``EX 4,1,7`` on fifteen
segments over soil A at 7 MHz, |Z| ≈ 980) and asked the seam to "adopt the
knot-feed spelling — split the fed wire at the EX segment's CENTRE, declare
the 2-member junction, drive through a node gap".

Measured before changing anything, and the ask does not survive the
measurement.  Two findings, each independently sufficient:

**The seam already spells the knot feed.**  A ``GW`` addressed inside by a
card is CUT there under a node-gap basis (``_serve.build_mesh``, "Two ways to
spell one series EMF"), the two halves are declared as a 2-member junction,
``feeds`` reaches the solver EMPTY and the drive is a ``node_gaps`` series
EMF across the join.  That is the #449 spelling, card for card, and it has
been the bspline seam's spelling since momwire#603 U1.  The gates below hold
it, because nothing else in the suite states it as an end-to-end fact.

**The split belongs at the ADDRESSED NODE, not at the fed segment's centre.**
``EX 4,1,7`` on a fifteen-segment 10 m wire drives the node 4.6667 m from the
top; the fed segment's centre is 4.3333 m from the top, a half element away.
Two independent measurements say the node is the engine's own drive point:

* the engine's PRINTED element-current table reconstructs — outward from
  each free end, where the current is zero, through its own element ⇒
  half-sum-of-end-nodes convention — to exactly ``1.0000`` A at node ``N``
  and nowhere else, on every rung of the density ladder (15 segments with
  ``EX ...,7``: 0.9999980; 45 with ``EX ...,21``: 1.0000026; 75 with
  ``EX ...,33``: 1.0000052, each chain closing on the far end to 1e-6).
  ``1.0000`` A is the ``EX 4`` card's own set current, so the node it lands
  on is the driven node.
* sweeping the card across the wire (``EX 4,1,N`` for N = 4…10, one engine
  run each) and following the RESISTANCE, which the feed-gap convention
  barely touches: split at the node, momwire tracks the engine within
  −0.19…−0.10 Ω across the whole sweep; split at the segment centre it
  drifts −0.27 → −0.81 Ω, monotonically, with position.

The segment-centre split only LOOKS better at N = 7 because two errors cross
there.  Its reactance error runs −92 → +69 Ω across the same sweep and
happens to pass through zero one card short of the headline deck.

**So where did the 15 Ω go?**  Into the ENGINE's own coarse-mesh error, not
into a spelling.  Held at one fed node (4.6667 m from the top) and refined,
the two instruments' ladders are (engine, printed impedances only, measured
2026-08-28; momwire through this seam):

===========  =====================  ===========  =====================
engine n     engine Z               momwire n    momwire Z
===========  =====================  ===========  =====================
15           23.129 − 983.49j       15           23.0319 − 968.1930j
45           23.066 − 972.87j       30           22.9893 − 966.6171j
75           23.032 − 970.25j       60           22.9553 − 965.4763j
105          23.012 − 968.95j       120          22.9264 − 964.5840j
135          22.998 − 968.14j       240          22.8992 − 963.8116j
195          22.979 − 967.15j
285          22.960 − 966.30j
===========  =====================  ===========  =====================

The engine's x1 print sits 17.2 Ω from its OWN deepest rung; momwire's x1
print sits 4.4 Ω from its own.  At the deepest rung each instrument reached,
the two answers are 2.5 Ω apart on |Z| ≈ 966 — 0.26 %, with both ladders
still climbing at about 0.8 Ω per doubling and in the same direction.  The
"gap" #703 measured is three quarters the engine's discretization error and
was compared x1 against x1.

Nothing is gated against the engine's numbers here: they are recorded as
context and the gates hold momwire's own values against DRIFT, which is the
house rule for a cross-engine comparison whose two ladders have not met.

**The tree already said this, and the measurement below is the loud version
of it.**  ``_check_basis_can_host`` refuses, by name, any family that would
resolve one of this seam's ``feeds`` arclengths to the nearest segment
CENTRE (momwire#611), and its comment carries the reason — *"that is what a
node address means in this dialect"* — with the corpus measurement behind
it: 75 of the 77 servable captures would have moved, every one of them by
exactly 0.500 h.  momwire#703 proposed making that same half-element move
deliberately, on the main path.  What the corpus could not say is how MUCH
0.500 h is worth, because it is resonant: G-703-4 below prices it at 13.65 Ω
where |Z| ≈ 968 and 0.24 Ω where |Z| ≈ 103.

Also related: momwire#673 wants the mirror capability cell (``centre_feeds``
— nothing declares that a family cannot place a gap at a segment centre).
That axis is the NEC-2 dialect's, which addresses centres; this dialect
addresses nodes, so nothing here becomes a consumer of it and #673 is
untouched by these gates.

Courtesy: every engine fact above is a printed impedance or a printed current
table off runs of the licensed binary.  No source, algorithm or internal
structure is described or relied on.
"""

from __future__ import annotations

import numpy as np
import pytest

from momwire.bspline import BSplineSolver
from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _serve

# --------------------------------------------------------------------------
# the two decks: the reactive headline and its resonant control
# --------------------------------------------------------------------------

DECK = (
    "CM elevated-detached ref h=0.5\n"
    "CE\n"
    "GW 1,{n},0.,0.,{top},0.,0.,{bot},.001\n"
    "GE 1,-1\n"
    "FR 0,1,0,0,7.\n"
    "GN 0,0,0,0,13.,.005\n"
    "EX 4,1,{seg},0,1.,0.\n"
    "XQ 0\n"
    "EN\n"
)

# (top, bottom, segments, EX segment).  E1 is the |Z| ~ 980 insulated-base
# headline; E2 is the resonant 21 m dipole control from the same panel, whose
# whole job is to be the same question asked where Z^2 makes the answer small.
E1 = (10.5, 0.5, 15, 7)
E2 = (21.25, 0.25, 21, 11)

# The seam's own banked prints.  E1's is the number momwire#703 quotes as
# "23.03 - 968.19j"; E2's is the resonant control it quotes as
# "100.77 + 19.59j".
Z_E1_X1 = complex(23.031871, -968.192987)
Z_E1_X2 = complex(22.989300, -966.617100)
Z_E1_X4 = complex(22.955300, -965.476300)
Z_E2_X1 = complex(100.772931, 19.587015)


def served(top, bot, n, seg):
    """The impedance this seam prints for one rung of the family."""
    deck = parse_nec5(DECK.format(top=top, bot=bot, n=n, seg=seg))
    return _serve.serve(deck).sources[0].impedance


def mesh_for(top, bot, n, seg):
    deck = parse_nec5(DECK.format(top=top, bot=bot, n=n, seg=seg))
    structure = _serve.structure_of(deck)
    return (
        deck,
        structure,
        _serve.build_mesh(deck, structure, solver_class=BSplineSolver),
    )


def split_fed(top, bot, z_split, n_above, n_below):
    """The #449 knot feed built by hand at an ARBITRARY split height.

    The same five arguments ``_ward_split_fed`` passes in
    ``test_taper_agreement_floor.py`` — two wires meeting at the split, the
    2-member junction declared, the drive a ``node_gaps`` series EMF, and
    ``feeds=[]`` EXPLICIT (omitting it engages the legacy default feed and
    silently double-drives the deck).  It exists here so the gates can build
    the MIS-located spelling, which the seam itself cannot be asked for.
    """
    wavelength = _serve.SPEED_OF_LIGHT_MHZ_M / 7.0
    solver = BSplineSolver(
        wires=[
            np.array([[0.0, 0.0, top], [0.0, 0.0, z_split]]),
            np.array([[0.0, 0.0, z_split], [0.0, 0.0, bot]]),
        ],
        n_per_edge_per_wire=[[n_above], [n_below]],
        wire_radius=0.001,
        wavelength=wavelength,
        feeds=[],
        junctions=[[(0, "end"), (1, "start")]],
        node_gaps=[(0, "end", 1.0 + 0j)],
        ground_z=0.0,
        ground_eps=(13.0, 0.005),
        ground_model="sommerfeld",
    )
    return complex(np.atleast_1d(solver.compute_impedance()[0])[0])


# --------------------------------------------------------------------------
# G-703-1 — the fed knot is the ADDRESSED NODE
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_g703_1_the_fed_knot_lands_on_the_addressed_node():
    """``EX 4,1,7`` cuts a fifteen-segment wire 7 elements down, not 6.5.

    The coordinate is the whole of momwire#703's mistake, so it is gated as a
    coordinate: the split point is the addressed NODE (4.6667 m from the top,
    z = 5.8333) and is NOT the fed segment's centre (4.3333 m, z = 6.1667).
    Every other segment boundary survives the cut — the two pieces carry 7
    and 8 elements, which is what keeps the printout's per-element current
    and charge rows on the deck's own report points.
    """
    top, bot, n, seg = E1
    _deck, _structure, mesh = mesh_for(top, bot, n, seg)

    assert [(p.first_node, p.last_node) for p in mesh.pieces] == [(0, 7), (7, 15)]
    assert [p.n_elements for p in mesh.pieces] == [7, 8]

    z_split = float(mesh.pieces[0].points[1][2])
    element = (top - bot) / n
    assert z_split == pytest.approx(top - seg * element)  # the node, 5.83333
    assert z_split != pytest.approx(top - (seg - 0.5) * element)  # not 6.16667

    # Every deck element is still exactly one mesh element, in order — the
    # round-trip gates' report points are untouched by the cut.
    assert mesh.element_of == [(0, k) for k in range(7)] + [(1, k) for k in range(8)]


# --------------------------------------------------------------------------
# G-703-2 — the spelling reaching the solver IS the knot feed
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_g703_2_the_seam_hands_the_solver_the_knot_feed_spelling():
    """Split, 2-member junction, ``feeds=[]``, drive through a node gap.

    momwire#703 asked for this spelling as if it were missing.  It is not,
    and no gate said so end to end: the previous ones stop at the mesh.  This
    one asks the constructed SOLVER, which is where the ``feeds=[]`` trap
    lives — a non-empty ``feeds`` here would be the #449 double-drive.
    """
    top, bot, n, seg = E1
    deck, _structure, mesh = mesh_for(top, bot, n, seg)

    # the mesh's side of it
    assert not mesh.feeds
    ((site,),) = (mesh.gaps,)
    assert site.spelling == "node"
    assert (site.piece, site.end) == (1, "start")
    assert mesh.junctions == [[(0, "end"), (1, "start")]]

    # the solver's side of it
    wavelength = _serve.SPEED_OF_LIGHT_MHZ_M / 7.0
    medium = _serve._medium(deck.ground, wavelength)
    solver = _serve._solver_for(deck, mesh, wavelength, medium, BSplineSolver)
    assert solver.feeds == []
    assert solver.node_gaps == [(1, "start", 0j)]
    assert len(solver.junctions) == 1 and len(solver.junctions[0]) == 2


# --------------------------------------------------------------------------
# G-703-3 — the reactive print, and how converged it is
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_g703_3_the_reactive_deck_print_is_banked():
    """The headline deck's served impedance, held against drift.

    Context, gated against nothing: the licensed engine prints
    23.129 − 983.49j for this deck at x1 and 22.960 − 966.30j at x19, so its
    own x1 print is 17.2 Ω from its own deepest rung.  The bar here is 0.5 Ω,
    which is 27× under the half-element mis-location G-703-4 measures and
    tight enough that any real movement in the drive spelling shows.
    """
    z = served(*E1)
    assert abs(z - Z_E1_X1) <= 0.5, (
        f"the reactive headline deck now serves {z:.4f}, "
        f"{abs(z - Z_E1_X1):.4f} Ω from the banked {Z_E1_X1:.4f}"
    )

    z2 = served(10.5, 0.5, 30, 14)
    assert abs(z2 - Z_E1_X2) <= 0.5
    z4 = served(10.5, 0.5, 60, 28)
    assert abs(z4 - Z_E1_X4) <= 0.5

    # The ladder is monotone in X and DECELERATING — momwire's x1 print is
    # already within ~4.4 Ω of its own n=240 rung, which is why the x1 print
    # is worth banking at all.
    first, second = abs(z2 - z), abs(z4 - z2)
    assert z.imag < z2.imag < z4.imag
    assert second < first, (
        f"the density ladder stopped decelerating: {first:.4f} then {second:.4f}"
    )


@pytest.mark.integration
def test_g703_3b_the_resonant_control_is_banked():
    """The same family at |Z| ≈ 103, which the feed spelling barely moves.

    The control is the point of the pair: it is why 77 corpus captures agreed
    with the engine through a drive question this deck answers at 13.65 Ω.
    """
    z = served(*E2)
    assert abs(z - Z_E2_X1) <= 0.5, (
        f"the resonant control now serves {z:.4f}, "
        f"{abs(z - Z_E2_X1):.4f} Ω from the banked {Z_E2_X1:.4f}"
    )


# --------------------------------------------------------------------------
# G-703-4 — why the corpus never saw it
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_g703_4_the_half_element_mislocation_costs_z_squared():
    """Moving the fed knot half an element is 13.65 Ω here and 0.24 Ω there.

    The measurement momwire#703 was reaching for, kept as the gate that would
    have caught its own prescription.  A feed-region capacitance difference
    enters as ΔZ ≈ −jωC·Z², so the same half-element move costs 57× more on
    the |Z| ≈ 968 deck than on the |Z| ≈ 103 one — which is both why the
    reactive class exposes the question and why no resonant capture in the
    corpus could ever have adjudicated it.
    """
    costs = {}
    for label, (top, bot, n, seg) in (("reactive", E1), ("resonant", E2)):
        element = (top - bot) / n
        at_node = split_fed(top, bot, top - seg * element, seg, n - seg)
        at_centre = split_fed(
            top, bot, top - (seg - 0.5) * element, 2 * seg - 1, 2 * (n - seg) + 1
        )
        # The seam's own answer IS the split-at-node one, to the last digit.
        assert abs(at_node - served(top, bot, n, seg)) < 1e-9
        costs[label] = abs(at_centre - at_node)

    assert costs["reactive"] > 10.0, (
        f"the half-element mis-location now costs only {costs['reactive']:.4f} Ω "
        f"on the reactive deck — it measured 13.65, and a gate that cannot see "
        f"it cannot protect the fed knot's coordinate"
    )
    assert costs["resonant"] < 0.5, (
        f"the resonant control now costs {costs['resonant']:.4f} Ω — it measured "
        f"0.24, and the pair only discriminates while it stays small"
    )
    assert costs["reactive"] / costs["resonant"] > 20.0
