"""Rung-1 physics: a parsed ``nec5`` deck, solved, as a :class:`RunData`.

U2 read the deck, U3 laid out the bytes; this is the middle — the unit that
turns ``(favored tag, node)`` addressing, a ground card and a request into a
momwire solve and reads the answer back into the numbers the printout wants.
It computes physics and formats nothing, exactly as :mod:`._printout`
formats and computes nothing.

Rungs 1 through 4 of the scored ladder (antennaknobs
``docs/status/2026-08-20-eznec-nec5-scored-matrix.md``) are what is served,
and with rung 3 the GROUND is finished: all four cards the dialect writes —
``GN -1`` free space, ``GN 1`` perfect ground, ``GN 0`` / ``GN 2`` Sommerfeld
finite ground and the bare ``GD`` MININEC-type ground — plus ``EX 0``/``EX 4``
sources at a node, ``LD 4`` fixed impedances, node-addressed ``TL`` and ``NT``
networks — separately and, since #504 U3, together in one ``NETWORK DATA``
table — and the ``RP 0`` / ``XQ`` / ``PQ 0`` requests.  Rung 4 is the PHASED
drive: several ``EX 4`` cards at once, which is how this dialect spells an
array, and since momwire#511 that drive may reach the structure THROUGH a
network — a four-square is four set currents, and whether a ``TL`` hangs off
it is the deck's business rather than the drive's.
momwire#516 added the NEAR field, ``NE`` and ``NH``, over the two grounds that
could carry one then — free space and ``GN 1`` — and momwire#545 finishes it:
all four ground cards carry a near field, the two finite ones through the
point evaluator :mod:`momwire._field_point` ("One near field, four grounds and
one point", below).  What is left refuses the OBSERVATION POINT rather than the
ground: the field at a contact-fed wire's base over a finite ground is singular
and no sampling of it converges.  Everything still above the rungs — that one
point, a mixed table whose deck INTERLEAVES the two card kinds, the three
multi-source shapes nothing has printed — refuses BY NAME through
:func:`refusal`, because a seam that answers a question it has no gate for is
worse than one that says so.

momwire#553 lands the BURIED wire, and with it a serve matrix rather than a
rung: a wire strictly below a ``GN 0`` / ``GN 2`` interface is solved through
the per-segment medium and the two buried Sommerfeld families, and **such a
deck's IMPEDANCE, its CURRENTS and its CHARGES are served while every other
output refuses by name** — its near field is momwire#524 phase 3 and its far
field the transmitted far-zone asymptotics, neither of which this arc built.
Three GEOMETRIES around it refuse too, and none of them says "buried wires
are not served" any more: a wire with points on both sides of the interface
(the crossing basis, phase 2); a buried wire over ``GN 1`` or a bare ``GD``,
neither of which has a lower medium to bury it in; and a buried wire on a
deck that also stands a wire END in the plane — the combination momwire#553
measured itself out of, whose sentence quotes both banked phase-0 anchors as
the numbers phase 2 has to meet.

Courtesy stance, the arc's throughout: every NEC-5 fact below was measured
off captured decks and captured printouts under ``tests/fixtures/eznec/``,
and is cited by capture id.  No NEC-5 source, algorithm or internal
structure is described or relied on.

The basis, and why it is the one
--------------------------------
``BSplineSolver`` at the repo's own default configuration — ``degree=2``,
``feed_model="point"``, no extended kernel, no enrichment — which is what
``momwire.deck.build_solver``'s ``"bspline"`` entry constructs and what the
portal solves every NEC-2 deck with.

The family used to be no choice at all, and that claim was wrong.  NEC-5
addresses NODES and ``node_gaps`` (momwire#305, the apex-feed arc) is the only
port momwire has AT a node — but a node where a through-current path runs is
also a knot, and a delta gap on that knot is the same series EMF (below, "Two
spellings of a series EMF").  Only a K >= 3 apex genuinely needs the node
port.  So :func:`build_mesh` takes the solver class and spells the drive for
it (momwire#603 U1), and the ``razor`` family — the NEC-5 formulation twin,
which has no node gaps — reaches this seam.

Three drive spellings, and the address picks between them
---------------------------------------------------------
* a node where two or more wire ends meet — including the two halves of a
  wire this module CUT because a card addressed a node inside it — is a
  junction, and the source is a series EMF named through the favored wire's
  end (0013's ``EX 4,5,-1``, the five-wire apex; 0010's ``EX 4,1,6``, six
  segments into an eleven-segment dipole);
* a wire end standing IN the ground plane is momwire's ground-contact feed
  — a delta gap at arclength 0 on the grounded piece, the idiom
  ``tests/test_contact_nec5_lane.py`` already gates against this engine's
  own base-fed monopole numbers (0019/0043/0044's ``EX 4,1,-1``);
* a FREE wire end is refused.  There is no through-current path at a lone
  conductor end for a series EMF to sit in, momwire says so at the
  constructor, and no captured deck asks for one.

Two spellings of a series EMF, and the BASIS picks between them
---------------------------------------------------------------
The first of those three is one source with two discretizations, and which
one a deck gets is the only thing the solver class decides here.  A basis
carrying ``node_gaps`` gets the node port, and the wire is CUT wherever a
card addressed a node inside it so that port has the two wire ends it needs.
A basis without them gets a ``feeds`` delta gap on the knot itself — the
wire's own interior knot, uncut, or the junction's through-current unknown
where K = 2 ends meet — and no cut anywhere.

Neither is the true one.  They differ by whether the expansion is clamped on
each side of the node or runs continuously through it: 2.03 % on a 5 m dipole
at 30 MHz under degree-2 B-splines, the same class as the basis envelope this
seam already reports against the licensed engine.  What is NOT a matter of
taste is that the cut costs something — a node one element from a wire's end
leaves a one-segment polyline, which carries no tent, which ``RazorSolver``
refuses outright, and which no basis wanted.

Measured over the 62 committed captures: razor-nec5 served 11 under the cut
and serves 47 without it, and on the 42 that carry a printout its input
impedance sits a median 0.00 % and a worst 0.03 % from the licensed engine's
own — which is the twin doing what it is for.  The remaining 15 are four
named things and no drive spelling: 5 whose ``GW`` declares a genuine
one-segment wire, 5 ground contact over a finite ground, 3 that bspline
refuses too, and the 2 five-wire apexes, which keep the node port.

The address picks the spelling one card at a time, and neither #504 U4 nor
momwire#511 changes that: a phased deck is several cards each picking its own,
and 0031's four are four ground-contact feeds because all four verticals stand
in the plane.  What the phased deck DOES change is the drive: with one source
the card is a scale on a unit probe, with four it is a simultaneous boundary
condition on four coupled ports and the volts that deliver them come out of a
4x4 solve (:func:`_multi_drive_state`).  Put a network under it and the 4x4 is
still a 4x4 — what moves is where its entries come from, because what an
``EX 4`` fixes is the SOURCE current and a driven connection point's source
current is not its structure current.  Two cards on ONE port would be a
boundary condition written twice; both ways to write that — the same address
twice, and the two sides of one cut — refuse by name.

Two conventions, and the one matrix between them
------------------------------------------------
A card's ``(favored tag, node)`` names a port in the DECK's convention: the
current it reports is the one flowing along the favored wire in that wire's
own end-1 → end-2 direction, and the voltage goes with it (0017 prints the
two sides of one junction with IDENTICAL currents, which is what fixes this
reading).  momwire's node-gap port reports the current flowing FROM the node
INTO the named wire — the same number up to the ``sigma`` of the wire end
that named it.  So the two conventions differ by one signed matrix ``T``
(:func:`_transform`), and every number in this module lives on the deck's
side of it: the loads, the network cards, the port tables, the power budget.
``T`` is applied twice, at the two places the solver is touched, and nowhere
else.

That matters because a network is the first thing at this seam that can SEE
the sign.  With one source a global flip is gauge — it cancels between the
voltage and the current and the printed impedance never moves — but a
network relates two ports to each other, and a relative sign between them is
a different circuit.  The W7EL triple is the gate: config C's two
admittances sit on OPPOSITE sides of the wire-2/wire-3 junction and combine
in SERIES (195.34 − j57.458), where A's and B's sit on one side and combine
in parallel (114.47 + j21.096).  A seam that loses the relative sign, or
that folds the two addresses into one node, answers A's number for C.

One cut, two addresses
----------------------
Which leads to the other half of the same fact.  ``2,3`` and ``3,-1`` name
the two sides of ONE cut through a two-wire node, so the antenna presents
ONE port to both of them: their currents are equal and their EMFs add.
momwire says as much at its constructor — "one series gap per junction" —
and this module does not ask it for a second one.  It declares the cut once
and gives the far side the SAME solver column with the opposite sign
(:func:`_assign_columns`), which is the rank-1 two-port those two addresses
really are.  At a junction of three or more wires two addresses would be two
genuinely different cuts, no capture writes one, and it refuses by name.

One finite ground, two ways to spell its loss
---------------------------------------------
``GN 0``'s sixth field is a conductivity when it is positive and Im(εc)
itself when it is NEGATIVE — the same convention the dialect's ``GD`` record
already flags, measured here on ``GN 0`` against the linux oracle
(2026-08-20).  ``GN 0,0,0,0,13.,-12.84,…`` and ``GN 0,0,0,0,13.,.005,…``
print the same ``CONDUCTIVITY= 5.000E-03`` cell, the same
``1.30000E+01-1.28400E+01``, and the same 47.789 − j0.78525 Ω: the engine
back-derives the conductivity from the imaginary part and runs the identical
medium.  So this module folds the negative spelling into its equivalent
σ at :func:`_medium` and nothing downstream — not the solve, not the far
field, not the printout — has to know which spelling arrived.

The two constants in that fold are BOTH measured, not assumed.  The printed
``COMPLEX DIELECTRIC CONSTANT`` is ``εr − j·σ·λ·59.96``: at 7 MHz and
σ = 0.005 the captures print ``1.30000E+01-1.28400E+01`` and 0048's 7.02 MHz
twin prints ``-1.28034E+01``, which pins the product σ·λ·59.96 to six digits
(the oracle sweep 1/3/7/14.25/299.8 MHz gives 59.9600 ± 0.0002; NEC-2's own
59.958 would print ``1.79754E+04`` at 1 MHz where the engine prints
``1.79760E+04``).  momwire's own solve folds ε̃ from σ with the SI ε₀ instead,
which differs by 5e-5 of the imaginary part — two orders below the frequency
slack ``SPEED_OF_LIGHT_MHZ_M`` already carries and many below this seam's
basis offset.  The printed cell is the ENGINE's, because the byte-gate
compares it; the solved medium is momwire's, because that is what solved it.

One medium, two things to do with it
------------------------------------
``GN 0`` and the bare ``GD`` carry the SAME two media fields, print the SAME
three cells from them under the SAME banner — and are two different grounds.
``GN 0`` solves in the medium.  ``GD``, EZNEC's "Real / MININEC type", solves
over a PERFECT image and spends the medium entirely on the far field: PEC
currents plus a second-medium pattern under an ordinary ``RP 0``, on
contacting and elevated geometry alike (scored matrix, probe family 1).

The consequence is an identity, and it is the sharpest fact this rung has:
``Z`` under ``GD`` IS ``Z`` under ``GN 1``.  Measured on the engine — the
contact vertical answers 35.571 − j1.4223 both ways (0043/0044 against
0015/0020/0045) and an elevated half-wave 89.933 + j52.053 both ways — and
reproduced here mechanically rather than approximately: :func:`_solver_for`
hands ``GD`` the same single ``ground_z=0.0`` kwarg it hands ``GN 1``, so the
two runs are the same constructor call and every solved number in them is
bit-identical.  ``tests/test_eznec_serve.py`` gates that on both geometry
classes at ``==``, and gates the other side of it too: the same deck under
``GN 0`` answers 47.789 − j0.78525, which is 34 % away in R.  Aliasing ``GD``
onto ``GN 0`` is the trap this rung exists to not fall into, and the identity
is what makes falling in visible.

What the medium then DOES is the pattern, and the two modes part company
there by a level rather than by a shape: 0045 and 0047 are one wire over one
medium at one frequency, and their printed cuts sit 1.28-1.29 dB apart at
every one of the 178 non-null rows — the MININEC mode hotter, because a PEC
structure keeps the current a lossy ground would have loaded, and the pattern
is normalized by an input power that moved with it.  The seam reproduces that
DIFFERENCE to 0.02 dB across the cut, which is the gate an aliased ``GD``
could not pass at all: it would print 0.00.

The far field itself is a Fresnel-weighted image and this module says so in
the portal's own vocabulary (:func:`_far_ground`).  No Sommerfeld integral
answers anything under ``GD`` — the captures show it in the one printed cell
that is not the banner, ``FILL= 0.000 SEC.`` against 0047's ``0.094``, and in
what 0045 does NOT print: the ``Sommerfeld integral tables written in
previous run`` postamble that all four ``GN 0`` captures carry.  The engine
announces ``Will compute Sommerfeld-ground tables`` from a stale cache check
and then computes none, because this mode never wanted any.

One near field, four grounds and one point
------------------------------------------
``NE`` and ``NH`` are served over all four ground cards.  The matrix was
measured (momwire#516's capture family 0107-0115, manifest
``near_field_family``; momwire#545's routing) rather than assumed, and every
served cell carries its own envelope, worst PRINTED cell of the whole table
against the capture's, magnitude relative to that cell and phase in degrees:

  ground     card    capture   worst |cell|   worst phase   composition
  ``GN -1``  ``NE``  0115         1.9482 %      0.30 deg    direct
  ``GN 1``   ``NE``  0109         5.4488 %      1.50 deg    direct + image
  ``GD``     ``NE``  0108         5.4526 %      1.23 deg    direct + C2·img + rem
  ``GN 0``   ``NE``  0110         5.3076 %      1.26 deg    direct + C2·img + rem
  ``GN 0``   ``NH``  0111         6.6992 %      0.33 deg    the same, curled
  ``GN 0``   ``NE``  0113         1.8123 %      0.30 deg    direct + C2·img + rem

Four spellings of one line.  ``direct`` is ``_element_fields`` — the portal's
mixed-potential readout, one owner per readout; ``img`` is the geometric
mirror of it; ``rem`` is :mod:`momwire._field_point`, which is the matrix
fill's own azimuth combination of the four Sommerfeld interpolation surfaces
kept as a VECTOR so that an observer can be a point in space rather than a
wire element.  ``C2 = (ε̃−1)/(ε̃+1)``, and the whole thing is NEC's own
decomposition of the half-space Green's function (theory manual eqs 136-147)
read out at a point.  All six numbers sit in the family the impedance
envelopes do, which is what says the near field is the same solve read out at
a different place rather than a second one.

The load-bearing finding is which MEDIUM each ground reads out in, and it is
the same one momwire#516 measured — now the ROUTE rather than the evidence for
a refusal.  The engine's near field over a bare ``GD`` is its ``GN 0`` near
field: 0108 and 0110 are one grid over one medium with only the ground
mnemonic changed and their tables agree to 4.4 % worst-cell, INCLUDING the
``EY`` column that a vertical monopole's symmetry forbids — 1.7359E-02 against
1.7342E-02, the same Sommerfeld interpolation dust in both, in a column a PEC
image makes exactly zero.  So the MININEC-type ground reflects off its medium
in the far field and SOLVES its near field in it, and the far-field rung's
``GD``/``GN 1`` identity says nothing about this table.  :func:`_near_medium`
is that sentence in code: ``GN 0``'s ε̃ comes off its own Sommerfeld solve and
``GD``'s is folded from the deck's medium, because its solver never saw one.
The seam reproduces the captures' own cross-agreement with its own numbers —
1.54 % of table scale between the served 0108 and the served 0110, well inside
the 4.4 % the two engine tables sit apart at.

What is NOT served is one CELL, and it is a point rather than a ground.  0022,
0107 and 0112 all ask for the field at exactly (0, 0, 0) — the base of a
contact-fed monopole standing on a finite ground — and the composition there
is SINGULAR: the image cancels the base node's continuity charge only by
``1 − C2 = 2/(1 + ε̃)``, so a finite residual charge sits exactly at the
observer.  The subdiv ladder there does not converge, it grows: 36.6 / 141 /
298 / 588 / 978 / 1230 V/m at subdiv 1 / 4 / 8 / 16 / 32 / 64, while the engine
prints 8.2521E+02 ∠162.15° for 0112 and 6.6673E+02 for 0107 — numbers that sit
BETWEEN our subdiv-16 and subdiv-32 rungs, which is the sharpest way to say
neither party has converged.  Printing a rung would publish an artefact of
``_NEAR_FIELD_SUBDIV``; printing the engine's would tune a sampling constant
against an oracle.  So :func:`_near_field_refusal` names the POINT — the card,
the coordinate, the wire whose base it is — and says the near field is served
at every point off the contact.  The ladders are gated in
``tests/test_field_point.py`` as gates that FAIL IF FIXED: a ladder that
converged would mean the composition changed and this sentence needs
re-measuring.

Over a PERFECT ground the same point is ordinary, which is why the refusal is
finite-ground-only: ``C2 = 1`` there, the image cancels the contact charge
EXACTLY, and there is no residual to diverge on.

The nec2 half of this tree still refuses ``NE``/``NH`` over its own finite
grounds by name (``momwire.portal._portal._near_field_lines``, momwire#388) —
same evaluator, different seam, and a follow-up rather than a disagreement.

The budget's own arithmetic
---------------------------
``INPUT POWER`` is the SUM of the ``ANTENNA INPUT PARAMETERS`` rows, one row
per ``EX`` card, each row's own ``½·Re(V·I*)``.  With one source that is a
sentence about nothing; with four it is the rung's sharpest arithmetic, and
0031 is why.  Its tag-1 vertical prints ``-1.7790E+00`` OHMS and
``-1.7790E+00`` WATTS — a negative driving-point resistance, because at that
phasing the element takes more power out of the array through the mutual
coupling than its own generator puts in — and the printed ``INPUT POWER =
1.2831E+02`` is the other three rows with that one SUBTRACTED.  Nothing in
this module clamps, flags or apologises for a negative row: it rides through
the sum, through ``EFFICIENCY``, and through the pattern's normalization
untouched, because that is what the capture shows the engine doing.

``RADIATED POWER`` is not ``INPUT − losses`` here.  Measured on all six
network captures, NEC-5 prints ``INPUT + Σ(network connection point powers)
− WIRE LOSS`` and ``NETWORK LOSS = −Σ``: on 0012 that is 114.47 − 36.714 =
77.756 against a printed 7.7758E+01, and on 0027 — whose source stands ON a
connection point, so the source's own watts are counted a second time
through it — it is 23.422 + 23.422 = 46.844 against a printed 4.6844E+01,
with ``EFFICIENCY = 200.00 PERCENT`` and no ``NETWORK LOSS`` line at all.
The line prints when that number is POSITIVE, which is 4 of 6 captures.  The
double count is the engine's; reproducing it is this seam's job, and the two
200 % budgets are the evidence it is real rather than a slip.  The pattern
is normalized by INPUT power and is untouched by any of it (0012's peak row
gives 4π|rE|²/2η at 0.50 dB = 114.36 W, which is the input power and not the
radiated one).

What this module does NOT reproduce
-----------------------------------
The solved numbers.  momwire's B-spline formulation and NEC-5's are two
different discretizations of the same physics and they sit Ω apart at these
mesh densities — measured on the rung-1 captures and pinned, capture by
capture, in ``tests/test_eznec_serve.py``.  The gate on this unit is
therefore the printout's STRUCTURE (every heading, count, echo, row count
and null-row convention byte-identical) plus an envelope on the physics, in
the shape ``docs/design/solver-architecture.md`` calls the golden-lane rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..bspline import BSplineSolver
from ..deck._cards import tokenize
from .. import _field_point, _ground_refl, _ground_spec
from ..deck._nec5 import (
    Nec5Deck,
    Nec5FarFieldRequest,
    Nec5FreeSpace,
    Nec5Ground,
    Nec5MininecGround,
    Nec5NearFieldRequest,
    Nec5Network,
    Nec5Node,
    Nec5PerfectGround,
    Nec5SommerfeldGround,
    Nec5Source,
    Nec5TransmissionLine,
    Nec5Wire,
)

# The network solve is momwire's and it has ONE owner too.  What six real
# fields on a `TL` or an `NT` MEAN — the crossed-line polarity flag hiding in
# the sign of Z0, the end shunts, the forced Y21 = Y12 — is card semantics,
# written down once in `deck._networks` for the nec2 dialect and read from
# there here (design doc `networks-move-into-the-engine.md`: *parse
# per-reader, semantics once*).  This module resolves the ADDRESSING, which is
# the dialect's own, and hands the semantics a record with its fields already
# in NEC's order.
from ..deck._networks import card_branches, port_name
from ..deck.model import NetworkCard
from ..networks import Driven, Network, NetworkReducer, PortOnWire

# The far-field readout is momwire's, and it has ONE owner: the portal's
# NEC-2 front end already turns element currents into E(THETA)/E(PHI), the
# gain columns and the polarisation ellipse, against a nec2c oracle and
# hundreds of fixtures.  A second copy here would be a second thing to keep
# right (the ranked extraction backlog, momwire#429, is where that copy would
# be paid off); importing is the honest dependency until it is.
from ..portal._portal import (
    ETA0,
    Ground,
    _element_fields,
    _far_moments,
    _FIELD_FLOOR2,
    _gain_db,
    _image_moments,
    _polarisation,
)
from ._printout import (
    ENVIRONMENT_FINITE_GROUND,
    ENVIRONMENT_FREE_SPACE,
    ENVIRONMENT_PERFECT_GROUND,
    ChargeRow,
    GroundMedium,
    LineRow,
    LoadRow,
    NearFieldBlock,
    NearFieldRow,
    NetworkRow,
    PatternBlock,
    PatternRow,
    PortRow,
    PowerBudget,
    RunData,
    WireCurrentRow,
)

__all__ = [
    "BASIS",
    "EPSC_CONDUCTIVITY_FACTOR",
    "SPEED_OF_LIGHT_MHZ_M",
    "refusal",
    "serve",
]


# NEC's own metre-megahertz product, MEASURED off the printed WAVELENGTH
# cell rather than assumed: 0019 prints ``WAVELENGTH= 4.2829E+01 METERS`` at
# 7 MHz (299.8/7 = 42.8286; the SI c gives 42.8275 and would print
# 4.2827E+01), and 0010's normalized segment length 4.54534E-02 is
# (0.5/11)/(299.8/299.7925) to every printed digit.  Used for the printed
# wavelength, for the wavelength-normalized geometry columns of the current
# and charge tables, and for the solve itself — 2.5e-5 of relative frequency,
# far below this seam's basis offset, and one constant is easier to defend
# than two.
SPEED_OF_LIGHT_MHZ_M = 299.8

# The engine's own conductivity-to-permittivity constant, MEASURED off the
# printed `COMPLEX DIELECTRIC CONSTANT` cell (module docstring, "One finite
# ground"): eps_c = eps_r - j*sigma*wavelength*59.96.  Written in the same
# wavelength-and-a-constant shape NEC-2 writes it in rather than as
# sigma/(omega*eps0), because that is the shape the measurement recovers —
# the product is constant to six digits across a 1 MHz to 299.8 MHz sweep,
# with the wavelength taken at SPEED_OF_LIGHT_MHZ_M.
EPSC_CONDUCTIVITY_FACTOR = 59.96

# The basis, spelled as `momwire.deck.build_solver` spells it.  Recorded as a
# name so a reader of a served printout can ask what solved it.
BASIS = "bspline"
_DEGREE = 2

# Node fusion grid, metres — `momwire.deck._polylines._NODE_EPS`, the same
# tolerance the nec2 front end decides "these two wires touch" with.  By the
# time a deck exists its coincident ends are already exactly equal (EZNEC
# writes the same decimal string twice); the grid absorbs the last few ulps.
_NODE_EPS = 1e-6


# --------------------------------------------------------------------------
# refusals — everything above rung 1, by name
# --------------------------------------------------------------------------
#
# Each message names its CARD and says what it would take to serve it, because
# the only channel this engine has is the printout and the only reader is a
# person looking at EZNEC's viewer.  U1's catch-all stub stays behind them all
# for the genuinely unforeseen.

_REFUSE_MIXED_ORDER = (
    "this deck writes an NT card before a TL card; a mixed NETWORK DATA table "
    "is served in the order every captured mixed deck writes it - every TL "
    "card, then every NT card, one sub-table each - and no captured printout "
    "says which sub-table an interleaved deck heads with, or which order its "
    "STRUCTURE EXCITATION DATA connection points print in"
)
_REFUSE_TWO_CUTS = (
    "{first} and {second} address a node where {count} wires meet, which is "
    "two different cuts through one junction; this seam serves the two sides "
    "of a SINGLE cut (a node where exactly two wires meet) and no captured "
    "deck asks for more"
)
_REFUSE_ZERO_LENGTH_LINE = (
    "a TL card between {first} and {second} writes no length and the two "
    "nodes are the same point, so there is no distance for NEC's "
    "zero-length rule to resolve it to"
)
_REFUSE_MIXED_DRIVE_KINDS = (
    "this deck carries {count} EX cards writing BOTH kinds - {voltages} EX 0 "
    "(a set voltage) and {currents} EX 4 (a set current); a multi-source drive "
    "is served only where every card sets a CURRENT, which is what both "
    "captured multi-EX decks write, and a mixed drive has no printed row "
    "anywhere to be gated against"
)
_REFUSE_MULTI_EX_VOLTAGE = (
    "this deck carries {count} EX 0 cards; a multi-VOLTAGE drive is not served "
    "at this seam - none of the 49 captured decks writes one, so nothing says "
    "what the engine prints for it"
)
_REFUSE_DUPLICATE_EX = (
    "two EX cards address {at}; one node is one port, so the second card is a "
    "second generator in series across the same gap - no captured deck writes "
    "that, and this seam will not decide how two set currents share one path"
)
_REFUSE_TWO_DRIVES_ONE_PORT = (
    "{first} and {second} are the two sides of ONE cut and an EX card drives "
    "each; the antenna presents a single port to both, so the two set currents "
    "are one boundary condition written twice - no captured deck writes that, "
    "and this seam will not decide which of them wins"
)
# The near-field refusal names the POINT.  Every ground carries a near field
# since momwire#545; what does not is one CELL of one grid — an observation
# point sitting exactly on a wire's ground contact over a finite ground, where
# the composed field is singular and its own subdiv ladder diverges (module
# docstring, "One near field, four grounds and one point").  The ladder's two
# ENDS are quoted so the operator can see the divergence rather than take it on
# trust; the whole ladder is measured on 0112 and gated, rung by rung, in
# `tests/test_field_point.py`.
_REFUSE_NEAR_FIELD_CONTACT = (
    "{card} asks for the field at ({x:g}, {y:g}, {z:g}) metres, which is where "
    "wire {tag} stands on this deck's finite ground - and the field AT a "
    "contact-fed wire's base over a finite ground is singular. The image "
    "cancels the base node's charge only by 2/(1+epsilon), so a finite "
    "residual charge sits exactly at the observation point and the composed "
    "field there grows without bound as the solved current is resampled - 36.6 "
    "V/m at one sample per element, 1230 V/m at sixty-four. The engine's own "
    "printed cell is its discretization's regularization of that same "
    "singularity and sits INSIDE that ladder rather than at the end of it, so "
    "this seam will not print a number that is an artifact of a sampling "
    "constant. The same deck's FAR field, its currents, its charges and its "
    "impedance are all served, and the near "
    "field is served at every point off the contact - move the {card} card's "
    "observation point off wire {tag}'s base"
)
# The buried rung LANDED with momwire#553: a wire strictly below a ``GN 0`` /
# ``GN 2`` interface is served, through the per-segment medium and the two
# buried Sommerfeld families.  What is left of the old sentence splits three
# ways, and each of the three names a DIFFERENT missing thing rather than
# repeating "buried wires are not served":
#
#   * a wire that CROSSES the interface (or stands an end in it from below) —
#     the crossing basis, momwire#524 phase 2, with its banked anchor;
#   * a buried wire over ``GN 1`` or a bare ``GD`` — no lower medium exists
#     under either card, so there is nothing to bury the wire in;
#   * the OUTPUTS a buried deck cannot answer — its near field (phase 3) and
#     its far field (the transmitted far-zone asymptotics).
#
# Impedance, currents and charges serve.  That sentence is the serve matrix
# and it is repeated in the module docstring.
_REFUSE_BURIED_CROSSING = (
    "wire {tag} crosses the ground interface (z runs {zmin:g} to {zmax:g} m "
    "across z = 0) - a wire wholly below the interface is served over this "
    "deck's ground card, and a wire wholly at or above it is served, but a "
    "wire with points on BOTH sides is neither: current crossing the "
    "interface is a boundary condition where the two media meet, not a "
    "junction, and it needs the crossing basis momwire#524 phase 2 owns. "
    "The gate that basis has to meet is already banked from this engine's "
    "own printout - a 2 m buried vertical joined at z = 0 to a 10 m monopole "
    "over eps_r 13 / sigma 0.005 S/m soil at 7 MHz prints 74.761 - 57.730j "
    "ohm. Until then: leave the buried part DETACHED from the part above the "
    "plane (a buried radial screen under a base-fed vertical is served that "
    "way), or raise the whole wire clear of z = 0"
)
_REFUSE_BURIED_NO_MEDIUM = (
    "wire {tag} runs below the ground plane (min z = {zmin:g} m) under a "
    "{card} card, which has no lower medium to put it in: {why}. A buried "
    "wire is served under GN 0 / GN 2, where the half-space below the "
    "interface is a real medium with a wavenumber of its own - write the "
    "finite-ground card, raise the wire to the surface, or model in free "
    "space (GN -1)"
)
_WHY_NO_MEDIUM = {
    "GN 1": (
        "the field inside a perfect conductor is identically zero, so a wire "
        "there is not buried, it is shorted out"
    ),
    "GD": (
        "the MININEC-type ground solves its currents over a PERFECT image and "
        "spends its medium entirely on the far field, so below the plane "
        "there is a perfect conductor here too"
    ),
}
_REFUSE_BURIED_WITH_CONTACT = (
    "wire {cw} stands an END in the ground plane and wire {bw} is buried "
    "below it, and that COMBINATION is not served though each half is. "
    "momwire continues a contact wire's current into the ground as a scaled "
    "IMAGE, which works only because nothing is ever inside the ground to "
    "look at it - and a buried wire is exactly that. What a buried wire "
    "should see is the contact current spreading in the lower medium, which "
    "is the crossing physics momwire#524 phase 2 owns. The error is not "
    "small and it is measured: at eps_tilde = 1, where the buried fill must "
    "reproduce the free-space fill exactly, this deck class disagrees by 2.5 "
    "RELATIVE on the contact basis and by 1e-8 on every other one, while the "
    "same deck with the monopole lifted 1 m clear of the plane agrees to "
    "1.0e-5 throughout. This engine's own printouts are banked as the gates "
    "phase 2 has to meet - 92.130 - 70.141j ohm for a 10 m monopole over one "
    "detached 5 m radial 15 cm down, 89.985 - 71.401j ohm for the four-"
    "radial fan, both over eps_r 13 / sigma 0.005 S/m soil at 7 MHz. Until "
    "then: raise the above-ground wire clear of z = 0 (an elevated feed over "
    "a buried counterpoise is served), or model the buried structure alone"
)
_REFUSE_BURIED_NEAR_FIELD = (
    "{card} asks for a field on a deck with a wire below the ground plane, "
    "and a buried deck's near field is not served. The point evaluator "
    "composes a direct term, an image term and a Sommerfeld remainder, and "
    "which three of those a point gets depends on which side of the "
    "interface it and the source are on - reconciling the two readouts (a "
    "REMAINDER above, a WHOLE transmitted field across) is momwire#524 phase "
    "3, and the three crossing regimes refuse by name until it lands. This "
    "deck's IMPEDANCE, its CURRENTS and its CHARGES are all served: drop the "
    "{card} card, or lift the wire above z = 0"
)
_REFUSE_BURIED_FAR_FIELD = (
    "RP asks for the far field of a deck with a wire below the ground plane, "
    "and a buried deck's radiation pattern is not served. The pattern of a "
    "buried source is the transmitted field's FAR-ZONE asymptotics - a "
    "saddle-point evaluation of the same integral, with its own lateral-wave "
    "and critical-angle structure - and momwire#553 built the transmitted "
    "family over a NEAR-zone tabulation only (2 free-space wavelengths of "
    "range), so there is nothing here to take a limit of. This deck's "
    "IMPEDANCE, its CURRENTS and its CHARGES are all served: drop the RP "
    "card, or lift the wire above z = 0"
)
_REFUSE_IN_PLANE_WIRE = (
    "wire {tag} lies in the ground plane (both ends at z = 0) - a horizontal "
    "wire IN a conducting interface is degenerate - raise it above the "
    "plane, or stand a wire END on the plane instead (ground contact is "
    "served)"
)
_REFUSE_NO_EX = "this deck carries no EX card - nothing drives the structure"
_REFUSE_NO_FR = "this deck carries no FR card - there is no frequency to solve at"
_REFUSE_NO_REQUEST = (
    "this deck asks for nothing: it carries neither an RP nor an XQ card, so "
    "there is no run to report"
)
_REFUSE_RP_RANGE = (
    "RP 0 with a nonzero range field is not served at this seam: every "
    "captured RP writes 0 there, and the range form's printout header has "
    "never been observed"
)


def refusal(deck: Nec5Deck) -> str | None:
    """The reason this deck is out of scope, or ``None`` when it is not.

    Order is deliberate — grounds first, then the cards, then the drive, then
    the request — so a deck that is out of scope in several ways names the
    reason a reader would fix first.  It is the ladder's own shape, and every
    rung that lands moves decks DOWN it rather than changing it: 0022 named its
    ``GN 0``, then its ``NE`` once the ground rung landed, then the PAIR once
    momwire#516 measured which grounds could carry a near field, and since
    momwire#545 names neither — the card is served, the ground under it is
    served, and what is left is the single OBSERVATION POINT its ``NE`` asks
    about, which is the base of its own monopole; the three mixed-card decks
    (0000, 0023, 0025) named their ground, then their TL-and-NT table, and now
    name nothing at all; and 0031 named its ground, then its four ``EX`` cards,
    and now names nothing either.

    That last move is the shape of a refusal shrinking rather than lifting.
    momwire#545's evaluator gives every one of the four ground cards a near
    field, so the near-field rung's sentence stopped being about a CAPABILITY
    and became about a CELL: 0110 and 0022 are the same deck and the same
    ground, and only the grid separates them (0110's five-by-five grid stands
    off the wire and is served; 0022's single point is the contact node and is
    not).  A refusal that names a point rather than a ground is one no rung can
    land on — nothing converges there, on either side of the seam.

    There is no ground refusal left to put first, no mixed-table refusal and,
    since #504 U4, no plain multi-source one — and since momwire#511 no
    multi-source-through-a-network one either.  All four ground cards this
    dialect writes are served, a table carrying both card kinds is served, and a
    deck driven by several ``EX 4`` cards at once is served as the constrained
    drive it is, whether or not the drive reaches the structure through a
    ``TL``/``NT``.  What is left on those rungs is one ORDER and four shapes
    nothing has ever printed.  The order: every captured deck writes all of its
    ``TL`` cards before any of its ``NT`` cards, both readings of the sub-table
    rule agree while that holds, and a deck that interleaves them separates the
    two readings with nothing to pick between them.  The shapes are
    :func:`_drive_refusal`'s.  So "grounds first" survives as the rule a fifth
    ground card would land under rather than as a description of a line that is
    still there.

    momwire#553's rung is the same shape again, and it is recorded here the
    way #545's was.  The BURIED wire moves down the ladder rather than off
    it: before it, ``GW 2,10,0.,0.,-0.15,…`` named its own geometry and
    stopped; after it the same card is served, and what a buried deck can
    still name is either a GEOMETRY around it (a wire crossing the interface,
    a buried wire over a card with no lower medium, a buried wire sharing a
    deck with a ground contact) or an OUTPUT it cannot answer (its near
    field, its far field).  That last pair is the first time this seam's
    refusal grammar has had to say "this deck is served, but not for THAT
    number", which is why the serve matrix — impedance, currents, charges —
    is written out in the module docstring rather than left implicit in the
    order of the branches below.  The geometry rung goes on being checked
    before the request rung, so a deck that is out of scope both ways still
    names the geometry a reader would fix first.
    """
    if deck.transmission_lines and deck.networks and _interleaves_networks(deck):
        return _REFUSE_MIXED_ORDER
    if not deck.sources:
        return _REFUSE_NO_EX
    drive = _drive_refusal(deck)
    if drive is not None:
        return drive
    if deck.frequency_mhz is None or deck.frequency_mhz <= 0.0:
        return _REFUSE_NO_FR
    geometry = _geometry_refusal(deck)
    if geometry is not None:
        return geometry
    buried = _has_buried_wire(deck)
    for request in deck.requests:
        if isinstance(request, Nec5NearFieldRequest):
            if buried:
                # THE SERVE MATRIX, in one branch: a buried deck's impedance,
                # currents and charges serve and every other output refuses
                # BY NAME. Near fields are momwire#524 phase 3, far fields
                # the transmitted far-zone follow-up.
                return _REFUSE_BURIED_NEAR_FIELD.format(card=_near_field_card(request))
            near = _near_field_refusal(deck, request)
            if near is not None:
                return near
        if isinstance(request, Nec5FarFieldRequest):
            if buried:
                return _REFUSE_BURIED_FAR_FIELD
            if request.range_m != 0.0:
                return _REFUSE_RP_RANGE
    if not deck.requests:
        return _REFUSE_NO_REQUEST
    return None


def _has_buried_wire(deck: Nec5Deck) -> bool:
    """Whether any wire lies STRICTLY below the plane, on the solver's own
    per-wire tolerance. Read by the two output refusals, which are about the
    deck rather than about a card."""
    if deck.ground is None or isinstance(deck.ground, Nec5FreeSpace):
        return False
    for wire in deck.wires:
        pl = np.array([wire.end1, wire.end2], dtype=float)
        tol = _ground_spec.ground_touch_tol(pl)
        if float(pl[:, 2].max()) < -tol:
            return True
    return False


def _geometry_refusal(deck: Nec5Deck) -> str | None:
    """``None`` when every wire can stand over this deck's ground, the
    sentence when one cannot.

    Since momwire#553 a wire STRICTLY below a ``GN 0`` / ``GN 2`` interface
    is served — the solver labels it with the lower medium and fills its
    pairs through the two buried Sommerfeld families — so what this function
    refuses is no longer "below the plane". It is three narrower things, and
    each names what is actually missing rather than restating the geometry:

    * a wire with points on BOTH sides of the interface, which needs the
      crossing basis momwire#524 phase 2 owns (a buried wire whose end stands
      IN the plane is this case, not the buried one: the tolerance decides,
      and it is the solver's own);
    * a buried wire under ``GN 1`` or a bare ``GD``, neither of which has a
      lower medium at all — momwire solves both over a PERFECT image, so
      below the plane is a perfect conductor and there is nothing to bury a
      wire in;
    * an in-plane wire, degenerate over any conducting ground, unchanged.

    Free space is exempt on all three: z < 0 is legal geometry with no
    interface under it. The tolerance is the solver's own
    (`_ground_spec.ground_touch_tol`, 1e-6 of each wire's length) so the seam
    refuses exactly what the solver would refuse — never more, never less,
    and in particular the seam and the solver agree wire for wire about which
    side of the line "an end in the plane" falls on.
    """
    if deck.ground is None or isinstance(deck.ground, Nec5FreeSpace):
        return None
    sommerfeld = isinstance(deck.ground, Nec5SommerfeldGround)
    card = "GD" if isinstance(deck.ground, Nec5MininecGround) else "GN 1"
    for wire in deck.wires:
        pl = np.array([wire.end1, wire.end2], dtype=float)
        tol = _ground_spec.ground_touch_tol(pl)
        zmin = float(pl[:, 2].min())
        zmax = float(pl[:, 2].max())
        if zmin < -tol:
            if zmax >= -tol:
                return _REFUSE_BURIED_CROSSING.format(
                    tag=wire.tag, zmin=zmin, zmax=zmax
                )
            if not sommerfeld:
                return _REFUSE_BURIED_NO_MEDIUM.format(
                    tag=wire.tag, zmin=zmin, card=card, why=_WHY_NO_MEDIUM[card]
                )
            continue
        if abs(pl[0, 2]) <= tol and abs(pl[1, 2]) <= tol:
            return _REFUSE_IN_PLANE_WIRE.format(tag=wire.tag)
    buried = [
        w.tag
        for w in deck.wires
        if float(np.array([w.end1, w.end2], dtype=float)[:, 2].max())
        < -_ground_spec.ground_touch_tol(np.array([w.end1, w.end2], dtype=float))
    ]
    if buried:
        contacts = [
            w.tag
            for w in deck.wires
            if _ground_spec.contact_ends([np.array([w.end1, w.end2], dtype=float)], 0.0)
        ]
        if contacts:
            return _REFUSE_BURIED_WITH_CONTACT.format(cw=contacts[0], bw=buried[0])
    return None


def _near_field_card(request: Nec5NearFieldRequest) -> str:
    """``NE``/``NH`` spelled the way a refusal names it."""
    return (
        "NH (near magnetic field)" if request.magnetic else "NE (near electric field)"
    )


def _grid_points(request: Nec5NearFieldRequest) -> np.ndarray:
    """An ``NE``/``NH`` card's observation points, in the printout's own order.

    ``NE 0,NX,NY,NZ,X1,Y1,Z1,DX,DY,DZ`` walks X fastest, then Y, then Z.  No
    capture separates the Y nesting from the Z one — every captured grid is
    ``NY = 1`` — so it was measured on the linux oracle (2026-08-21) at
    ``NX = 2, NY = 3, NZ = 2``, and it is the same nesting the nec2 portal
    measured off nec2c.  The points are in METRES and unnormalized.

    One walk, two consumers: :func:`_near_field` prints these rows and
    :func:`_near_field_refusal` asks whether any of them lands on a ground
    contact.  A refusal that walked its own grid could refuse a point the
    table never prints, or miss one it does.
    """
    n_x, n_y, n_z = request.counts
    start = np.asarray(request.origin, dtype=float)
    step = np.asarray(request.step, dtype=float)
    return np.array(
        [
            start + np.array([ix, iy, iz]) * step
            for iz in range(n_z)
            for iy in range(n_y)
            for ix in range(n_x)
        ]
    )


def _contact_ends(deck: Nec5Deck) -> list[tuple[int, np.ndarray, float]]:
    """``(tag, point, tolerance)`` for every wire END standing in the plane.

    The tolerance is the SOLVER's own, per wire — `_ground_spec.
    ground_touch_tol`, 1e-6 of the wire's length, the same spelling
    :func:`_geometry_refusal` uses — so "this end stands on the ground" means
    here exactly what it means to the basis that will grow an image off it.
    """
    ends: list[tuple[int, np.ndarray, float]] = []
    for wire in deck.wires:
        pl = np.array([wire.end1, wire.end2], dtype=float)
        tol = _ground_spec.ground_touch_tol(pl)
        for point in pl:
            if abs(float(point[2])) <= tol:
                ends.append((wire.tag, point, tol))
    return ends


def _near_field_refusal(deck: Nec5Deck, request: Nec5NearFieldRequest) -> str | None:
    """``None`` when every requested point can be answered, the sentence when
    one of them cannot.

    All four ground cards carry a near field since momwire#545, so what is
    refused here is not a ground but a CELL: an observation point sitting on
    the ground CONTACT of a wire, over a finite ground.  The composition there
    is singular — the image cancels the base node's continuity charge only by
    ``1 − C₂ = 2/(1 + ε̃)``, so a finite residual charge sits exactly at the
    observer — and its subdiv ladder diverges rather than converging (36.6 →
    1230 V/m over subdiv 1 → 64, gated in ``tests/test_field_point.py``).  The
    engine's own printed cell sits BETWEEN two of those rungs, which is what
    says it is its discretization's regularization of the same singularity and
    not a converged answer either.  Printing a rung would publish an artefact
    of :data:`_NEAR_FIELD_SUBDIV`; printing the engine's would tune a sampling
    constant against an oracle.  Both are forbidden, so the point refuses.

    Exactly that and no more.  A point ON a wire away from the ground is not
    refused (no capture asks, and the mixed-potential readout's own
    regularization owns that cell); an elevated grid over a finite ground is
    not refused (0113 is served); and NOTHING over ``GN 1`` or free space is
    refused, because over a PERFECT ground the image cancels the contact
    charge EXACTLY — ``C₂ = 1``, ``1 − C₂ = 0`` — and the singularity this
    sentence is about is not there to find.  The corpus decides the scope of
    anything wider.
    """
    if not isinstance(deck.ground, (Nec5MininecGround, Nec5SommerfeldGround)):
        return None
    contacts = _contact_ends(deck)
    if not contacts:
        return None
    for point in _grid_points(request):
        for tag, contact, tol in contacts:
            if float(np.linalg.norm(point - contact)) <= tol:
                return _REFUSE_NEAR_FIELD_CONTACT.format(
                    card=_near_field_card(request),
                    x=point[0],
                    y=point[1],
                    z=point[2],
                    tag=tag,
                )
    return None


def _drive_refusal(deck: Nec5Deck) -> str | None:
    """The multi-``EX`` shapes this seam still has no capture for.

    One ``EX`` is always in scope and always has been.  Several are in scope
    since #504 U4 in one shape and since momwire#511 in two: every card an
    ``EX 4``, one card per port, with or without a ``TL``/``NT`` network on the
    deck.  U4's shape is the network-free one the corpus prints twice (0031's
    four phased verticals over a bare ``GD``, 0032's two over a perfect
    ground); #511's is the same drive reaching the structure THROUGH a network,
    which the corpus now prints four more times (0116/0117's four-square with a
    ``TL``, 0120/0121's cardioid over an ``NT`` and two ``TL``s).  The three
    refusals below are three ways a deck can leave BOTH, and each is worth
    naming separately rather than folding into one "unsupported drive" line,
    because each would be fixed by a different capture.

    A mixed drive would need the engine's rule for a voltage source and a
    current source in one matrix; a multi-``EX 0`` would need its rule for
    several set voltages; and two cards at one address would need it to say
    which of two set currents wins.  All three are 0 of 53.

    The one that LEFT is the phased drive through a network, and it left the
    way a refusal should: not because the answer became obvious but because
    four printouts arrived to check it against.  Its sentence used to say that
    "a constrained drive composed with the network reducer is unobserved", and
    that was the exact truth until 2026-08-20.

    A FOURTH way out lands two cards on one port through two different
    addresses, and only the mesh can see it — :func:`_check_one_port_per_drive`
    catches that one, on the far side of :func:`build_mesh`.
    """
    kinds = [source.kind for source in deck.sources]
    if len(kinds) == 1:
        return None
    voltages = kinds.count(0)
    if voltages and voltages != len(kinds):
        return _REFUSE_MIXED_DRIVE_KINDS.format(
            count=len(kinds), voltages=voltages, currents=len(kinds) - voltages
        )
    if voltages:
        return _REFUSE_MULTI_EX_VOLTAGE.format(count=voltages)
    seen: set[Nec5Node] = set()
    for source in deck.sources:
        if source.at in seen:
            return _REFUSE_DUPLICATE_EX.format(
                at=f"{source.at.tag},{source.at.written}"
            )
        seen.add(source.at)
    return None


def _interleaves_networks(deck: Nec5Deck) -> bool:
    """Does this deck write an ``NT`` card BEFORE a ``TL`` card?

    Read off the deck's own text rather than off the model, and it has to be:
    :class:`~momwire.deck._nec5.Nec5Deck` keeps its two card kinds in two
    tuples, which is the right shape for everything else this seam does with
    them and the one shape that cannot answer this question.  The card images
    are already the printout's source of truth for the echo, so reading them
    again here is the same reading applied to the same file.
    """
    kinds = [
        card.mnemonic
        for card in tokenize(deck.source_text)
        if card.mnemonic in ("TL", "NT")
    ]
    if "TL" not in kinds or "NT" not in kinds:
        return False
    return kinds.index("NT") < len(kinds) - 1 - kinds[::-1].index("TL")


class ServeRefusal(Exception):
    """A rung-1 deck this engine still cannot stand behind, named.

    Raised from inside the translation for the refusals :func:`refusal`
    cannot see from the cards alone — an address that lands on a free wire
    end, a source on a grounded junction — so the shell reports them through
    the same ``NEC ERROR`` line as the rest.
    """


# --------------------------------------------------------------------------
# the structure, in the deck's own node vocabulary
# --------------------------------------------------------------------------


def node_points(wire: Nec5Wire) -> list[tuple[float, float, float]]:
    """A ``GW``'s ``segment_count + 1`` node points, end 1 first.

    The two authored endpoints are handed back UNTOUCHED rather than
    recomputed at ``t = 0`` and ``t = 1``: ``a + (b - a) * 1.0`` is not
    always bitwise ``b``, and two wires that share an endpoint have to fuse
    into one node without leaning on the tolerance to save them (the same
    line ``momwire.deck._polylines`` draws).
    """
    n = wire.segment_count
    a = np.asarray(wire.end1, dtype=float)
    b = np.asarray(wire.end2, dtype=float)
    points = [tuple(a + (b - a) * (k / n)) for k in range(n + 1)]
    points[0] = wire.end1
    points[-1] = wire.end2
    return points


def _node_key(point) -> tuple[int, int, int]:
    return tuple(round(c / _NODE_EPS) for c in point)  # type: ignore[return-value]


@dataclass(frozen=True)
class Structure:
    """A deck's geometry counted the way the printout counts it.

    Every field is derived from the CARDS — nothing here has met a solver —
    because the four numbers the STRUCTURE SPECIFICATION block prints are
    properties of the deck, not of whatever basis answers it.
    """

    wires: tuple[Nec5Wire, ...]
    points: tuple[tuple[tuple[float, float, float], ...], ...]
    # fused node key -> how many ELEMENT ends meet there, image included
    degree: dict[tuple[int, int, int], int]
    ground_plane: bool

    @property
    def node_count(self) -> int:
        return len(self.degree)

    @property
    def wire_element_count(self) -> int:
        return sum(w.segment_count for w in self.wires)

    @property
    def patch_element_count(self) -> int:
        """Always zero.  This dialect has no ``SP``/``SM`` and refuses both."""
        return 0

    @property
    def unknown_count(self) -> int:
        """``Σ max(degree − 1, 0)`` over the fused nodes.

        DERIVED from the ten printouts committed when this rule was written,
        and it reproduces all thirty-nine committed since:
        an interior node of a wire has two element ends and one unknown; a
        free wire end has one and none; a junction of ``m`` wires has ``m``
        and ``m − 1`` (0013's five-wire apex contributes 4, which is what
        turns 30 nodes into 28 unknowns); and a wire end standing in a
        declared ground plane counts its IMAGE as one more element end, which
        is what turns 0019's 11 nodes into 10 unknowns rather than 9.

        The ``GROUND PLANE SPECIFIED`` note and this ``+1`` come off the same
        card — ``GE``'s first field — because that is the card whose note
        says the current "will be interpolated to image in ground plane".
        All 49 captured decks pair a nonzero ``GE`` with a ground ``GN``/
        ``GD``, so no capture can tell ``GE`` from the ground card here.
        """
        return sum(max(count - 1, 0) for count in self.degree.values())

    def index_of(self, tag: int) -> int:
        for index, wire in enumerate(self.wires):
            if wire.tag == tag:
                return index
        raise ServeRefusal(f"no GW card declares tag {tag}")

    def first_element(self, tag: int) -> int:
        """The GLOBAL element number of a tag's first segment, 1-based.

        The ``ANTENNA INPUT PARAMETERS`` row's SEG column is global, not
        wire-local: 0013 drives ``EX 4,5,-1`` and prints ``5    25``, wire 5's
        first segment after the four six-segment radials; 0035 drives
        ``EX 4,12,2`` and prints ``12    21``.
        """
        return 1 + sum(w.segment_count for w in self.wires[: self.index_of(tag)])


def structure_of(deck: Nec5Deck) -> Structure:
    """Count a deck's nodes, elements and unknowns."""
    points = tuple(tuple(node_points(w)) for w in deck.wires)
    ground_plane = deck.ge_flag != 0
    degree: dict[tuple[int, int, int], int] = {}
    for wire, wire_points in zip(deck.wires, points, strict=True):
        last = wire.segment_count
        for k, point in enumerate(wire_points):
            key = _node_key(point)
            ends = 1 if k in (0, last) else 2
            degree[key] = degree.get(key, 0) + ends
    if ground_plane:
        for wire, wire_points in zip(deck.wires, points, strict=True):
            for k in (0, wire.segment_count):
                if abs(wire_points[k][2]) <= _NODE_EPS:
                    key = _node_key(wire_points[k])
                    degree[key] += 1
    return Structure(
        wires=tuple(deck.wires),
        points=points,
        degree=degree,
        ground_plane=ground_plane,
    )


# --------------------------------------------------------------------------
# the mesh: one polyline per GW, cut at every addressed interior node
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Piece:
    """One momwire polyline: a run of one ``GW``'s elements.

    A ``GW`` is one piece unless a card addressed a node INSIDE it AND the
    basis spells a series EMF as a ``node_gaps`` port, in which case it is cut
    there — that port needs the node to be a wire END on both sides, which is
    the same forcing ``momwire.deck._polylines`` applies for a node gap.

    The cut is the NODE-GAP spelling's alone (:func:`build_mesh`, "Two ways to
    spell one series EMF").  A basis without node gaps drives the same node as
    a delta gap at an interior knot of the WHOLE wire, and cutting for it would
    be worse than pointless: a cut that lands one element from a wire's end
    manufactures a one-segment polyline, which carries no tent and which
    ``RazorSolver`` refuses at its constructor.
    """

    tag: int
    first_node: int
    last_node: int
    points: np.ndarray
    radius: float

    @property
    def n_elements(self) -> int:
        return self.last_node - self.first_node

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.points[1] - self.points[0]))


@dataclass
class _Site:
    """One card address, and the momwire port it reaches.

    A site is a DECK port: everything on it — its load, its drive, its rows
    in the two port tables — is in the deck's own convention.  Two fields
    carry it across to the solver's, and they are the whole of the sign story
    (module docstring, "Two conventions"):

    :attr:`column` is the momwire port it lands on, and :attr:`weight` is the
    ``+-1`` that relates the two.  They are usually one address to one port,
    with the weight equal to :attr:`sign`; the exception is the far side of a
    cut another address already declared, which shares the column and takes
    the opposite weight.
    """

    at: Nec5Node
    # "gap" (a ``feeds`` delta gap) or "node" (a ``node_gaps`` series EMF).
    # Which one a through-current node becomes is the BASIS's choice, not the
    # deck's — see :func:`build_mesh`, "Two ways to spell one series EMF".
    spelling: str
    piece: int
    end: str  # "start" | "end" | "interior"
    # +1 when momwire's port quantities are already in NEC's direction: the
    # current the deck's EX drives flows along the favored wire's own
    # end-1 -> end-2 sense.  A node gap named through a wire's END sees the
    # node's outflow the other way round (momwire's sigma), and that sign is
    # the only thing standing between a served current table and one printed
    # 180 degrees out.
    sign: float
    # Where along :attr:`piece` a "gap" spelling sits, in metres from that
    # piece's first point.  Unread by "node", which names an END and not a
    # length.  Decided here rather than in :func:`_solver_for` because an
    # interior gap is the one site whose arclength is not one of the two ends.
    arclength: float = 0.0
    # A gap between a grounded wire END and the plane.  It shares no column
    # with anything: momwire gives each end standing at a grounded point a
    # tent of ITS OWN (`RazorSolver._feed_knots`), because the gap a source
    # there occupies is between the plane and that one wire, so there is no
    # branch pair left to name.  The other two gap sites sit on a knot INSIDE
    # the structure, where a second address is the far side of one port.
    contact: bool = False
    index: int = -1
    column: int = -1
    weight: float = 0.0
    load: complex = 0j
    driven: bool = False


@dataclass
class _Mesh:
    pieces: list[_Piece] = field(default_factory=list)
    junctions: list[list[tuple[int, str]]] = field(default_factory=list)
    sites: list[_Site] = field(default_factory=list)
    # deck element index (0-based, global) -> (piece, element within piece)
    element_of: list[tuple[int, int]] = field(default_factory=list)
    # The solver's ports, in momwire's own order — every gap feed, then every
    # node gap — each named by the site that DECLARED it.  A site that shares
    # a column appears in neither list.
    feeds: list[_Site] = field(default_factory=list)
    gaps: list[_Site] = field(default_factory=list)

    @property
    def n_columns(self) -> int:
        return len(self.feeds) + len(self.gaps)


def _network_ends(deck: Nec5Deck):
    """Every ``TL``/``NT`` endpoint, card by card, end A before end B.

    Lines-then-networks, and since #504 U3 that is a MEASURED order rather than
    a vacuous one.  It used to be true by default — no deck that reached here
    wrote both kinds — and the three that do now reach it, so the claim was
    checked against them instead: 0000's two ``TL`` and one ``NT`` print four
    connection points, 0023's four and one print six, 0025's four and three
    print eight, and all three blocks are in this order to the row.  Every
    captured mixed deck writes its ``TL`` cards first, so the deck's own card
    order and this one are the same order; a deck that interleaves them refuses
    in :func:`refusal` rather than pick which one this becomes.

    The order is not decoration.  The ``STRUCTURE EXCITATION DATA`` block
    prints its connection points in exactly this discovery order, a point named
    twice printed once (0028's six rows against its five cards, and 0025's
    eight against its seven).
    """
    for card in (*deck.transmission_lines, *deck.networks):
        yield card.end_a
        yield card.end_b


def _addressed_nodes(deck: Nec5Deck) -> dict[int, set[int]]:
    """Every ``(tag, node)`` a card puts a port at, gathered per tag."""
    at: dict[int, set[int]] = {}
    for source in deck.sources:
        at.setdefault(source.at.tag, set()).add(source.at.node)
    for load in deck.loads:
        at.setdefault(load.at.tag, set()).add(load.at.node)
    for end in _network_ends(deck):
        at.setdefault(end.tag, set()).add(end.node)
    return at


def build_mesh(
    deck: Nec5Deck, structure: Structure, *, solver_class: type = BSplineSolver
) -> _Mesh:
    """The deck's wires as momwire polylines, with one port per address.

    A ``GW`` is NOT chained onto its neighbours even where two of them share
    an endpoint and a radius.  Chaining would span the joint with one
    continuous B-spline; NEC-5 puts a node there, and so does this — the
    junction's KCL row carries the current across, and the deck's own node
    structure survives into the mesh, which is the whole point of a dialect
    that addresses nodes.

    Two ways to spell one series EMF
    --------------------------------
    A ``GW`` IS cut where a card addressed a node inside it — but only for a
    basis whose ports include ``node_gaps``, which is the one thing
    ``solver_class`` decides here (momwire#603 U1).  The cut exists solely to
    manufacture the two wire ends that port needs, and it is not free: a node
    addressed one element from a wire's end leaves a one-segment polyline,
    which carries no tent at all and which ``RazorSolver`` refuses outright.

    A basis without node gaps spells the same series EMF as a ``feeds`` delta
    gap at that node's own interior knot, on the WHOLE uncut wire.  Neither
    spelling is the true one and the seam does not have to choose: they are
    two discretizations of one source, differing by whether the expansion is
    clamped on each side of the node or runs continuously through it.  On a
    5 m dipole at 30 MHz with 10 elements, fed at the centre, degree-2
    B-splines answer 80.320 + 44.899j cut and 79.117 + 46.321j whole — 2.03 %,
    the same class as the 2–6 % basis envelope this seam already reports
    against the licensed engine, and a difference confined to the two basis
    ends the cut adds.

    The choice is read off the solver's own :class:`~momwire._capabilities.
    Capabilities` row rather than a list kept here, for the reason that module
    exists: a second list is a second thing to forget when a family is added.
    ``momwire.deck._solver``'s ``_NATIVE_LOADING`` is the same shape one axis
    over — a per-family spelling of a port, decided where the family is known.
    """
    mesh = _Mesh()
    node_gaps = bool(solver_class.capabilities.node_gaps)
    addressed = _addressed_nodes(deck)
    piece_of_node: dict[tuple[int, int], tuple[int, str]] = {}
    # (tag, node) -> (piece, metres along it), for an addressed node STRICTLY
    # inside a piece.  Only the delta-gap spelling leaves one there; under the
    # node-gap spelling the cut has already turned every one into a piece end.
    inside_piece: dict[tuple[int, int], tuple[int, float]] = {}

    for wire, points in zip(structure.wires, structure.points, strict=True):
        last = wire.segment_count
        addressed_inside = sorted(
            k for k in addressed.get(wire.tag, ()) if 0 < k < last
        )
        bounds = [0, *addressed_inside, last] if node_gaps else [0, last]
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            index = len(mesh.pieces)
            mesh.pieces.append(
                _Piece(
                    tag=wire.tag,
                    first_node=a,
                    last_node=b,
                    points=np.array([points[a], points[b]], dtype=float),
                    radius=wire.radius,
                )
            )
            # A node that is both one piece's END and the next piece's START
            # is recorded as the start, because that is the end whose sigma is
            # +1 — see `_Site.sign`.  Writing "start" unconditionally and
            # "end" only where nothing claimed the node yet is what makes the
            # later piece win at a cut.
            piece_of_node[(wire.tag, a)] = (index, "start")
            piece_of_node.setdefault((wire.tag, b), (index, "end"))
            for k in addressed_inside:
                if a < k < b:
                    inside_piece[(wire.tag, k)] = (
                        index,
                        float(
                            np.linalg.norm(
                                np.asarray(points[k], dtype=float)
                                - np.asarray(points[a], dtype=float)
                            )
                        ),
                    )
            for element in range(b - a):
                mesh.element_of.append((index, element))

    # -- junctions: fused node key -> the piece ends that meet there --------
    #
    # A node with ONE end is left out on purpose: momwire reads a lone wire end
    # off the geometry as free, or — when it stands in the plane — as a ground
    # contact (`_wire_endpoint_status`, momwire#151).  Declaring it as a
    # one-member junction would pin its current to zero and disconnect the
    # base-fed verticals from their own image.
    ends_at: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
    for index, piece in enumerate(mesh.pieces):
        for which, point in (("start", piece.points[0]), ("end", piece.points[1])):
            ends_at.setdefault(_node_key(point), []).append((index, which))
    for key in sorted(ends_at):
        if len(ends_at[key]) >= 2:
            mesh.junctions.append(sorted(ends_at[key], key=_canonical_end))

    # -- one site per addressed node ---------------------------------------
    for tag, nodes in sorted(addressed.items()):
        for node in sorted(nodes):
            site = _site_for(
                structure,
                mesh,
                piece_of_node,
                inside_piece,
                tag,
                node,
                node_gaps=node_gaps,
            )
            site.index = len(mesh.sites)
            mesh.sites.append(site)
    _assign_columns(mesh)
    return mesh


def _canonical_end(end: tuple[int, str]) -> tuple[int, int]:
    """momwire's own order for the ends of one junction — wire, then start.

    ``momwire._junction_rule.canonical_groups`` is the owner of this rule and
    documents what rides on it: a junction's FIRST end is the one detection
    would have found first, and :meth:`RazorSolver._junction_wings` calls that
    end "side A" — the branch +1 A of the junction's through current flows
    IN along.  :func:`_assign_columns` needs that to sign a delta gap driven
    at the junction, so the seam sorts by momwire's key rather than by the
    tuple's own order, in which ``"end"`` sorts before ``"start"``.

    The two orders differ only where ONE piece brings both of its ends to one
    node — a wire closed into a loop — which no capture writes.  Sorting the
    momwire way regardless is free: a declared spec goes through
    ``canonical_groups`` on the way in, so the solver reorders anything else
    to this anyway, and agreeing by construction beats agreeing by accident.
    """
    return (end[0], 0 if end[1] == "start" else 1)


def _assign_columns(mesh: _Mesh) -> None:
    """Give every site a solver port and the sign that reaches it.

    One site is one port, except where two addresses name the two sides of
    one cut — ``2,3`` and ``3,-1`` at a two-wire node, which is config C of
    the W7EL triple.  momwire allows ONE series gap per junction and is right
    to: at a two-wire node the second gap is not a second cut, it is the far
    side of the first, carrying the same current with the EMFs in series.  So
    the second address shares the first's column and takes the opposite
    weight, and the two-port those two addresses present is the rank-1 one
    the physics has (0017 prints both sides with the same current, 7.7427E-01
    − 4.9839E-01j, and voltages in the 2:1 ratio of their two admittances —
    which is series, and is what makes C 195.34 − j57.458 where A and B are
    114.47 + j21.096).

    The weight follows from KCL and nothing else.  ``sign`` is momwire's
    sigma: the current from the node into the named wire is the current along
    that wire's own direction, times ``sign``.  At a two-wire node the two
    outflows sum to zero, so the far side's momwire current is minus the
    declared side's — which is the ``-sign`` below, and which comes out as
    two EQUAL deck-convention currents whenever the two wires run through the
    node (the 0017 case) and two opposite ones when they meet head to head.

    A delta gap driven AT that junction — the same two addresses under a basis
    without node gaps — shares its column the same way and for the same
    reason, but reads its weight off the junction instead of off the order the
    addresses arrived in.  There is one through-current unknown at the node
    and momwire orients it once: ``+1 A`` flows IN along the junction's first
    end and OUT along the other (:meth:`RazorSolver._junction_wings`, "side
    A"), whoever names it.  So the deck's ``+1`` is ``-sign`` on side A and
    ``+sign`` on side B — the same two numbers the node spelling reaches,
    since ``sign`` is ``+1`` at a ``start`` and ``-1`` at an ``end`` — and
    which end is side A is :func:`_canonical_end`, not arrival order.  A
    ground CONTACT gap is exempt and shares nothing: momwire gives each end
    standing at a grounded point a tent of its own (:attr:`_Site.contact`).
    """
    junction_of = {
        member: index
        for index, members in enumerate(mesh.junctions)
        for member in members
    }
    declared: dict[int, _Site] = {}
    for site in mesh.sites:
        junction = None if site.contact else junction_of.get((site.piece, site.end))
        if junction is not None and site.spelling == "gap":
            site.weight = (
                -site.sign
                if mesh.junctions[junction][0] == (site.piece, site.end)
                else site.sign
            )
        else:
            site.weight = site.sign
        if junction is None:
            # Its own knot, and its own column: a ground contact, or a delta
            # gap at a node this wire alone passes through.
            site.column = len(mesh.feeds)
            mesh.feeds.append(site)
            continue
        first = declared.get(junction)
        if first is None:
            declared[junction] = site
            if site.spelling == "gap":
                site.column = len(mesh.feeds)
                mesh.feeds.append(site)
            else:
                site.column = len(mesh.gaps)
                mesh.gaps.append(site)
            continue
        members = mesh.junctions[junction]
        if len(members) != 2:
            raise ServeRefusal(
                _REFUSE_TWO_CUTS.format(
                    first=f"{first.at.tag},{first.at.written}",
                    second=f"{site.at.tag},{site.at.written}",
                    count=len(members),
                )
            )
        site.column = first.column
        if site.spelling == "node":
            site.weight = -site.sign
    # momwire orders its ports [gap feeds..., junction ports..., node gaps...]
    # and this seam declares no junction ports, so a node gap's column is its
    # place in that list offset by the feeds.  Applied here rather than at the
    # readout because a port index that is wrong is a printout addressed to
    # the wrong wire and nothing red anywhere.
    for site in mesh.sites:
        if site.spelling == "node":
            site.column += len(mesh.feeds)


def _transform(mesh: _Mesh) -> np.ndarray:
    """``T``: solver ports on one side, deck ports on the other.

    ``V_solver = T . V_deck`` for the drives and ``I_deck = T^T . I_solver``
    for the readouts — one matrix for both, which is what keeps the composed
    admittance ``T^T Y T`` symmetric and the power identical on the two
    sides.  It is a signed selection matrix: one nonzero per site, and one
    column per site rather than per port, because a shared cut has two sites
    on one port.
    """
    t = np.zeros((mesh.n_columns, len(mesh.sites)))
    for site in mesh.sites:
        t[site.column, site.index] = site.weight
    return t


def _interior_site(
    structure: Structure,
    mesh: _Mesh,
    inside: tuple[int, float],
    tag: int,
    node: int,
) -> _Site:
    """A series EMF at a node INSIDE a piece, as a delta gap on that piece.

    Reached only under the delta-gap spelling, because the node-gap spelling
    cut the wire here and left no node inside a piece to reach
    (:func:`build_mesh`, "Two ways to spell one series EMF").

    The sign is ``+1`` and there is no case to pick between: the deck's port
    current is the one flowing along the favored wire's own end-1 → end-2
    direction, momwire's delta gap drives the knot's tent in the direction of
    increasing arc length, and on an uncut ``GW`` those are the same
    direction.  That is the same ``+1`` the node-gap spelling reaches by
    another route — its cut hands the address to the LATER piece's ``start``,
    whose sigma is ``+1`` — so the two spellings agree on the sign as well as
    on the source.

    The node must be one this wire alone passes through.  Its degree counts
    ELEMENT ends (:meth:`Structure.unknown_count`), so an untouched interior
    node has exactly two; anything more is another wire's end landing on it,
    which the cut used to weld and a delta gap does not.  Serving that as a
    gap would model a T-junction as an unconnected wire — a well-posed WRONG
    answer — so it refuses instead.  No captured deck writes one: all 78
    addressed interior nodes in the corpus have degree 2.
    """
    piece_index, arclength = inside
    point = structure.points[structure.index_of(tag)][node]
    degree = structure.degree[_node_key(point)]
    if degree != 2:
        raise ServeRefusal(
            f"{tag},{Nec5Node(tag, node).written} addresses a node INSIDE "
            f"wire {tag} where {degree // 2} more element ends meet; this "
            f"basis spells a series EMF as a delta gap, which drives one "
            f"wire's own knot and would leave the others unconnected"
        )
    return _Site(
        at=Nec5Node(tag=tag, node=node),
        spelling="gap",
        piece=piece_index,
        end="interior",
        sign=1.0,
        arclength=arclength,
    )


def _site_for(
    structure: Structure,
    mesh: _Mesh,
    piece_of_node: dict[tuple[int, int], tuple[int, str]],
    inside_piece: dict[tuple[int, int], tuple[int, float]],
    tag: int,
    node: int,
    *,
    node_gaps: bool,
) -> _Site:
    """Which momwire port a ``(favored tag, node)`` address becomes."""
    inside = inside_piece.get((tag, node))
    if inside is not None:
        return _interior_site(structure, mesh, inside, tag, node)
    where = piece_of_node.get((tag, node))
    if where is None:  # pragma: no cover - the parser bounds-checks the node
        raise ServeRefusal(f"no wire piece carries node {node} of tag {tag}")
    piece_index, which = where
    piece = mesh.pieces[piece_index]
    point = piece.points[0 if which == "start" else 1]
    key = _node_key(point)
    grounded = structure.ground_plane and abs(point[2]) <= _NODE_EPS
    meeting = structure.degree[key] - (1 if grounded else 0)

    if meeting >= 2:
        if grounded:
            raise ServeRefusal(
                f"{tag},{Nec5Node(tag, node).written} addresses a node where "
                f"several wires meet IN the ground plane; a series source at a "
                f"grounded junction is not served at this seam"
            )
        return _Site(
            at=Nec5Node(tag=tag, node=node),
            # A K = 2 node is a through-current path, and a basis without node
            # gaps drives it as a delta gap on the junction's own through-
            # current unknown — the same port the interior case reaches, and
            # measured to be the same port: a 12 m dipole at 30 MHz split into
            # two 6-element pieces and driven at the joint answers the uncut
            # 12-element wire driven at its middle knot to the last ulp (2e-16
            # in the impedance, 7e-16 across all thirteen knot currents), and
            # the two ways to NAME the joint answer each other bit for bit.
            # K >= 3 stays a node gap: momwire refuses a delta gap
            # there ("a delta-gap voltage there is ambiguous — it would have
            # to name which pair of branches it drives"), and so it must.
            spelling="node" if node_gaps or meeting >= 3 else "gap",
            piece=piece_index,
            end=which,
            sign=1.0 if which == "start" else -1.0,
            arclength=0.0 if which == "start" else piece.length,
        )
    if grounded:
        return _Site(
            at=Nec5Node(tag=tag, node=node),
            spelling="gap",
            piece=piece_index,
            end=which,
            sign=1.0,
            arclength=0.0 if which == "start" else piece.length,
            contact=True,
        )
    raise ServeRefusal(
        f"{tag},{Nec5Node(tag, node).written} addresses a FREE end of wire "
        f"{tag} - nothing carries current past a lone conductor end, so there "
        f"is no path for a source, a load or a network connection to sit in"
    )


# --------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------


def _medium(ground: Nec5Ground, wavelength: float) -> GroundMedium | None:
    """A finite ground's ``(εr, σ, εc)``, or ``None`` for the two that have none.

    THREE cards arrive here and one medium leaves: ``GN 0``, ``GN 2`` and the
    bare ``GD``, which carry the same two media fields on different mnemonics
    and print the same three cells from them.  What the medium is FOR differs —
    ``GN 0`` solves in it, ``GD`` only reflects off it (module docstring, "One
    medium, two things to do with it") — and that difference is
    :func:`_solver_for`'s and :func:`_far_ground`'s, not this function's.

    Both spellings of the sixth field arrive here too and only one leaves
    (module docstring, "One finite ground"): a POSITIVE field is a conductivity
    and ``εc = εr − j·σ·λ·59.96``; a NEGATIVE one IS ``Im εc``, and the
    conductivity printed beside it is the engine's own back-derivation, which
    is the same division run the other way.  Measured against the linux oracle
    2026-08-20, and it is an identity rather than an approximation:
    ``-12.84`` at 7 MHz prints ``CONDUCTIVITY= 5.000E-03`` and ``-3851.99``
    prints ``1.500E+00``, both to every printed digit.

    The two records spell that ONE convention two ways and the difference is
    the parser's, not the engine's: ``GD`` carries a ``sigma_sets_im_epsc``
    flag (the dialect study measured the convention on that card first), and
    ``GN 0`` carries no flag at all, so its own sign IS the flag — the parser
    sets ``GD``'s from exactly the same ``sigma < 0`` test.  Read the flag
    where there is one; read the sign where there is not.
    """
    if isinstance(ground, Nec5MininecGround):
        negative = ground.sigma_sets_im_epsc
    elif isinstance(ground, Nec5SommerfeldGround):
        negative = ground.sigma < 0.0
    else:
        return None
    if negative:
        eps_c = complex(ground.eps_r, ground.sigma)
        sigma = -ground.sigma / (wavelength * EPSC_CONDUCTIVITY_FACTOR)
    else:
        sigma = ground.sigma
        eps_c = complex(ground.eps_r, -sigma * wavelength * EPSC_CONDUCTIVITY_FACTOR)
    return GroundMedium(eps_r=ground.eps_r, sigma=sigma, eps_c=eps_c)


def _solver_for(
    deck: Nec5Deck, mesh: _Mesh, wavelength: float, medium: GroundMedium | None
) -> BSplineSolver:
    """The constructed solver, one port per declared cut.

    Two kwargs carry the deck's ports and momwire orders their rows
    ``[gap feeds…, junction ports…, node gaps…]`` — the order
    :func:`_assign_columns` already wrote down.

    Three ground kwargs carry the environment, and which of them appear is the
    whole of the difference between the rungs.  ``GN 1`` is ``ground_z``
    alone.  ``GN 0`` adds the medium and asks for momwire's SOMMERFELD model
    rather than its reflection-coefficient one — not a preference: the
    captured decks stand a wire END in the plane, and the refl-coef model is
    documented valid only from about 0.1λ up (``docs/refl-coef-ground-plan.md``,
    momwire#151), so it is the wrong tool for a base-fed vertical by exactly
    the geometry every one of these captures writes.  The medium goes in as
    ``(εr, σ)`` and momwire folds it with the SI ε₀; the 5e-5 that separates
    that from the engine's printed εc is discussed in the module docstring.

    And the bare ``GD`` is ``ground_z`` alone — the SAME constructor call
    ``GN 1`` makes, with no ``ground_eps`` and no ``ground_model``.  That line
    is the mechanical origin of the identity the module docstring measures: the
    MININEC-type ground solves its currents over a PERFECT image and spends its
    medium entirely on the far field, so the deck's ε and σ reach
    :func:`_far_ground` and reach nothing else.  A ``GD`` routed down the
    branch above it would be ``GN 0``, which is 34 % wrong in R.
    """
    radii = [piece.radius for piece in mesh.pieces]
    # Arclength 0 is the grounded end of the piece that starts there; the "end"
    # spelling is the mirror case — a wire whose END stands in the plane —
    # which no capture writes, and which is served because leaving it out would
    # be a silent wrong feed rather than a missing one.  A site INSIDE a piece
    # is neither, and it is why the arclength is decided at the site rather
    # than re-derived from `end` here (momwire#603 U1).
    feeds = [(site.piece, site.arclength, 0j) for site in mesh.feeds]
    gaps = [(site.piece, site.end, 0j) for site in mesh.gaps]

    ground: dict[str, object] = {}
    if isinstance(deck.ground, (Nec5PerfectGround, Nec5MininecGround)):
        ground["ground_z"] = 0.0
    elif medium is not None:
        ground = {
            "ground_z": 0.0,
            "ground_eps": (medium.eps_r, medium.sigma),
            "ground_model": "sommerfeld",
        }

    return BSplineSolver(
        wires=[piece.points for piece in mesh.pieces],
        n_per_edge_per_wire=[[piece.n_elements] for piece in mesh.pieces],
        feeds=feeds,
        junctions=mesh.junctions or None,
        node_gaps=gaps or None,
        degree=_DEGREE,
        wire_radius=radii[0] if len(set(radii)) == 1 else radii,
        wavelength=wavelength,
        **ground,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# the networks: two cards, one composed circuit
# --------------------------------------------------------------------------


def _segment_of(structure: Structure, at: Nec5Node) -> int:
    """The GLOBAL element number a ``(tag, node)`` address prints as.

    Node 0 and node 1 share one: 0012's ``3,-1`` prints segment 10 and 0018's
    ``2,2`` prints 8, so the rule is the tag's first element plus the node,
    with node 0 reading as node 1 and the SIGN carried separately (the
    ``NETWORK DATA`` table prints it as ``-10``; the excitation table prints
    ``10`` and a trailing 2).
    """
    return structure.first_element(at.tag) + max(at.node, 1) - 1


def _node_point(structure: Structure, at: Nec5Node) -> np.ndarray:
    return np.asarray(
        structure.points[structure.index_of(at.tag)][at.node], dtype=float
    )


@dataclass(frozen=True)
class _Card:
    """One ``TL``/``NT`` card, resolved: where it attaches and what it prints.

    :attr:`card` is the nec2 dialect's own record, carrying this card's six
    fields in NEC's order so that :func:`~momwire.deck._networks.card_branches`
    — the one place in the tree that says what they mean — can read them.  Its
    ADDRESSES are not used and cannot be: they are ``(wire, arclength)`` pairs
    from a dialect whose connections land on segment centres, and this one's
    land on nodes.  The two indices beside it are this seam's addressing, and
    the length is resolved before the record is built for the same reason.
    """

    card: NetworkCard
    site_a: int
    site_b: int
    row: NetworkRow | LineRow


def _line_card(
    line: Nec5TransmissionLine, structure: Structure, sites: dict[Nec5Node, int]
) -> _Card:
    """One ``TL``.

    NEC's zero-length rule, measured: a card whose length field is zero is
    the straight-line distance between the two addressed NODES, and the
    printout shows the RESOLVED number.  0028's four crossed feeders all
    write ``0.`` and print 9.9764E-01, 7.9826E-01, 6.3831E-01 and 5.1090E-01,
    which are the node-to-node distances to every printed digit.  SEGMENT
    centres — what the nec2 dialect's own reader measures between, a NEC-2
    network landing on a segment rather than on a node — would give four
    different numbers, so the captured column is what picks the reading.
    """
    length = line.length_m or float(
        np.linalg.norm(
            _node_point(structure, line.end_a) - _node_point(structure, line.end_b)
        )
    )
    if length == 0.0:
        raise ServeRefusal(
            _REFUSE_ZERO_LENGTH_LINE.format(
                first=f"{line.end_a.tag},{line.end_a.written}",
                second=f"{line.end_b.tag},{line.end_b.written}",
            )
        )
    return _Card(
        # F1 keeps its sign: `card_branches` reads the crossed-line polarity
        # flag off it and takes the magnitude for the impedance, which is not
        # the same circuit as a negated Z0.
        card=NetworkCard(kind="TL", payload=(line.z0, length, *line.shunt)),
        site_a=sites[line.end_a],
        site_b=sites[line.end_b],
        row=LineRow(
            tag_from=line.end_a.tag,
            segment_from=_signed_segment(structure, line.end_a),
            tag_to=line.end_b.tag,
            segment_to=_signed_segment(structure, line.end_b),
            z0=abs(line.z0),
            length_m=length,
            shunt_a=complex(line.shunt[0], line.shunt[1]),
            shunt_b=complex(line.shunt[2], line.shunt[3]),
            crossed=line.crossed,
        ),
    )


def _network_card(
    net: Nec5Network, structure: Structure, sites: dict[Nec5Node, int]
) -> _Card:
    """One ``NT``: three complex entries, and ``Y21 = Y12`` by construction."""
    return _Card(
        card=NetworkCard(
            kind="NT",
            payload=(
                net.y11.real,
                net.y11.imag,
                net.y12.real,
                net.y12.imag,
                net.y22.real,
                net.y22.imag,
            ),
        ),
        site_a=sites[net.end_a],
        site_b=sites[net.end_b],
        row=NetworkRow(
            tag_from=net.end_a.tag,
            segment_from=_signed_segment(structure, net.end_a),
            tag_to=net.end_b.tag,
            segment_to=_signed_segment(structure, net.end_b),
            y11=net.y11,
            y12=net.y12,
            y22=net.y22,
        ),
    )


def _signed_segment(structure: Structure, at: Nec5Node) -> int:
    """The ``NETWORK DATA`` address: the segment, negated for a ``-1`` node.

    "Negative segment numbers are a flag, not an index" (capture study,
    "Dialect notes"): the table echoes the deck's own spelling through a
    tag-to-global conversion that the sign survives — 0012's ``NT 3,-1``
    prints ``3   -10`` and 0016's ``NT 2,3`` prints ``2     9`` for the same
    physical point.
    """
    segment = _segment_of(structure, at)
    return -segment if at.written == -1 else segment


def _cards(deck: Nec5Deck, structure: Structure, mesh: _Mesh) -> tuple[_Card, ...]:
    """The deck's network cards, resolved onto sites, in deck order."""
    sites = {site.at: site.index for site in mesh.sites}
    return tuple(
        [_line_card(line, structure, sites) for line in deck.transmission_lines]
        + [_network_card(net, structure, sites) for net in deck.networks]
    )


def _connection_points(cards: tuple[_Card, ...]) -> tuple[int, ...]:
    """The connection-point sites, in the order the printout lists them.

    Discovery order — a card's end A before its end B, cards in deck order, a
    site named twice named once — which reproduces all six captured blocks
    including 0028's, where five cards give six rows.  Unlike the NEC-2
    printout this seam's sibling serves, there is no sourced/unsourced
    partition: 0027 prints its DRIVEN point first because its first card
    names it first, and 0028 prints its driven point fifth for the same
    reason.
    """
    seen: list[int] = []
    for card in cards:
        for site in (card.site_a, card.site_b):
            if site not in seen:
                seen.append(site)
    return tuple(seen)


def _reducer_for(
    cards: tuple[_Card, ...],
    n_sites: int,
    voltages: np.ndarray,
    driven: tuple[int, ...],
) -> NetworkReducer:
    """The deck's cards as one flat network over the site ports.

    Every port that is NOT a network endpoint is pinned with a ``Driven``
    source at its applied voltage, zero included — the reducer floats an
    untouched port at ``I_ext = 0``, which is an OPEN gap, where this
    pipeline's undriven port is a SHORTED one.  A zero-volt pin is the
    reducer's own hard ``V = 0``, so the two conventions meet exactly where
    the network stops.  An endpoint that IS driven gets its source anyway,
    and the reducer's termination branch then carries antenna plus network.

    ``driven`` is what says which endpoints those are, and it is a parameter
    rather than "the ports whose voltage is not zero" because momwire#511's
    probes ask for exactly that distinction.  With one source the two readings
    agree — the probe is a 1 at the driven port — and the sources list this
    builds for a single-source deck is character for character the one #504 U3
    built.  With several, one probe at a time, the OTHER driven ports carry a
    probe voltage of zero and still have to be sources: a driven endpoint that
    silently became a floating network node between the probes and the final
    solve would be two different circuits superposed, which is not superposition
    at all.
    """
    branches: list[object] = []
    endpoints: set[int] = set()
    for entry in cards:
        # `wires=()` is safe and is the point of resolving the length above:
        # the zero-length rule is the only thing `card_branches` would reach
        # for geometry for, and this dialect's rule measures something else.
        branches += card_branches(entry.card, entry.site_a, entry.site_b, ())
        endpoints.update((entry.site_a, entry.site_b))
    held = frozenset(driven)
    sources = [
        Driven(port_name(k), complex(voltages[k]))
        for k in range(n_sites)
        if k not in endpoints or k in held or voltages[k] != 0
    ]
    ports = {port_name(k): PortOnWire(name=port_name(k)) for k in range(n_sites)}
    network = Network(ports=ports, branches=branches, sources=sources)
    return NetworkReducer(network, {port_name(k): k for k in range(n_sites)}, n_sites)


def _reduced_state(
    cards: tuple[_Card, ...],
    n_sites: int,
    voltages: np.ndarray,
    driven: tuple[int, ...],
    y_eff: np.ndarray,
    wavelength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One reducer solve: ``(V_applied, I_structure, I_source)`` at every site.

    The one place this module hands a loaded antenna and a deck's network cards
    to :class:`~momwire.networks.NetworkReducer`, and it is one place on
    purpose: the single-source path, momwire#511's N unit probes and #511's
    final solve all have to go through the SAME reduced system or the
    superposition between them means nothing.  Sharing the function is how that
    is said structurally rather than in a comment — the probes cannot drift from
    the solve they are probing, because there is only one of them.

    Three vectors out and they are three different numbers (:class:`_PortState`
    says which table prints which).  ``V_applied`` is the solve's, with the
    PINNED ports written back from their own boundary condition rather than read
    out — round-off in a number that was exact going in decides the sign of a
    zero, and the printout shows that byte (momwire#456 phase C).  ``I_source``
    is ``I_structure`` everywhere except at a driven site, where it is the
    reducer's TERMINATION-branch current: antenna plus network, which is what
    the generator actually delivered (0027, where one port is 1.4142 A of source
    and 3.3124E-09 A of structure, all of the difference having gone down two
    transmission lines).
    """
    reducer = _reducer_for(cards, n_sites, voltages, driven)
    system = reducer.apply_branches(y_eff, wavelength)
    v, j = system.solve()
    v_applied = np.asarray(v[:n_sites], dtype=np.complex128).copy()
    # A pinned port's voltage is a boundary condition, not a result: reading it
    # back out of the solve only adds round-off to a number that was exact going
    # in, and it decides the sign of a zero — which is a byte the printout shows
    # (the portal restores its driven port voltages for the same reason,
    # momwire#456 phase C).
    for port, volts in zip(reducer.driven_port_idx, reducer.driven_voltages):
        v_applied[port] = volts
    i_port = y_eff @ v_applied
    i_source = i_port.copy()
    # At a driven port the reducer's termination branch carries antenna PLUS
    # network, which is what the source actually delivered.
    for index in driven:
        i_source[index] = j[system.terminations[index][0]]
    return v_applied, i_port, i_source


@dataclass(frozen=True)
class _PortState:
    """What the solve says at every site, in the DECK's convention.

    Four vectors and they are four different numbers, which is the whole
    reason this is a record rather than a return tuple:

    * :attr:`v_applied` — the voltage ACROSS the site, load drop included.
      Both port tables print it.
    * :attr:`v_gap` — what is left of it at the gap, and therefore what
      drives the structure and reconstructs the currents.
    * :attr:`i_port` — the STRUCTURE current at the gap.  The wire-current
      table and the ``STRUCTURE EXCITATION DATA`` block print it.
    * :attr:`i_source` — what the GENERATOR delivered, antenna plus network.
      The ``ANTENNA INPUT PARAMETERS`` row prints it, and it differs from
      ``i_port`` only where the deck drives a network connection point —
      0027, where the same port prints 1.4142 A as a source and 3.3124E-09 A
      as a structure current, all of the difference having gone down the two
      transmission lines.
    """

    v_applied: np.ndarray
    v_gap: np.ndarray
    i_port: np.ndarray
    i_source: np.ndarray
    z_load: np.ndarray


def _port_state(
    deck: Nec5Deck,
    mesh: _Mesh,
    cards: tuple[_Card, ...],
    y_solver: np.ndarray,
    wavelength: float,
) -> _PortState:
    """Solve the deck's ports, networks and all.

    The antenna arrives as momwire's port admittance and is turned once, by
    ``T^T Y T``, into the deck's own convention; everything below that line
    is in the deck's terms and can be read straight off a card or into a
    printed row.

    An ``LD`` is an impedance in the port's own current path, so the deck's
    loads fold in exactly as they do in the portal's NEC-2 half:
    ``V_gap = (1 + Z·Y)^-1 · V_applied`` and ``I = Y·V_gap``.  The generator
    stays ideal — NEC's ``EX`` is an ideal source, and a source impedance
    would divide the drive.  With networks the SAME loaded structure is
    handed to :class:`~momwire.networks.NetworkReducer` as one admittance,
    ``Y_eff = Y·(1 + Z·Y)^-1``, which is the port admittance of the loaded
    structure and therefore what NEC's own network solve sees after ``LD``
    has been added to the matrix diagonal.  The loads are not restamped in
    the reducer: they are already inside, in series inside the port, which is
    where an ``LD`` and a ``TL`` on one node have to meet (0027 pins its
    driven virtual node with ``LD 4,3,1,0,1.E+10,0.`` and hangs both lines
    off it).

    The drive is a UNIT probe and the answer is rescaled afterwards, which is
    what the scored matrix means by ``EX 4`` being a "readout transform":
    with one source the whole system is linear in its voltage, so the volts
    that deliver ``I0`` are found by dividing rather than by a second fill.
    ``EX 0`` takes the same road with the scale known in advance.

    A PHASED drive is that same trick with the division become an inversion,
    and it goes down :func:`_multi_drive_state` rather than through here.  The
    single-source path below is left exactly as #504 U3 left it — not for
    elegance but because forty-six captured printouts are byte-gated against
    it, and a rearrangement that is algebraically identical is not necessarily
    identical in its last bits.
    """
    t = _transform(mesh)
    y = t.T @ y_solver @ t
    n = len(mesh.sites)
    z_load = np.array([site.load for site in mesh.sites], dtype=np.complex128)
    if len(deck.sources) > 1:
        return _multi_drive_state(deck, mesh, cards, y, z_load, wavelength)
    (source,) = deck.sources
    (driven,) = [site.index for site in mesh.sites if site.driven]
    probe = np.zeros(n, dtype=np.complex128)
    probe[driven] = 1.0
    loaded = np.eye(n, dtype=np.complex128) + z_load[:, None] * y

    if not cards:
        # No network reaches this deck, so the source current IS the port
        # current and the applied voltage is the gap's plus the load drop.
        # This branch is U4's arithmetic, conjugated by T and no more, which
        # is what keeps every bare capture's printout where it was.
        v_gap = np.linalg.solve(loaded, probe)
        i_port = y @ v_gap
        i_source = i_port
        v_applied = v_gap + z_load * i_port
    else:
        y_eff = y if not np.any(z_load) else np.linalg.solve(loaded.T, y.T).T
        v_applied, i_port, i_source = _reduced_state(
            cards, n, probe, (driven,), y_eff, wavelength
        )
        v_gap = v_applied - z_load * i_port

    scale = (
        source.drive
        if source.kind == 0
        else source.drive / complex(i_source[driven])  # EX 4: the current is set
    )
    return _PortState(
        v_applied=v_applied * scale,
        v_gap=v_gap * scale,
        i_port=i_port * scale,
        i_source=i_source * scale,
        z_load=z_load,
    )


def _multi_drive_state(
    deck: Nec5Deck,
    mesh: _Mesh,
    cards: tuple[_Card, ...],
    y: np.ndarray,
    z_load: np.ndarray,
    wavelength: float,
) -> _PortState:
    """Several ``EX 4`` cards at once: the CONSTRAINED current drive.

    A phased array is the one place in this dialect where a drive is not a
    scale.  Four cards fix four currents simultaneously, the four ports are
    coupled through the structure, and no single number rescales the answer —
    0031's own row table is the proof, since its four ports report four
    different impedances and one of them is NEGATIVE.

    The algebra is the single-source branch's, read the other way round.  The
    loaded structure's port admittance ``Y_eff = Y·(1 + Z·Y)^-1`` maps applied
    voltages to port currents; an UNDRIVEN site is shorted, ``V = 0``, which is
    the same boundary condition the unit probe already imposes on every site it
    is not at; so the drive is the square sub-block on the driven sites::

        Y_eff[driven, driven] · V_driven = I_spec

    one solve, no second fill, and the whole of what "phased" costs over
    "scaled".  The undriven sites then read out of the same ``Y_eff`` and the
    gap voltages come back through the load drop exactly as they do above.

    ``I_spec`` is in the DECK's convention, which is where every number in this
    module lives (module docstring, "Two conventions"): the current an ``EX 4``
    sets is the one flowing along the favored wire's own end-1 → end-2
    direction, and ``T`` has already put ``y`` on that side of the transform.

    The set currents are restored afterwards rather than read back.  They come
    out of the solve equal to what went in, to round-off — that is what the
    equation says — and round-off is exactly what would print ``1.1102E-16``
    where the capture prints ``0.0000E+00`` (0032's second row sets a pure
    imaginary current and its real part is a zero that has to stay one).  The
    U1 restore rule, once per row.

    And with a NETWORK on the deck (momwire#511) the sub-block ``Y_eff`` is not
    the map to invert, because what an ``EX 4`` fixes is the SOURCE current —
    antenna plus network, the reducer's termination branch — and at a driven
    site that is also a connection point those are two different numbers.  0120
    prints both: the same port reads 1.4142 A of source and 3.1645E-11 A of
    structure, the rest having gone down an ``NT`` and two ``TL``s.  So the map
    is measured instead, one column at a time, exactly as this module measures
    everything else it cannot write down: N unit probes through the SAME reduced
    system the final solve uses (:func:`_reduced_state`), each giving one column
    of ``M``, the map from the driven sites' applied VOLTAGES to their source
    currents.  ``M·V = I_spec`` is then the same N×N solve U4 already does, and
    one last reducer solve at ``V`` produces the full state.

    N + 1 reducer solves where the network-free branch does one, and that is the
    honest price rather than an accepted inefficiency: N is 2 or 4 on every deck
    the corpus writes, the antenna's own fill and factor happened once before any
    of this, and a one-solve formulation that reproduced the captures would still
    have had to be checked against this one.  Superposition is what makes it
    exact — every probe and the final solve carry the same branches, the same
    zero-volt pins on the undriven non-endpoint sites and the same
    network-governed endpoints, so the columns are columns of ONE linear map and
    not four different circuits.  The load is inside ``Y_eff`` on all of them
    (module docstring; :func:`_port_state`), which is what keeps 0120's four
    ``LD`` cards — two 18 Ω base resistors and two 1e10 Ω virtual pins — in the
    probes as well as in the answer.
    """
    n = len(mesh.sites)
    driven = [site.index for site in mesh.sites if site.driven]
    loaded = np.eye(n, dtype=np.complex128) + z_load[:, None] * y
    y_eff = y if not np.any(z_load) else np.linalg.solve(loaded.T, y.T).T

    site_of = {site.at: site.index for site in mesh.sites}
    row_of = {index: k for k, index in enumerate(driven)}
    spec = np.zeros(len(driven), dtype=np.complex128)
    for source in deck.sources:
        spec[row_of[site_of[source.at]]] = source.drive

    if not cards:
        v_applied = np.zeros(n, dtype=np.complex128)
        v_applied[driven] = np.linalg.solve(y_eff[np.ix_(driven, driven)], spec)
        i_port = y_eff @ v_applied
    else:
        m = np.zeros((len(driven), len(driven)), dtype=np.complex128)
        for column, index in enumerate(driven):
            probe = np.zeros(n, dtype=np.complex128)
            probe[index] = 1.0
            _v, _i, i_probe = _reduced_state(
                cards, n, probe, tuple(driven), y_eff, wavelength
            )
            m[:, column] = i_probe[driven]
        volts = np.zeros(n, dtype=np.complex128)
        volts[driven] = np.linalg.solve(m, spec)
        v_applied, i_port, _i_source = _reduced_state(
            cards, n, volts, tuple(driven), y_eff, wavelength
        )
    i_source = i_port.copy()
    i_source[driven] = spec
    return _PortState(
        v_applied=v_applied,
        v_gap=v_applied - z_load * i_port,
        i_port=i_port,
        i_source=i_source,
        z_load=z_load,
    )


# --------------------------------------------------------------------------
# the readouts
# --------------------------------------------------------------------------


def _ratio(numerator: complex, denominator: complex) -> complex:
    """``a / b``, answering an open port with an infinity rather than raising.

    No capture reaches it — the pinned virtual nodes carry ~1e-32 A and print
    ~1e+25 Ω rather than nothing at all — but a port that a network leaves
    exactly open is one subtraction away, and a printout is the only channel
    this engine has.
    """
    if denominator == 0:
        return complex(math.inf, math.inf) if numerator else 0j
    return numerator / denominator


def _port_row(
    structure: Structure, mesh: _Mesh, state: _PortState, index: int
) -> PortRow:
    """One ``STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS`` row.

    The voltage is the one ACROSS the site, load drop included: 0027's driven
    virtual node prints 1.0000E+10 − 2.6855E+04j there, which is its ``LD``
    pin plus the little wire's own gap impedance, and only an applied voltage
    can say that.  The current is the STRUCTURE's, not the source's.
    """
    site = mesh.sites[index]
    voltage = complex(state.v_applied[index])
    current = complex(state.i_port[index])
    return PortRow(
        tag=site.at.tag,
        segment=_segment_of(structure, site.at),
        # The trailing index tracks the DECK's spelling, 9 of 9 rows in the
        # capture study: node 0 written -1 prints 2, a positive node field
        # prints 1.  Both sides of 0017's junction are connection points and
        # they print 2 and 1 accordingly, on one geometric node.
        end_index=2 if site.at.written == -1 else 1,
        voltage=voltage,
        current=current,
        impedance=_ratio(voltage, current),
        admittance=_ratio(current, voltage),
        power=0.5 * (voltage * current.conjugate()).real,
    )


def _source_row(
    structure: Structure, site: _Site, state: _PortState, source: Nec5Source
) -> PortRow:
    """One ``ANTENNA INPUT PARAMETERS`` row, for one ``EX`` card.

    Whichever of the two quantities the card SET is a boundary condition, not a
    result: reading it back out of the solve only adds round-off to a number
    that was exact going in, and it decides the sign of a zero — 0010's
    ``EX 4 … 1.414214,0.`` came back with an imaginary part of 1.1e-16 and
    printed ``1.1102E-16`` where the capture prints ``0.0000E+00``.  The portal
    restores its driven port voltages for the same reason (momwire#456 phase
    C); this is that rule on the other card, and #504 U4 applies it once per
    row rather than once per deck.

    ``POWER`` is this row's own ``½·Re(V·I*)`` and it is allowed to be
    NEGATIVE.  0031's tag-1 row prints ``-1.7790E+00`` watts against a
    ``-1.7790E+00`` Ω resistance: the element is absorbing from the other three
    through the mutual coupling, its generator is being driven rather than
    driving, and the array's INPUT POWER is the sum of the four rows with that
    one taken off.  Nothing here clamps, flags or apologises for it — a
    negative row is arithmetic, and the budget that carries it is the engine's.
    """
    voltage = complex(state.v_applied[site.index])
    current = complex(state.i_source[site.index])
    if source.kind == 0:
        voltage = source.drive
    else:
        current = source.drive
    return PortRow(
        tag=source.at.tag,
        segment=_segment_of(structure, source.at),
        # The trailing index tracks the DECK's spelling, 9 of 9 rows in the
        # capture study: node 0 written -1 prints 2, a positive node field
        # prints 1.  0031 and 0032 print ``2`` on every one of their six rows,
        # which is six more of the same.
        end_index=2 if source.at.written == -1 else 1,
        voltage=voltage,
        current=current,
        impedance=_ratio(voltage, current),
        admittance=_ratio(current, voltage),
        power=0.5 * (voltage * current.conjugate()).real,
    )


def _check_one_port_per_drive(
    deck: Nec5Deck, by_address: dict[Nec5Node, _Site]
) -> None:
    """Two ``EX`` cards may not land on one solver column.

    :func:`_drive_refusal` already caught the two cards that write the SAME
    address; this is the other way to reach one port, and only the mesh can see
    it — ``2,3`` and ``3,-1`` are two addresses, two sites and one cut, sharing
    a column by :func:`_assign_columns` because that is the rank-1 two-port
    they really are.  One EX card there is a source in the gap; two are one
    boundary condition written twice, with nothing to say which wins.
    """
    seen: dict[int, Nec5Node] = {}
    for source in deck.sources:
        site = by_address[source.at]
        first = seen.get(site.column)
        if first is not None:
            raise ServeRefusal(
                _REFUSE_TWO_DRIVES_ONE_PORT.format(
                    first=f"{first.tag},{first.written}",
                    second=f"{source.at.tag},{source.at.written}",
                )
            )
        seen[site.column] = source.at


def _element_geometry(structure: Structure, wavelength: float):
    """``(centres, lengths, tags)`` for every deck element, in global order.

    Both columns are WAVELENGTH-NORMALIZED, which is what the mixed-case note
    over each table says and what the captures show: 0019's 1.03 m segments
    print 2.40494E-02 = 1.03/(299.8/7).
    """
    centres, lengths, tags = [], [], []
    for wire, points in zip(structure.wires, structure.points, strict=True):
        for k in range(wire.segment_count):
            a = np.asarray(points[k], dtype=float)
            b = np.asarray(points[k + 1], dtype=float)
            centres.append(0.5 * (a + b) / wavelength)
            lengths.append(float(np.linalg.norm(b - a)) / wavelength)
            tags.append(wire.tag)
    return centres, lengths, tags


def _element_currents_and_charges(
    solver: BSplineSolver, mesh: _Mesh, coeffs: np.ndarray, omega: float
):
    """Per deck element: the current at its centre and the charge density there.

    The current is the mean of the element's two end-knot currents, which is
    momwire's own element convention (``_ElementCurrents``).  The charge
    density is ``q = −(1/jω)·dI/ds`` evaluated AT the centre through
    :meth:`~momwire.bspline.BSplineSolver.current_slopes` — differentiated in
    the basis rather than around it, so the printed C/m is the one this
    expansion actually carries rather than a difference of two samples of it.

    Both are read along each piece's own arc, and every piece runs in its
    ``GW``'s end-1 → end-2 direction because :func:`build_mesh` never
    reverses one; so no re-signing is needed and NEC's current convention is
    already this one.
    """
    knot_currents = solver.currents_at_knots(coeffs)
    centres_per_piece = []
    for piece in mesh.pieces:
        step = piece.length / piece.n_elements
        centres_per_piece.append((np.arange(piece.n_elements) + 0.5) * step)
    slopes = solver.current_slopes(coeffs, centres_per_piece)

    currents, charges = [], []
    for piece_index, element in mesh.element_of:
        knots = np.asarray(knot_currents[piece_index])
        currents.append(0.5 * (knots[element] + knots[element + 1]))
        charges.append(-slopes[piece_index][element] / (1j * omega))
    return currents, charges


def _far_ground(deck: Nec5Deck, medium: GroundMedium | None) -> Ground:
    """The environment in the far-field readout's own vocabulary.

    Three shapes and no fourth: ``pec`` is the geometric image, ``free`` is no
    image at all, and a MEDIUM is that image weighted by the Fresnel
    coefficients of it.  The name is the PORTAL's for a family of ground
    models and not a claim about which integral answered the near field — the
    far field of a Sommerfeld solve is a Fresnel-weighted image in NEC too
    (the reflected wave is a plane wave at infinity), which is why one shape
    covers both halves here.

    Which is also why the two finite rungs spell that shape differently while
    reaching the same arithmetic.  ``refl`` and ``sommerfeld`` are ONE branch
    in ``_far_moments`` — both fall through to ``_image_coeffs``, and the seam
    gates that they answer bit for bit — so the choice between them buys
    nothing but a name, and a name is worth spending on the truth.  ``GN 0``
    says ``sommerfeld`` because a Sommerfeld integral really did answer its
    near field and the far field is that same run's.  ``GD`` says ``refl``
    because nothing Sommerfeld happened anywhere in its solve: the currents
    came off a PEC image (:func:`_solver_for`) and the only thing the medium
    ever did was weight the reflection.  Claiming otherwise in a served run's
    own vocabulary would be the banner's lie repeated where nobody forced it —
    the captures make the ENGINE print ``SOMMERFELD SOLUTION`` over a ``GD``
    run and this seam reproduces that byte for byte, but it does not have to
    believe it.
    """
    if isinstance(deck.ground, Nec5PerfectGround):
        return Ground("pec")
    if medium is None:
        return Ground("free")
    if isinstance(deck.ground, Nec5MininecGround):
        return Ground("refl", medium.eps_r, medium.sigma)
    return Ground("sommerfeld", medium.eps_r, medium.sigma)


def _pattern(
    request: Nec5FarFieldRequest,
    solver: BSplineSolver,
    coeffs: np.ndarray,
    ground: Ground,
    frequency_mhz: float,
    wavelength: float,
    p_in: float,
) -> PatternBlock:
    """One ``RP 0`` card's answer.

    The moments, the E components and the polarisation ellipse are the
    portal's — one owner for the far-field readout — and the two conventions
    a reader would want checked are checked against captures: the printed E
    is ``r·E`` in volts (0010's 8.89384E+01 at ``theta = 90`` gives
    ``4πr²·|E|²/2η / P_in`` = 1.650 = 2.18 dB, the HOR column on that row),
    and a gain below NEC's own ``DB10`` floor prints -999.99 with a blank
    SENSE (0013's zenith row, where both components are 1e-13 dust).

    Negative theta needs no special case: 0044 sweeps 90 degrees down to -90
    and its rows are what the spherical formulae give when a negative theta
    is put straight into them (the E(THETA) phase flips by 180 across the
    zenith, 87.28 to -92.72, which is theta_hat turning over).

    Neither does the HORIZON, and that is the ground rung's own evidence.
    0044 over perfect ground prints its PEAK, 5.15 dB, at theta = 90; 0047,
    the same wire over 13/0.005 earth, prints -999.99 there and 0.06 dB from
    the capture at every other angle of the same cut.  Nothing here tests for
    grazing incidence: the Fresnel coefficients go to -1 as theta_i goes to
    90, the direct wave and its weighted image cancel term for term, and the
    null falls out of ``_far_moments`` on the row the capture put it.
    """
    thetas = request.theta0_deg + request.d_theta_deg * np.arange(request.n_theta)
    phis = request.phi0_deg + request.d_phi_deg * np.arange(request.n_phi)
    k = 2.0 * math.pi / wavelength
    mid, moment, _nodes, _delta = solver.element_currents(coeffs)
    m_theta, m_phi = _far_moments(
        mid,
        moment,
        k,
        np.radians(thetas),
        np.radians(phis),
        ground,
        0.0,
        frequency_mhz * 1e6,
    )
    e_theta = -1j * ETA0 * k / (4.0 * math.pi) * m_theta
    e_phi = -1j * ETA0 * k / (4.0 * math.pi) * m_phi
    norm = ETA0 * k * k / (8.0 * math.pi * p_in) if p_in > 0 else 0.0
    g_v = norm * np.abs(m_theta) ** 2
    g_h = norm * np.abs(m_phi) ** 2
    floor_scale = 1.0 / wavelength

    rows = []
    for j in range(request.n_phi):
        for i in range(request.n_theta):
            et, ep = complex(e_theta[i, j]), complex(e_phi[i, j])
            # A component that is EXACTLY zero — the vertical's zenith, a
            # linear dipole's co-polar null, where a spherical unit vector
            # has an exactly zero component and every moment term cancels
            # term for term — still carries the sign of its zeros out of the
            # sum, and ``atan2(-0.0, 0.0)`` prints ``-0.00`` where the
            # capture prints ``0.00``.  The angle of zero is not a number;
            # normalize the zero rather than invent an angle for it.  Only
            # an exact zero is touched: a dust component is a real (tiny)
            # field with a real phase, and 0010's TILT column reads +90 or
            # -90 off it (compare its ``phi = 0`` and ``phi = 1`` rows).
            if et == 0:
                et = 0j
            if ep == 0:
                ep = 0j
            axial, tilt, sense = _polarisation(et, ep, floor_scale)
            rows.append(
                PatternRow(
                    theta_deg=float(thetas[i]),
                    phi_deg=float(phis[j]),
                    vert_db=_gain_db(float(g_v[i, j])),
                    hor_db=_gain_db(float(g_h[i, j])),
                    total_db=_gain_db(float(g_v[i, j] + g_h[i, j])),
                    axial_ratio=axial,
                    tilt_deg=tilt,
                    sense=sense,
                    e_theta_magnitude=abs(et),
                    e_theta_phase_deg=math.degrees(math.atan2(et.imag, et.real)),
                    e_phi_magnitude=abs(ep),
                    e_phi_phase_deg=math.degrees(math.atan2(ep.imag, ep.real)),
                )
            )

    if request.xnda % 10 == 0:
        # XNDA's A digit asks for the average gain; 1000 (0010, 0044) does
        # not and 1001 (0013, 0035) does.
        return PatternBlock(rows=tuple(rows))
    average, solid = _average_gain(
        g_v + g_h, thetas, request.d_theta_deg, request.d_phi_deg, request.n_phi
    )
    return PatternBlock(
        rows=tuple(rows),
        average_power_gain=average,
        solid_angle_pi=solid / math.pi,
        power_radiated_4pi=average * p_in,
    )


# How finely the solved current is resampled before it is summed at an
# observation point.  The far field never needs it — every mesh element is
# already electrically small and only the radiation-zone limit survives — but
# a point a metre from a metre-long element resolves the variation along it.
# The portal's own near-field constant, and MEASURED to be converged here:
# 0115 reads 1.33 % worst-cell magnitude at ``subdiv = 1``, 1.92 % at 4,
# 1.9479 % at 8 and 1.9572 % at 32, and 0109 reads 10.34 / 5.02 / 5.4494 /
# 5.5834 %.  Everything past 8 moves the answer by less than a tenth of a
# percent, so what is left at 8 is the formulation difference and not the
# sampling — which is what a subdivision constant has to be able to say.
_NEAR_FIELD_SUBDIV = 8


def _near_medium(
    deck: Nec5Deck, solver: BSplineSolver, medium: GroundMedium | None
) -> tuple[complex, complex] | None:
    """``(ε̃, C₂)`` for the medium a near field is EVALUATED in, or ``None``
    when the ground has no medium to evaluate in (free space, ``GN 1``).

    Both finite cards land here and they arrive by different roads, which is
    the whole content of the function.  ``GN 0`` solved IN its medium, so
    `_solver_for` gave the solver a ``ground_eps`` and
    :func:`~momwire._ground_spec.ground_config` — the one owner of ``C₂`` in
    this tree — reads it straight back off the solve.  The bare ``GD`` solved
    over a PERFECT image and its solver never saw the medium at all, so the
    same call would hand back the PEC row (``eps_tilde=None``,
    ``image_coefficient=1.0``): right for its CURRENTS (the ``Z ≡ GN 1``
    identity) and wrong for its near field, which the engine solves in the
    medium (module docstring, "One near field, four grounds and one point").
    So ``GD``'s ε̃ is folded here from the deck's own medium instead, through
    :func:`~momwire._ground_refl.eps_tilde` — the same function
    ``ground_config`` would have called, given the same ``(εr, σ)`` — and
    ``C₂`` is written out in the one expression ``ground_config`` writes it
    in.  Two roads, one arithmetic; a ``GD`` near field that came back with
    ``C₂ = 1`` and no remainder would be the aliasing bug, and it is gated as
    one (``tests/test_field_point.py``).
    """
    if medium is None:
        return None
    config = _ground_spec.ground_config(solver, solver.omega)
    if config is not None and config.mode == "compose":
        assert config.eps_tilde is not None
        return config.eps_tilde, complex(config.image_coefficient)
    eps_t = _ground_refl.eps_tilde(
        (medium.eps_r, medium.sigma), solver.omega, solver.eps
    )
    return eps_t, (eps_t - 1.0) / (eps_t + 1.0)


def _near_field(
    request: Nec5NearFieldRequest,
    solver: BSplineSolver,
    coeffs: np.ndarray,
    deck: Nec5Deck,
    mesh: _Mesh,
    wavelength: float,
    medium: GroundMedium | None,
) -> NearFieldBlock:
    """One ``NE``/``NH`` card's answer, over all four ground cards.

    The readout is the PORTAL's — ``_element_fields``, the mixed-potential
    form the nec2 front end already answers ``NE``/``NH`` with — for the same
    reason the far-field readout is: one owner per readout.  The ground
    REMAINDER at a point is :mod:`momwire._field_point`'s, for the same reason
    again.  What this function owns is the grid, the composition and the units.

    THE GRID is :func:`_grid_points`', shared with the refusal so that the
    points a table prints and the points a refusal inspects are one list.

    THE COMPOSITION is one line with four spellings, and which one a deck gets
    is its ground card's:

      ``GN -1``  the elements alone
      ``GN 1``   the elements plus their geometric mirror
      ``GN 0``   ``direct + C₂·image + remainder``, ε̃ off the Sommerfeld solve
      ``GD``     the same, ε̃ off the deck's medium (the currents are the
                 PEC-image solve's; the near field is still in the medium)

    The IMAGE is the same one the far field uses — the geometric mirror with
    the horizontal moments flipped and the continuity charge NEGATED, which is
    one statement twice over (reversing a horizontal current reverses dI/ds,
    and mirroring a vertical one reverses the arc direction).  Over the two
    finite grounds that image is scaled by ``C₂ = (ε̃−1)/(ε̃+1)`` and a smooth
    REMAINDER is added to it, which is NEC's own decomposition of the
    half-space Green's function (theory manual eqs 136-147) evaluated at a
    point rather than between two wire elements.

    The association is `_field_ground.FieldGround`'s ``"compose"`` contract and
    not a style: the coefficient goes on the LEFT of the block and ``coef·img
    + rem`` is associated before the outer sum, because that contract is about
    float64 evaluation order.

    THE UNITS.  Volts and amps per metre at the point, PEAK, in whatever basis
    the deck's own ``EX`` card set — nothing is scaled.  EZNEC writes
    ``1.414214`` for 1 A and its GUI divides the printed field back down by
    √2 to display RMS (0107: 6.6673E+02 printed, 471.465 shown).

    A component the geometry kills exactly — ``EY`` everywhere on an
    axis-symmetric vertical's grid, ``HX`` and ``HZ`` on the same one — comes
    out of the sum as a signed zero or as fill round-off, and the angle of a
    zero is not a number.  Floored at the portal's own bar and printed as
    ``0.0000E+00     0.00``, which is what the captures print in those columns
    (0109's whole ``EY`` column, 0111's ``HZ``).  The engine's own finite-ground
    tables print interpolation DUST in those columns instead (0110's ``EY`` at
    1.7342E-02 of a 1.5E+01 table), which is why the served envelopes gate them
    by magnitude and never by phase.
    """
    points = _grid_points(request)
    k = 2.0 * math.pi / wavelength
    radius = min(piece.radius for piece in mesh.pieces)
    mid, moment, nodes, delta = solver.element_currents(
        coeffs, subdiv=_NEAR_FIELD_SUBDIV
    )
    elements = (mid, moment, nodes, delta)
    field = _element_fields(points, elements, k, radius, request.magnetic)
    near = _near_medium(deck, solver, medium)
    if isinstance(deck.ground, Nec5PerfectGround) or near is not None:
        mid_img, moment_img = _image_moments(mid, moment, 0.0)
        nodes_img = nodes.copy()
        nodes_img[:, 2] = -nodes[:, 2]
        image = _element_fields(
            points,
            (mid_img, moment_img, nodes_img, -delta),
            k,
            radius,
            request.magnetic,
        )
        if near is None:
            field = field + image
        else:
            eps_t, coef = near
            evaluate = (
                _field_point.reflected_h_field_at
                if request.magnetic
                else _field_point.reflected_field_at
            )
            remainder = evaluate(points, mid, moment, eps_t, 0.0, k, solver.omega)
            np.multiply(coef, image, out=image)
            field = field + (image + remainder)

    rows = []
    for point, value in zip(points, field, strict=True):
        cells = [complex(component) for component in value]
        for index, cell in enumerate(cells):
            if cell.real * cell.real + cell.imag * cell.imag <= _FIELD_FLOOR2:
                cells[index] = 0j
        rows.append(
            NearFieldRow(
                point=(float(point[0]), float(point[1]), float(point[2])),
                magnitudes=tuple(abs(cell) for cell in cells),
                phases_deg=tuple(
                    math.degrees(math.atan2(cell.imag, cell.real)) for cell in cells
                ),
            )
        )
    return NearFieldBlock(rows=tuple(rows), magnetic=request.magnetic)


def _average_gain(gain, thetas, d_theta, d_phi, n_phi) -> tuple[float, float]:
    """``(average power gain, solid angle)`` over the sampled directions.

    The quadrature is the portal's, which recovered it from nec2c fixtures:
    each theta sample owns the band between its half-step neighbours, clipped
    to the requested range, so the bands telescope and a full sphere comes
    out at exactly ``4*PI`` — which is what 0013 and 0035 print.
    """
    lo = np.radians(np.maximum(thetas - 0.5 * d_theta, thetas[0]))
    hi = np.radians(np.minimum(thetas + 0.5 * d_theta, thetas[-1]))
    band = np.cos(lo) - np.cos(hi)
    columns = max(n_phi - 1, 1)
    step = math.radians(d_phi) if d_phi else 2.0 * math.pi
    total = float(np.sum(gain[:, :columns] * band[:, None])) * step
    solid = float(np.sum(band)) * columns * step
    return (total / solid if solid else 0.0), abs(solid)


# --------------------------------------------------------------------------
# the unit
# --------------------------------------------------------------------------


def serve(deck: Nec5Deck) -> RunData:
    """Solve one rung-1 deck and return everything its printout reports.

    Raises :class:`ServeRefusal` for a deck :func:`refusal` passed but the
    translation cannot stand behind; call :func:`refusal` first for the
    cheap, card-level answer.
    """
    reason = refusal(deck)
    if reason is not None:
        raise ServeRefusal(reason)

    structure = structure_of(deck)
    mesh = build_mesh(deck, structure)
    by_address = {site.at: site for site in mesh.sites}
    for load in deck.loads:
        by_address[load.at].load += load.impedance
    for source in deck.sources:
        by_address[source.at].driven = True
    _check_one_port_per_drive(deck, by_address)

    frequency = float(deck.frequency_mhz or 0.0)
    wavelength = SPEED_OF_LIGHT_MHZ_M / frequency
    omega = 2.0 * math.pi * frequency * 1e6

    medium = _medium(deck.ground, wavelength)
    cards = _cards(deck, structure, mesh)
    solver = _solver_for(deck, mesh, wavelength, medium)
    solution = solver.compute_port_solution()
    state = _port_state(deck, mesh, cards, solution.y, wavelength)
    # Back across T: the structure is driven by the SOLVER's gap EMFs, which
    # is the one place besides the admittance that the two conventions meet.
    coeffs = solution.coeffs @ (_transform(mesh) @ state.v_gap)

    # One row per EX card, in DECK order — which is the order 0031 prints its
    # four in, and which is not the order the sites were built in (those are
    # sorted by tag and node).  Every row is the same three spelling rules the
    # single-source row always followed; what is new is that there are several
    # of them and that the budget is their SUM.
    source_rows = tuple(
        _source_row(structure, by_address[source.at], state, source)
        for source in deck.sources
    )
    p_in = float(sum(row.power for row in source_rows))
    # Every wire is a perfect conductor at this seam (the dialect has no
    # LD 5 and no IS), so the budget's WIRE LOSS line carries the LD loads'
    # watts and nothing else — which is where 0012's two 1.E+10 pins print
    # theirs (5.0797E-54 W) and where 0027's single pin prints 7.1165E-08.
    p_load = 0.5 * float(np.sum(np.real(state.z_load) * np.abs(state.i_port) ** 2))
    points = _connection_points(cards)
    connections = tuple(_port_row(structure, mesh, state, index) for index in points)
    # The budget's own arithmetic (module docstring): RADIATED is INPUT plus
    # what the connection points delivered, less the load's watts, and the
    # NETWORK LOSS line is the negative of the first sum — printed only when
    # it is positive, which is 4 of the 6 captured network budgets.
    p_network = -sum(row.power for row in connections)
    p_radiated = p_in - p_network - p_load

    centres, lengths, tags = _element_geometry(structure, wavelength)
    currents, charges = _element_currents_and_charges(solver, mesh, coeffs, omega)

    def phase(value: complex) -> float:
        return math.degrees(math.atan2(value.imag, value.real))

    return RunData(
        node_count=structure.node_count,
        wire_element_count=structure.wire_element_count,
        patch_element_count=structure.patch_element_count,
        unknown_count=structure.unknown_count,
        frequency_mhz=frequency,
        wavelength_m=wavelength,
        environment=(
            ENVIRONMENT_FINITE_GROUND
            if medium is not None
            else ENVIRONMENT_PERFECT_GROUND
            if isinstance(deck.ground, Nec5PerfectGround)
            else ENVIRONMENT_FREE_SPACE
        ),
        ground=medium,
        loads=tuple(
            LoadRow(
                tag=load.at.tag,
                # The loading table prints the DECODED node and drops the
                # deck's spelling, which is the opposite of what NETWORK DATA
                # does with the same address (:func:`_signed_segment`).  0025
                # settles it: ``LD 4,1,-1`` prints ``1    1`` and ``LD 4,5,3``
                # prints ``3    3``, so the rule is the segment a node names,
                # wire-local — node 0 reading as node 1 exactly as it does in
                # :func:`_segment_of`, and no sign surviving anywhere.
                #
                # #504 U1's four loaded captures could not say this: all eight
                # of their ``LD`` cards write a positive node.  The nine mixed
                # and feed-system captures that landed with U3 write ``-1``
                # twenty-two times and print ``1`` twenty-two times.
                node_from=max(load.at.node, 1),
                node_thru=max(load.at.node, 1),
                resistance=load.impedance.real,
                reactance=load.impedance.imag or None,
            )
            for load in deck.loads
        ),
        networks=tuple(card.row for card in cards),
        network_excitation=connections,
        sources=source_rows,
        currents=tuple(
            WireCurrentRow(
                element=index + 1,
                tag=tag,
                center=tuple(centre),
                length=length,
                real=value.real,
                imag=value.imag,
                magnitude=abs(value),
                phase_deg=phase(value),
            )
            for index, (tag, centre, length, value) in enumerate(
                zip(tags, centres, lengths, currents, strict=True)
            )
        ),
        charges=(
            tuple(
                ChargeRow(
                    element=index + 1,
                    tag=tag,
                    center=tuple(centre),
                    length=length,
                    magnitude=abs(value),
                    phase_deg=phase(value),
                )
                for index, (tag, centre, length, value) in enumerate(
                    zip(tags, centres, lengths, charges, strict=True)
                )
            )
            if deck.pq is not None
            else ()
        ),
        power=PowerBudget(
            input_power=p_in,
            radiated_power=p_radiated,
            wire_loss=p_load,
            network_loss=p_network if p_network > 0 else None,
            efficiency_percent=(100.0 * p_radiated / p_in) if p_in > 0 else 0.0,
        ),
        near_fields=tuple(
            _near_field(request, solver, coeffs, deck, mesh, wavelength, medium)
            for request in deck.requests
            if isinstance(request, Nec5NearFieldRequest)
        ),
        patterns=tuple(
            _pattern(
                request,
                solver,
                coeffs,
                _far_ground(deck, medium),
                frequency,
                wavelength,
                p_in,
            )
            for request in deck.requests
            if isinstance(request, Nec5FarFieldRequest)
        ),
    )
