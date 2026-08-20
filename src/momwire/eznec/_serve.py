"""Rung-1 physics: a parsed ``nec5`` deck, solved, as a :class:`RunData`.

U2 read the deck, U3 laid out the bytes; this is the middle — the unit that
turns ``(favored tag, node)`` addressing, a ground card and a request into a
momwire solve and reads the answer back into the numbers the printout wants.
It computes physics and formats nothing, exactly as :mod:`._printout`
formats and computes nothing.

Rung 1 of the scored ladder (antennaknobs
``docs/status/2026-08-20-eznec-nec5-scored-matrix.md``) is what is served:
``GN -1`` free space and ``GN 1`` perfect ground, one ``EX 0``/``EX 4``
source at a node, ``LD 4`` fixed impedances, node-addressed ``TL``/``NT``
networks, and the ``RP 0`` / ``XQ`` / ``PQ 0`` requests.  Everything above
it — the Sommerfeld ground, the bare ``GD`` MININEC mode, phased multi-``EX``
drive, the near field — refuses BY NAME through :func:`refusal`, because a
seam that answers a question it has no gate for is worse than one that says
so.  A network deck over a ground this seam cannot solve refuses at the
GROUND rung, which is what the ordering in :func:`refusal` is for.

Courtesy stance, the arc's throughout: every NEC-5 fact below was measured
off captured decks and captured printouts under ``tests/fixtures/eznec/``,
and is cited by capture id.  No NEC-5 source, algorithm or internal
structure is described or relied on.

The basis, and why it is the one
--------------------------------
``BSplineSolver`` at the repo's own default configuration — ``degree=2``,
``feed_model="point"``, no extended kernel, no enrichment — which is what
``momwire.deck.build_solver``'s ``"bspline"`` entry constructs and what the
portal solves every NEC-2 deck with.  The family is not a free choice: NEC-5
addresses NODES, and ``node_gaps`` (momwire#305, the apex-feed arc) is the
only port momwire has AT a node.  The sinusoidal family has none and cannot
stand at this seam.

Three drive spellings, and the address picks between them
---------------------------------------------------------
* a node where two or more wire ends meet — including the two halves of a
  wire this module CUT because a card addressed a node inside it — is a
  junction, and the source is a ``node_gaps`` series EMF named through the
  favored wire's end (0013's ``EX 4,5,-1``, the five-wire apex; 0010's
  ``EX 4,1,6``, six segments into an eleven-segment dipole);
* a wire end standing IN the ground plane is momwire's ground-contact feed
  — a delta gap at arclength 0 on the grounded piece, the idiom
  ``tests/test_contact_nec5_lane.py`` already gates against this engine's
  own base-fed monopole numbers (0019/0043/0044's ``EX 4,1,-1``);
* a FREE wire end is refused.  There is no through-current path at a lone
  conductor end for a series EMF to sit in, momwire says so at the
  constructor, and no captured deck asks for one.

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

The budget's own arithmetic
---------------------------
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
from ..deck._nec5 import (
    Nec5Deck,
    Nec5FarFieldRequest,
    Nec5MininecGround,
    Nec5NearFieldRequest,
    Nec5Network,
    Nec5Node,
    Nec5PerfectGround,
    Nec5SommerfeldGround,
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
    _far_moments,
    _gain_db,
    _polarisation,
)
from ._printout import (
    ENVIRONMENT_FREE_SPACE,
    ENVIRONMENT_PERFECT_GROUND,
    ChargeRow,
    LineRow,
    LoadRow,
    NetworkRow,
    PatternBlock,
    PatternRow,
    PortRow,
    PowerBudget,
    RunData,
    WireCurrentRow,
)

__all__ = ["BASIS", "SPEED_OF_LIGHT_MHZ_M", "refusal", "serve"]


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

_REFUSE_SOMMERFELD = (
    "GN 0 asks for the finite-ground Sommerfeld solution, which this engine "
    "does not yet serve at this seam: rung 1 is GN -1 (free space) and GN 1 "
    "(perfect ground)"
)
_REFUSE_MININEC = (
    "GD asks for the MININEC-type ground (PEC currents with a second-medium "
    "far field), which this engine does not yet serve at this seam: rung 1 is "
    "GN -1 (free space) and GN 1 (perfect ground)"
)
_REFUSE_MIXED_NETWORKS = (
    "this deck carries TL and NT cards together; the NETWORK DATA table has "
    "one heading per card type and no captured printout shows what a table "
    "carrying both looks like, so the layout is refused rather than guessed"
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
_REFUSE_MULTI_EX = (
    "this deck carries {count} EX cards; a phased multi-source drive is not "
    "served at this seam yet, and one EX is what every rung-1 capture writes"
)
_REFUSE_NE = (
    "NE (near electric field) is not served at this seam yet: the one captured "
    "NE deck stands over GN 0, so a near field answered here would have no "
    "captured printout to be gated against"
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
    """The reason this deck is not rung-1, or ``None`` when it is.

    Order is deliberate — grounds first, then the cards, then the drive, then
    the request — so a deck that is out of scope in several ways names the
    reason a reader would fix first.  The corpus leans on it: every 4-square
    feed-system deck carries six ``TL`` cards over a bare ``GD``, and what it
    needs to hear is that the GROUND is unserved, not that its networks are
    (they are not).
    """
    ground = deck.ground
    if isinstance(ground, Nec5SommerfeldGround):
        return _REFUSE_SOMMERFELD
    if isinstance(ground, Nec5MininecGround):
        return _REFUSE_MININEC
    if deck.transmission_lines and deck.networks:
        return _REFUSE_MIXED_NETWORKS
    if not deck.sources:
        return _REFUSE_NO_EX
    if len(deck.sources) > 1:
        return _REFUSE_MULTI_EX.format(count=len(deck.sources))
    if deck.frequency_mhz is None or deck.frequency_mhz <= 0.0:
        return _REFUSE_NO_FR
    for request in deck.requests:
        if isinstance(request, Nec5NearFieldRequest):
            return _REFUSE_NE
        if isinstance(request, Nec5FarFieldRequest) and request.range_m != 0.0:
            return _REFUSE_RP_RANGE
    if not deck.requests:
        return _REFUSE_NO_REQUEST
    return None


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

        DERIVED from the ten captured printouts, and it reproduces all ten:
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

    A ``GW`` is one piece unless a card addressed a node INSIDE it, in which
    case it is cut there — a series EMF at a node needs the node to be a wire
    END on both sides, which is the same forcing ``momwire.deck._polylines``
    applies for a node gap.
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
    # "gap" (a delta gap on a grounded wire end) or "node" (a series node gap)
    spelling: str
    piece: int
    end: str  # "start" | "end"
    # +1 when momwire's port quantities are already in NEC's direction: the
    # current the deck's EX drives flows along the favored wire's own
    # end-1 -> end-2 sense.  A node gap named through a wire's END sees the
    # node's outflow the other way round (momwire's sigma), and that sign is
    # the only thing standing between a served current table and one printed
    # 180 degrees out.
    sign: float
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

    Card order is the deck's: no deck that reaches here writes both kinds
    (the three that do stand over a ``GD`` ground and refuse a rung earlier,
    and a mixed deck under a served ground refuses for its table layout), so
    lines-then-networks IS the order they were read in.  The order is not
    decoration — the
    ``STRUCTURE EXCITATION DATA`` block prints its connection points in
    exactly this discovery order, a point named twice printed once (0028's
    six rows against its five cards).
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


def build_mesh(deck: Nec5Deck, structure: Structure) -> _Mesh:
    """The deck's wires as momwire polylines, with one port per address.

    A ``GW`` is NOT chained onto its neighbours even where two of them share
    an endpoint and a radius.  Chaining would span the joint with one
    continuous B-spline; NEC-5 puts a node there, and so does this — the
    junction's KCL row carries the current across, and the deck's own node
    structure survives into the mesh, which is the whole point of a dialect
    that addresses nodes.
    """
    mesh = _Mesh()
    addressed = _addressed_nodes(deck)
    piece_of_node: dict[tuple[int, int], tuple[int, str]] = {}

    for wire, points in zip(structure.wires, structure.points, strict=True):
        last = wire.segment_count
        inside = sorted(k for k in addressed.get(wire.tag, ()) if 0 < k < last)
        bounds = [0, *inside, last]
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
            mesh.junctions.append(sorted(ends_at[key]))

    # -- one site per addressed node ---------------------------------------
    for tag, nodes in sorted(addressed.items()):
        for node in sorted(nodes):
            site = _site_for(structure, mesh, piece_of_node, tag, node)
            site.index = len(mesh.sites)
            mesh.sites.append(site)
    _assign_columns(mesh)
    return mesh


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
    """
    junction_of = {
        member: index
        for index, members in enumerate(mesh.junctions)
        for member in members
    }
    declared: dict[int, _Site] = {}
    for site in mesh.sites:
        if site.spelling == "gap":
            site.column, site.weight = len(mesh.feeds), site.sign
            mesh.feeds.append(site)
            continue
        junction = junction_of[(site.piece, site.end)]
        first = declared.get(junction)
        if first is None:
            declared[junction] = site
            site.column, site.weight = len(mesh.gaps), site.sign
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
        site.column, site.weight = first.column, -site.sign
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


def _site_for(
    structure: Structure,
    mesh: _Mesh,
    piece_of_node: dict[tuple[int, int], tuple[int, str]],
    tag: int,
    node: int,
) -> _Site:
    """Which momwire port a ``(favored tag, node)`` address becomes."""
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
            spelling="node",
            piece=piece_index,
            end=which,
            sign=1.0 if which == "start" else -1.0,
        )
    if grounded:
        return _Site(
            at=Nec5Node(tag=tag, node=node),
            spelling="gap",
            piece=piece_index,
            end=which,
            sign=1.0,
        )
    raise ServeRefusal(
        f"{tag},{Nec5Node(tag, node).written} addresses a FREE end of wire "
        f"{tag} - nothing carries current past a lone conductor end, so there "
        f"is no path for a source, a load or a network connection to sit in"
    )


# --------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------


def _solver_for(deck: Nec5Deck, mesh: _Mesh, wavelength: float) -> BSplineSolver:
    """The constructed solver, one port per declared cut.

    Two kwargs carry the deck's ports and momwire orders their rows
    ``[gap feeds…, junction ports…, node gaps…]`` — the order
    :func:`_assign_columns` already wrote down.
    """
    radii = [piece.radius for piece in mesh.pieces]
    feeds = [
        (
            site.piece,
            # Arclength 0 is the grounded end of the piece that starts there.
            # The "end" spelling is the mirror case — a wire whose END stands
            # in the plane — which no capture writes; it is here because
            # leaving it out would be a silent wrong feed rather than a
            # missing one.
            0.0 if site.end == "start" else mesh.pieces[site.piece].length,
            0j,
        )
        for site in mesh.feeds
    ]
    gaps = [(site.piece, site.end, 0j) for site in mesh.gaps]

    return BSplineSolver(
        wires=[piece.points for piece in mesh.pieces],
        n_per_edge_per_wire=[[piece.n_elements] for piece in mesh.pieces],
        feeds=feeds,
        junctions=mesh.junctions or None,
        node_gaps=gaps or None,
        degree=_DEGREE,
        wire_radius=radii[0] if len(set(radii)) == 1 else radii,
        wavelength=wavelength,
        ground_z=0.0 if isinstance(deck.ground, Nec5PerfectGround) else None,
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
    cards: tuple[_Card, ...], n_sites: int, voltages: np.ndarray
) -> NetworkReducer:
    """The deck's cards as one flat network over the site ports.

    Every port that is NOT a network endpoint is pinned with a ``Driven``
    source at its applied voltage, zero included — the reducer floats an
    untouched port at ``I_ext = 0``, which is an OPEN gap, where this
    pipeline's undriven port is a SHORTED one.  A zero-volt pin is the
    reducer's own hard ``V = 0``, so the two conventions meet exactly where
    the network stops.  An endpoint that IS driven gets its source anyway,
    and the reducer's termination branch then carries antenna plus network.
    """
    branches: list[object] = []
    endpoints: set[int] = set()
    for entry in cards:
        # `wires=()` is safe and is the point of resolving the length above:
        # the zero-length rule is the only thing `card_branches` would reach
        # for geometry for, and this dialect's rule measures something else.
        branches += card_branches(entry.card, entry.site_a, entry.site_b, ())
        endpoints.update((entry.site_a, entry.site_b))
    sources = [
        Driven(port_name(k), complex(voltages[k]))
        for k in range(n_sites)
        if k not in endpoints or voltages[k] != 0
    ]
    ports = {port_name(k): PortOnWire(name=port_name(k)) for k in range(n_sites)}
    network = Network(ports=ports, branches=branches, sources=sources)
    return NetworkReducer(network, {port_name(k): k for k in range(n_sites)}, n_sites)


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
    """
    t = _transform(mesh)
    y = t.T @ y_solver @ t
    n = len(mesh.sites)
    z_load = np.array([site.load for site in mesh.sites], dtype=np.complex128)
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
        reducer = _reducer_for(cards, n, probe)
        system = reducer.apply_branches(y_eff, wavelength)
        v, j = system.solve()
        v_applied = np.asarray(v[:n], dtype=np.complex128).copy()
        # A pinned port's voltage is a boundary condition, not a result:
        # reading it back out of the solve only adds round-off to a number
        # that was exact going in, and it decides the sign of a zero — which
        # is a byte the printout shows (the portal restores its driven port
        # voltages for the same reason, momwire#456 phase C).
        for port, volts in zip(reducer.driven_port_idx, reducer.driven_voltages):
            v_applied[port] = volts
        i_port = y_eff @ v_applied
        i_source = i_port.copy()
        # At a driven port the reducer's termination branch carries antenna
        # PLUS network, which is what the source actually delivered.
        i_source[driven] = j[system.terminations[driven][0]]
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


def _pattern(
    request: Nec5FarFieldRequest,
    solver: BSplineSolver,
    coeffs: np.ndarray,
    deck: Nec5Deck,
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
    """
    thetas = request.theta0_deg + request.d_theta_deg * np.arange(request.n_theta)
    phis = request.phi0_deg + request.d_phi_deg * np.arange(request.n_phi)
    k = 2.0 * math.pi / wavelength
    pec = isinstance(deck.ground, Nec5PerfectGround)
    mid, moment, _nodes, _delta = solver.element_currents(coeffs)
    m_theta, m_phi = _far_moments(
        mid,
        moment,
        k,
        np.radians(thetas),
        np.radians(phis),
        Ground("pec") if pec else Ground("free"),
        0.0,
        (deck.frequency_mhz or 0.0) * 1e6,
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
    (source,) = deck.sources
    by_address[source.at].driven = True

    frequency = float(deck.frequency_mhz or 0.0)
    wavelength = SPEED_OF_LIGHT_MHZ_M / frequency
    omega = 2.0 * math.pi * frequency * 1e6

    cards = _cards(deck, structure, mesh)
    solver = _solver_for(deck, mesh, wavelength)
    solution = solver.compute_port_solution()
    state = _port_state(deck, mesh, cards, solution.y, wavelength)
    # Back across T: the structure is driven by the SOLVER's gap EMFs, which
    # is the one place besides the admittance that the two conventions meet.
    coeffs = solution.coeffs @ (_transform(mesh) @ state.v_gap)

    site = by_address[source.at]
    voltage = complex(state.v_applied[site.index])
    current = complex(state.i_source[site.index])
    # Whichever of the two the card SET is a boundary condition, not a
    # result: reading it back out of the solve only adds round-off to a
    # number that was exact going in, and it decides the sign of a zero —
    # 0010's ``EX 4 … 1.414214,0.`` came back with an imaginary part of
    # 1.1e-16 and printed ``1.1102E-16`` where the capture prints
    # ``0.0000E+00``.  The portal restores its driven port voltages for the
    # same reason (momwire#456 phase C); this is that rule on the other card.
    if source.kind == 0:
        voltage = source.drive
    else:
        current = source.drive
    p_in = 0.5 * float((voltage * np.conj(current)).real)
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
            ENVIRONMENT_PERFECT_GROUND
            if isinstance(deck.ground, Nec5PerfectGround)
            else ENVIRONMENT_FREE_SPACE
        ),
        loads=tuple(
            LoadRow(
                tag=load.at.tag,
                node_from=load.at.written,
                node_thru=load.at.written,
                resistance=load.impedance.real,
                reactance=load.impedance.imag or None,
            )
            for load in deck.loads
        ),
        networks=tuple(card.row for card in cards),
        network_excitation=connections,
        sources=(
            PortRow(
                tag=source.at.tag,
                segment=_segment_of(structure, source.at),
                # The trailing index tracks the DECK's spelling, 9 of 9 rows
                # in the capture study: node 0 written -1 prints 2, a
                # positive node field prints 1.
                end_index=2 if source.at.written == -1 else 1,
                voltage=complex(voltage),
                current=complex(current),
                impedance=complex(voltage / current),
                admittance=complex(current / voltage),
                power=p_in,
            ),
        ),
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
        patterns=tuple(
            _pattern(request, solver, coeffs, deck, wavelength, p_in)
            for request in deck.requests
            if isinstance(request, Nec5FarFieldRequest)
        ),
    )
