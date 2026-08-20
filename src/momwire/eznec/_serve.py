"""Rung-1 physics: a parsed ``nec5`` deck, solved, as a :class:`RunData`.

U2 read the deck, U3 laid out the bytes; this is the middle — the unit that
turns ``(favored tag, node)`` addressing, a ground card and a request into a
momwire solve and reads the answer back into the numbers the printout wants.
It computes physics and formats nothing, exactly as :mod:`._printout`
formats and computes nothing.

Rung 1 of the scored ladder (antennaknobs
``docs/status/2026-08-20-eznec-nec5-scored-matrix.md``) is what is served:
``GN -1`` free space and ``GN 1`` perfect ground, one ``EX 0``/``EX 4``
source at a node, ``LD 4`` fixed impedances, and the ``RP 0`` / ``XQ`` /
``PQ 0`` requests.  Everything above it — the Sommerfeld ground, the bare
``GD`` MININEC mode, ``TL``/``NT`` networks, phased multi-``EX`` drive, the
near field — refuses BY NAME through :func:`refusal`, because a seam that
answers a question it has no gate for is worse than one that says so.

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
    Nec5Node,
    Nec5PerfectGround,
    Nec5SommerfeldGround,
    Nec5Wire,
)

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
    LoadRow,
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
_REFUSE_TL = (
    "TL (transmission line) is not served at this seam yet - a node-addressed "
    "network is the gate unit of this arc and is answered against the W7EL "
    "oracle triple, not against this engine's own arithmetic"
)
_REFUSE_NT = (
    "NT (two-port network) is not served at this seam yet - a node-addressed "
    "network is the gate unit of this arc and is answered against the W7EL "
    "oracle triple, not against this engine's own arithmetic"
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

    Order is deliberate — grounds first, then networks, then the drive, then
    the request — so a deck that is out of scope in several ways names the
    reason a reader would fix first.
    """
    ground = deck.ground
    if isinstance(ground, Nec5SommerfeldGround):
        return _REFUSE_SOMMERFELD
    if isinstance(ground, Nec5MininecGround):
        return _REFUSE_MININEC
    if deck.transmission_lines:
        return _REFUSE_TL
    if deck.networks:
        return _REFUSE_NT
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
    """One momwire port, and the card address that asked for it."""

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
    port: int = -1
    load: complex = 0j
    driven: bool = False


@dataclass
class _Mesh:
    pieces: list[_Piece] = field(default_factory=list)
    junctions: list[list[tuple[int, str]]] = field(default_factory=list)
    sites: list[_Site] = field(default_factory=list)
    # deck element index (0-based, global) -> (piece, element within piece)
    element_of: list[tuple[int, int]] = field(default_factory=list)


def _addressed_nodes(deck: Nec5Deck) -> dict[int, set[int]]:
    """Every ``(tag, node)`` a card puts a port at, gathered per tag."""
    at: dict[int, set[int]] = {}
    for source in deck.sources:
        at.setdefault(source.at.tag, set()).add(source.at.node)
    for load in deck.loads:
        at.setdefault(load.at.tag, set()).add(load.at.node)
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
            mesh.sites.append(_site_for(structure, mesh, piece_of_node, tag, node))
    return mesh


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
        f"is no path for a source or a load to sit in"
    )


# --------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------


def _solver_for(deck: Nec5Deck, mesh: _Mesh, wavelength: float) -> BSplineSolver:
    """The constructed solver, with every site's port row recorded on it.

    Two kwargs carry the deck's sources and loads and momwire orders their
    rows ``[gap feeds…, junction ports…, node gaps…]``.  This seam declares no
    junction ports, so a site's row is its place in whichever of the two lists
    it went into, offset by the first list's length — written out here rather
    than derived at the readout, because a port index that is wrong is a
    printout addressed to the wrong wire and nothing red anywhere.
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
        for site in mesh.sites
        if site.spelling == "gap"
    ]
    gaps = [
        (site.piece, site.end, 0j) for site in mesh.sites if site.spelling == "node"
    ]
    port = 0
    for site in mesh.sites:
        if site.spelling == "gap":
            site.port = port
            port += 1
    for site in mesh.sites:
        if site.spelling == "node":
            site.port = port
            port += 1

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


def _port_algebra(deck: Nec5Deck, mesh: _Mesh, y: np.ndarray):
    """``(v_gap, i_port, v_source, z_load)``, in MOMWIRE's sign convention.

    An ``LD`` is an impedance in the port's own current path, so the deck's
    loads fold in exactly as they do in the portal's NEC-2 half:
    ``V_gap = (1 + Z·Y)^-1 · V_source`` and ``I = Y·V_gap``.  The generator
    stays ideal — NEC's ``EX`` is an ideal source, and a source impedance
    would divide the drive.

    Two voltages come out and they are two different numbers.  ``V_gap`` is
    what drives the structure and reconstructs the currents; ``V_source`` is
    what the generator applied, load drop included, and it is the one the
    ``ANTENNA INPUT PARAMETERS`` row prints — so a load stamped on the driven
    node lands INSIDE the reported impedance, which is where NEC puts it (it
    adds ``LD`` to the impedance matrix's own diagonal before anything reads
    a driving point off it).  Without a load the two are one number, which is
    every captured rung-1 deck.

    ``EX 4`` is the current source, and the scored matrix's "readout
    transform" is this line: with one port driven, ``I = Y_eff·V`` is linear
    in the drive, so the voltage that delivers ``I0`` is ``I0`` divided by
    the loaded structure's own driving-point admittance, and the solve is the
    voltage-driven one rescaled rather than a second fill.
    """
    (source,) = deck.sources
    (site,) = [s for s in mesh.sites if s.driven]
    n = y.shape[0]
    z_load = np.zeros(n, dtype=np.complex128)
    for other in mesh.sites:
        z_load[other.port] += other.load
    system = np.eye(n, dtype=np.complex128) + z_load[:, None] * y

    unit = np.zeros(n, dtype=np.complex128)
    unit[site.port] = 1.0
    if source.kind == 0:
        # A voltage source drives the gap it sits in; the deck's volts are in
        # the favored wire's direction, momwire's in the polyline's.
        scale = site.sign * source.drive
    else:
        i_of_unit = y @ np.linalg.solve(system, unit)
        scale = site.sign * source.drive / i_of_unit[site.port]

    v_gap = np.linalg.solve(system, unit * scale)
    i_port = y @ v_gap
    return v_gap, i_port, v_gap + z_load * i_port, z_load


# --------------------------------------------------------------------------
# the readouts
# --------------------------------------------------------------------------


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

    solver = _solver_for(deck, mesh, wavelength)
    solution = solver.compute_port_solution()
    v_gap, i_port, v_source, z_load = _port_algebra(deck, mesh, solution.y)
    coeffs = solution.coeffs @ v_gap

    site = by_address[source.at]
    voltage = site.sign * v_source[site.port]
    current = site.sign * i_port[site.port]
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
    # The only dissipation rung 1 can have: every wire is a perfect conductor
    # (this dialect has no LD 5 and no IS) and there are no networks, so the
    # budget's WIRE LOSS line carries the LD loads' watts and nothing else —
    # which is where 0012's two 1.E+10 pins print theirs (5.0797E-54 W).
    p_load = 0.5 * float(np.sum(np.real(z_load) * np.abs(i_port) ** 2))
    p_radiated = p_in - p_load

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
        sources=(
            PortRow(
                tag=source.at.tag,
                segment=structure.first_element(source.at.tag)
                + max(source.at.node, 1)
                - 1,
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
            efficiency_percent=(100.0 * p_radiated / p_in) if p_in > 0 else 0.0,
        ),
        patterns=tuple(
            _pattern(request, solver, coeffs, deck, wavelength, p_in)
            for request in deck.requests
            if isinstance(request, Nec5FarFieldRequest)
        ),
    )
