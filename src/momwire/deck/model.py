"""The dialect-neutral deck model.

``momwire.deck.parse()`` returns a :class:`DeckModel` whatever dialect it
read.  The normative contract for this module — the model's vocabulary, and
card by card what every dialect front-end must put in it — is the reference
page ``site/src/content/docs/reference/deck-grammar-nec2.md`` ("The nec2 deck
dialect"), section ``#the-deckmodel``.  The code may not drift from that page
without a test going red; the tests cite its anchors.

The model speaks momwire's language and only momwire's: wires, arclengths,
feeds, gaps, loads, grounds.  NEC's tags, segment numbers and card mnemonics
stop at the dialect front-end.  A second dialect (a NEC-5 flavour is the
probable second) is a second parser, not a second model, so anything a
consumer needs in *card* terms — an echo, a table addressed by segment —
travels alongside the model rather than inside it.

Nothing here is populated by the geometry front-end alone: ``wires`` and the
framing fields are filled by unit B of momwire#359, and the excitation,
loading, ground and request fields by the units that follow.  The dataclass is
complete from the start so the seam never moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "DeckModel",
    "DeckWire",
    "WireMaterial",
    "LoadSpec",
    "SecondMedium",
    "FarFieldRequest",
    "NearFieldRequest",
    "PrintControl",
    "ExecuteGroup",
]


# A point is a plain 3-tuple of floats rather than an ndarray: the model is a
# frozen, hashable, comparable record (a solver cache keys on it), and numpy
# arrays are neither.  Consumers that want (M, 3) call np.asarray themselves.
Point = tuple[float, float, float]


@dataclass(frozen=True)
class WireMaterial:
    """A wire's optional material: finite conductivity and/or a dielectric
    jacket.

    Both are per *wire* — never per element — because both are wire
    properties in every dialect that has them (NEC's ``LD 5`` and ``IS``),
    and because momwire's solvers take one value per wire.  This is the
    model's "insulation" record; the spec folds it into the wire row rather
    than giving it a top-level field.
    """

    # Mhos/metre.  None is a perfect conductor (momwire's default).
    conductivity: float | None = None
    # Jacket outer radius in metres and its relative permittivity.  Both or
    # neither: a jacket with no radius is not a jacket.
    insulation_radius: float | None = None
    insulation_eps_r: float | None = None


@dataclass(frozen=True)
class DeckWire:
    """One conductor: a polyline of ``M`` vertices, ``M-1`` edges.

    ``edge_elements[k]`` is how many solver elements edge ``k`` carries, so a
    wire's element boundaries are exactly the deck's — no remeshing happens
    between a card and a matrix row.  ``radius`` is carried per wire with no
    averaging across the structure.

    The ``nec2`` front-end emits two-vertex wires (a ``GW`` card is a straight
    wire, and a wire another wire's end lands on mid-span is *split* at that
    point so the crossing is a shared wire END — see the spec's
    ``#connections``).  Longer polylines are legal and a NEC-5 dialect will
    produce them; nothing downstream may assume ``M == 2``.
    """

    vertices: tuple[Point, ...]
    radius: float
    edge_elements: tuple[int, ...]
    material: WireMaterial | None = None

    def __post_init__(self) -> None:
        if len(self.vertices) < 2:
            raise ValueError("a wire needs at least two vertices")
        if len(self.edge_elements) != len(self.vertices) - 1:
            raise ValueError(
                f"edge_elements has {len(self.edge_elements)} entries for "
                f"{len(self.vertices) - 1} edges"
            )
        if self.radius <= 0.0:
            raise ValueError(f"wire radius must be > 0, got {self.radius}")

    @property
    def edge_lengths(self) -> tuple[float, ...]:
        return tuple(
            math.dist(a, b) for a, b in zip(self.vertices[:-1], self.vertices[1:])
        )

    @property
    def length(self) -> float:
        return math.fsum(self.edge_lengths)

    @property
    def n_elements(self) -> int:
        return sum(self.edge_elements)

    def point_at(self, arclength: float) -> Point:
        """The point ``arclength`` metres along the polyline from vertex 0.

        Clamped at both ends: an arclength off the wire is a caller bug, but
        a clamp keeps a rounding-noise overshoot from producing a point off
        the conductor.
        """
        remaining = arclength
        lengths = self.edge_lengths
        for k, seg_len in enumerate(lengths):
            if remaining <= seg_len or k == len(lengths) - 1:
                t = 0.0 if seg_len == 0.0 else max(0.0, min(1.0, remaining / seg_len))
                a, b = self.vertices[k], self.vertices[k + 1]
                return tuple(ai + (bi - ai) * t for ai, bi in zip(a, b))  # type: ignore[return-value]
            remaining -= seg_len
        raise AssertionError("unreachable: a wire always has one edge")


@dataclass(frozen=True)
class LoadSpec:
    """A lumped impedance stamped at a point on a wire.

    ``kind`` is ``"series"`` (R + jωL + 1/jωC), ``"parallel"`` (their parallel
    combination) or ``"fixed"`` (a frequency-independent ``r + 1j*x``).  The
    load is stamped as a port impedance *outside* the solver — see the spec's
    ``#one-geometry-one-port-set`` — so it costs a gap in the fill and nothing
    else.
    """

    kind: str
    r: float = 0.0
    l: float = 0.0  # noqa: E741 — NEC's own field name for the inductance
    c: float = 0.0
    x: float = 0.0

    def impedance(self, freq_hz: float) -> complex:
        omega = 2.0 * math.pi * freq_hz
        if self.kind == "fixed":
            return complex(self.r, self.x)
        # A zero L or C drops out of the branch it is in rather than dividing
        # by zero: NEC reads an absent element as absent, not as a short.
        if self.kind == "series":
            z = complex(self.r, 0.0)
            if self.l:
                z += 1j * omega * self.l
            if self.c:
                z += 1.0 / (1j * omega * self.c)
            return z
        if self.kind == "parallel":
            y = 0j
            if self.r:
                y += 1.0 / self.r
            if self.l:
                y += 1.0 / (1j * omega * self.l)
            if self.c:
                y += 1j * omega * self.c
            return 1.0 / y if y != 0 else complex("inf")
        raise ValueError(f"unknown load kind {self.kind!r}")


@dataclass(frozen=True)
class SecondMedium:
    """A second ground medium and the edge where medium 1 stops.

    It reaches an answer only through a far-field request's cliff modes; the
    moment method never sees it (spec ``#gd--additional-ground-medium``).
    """

    eps_r: float = 0.0
    sigma: float = 0.0
    edge_distance: float = 0.0
    height: float = 0.0


@dataclass(frozen=True)
class FarFieldRequest:
    """A far-field pattern.  ``mode`` 0 is the plain space wave, 2 and 3 the
    linear and circular cliffs that read :class:`SecondMedium`."""

    mode: int = 0
    n_theta: int = 1
    n_phi: int = 1
    theta0_deg: float = 0.0
    phi0_deg: float = 0.0
    d_theta_deg: float = 0.0
    d_phi_deg: float = 0.0
    # 0 selects the gain-only form: the field columns carry the far-field
    # amplitude itself rather than the field at a range.
    range_m: float = 0.0


@dataclass(frozen=True)
class NearFieldRequest:
    """A rectangular grid of near-field samples, X varying fastest."""

    magnetic: bool = False
    counts: tuple[int, int, int] = (1, 1, 1)
    origin: Point = (0.0, 0.0, 0.0)
    step: Point = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PrintControl:
    """Which element-current rows a run reports.

    ``elements`` is ``None`` for "no restriction" and otherwise a tuple of
    ``(wire, element)`` pairs, 0-based within the wire — the model's own
    addressing, not the dialect's.
    """

    suppressed: bool = False
    elements: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True)
class ExecuteGroup:
    """One execute card's worth of state.

    A group is a voltage vector over :attr:`DeckModel.feeds`, a frequency
    list, and what it asks to be reported.  ``voltages`` is parallel to
    ``feeds``: a port undriven in this group carries 0, which with a zero load
    collapses to a shorted gap — present in the matrix, invisible to the
    physics (spec ``#one-geometry-one-port-set``).
    """

    frequencies: tuple[float, ...] = ()
    voltages: tuple[complex, ...] = ()
    request: FarFieldRequest | NearFieldRequest | None = None
    print_control: PrintControl | None = None
    extended_kernel: bool = False
    # Advisory only, recorded so a consumer can echo it in the right block:
    # (processor count, block size).
    multiprocessing: tuple[int, int] | None = None
    # True when the operator this group is answered from had to be rebuilt —
    # a fresh frequency list, or the first group of the deck.
    refilled: bool = True
    # A kernel change between two execute cards rebuilds the operator without
    # a new frequency list.
    refilled_partial: bool = False


@dataclass(frozen=True)
class DeckModel:
    """A parsed deck, in momwire's vocabulary.

    See the spec's ``#the-deckmodel`` for the normative field list.  Two of
    that table's rows are exposed here as derived properties rather than
    stored fields, because both are stated there as *per execute group*
    quantities: :attr:`frequencies` and :attr:`requests`.
    """

    # Ordered conductors.  Feed/load/gap indices address this tuple.
    wires: tuple[DeckWire, ...] = ()

    # (wire, arclength, volts) — a delta gap at a point on a wire.  The union
    # of every group's drive points, in discovery order; a group's own drive
    # is its ExecuteGroup.voltages vector over this list.
    feeds: tuple[tuple[int, float, complex], ...] = ()

    # (wire, vertex, volts) — a gap at a polyline KNOT rather than mid-span.
    # The nec2 dialect emits none; the seam exists because a NEC-5 dialect's
    # edge sources are exactly this.
    node_gaps: tuple[tuple[int, int, complex], ...] = ()

    # (wire, arclength, spec) — stamped at the same kind of point a feed is.
    loads: tuple[tuple[int, float, LoadSpec], ...] = ()

    # None (free space), "pec", or (model, eps_r, sigma) with model one of
    # "finite-fast" / "finite" — momwire's own spellings.
    ground: None | str | tuple[str, float, float] = None
    ground_z: float = 0.0
    # The second medium of a cliff, read only by a far-field request's modes
    # 2 and 3.
    second_medium: SecondMedium | None = None
    # Whether the deck declared that it sits over a ground PLANE.  Not the
    # ground's physics — that is `ground` alone — but a structure-report
    # annotation the dialect records as written.
    ground_plane_flag: bool = False

    # One entry per execute card, in deck order; None marks an execute card
    # that ran nothing (nothing had changed since the previous one).
    groups: tuple[ExecuteGroup | None, ...] = ()

    # The deck's free text, in card order, and the two directives a parser
    # consumes out of it rather than passing through.
    comments: tuple[str, ...] = ()
    quiet: bool = False
    reduced_field: int | None = None

    # Dialect front-ends that have not yet grown a card's semantics record
    # the card mnemonics they deferred here.  Empty in a finished dialect;
    # it exists so an unfinished one is honest rather than silent.
    deferred: tuple[str, ...] = field(default=())

    @property
    def frequencies(self) -> tuple[tuple[float, ...], ...]:
        """The frequency list, in MHz, per execute group (None for a group
        that ran nothing)."""
        return tuple(g.frequencies if g is not None else () for g in self.groups)

    @property
    def requests(self) -> tuple[FarFieldRequest | NearFieldRequest | None, ...]:
        """What each execute group asks for: None is a plain solve."""
        return tuple(g.request if g is not None else None for g in self.groups)
