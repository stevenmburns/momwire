"""The ``nec2`` dialect's geometry front-end.

Normative spec: ``site/src/content/docs/reference/deck-grammar-nec2.md``
("The nec2 deck dialect"), sections ``#gw--straight-wire``, ``#gm--move-and-
replicate``, ``#gx--structure-reflection``,
``#gr--cylindrical-structure-rotation``, ``#gs--scale``,
``#ge--end-of-geometry``, ``#connections`` and ``#addressing``.

This module is the ONLY place in ``momwire.deck`` where NEC's ``(tag,
segment)`` vocabulary exists.  It turns a deck's geometry cards into

  1. a flat list of straight NEC wires, after every ``GM``/``GS``/``GX``
     transform and after the two connection passes
     (:func:`snap_connections`);
  2. that same list shattered at the interior boundaries other wires' ends
     land on, so every electrical connection in the structure is a shared
     wire END (:func:`junction_cuts`, :func:`shatter`); and
  3. the resolver that answers ``(tag, segment) -> (wire, arclength)`` for
     every ``EX``/``LD``/``IS``/``PT`` address in the deck.

Step 2 is what makes step 3's "wire" a :class:`~momwire.deck.model.DeckWire`
index rather than a card ordinal, and it is why the model never has to carry
a junction table: after it, two conductors are connected exactly when they
share an endpoint coordinate.

The transform, tolerance and snapping arithmetic here is transcribed from
nec2c's ``geometry.c`` (``wire``/``move``/``conect``) by way of antennaknobs'
``nec_import``, which the spec's appendix records as shared code rather than
an independent re-derivation — the two readers must agree bitwise on
geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ._cards import Card, DeckError

__all__ = [
    "FlatWire",
    "Piece",
    "Symmetry",
    "Nec2Structure",
    "build_geometry",
    "snap_connections",
    "junction_cuts",
    "shatter",
]

_DEG = math.pi / 180.0

# NEC's connection tolerance: an L1 (Manhattan) separation no greater than
# this times the connecting segment's length.  nec2c's SMIN.
_SMIN = 1e-3

# Quantization for "these two points are the same node" when detecting which
# interior boundaries another wire touches.  1e-9 m is nine orders below the
# 1e-3-of-a-segment tolerance the snapping pass has already applied, so it
# separates genuinely distinct nodes and fuses the ones snapping made equal.
_NODE_EPS = 1e-9


@dataclass
class FlatWire:
    """One straight NEC wire after every geometry transform: a ``GW`` row.

    Mutable because ``GM``/``GS``/snapping rewrite coordinates in place, which
    is how NEC's own geometry buffer works and what keeps the transform order
    faithful.
    """

    tag: int
    n_seg: int
    p1: list[float]
    p2: list[float]
    radius: float

    @property
    def length(self) -> float:
        return math.dist(self.p1, self.p2)


@dataclass(frozen=True)
class Piece:
    """One shattered span of a :class:`FlatWire` — a model wire in waiting.

    ``first_seg`` is how many of the parent wire's segments precede this
    piece, so the parent's 1-based local segment ``s`` lives in the piece with
    ``first_seg < s <= first_seg + n_seg``.
    """

    wire: int
    first_seg: int
    n_seg: int
    p1: tuple[float, float, float]
    p2: tuple[float, float, float]

    @property
    def length(self) -> float:
        return math.dist(self.p1, self.p2)


@dataclass(frozen=True)
class Symmetry:
    """The symmetric cell a ``GX``/``GR`` declared, while it is still live.

    Spec ``#gx--structure-reflection`` and
    ``#gr--cylindrical-structure-rotation``.  NEC does not merely replicate a
    reflected or rotated structure: it declares the structure SYMMETRIC and
    fills and factors the matrix on one cell, which is what its ``SEGMENTS IN
    A SYMMETRIC CELL`` line reports (nec2c's ``NP``, set by ``reflc`` — the
    one routine behind both cards).

    The layout is flat and contiguous, with no hierarchy: the cell is the
    structure exactly as it stood when the card fired, and its images follow
    it, so the cell is a PREFIX of :attr:`Nec2Structure.wires` — the first
    ``cell_wires`` of them, carrying ``cell_segments`` segments between them.
    A second ``GX``/``GR`` resets the cell to the whole prior structure
    rather than nesting inside the first one's.

    The cell is recorded as a prefix LENGTH rather than a wire list because
    the prefix survives everything that happens to the list afterwards:
    ``GM``/``GS`` rewrite coordinates in place, snapping rewrites endpoints,
    and :func:`shatter` indexes pieces back to their parent wire — none of
    which moves a wire's position in the list.
    """

    cell_wires: int
    cell_segments: int


# ---------------------------------------------------------------------------
# geometry cards
# ---------------------------------------------------------------------------


def _gw(card: Card, wires: list[FlatWire]) -> None:
    """``GW`` — one straight wire divided into equal segments (spec
    ``#gw--straight-wire``).  Nothing on the card is ignored."""
    tag, n_seg = card.i(0), card.i(1)
    if n_seg < 1:
        raise DeckError(f"segment count must be >= 1, got {n_seg}")
    radius = card.f(8)
    if radius <= 0.0:
        # NEC spells a TAPERED wire by leaving the radius field clear and
        # continuing the wire on a GC card.  Reading it as a wire of zero
        # radius would silently model something the deck did not describe.
        raise DeckError(
            "GW with a non-positive radius announces a tapered wire "
            "(GW + GC continuation), which is not part of this engine's "
            "nec2 dialect"
        )
    wires.append(
        FlatWire(
            tag,
            n_seg,
            [card.f(2), card.f(3), card.f(4)],
            [card.f(5), card.f(6), card.f(7)],
            radius,
        )
    )


def _first_wire_with_tag(wires: list[FlatWire], tag: int) -> int:
    if tag <= 0:
        return 0
    for i, w in enumerate(wires):
        if w.tag == tag:
            return i
    raise DeckError(f"no wire has tag {tag}")


def _gm(card: Card, wires: list[FlatWire]) -> None:
    """``GM`` — rotate about X then Y then Z, translate, optionally replicate
    (spec ``#gm--move-and-replicate``).

    ``ITS`` is read as a real and rounded, which is NEC's own reading of the
    field; the block is every wire from the first one carrying that tag to the
    end of the list.
    """
    itgi, nrpt = card.i(0), card.i(1)
    rox, roy, roz = card.f(2) * _DEG, card.f(3) * _DEG, card.f(4) * _DEG
    xs, ys, zs = card.f(5), card.f(6), card.f(7)
    its = int(card.f(8) + 0.5)

    sps, cps = math.sin(rox), math.cos(rox)
    sth, cth = math.sin(roy), math.cos(roy)
    sph, cph = math.sin(roz), math.cos(roz)
    m = (
        (cph * cth, cph * sth * sps - sph * cps, cph * sth * cps + sph * sps),
        (sph * cth, sph * sth * sps + cph * cps, sph * sth * cps - cph * sps),
        (-sth, cth * sps, cth * cps),
    )

    def xf(p):
        return [
            p[0] * m[0][0] + p[1] * m[0][1] + p[2] * m[0][2] + xs,
            p[0] * m[1][0] + p[1] * m[1][1] + p[2] * m[1][2] + ys,
            p[0] * m[2][0] + p[1] * m[2][1] + p[2] * m[2][2] + zs,
        ]

    i1 = _first_wire_with_tag(wires, its)
    if nrpt == 0:
        for w in wires[i1:]:
            w.p1, w.p2 = xf(w.p1), xf(w.p2)
        return
    block = wires[i1:]
    for _ in range(nrpt):
        # Each repetition transforms the PREVIOUS copy, so rotations and
        # translations compound — that is how one loop side replicates into a
        # polygon, or one bay into a stack.  A wire tagged 0 keeps tag 0.
        block = [
            FlatWire(
                w.tag + itgi if w.tag != 0 else 0,
                w.n_seg,
                xf(w.p1),
                xf(w.p2),
                w.radius,
            )
            for w in block
        ]
        wires.extend(block)


def _gs(card: Card, wires: list[FlatWire]) -> None:
    """``GS`` — multiply every coordinate and radius by a factor (spec
    ``#gs--scale``).  The tag-range form is xnec2c's extension, honoured
    here; segment counts are unchanged."""
    lo, hi, factor = card.i(0), card.i(1), card.f(2)
    if factor <= 0.0:
        raise DeckError(f"scale factor must be > 0, got {factor}")
    ranged = lo > 0 and hi >= lo
    for w in wires:
        if ranged and not (lo <= w.tag <= hi):
            continue
        w.p1 = [c * factor for c in w.p1]
        w.p2 = [c * factor for c in w.p2]
        w.radius *= factor


def _planes(card: Card) -> tuple[tuple[int, int, str], ...]:
    """``GX``'s ``I2`` decoded into ``(axis, flag, plane name)``, in nec2c's
    ``reflc`` firing order: Z=0 first, then Y=0, then X=0."""
    code = card.i(1)
    ix, iy, iz = (code // 100) % 10, (code // 10) % 10, code % 10
    return ((2, iz, "Z=0"), (1, iy, "Y=0"), (0, ix, "X=0"))


def _reflect(wires: list[FlatWire], axis: int, iti: int, plane: str) -> None:
    """One reflection: copy every wire so far with one coordinate negated.

    The plane test is nec2c's (``reflc``), by way of antennaknobs'
    ``nec_import._reflect``: a wire whose two ends both sit on the plane LIES
    in it, and a wire whose two ends straddle it CROSSES it.  Either is a
    geometry error, because the image would land on top of the original or
    inside it.  Touching the plane at ONE end is legal and common — that is
    how a symmetric loop or a V is built.

    Reading it per WIRE is the reference importer's form, corpus-validated;
    nec2c reads the same expression per SEGMENT, which differs only for a
    wire crossing the plane exactly at a segment boundary (nec2c accepts it
    and then duplicates the wire onto itself) and for a wire whose SEGMENTS
    are shorter than the 1e-5 absolute tolerance.  The two readers must agree
    on geometry, so this follows the reference.
    """
    images: list[FlatWire] = []
    for w in wires:
        e1, e2 = w.p1[axis], w.p2[axis]
        if abs(e1) + abs(e2) <= 1.0e-5 or e1 * e2 < -1.0e-6:
            raise DeckError(f"a wire lies in or crosses the {plane} symmetry plane")
        q1, q2 = list(w.p1), list(w.p2)
        q1[axis], q2[axis] = -e1, -e2
        images.append(
            FlatWire(w.tag + iti if w.tag != 0 else 0, w.n_seg, q1, q2, w.radius)
        )
    wires.extend(images)


def _gx(card: Card, wires: list[FlatWire]) -> None:
    """``GX`` — reflect the whole structure so far in one, two or three
    coordinate planes (spec ``#gx--structure-reflection``).

    Transcribed from nec2c's ``reflc`` by way of antennaknobs'
    ``nec_import._gx``.  Each reflection that fires copies the ENTIRE
    structure defined so far, and the tag increment DOUBLES afterwards, so
    every image keeps a tag of its own: with ``I1 = 1`` and all three planes
    selected, one wire tagged 1 becomes eight wires tagged 1 through 8.  A
    wire tagged 0 stays tagged 0 in every image.
    """
    iti = card.i(0)
    for axis, flag, plane in _planes(card):
        if flag:
            _reflect(wires, axis, iti, plane)
            iti *= 2


def _gr(card: Card, wires: list[FlatWire]) -> None:
    """``GR`` — rotate the whole structure so far about Z, ``nop - 1`` times,
    to form a cylindrical structure of ``nop`` copies (spec
    ``#gr--cylindrical-structure-rotation``).

    Transcribed from nec2c's ``reflc`` by way of antennaknobs' ``nec_import.
    _gr``.  ``GR``'s own reader calls the same ``reflc`` that ``GX`` does,
    with ``ix = -1``, which sends it down ``reflc``'s ROTATION branch instead
    of the reflection one.  Unlike ``GX`` the tag increment does not double:
    each copy is built by rotating the PREVIOUS copy's wires (not the
    original structure) and adding ``I1`` to the previous copy's tags, so the
    increment accumulates — the ``k``-th copy (1-based, ``k = 1`` is the
    first image) carries tag ``original + k * I1``. A wire tagged 0 stays
    tagged 0 in every copy.
    """
    itg, nop = card.i(0), card.i(1)
    if nop < 1:
        raise DeckError(f"structure count must be >= 1, got {nop}")
    sam = 2.0 * math.pi / nop
    cs, ss = math.cos(sam), math.sin(sam)

    def rot(p):
        return [p[0] * cs - p[1] * ss, p[0] * ss + p[1] * cs, p[2]]

    block = wires[:]
    for _ in range(nop - 1):
        # Each repetition rotates the PREVIOUS copy, exactly as `_gm`'s
        # replicating form compounds a repeated transform — that is how one
        # bay becomes a tower's worth of copies.
        block = [
            FlatWire(
                w.tag + itg if w.tag != 0 else 0,
                w.n_seg,
                rot(w.p1),
                rot(w.p2),
                w.radius,
            )
            for w in block
        ]
        wires.extend(block)


_GEOMETRY_BUILDERS = {"GW": _gw, "GM": _gm, "GS": _gs, "GX": _gx, "GR": _gr}


def _symmetry_after(
    card: Card,
    wires: list[FlatWire],
    before: tuple[int, int],
    symmetry: Symmetry | None,
) -> Symmetry | None:
    """The symmetry state after ``card`` ran; ``before`` is the ``(wires,
    segments)`` extent the structure had before it (spec
    ``#gx--structure-reflection``, "The symmetric cell", and
    ``#gr--cylindrical-structure-rotation``).

    Only ``GX`` and ``GR`` ever CREATE symmetry.  What the other geometry
    cards do to it was measured on nec2c (issue #415) and is confirmed in its
    source — the state is ``NP`` (the cell) and ``IPSYM`` (the flag):

    | card after ``GX``/``GR``    | symmetry  | nec2c                     |
    |-----------------------------|-----------|---------------------------|
    | ``GS`` (any)                | kept      | ``scale`` touches neither |
    | ``GM`` whole-structure      | kept      | ``move`` returns early    |
    | ``GM`` selective / ``NRPT`` | destroyed | ``move``: ``NP = N``      |
    | ``GW``                      | destroyed | ``wire``: ``NP = N``      |
    | ``GX``/``GR``               | reset     | ``reflc``: ``NP = N``     |

    Symmetry survives a congruence of the ENTIRE structure and dies the
    moment anything is added (``GW``, a replicating ``GM``) or transformed
    selectively (a ``GM`` whose block does not start at the first wire).

    This is deliberately the conservative half of nec2c's state machine: it
    never reports symmetry DEAD where nec2c keeps it alive.  Two of nec2c's
    further rules only ever shrink a cell — ``conect`` doubles ``NP`` when a
    ground plane meets a Z=0 reflection, and a ``GM`` rotating about X or Y
    scales ``IPSYM`` out of range — so leaving them out can only over-report
    liveness, which costs a refusal rather than a wrong answer.
    """
    mnemonic = card.mnemonic
    if mnemonic == "GX":
        if not any(flag for _, flag, _ in _planes(card)):
            # `reflc` sets NP = N and IPSYM = 0 BEFORE it looks at the plane
            # code, so a GX that selects no plane is not a no-op: it retires
            # whatever symmetry was in force.
            return None
        return Symmetry(*before)
    if mnemonic == "GR":
        # `GR`'s own card reader calls `reflc` with `ix = -1`, which always
        # takes the ROTATION branch and always sets `IPSYM = -1` with
        # `NP` = the pre-card segment count — there is no GX-shaped "selects
        # nothing" case to special-case here, because a `GR` card carries no
        # analogous "off" state: `nop < 1` already refused in `_gr`, so every
        # card that reaches this point is a `nop >= 1` rotation.  This
        # includes the degenerate `nop = 1`: nec2c's source sets IPSYM = -1
        # unconditionally before it ever multiplies `N` by `nop`, so a `GR n
        # 1` declares symmetry with a cell equal to the whole (unchanged)
        # structure — resolved deliberately here rather than left to fall
        # through to the GM branch below.
        return Symmetry(*before)
    if mnemonic == "GS":
        return symmetry
    if mnemonic == "GW":
        return None
    # GM.  `move` preserves only when it neither replicates nor restricts:
    # `if (nrpt == 0 && ix == 1) return;`, where `ix` is the first segment of
    # the affected block.  The block start is re-resolved here rather than
    # threaded out of `_gm` because it does not move: a replicating GM
    # APPENDS its copies, so the first wire carrying ITS keeps its index.
    if card.i(1) != 0:
        return None
    if _first_wire_with_tag(wires, int(card.f(8) + 0.5)) != 0:
        return None
    return symmetry


# ---------------------------------------------------------------------------
# connections
# ---------------------------------------------------------------------------


def snap_connections(wires: list[FlatWire]) -> None:
    """Rewrite coincident wire ends onto exact shared coordinates, in place.

    Spec ``#connections``.  NEC connects SEGMENTS whose ends coincide within
    ``1e-3`` of a segment length (an L1 separation); the grouping of segments
    into ``GW`` wires is irrelevant to it.  For a pair of different lengths we
    take the MINIMUM of the two — the conservative intersection, which never
    connects what a NEC build would leave open.

    Two passes, both moving an endpoint by at most 0.1 % of one segment:

      1. endpoint clusters are unioned and rewritten to one exact coordinate;
      2. a remaining endpoint within tolerance of another wire's interior
         segment boundary is snapped onto it, computed with the same
         ``a + (b - a) * k/NS`` expression :func:`shatter` uses, so the touch
         is later detected as a cut.

    Without this a deck whose corners sit a micrometre apart — the routine
    outcome of ``GM``-rotating six-significant-digit card coordinates — would
    solve as a broken electrical graph, current pinned to zero at every
    unmatched end.
    """

    def l1(a, b):
        return sum(abs(x - y) for x, y in zip(a, b))

    end_seg_len = []  # per endpoint (2 per wire): its wire's segment length
    pts = []  # coordinate tuple per endpoint, index = 2*wire + (0|1)
    for w in wires:
        seg = w.length / w.n_seg
        for p in (w.p1, w.p2):
            end_seg_len.append(seg)
            pts.append(tuple(float(c) for c in p))

    if not pts:
        return
    max_tol = _SMIN * max(end_seg_len)
    if max_tol <= 0.0:
        return

    def cell(p):
        return tuple(math.floor(c / max_tol) for c in p)

    def neighborhood(grid, p):
        cx, cy, cz = cell(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    yield from grid.get((cx + dx, cy + dy, cz + dz), ())

    # Pass 1: endpoint clusters, union-find under the per-pair tolerance.
    grid: dict = {}
    for k, p in enumerate(pts):
        grid.setdefault(cell(p), []).append(k)
    parent = list(range(len(pts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, p in enumerate(pts):
        for j in neighborhood(grid, p):
            if j <= k:
                continue
            if l1(p, pts[j]) <= _SMIN * min(end_seg_len[k], end_seg_len[j]):
                parent[find(j)] = find(k)

    for k in range(len(pts)):
        root = find(k)
        if root != k:
            wi, end = divmod(k, 2)
            coords = list(pts[root])
            if end == 0:
                wires[wi].p1 = coords
            else:
                wires[wi].p2 = coords
            pts[k] = pts[root]

    # Pass 2: endpoints onto OTHER wires' interior segment boundaries,
    # recomputed after pass 1 so moved ends shift their boundaries too.
    bgrid: dict = {}
    boundaries: list[tuple[int, tuple]] = []
    for wi, w in enumerate(wires):
        for k in range(1, w.n_seg):
            t = k / w.n_seg
            b = tuple(a + (b_ - a) * t for a, b_ in zip(w.p1, w.p2))
            bgrid.setdefault(cell(b), []).append(len(boundaries))
            boundaries.append((wi, b))

    for k, p in enumerate(pts):
        wi, end = divmod(k, 2)
        best = None
        for j in neighborhood(bgrid, p):
            bw, b = boundaries[j]
            if bw == wi or b == p:
                continue
            d = l1(p, b)
            seg_b = end_seg_len[2 * bw]
            if d <= _SMIN * min(end_seg_len[k], seg_b) and (
                best is None or d < best[0]
            ):
                best = (d, b)
        if best is not None:
            coords = list(best[1])
            if end == 0:
                wires[wi].p1 = coords
            else:
                wires[wi].p2 = coords


def _boundary(w: FlatWire, k: int) -> tuple[float, float, float]:
    """The point after ``k`` of the wire's segments.

    One expression, used everywhere a boundary is needed, so adjoining pieces
    land on bitwise-equal coordinates and are recognised as connected.  Note
    ``k == n_seg`` evaluates ``a + (b - a) * 1.0`` rather than returning ``b``:
    that is the reference implementation's arithmetic, and the two readers
    must agree bitwise (the residue is sub-ulp, far below the 1e-9 node
    quantization below).
    """
    t = k / w.n_seg
    return tuple(a + (b - a) * t for a, b in zip(w.p1, w.p2))  # type: ignore[return-value]


def junction_cuts(wires: list[FlatWire]) -> dict[int, frozenset[int]]:
    """wire index -> interior segment boundaries some OTHER wire ends on.

    Spec ``#connections``, second pass.  NEC connects segments regardless of
    the ``GW`` grouping, so a deck may run one long wire straight through
    another and rely on the crossing carrying current.  momwire junctions
    wires at their ENDS, so those crossings have to become ends — which is
    lossless: same segments, same boundaries, and the KCL at the shared node
    is exactly NEC's connection.
    """

    def key(p):
        return tuple(round(c / _NODE_EPS) for c in p)

    owners: dict[tuple, set[int]] = {}
    for i, w in enumerate(wires):
        for k in range(w.n_seg + 1):
            owners.setdefault(key(_boundary(w, k)), set()).add(i)
    cuts: dict[int, frozenset[int]] = {}
    for i, w in enumerate(wires):
        shared = {k for k in range(1, w.n_seg) if len(owners[key(_boundary(w, k))]) > 1}
        if shared:
            cuts[i] = frozenset(shared)
    return cuts


def shatter(
    wires: list[FlatWire], cuts: dict[int, frozenset[int]]
) -> tuple[list[Piece], list[list[int]]]:
    """Split each wire at its junction cuts.

    Returns the flat piece list and, per wire, the indices of its pieces in
    order.  A wire with no cuts yields one piece carrying its own endpoints
    verbatim — no arithmetic, so an uncut wire's coordinates are exactly the
    card's (after transforms and snapping).
    """
    pieces: list[Piece] = []
    per_wire: list[list[int]] = []
    for i, w in enumerate(wires):
        cutset = cuts.get(i)
        indices: list[int] = []
        if not cutset:
            indices.append(len(pieces))
            pieces.append(
                Piece(i, 0, w.n_seg, tuple(w.p1), tuple(w.p2))  # type: ignore[arg-type]
            )
        else:
            prev = 0
            for b in [*sorted(cutset), w.n_seg]:
                indices.append(len(pieces))
                pieces.append(
                    Piece(i, prev, b - prev, _boundary(w, prev), _boundary(w, b))
                )
                prev = b
        per_wire.append(indices)
    return pieces, per_wire


# ---------------------------------------------------------------------------
# the structure
# ---------------------------------------------------------------------------


@dataclass
class Nec2Structure:
    """A deck's geometry, plus the ``(tag, segment)`` resolver over it."""

    wires: list[FlatWire]
    cuts: dict[int, frozenset[int]] = field(default_factory=dict)
    pieces: list[Piece] = field(default_factory=list)
    pieces_of_wire: list[list[int]] = field(default_factory=list)
    # The GE ground-plane flag.  It records that the structure sits over a
    # plane and drives nothing else; the ground's physics comes from GN.
    ground_plane_flag: bool = False
    # The symmetric cell still in force at the end of the geometry section,
    # or None if no GX declared one or a later card destroyed it.  A live
    # symmetry changes how NEC reads the cards that move the MATRIX (see
    # `Symmetry` and spec ``#gx--structure-reflection``); the geometry
    # itself is fully expanded either way.
    symmetry: Symmetry | None = None

    @property
    def n_segments(self) -> int:
        return sum(w.n_seg for w in self.wires)

    # -- addressing (spec #addressing) --------------------------------------

    def locate(self, tag: int, seg: int) -> tuple[int, int]:
        """``(tag, segment)`` -> ``(flat wire index, 1-based local segment)``.

        A segment number of 0 or below reads as 1.  With a nonzero tag the
        count runs over segments of wires carrying that tag, in card order;
        with ``tag = 0`` it runs over every segment of the structure.
        """
        seg = max(seg, 1)
        acc = 0
        for i, w in enumerate(self.wires):
            if tag != 0 and w.tag != tag:
                continue
            if acc + w.n_seg >= seg:
                return i, seg - acc
            acc += w.n_seg
        raise DeckError(
            f"segment {seg} is out of range for "
            + (f"tag {tag}" if tag else "the structure")
        )

    def element_of(self, wire: int, local: int) -> tuple[int, int]:
        """``(flat wire index, 1-based local segment)`` -> ``(model wire
        index, 0-based element)``.  The part of :meth:`element` that does not
        need a tag, factored out so a resolver that already has a flat wire
        index (:meth:`Nec2Structure.ld_segment_range`) does not have to
        re-walk the tag search in :meth:`locate`."""
        for piece_index in self.pieces_of_wire[wire]:
            piece = self.pieces[piece_index]
            if piece.first_seg < local <= piece.first_seg + piece.n_seg:
                return piece_index, local - piece.first_seg - 1
        raise AssertionError("unreachable: the pieces of a wire cover it")

    def element(self, tag: int, seg: int) -> tuple[int, int]:
        """``(tag, segment)`` -> ``(model wire index, 0-based element)``."""
        wire, local = self.locate(tag, seg)
        return self.element_of(wire, local)

    def resolve_of(self, wire: int, local: int) -> tuple[int, float]:
        """``(flat wire index, 1-based local segment)`` -> ``(model wire
        index, arclength in metres)``.  The part of :meth:`resolve` that does
        not need a tag; see :meth:`element_of`."""
        piece_index, element = self.element_of(wire, local)
        piece = self.pieces[piece_index]
        return piece_index, (element + 0.5) * piece.length / piece.n_seg

    def resolve(self, tag: int, seg: int) -> tuple[int, float]:
        """``(tag, segment)`` -> ``(model wire index, arclength in metres)``.

        A wire's ``NS`` segments divide it into ``NS`` equal parts; local
        segment ``k`` occupies ``[(k-1)L/NS, kL/NS]`` and its CENTRE — the
        point a feed, a load or a port lands on — is ``(k - 1/2)L/NS`` metres
        from the wire's first vertex.
        """
        wire, local = self.locate(tag, seg)
        return self.resolve_of(wire, local)

    def wire_of_segment_range(
        self, tag: int, first: int, last: int
    ) -> tuple[tuple[int, int], ...]:
        """Every ``(model wire, element)`` in an inclusive segment range."""
        if last < first:
            first, last = last, first
        return tuple(self.element(tag, s) for s in range(max(first, 1), last + 1))

    def ld_segment_range(
        self, tag: int, first: int, last: int
    ) -> tuple[tuple[int, int], ...]:
        """``(tag, first, last)`` -> every ``(flat wire index, 1-based local
        segment)`` an ``LD``/``IS`` range covers.

        NEC's own range rule for these two cards — verified against nec2c's
        ``calculations.c: load()`` (oracle probes ``ld_range_probe*.nec``,
        momwire#359 unit C): ``tag = 0, first = 0`` is the WHOLE STRUCTURE;
        ``first = 0`` with a nonzero tag is every segment of that tag, with
        ``last`` ignored entirely (confirmed on the oracle: ``LD 0 1 0 3 ...``
        loads all of tag 1, not segments 1-3); otherwise it is the ordinary
        local segment range ``[first, max(first, last)]``, resolved exactly
        as a single address is (spec ``#addressing``).  This is a different
        rule from ``PT``'s ``wire_of_segment_range`` (a plain min/max swap,
        clamped at 1): PT's all-zero range means "no restriction", while
        LD/IS's means "the whole tag or structure".
        """
        if tag == 0 and first == 0:
            return tuple(
                (i, s) for i, w in enumerate(self.wires) for s in range(1, w.n_seg + 1)
            )
        if first == 0:
            pairs = tuple(
                (i, s)
                for i, w in enumerate(self.wires)
                if w.tag == tag
                for s in range(1, w.n_seg + 1)
            )
            if not pairs:
                raise DeckError(f"no wire has tag {tag}")
            return pairs
        last = max(last, first)
        return tuple(self.locate(tag, s) for s in range(first, last + 1))


def build_geometry(cards: list[Card]) -> Nec2Structure:
    """Run a deck's geometry section and close it.

    ``cards`` is the geometry cards in deck order.  ``GE`` terminates the
    section and carries the ground-plane flag; a deck with no ``GE`` builds as
    though it wrote ``GE 0``.
    """
    wires: list[FlatWire] = []
    ground_plane_flag = False
    symmetry: Symmetry | None = None
    for card in cards:
        if card.mnemonic == "GE":
            # I2 and every real field are ignored, and the flag specifies no
            # ground: it records that the structure sits over a plane.
            ground_plane_flag = card.i(0) != 0
            continue
        before = (len(wires), sum(w.n_seg for w in wires))
        _GEOMETRY_BUILDERS[card.mnemonic](card, wires)
        symmetry = _symmetry_after(card, wires, before, symmetry)

    snap_connections(wires)
    cuts = junction_cuts(wires)
    pieces, pieces_of_wire = shatter(wires, cuts)
    return Nec2Structure(
        wires=wires,
        cuts=cuts,
        pieces=pieces,
        pieces_of_wire=pieces_of_wire,
        ground_plane_flag=ground_plane_flag,
        symmetry=symmetry,
    )
