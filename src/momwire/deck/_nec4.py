"""The ``nec4`` dialect front-end: the decks EZNEC's External NEC-4.2 slot writes.

momwire#1295.  EZNEC Pro+ v7, with its engine option set to External NEC-4.2,
writes a NEC-4.2-dialect deck and launches the engine as
``"<engine>.exe" "EZ.NEC" "NEC.OUT"`` with the engine's own folder as cwd.
:mod:`momwire.eznec` serves that slot when its executable's NAME carries
``nec4`` (:func:`~momwire.deck._solver.dialect_from_program_name`); this module
is the deck reader behind it.

Derived from CAPTURED INPUT ONLY: the seventeen decks EZNEC wrote in the
2026-10-03 capture sitting (antennaknobs ``scratch/eznec-capture/0223``-``0239``,
committed here under ``tests/fixtures/eznec_nec42/``) and the black-box solve
notes taken from them (prose and numbers, no printout).  No NEC-4.2 code,
manual text or printout is described or relied on, and none may be added: its
licence is own use at home.

NEC-4.2 reads NEC-2's cards and addresses a source by SEGMENT, whose centre is
the gap, so the dialect is the ``nec2`` one (:mod:`._nec2`) with the deltas the
captures show, and nothing else:

* ``EX 6`` is a CURRENT source, the card's F1/F2 its set current in amperes.
  Measured, not inferred from the card: on the captures carrying it the solved
  current comes back pinned at the card's 1.414214 A while the voltage floats,
  and 0223/0224 — one antenna, ``EX 6`` against ``EX 0`` — give the same
  impedance to every printed digit.  A sole current source is served as a
  voltage drive of the same port rescaled after the solve
  (:attr:`DeckModel.current_feeds`): the response is linear in the drive, so
  the impedance is the voltage drive's own and only the scale moves.  Several
  sources with any of them a current source — EZNEC's stock Cardioid in this
  slot writes two ``EX 6`` cards — is a PHASED drive and no scale serves it:
  the portal solves for the port voltages that deliver every set current
  while every ``EX 0`` keeps its set voltage (``DeckSolver._phased_drive``, the
  NEC-5 slot's ``EX 4`` algebra on the portal's port space).  One segment is
  still one port for the whole deck, so a segment driven both ways refuses.
* ``GE`` I1 reads -1 / 0 / 1, and -1 is NEC-4's "structure below ground"
  (0239 is a wholly buried wire under ``GE -1,-1``).  The ``nec2`` reading of
  the same sign — a ground plane WITHOUT the contact current expansion — is the
  part that carries over: a free wire end standing in the plane under ``GE -1``
  refuses exactly as it does there, and a buried or crossing structure is
  served by the solver's own below-interface machinery.  I2 is -1 on every
  capture; 0 (the omitted field) is accepted beside it, anything else refuses
  because no capture says what it means.
* ``GN 3`` is the Sommerfeld half-space, the same PHYSICAL ground as ``GN 2``:
  EZNEC offers the two as NEC-4.2's two ground methods, and they are two
  evaluations of one model rather than two models (the licensed engine gave
  0231/0232 the same impedance to every printed digit; antennaknobs measured
  them about an ohm apart on buried designs).  momwire has one evaluation of
  that half-space, so both cards map to ``("finite", eps_r, sigma)``.  The
  card as written still reaches the printout echo; only the solve is shared.
* ``GD`` accompanies ``GN 1`` (0230, 0234, 0235), EZNEC's MININEC-type ground.
  It is read exactly as the ``nec2`` dialect reads it — a second medium that
  only a cliff pattern (``RP 2``/``RP 3``) consults — so the currents are the
  perfect ground's, which is what the licensed engine answered (0230's
  impedance equals 0229's, the same ground with no ``GD``, to every digit).
  Its leading integer (2 on every capture) moves nothing here.
* A ``GN`` card's trailing non-numeric token is a Sommerfeld-table FILE
  name, ``NOFILE`` included, on EVERY ground type (momwire#1317 for GN 2,
  momwire#1443 for the rest).  No EZNEC capture writes one, but SimNEC
  appends ``NOFILE`` to every GN card whatever the engine.  The licensed
  engine, run as a black box (2026-10-12) on a 21-segment 30 MHz dipole at
  10 m with ``GN g 0 0 0 13. .005 0 0 0 0 [NOFILE]`` for g = -1, 0, 1, 2, 3,
  exited 0 on all ten runs, and printed the same input impedance with and
  without ``NOFILE`` to every digit on all five types.  Without a trailer,
  GN 2 and GN 3 reported they could not open their default table file; with
  ``NOFILE`` they report that they could not open THAT one (probes p16 and
  p30, momwire#1463), and compute the table either way.  The name is a cache, not physics,
  so it is accepted and ignored on every type: the solve is the card's
  without it, and the printout echoes the name (``eznec._nec4._ground_file``).
* ``GA`` (wire arc) and ``GH`` are geometry here (momwire#1592, probes
  p46-p53, measured on the licensed binary as a black box), as they are in
  the ``nec2`` dialect since momwire#1651.  ``GA`` is NEC-2's arc, except
  that NEC-4.2 stops on a sweep of 360 degrees, which nec2c reads.  ``GH`` is
  NEC-4.2's own card, which is NOT NEC-2's (the ``nec2`` dialect reads the
  NEC-2 one): a log-spiral or helix read as
  turns, length and two helix and two wire radii (:func:`helix_vertices`
  has the measured rule).  Both are built as one one-segment wire per
  chord, carrying the card's tag, so a ``(tag, segment)`` address counts
  along the arc or helix as along a ``GW``.
* ``EX 4`` refuses with a sentence of its own (``_EX4_REFUSAL``): NEC-4.2
  reads it as a current element at a point, which no NEC-5 user writing
  EX 4 means.
* ``EK`` is read and IGNORED, and every run solves with the extended kernel
  (momwire#1326, Steve's decision 2026-10-04).  NEC-4.2 has one thin-wire
  model and no extended-kernel switch: the binary, given ``EK 0`` as a black
  box (2026-10-04), echoes the card and prints ``THE EK AND KH COMMANDS HAVE
  NO EFFECT IN NEC-4``.  Against momwire's two kernels that model behaves
  as the extended one where it matters (a 28 MHz dipole ladder: at Delta/a
  1.5-2 the extended kernel lands up to 5 ohm nearer NEC-4.2, and on thin
  wire it costs 0.5-2 ohm, inside the model gap either way), so it is this
  dialect's DEFAULT (`DeckModel.extended_kernel_default`), and a basis or a
  deck that cannot take it falls back to the reduced kernel with an
  advisory rather than refusing.  A deck with wires below the ground keeps it
  on its above-ground wires where the basis can (momwire#1170 stage 0).

* ``NE``/``NH`` read coordinate system 1, the SPHERICAL grid, beside the
  rectangular 0 (momwire#1352): EZNEC writes ``NE 1`` when its user enters a
  near-field grid in spherical form.  Measured on the licensed binary as a
  black box (probes p11-p15): the card's three counts, origins and steps are
  ``(R, θ, φ)`` with R in metres and both angles in degrees, θ from the
  zenith and φ from +x; R walks fastest, then θ, then φ; and the table prints
  the Cartesian X, Y, Z of each point under the rectangular grid's own
  headings.  That is the NEC-5 seam's ``NE 1`` exactly (momwire#1257), so the
  two slots place their points with one function.  The nec2 dialect keeps
  refusing it (``_Nec2Parser._near_coordinates``).

* A further ``RP``, ``NE`` or ``NH`` card after a run, with no card between
  that changes the drive or the matrix, prints ONE MORE TABLE of the run it
  follows (momwire#1463).  Measured on the licensed binary as a black box
  (probes p20-p28, 2026-10-13): ``RP`` then ``NE``, ``NE`` then ``RP``,
  ``XQ`` then both, and two ``RP`` cards each print every table, in card
  order, each under its own card's echo, with the input parameters, currents
  and power budget printed once.  A further ``XQ`` there prints nothing.
  The ``nec2`` reading drops every such card (its arming rule, measured on a
  trailing ``XQ``); this dialect keeps the model's group list as the parent
  builds it, one entry per execute card and ``None`` for these, and reads
  each such card for its refusals (``_unarmed_execute``).  The slot's
  printout finds the cards again by position and prints their tables from
  the run already solved (``eznec._nec4``).

Everything else the captures carry is a ``nec2`` card with the ``nec2``
meaning: ``GW``, ``LD 4`` and ``LD 5``, ``FR 0``, ``GN -1``/``1``/``2`` (``GN 1``
in both spellings EZNEC writes), ``TL``/``NT`` addressed by ``(tag, segment)``,
``RP`` (the mode-1001 3D grid and the negative theta step included), ``XQ`` and
``PQ``.  The ``nec2`` refusals stand for every card and form no capture shows.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import MappingProxyType

from ._cards import Card, DeckError
from ._nec2 import _Nec2Parser
from ._nec2_geometry import _GEOMETRY_BUILDERS, FlatWire
from ._nec2_geometry import arc_vertices as _arc_vertices
from .model import DeckModel

__all__ = ["parse_nec4"]

# EX types this dialect drives.  NEC-4's other source types (plane waves, the
# current-slope-discontinuity voltage source) appear on no capture.
_EX_VOLTAGE = 0
_EX_CURRENT = 6

# NEC-4.2's EX 4 is a current ELEMENT, not NEC-5's EX 4 (a current source on
# a segment).  Measured on the licensed NEC-4.2 as a black box (refused/r2,
# momwire#1592): ``EX 4 1 11 0 1.0 0`` on a dipole prints a CURRENT SOURCE
# at the POSITION (1, 0, 0) its first three reals give, at the orientation
# and dipole moment of the next three (0 here, so every current is zero),
# and the card's tag and segment move nothing.  A NEC-5 user writing EX 4
# means NEC-4.2's EX 6, so the sentence says so.
_EX_ELEMENT = 4
_EX4_REFUSAL = (
    "EX 4 is not served by this engine's nec4 dialect: NEC-4.2 reads EX 4 as a "
    "current element at the point its first three real fields give, tied to no "
    "segment, not as NEC-5's EX 4 current source on a segment.  For a current "
    "source on a segment, write EX 6 with that tag and segment and the current "
    "in amperes in its first two real fields (EX 0 drives a voltage source)"
)

# GE I2 values a capture shows (-1), plus the omitted field.
_GE_I2 = frozenset({0, -1})


# --------------------------------------------------------------------------
# GA and GH, as NEC-4.2 reads them (momwire#1592)
# --------------------------------------------------------------------------


def arc_vertices(card: Card) -> list[tuple[float, float, float]]:
    """A ``GA`` card's ``NS + 1`` vertices, as NEC-4.2 reads the card: NEC-2's
    arc (:func:`momwire.deck._nec2_geometry.arc_vertices`), measured on the
    licensed NEC-4.2 as a black box (probes p46-p48: the printed segment
    centres and lengths are these chords' to every digit, for an arc that
    starts below the X axis, one wider than 180 degrees and one whose angle
    falls), with NEC-4.2's own widest sweep: under 360 degrees, where nec2c
    reads 360 itself (momwire#1651)."""
    return _arc_vertices(card, "nec4")


def _ga(card: Card, wires: list[FlatWire]) -> None:
    """``GA``: one one-segment wire per chord of :func:`arc_vertices`, all
    carrying the card's tag, so ``(tag, segment)`` counts along the arc."""
    vertices = arc_vertices(card)
    for a, b in zip(vertices, vertices[1:], strict=False):
        wires.append(FlatWire(card.i(0), 1, list(a), list(b), card.f(5)))


def helix_vertices(card: Card) -> list[tuple[float, float, float]]:
    """A ``GH`` card's ``NS + 1`` vertices, in NEC-4.2's reading of the card.

    NEC-4.2's ``GH`` is NOT NEC-2's.  Its reals are ``TURNS, ZLEN, HR1,
    HR2, WR1, WR2``: the number of turns, the length along Z, the helix
    radius at the two ends and the WIRE radius at the two ends; the
    printout calls it a LOG-SPIRAL OR HELIX and echoes them under those
    names (probes p49-p53, black box).  NEC-2's ``S, HL, A1, B1, A2, B2,
    RAD`` (turn spacing, length, four radii, wire radius) written to it
    reads as something else: its ``S`` as a number of turns, its ``A2`` and
    ``B2`` as wire radii.  F7 has no effect (black box: the same deck with
    0.002 and 0 there printed the same bytes).

    Measured from the printed segmentation (the centre, length and both
    angles of every segment, p49-p53): vertex ``k`` of ``NS`` stands at the
    angle ``2 pi TURNS k / NS`` from +X (counter-clockwise seen from +Z;
    negative turns run clockwise), at the helix radius ``HR1 q**k`` with
    ``q = (HR2/HR1)**(1/NS)``, and at the height ``ZLEN k / NS`` for a
    constant radius or ``ZLEN (1 - q**k) / (1 - q**NS)`` for a tapered one
    (each step's rise in proportion to its radius: the log-spiral, whose
    segments keep one pitch angle).  A negative ``ZLEN`` runs down from
    Z = 0, a zero one is a flat spiral.  Each segment is the chord between
    two vertices.
    """
    n_seg = card.i(1)
    turns, zlen, r1, r2 = card.f(2), card.f(3), card.f(4), card.f(5)
    if n_seg < 1:
        raise DeckError(f"GH segment count must be >= 1, got {n_seg}")
    if turns == 0.0:
        raise DeckError(
            "GH with zero turns is not served by this engine's nec4 dialect, "
            "which reads NEC-4.2's GH: F1 is the number of turns, and no "
            "NEC-4.2 run shows a helix of none"
        )
    if r1 <= 0.0 or r2 <= 0.0:
        raise DeckError(
            f"GH needs a positive helix radius at both ends (F3, F4), got "
            f"{r1:g} and {r2:g}"
        )
    if card.f(6) <= 0.0 or card.f(7) <= 0.0:
        # NEC-4.2 stops on it with a segment data error (black box).  A
        # NEC-2 GH, whose wire radius is F7, lands here with its B1/B2.
        raise DeckError(
            f"GH needs a positive wire radius at both ends (F5, F6), got "
            f"{card.f(6):g} and {card.f(7):g}; this engine's nec4 dialect reads "
            f"NEC-4.2's GH (turns, length, two helix radii, two wire radii), "
            f"not NEC-2's"
        )
    q = (r2 / r1) ** (1.0 / n_seg)
    step = 2.0 * math.pi * turns / n_seg
    out = []
    for k in range(n_seg + 1):
        rho = r1 * q**k
        # `q == 1.0` rather than `r1 == r2`: two radii a rounding apart give
        # a q of exactly 1, whose log-spiral form would divide by zero.
        z = zlen * k / n_seg if q == 1.0 else zlen * (1.0 - q**k) / (1.0 - q**n_seg)
        out.append((rho * math.cos(k * step), rho * math.sin(k * step), z))
    return out


def helix_wire_radii(card: Card) -> list[float]:
    """The wire radius of each ``GH`` segment: ``WR1`` on the first and
    ``WR2`` on the last, in geometric steps between (p50, p52)."""
    n_seg = card.i(1)
    w1, w2 = card.f(6), card.f(7)
    if n_seg == 1:
        return [w1]
    return [w1 * (w2 / w1) ** (k / (n_seg - 1)) for k in range(n_seg)]


def _gh(card: Card, wires: list[FlatWire]) -> None:
    """``GH``: one one-segment wire per chord of :func:`helix_vertices`,
    each at its own wire radius, all carrying the card's tag."""
    vertices = helix_vertices(card)
    for a, b, radius in zip(
        vertices, vertices[1:], helix_wire_radii(card), strict=False
    ):
        wires.append(FlatWire(card.i(0), 1, list(a), list(b), radius))


# ---------------------------------------------------------------------------
# CW: NEC-4 and NEC-5's catenary wire (momwire#1697)
# ---------------------------------------------------------------------------
# ``CW ITG NS X1 Y1 Z1 X2 Y2 Z2 RAD ICAT RHM ZM``. The wire hangs between its
# two ends under gravity along -z. With u the horizontal distance from end 1
# (along the ends' separation projected onto the xy plane) and k = 1/a the
# curvature at the vertex,
#
#     z(u) = z1 + (cosh(k(u - u0)) - cosh(k u0)) / k,
#
# k > 0 for a wire that sags below the straight line between its ends and
# k < 0 for one that arches above it. ICAT says which number fixes k:
#
# * 1: the wire passes through the point at horizontal distance RHM from end
#   1 at height ZM (absolute z, metres);
# * 2: at horizontal distance RHM from end 1 it hangs ZM below the straight
#   line between the ends (a negative ZM arches above it);
# * 3: its arc length is RHM (ZM unused); no longer than the straight line,
#   and the wire is that straight line.
#
# A sag is cut into NS pieces of equal ARC LENGTH, an arch into NS pieces of
# equal HORIZONTAL step: what both licensed engines build, read off their own
# segment tables as black boxes (antennaknobs#2021, 20 synthetic decks), not
# what the NEC-5 User's Guide says (equal arc length always).
#
# The same construction as antennaknobs' importer (`nec_import._cw`), so the
# app and the launchers cannot disagree on the curve. Everything is written
# in forms that stay accurate as k -> 0 (a nearly straight wire): z and the
# arc length are products of sinh/cosh terms whose arguments are O(1), never
# differences of two large cosh values.

_CW_KD_MAX = 600.0  # |k|·D bound: sinh overflows past ~710


def _sinhc(x: float) -> float:
    """sinh(x)/x, 1 at 0."""
    return math.sinh(x) / x if x != 0.0 else 1.0


def _cw_shape(k: float, d: float, h: float):
    """``(z(u), s(u))``: height above end 1 and arc length from end 1 of the
    catenary of curvature ``k`` through (0, 0) and (``d``, ``h``)."""
    half = 0.5 * k * d
    # asinh(k h / (2 sinh(k d / 2))), the vertex offset, as an O(1) number.
    c = math.asinh(h / (d * _sinhc(half)))

    def z(u: float) -> float:
        # 2 sinh(k(u - 2 u0)/2) sinh(k u/2) / k
        return u * math.sinh(0.5 * k * u - half + c) * _sinhc(0.5 * k * u)

    def s(u: float) -> float:
        # 2 cosh(k(u - 2 u0)/2) sinh(k u/2) / k
        return u * math.cosh(0.5 * k * u - half + c) * _sinhc(0.5 * k * u)

    return z, s


def _bisect(f, lo: float, hi: float) -> float:
    """The root of ``f`` between ``lo`` and ``hi`` (a sign change)."""
    f_lo = f(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        f_mid = f(mid)
        if (f_mid > 0.0) == (f_lo > 0.0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _cw_solve_k(g, d: float, signs) -> float | None:
    """The curvature k where ``g(k)`` crosses zero, searched outward from a
    straight wire on each side in ``signs``; None when it never does."""
    for sign in signs:
        prev_k, prev_g = None, None
        for j in range(-400, 1):
            k = sign * (_CW_KD_MAX / d) * 10.0 ** (j / 25.0)
            try:
                gk = g(k)
            except OverflowError:
                continue
            if gk != gk:  # NaN
                continue
            if gk == 0.0:
                return k
            if prev_g is not None and (gk > 0.0) != (prev_g > 0.0):
                return _bisect(g, prev_k, k)
            prev_k, prev_g = k, gk
    return None


def _catenary(card: Card) -> tuple[list[tuple[float, float, float]], float]:
    """``(vertices, arc length)`` of a ``CW`` card: its ``NS + 1`` vertices
    (see the block above), each segment the chord between two of them, and
    the length of the curve itself. A wire no longer than the straight line
    between its ends (ICAT 3) is that line, cut evenly."""
    n_seg = card.i(1)
    p1 = (card.f(2), card.f(3), card.f(4))
    p2 = (card.f(5), card.f(6), card.f(7))
    radius, icat, rhm, zm = card.f(8), card.f(9), card.f(10), card.f(11)
    if n_seg < 1:
        raise DeckError(f"CW segment count must be >= 1, got {n_seg}")
    if radius <= 0.0:
        raise DeckError(f"CW needs a positive wire radius (F7), got {radius:g}")
    if icat not in (1.0, 2.0, 3.0):
        raise DeckError(
            f"CW ICAT must be 1 (a point the wire passes through), 2 (the sag "
            f"below the straight line at a distance) or 3 (the wire's "
            f"length), got {icat:g}"
        )
    dx, dy, h = p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]
    d = math.hypot(dx, dy)
    if d == 0.0:
        raise DeckError(
            "CW's two ends are one above the other, so there is no horizontal "
            "span for the wire to hang across"
        )
    ex, ey = dx / d, dy / d

    if icat == 3.0:
        if rhm <= math.hypot(d, h):
            # No longer than the straight line: both engines build that line.
            line = [
                tuple(a + (b - a) * i / n_seg for a, b in zip(p1, p2, strict=True))
                for i in range(n_seg + 1)
            ]
            return line, math.dist(p1, p2)
        signs: tuple[float, ...] = (1.0,)

        def g(k):
            return _cw_shape(k, d, h)[1](d) - rhm

    else:
        if rhm == 0.0 or rhm == d:
            raise DeckError(
                f"CW with RHM = {rhm:g} puts the point at an end of the wire, "
                "which fixes nothing about its shape"
            )
        drop = zm if icat == 2.0 else p1[2] + h * rhm / d - zm
        if drop == 0.0:
            raise DeckError(
                "CW's point lies on the straight line between the ends, so no "
                "catenary passes through it"
            )
        # Inside the span a sag (k > 0) drops below the line and an arch
        # rises above it; outside the span the two swap, so try both.
        inside = 0.0 < rhm < d
        signs = (
            (math.copysign(1.0, drop),)
            if inside
            else (-math.copysign(1.0, drop), math.copysign(1.0, drop))
        )

        def g(k):
            return h * rhm / d - _cw_shape(k, d, h)[0](rhm) - drop

    k = _cw_solve_k(g, d, signs)
    if k is None:
        raise DeckError(
            f"no catenary between CW's ends fits ICAT {icat:g} with "
            f"RHM = {rhm:g} and ZM = {zm:g}"
        )
    z, s = _cw_shape(k, d, h)
    if k > 0.0:
        total = s(d)
        us = [0.0]
        for i in range(1, n_seg):
            target = total * i / n_seg
            us.append(_bisect(lambda u, t=target: s(u) - t, 0.0, d))
        us.append(d)
    else:
        us = [d * i / n_seg for i in range(n_seg + 1)]
    out = [p1]
    for u in us[1:-1]:
        out.append((p1[0] + ex * u, p1[1] + ey * u, p1[2] + z(u)))
    out.append(p2)
    return out, s(d)


def catenary_vertices(card: Card) -> list[tuple[float, float, float]]:
    """A ``CW`` card's ``NS + 1`` vertices (see `_catenary`)."""
    return _catenary(card)[0]


def catenary_length(card: Card) -> float:
    """The arc length of a ``CW`` card's curve, the "Catenary length" both
    engines echo beside the card (see `_catenary`)."""
    return _catenary(card)[1]


def _cw(card: Card, wires: list[FlatWire]) -> None:
    """``CW``: one one-segment wire per chord of :func:`catenary_vertices`,
    all carrying the card's tag, so ``(tag, segment)`` counts along the
    curve."""
    vertices = catenary_vertices(card)
    for a, b in zip(vertices, vertices[1:], strict=False):
        wires.append(FlatWire(card.i(0), 1, list(a), list(b), card.f(8)))


_NEC4_GEOMETRY_BUILDERS = MappingProxyType(
    {**_GEOMETRY_BUILDERS, "GA": _ga, "GH": _gh, "CW": _cw}
)


class _Nec4Parser(_Nec2Parser):
    """The ``nec2`` state machine with the NEC-4.2 deltas in the module
    docstring; every other card is read by the parent unchanged."""

    _geometry_builders = _NEC4_GEOMETRY_BUILDERS

    def __init__(self) -> None:
        super().__init__()
        # The extended kernel by default, and no card turns it off (module
        # docstring, momwire#1326).
        self._extended_kernel = True
        # (tag, seg) -> the source kind that card address was driven with.
        # A port's kind is fixed for the deck because the model's feed list is
        # one port set across every execute group.
        self._source_kind: dict[tuple[int, int], int] = {}

    def _geometry(self, card: Card) -> None:
        if card.mnemonic == "GE":
            if card.i(0) not in (-1, 0, 1):
                raise DeckError(
                    f"GE {card.i(0)} is not a NEC-4.2 ground flag; this engine's "
                    f"nec4 dialect reads GE -1 (structure below ground), GE 0 "
                    f"(no ground) and GE 1 (structure above ground)"
                )
            if card.i(1) not in _GE_I2:
                raise DeckError(
                    f"GE with a second field of {card.i(1)} is not served by "
                    f"this engine's nec4 dialect; EZNEC's NEC-4.2 slot writes -1 "
                    f"there, and no capture shows what any other value means"
                )
        super()._geometry(card)

    def _near_coordinates(self, card: Card) -> int:
        """Rectangular (0) or SPHERICAL (1): the module docstring's ``NE 1``."""
        coordinates = card.i(0)
        if coordinates not in (0, 1):
            raise DeckError(
                f"{card.mnemonic} coordinate system {coordinates} is not one "
                f"this engine's nec4 dialect reads; the grid is rectangular (0: "
                f"X, Y, Z in metres) or spherical (1: R in metres, then theta "
                f"from the zenith and phi from +x, in degrees)"
            )
        return coordinates

    def _unarmed_execute(self, card: Card) -> None:
        """A further ``RP``/``NE``/``NH`` after a run asks for one more table
        of that run (module docstring), so it is read as strictly as the card
        that ran: a form this dialect refuses is refused here too, not
        skipped because nothing was solved for it."""
        if card.mnemonic == "RP":
            self._far_field(card)
        elif card.mnemonic in ("NE", "NH"):
            self._near_field(card)

    def _ek(self, card: Card) -> None:
        """Ignored: NEC-4.2's ``EK`` has no effect (module docstring), so
        the kernel stays the dialect's default whatever the card says.  The
        printout echoes the card and NEC-4.2's own line about it."""

    def _ld(self, card: Card) -> None:
        if card.i(0) in (2, 3) and card.f(6) != 0.0:
            # The nec2 reading folds this field with the segment length
            # (momwire#1091), measured on nec2c.  NEC-4.2 was not measured on
            # it and no capture carries one, so this dialect keeps the
            # refusal rather than assume NEC-2's scaling.
            raise DeckError(
                f"LD {card.i(0)} asks for a capacitance of {card.f(6):g} in its "
                f"per-unit-length RLC, which this engine's nec4 dialect does "
                f"not serve: NEC-2 scales that field by the segment length, and "
                f"no NEC-4.2 run shows whether NEC-4.2 does the same"
            )
        super()._ld(card)

    def _gn(self, card: Card) -> None:
        if card.trailer is not None:
            # The Sommerfeld-table FILE (module docstring), on any GN type
            # (momwire#1443): a cache, so the card is read without it.  The
            # printout reads the name back off the deck text
            # (``eznec._nec4._ground_file``).
            card = Card(card.mnemonic, card.values, card.raw)
        code = card.i(0)
        if code == 3:
            if card.i(1) != 0:
                raise DeckError(
                    f"GN 3 with a {card.i(1)}-wire radial ground screen is not "
                    f"supported by this engine"
                )
            # One physical ground, two NEC-4.2 evaluations of it (module
            # docstring): read as the GN 2 it shares a model with.
            card = Card(card.mnemonic, (2.0, *card.values[1:]), card.raw, card.trailer)
        if card.i(0) in (0, 2) and card.f(4) <= 0.0:
            # A short card reads its missing fields as 0 (momwire#1576), and
            # a finite ground at a permittivity of 0 has no answer: the solve
            # used to fail inside the Sommerfeld fill with "array must not
            # contain infs or NaNs" (momwire#1592's audit).  The nec5
            # dialect refuses the same card by name in its `_gn`.
            raise DeckError(
                f"GN {code} needs a positive relative permittivity (field 5) "
                f"for a finite ground, and this card's reads {card.f(4):g}"
            )
        super()._gn(card)

    def _ex(self, card: Card) -> None:
        ex_type = card.i(0)
        if ex_type == _EX_ELEMENT:
            raise DeckError(_EX4_REFUSAL)
        if ex_type not in (_EX_VOLTAGE, _EX_CURRENT):
            raise DeckError(
                f"EX type {ex_type} is not served by this engine's nec4 dialect, "
                f"which drives EX 0 (voltage source) and EX 6 (current source) "
                f"only"
            )
        address = (card.i(1), card.i(2))
        kind = self._source_kind.setdefault(address, ex_type)
        if kind != ex_type:
            raise DeckError(
                f"EX {ex_type} on tag {address[0]} segment {address[1]}, which an "
                f"earlier EX {kind} drove, is not supported by this engine: one "
                f"segment is one port for the whole deck, and a port is either a "
                f"voltage source or a current source"
            )
        if ex_type == _EX_VOLTAGE:
            super()._ex(card)
            return
        amps = complex(card.f(4), card.f(5))
        if amps == 0:
            raise DeckError(
                f"EX 6 on tag {address[0]} segment {address[1]} sets a current of "
                f"zero, which is an open circuit rather than a source; this "
                f"engine refuses it rather than guess a drive"
            )
        # The parent's retention rule, unchanged: the first EX after an
        # execute card replaces the list, every further one adds to it.
        self._saw_ex = True
        if self._sources_stale:
            self._sources = []
            self._sources_stale = False
        self._sources.append((address[0], address[1], amps))

    def model(self) -> DeckModel:
        model = super().model()
        # The parent's union port order (`_feeds_and_groups`): every group's
        # sources, first appearance first.
        order: list[tuple[int, int]] = []
        for pending in self.groups:
            if pending is None:
                continue
            for tag, seg, _v in pending.sources:
                if (tag, seg) not in order:
                    order.append((tag, seg))
        current = tuple(
            i
            for i, address in enumerate(order)
            if self._source_kind[address] == _EX_CURRENT
        )
        return replace(
            model,
            current_feeds=current if current else model.current_feeds,
            extended_kernel_default=True,
            junction_rule="nec42",
        )


def parse_nec4(text: str) -> DeckModel:
    """Parse one ``nec4`` deck body into a :class:`DeckModel`."""
    parser = _Nec4Parser()
    parser.feed(text)
    return parser.model()
