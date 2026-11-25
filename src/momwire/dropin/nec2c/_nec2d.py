"""The NEC-2D printout: LLNL's Fortran NEC-2 layout, for 4nec2 (momwire#2138).

There are two NEC-2 printout layouts.  :mod:`._portal` writes nec2c's
(Kyriazis' C translation, matched to Ward AE6TY's ``nec2c.ae6ty``, which
SimNEC reads).  4nec2's NEC-2 engines (``nec2dxs``) write the original
Fortran's, NEC-2D's, and 4nec2 accepts nothing else: its readers find the
sections by their ``- - - NAME - - -`` headings and the impedance row by
position.  A program name carrying ``nec2d`` (``momwire-nec2d[-<basis>]``)
selects this module's layout; the deck reading, the refusals, the NEC-2
rules and the solve are the portal's, unchanged, so a ``momwire-nec2d``
printout and a ``momwire-nec2c`` one of the same deck carry one solve's
numbers in two layouts.

Where the layout comes from
---------------------------
Every heading, column header, field width and blank-line run below is the
``FORMAT`` statement that prints it in ``nec2dx.f`` (NEC-2D, Lawrence
Livermore National Laboratory, "FILE CREATED 4/11/80", double-precision
revision 6/4/85, a US government work in the public domain), cited by its
routine and statement label, and checked against ``nec2dxs`` printouts
captured from 4nec2 (2026-08, black-box output only).  Fortran's record
rules decide the blanks: ``n`` slashes before a format's first text are ``n``
blank lines, ``n`` between two texts are ``n - 1``, and ``n`` after the last
text are ``n``; a format of slashes alone is ``n + 1``.

Which sections print, and when, is NEC-2D's main program's own state machine
(``IGO``/``IFLOW``, :class:`_Flow`): an ``FR`` card restarts at FREQUENCY, a
load or ground card at STRUCTURE IMPEDANCE LOADING, an excitation or network
card at the excitation, and a pattern card after a completed run prints its
table alone.  The solve behind each execute card is the portal's group for
that card (:attr:`~._portal.PortalDeck.groups`).

What the layout shares with momwire's NEC-4.2 printout
(:mod:`momwire.dropin.nec5._nec4_printout`, the same Fortran family) it
reuses from there: the wire, arc, reflection and scale rows of the structure
table, the segmentation row and the input-parameter row are the same
statements in both engines.

Departures, each deliberate:

* line 2, blank in ``nec2dxs``'s printout, carries the engine stamp, as every
  momwire printout's line 2 does (:func:`momwire.dropin.nec5._printout.engine_stamp`);
* the TIMING and RUN TIME figures are this process's wall clock;
* the dust floors of the portal (momwire#403) zero what NEC-2D prints as
  rounding noise, exactly as the nec2c layout does;
* a deck the portal refuses prints `` ***** NEC ERROR - <reason>`` where its
  answer would stand, as momwire's other Fortran-family printouts do;
* ``MP``, the ae6ty multicore hint, is echoed and otherwise ignored (NEC-2D
  knows no such card); ``PT`` with a positive flag (a receiving-pattern
  table, which needs a plane-wave source this dialect refuses) prints no
  currents table.
"""

from __future__ import annotations

import io
import math
import sys
import time
from dataclasses import dataclass, replace

import numpy as np

from ... import _features, _nec2_end_cap
from ..._far_readout import _gain_db, _phase_deg, _polarisation
from ...deck import Card, parse_card
from ...serve import Seam, run_session
from ..nec5 import _printout
from ..nec5._nec4_printout import (
    _WIRE_COLUMNS,
    ArcRow,
    ReflectRow,
    ScaleRow,
    WireRow,
)
from ..nec5._nec4_printout import _port_row as _nec4_port_row
from ..nec5._nec4_printout import _segment_row as _nec4_segment_row
from ..nec5._nec4_printout import _structure_rows as _nec4_structure_rows
from ..nec5._printout import PortRow
from . import _portal as _p

__all__ = ["file_printout", "file_refusal", "nec2d_seam", "render", "resident_loop"]

# NEC-2D's speed of light, ``CVEL`` (``DATA CVEL/299.8/``): the printed
# WAVELENGTH and every constant derived from it in the printout use it.
_CVEL = 299.8
# RDPAT's own constants (``DATA PI,TA,TD``), so the average-gain arithmetic
# prints NEC-2D's digits.
_PI = 3.141592654
_TA = 1.745329252e-02

# --------------------------------------------------------------------------
# Fortran edit descriptors
# --------------------------------------------------------------------------


def _i(value: int, width: int) -> str:
    """``Iw``: right-justified, or ``w`` asterisks when it does not fit (the
    ``NT ***`` echo of a 9901 tag in an I3 field)."""
    text = f"{int(value):{width}d}"
    return "*" * width if len(text) > width else text


def _f(value: float, width: int, digits: int) -> str:
    """``Fw.d``, asterisks on overflow.  A negative zero prints as zero: it
    is this engine's arithmetic, never a value NEC-2D would have printed."""
    text = f"{value + 0.0:{width}.{digits}f}"
    return "*" * width if len(text) > width else text


def _e(value: float, width: int, digits: int) -> str:
    """``1PEw.d``, Fortran's three-digit-exponent spelling included."""
    return _printout._e_fortran(value + 0.0, width, digits)


def _blank(count: int) -> list[str]:
    return [""] * count


# --------------------------------------------------------------------------
# the run's chrome (main program FORMATs 126-128, 135, 137, 201)
# --------------------------------------------------------------------------

_RULE = " " * 33 + "*" * 45
_TITLE = " " * 36 + "NUMERICAL ELECTROMAGNETICS CODE (NEC-2D)"
# 128: (////,37X,'- - - - COMMENTS - - - -',//)
_COMMENTS = (*_blank(4), " " * 37 + "- - - - COMMENTS - - - -", *_blank(2))
# 129: (25X,20A4) of COM(1..19), the card's columns 3-78.
_COMMENT_INDENT = " " * 25
_COMMENT_COLUMNS = slice(2, 78)
_COMMENT_WIDTH = 76
# 135: (/////), between a finished run and the next card.
_RUN_GAP = 6
_ERROR_PREFIX = _printout._ERROR_PREFIX


def _stamp() -> str:
    return _printout.engine_stamp(_p._active_basis_spelled or _p._active_basis_name)


def _banner() -> list[str]:
    """126 and 127: the form feed, then (///,33X,rule,//,36X,title,//,33X,rule).
    Line 2, blank in NEC-2D's, is the engine stamp."""
    return ["1", _stamp(), "", "", _RULE, "", _TITLE, "", _RULE]


def _comment_lines(body: str) -> list[str]:
    """The leading ``CM``/``CE`` cards as 129 prints them, ``CE`` included."""
    out = []
    for line in body.splitlines():
        if not line.strip():
            continue
        head = line[:2].upper()
        if head not in ("CM", "CE"):
            break
        out.append(_COMMENT_INDENT + line[_COMMENT_COLUMNS].ljust(_COMMENT_WIDTH))
        if head == "CE":
            break
    return out


def _card_fields(card: Card) -> tuple[list[int], list[float]]:
    return [card.i(k) for k in range(4)], [card.f(4 + k) for k in range(6)]


def _echo(number: int, mnemonic: str, ints: list[int], reals: list[float]) -> str:
    """137: (1X,'***** DATA CARD NO.',I3,3X,A2,1X,I3,3(1X,I5),6(1X,1P,E12.5))."""
    return (
        f" ***** DATA CARD NO.{_i(number, 3)}   {mnemonic:2s} {_i(ints[0], 3)}"
        + "".join(" " + _i(v, 5) for v in ints[1:])
        + "".join(" " + _e(v, 12, 5) for v in reals)
    )


def _run_time(seconds: float) -> list[str]:
    """201: (/,' RUN TIME =',F10.3)."""
    return ["", " RUN TIME =" + _f(seconds, 10, 3)]


def _error(exc: BaseException) -> list[str]:
    return ["", "", "", _ERROR_PREFIX + str(exc)]


# --------------------------------------------------------------------------
# STRUCTURE SPECIFICATION (DATAGN, CONECT)
# --------------------------------------------------------------------------

# DATAGN 40: (////,33X,heading,//,37X,..,/,37X,..,/,37X,..,//), then 41.
_STRUCTURE_HEAD = (
    *_blank(4),
    " " * 33 + "- - - STRUCTURE SPECIFICATION - - -",
    "",
    " " * 37 + "COORDINATES MUST BE INPUT IN",
    " " * 37 + "METERS OR BE SCALED TO METERS",
    " " * 37 + "BEFORE STRUCTURE INPUT IS ENDED",
    *_blank(2),
    *_WIRE_COLUMNS,
)
# CONECT 50 and 52.
_JUNCTION_HEAD = (
    *_blank(2),
    "         - MULTIPLE WIRE JUNCTIONS -",
    " JUNCTION    SEGMENTS  (- FOR END 1, + FOR END 2)",
)
_NO_JUNCTIONS = "  NONE"
# DATAGN 53 and 54.
_SEGMENTATION_HEAD = (
    *_blank(4),
    " " * 33 + "- - - - SEGMENTATION DATA - - - -",
    "",
    " " * 40 + "COORDINATES IN METERS",
    "",
    " " * 25 + "I+ AND I- INDICATE THE SEGMENTS BEFORE AND AFTER I",
    *_blank(2),
    "  SEG.   COORDINATES OF SEG. CENTER     SEG.     ORIENTATION ANGLES    WIRE"
    "    CONNECTION DATA   TAG",
    "  NO.       X         Y         Z       LENGTH     ALPHA     BETA      RADIUS"
    "    I-   I    I+    NO.",
)


@dataclass
class _Geometry:
    """DATAGN's running counters: the segment count ``N``, the symmetric
    cell ``NP``, the symmetry flag ``IPSYM`` and every segment's tag (for a
    ``GM`` card's ``ISEGNO``), kept as WIRE, REFLC and MOVE keep them."""

    n: int = 0
    np: int = 0
    ipsym: int = 0
    tags: list[int] | None = None

    def __post_init__(self) -> None:
        self.tags = []

    def wire(self, n_seg: int, tag: int) -> tuple[int, int]:
        first = self.n + 1
        self.n += n_seg
        self.tags += [tag] * n_seg
        self.np = self.n
        self.ipsym = 0
        return first, self.n

    def reflect(self, ix: int, iy: int, iz: int, increment: int) -> None:
        """REFLC, the plane form: Z first, then Y, then X, the increment
        doubling after each plane."""
        self.np = self.n
        self.ipsym = 0
        if not (ix or iy or iz):
            return
        self.ipsym = 2 if iz else 1
        step = increment
        for on in (iz, iy, ix):
            if on:
                self.tags += [t + step if t else 0 for t in self.tags]
                self.n *= 2
                step *= 2

    def rotate(self, copies: int, increment: int) -> None:
        """REFLC, the rotation form (``IX < 0``)."""
        self.np = self.n
        self.ipsym = -1
        block = list(self.tags)
        for _ in range(max(copies, 1) - 1):
            block = [t + increment if t else 0 for t in block]
            self.tags += block
        self.n *= max(copies, 1)

    def move(self, rot_x: float, rot_y: float, copies: int, its: int, inc: int):
        """MOVE: a rotation about X or Y triples the flag; copies, or a move
        of part of the structure, end the symmetry."""
        if abs(rot_x) + abs(rot_y) > 1e-10:
            self.ipsym *= 3
        first = 1
        if its:
            first = next((k + 1 for k, t in enumerate(self.tags) if t == its), 1)
        if copies:
            block = self.tags[first - 1 :]
            for _ in range(copies):
                block = [t + inc if t else 0 for t in block]
                self.tags += block
            self.n = len(self.tags)
        if copies == 0 and first == 1:
            return
        self.np = self.n
        self.ipsym = 0


def _helix_rows(card: Card, number: int, first: int, last: int) -> list[str]:
    """DATAGN 124, then HELIX 104 (a tapered helix) or 105."""
    s, hl, a1, b1, a2, b2, rad = (card.f(k) for k in range(2, 9))
    rows = [
        "     HELIX STRUCTURE-   AXIAL SPACING BETWEEN TURNS ="
        + _f(s, 8, 3)
        + " TOTAL AXIAL LENGTH ="
        + _f(hl, 8, 3),
        " "
        + _i(number, 5)
        + "  RADIUS OF HELIX ="
        + "".join("  " + _f(v, 8, 3) for v in (a1, b1, a2, b2))
        + " " * 7
        + _f(rad, 11, 5)
        + _i(card.i(1), 8)
        + "    "
        + _i(first, 5)
        + " "
        + _i(last, 5)
        + "   "
        + _i(card.i(0), 5),
    ]
    if a2 != a1:
        cone = math.atan(a2 / (abs(hl) + (abs(hl) * a1) / (a2 - a1)))
        rows.append("     THE CONE ANGLE OF THE SPIRAL IS" + _f(cone, 10, 4))
        return rows
    if b1 == 0.0:
        b1 = a1
    if a1 == b1:
        hdia = 2.0 * a1
        pitch = math.atan(s / (math.pi * hdia))
        turn = hdia * math.pi / math.cos(pitch)
        pitch = 180.0 * pitch / math.pi
    else:
        hmaj, hmin = (2.0 * a1, 2.0 * b1) if a1 >= b1 else (2.0 * b1, 2.0 * a1)
        hdia = math.sqrt((hmaj * hmaj + hmin * hmin) / 2 * hmaj)
        turn = 2.0 * math.pi * hdia
        pitch = (180.0 / math.pi) * math.atan(s / (math.pi * hdia))
    rows += [
        "     THE PITCH ANGLE IS" + _f(pitch, 10, 4),
        "     THE LENGTH OF WIRE/TURN IS" + _f(turn, 10, 4),
    ]
    return rows


def _structure(deck: _p.PortalDeck, solver: _p.DeckSolver) -> list[str]:
    """STRUCTURE SPECIFICATION through the junction table."""
    out = list(_STRUCTURE_HEAD)
    geo = _Geometry()
    number = 0
    ge = 0
    for card in deck.geometry:
        m = card.mnemonic
        if m in ("GW", "GA", "GH"):
            number += 1
            first, last = geo.wire(card.i(1), card.i(0))
            if m == "GW":
                row = WireRow(
                    number,
                    (card.f(2), card.f(3), card.f(4)),
                    (card.f(5), card.f(6), card.f(7)),
                    card.f(8),
                    card.i(1),
                    first,
                    last,
                    card.i(0),
                )
                out += _nec4_structure_rows(row)
            elif m == "GA":
                row = ArcRow(
                    number,
                    card.f(2),
                    card.f(3),
                    card.f(4),
                    card.f(5),
                    card.i(1),
                    first,
                    last,
                    card.i(0),
                )
                out += _nec4_structure_rows(row)
            else:
                out += _helix_rows(card, number, first, last)
        elif m == "GX":
            code = card.i(1)
            iy, iz = code // 10, code % 10
            ix, iy = iy // 10, iy % 10
            ix, iy, iz = int(ix != 0), int(iy != 0), int(iz != 0)
            out += _nec4_structure_rows(
                ReflectRow((bool(ix), bool(iy), bool(iz)), card.i(0))
            )
            geo.reflect(ix, iy, iz, card.i(0))
        elif m == "GR":
            # DATAGN 45: (6X,'STRUCTURE ROTATED ABOUT Z-AXIS',I3,
            # ' TIMES.  LABELS INCREMENTED BY',I5)
            out.append(
                "      STRUCTURE ROTATED ABOUT Z-AXIS"
                + _i(card.i(1), 3)
                + " TIMES.  LABELS INCREMENTED BY"
                + _i(card.i(0), 5)
            )
            geo.rotate(card.i(1), card.i(0))
        elif m == "GS":
            out += _nec4_structure_rows(ScaleRow(card.f(2)))
        elif m == "GM":
            # DATAGN 47: (6X,'THE STRUCTURE HAS BEEN MOVED, MOVE DATA CARD
            # IS -'/6X,I3,I5,7F10.5)
            out += [
                "      THE STRUCTURE HAS BEEN MOVED, MOVE DATA CARD IS -",
                "      "
                + _i(card.i(0), 3)
                + _i(card.i(1), 5)
                + "".join(_f(card.f(k), 10, 5) for k in range(2, 9)),
            ]
            geo.move(card.f(2), card.f(3), card.i(1), int(card.f(8) + 0.5), card.i(0))
        elif m == "GE":
            ge = card.i(0)
    # CONECT 54, 55 and the GE plane's effect on the symmetric cell.
    if ge != 0:
        out += ["", "   GROUND PLANE SPECIFIED."]
        if ge > 0:
            out += [
                "",
                "   WHERE WIRE ENDS TOUCH GROUND, CURRENT WILL BE INTERPOLATED"
                " TO IMAGE IN GROUND PLANE.",
                "",
            ]
        if geo.ipsym == 2:
            geo.np *= 2
        if abs(geo.ipsym) > 2:
            geo.np = geo.n
        if geo.np == geo.n:
            geo.ipsym = 0
    n = solver.n_segments
    cell = geo.np if geo.ipsym and geo.np else n
    flag = geo.ipsym if cell != n else 0
    # CONECT 58: (/,3X,'TOTAL SEGMENTS USED=',I5,5X,'NO. SEG. IN A SYMMETRIC
    # CELL=',I5,5X,'SYMMETRY FLAG=',I3)
    out += [
        "",
        "   TOTAL SEGMENTS USED="
        + _i(n, 5)
        + "     NO. SEG. IN A SYMMETRIC CELL="
        + _i(cell, 5)
        + "     SYMMETRY FLAG="
        + _i(flag, 3),
    ]
    folds = n // cell if cell else 1
    if folds != 1 and flag < 0:
        out += [" STRUCTURE HAS" + _i(folds, 4) + " FOLD ROTATIONAL SYMMETRY", ""]
    elif folds != 1 and flag > 0:
        planes = 3 if folds == 8 else folds // 2
        out += [" STRUCTURE HAS" + _i(planes, 2) + " PLANES OF SYMMETRY", ""]
    out += _JUNCTION_HEAD
    ends, order = _p._segment_end_nodes(solver.wires)
    junctions = [sorted(ends[key], key=abs) for key in order if len(ends[key]) >= 3]
    for index, members in enumerate(junctions, 1):
        # CONECT 51: (1X,I5,5X,20I5,/,(11X,20I5))
        for k in range(0, len(members), 20):
            lead = " " + _i(index, 5) + " " * 5 if k == 0 else " " * 11
            out.append(lead + "".join(_i(m, 5) for m in members[k : k + 20]))
    if not junctions:
        out.append(_NO_JUNCTIONS)
    return out


def _segmentation(deck: _p.PortalDeck, solver: _p.DeckSolver) -> list[str]:
    """SEGMENTATION DATA: DATAGN 55 per segment, with CONECT's connection
    columns (each end names the next member of its junction, cyclically:
    CONECT searches from the segment after itself)."""
    from ..nec5._nec4 import _connections, _segment_row

    connections = _connections(solver, deck)
    rows = []
    for seg, (i_minus, i_plus) in zip(solver.segments, connections, strict=True):
        row = _segment_row(seg, i_minus, i_plus)
        # A negative zero is this engine's arithmetic, not a printed value.
        row = replace(
            row,
            centre=tuple(c + 0.0 for c in row.centre),
            alpha_deg=row.alpha_deg + 0.0,
            beta_deg=row.beta_deg + 0.0,
        )
        rows.append(_nec4_segment_row(row))
    return [*_SEGMENTATION_HEAD, *rows]


# --------------------------------------------------------------------------
# a run's preamble (main program 145, 196, 321, 146-153; LOAD; PRNT)
# --------------------------------------------------------------------------


def _frequency(freq_mhz: float, rkh: float, ek: bool) -> list[str]:
    """145, 196 and (under ``EK``) 321."""
    out = [
        *_blank(4),
        " " * 33 + "- - - - - - FREQUENCY - - - - - -",
        "",
        " " * 36 + "FREQUENCY=" + _e(freq_mhz, 11, 4) + " MHZ",
        " " * 36 + "WAVELENGTH=" + _e(_CVEL / freq_mhz, 11, 4) + " METERS",
        *_blank(4),
        " " * 20
        + "APPROXIMATE INTEGRATION EMPLOYED FOR SEGMENTS MORE THAN"
        + _f(rkh, 8, 3)
        + " WAVELENGTHS APART",
    ]
    if ek:
        out += ["", " " * 20 + "THE EXTENDED THIN WIRE KERNEL WILL BE USED"]
    return out


# LOAD 25, the loading table's column header.
_LOAD_COLUMNS = (
    "       LOCATION          RESISTANCE   INDUCTANCE  CAPACITANCE       "
    "IMPEDANCE (OHMS)     CONDUCTIVITY    TYPE",
    "    ITAG FROM THRU          OHMS        HENRYS       FARADS        REAL"
    "      IMAGINARY    MHOS/METER",
)
# LOAD's CTYPE strings, by load type.
_LOAD_TYPE = {
    0: " SERIES ",
    1: "PARALLEL",
    2: " SERIES (PER METER) ",
    3: "PARALLEL (PER METER)",
    4: "FIXED IMPEDANCE ",
    5: "  WIRE  ",
}


def _load_row(card: Card) -> str:
    """PRNT 92: (/,3X,3A,3X,6A,3X,A) — the location as I5 cells (blank when
    zero, ``  ALL`` when all three are), six 1PE13.4 cells blank under
    1E-30, and the type."""
    kind = card.i(0)
    tag, first, last = card.i(1), card.i(2), card.i(3) or card.i(2)
    if tag == 0 and first == 0 and last == 0:
        cells = ["  ALL", " " * 5, " " * 5]
    else:
        cells = [_i(v, 5) if v else " " * 5 for v in (tag, first, last)]
    if kind in (0, 1, 2, 3):
        values = (card.f(4), card.f(5), card.f(6), 0.0, 0.0, 0.0)
    elif kind == 4:
        values = (0.0, 0.0, 0.0, card.f(4), card.f(5), 0.0)
    else:
        values = (0.0, 0.0, 0.0, 0.0, 0.0, card.f(4))
    reals = "".join(_e(v, 13, 4) if abs(v) > 1e-30 else " " * 13 for v in values)
    return "   " + "".join(cells) + "   " + reals + "   " + _LOAD_TYPE[kind]


def _loaded_twice(loads: tuple[Card, ...], solver: _p.DeckSolver) -> bool:
    """LOAD's ``IWARN``: a card lands on a segment an earlier card in force
    already loaded.  Every load type counts, wire conductivity (``LD 5``)
    included, which is how a trap on a conducting wire draws the note."""
    loaded: set[int] = set()
    twice = False
    for card in loads:
        tag, first, last = card.i(1), card.i(2), card.i(3) or card.i(2)
        values = [card.f(k) for k in range(4, 7)]
        hits = []
        count = 0
        for seg in solver.segments:
            if tag:
                if seg.tag != tag:
                    continue
                count += 1
                if first and not first <= count <= last:
                    continue
            elif (first or last) and not first <= seg.number <= last:
                continue
            hits.append(seg.number)
        if loaded.intersection(hits):
            twice = True
        if any(values):
            loaded.update(hits)
    return twice


def _loading(loads: tuple[Card, ...], twice: bool) -> list[str]:
    """146, then LOAD (25, a PRNT per card, 26) or 147."""
    out = [*_blank(3), " " * 31 + "- - - STRUCTURE IMPEDANCE LOADING - - -"]
    if not loads:
        return [*out, "", " " * 35 + "THIS STRUCTURE IS NOT LOADED"]
    out += [*_blank(2), *_LOAD_COLUMNS]
    for card in loads:
        out += ["", _load_row(card)]
    if twice:
        out += [
            "",
            " " * 10 + "NOTE, SOME OF THE ABOVE SEGMENTS HAVE BEEN LOADED TWICE - "
            "IMPEDANCES ADDED",
        ]
    return out


def _environment(ground: _p.Ground, freq_mhz: float) -> list[str]:
    """148, then 152, 151, or 391/392 and 150."""
    out = [*_blank(3), " " * 34 + "- - - ANTENNA ENVIRONMENT - - -", ""]
    if ground.kind == "free":
        return [*out, " " * 44 + "FREE SPACE"]
    if ground.kind == "pec":
        return [*out, " " * 42 + "PERFECT GROUND"]
    pad = " " * 40
    label = (
        "FINITE GROUND.  REFLECTION COEFFICIENT APPROXIMATION"
        if ground.kind == "refl"
        else "FINITE GROUND.  SOMMERFELD SOLUTION"
    )
    # EPSC=DCMPLX(EPSR,-SIG*WLAM*59.96), at NEC-2D's printed wavelength.
    eps_c = complex(ground.eps_r, -ground.sigma * (_CVEL / freq_mhz) * 59.96)
    return [
        *out,
        pad + label,
        pad + "RELATIVE DIELECTRIC CONST.=" + _f(ground.eps_r, 7, 3),
        pad + "CONDUCTIVITY=" + _e(ground.sigma, 10, 3) + " MHOS/METER",
        pad
        + "COMPLEX DIELECTRIC CONSTANT="
        + _e(eps_c.real, 12, 5)
        + _e(eps_c.imag, 12, 5),
    ]


def _timing(fill_s: float, factor_s: float) -> list[str]:
    """153."""
    return [
        *_blank(3),
        " " * 32 + "- - - MATRIX TIMING - - -",
        "",
        " " * 24
        + "FILL="
        + _f(fill_s, 9, 3)
        + " SEC.,  FACTOR="
        + _f(factor_s, 9, 3)
        + " SEC.",
    ]


# --------------------------------------------------------------------------
# a run's answer (main program 157-166, 315; NETWK 60-63)
# --------------------------------------------------------------------------

# NETWK 60, the port tables' column header.
_PORT_COLUMNS = (
    "",
    "   TAG   SEG.    VOLTAGE (VOLTS)         CURRENT (AMPS)         IMPEDANCE "
    "(OHMS)        ADMITTANCE (MHOS)      POWER",
    "   NO.   NO.    REAL        IMAG.       REAL        IMAG.       REAL     "
    "   IMAG.       REAL        IMAG.     (WATTS)",
)
# 159 (TL) and 160 (NT).
_TL_COLUMNS = (
    "",
    "      - FROM -    - TO -           TRANSMISSION LINE               -  -  "
    "SHUNT ADMITTANCES (MHOS)  -  -              LINE",
    "      TAG  SEG.   TAG  SEG.      IMPEDANCE      LENGTH            - END ONE"
    " -                 - END TWO -            TYPE",
    "      NO.   NO.   NO.   NO.         OHMS        METERS         REAL       "
    "   IMAG.         REAL          IMAG.",
)
_NT_COLUMNS = (
    "",
    "      - FROM -    - TO -                          -  -  ADMITTANCE MATRIX"
    " ELEMENTS (MHOS)  -  -",
    "      TAG  SEG.   TAG  SEG.             (ONE,ONE)                   "
    "(ONE,TWO)                   (TWO,TWO)",
    "      NO.   NO.   NO.   NO.        REAL          IMAG.         REAL       "
    "   IMAG.         REAL          IMAG.",
)
_NETWORK_TYPE = {"": "        ", "STRAIGHT": "STRAIGHT", "CROSSED": "CROSSED "}


def _port_row(tag, segment, volts, current, impedance, admittance, power) -> str:
    """NETWK 62: (2(1X,I5),1P,9E12.5) — NEC-4.2's port row."""
    return _nec4_port_row(
        PortRow(tag, segment, 0, volts, current, impedance, admittance, power)
    )


def _network_data(solver: _p.DeckSolver, group_index: int) -> list[str]:
    """158, then per kind 159 or 160 and its rows (157), the first card's
    kind first, as the main program's two-pass loop prints them."""
    live = _p.live_cards(solver.model, solver.plan, group_index)
    out = [*_blank(3), " " * 44 + "- - - NETWORK DATA - - -"]
    for kind in _p._kind_order(card.kind for card, _pair in live):
        out += _TL_COLUMNS if kind == "TL" else _NT_COLUMNS
        for card, _pair in live:
            if card.kind != kind:
                continue
            ends, values, kind_name = _p.network_row_values(solver, card)
            # 157: (4X,4(I5,1X),1P,6(3X,E11.4),3X,A6,A2)
            out.append(
                "    "
                + "".join(_i(v, 5) + " " for v in ends)
                + "".join("   " + _e(v, 11, 4) for v in values)
                + "   "
                + _NETWORK_TYPE[kind_name]
            )
    return out


def _segment_rows(table: _p.SegTable) -> list[str]:
    """165: (1X,2I5,3F9.4,F9.5,1X,1P,3E12.4,0P,F9.3), one per row."""
    out = []
    for number, tag, centre, length, value in zip(
        table.numbers, table.tags, table.centre_wl, table.length_wl, table.values
    ):
        value = complex(value)
        magnitude = abs(value)
        phase = math.degrees(math.atan2(value.imag, value.real)) if magnitude else 0.0
        out.append(
            " "
            + _i(number, 5)
            + _i(tag, 5)
            + "".join(_f(float(c), 9, 4) for c in centre)
            + _f(float(length), 9, 5)
            + " "
            + _e(value.real, 12, 4)
            + _e(value.imag, 12, 4)
            + _e(magnitude, 12, 4)
            + _f(phase, 9, 3)
        )
    return out


def _range_segments(control, solver: _p.DeckSolver) -> list:
    """The rows a ``PT 0`` / ``PQ`` card's range leaves (main program labels
    65-66 and 317): with a tag, its segments counted from 1; without, the
    structure's.  A zero ``first`` prints every segment (``IPTFLG=-2``)."""
    if control.first == 0:
        return solver.segments
    last = control.last or control.first
    chosen, count = [], 0
    for seg in solver.segments:
        if control.tag and seg.tag != control.tag:
            continue
        count += 1
        if control.first <= count <= last:
            chosen.append(seg)
    return chosen


_CURRENTS_HEAD = (
    *_blank(3),
    " " * 29 + "- - - CURRENTS AND LOCATION - - -",
    "",
    " " * 33 + "DISTANCES IN WAVELENGTHS",
    *_blank(2),
    "  SEG.  TAG    COORD. OF SEG. CENTER     SEG.            - - - CURRENT (AMPS)"
    " - - -",
    "  NO.   NO.     X        Y        Z      LENGTH     REAL        IMAG.       "
    "MAG.        PHASE",
)
_CHARGES_HEAD = (
    *_blank(3),
    " " * 34 + "- - - CHARGE DENSITIES - - -",
    "",
    " " * 36 + "DISTANCES IN WAVELENGTHS",
    *_blank(2),
    "  SEG.  TAG    COORD. OF SEG. CENTER     SEG.          CHARGE DENSITY "
    "(COULOMBS/METER)",
    "  NO.   NO.     X        Y        Z      LENGTH     REAL        IMAG.       "
    "MAG.        PHASE",
)


def _power_budget(result: _p.GroupResult) -> list[str]:
    """166."""
    pad = " " * 43
    return [
        *_blank(3),
        " " * 40 + "- - - POWER BUDGET - - -",
        "",
        pad + "INPUT POWER   =" + _e(result.p_in, 11, 4) + " WATTS",
        pad + "RADIATED POWER=" + _e(result.p_radiated, 11, 4) + " WATTS",
        pad + "STRUCTURE LOSS=" + _e(result.p_structure, 11, 4) + " WATTS",
        pad + "NETWORK LOSS  =" + _e(result.p_network, 11, 4) + " WATTS",
        pad + "EFFICIENCY    =" + _f(result.efficiency, 7, 2) + " PERCENT",
    ]


def _answer(
    deck: _p.PortalDeck,
    solver: _p.DeckSolver,
    group: _p.ExecuteGroup,
    records: _p.RunRecords,
    group_index: int,
) -> list[str]:
    """Main program labels 53-70: the network data, NETWK's two tables, the
    currents, the charges and the power budget."""
    result = records.result
    out: list[str] = []
    if result.network_points:
        out += _network_data(solver, group_index)
        out += [
            *_blank(3),
            " " * 27
            + "- - - STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS - - -",
            *_PORT_COLUMNS,
            *(_port_row(*row) for row in _p.network_excitation_rows(result)),
        ]
    if records.aip_rows:
        out += [
            *_blank(3),
            " " * 42 + "- - - ANTENNA INPUT PARAMETERS - - -",
            *_PORT_COLUMNS,
        ]
        out += [
            _port_row(
                row.tag,
                row.segment,
                row.volts,
                row.current,
                row.impedance,
                row.admittance,
                row.power,
            )
            for row in records.aip_rows
        ]
    # PT: IPTFLG, -2 unless a card says otherwise; a card with no first
    # segment prints them all (-2) unless it is -1, which prints none.  A
    # positive flag is the receiving-pattern table, which needs a plane-wave
    # source: nothing here.
    pt = group.pt
    flag = -2 if pt is None else pt.flag
    if pt is not None and pt.first == 0 and flag != -1:
        flag = -2
    if flag != -1 and flag <= 0:
        segments = _range_segments(pt, solver) if flag == 0 else solver.segments
        table = _p._seg_table(
            solver, segments, result.segment_currents, result.wavelength
        )
        out += [*_CURRENTS_HEAD, *_segment_rows(table)]
    # PQ: IPTFLQ, -1 (no table) unless a card says otherwise; -2 prints every
    # segment, any other flag the card's range.
    pq = group.pq
    qflag = -1 if pq is None else pq.flag
    if pq is not None and pq.first == 0 and qflag != -1:
        qflag = -2
    if qflag != -1:
        segments = solver.segments if qflag == -2 else _range_segments(pq, solver)
        table = _p._seg_table(
            solver, segments, result.segment_charges, result.wavelength
        )
        out += [*_CHARGES_HEAD, *_segment_rows(table)]
    out += _power_budget(result)
    return out


# --------------------------------------------------------------------------
# NFPAT
# --------------------------------------------------------------------------

_NEAR_HEAD = {
    "NE": (
        *_blank(3),
        " " * 35 + "- - - NEAR ELECTRIC FIELDS - - -",
        "",
        "            -  LOCATION  -                     -  EX  -               "
        "-  EY  -               -  EZ  -",
        "        X          Y          Z          MAGNITUDE   PHASE      "
        "MAGNITUDE   PHASE      MAGNITUDE   PHASE",
        "      METERS     METERS     METERS        VOLTS/M   DEGREES      "
        "VOLTS/M   DEGREES      VOLTS/M   DEGREES",
    ),
    "NH": (
        *_blank(3),
        " " * 35 + "- - - NEAR MAGNETIC FIELDS - - -",
        "",
        "            -  LOCATION  -                     -  HX  -               "
        "-  HY  -               -  HZ  -",
        "        X          Y          Z          MAGNITUDE   PHASE      "
        "MAGNITUDE   PHASE      MAGNITUDE   PHASE",
        "      METERS     METERS     METERS         AMPS/M   DEGREES       "
        "AMPS/M   DEGREES       AMPS/M   DEGREES",
    ),
}


def _near_field(card: Card, solver: _p.DeckSolver, result: _p.GroupResult):
    """NFPAT 10 or 12, then 11 per point: (2X,3(2X,F9.4),1X,3(3X,1P,E11.4,
    2X,0P,F7.2))."""
    points, field = _p.near_field_values(card, solver, result)
    out = list(_NEAR_HEAD[card.mnemonic])
    for point, components in zip(points, field, strict=True):
        out.append(
            "  "
            + "".join("  " + _f(float(c), 9, 4) for c in point)
            + " "
            + "".join(
                "   " + _e(abs(v), 11, 4) + "  " + _f(_phase_deg(v), 7, 2)
                for v in components
            )
        )
    return out


# --------------------------------------------------------------------------
# RDPAT
# --------------------------------------------------------------------------

_HPOL = {"LINEAR": "LINEAR", "RIGHT": "RIGHT ", "LEFT": "LEFT  ", "": "      "}
# RDPAT's IGTP, IGAX and IGNTP Hollerith tables, as A6 cells.
_GAIN_KIND = ("    - POWER ", "- DIRECTIVE ")
_GAIN_AXES = ((" MAJOR", " MINOR"), (" VERT.", " HOR. "))
_NORMALIZED_KIND = {
    1: " MAJOR AXIS ",
    2: " MINOR AXIS ",
    3: "   VERTICAL ",
    4: " HORIZONTAL ",
    5: "      TOTAL ",
}


def _pattern_header(axes: tuple[str, str], kind: str) -> list[str]:
    """RDPAT 40."""
    return [
        "",
        "  - - ANGLES - -       "
        + kind
        + "GAINS -       - - - POLARIZATION - - -    - - - E(THETA) - - -    "
        "- - - E(PHI) - - -",
        "  THETA     PHI       "
        + axes[0]
        + "  "
        + axes[1]
        + "   TOTAL      AXIAL     TILT   SENSE"
        + "     MAGNITUDE    PHASE " * 2,
        " DEGREES  DEGREES " + "      DB" * 3 + "        RATIO     DEG.        "
        "      VOLTS/M    DEGREES      VOLTS/M    DEGREES",
    ]


def _far_ground(mode: int, second: _p.SecondMedium | None) -> list[str]:
    """RDPAT 35 and 37: printed for every cliff mode, whatever the ground."""
    fields = second or _p.SecondMedium()
    pad = " " * 40
    return [
        *_blank(3),
        " " * 31 + "- - - FAR FIELD GROUND PARAMETERS - - -",
        *_blank(2),
        pad + ("LINEAR" if mode == 2 else "CIRCLE") + " CLIFF",
        pad + "EDGE DISTANCE=" + _f(fields.edge_distance, 9, 2) + " METERS",
        pad + "HEIGHT=" + _f(fields.height, 8, 2) + " METERS",
        pad + "SECOND MEDIUM -",
        pad + "RELATIVE DIELECTRIC CONST.=" + _f(fields.eps_r2, 7, 3),
        pad + "CONDUCTIVITY=" + _e(fields.sigma2, 10, 3) + " MHOS",
    ]


def _db(power: float) -> float:
    """DB10: ``-999.99`` under 1E-20, else ten log ten."""
    return _gain_db(power)


def _pattern(
    card: Card, solver: _p.DeckSolver, result: _p.GroupResult, freq_mhz: float
) -> list[str]:
    """One ``RP`` card's RADIATION PATTERNS, as RDPAT prints it: the far
    field ground block, the heading and its range lines, a row per
    direction (a ground skips every theta past 90.01), the average gain,
    and the normalised gain table."""
    grid = _p.pattern_grid(card, solver, result, freq_mhz)
    xnda = card.i(3)
    iavp = xnda % 10
    ipd = (xnda // 10) % 10
    inor = (xnda // 100) % 10
    iax = int((xnda // 1000) != 0)
    ipd = int(ipd != 0)
    n_theta, n_phi = len(grid.thetas), len(grid.phis)
    if n_theta < 2 or n_phi < 2:
        iavp = 0
    grounded = result.ground.kind != "free"
    wavelength = _CVEL / freq_mhz
    out: list[str] = []
    if grid.mode >= 2:
        out += _far_ground(grid.mode, grid.second)
    out += [*_blank(3), " " * 48 + "- - - RADIATION PATTERNS - - -"]
    shift = 0.0
    if grid.at_range:
        cycles = grid.rng / wavelength
        shift = -360.0 * (cycles - math.trunc(cycles))
        out += [
            " " * 54 + "RANGE=" + _e(grid.rng, 13, 6) + " METERS",
            " " * 54
            + "EXP(-JKR)/R="
            + _e(1.0 / grid.rng, 12, 5)
            + " AT PHASE"
            + _f(shift, 7, 2)
            + " DEGREES",
            "",
        ]
    out += _pattern_header(_GAIN_AXES[iax], _GAIN_KIND[ipd])
    directive = result.p_in / result.p_radiated if ipd and result.p_radiated else 1.0
    raw2 = grid.floor_scale * grid.floor_scale
    floor = _p._FIELD_FLOOR2
    gains: list[float] = []
    pint = 0.0
    tmp1 = card.f(7) * _TA
    tmp2 = 0.5 * card.f(6) * _TA
    for j in range(n_phi):
        phi = float(grid.phis[j])
        for i in range(n_theta):
            theta = float(grid.thetas[i])
            if grounded and theta > 90.01:
                continue
            et, ep = complex(grid.e_theta[i, j]), complex(grid.e_phi[i, j])
            if (et.real * et.real + et.imag * et.imag) * raw2 <= floor:
                et = 0j
            if (ep.real * ep.real + ep.imag * ep.imag) * raw2 <= floor:
                ep = 0j
            g_v, g_h = float(grid.g_v[i, j]), float(grid.g_h[i, j])
            axial, tilt, sense = _polarisation(et, ep, grid.floor_scale, 1.0e-5)
            axial = abs(axial)
            total = (g_v + g_h) * directive
            major = total / (1.0 + axial * axial)
            minor = total - major
            by_kind = {
                1: _db(major),
                2: _db(minor),
                3: _db(g_v * directive),
                4: _db(g_h * directive),
                5: _db(total),
            }
            if inor:
                gains.append(by_kind[inor])
            if iavp:
                ta = theta * _TA
                lo = ta if i == 0 else ta - tmp2
                hi = ta if i == n_theta - 1 else ta + tmp2
                area = abs(tmp1 * (math.cos(lo) - math.cos(hi)))
                if j in (0, n_phi - 1):
                    area *= 0.5
                pint += (g_v + g_h) * area
                if iavp == 2:
                    continue
            cols = (by_kind[3], by_kind[4]) if iax else (by_kind[1], by_kind[2])
            if grid.at_range:
                prop = grid.prop
                e_t = _phase_deg(et / prop) + shift if et else shift
                e_p = _phase_deg(ep / prop) + shift if ep else shift
            else:
                e_t, e_p = _phase_deg(et), _phase_deg(ep)
            # 42: (1X,F7.2,F9.2,3X,3F8.2,F11.5,F9.2,2X,A6,2(1P,E15.5,0P,F9.2))
            out.append(
                " "
                + _f(theta, 7, 2)
                + _f(phi, 9, 2)
                + "   "
                + _f(cols[0], 8, 2)
                + _f(cols[1], 8, 2)
                + _f(by_kind[5], 8, 2)
                + _f(axial, 11, 5)
                + _f(tilt, 9, 2)
                + "  "
                + _HPOL[sense]
                + _e(abs(et), 15, 5)
                + _f(e_t, 9, 2)
                + _e(abs(ep), 15, 5)
                + _f(e_p, 9, 2)
            )
    if iavp:
        theta0 = card.f(4) * _TA
        theta1 = theta0 + card.f(6) * _TA * (n_theta - 1)
        solid = abs(
            card.f(7) * _TA * (n_phi - 1) * (math.cos(theta0) - math.cos(theta1))
        )
        average = pint / solid if solid else float("nan")
        # 44: (//,3X,'AVERAGE POWER GAIN=',1P,E12.5,7X,'SOLID ANGLE USED IN
        # AVERAGING=(',0P,F7.4,')*PI STERADIANS.',//)
        out += [
            *_blank(2),
            "   AVERAGE POWER GAIN="
            + _printout._e_nan(average, 12, 5)
            + " " * 7
            + "SOLID ANGLE USED IN AVERAGING=("
            + _f(solid / _PI, 7, 4)
            + ")*PI STERADIANS.",
            *_blank(2),
        ]
    if inor:
        out += _normalized(card, gains, inor, n_theta, n_phi)
    return out


def _normalized(card: Card, gains: list[float], inor: int, n_theta, n_phi):
    """RDPAT 45 and 46: three columns of the stored gains over the maximum
    (or the card's GNOR), the angles reckoned from the index over the full
    grid as RDPAT reckons them."""
    gnor = card.f(9)
    gmax = max(gains, default=-1.0e10)
    if abs(gnor) > 1e-20:
        gmax = gnor
    out = [
        *_blank(2),
        " " * 37 + "- - - - NORMALIZED GAIN - - - -",
        "",
        " " * 37 + _NORMALIZED_KIND[inor] + "GAIN",
        " " * 38 + "NORMALIZATION FACTOR =" + _f(gmax, 9, 2) + " DB",
        "",
        # A record ending in an X edit descriptor ends at the last text.
        ("    - - ANGLES - -      GAIN       " * 3).rstrip(),
        ("    THETA     PHI        DB        " * 3).rstrip(),
        ("   DEGREES  DEGREES                " * 3).rstrip(),
    ]
    total = n_phi * n_theta
    stored = gains + [0.0] * (total - len(gains))
    rows = (total + 2) // 3
    short = rows * 3 - total
    theta0, phi0, d_theta, d_phi = card.f(4), card.f(5), card.f(6), card.f(7)

    def cell(index: int) -> str:
        block = (index - 1) // n_theta
        theta = theta0 + (index - block * n_theta - 1) * d_theta
        phi = phi0 + block * d_phi
        return (
            " "
            + _f(theta, 9, 2)
            + _f(phi, 9, 2)
            + " "
            + _f(stored[index - 1] - gmax, 9, 2)
            + " " * 6
        )

    second, third = rows, 2 * rows - (1 if short == 2 else 0)
    for i in range(1, rows + 1):
        second += 1
        third += 1
        line = cell(i)
        if i == rows and short:
            if short == 1:
                line += cell(second)
        else:
            line += cell(second) + cell(third)
        out.append(line.rstrip())
    return out


# --------------------------------------------------------------------------
# the main program's card loop
# --------------------------------------------------------------------------


@dataclass
class _Flow:
    """The main program's ``IGO`` (where the next execution resumes: 1 at
    FREQUENCY, 2 at the loading, 3 at the excitation, 4 at the near field,
    5 at the pattern), ``IFLOW`` (what the last card was), ``NFRQ``, ``RKH``
    and the near-field card waiting on a frequency loop."""

    igo: int = 1
    iflow: int = 1
    nfrq: int = 1
    rkh: float = 1.0
    near: Card | None = None


class _Printer:
    """One printout: its start time and the last card read, which an end of
    file echoes as ``EN`` (NEC-2D's reader keeps the buffer)."""

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self.last_fields: tuple[list[int], list[float]] = ([0] * 4, [0.0] * 6)

    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def structure(self, body: str) -> tuple[list[str], int]:
        """``(lines, cards echoed)`` for one structure, banner first."""
        out = [*_banner(), *_COMMENTS, *_comment_lines(body)]
        try:
            deck = _p.parse_deck(body)
        except _p._DECK_REFUSALS as exc:
            self._remember_last(body)
            return [*out, *_error(exc)], 0
        try:
            solver = _p._solver_for(deck)
        except (
            _p.PortalError,
            ValueError,
            NotImplementedError,
            np.linalg.LinAlgError,
            _features.FeatureNotCertified,
        ) as exc:
            self._remember_last(body)
            return [*out, *_error(exc)], 0
        out += _structure(deck, solver)
        out += _segmentation(deck, solver)
        out += _blank(_RUN_GAP)
        number = self._cards(deck, solver, out)
        self._remember_last(body)
        return out, number

    def _remember_last(self, body: str) -> None:
        for line in body.splitlines():
            card = parse_card(line)
            if card is not None and card.mnemonic not in _p._TERMINATORS:
                if card.mnemonic not in ("CM", "CE") and card.mnemonic not in (
                    _p._GEOMETRY_CARDS
                ):
                    self.last_fields = _card_fields(card)

    def _cards(self, deck, solver, out: list[str]) -> int:
        flow = _Flow()
        group_index = 0
        number = 0
        for card in deck.data_cards:
            number += 1
            ints, reals = _card_fields(card)
            out.append(_echo(number, card.mnemonic, ints, reals))
            m = card.mnemonic
            if m == "FR":
                flow.nfrq = max(card.i(1), 1)
                flow.igo, flow.iflow = 1, 1
            elif m in ("KH", "EK"):
                if m == "KH":
                    flow.rkh = card.f(4)
                flow.igo, flow.iflow = min(flow.igo, 2), 1
            elif m == "LD":
                if flow.iflow != 3:
                    flow.iflow = 3
                    flow.igo = min(flow.igo, 2)
            elif m == "GN":
                flow.iflow = 4
                flow.igo = min(flow.igo, 2)
            elif m == "EX":
                if flow.iflow != 5:
                    flow.iflow = 5
                    flow.igo = min(flow.igo, 3)
            elif m in ("NT", "TL"):
                if flow.iflow != 6:
                    flow.iflow = 6
                    flow.igo = min(flow.igo, 3)
            elif m == "GD":
                flow.iflow = 9
            elif m in _p._EXECUTE_CARDS:
                group = deck.groups[group_index]
                index = group_index
                group_index += 1
                if m in ("NE", "NH"):
                    if flow.iflow == 8 and flow.nfrq != 1:
                        # 143
                        out += [
                            *_blank(3),
                            " "
                            * 10
                            + "WHEN MULTIPLE FREQUENCIES ARE REQUESTED, ONLY ONE "
                            "NEAR FIELD CARD CAN BE USED -",
                            " " * 10 + "LAST CARD READ IS USED",
                        ]
                    flow.near = card
                    flow.iflow = 8
                    if flow.nfrq != 1:
                        continue
                    far = None
                elif m == "RP":
                    flow.iflow = 10
                    far = card
                else:  # XQ
                    if flow.iflow == 10 and card.i(0) == 0:
                        continue
                    if flow.nfrq == 1 and card.i(0) == 0 and flow.iflow > 7:
                        continue
                    flow.iflow = 11 if flow.iflow > 7 else 7
                    far = None
                if group is None:
                    continue
                try:
                    out += self._execute(deck, solver, group, index, flow, far)
                except (
                    _p.PortalError,
                    ValueError,
                    NotImplementedError,
                    np.linalg.LinAlgError,
                    _features.FeatureNotCertified,
                ) as exc:
                    out += _error(exc)
                    out += _blank(_RUN_GAP)
        return number

    def _execute(self, deck, solver, group, index, flow: _Flow, far) -> list[str]:
        """``GO TO (41,46,53,71,78), IGO``, through the frequency loop."""
        out: list[str] = []
        freqs = group.freqs_mhz
        start = flow.igo
        last = len(freqs) - 1
        for step, freq in enumerate(freqs):
            if step:
                start = 1
            records = _p._run_records(solver, group, freq, index)
            result = records.result
            if start <= 1:
                out += _frequency(freq, flow.rkh, group.ek)
            if start <= 2:
                twice = _loaded_twice(group.loads, solver)
                out += _loading(group.loads, twice)
                out += _environment(group.ground, freq)
                out += _timing(result.fill_ms / 1000.0, result.factor_ms / 1000.0)
            if start <= 3:
                out += _answer(deck, solver, group, records, index)
                if flow.iflow == 7:
                    flow.igo = 4
                    if flow.nfrq == 1:
                        out += _blank(_RUN_GAP)
                    continue
            if start <= 4 and flow.near is not None:
                out += _near_field(flow.near, solver, result)
                if step == last:
                    flow.near = None
                if flow.nfrq == 1:
                    flow.igo = 5
                    out += _blank(_RUN_GAP)
                    continue
            if far is not None:
                out += _pattern(far, solver, result, freq)
            flow.igo = 5
            if flow.nfrq == 1:
                out += _blank(_RUN_GAP)
        if flow.nfrq != 1:
            flow.nfrq = 1
        return out


# --------------------------------------------------------------------------
# frames, files and the stream
# --------------------------------------------------------------------------


@_features.in_context("dropin")
def _render_structure(printer: _Printer, body: str) -> tuple[list[str], int]:
    """One structure under the launcher's end-cap default (the portal's
    :func:`~._portal.render_deck` rule)."""
    _p._save_deck(body)
    with _features.pragma_scope(body):
        if _p._active_basis_name == _p._END_CAP_BASIS:
            with _nec2_end_cap.launcher_default():
                return printer.structure(body)
        return printer.structure(body)


def nec2d_seam() -> Seam:
    """This layout as a :class:`momwire.serve.Seam`: no greeting (every
    structure prints its own banner), ``NX`` answers and continues, ``EN``
    answers and ends, and the end of the input closes a structure as
    NEC-2D's reader does, echoing ``EN`` over the last card's fields."""
    printer = _Printer()

    def answer(body: str, terminator: str) -> tuple[str, str]:
        out, number = _render_structure(printer, body)
        if terminator == "EOF":
            ints, reals = printer.last_fields
            out.append(_echo(number + 1, "EN", ints, reals))
        else:
            out.append(_echo(number + 1, terminator, [0] * 4, [0.0] * 6))
        if terminator != "NX":
            out += _run_time(printer.elapsed())
        _p._decks_rendered += 1
        _p._write_cache_stats()
        return "\n".join(out) + "\n", ""

    return Seam(
        name="nec2d",
        greeting=lambda: [],
        terminators=_p._TERMINATORS,
        closing=frozenset({"EN"}),
        eof_terminator="EOF",
        answer=answer,
    )


def resident_loop(stdin, stdout, stderr, solve_lock=None, scope=None) -> int:
    """Structures off ``stdin`` until it ends, in the NEC-2D layout."""
    the_seam = nec2d_seam() if scope is None else scope(nec2d_seam())
    return run_session(the_seam, stdin, stdout, stderr, solve_lock)


def render(text: str) -> str:
    """A deck file's whole NEC-2D printout."""
    out, err = io.StringIO(), io.StringIO()
    resident_loop(io.StringIO(text, newline=None), out, err)
    if err.getvalue():
        sys.stderr.write(err.getvalue())
        sys.stderr.flush()
    return out.getvalue()


file_printout = render


def file_refusal(text: str | None, reason: str) -> str:
    """The printout for a deck file the shell could not serve at all: the
    banner, the comments it could read, and the refusal, closed as a run."""
    out = [*_banner(), *_COMMENTS, *_comment_lines(text or "")]
    out += _error(_p.PortalError(reason))
    out.append(_echo(1, "EN", [0] * 4, [0.0] * 6))
    out += _run_time(0.0)
    return "\n".join(out) + "\n"
