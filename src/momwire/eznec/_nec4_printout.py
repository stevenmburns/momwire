"""The NEC-4.2 printout, top down: what EZNEC's External NEC-4.2 slot reads.

momwire#1295 phase 2.  A FORMATTER, on :mod:`._printout`'s rule: it is handed
a :class:`Nec4Printout` full of numbers and lays them out in the licensed
engine's columns.  It computes no physics; :mod:`._nec4` solves the deck and
fills the dataclass.

Every heading, column header, field width and blank-line run below was
MEASURED off real NEC-4.2 printouts, used as black-box OUTPUT only: the
seventeen captured decks (``tests/fixtures/eznec_nec42/``) run through the
licensed NEC-4.2 (LLNL-CODE-491368) console build, release 7768648, on
2026-10-03, and kept beside them under ``printouts/``.  No NEC-4.2 source,
algorithm or internal structure is described or relied on, and none may be.

Most of the layout is the NEC-5 printout's, byte for byte: the comment box,
the card echo, FREQUENCY, ANTENNA ENVIRONMENT, the loading column header,
MATRIX TIMING, NETWORK DATA (but one header word), the two port tables'
headers, POWER BUDGET and RADIATION PATTERNS (3-D trailer included).  Those
are rendered by :mod:`._printout`'s own section functions, so a width fixed
there is fixed here.  What is NEC-4.2's own:

* the banner's code name and a blank line 2 (:func:`._printout.render_header`
  with ``dialect="nec4"``);
* STRUCTURE SPECIFICATION as a wire table, the ground-plane notes and the
  junction table, then a SEGMENTATION DATA section NEC-5 does not print;
* no ``ALLOCATE CM:`` line, and a blank line after every loading row;
* the port rows (I6, I6, nine E12.5, no end digit);
* CURRENTS AND LOCATION and CHARGE DENSITIES in upper case with real, imag,
  magnitude and phase columns, and an ``E`` row for the charge at each free
  wire end;
* the admittance-matrix header's ``HTAG``;
* a NETWORK LOSS line whenever the deck carries a network;
* FAR FIELD GROUND PARAMETERS before the pattern when the deck carries ``GD``.

Section order is the solve notes' and the printouts': the fixed six-section
prefix, then (networks), input parameters, currents, charges, power budget,
(far-field ground), pattern.  An ``XQ`` deck stops after the power budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import _printout
from ._printout import (
    GroundMedium,
    LineRow,
    LoadRow,
    NetworkRow,
    PatternBlock,
    PortRow,
    PowerBudget,
    RunData,
)

__all__ = [
    "FarFieldGround",
    "Nec4Printout",
    "SegmentRow",
    "SegmentValueRow",
    "WireRow",
    "render_nec4_printout",
    "value_row",
]


@dataclass(frozen=True)
class WireRow:
    """One ``GW`` card's row of the STRUCTURE SPECIFICATION wire table."""

    number: int
    end1: tuple[float, float, float]
    end2: tuple[float, float, float]
    radius: float
    segments: int
    first: int
    last: int
    tag: int


@dataclass(frozen=True)
class SegmentRow:
    """One SEGMENTATION DATA row: metres and degrees, unnormalised.

    ``i_minus``/``i_plus`` are the connection columns, signed by which end of
    the neighbour lands on this segment, the segment's own number where the
    end stands on a ``GE 1`` ground plane and 0 where it is free.
    """

    number: int
    centre: tuple[float, float, float]
    length: float
    alpha_deg: float
    beta_deg: float
    radius: float
    i_minus: int
    i_plus: int
    tag: int


@dataclass(frozen=True)
class SegmentValueRow:
    """One row of CURRENTS AND LOCATION or CHARGE DENSITIES.

    Lengths in wavelengths.  ``end`` marks the charge table's wire-END row,
    printed as the segment number followed by ``E`` and placed at the end
    point: every captured charge table carries one before the first segment
    of a wire whose start is free and one after the last segment of a wire
    whose end is free, and none at an end on a ``GE 1`` ground plane (0234).
    """

    number: int
    tag: int
    centre: tuple[float, float, float]
    length: float
    real: float
    imag: float
    magnitude: float
    phase_deg: float
    end: bool = False


@dataclass(frozen=True)
class FarFieldGround:
    """The FAR FIELD GROUND PARAMETERS block a ``GD`` card prints.

    ``kind`` is the cliff's name as the engine prints it.  The captured
    ``GD 2,...`` prints ``CIRCLE CLIFF`` (0230, 0234, 0235); no other leading
    integer has been captured.
    """

    kind: str
    edge_distance: float
    height: float
    eps_r2: float
    sigma2: float


@dataclass(frozen=True)
class Nec4Printout:
    """Everything a NEC-4.2 printout prints that the deck text does not."""

    # -- the structure ------------------------------------------------------
    wires: tuple[WireRow, ...]
    ge_flag: int
    total_segments: int
    # One tuple of signed segment numbers per multiple-wire junction.
    junctions: tuple[tuple[int, ...], ...]
    segments: tuple[SegmentRow, ...]
    # -- the operating point ------------------------------------------------
    frequency_mhz: float
    wavelength_m: float
    environment: str
    ground: GroundMedium | None
    loads: tuple[LoadRow, ...]
    fill_seconds: float
    factor_seconds: float
    # -- the run --------------------------------------------------------------
    networks: tuple[NetworkRow | LineRow, ...]
    network_excitation: tuple[PortRow, ...]
    sources: tuple[PortRow, ...]
    currents: tuple[SegmentValueRow, ...] | None
    charges: tuple[SegmentValueRow, ...] | None
    power: PowerBudget
    far_field_ground: FarFieldGround | None
    patterns: tuple[PatternBlock, ...]
    run_seconds: float


# --------------------------------------------------------------------------
# headings and column headers that are NEC-4.2's own
# --------------------------------------------------------------------------

_WIRE_COLUMNS: tuple[str, ...] = (
    "  WIRE" + " " * 79 + "NO. OF    FIRST  LAST     TAG",
    "  NO.        X1         Y1         Z1          X2         Y2         Z2"
    "      RADIUS   SEG.     SEG.   SEG.     NO.",
)

# The plane notes follow the wire table, by GE's sign: GE 1 prints both lines
# (0234, 0235), GE -1 the first alone (0239), GE 0 neither.
_GROUND_PLANE = "   GROUND PLANE SPECIFIED."
_INTERPOLATION = (
    "   WHERE WIRE ENDS TOUCH GROUND, CURRENT WILL BE INTERPOLATED TO IMAGE IN"
    " GROUND PLANE."
)

_JUNCTION_HEADING: tuple[str, ...] = (
    "         - MULTIPLE WIRE JUNCTIONS -",
    " JUNCTION    SEGMENTS  (- FOR END 1, + FOR END 2)",
)
_NO_JUNCTIONS = "  NONE"

_SEGMENTATION_HEADING = " " * 33 + "- - - - SEGMENTATION DATA - - - -"
_SEGMENTATION_NOTES: tuple[str, ...] = (
    " " * 40 + "COORDINATES IN METERS",
    "",
    " " * 25 + "I+ AND I- INDICATE THE SEGMENTS BEFORE AND AFTER I",
)
_SEGMENTATION_COLUMNS: tuple[str, ...] = (
    "  SEG.   COORDINATES OF SEG. CENTER     SEG.     ORIENTATION ANGLES    WIRE"
    "    CONNECTION DATA   TAG",
    "  NO.       X         Y         Z       LENGTH     ALPHA     BETA      RADIUS"
    "    I-   I    I+    NO.",
)

# The admittance-matrix header differs from NEC-5's in one word: ``HTAG``
# where NEC-5 prints ``TAG `` (0234).  The other two lines, and the whole
# transmission-line header, are NEC-5's.
_NT_COLUMNS: tuple[str, ...] = (
    _printout._NETWORK_COLUMNS[0],
    "      HTAG  SEG.   TAG  SEG.             (ONE,ONE)                   "
    "(ONE,TWO)                   (TWO,TWO)",
    _printout._NETWORK_COLUMNS[2],
)

_CURRENTS_HEADING = " " * 29 + "- - - CURRENTS AND LOCATION - - -"
_CURRENTS_NOTE = " " * 21 + "LENGTHS NORMALIZED BY WAVELENGTH (OR 2.*PI/CABS(K))"
_CURRENTS_COLUMNS: tuple[str, ...] = (
    "  SEG.  TAG    COORD. OF SEG. CENTER     SEG.            "
    "- - - CURRENT (AMPS) - - -",
    "  NO.   NO.     X        Y        Z      LENGTH     REAL        IMAG."
    "       MAG.        PHASE",
)
# "NORMALIZED TO", not "BY", and two blanks before the columns, not one.
_CHARGES_HEADING = " " * 34 + "- - - CHARGE DENSITIES - - -"
_CHARGES_NOTE = " " * 23 + "LENGTHS NORMALIZED TO WAVELENGTH (OR 2.*PI/CABS(K))"
_CHARGES_COLUMNS: tuple[str, ...] = (
    "  SEG.  TAG    COORD. OF SEG. CENTER     SEG.          "
    "CHARGE DENSITY (COULOMBS/METER)",
    _CURRENTS_COLUMNS[1],
)

_FAR_GROUND_HEADING = " " * 31 + "- - - FAR FIELD GROUND PARAMETERS - - -"
_FAR_GROUND_INDENT = " " * 40

_SECTION_GAP = _printout._SECTION_GAP
_PATTERN_GAP = _printout._PATTERN_GAP
_STRUCTURE_GAP = _printout._STRUCTURE_GAP


# --------------------------------------------------------------------------
# rows
# --------------------------------------------------------------------------


def _wire_row(row: WireRow) -> str:
    """I6, three F11.5, an F12.5 and two F11.5, the radius F11.5, then the
    segment count I7, FIRST I9, LAST I6 and the tag I8 (0234, 0235)."""
    x1, y1, z1 = row.end1
    x2, y2, z2 = row.end2
    return (
        f"{row.number:6d}{x1:11.5f}{y1:11.5f}{z1:11.5f}"
        f"{x2:12.5f}{y2:11.5f}{z2:11.5f}{row.radius:11.5f}"
        f"{row.segments:7d}{row.first:9d}{row.last:6d}{row.tag:8d}"
    )


def _junction_row(number: int, members: tuple[int, ...]) -> str:
    """NOT MEASURED: no captured deck has a junction (every one prints
    ``NONE``).  The NEC-2 row (I8 junction number, then the signed segment
    numbers) stands in until a printout shows NEC-4.2's own."""
    return f"{number:8d}{members[0]:11d}" + "".join(f"{m:5d}" for m in members[1:])


def _segment_row(row: SegmentRow) -> str:
    """I6, four F10.5 (centre and length), ALPHA F11.5, BETA and RADIUS
    F10.5, then I- I6, I I5, I+ I5 and the tag I7.  A coordinate wide enough
    to fill its field runs into its neighbour (0234's ``614192.92200``)."""
    x, y, z = row.centre
    return (
        f"{row.number:6d}{x:10.5f}{y:10.5f}{z:10.5f}{row.length:10.5f}"
        f"{row.alpha_deg:11.5f}{row.beta_deg:10.5f}{row.radius:10.5f}"
        f"{row.i_minus:6d}{row.number:5d}{row.i_plus:5d}{row.tag:7d}"
    )


def _load_row(row: LoadRow) -> str:
    """One loading row.

    The FIXED IMPEDANCE row is NEC-5's canvas with one difference: a zero
    LOCATION field prints BLANK, so ``LD 4,3,0,0`` (a whole tag) prints its
    tag alone (0234).  The producer resolves an omitted THRU to FROM, which is
    what ``LD 4,1,3,0`` prints (0225).  The WIRE row (0227) is NEC-5's exactly,
    and the series/parallel shapes, which no capture carries, are NEC-5's too.
    """
    if row.kind != "FIXED IMPEDANCE":
        return _printout._load_row(row)
    canvas = [" "] * 118

    def place(text: str, end: int) -> None:
        canvas[end - len(text) : end] = list(text)

    for value, end in ((row.tag, 8), (row.node_from, 13), (row.node_thru, 18)):
        if value:
            place(f"{value:d}", end)
    place(_printout._e(row.resistance, 11, 4), 73)
    if row.reactance is not None:
        place(_printout._e(row.reactance, 11, 4), 86)
    place(row.kind.ljust(16), 118)
    return "".join(canvas)


def _port_row(row: PortRow) -> str:
    """I6, I6, then nine E12.5 cells; a negative cell fills its field and
    runs into its neighbour (``1.89388E-02-1.56495E-02``, 0234).  NEC-4.2's
    row has no end digit after the segment, unlike NEC-5's."""
    cells = (
        row.voltage.real,
        row.voltage.imag,
        row.current.real,
        row.current.imag,
        row.impedance.real,
        row.impedance.imag,
        row.admittance.real,
        row.admittance.imag,
        row.power,
    )
    return f"{row.tag:6d}{row.segment:6d}" + "".join(
        _printout._e(value, 12, 5) for value in cells
    )


def _value_row(row: SegmentValueRow) -> str:
    """I6, the end flag in one column, the tag I4, three F9.4 coordinates,
    the length F9.5, an E13.4 and two E12.4, and the phase F9.3."""
    x, y, z = row.centre
    return (
        f"{row.number:6d}{'E' if row.end else ' '}{row.tag:4d}"
        f"{x:9.4f}{y:9.4f}{z:9.4f}{row.length:9.5f}"
        f"{row.real:13.4E}{row.imag:12.4E}{row.magnitude:12.4E}"
        f"{row.phase_deg:9.3f}"
    )


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------


def _structure(data: Nec4Printout) -> list[str]:
    lines = [_printout._STRUCTURE_HEADING, "", *_printout._COORDINATE_NOTE, "", ""]
    lines += [*_WIRE_COLUMNS, *(_wire_row(w) for w in data.wires), ""]
    if data.ge_flag != 0:
        lines += [_GROUND_PLANE, ""]
    if data.ge_flag > 0:
        lines += [_INTERPOLATION, "", ""]
    total = data.total_segments
    lines += [
        f"   TOTAL SEGMENTS USED={total:5d}     NO. SEG. IN A SYMMETRIC CELL="
        f"{total:5d}     SYMMETRY FLAG={0:3d}",
        "",
        "",
        *_JUNCTION_HEADING,
    ]
    if data.junctions:
        lines += [
            _junction_row(i, members) for i, members in enumerate(data.junctions, 1)
        ]
    else:
        lines.append(_NO_JUNCTIONS)
    return lines


def _segmentation(data: Nec4Printout) -> list[str]:
    return [
        _SEGMENTATION_HEADING,
        "",
        *_SEGMENTATION_NOTES,
        "",
        "",
        *_SEGMENTATION_COLUMNS,
        *(_segment_row(row) for row in data.segments),
    ]


def _loading(data: Nec4Printout) -> list[str]:
    lines = [_printout._LOADING_HEADING, ""]
    if not data.loads:
        return [*lines, _printout._NOT_LOADED]
    lines += ["", *_printout._LOADING_COLUMNS]
    for row in data.loads:
        lines += ["", _load_row(row)]
    return lines


def _network_data(data: Nec4Printout) -> list[str]:
    lines = [_printout._NETWORK_HEADING, ""]
    for index, run in enumerate(_printout._network_runs(data.networks)):
        if index:
            lines.append("")
        columns = (
            _printout._LINE_COLUMNS if isinstance(run[0], LineRow) else _NT_COLUMNS
        )
        lines += [*columns, *(_printout._network_row(row) for row in run)]
    return lines


def _port_table(heading: str, rows: tuple[PortRow, ...]) -> list[str]:
    return [heading, "", *_printout._PORT_COLUMNS, *(_port_row(r) for r in rows)]


def _far_field_ground(ground: FarFieldGround) -> list[str]:
    pad = _FAR_GROUND_INDENT
    return [
        _FAR_GROUND_HEADING,
        "",
        "",
        pad + ground.kind,
        pad + f"EDGE DISTANCE={ground.edge_distance:9.2f} METERS",
        pad + f"HEIGHT={ground.height:8.2f} METERS",
        pad + "SECOND MEDIUM -",
        pad + f"RELATIVE DIELECTRIC CONST.={ground.eps_r2:7.3f}",
        pad + f"CONDUCTIVITY={ground.sigma2:10.3E} MHOS",
    ]


def render_nec4_printout(
    deck_text: str, data: Nec4Printout, *, basis: str | None = None
) -> str:
    """A complete NEC-4.2 printout: the header, then everything to ``RUN TIME``.

    ``deck_text`` supplies what the printout ECHOES (the comment box and the
    post-``GE`` card images, ``EN`` last and after the results, exactly as in
    NEC-5's); ``data`` every number the run produced.  ``basis`` reaches line
    2's engine stamp and nothing else.
    """
    shared = RunData(
        frequency_mhz=data.frequency_mhz,
        wavelength_m=data.wavelength_m,
        environment=data.environment,
        ground=data.ground,
        fill_seconds=data.fill_seconds,
        factor_seconds=data.factor_seconds,
    )
    cards = _printout._post_ge_cards(deck_text)
    terminator = cards[-1] if cards and cards[-1].mnemonic == "EN" else None
    gap = _printout._blank
    body: list[str] = []
    body += _structure(data)
    body += gap(_STRUCTURE_GAP)
    body += _segmentation(data)
    body += gap(_STRUCTURE_GAP)
    body += [
        _printout._card_echo(index, card)
        for index, card in enumerate(cards, start=1)
        if card is not terminator
    ]
    body += gap(_STRUCTURE_GAP)
    body += _printout._frequency(shared)
    body += gap(_SECTION_GAP)
    body += _printout._environment(shared)
    body += gap(_SECTION_GAP)
    body += _loading(data)
    body += gap(_SECTION_GAP)
    body += _printout._timing(shared)
    body += gap(_SECTION_GAP)
    if data.networks:
        body += _network_data(data)
        body += gap(_SECTION_GAP)
        body += _port_table(
            _printout._NETWORK_EXCITATION_HEADING, data.network_excitation
        )
        body += gap(_SECTION_GAP)
    body += _port_table(_printout._ANTENNA_INPUT_HEADING, data.sources)
    body += gap(_SECTION_GAP)
    if data.currents is not None:
        body += [_CURRENTS_HEADING, "", _CURRENTS_NOTE, "", *_CURRENTS_COLUMNS]
        body += [_value_row(row) for row in data.currents]
        body += gap(_SECTION_GAP)
    if data.charges is not None:
        body += [_CHARGES_HEADING, "", _CHARGES_NOTE, "", "", *_CHARGES_COLUMNS]
        body += [_value_row(row) for row in data.charges]
        body += gap(_SECTION_GAP)
    body += _printout._power_budget(data.power)
    body += gap(_SECTION_GAP)
    # Every GD capture asks for a pattern, so the block is printed only in
    # front of one; whether an XQ deck prints it too is unmeasured.
    if data.patterns and data.far_field_ground is not None:
        body += _far_field_ground(data.far_field_ground)
        body += gap(_SECTION_GAP)
    for block in data.patterns:
        body += _printout._pattern(block)
        body += gap(_PATTERN_GAP)
    body.append("")
    if terminator is not None:
        body.append(_printout._card_echo(len(cards), terminator))
    body += ["", f"{_printout._RUN_TIME_LABEL}{data.run_seconds:10.3f}"]
    header = _printout.render_header(deck_text, basis=basis, dialect="nec4")
    return header + "\n".join(body) + "\n"


def value_row(
    number: int,
    tag: int,
    centre: tuple[float, float, float],
    length: float,
    value: complex,
    *,
    end: bool = False,
) -> SegmentValueRow:
    """A :class:`SegmentValueRow` from one complex value: the four printed
    numbers are its real part, imaginary part, magnitude and phase."""
    magnitude = abs(value)
    phase = math.degrees(math.atan2(value.imag, value.real)) if magnitude else 0.0
    return SegmentValueRow(
        number,
        tag,
        centre,
        length,
        value.real,
        value.imag,
        magnitude,
        phase,
        end,
    )
