"""EZNEC's External NEC-4.2 slot, phase 2: the NEC-4.2 printout.

momwire#1295.  The oracle is the licensed NEC-4.2's own printout of each of
the seventeen captured decks and of nine probe decks written to show what
the captures do not (junctions, an FR sweep, RLC loads, NE, two runs, ...):
``tests/fixtures/eznec_nec42/printouts/``, README there for provenance,
used as black-box OUTPUT only.

Three kinds of gate, cheapest first:

* **round trip** (default lane, no solve): every licensed printout is read
  back into a :class:`~momwire.eznec._nec4_printout.Nec4Printout` by fixed
  columns and re-rendered; the result must be the licensed bytes, line 2 (the
  engine stamp) and the Sommerfeld-table cache lines aside.  A width, a blank
  run or a heading that is wrong anywhere is a failing test here.
* **served layout** (integration): each deck SOLVED and printed by this
  engine against the licensed printout — the same headings in the same order,
  the same number of lines, and line for line the same column structure
  (every literal character equal, every number in the same field and format).
* **one reader** (integration): a heading-keyed reader, written the way a
  consumer such as EZNEC must read a NEC printout, pulls the same set of
  fields out of both printouts.  The numbers themselves are REPORTED (``-s``
  prints the table), and held only to a loose sanity bar: momwire's bases are
  not NEC-4.2's, and the cross-engine gap is not a contract.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from momwire.eznec import _nec4, _printout
from momwire.eznec._nec4_printout import (
    FarFieldGround,
    Nec4Printout,
    Nec4Run,
    Preamble,
    SegmentRow,
    SegmentValueRow,
    WireRow,
    render_nec4_printout,
)
from momwire.eznec._printout import (
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
)
from momwire.eznec._shell import render

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "eznec_nec42"
PRINTOUTS = FIXTURES / "printouts"
PROBES_DIR = FIXTURES / "probes"
MULTI_DIR = FIXTURES / "multisource"
# The seventeen decks EZNEC wrote, and nine probe decks written to show
# NEC-4.2's layout of what EZNEC's captures did not (README beside them).
CAPTURES = tuple(sorted(p.name[:4] for p in FIXTURES.glob("*.nec")))
PROBES = tuple(sorted(p.name.split("_")[0] for p in PROBES_DIR.glob("*.nec")))
# Two decks EZNEC's NEC-4.2 slot wrote with SEVERAL current sources, the
# phased drive (README there): the stock Cardioid and a 40 m four-square.
MULTI = tuple(sorted(p.name.split("_")[0] for p in MULTI_DIR.glob("*.nec")))
CASES = CAPTURES + PROBES + MULTI
BASIS = "bspline"


def deck_text(capture: str) -> str:
    (path,) = [
        *FIXTURES.glob(f"{capture}_*.nec"),
        *PROBES_DIR.glob(f"{capture}_*.nec"),
        *MULTI_DIR.glob(f"{capture}_*.nec"),
    ]
    return path.read_bytes().decode("latin-1")


def licensed(capture: str) -> str:
    (path,) = PRINTOUTS.glob(f"{capture}_*.out")
    return path.read_bytes().decode("latin-1").replace("\r\n", "\n")


# The Sommerfeld-table cache messages: the licensed engine reports that it
# found no table file and computed one.  This engine keeps no such file and
# prints no such lines (the NEC-5 seam's rule for its SOMMPD.NEX block), so
# each block leaves the reference together with the blank line it brings:
# the GN 2 block sits between the heading and the medium (0231), the GN 3
# block after the medium (0232, 0239), and a GN 3 run that found tables
# from the previous run says so above RUN TIME (0239).  The one exception is
# a GN 2 that NAMES its file, which no capture carries: the block's first line
# is kept for it (momwire#1317, gated below).
_CACHE_BLOCKS = (
    re.compile(
        r"\n\n GNDINO: [^\n]*\n WILL COMPUTE SOMMERFELD-GROUND TABLES\n"
        r"\n Time to generate Sommerfeld ground tables =[^\n]*\n"
    ),
    re.compile(r"\n\n GNDINO3: [^\n]*\n Will compute Sommerfeld-ground tables(?=\n)"),
    re.compile(r"\n GN3 tables written in previous run:[^\n]*\n(?=\n RUN TIME)"),
)


def reference(capture: str) -> str:
    text = licensed(capture)
    for block in _CACHE_BLOCKS:
        text = block.sub("", text)
    return text


def _unstamped(text: str) -> list[str]:
    """Lines, line 2 blanked: the licensed engine's is empty, ours carries
    this engine's stamp, and every byte gate masks it by position."""
    lines = text.split("\n")
    assert lines[0] == "1"
    lines[1] = ""
    return lines


def test_the_fixtures_pair_each_deck_with_its_printout():
    """One printout per deck, and each echoes its own deck's comment block —
    the stamp EZNEC checks a printout's age by."""
    assert len(CAPTURES) == 17 and len(PROBES) == 10 and MULTI == ("m1", "m2")
    assert sorted(p.name.split("_")[0] for p in PRINTOUTS.glob("*.out")) == sorted(
        CASES
    )
    for capture in CASES:
        box = _printout._comment_box(deck_text(capture))
        lines = licensed(capture).split("\n")
        start = lines.index(box[0])
        assert lines[start : start + len(box)] == box, capture
        assert "NUMERICAL ELECTROMAGNETICS CODE (NEC-4.2)" in lines[6], capture


def test_the_cache_blocks_leave_only_their_own_lines():
    for capture in ("0231", "0232", "0239", "p7", "p9"):
        raw, clean = licensed(capture), reference(capture)
        assert "GNDINO" in raw and "GNDINO" not in clean
        assert "ommerfeld-ground tables" not in clean
        assert (
            len(raw.split("\n")) - len(clean.split("\n"))
            == {
                "0231": 6,
                "0232": 3,
                "0239": 5,
                "p7": 6,
                "p9": 6,
            }[capture]
        )


# ==========================================================================
# reading a printout back, by fixed columns
# ==========================================================================


def _cells(line: str, edges: tuple[int, ...]) -> list[str]:
    starts = (0, *edges[:-1])
    return [line[a:b] for a, b in zip(starts, edges, strict=True)]


def _section(lines: list[str], heading: str) -> int:
    return next(i for i, ln in enumerate(lines) if ln.strip() == heading)


def _rows_after(lines: list[str], start: int) -> list[str]:
    """The non-blank lines from ``start`` to the next blank."""
    out = []
    for line in lines[start:]:
        if not line.strip():
            break
        out.append(line)
    return out


_WIRE_EDGES = (6, 17, 28, 39, 51, 62, 73, 84, 91, 100, 106, 114)
_SEGMENT_EDGES = (6, 16, 26, 36, 46, 57, 67, 77, 83, 88, 93, 100)
_PORT_EDGES = (6, 12, *range(24, 121, 12))
_VALUE_EDGES = (6, 7, 11, 20, 29, 38, 47, 60, 72, 84, 93)
_NETWORK_EDGES = (9, 15, 21, 27, *range(42, 113, 14))
_PATTERN_EDGES = (8, 17, 28, 36, 44, 55, 64, 66, 72, 87, 96, 111, 120)


def _port_rows(lines: list[str], heading: str) -> tuple[PortRow, ...]:
    start = _section(lines, heading)
    rows = []
    for line in _rows_after(lines, start + 4):
        tag, seg, *v = _cells(line, _PORT_EDGES)
        v = [float(x) for x in v]
        rows.append(
            PortRow(
                int(tag),
                int(seg),
                0,
                complex(v[0], v[1]),
                complex(v[2], v[3]),
                complex(v[4], v[5]),
                complex(v[6], v[7]),
                v[8],
            )
        )
    return tuple(rows)


def _value_rows(lines: list[str], start: int) -> tuple[SegmentValueRow, ...]:
    rows = []
    for line in _rows_after(lines, start):
        number, flag, tag, x, y, z, length, re_, im, mag, ph = _cells(
            line, _VALUE_EDGES
        )
        rows.append(
            SegmentValueRow(
                int(number),
                int(tag),
                (float(x), float(y), float(z)),
                float(length),
                float(re_),
                float(im),
                float(mag),
                float(ph),
                flag == "E",
            )
        )
    return tuple(rows)


def _load_rows(lines: list[str]) -> tuple[LoadRow, ...]:
    start = _section(lines, "- - - STRUCTURE IMPEDANCE LOADING - - -")
    rows = []
    for line in lines[start + 2 :]:
        if "MATRIX TIMING" in line:
            break
        if not line.strip() or "LOCATION" in line or "ITAG" in line:
            continue
        if "NOT LOADED" in line:
            break
        tag, first, last = (
            int(c) if c.strip() else 0 for c in _cells(line, (8, 13, 18))
        )
        kind = line[100:].strip()
        if kind in ("SERIES", "PARALLEL"):
            r, ind, c = (line[a:b].strip() for a, b in ((23, 34), (36, 47), (49, 60)))
            rows.append(
                LoadRow(
                    tag,
                    first,
                    last,
                    resistance=float(r) if r else None,
                    inductance=float(ind) if ind else None,
                    capacitance=float(c) if c else None,
                    kind=kind,
                )
            )
            continue
        if line.rstrip().endswith("WIRE"):
            rows.append(
                LoadRow(tag, first, last, conductivity=float(line[88:99]), kind="WIRE")
            )
            continue
        reactance = line[75:86].strip()
        rows.append(
            LoadRow(
                tag,
                first,
                last,
                resistance=float(line[62:73]),
                reactance=float(reactance) if reactance else None,
            )
        )
    return tuple(rows)


def _networks(lines: list[str]) -> tuple[NetworkRow | LineRow, ...]:
    try:
        start = _section(lines, "- - - NETWORK DATA - - -")
    except StopIteration:
        return ()
    end = _section(
        lines, "- - - STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS - - -"
    )
    rows: list[NetworkRow | LineRow] = []
    for line in lines[start + 1 : end]:
        if not line.strip() or not line.strip()[0].isdigit():
            continue
        cells = _cells(line.ljust(112), _NETWORK_EDGES)
        ends = [int(c) for c in cells[:4]]
        v = [float(c) for c in cells[4:]]
        tail = line[112:].strip()
        if tail:
            rows.append(
                LineRow(
                    *ends,
                    z0=v[0],
                    length_m=v[1],
                    shunt_a=complex(v[2], v[3]),
                    shunt_b=complex(v[4], v[5]),
                    crossed=tail == "CROSSED",
                )
            )
        else:
            rows.append(
                NetworkRow(
                    *ends, complex(v[0], v[1]), complex(v[2], v[3]), complex(v[4], v[5])
                )
            )
    return tuple(rows)


def _pattern(lines: list[str]) -> tuple[PatternBlock, ...]:
    try:
        start = _section(lines, "- - - RADIATION PATTERNS - - -")
    except StopIteration:
        return ()
    rows = []
    for line in _rows_after(lines, start + 5):
        c = _cells(line, _PATTERN_EDGES)
        rows.append(
            PatternRow(
                float(c[0]),
                float(c[1]),
                float(c[2]),
                float(c[3]),
                float(c[4]),
                float(c[5]),
                float(c[6]),
                c[8].strip(),
                float(c[9]),
                float(c[10]),
                float(c[11]),
                float(c[12]),
            )
        )
    trailer = {}
    after = start + 5 + len(rows)
    if "AVERAGE POWER GAIN" in lines[after + 2]:
        nums = re.findall(r"[-+]?\d+\.\d+(?:E[-+]\d+)?", lines[after + 2])
        trailer = {
            "average_power_gain": float(nums[0]),
            "solid_angle_pi": float(nums[1]),
            "power_radiated_4pi": float(
                re.findall(r"\d\.\d+E[-+]\d+", lines[after + 4])[0]
            ),
        }
    return (PatternBlock(rows=tuple(rows), **trailer),)


def _number_after(line: str, label: str) -> float:
    return float(line.split(label, 1)[1].split()[0])


def _near_fields(lines: list[str]) -> tuple[NearFieldBlock, ...]:
    blocks = []
    for at, heading in enumerate(lines):
        if "- - - NEAR " not in heading:
            continue
        rows = []
        for line in _rows_after(lines, at + 5):
            c = _cells(line, (14, 26, 38, 53, 62, 76, 85, 99, 108))
            rows.append(
                NearFieldRow(
                    point=(float(c[0]), float(c[1]), float(c[2])),
                    magnitudes=(float(c[3]), float(c[5]), float(c[7])),
                    phases_deg=(float(c[4]), float(c[6]), float(c[8])),
                )
            )
        blocks.append(NearFieldBlock(tuple(rows), magnetic="MAGNETIC" in heading))
    return tuple(blocks)


_ECHO = " ***** INPUT LINE"


def _parse_run(lines: list[str], echo_through: int | None) -> Nec4Run:
    """One run's region of a printout: its sections found by heading."""
    preamble = None
    if any("- - - - - - FREQUENCY - - - - - -" in ln for ln in lines):
        frequency = next(ln for ln in lines if "FREQUENCY=" in ln)
        wavelength = next(ln for ln in lines if "WAVELENGTH=" in ln)
        env_at = _section(lines, "- - - ANTENNA ENVIRONMENT - - -")
        environment = lines[env_at + 2].strip()
        ground = None
        if environment.startswith("FINITE"):
            eps = _number_after(lines[env_at + 3], "CONST.=")
            sigma = _number_after(lines[env_at + 4], "CONDUCTIVITY=")
            cell = lines[env_at + 5].split("CONSTANT=")[1]
            ground = GroundMedium(
                eps, sigma, complex(float(cell[:12]), float(cell[12:24]))
            )
        fill = next(ln for ln in lines if "FILL=" in ln)
        preamble = Preamble(
            frequency_mhz=_number_after(frequency, "FREQUENCY="),
            wavelength_m=_number_after(wavelength, "WAVELENGTH="),
            environment=environment,
            ground=ground,
            loads=_load_rows(lines),
            fill_seconds=_number_after(fill, "FILL="),
            factor_seconds=_number_after(fill, "FACTOR="),
        )
    excitation_heading = (
        "- - - STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS - - -"
    )
    has_excitation = any(ln.strip() == excitation_heading for ln in lines)
    try:
        charges = _value_rows(
            lines, _section(lines, "- - - CHARGE DENSITIES - - -") + 7
        )
    except StopIteration:
        charges = None
    power_at = _section(lines, "- - - POWER BUDGET - - -")
    budget = {}
    for line in _rows_after(lines, power_at + 2):
        label, value = line.split("=")
        budget[label.strip()] = float(value.split()[0])
    try:
        ff_at = _section(lines, "- - - FAR FIELD GROUND PARAMETERS - - -")
        far = FarFieldGround(
            lines[ff_at + 3].strip(),
            _number_after(lines[ff_at + 4], "DISTANCE="),
            _number_after(lines[ff_at + 5], "HEIGHT="),
            _number_after(lines[ff_at + 7], "CONST.="),
            _number_after(lines[ff_at + 8], "CONDUCTIVITY="),
        )
    except StopIteration:
        far = None
    return Nec4Run(
        echo_through=echo_through,
        preamble=preamble,
        networks=_networks(lines),
        network_excitation=(
            _port_rows(lines, excitation_heading) if has_excitation else ()
        ),
        sources=_port_rows(lines, "- - - ANTENNA INPUT PARAMETERS - - -"),
        currents=_value_rows(
            lines, _section(lines, "- - - CURRENTS AND LOCATION - - -") + 6
        ),
        charges=charges,
        power=PowerBudget(
            budget["INPUT POWER"],
            budget["RADIATED POWER"],
            budget["WIRE LOSS"],
            budget["EFFICIENCY"],
            budget.get("NETWORK LOSS"),
        ),
        near_fields=_near_fields(lines),
        far_field_ground=far,
        patterns=_pattern(lines),
    )


def parse_printout(deck: str, text: str) -> Nec4Printout:
    """A licensed NEC-4.2 printout read back into the dataclass it would be
    rendered from: every number by its column, nothing recomputed.

    A run's region starts at its card echo or its FREQUENCY heading,
    whichever comes first after the previous run's input parameters, and
    holds exactly one ANTENNA INPUT PARAMETERS table.
    """
    lines = text.split("\n")
    wires_at = next(i for i, ln in enumerate(lines) if ln.startswith("  NO.        X1"))
    wires = []
    for line in _rows_after(lines, wires_at + 1):
        c = _cells(line, _WIRE_EDGES)
        wires.append(
            WireRow(
                int(c[0]),
                (float(c[1]), float(c[2]), float(c[3])),
                (float(c[4]), float(c[5]), float(c[6])),
                float(c[7]),
                int(c[8]),
                int(c[9]),
                int(c[10]),
                int(c[11]),
            )
        )
    total = next(
        int(ln.split("=")[1].split()[0]) for ln in lines if "TOTAL SEGMENTS USED=" in ln
    )
    junction_at = next(i for i, ln in enumerate(lines) if ln.startswith(" JUNCTION"))
    junctions = tuple(
        tuple(int(n) for n in line.split()[1:])
        for line in _rows_after(lines, junction_at + 1)
        if line != "  NONE"
    )
    seg_at = next(i for i, ln in enumerate(lines) if ln.startswith("  NO.       X"))
    segments = []
    for line in _rows_after(lines, seg_at + 1):
        c = _cells(line, _SEGMENT_EDGES)
        segments.append(
            SegmentRow(
                int(c[0]),
                (float(c[1]), float(c[2]), float(c[3])),
                float(c[4]),
                float(c[5]),
                float(c[6]),
                float(c[7]),
                int(c[8]),
                int(c[10]),
                int(c[11]),
            )
        )
    ge = next(ln for ln in deck.splitlines() if ln.startswith("GE"))

    def echo_number(line: str) -> int | None:
        if not line.startswith(_ECHO) or line[22:24] == "EN":
            return None
        return int(line[len(_ECHO) : len(_ECHO) + 3])

    inputs = [
        i
        for i, ln in enumerate(lines)
        if ln.strip() == "- - - ANTENNA INPUT PARAMETERS - - -"
    ]
    starts = [seg_at]
    for previous in inputs[:-1]:
        starts.append(
            next(
                i
                for i in range(previous + 1, len(lines))
                if echo_number(lines[i]) is not None
                or "- - - - - - FREQUENCY - - - - - -" in lines[i]
            )
        )
    starts.append(len(lines))
    runs = []
    for k in range(len(inputs)):
        region = lines[starts[k] : starts[k + 1]]
        echoed = [n for ln in region if (n := echo_number(ln)) is not None]
        runs.append(_parse_run(region, echoed[-1] if echoed else None))
    run_time = next(ln for ln in lines if "RUN TIME =" in ln)
    return Nec4Printout(
        wires=tuple(wires),
        ge_flag=int(ge[2:].split(",")[0]),
        total_segments=total,
        junctions=junctions,
        segments=tuple(segments),
        runs=tuple(runs),
        run_seconds=_number_after(run_time, "RUN TIME ="),
    )


@pytest.mark.parametrize("capture", CASES)
def test_the_licensed_printout_round_trips_through_the_renderer(capture):
    """The formatter gate: NEC-4.2's own numbers, laid out by this module,
    are NEC-4.2's own bytes."""
    deck = deck_text(capture)
    want = reference(capture)
    got = render_nec4_printout(deck, parse_printout(deck, want), basis=BASIS)
    assert _unstamped(got) == _unstamped(want)


def test_the_nec5_header_is_untouched():
    """The banner became a function of the slot; NEC-5's bytes did not move."""
    deck = deck_text("0223")
    nec5 = render_header_lines(deck, "nec5")
    assert nec5[1].startswith(" momwire ")
    assert nec5[2:5] == ["", "", ""]
    assert nec5[7] == " " * 32 + "*  NUMERICAL ELECTROMAGNETICS CODE (NEC-5)    *"
    nec4 = render_header_lines(deck, "nec4")
    assert nec4[2:4] == ["", ""]
    assert nec4[6] == " " * 32 + "*  NUMERICAL ELECTROMAGNETICS CODE (NEC-4.2)  *"
    assert len(nec4) == len(nec5) - 1


def render_header_lines(deck: str, dialect: str) -> list[str]:
    return _printout.render_header(deck, basis=BASIS, dialect=dialect).split("\n")


# ==========================================================================
# the served printout against the licensed one
# ==========================================================================

_HEADING = re.compile(r"^ +(- )+-? ?[A-Z][A-Z .()]*[A-Z]( -)+ *$")
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:E[-+]\d+)?")
_PATTERN_ROW = re.compile(r"^ *-?\d+\.\d\d +-?\d+\.\d\d +-?\d+\.\d\d ")


def headings(text: str) -> list[str]:
    return [ln.strip() for ln in text.split("\n") if _HEADING.match(ln)]


def _kind(token: str) -> str:
    mantissa, _, exponent = token.partition("E")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else -1
    return f"{'E' if exponent else 'F'}{decimals}"


def skeleton(line: str) -> tuple[str, tuple[tuple[int, str], ...]]:
    """A line's column structure: its literal text with every number blanked
    in place, and each number's END COLUMN and format (``F4``, ``E5``, ``F-1``
    for an integer).  Two lines with equal skeletons print the same fields,
    right-justified to the same columns, among the same literal characters;
    only the digits (and so a number's own width inside its field) may
    differ.  A pattern row's SENSE column is a value
    (``LINEAR``/``LEFT``/``RIGHT``/blank), so it is masked."""
    if _PATTERN_ROW.match(line) and len(line) >= 72:
        line = line[:66] + "~" * 6 + line[72:]
    fields = tuple((m.end(), _kind(m.group())) for m in _NUMBER.finditer(line))
    literal = _NUMBER.sub(lambda m: " " * len(m.group()), line)
    return literal.rstrip(), fields


# The slot's default basis and its point-matched twin, the launcher a
# NEC-4.2 user compares against first (``momwire-nec4-sinusoidal.exe``).
SERVED_BASES = ("bspline", "sinusoidal")


@pytest.fixture(scope="module", params=SERVED_BASES)
def served(request) -> dict[str, str]:
    basis = request.param
    texts = {c: render(deck_text(c), basis=basis, dialect="nec4") for c in CASES}
    texts["basis"] = basis
    return texts


@pytest.mark.integration
@pytest.mark.parametrize("capture", CASES)
def test_the_served_printout_has_the_licensed_sections_in_order(served, capture):
    ours = served[capture]
    assert "NEC ERROR" not in ours, ours[-400:]
    assert headings(ours) == headings(reference(capture))


@pytest.mark.integration
@pytest.mark.parametrize("capture", CASES)
def test_the_served_printout_has_the_licensed_columns_line_for_line(served, capture):
    ours, theirs = _unstamped(served[capture]), _unstamped(reference(capture))
    assert len(ours) == len(theirs)
    for number, (a, b) in enumerate(zip(ours, theirs, strict=True), start=1):
        assert skeleton(a) == skeleton(b), (number, a, b)


# --------------------------------------------------------------------------
# one reader for both


def read_fields(text: str) -> dict:
    """What a heading-keyed consumer reads off a NEC printout.

    Sections are found by heading, never by position (TL/NT move the input
    parameters later in the file), and a row is the numbers on it in order,
    which reads a negative cell glued to its neighbour correctly.  Kept
    deliberately independent of the fixed-column reader above.
    """
    lines = text.split("\n")

    def table(heading: str, width: int) -> list[list[float]]:
        rows = []
        for at in (i for i, ln in enumerate(lines) if ln.strip() == heading):
            started = False
            for line in lines[at + 1 :]:
                numbers = [float(n) for n in _NUMBER.findall(line)]
                if len(numbers) == width:
                    rows.append(numbers)
                    started = True
                elif started and not line.strip():
                    break
        return rows

    return {
        "frequency": [
            float(_NUMBER.findall(ln)[0]) for ln in lines if "FREQUENCY=" in ln
        ],
        "input": table("- - - ANTENNA INPUT PARAMETERS - - -", 11),
        "currents": table("- - - CURRENTS AND LOCATION - - -", 10),
        "charges": table("- - - CHARGE DENSITIES - - -", 10),
        "pattern": table("- - - RADIATION PATTERNS - - -", 11),
        "near": table("- - - NEAR ELECTRIC FIELDS - - -", 9),
        "budget": [
            ln.split("=")[0].strip()
            for ln in lines
            if ln.startswith(" " * 43) and "=" in ln
        ],
    }


@pytest.mark.integration
@pytest.mark.parametrize("capture", CASES)
def test_one_reader_finds_the_same_fields_in_both(served, capture):
    ours, theirs = read_fields(served[capture]), read_fields(reference(capture))
    assert ours.keys() == theirs.keys()
    for key in ours:
        if key == "budget":
            assert ours[key] == theirs[key]
            continue
        assert len(ours[key]) == len(theirs[key]), key
        # Addresses (tag, segment; theta, phi) are the same rows in both.
        if key == "input":
            assert [r[:2] for r in ours[key]] == [r[:2] for r in theirs[key]]
        if key in ("currents", "charges"):
            assert [r[:2] for r in ours[key]] == [r[:2] for r in theirs[key]]
        if key == "pattern":
            assert [r[:2] for r in ours[key]] == [r[:2] for r in theirs[key]]


# --------------------------------------------------------------------------
# the numbers, reported

# The phase-1 bar and its two exemptions (test_eznec_nec42_1295.py): 0233 is
# all reactance, 0239 an unconverged buried mesh.
_SANITY_REL = 0.05
_SANITY_EXEMPT = {"0233", "0239"}


@pytest.mark.integration
def test_the_numbers_against_the_licensed_engine(served, capsys):
    report = []
    for capture in CASES:
        ours, theirs = read_fields(served[capture]), read_fields(reference(capture))
        for run, (a, b) in enumerate(zip(ours["input"], theirs["input"], strict=True)):
            label = capture if run == 0 else f"{capture}#{run + 1}"
            report.append((label, complex(*b[6:8]), complex(*a[6:8]), None, None))
        z_ours = complex(*ours["input"][0][6:8])
        z_theirs = complex(*theirs["input"][0][6:8])
        peak = max(abs(complex(*r[6:8])) for r in theirs["currents"])
        i_rel = (
            max(
                abs(complex(*a[6:8]) - complex(*b[6:8]))
                for a, b in zip(ours["currents"], theirs["currents"], strict=True)
            )
            / peak
        )
        gain = [
            abs(a[4] - b[4])
            for a, b in zip(ours["pattern"], theirs["pattern"], strict=True)
            if b[4] > -100.0
        ]
        report[-len(ours["input"])] = (
            capture,
            z_theirs,
            z_ours,
            i_rel,
            max(gain, default=None),
        )
    with capsys.disabled():
        print(
            f"\n{served['basis']}: capture, NEC-4.2 Z, momwire Z, |dZ|/|Z|, "
            "max|dI|/peak, max|dG| dB"
        )
        for capture, zt, zo, i_rel, dg in report:
            rel = abs(zo - zt) / abs(zt)
            g = "-" if dg is None else f"{dg:.2f}"
            i = "-" if i_rel is None else f"{i_rel:.4f}"
            print(f"  {capture}  {zt:.4f}  {zo:.4f}  {rel:.4f}  {i}  {g}")
    # The bar holds the captures, the decks EZNEC writes; the probes are
    # layout probes and are reported only.
    for capture, zt, zo, _i_rel, _dg in report:
        assert math.isfinite(zo.real) and math.isfinite(zo.imag), capture
        if capture in CAPTURES and capture not in _SANITY_EXEMPT:
            assert abs(zo - zt) / abs(zt) < _SANITY_REL, (capture, zo, zt)


# The licensed NEC-4.2's own ANTENNA INPUT PARAMETERS rows for the two
# phased decks, (tag, segment, Z) per source: the targets the README quotes.
MULTI_Z = {
    "m1": ((1, 1, 21.0326 - 18.7112j), (2, 7, 51.6136 + 20.8613j)),
    "m2": (
        (1, 1, -1.38797 - 20.3048j),
        (2, 7, 40.9235 - 21.9351j),
        (3, 13, 40.9235 - 21.9351j),
        (4, 19, 58.7204 + 53.3152j),
    ),
}


def test_the_phased_targets_are_the_licensed_printouts_own_rows():
    for case, rows in MULTI_Z.items():
        theirs = read_fields(reference(case))["input"]
        assert [(int(r[0]), int(r[1])) for r in theirs] == [r[:2] for r in rows]
        for row, (_t, _s, z) in zip(theirs, rows, strict=True):
            assert complex(*row[6:8]) == z, case


@pytest.mark.integration
def test_a_phased_drive_against_the_licensed_engine(served, capsys):
    """Every source of both phased decks: the card's current printed as set,
    and the impedance within the slot's sanity bar of NEC-4.2's.  The
    four-square's source 1 has a NEGATIVE resistance in NEC-4.2's answer (a
    driven element absorbing power from its neighbours) and must keep its
    sign here: nothing clamps it.  Its sources 2 and 3 are mirror images of
    each other and must print identically."""
    lines = []
    for case, rows in MULTI_Z.items():
        ours = read_fields(served[case])["input"]
        assert [(int(r[0]), int(r[1])) for r in ours] == [r[:2] for r in rows]
        theirs = read_fields(reference(case))["input"]
        for a, b, (_t, seg, z) in zip(ours, theirs, rows, strict=True):
            # The current columns are the card's set current, as printed.
            assert a[4:6] == b[4:6], (case, seg)
            got = complex(*a[6:8])
            rel = abs(got - z) / abs(z)
            lines.append(f"  {case} seg {seg:>2}  {z:.4f}  {got:.4f}  {rel:.4f}")
            assert rel < _SANITY_REL, (case, seg, got, z)
    with capsys.disabled():
        print(f"\n{served['basis']}: phased, NEC-4.2 Z, momwire Z, |dZ|/|Z|")
        print("\n".join(lines))
    square = read_fields(served["m2"])["input"]
    assert square[0][6] < 0.0
    assert square[1][2:] == square[2][2:]


# ==========================================================================
# a GN 2 naming a Sommerfeld-table file (momwire#1317)
# ==========================================================================


def _timeless(text: str) -> list[str]:
    """Lines with the stamp and the two clock readings blanked."""
    lines = _unstamped(text)
    return [
        "" if ("FILL=" in ln or ln.startswith(_printout._RUN_TIME_LABEL)) else ln
        for ln in lines
    ]


def test_a_gn2_file_name_prints_that_the_file_was_not_read():
    """The licensed engine's line, in its captured place (the cache block of
    p7, this deck's own printout, first line only), and the name echoed on
    the card's continuation line; every other line is the bare card's
    printout."""
    bare_card = "GN 2 0 0 0 13. .005 0 0 0 0"
    bare = render(
        _edited("p7", "GN 2,0,0,0,13.,.005", bare_card), basis=BASIS, dialect="nec4"
    )
    assert "GNDINO" not in bare
    for name in ("NOFILE", "SOMEX10.NEC"):
        named = render(
            _edited("p7", "GN 2,0,0,0,13.,.005", f"{bare_card} {name}"),
            basis=BASIS,
            dialect="nec4",
        )
        assert "NEC ERROR" not in named, name
        gndino = " GNDINO: UNABLE TO OPEN FILE " + name.ljust(40)
        echo = " " * 43 + "  0.00000E+00  " + name.ljust(40)
        lines = _timeless(named)
        env_at = _section(lines, "- - - ANTENNA ENVIRONMENT - - -")
        assert lines[env_at + 1 : env_at + 6] == [
            "",
            "",
            gndino,
            "",
            " " * 40 + "FINITE GROUND.  SOMMERFELD SOLUTION GN2",
        ], name
        assert lines.count(gndino) == 1 and lines.count(echo) == 1, name
        kept = [*lines[: env_at + 2], *lines[env_at + 5 :]]
        kept.remove(echo)
        assert kept == _timeless(bare), name


# ==========================================================================
# refusals
# ==========================================================================


def _edited(capture: str, old: str, new: str) -> str:
    text = deck_text(capture)
    assert old in text
    return text.replace(old, new)


@pytest.mark.parametrize(
    "text, match",
    [
        pytest.param(
            _edited(
                "0233",
                "FR 0,1,0,0,13.9\r\nGN -1\r\nEX 0,1,6,0,1.414214,0.\r\nPQ 0\r\nXQ 0",
                "FR 0,2,0,0,13.9,1.\r\nGN -1\r\nEX 0,1,6,0,1.414214,0.\r\nPQ 0"
                "\r\nXQ 0\r\nEX 0,1,5,0,1.414214,0.\r\nXQ 0",
            ),
            "several execute cards over a frequency sweep",
            id="sweep-and-two-runs",
        ),
        pytest.param(
            _edited("0233", "XQ 0", "XQ 0\r\nFR 0,1,0,0,14.,0.\r\nXQ 0"),
            "a later execute card after a card that refills the matrix",
            id="refill-between-runs",
        ),
        pytest.param(
            _edited(
                "0231",
                "RP 0,1,361,1000,75.,0.,0.,1.,0.",
                "NE 0,1,1,1,1.,0.,1.,0.,0.,0.",
            ),
            "NE over a finite ground",
            id="near-field-finite-ground",
        ),
        pytest.param(
            _edited(
                "0223",
                "RP 0,1,361,1000,90.,0.,0.,1.,0.",
                "NE 1,1,1,1,1.,0.,0.,0.,0.,0.",
            ),
            "NE coordinate system 1",
            id="near-field-spherical",
        ),
        pytest.param(
            _edited("0223", "RP 0,1,361,1000", "RP 0,1,361,0"),
            "RP with XNDA 0",
            id="xnda",
        ),
        pytest.param(
            _edited("0230", "GD 2,", "GD 1,"),
            "GD 1",
            id="gd-kind",
        ),
    ],
)
def test_a_shape_no_capture_prints_refuses_by_name(text, match):
    """Through the deck reader and the shape check, no solve: each of these
    is refused before a matrix is filled."""
    with pytest.raises(_nec4.Nec4Refusal, match=match):
        _nec4.solve(text, basis=BASIS)


def test_a_refusal_is_a_nec_error_under_the_nec42_banner():
    deck = _edited("0223", "EX 6,1,6,0,1.414214,0.", "EX 1,1,6,0,1.414214,0.")
    text = render(deck, basis=BASIS, dialect="nec4")
    lines = text.split("\n")
    assert "NUMERICAL ELECTROMAGNETICS CODE (NEC-4.2)" in lines[6]
    assert " ***** NEC ERROR - EX type 1 is not served" in text
    assert "ANTENNA INPUT PARAMETERS" not in text
    # The comment box still echoes, so EZNEC shows the refusal rather than
    # discarding the file as stale.
    assert "Written by EZNEC/Pro+ v. 7.0 in NEC-4.2 format." in text
