"""The rest of the NEC-5 printout, byte-gated against the captures (#497, U3).

One gate carries this unit: **extract and re-render**.  For each of the ten
captured printouts under ``tests/fixtures/eznec/printouts/``, :func:`extract`
reads the file back into a :class:`~momwire.eznec.RunData` — every number
lifted from the engine's own text, none of them recomputed — and
:func:`~momwire.eznec.render_printout` writes it out again.  The result has
to be the captured file, byte for byte.

That shape is deliberate.  U3 renders FORMAT ONLY: it is handed numbers and
lays them out, so the honest question to ask it is "given exactly the numbers
the engine had, do you print the file the engine printed?".  U4 will answer
the other half (are they the right numbers) by feeding a solve into the same
:class:`RunData`, which is why :func:`extract` builds one the way a solver
would rather than a parallel structure of its own.

The comparison is byte-exact after CRLF-to-LF alone.  The manifest lists
three more normalizations — the ``FILL=``/``RUN TIME`` timing lines, signed
zero in the pattern TILT column, the ``GMPINO`` preamble — and none of them
is needed here: timings and signed zeros are carried through the round trip
verbatim, and no rung-1 capture has a preamble.  They exist for U4, where the
numbers come from a solver and a machine's clock instead of from the file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _printout
from momwire.eznec._printout import (
    ChargeRow,
    LoadRow,
    NetworkRow,
    PatternBlock,
    PatternRow,
    PortRow,
    PowerBudget,
    RunData,
    WireCurrentRow,
    render_printout,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "eznec"
MANIFEST = json.loads((FIXTURE_DIR / "manifest.json").read_text())

GATED = tuple(entry for entry in MANIFEST["captures"] if entry.get("printout"))
GATED_IDS = tuple(entry["id"] for entry in GATED)


def capture(cid: str) -> dict:
    for entry in MANIFEST["captures"]:
        if entry["id"] == cid:
            return entry
    raise AssertionError(f"no capture {cid} in the manifest")


def deck_text(cid: str) -> str:
    return (FIXTURE_DIR / capture(cid)["deck"]).read_bytes().decode("latin-1")


def printout_text(cid: str) -> str:
    """A captured printout, CRLF-normalized per the manifest."""
    raw = (FIXTURE_DIR / capture(cid)["printout"]).read_bytes().decode("latin-1")
    return raw.replace("\r\n", "\n")


# --------------------------------------------------------------------------
# reading a printout back into a RunData
# --------------------------------------------------------------------------
#
# Rows are read by SPLITTING on whitespace wherever the columns cannot run
# together, and by slicing only where a cell can be blank (the pattern SENSE
# column, the loading table's empty cells).  Reading them back with the
# renderer's own column table would make the round trip agree with itself
# about a width that was wrong; splitting is an independent reading.


def _floats(line: str) -> list[float]:
    return [float(token) for token in line.split()]


def _section(lines: list[str], heading: str) -> int | None:
    """The index of the line carrying ``heading``, or None if absent."""
    for index, line in enumerate(lines):
        if heading in line:
            return index
    return None


def _rows(lines: list[str], start: int) -> list[str]:
    """The run of non-blank lines beginning at ``start``.

    ``ALLOCATE CM:`` ends a run as surely as a blank line does: it is the one
    line the engine prints hard against the table above it, with no blank
    between (0012, where it follows the second load row).
    """
    rows = []
    while (
        start < len(lines)
        and lines[start].strip()
        and not lines[start].startswith("ALLOCATE CM:")
    ):
        rows.append(lines[start])
        start += 1
    return rows


def _count(lines: list[str], label: str) -> int:
    for line in lines:
        if line.startswith(label):
            return int(line[len(label) :])
    raise AssertionError(f"no {label!r} line in this printout")


def _port_rows(lines: list[str], start: int) -> tuple[PortRow, ...]:
    rows = []
    for line in _rows(lines, start):
        tag, segment, end_index, *values = line.split()
        rows.append(
            PortRow(
                tag=int(tag),
                segment=int(segment),
                end_index=int(end_index),
                voltage=complex(float(values[0]), float(values[1])),
                current=complex(float(values[2]), float(values[3])),
                impedance=complex(float(values[4]), float(values[5])),
                admittance=complex(float(values[6]), float(values[7])),
                power=float(values[8]),
            )
        )
    return tuple(rows)


def _pattern_row(line: str) -> PatternRow:
    """One pattern row: seven numbers, a possibly blank SENSE, four numbers.

    The SENSE column is the one cell in the whole printout that is blank on
    some rows and filled on others (0013's ``-999.99`` null rows against its
    ordinary ones), so it is the one cell read by column rather than by
    splitting.
    """
    angles = _floats(line[:64])
    sense = line[64:72].strip()
    fields = _floats(line[72:])
    return PatternRow(
        theta_deg=angles[0],
        phi_deg=angles[1],
        vert_db=angles[2],
        hor_db=angles[3],
        total_db=angles[4],
        axial_ratio=angles[5],
        tilt_deg=angles[6],
        sense=sense,
        e_theta_magnitude=fields[0],
        e_theta_phase_deg=fields[1],
        e_phi_magnitude=fields[2],
        e_phi_phase_deg=fields[3],
    )


def _power_budget(lines: list[str], start: int) -> PowerBudget:
    values = {}
    for line in _rows(lines, start):
        label, _, value = line.partition("=")
        values[label.strip()] = float(value.split()[0])
    return PowerBudget(
        input_power=values["INPUT POWER"],
        radiated_power=values["RADIATED POWER"],
        wire_loss=values["WIRE LOSS"],
        network_loss=values.get("NETWORK LOSS"),
        efficiency_percent=values["EFFICIENCY"],
    )


def extract(text: str) -> RunData:
    """A captured printout, read back into the numbers that produced it.

    The inverse of :func:`~momwire.eznec.render_printout` over everything the
    renderer does not take from the deck.  Written for the gate, and shaped
    so U4 can read it as the worked example of what a served run must fill
    in.
    """
    lines = text.split("\n")

    loads: tuple[LoadRow, ...] = ()
    at = _section(lines, "- - - STRUCTURE IMPEDANCE LOADING - - -")
    assert at is not None
    if "THIS STRUCTURE IS NOT LOADED" not in lines[at + 2]:
        loads = tuple(
            LoadRow(
                tag=int(row[:8]),
                node_from=int(row[8:13]),
                node_thru=int(row[13:18]),
                resistance=float(row[18:73]),
                reactance=float(cell) if (cell := row[73:102].strip()) else None,
                kind=row[102:].strip(),
            )
            for row in _rows(lines, at + 6)
        )

    networks: tuple[NetworkRow, ...] = ()
    if (at := _section(lines, "- - - NETWORK DATA - - -")) is not None:
        rows = []
        for row in _rows(lines, at + 5):
            fields = row.split()
            values = [float(value) for value in fields[4:]]
            rows.append(
                NetworkRow(
                    tag_from=int(fields[0]),
                    segment_from=int(fields[1]),
                    tag_to=int(fields[2]),
                    segment_to=int(fields[3]),
                    y11=complex(values[0], values[1]),
                    y12=complex(values[2], values[3]),
                    y22=complex(values[4], values[5]),
                )
            )
        networks = tuple(rows)

    network_excitation: tuple[PortRow, ...] = ()
    at = _section(lines, "STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS")
    if at is not None:
        network_excitation = _port_rows(lines, at + 4)

    sources: tuple[PortRow, ...] = ()
    if (at := _section(lines, "- - - ANTENNA INPUT PARAMETERS - - -")) is not None:
        sources = _port_rows(lines, at + 4)

    currents: tuple[WireCurrentRow, ...] = ()
    if (at := _section(lines, "- - - Wire Currents - - -")) is not None:
        currents = tuple(
            WireCurrentRow(
                element=int(fields[0]),
                tag=int(fields[1]),
                center=(float(fields[2]), float(fields[3]), float(fields[4])),
                length=float(fields[5]),
                real=float(fields[6]),
                imag=float(fields[7]),
                magnitude=float(fields[8]),
                phase_deg=float(fields[9]),
            )
            for fields in (row.split() for row in _rows(lines, at + 6))
        )

    charges: tuple[ChargeRow, ...] = ()
    if (at := _section(lines, "- - - Wire Charge Densities - - -")) is not None:
        charges = tuple(
            ChargeRow(
                element=int(fields[0]),
                tag=int(fields[1]),
                center=(float(fields[2]), float(fields[3]), float(fields[4])),
                length=float(fields[5]),
                magnitude=float(fields[6]),
                phase_deg=float(fields[7]),
            )
            for fields in (row.split() for row in _rows(lines, at + 6))
        )

    power = None
    if (at := _section(lines, "- - - POWER BUDGET - - -")) is not None:
        power = _power_budget(lines, at + 2)

    patterns: list[PatternBlock] = []
    if (at := _section(lines, "- - - RADIATION PATTERNS - - -")) is not None:
        rows = tuple(_pattern_row(row) for row in _rows(lines, at + 5))
        trailer = "\n".join(lines[at + 5 + len(rows) :])
        gain = re.search(
            r"AVERAGE POWER GAIN=\s*(\S+).*AVERAGING=\(\s*(\S+)\)", trailer
        )
        radiated = re.search(r"STERADIANS =\s*(\S+) WATTS", trailer)
        patterns.append(
            PatternBlock(
                rows=rows,
                average_power_gain=float(gain.group(1)) if gain else None,
                solid_angle_pi=float(gain.group(2)) if gain else None,
                power_radiated_4pi=float(radiated.group(1)) if radiated else None,
            )
        )

    frequency = next(line for line in lines if line.strip().startswith("FREQUENCY="))
    wavelength = next(line for line in lines if line.strip().startswith("WAVELENGTH="))
    timing = next(line for line in lines if line.strip().startswith("FILL="))
    run_time = next(line for line in lines if line.startswith(" RUN TIME ="))
    environment = _section(lines, "- - - ANTENNA ENVIRONMENT - - -")
    assert environment is not None

    return RunData(
        node_count=_count(lines, " Number of nodes :"),
        wire_element_count=_count(lines, " Number of wire elements :"),
        patch_element_count=_count(lines, " Number of patch elements:"),
        unknown_count=_count(lines, " Number unknowns:"),
        frequency_mhz=float(frequency.split()[1]),
        wavelength_m=float(wavelength.split()[1]),
        environment=lines[environment + 2].strip(),
        loads=loads,
        networks=networks,
        network_excitation=network_excitation,
        sources=sources,
        currents=currents,
        charges=charges,
        power=power,
        patterns=tuple(patterns),
        fill_seconds=float(timing.split("=")[1].split()[0]),
        factor_seconds=float(timing.split("=")[2].split()[0]),
        run_seconds=float(run_time.split("=")[1]),
    )


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cid", GATED_IDS)
def test_every_captured_printout_round_trips_byte_for_byte(cid):
    """The unit's whole contract, ten times over.

    Failure here is a formatting bug and nothing else: the numbers came out
    of the same file they are being compared against, so any difference is a
    column width, a float format, or a blank line this renderer has wrong.
    """
    captured = printout_text(cid)
    rendered = render_printout(parse_nec5(deck_text(cid)), extract(captured))
    assert rendered == captured


@pytest.mark.parametrize("cid", GATED_IDS)
def test_the_allocation_is_the_square_of_the_unknown_count(cid):
    """``ALLOCATE CM:`` is unknowns² on all ten captures — 100 = 10², 169 =
    13², 784 = 28², 8100 = 90² — which is why the renderer computes it from
    the unknown count instead of carrying it as a number of its own."""
    data = extract(printout_text(cid))
    printed = next(
        int(line.removeprefix("ALLOCATE CM:"))
        for line in printout_text(cid).split("\n")
        if line.startswith("ALLOCATE CM:")
    )
    assert printed == data.unknown_count**2


# --------------------------------------------------------------------------
# the rules a reader would otherwise have to reverse-engineer from the diff
# --------------------------------------------------------------------------


def test_the_card_echo_reproduces_the_card_as_written_not_as_parsed():
    """0044 asks for ``RP 0,181,1,1000,…``; 0010 asks for ``RP 0,1,361,1000,…``.

    The echo prints the deck's own integer fields, which is why it is
    rendered from re-tokenized card images rather than from the parsed
    model — the model clamps a count, decodes a node address, and would
    print a different card image from the one that was read.
    """
    for cid, expected in (
        ("0044", " ***** INPUT LINE  5  RP   0  181    1 1000"),
        ("0010", " ***** INPUT LINE  5  RP   0    1  361 1000"),
    ):
        rendered = render_printout(
            parse_nec5(deck_text(cid)), extract(printout_text(cid))
        )
        assert any(line.startswith(expected) for line in rendered.split("\n"))


def test_the_terminator_is_echoed_after_the_results():
    """``EN`` keeps the line number it would have had at the top of the file
    and prints at the BOTTOM, after everything the run produced."""
    text = render_printout(
        parse_nec5(deck_text("0014")), extract(printout_text("0014"))
    )
    lines = text.split("\n")
    en = next(i for i, line in enumerate(lines) if "  EN   0" in line)
    budget = next(i for i, line in enumerate(lines) if "POWER BUDGET" in line)
    assert budget < en
    assert lines[en].startswith(" ***** INPUT LINE 10  EN")


def test_a_ground_plane_deck_prints_the_interpolation_note():
    """``GE 1,-1`` (0019) adds two sentences between the geometry echo and the
    counts; ``GE 0,-1`` (0010) adds neither."""
    grounded = render_printout(
        parse_nec5(deck_text("0019")), extract(printout_text("0019"))
    )
    free = render_printout(
        parse_nec5(deck_text("0010")), extract(printout_text("0010"))
    )
    assert "   GROUND PLANE SPECIFIED." in grounded
    assert "INTERPOLATED TO IMAGE IN GROUND PLANE." in grounded
    assert "GROUND PLANE SPECIFIED" not in free


def test_a_null_gain_row_prints_a_blank_sense_column():
    """-999.99 with an empty SENSE, and an ordinary row with ``LINEAR`` in the
    same columns — the blank is a value, so it is carried as one."""
    data = extract(printout_text("0013"))
    (block,) = data.patterns
    null = block.rows[0]
    ordinary = block.rows[1]
    assert (null.total_db, null.sense) == (-999.99, "")
    assert ordinary.sense == "LINEAR"

    rendered = _printout._pattern(block)
    body = rendered[len(_printout._PATTERN_COLUMNS) + 2 :]
    assert body[0][64:72] == " " * 8
    assert body[1][64:72] == "  LINEAR"


def test_the_network_tables_keep_the_deck_s_signed_address():
    """0012 writes the connection as ``3,-1`` and 0016 as ``2,3``.  The
    printed segment numbers differ (``3   -10`` against ``2     9``) and the
    excitation row's trailing index follows the sign (2 against 1), which is
    exactly the distinction the W7EL gate turns on."""
    a = extract(printout_text("0012"))
    b = extract(printout_text("0016"))
    assert (a.networks[0].tag_from, a.networks[0].segment_from) == (3, -10)
    assert (b.networks[0].tag_from, b.networks[0].segment_from) == (2, 9)
    assert a.network_excitation[0].end_index == 2
    assert b.network_excitation[0].end_index == 1
    assert a.sources[0].impedance == b.sources[0].impedance


def test_the_two_pattern_forms_differ_only_in_the_trailer():
    """``XNDA`` 1001 (0013) closes with the average-gain pair; ``XNDA`` 1000
    (0010) closes with nothing.  Same columns, same row format."""
    (three_d,) = extract(printout_text("0013")).patterns
    (two_d,) = extract(printout_text("0010")).patterns
    assert three_d.average_power_gain is not None
    assert three_d.power_radiated_4pi is not None
    assert two_d.average_power_gain is None
    assert two_d.power_radiated_4pi is None
    assert len(three_d.rows) == 37 * 73
    assert len(two_d.rows) == 361


def test_the_loaded_captures_print_a_blank_reactance_cell():
    """All four W7EL captures carry ``LD 4,4,n,0,1.E+10,0.`` and print the
    IMPEDANCE REAL cell only; the IMAGINARY cell is blank where the card
    wrote a zero, so the row carries it as absent rather than as 0.0."""
    for cid in ("0012", "0014", "0016", "0017"):
        loads = extract(printout_text(cid)).loads
        assert len(loads) == 2
        assert [row.resistance for row in loads] == [1.0e10, 1.0e10]
        assert [row.reactance for row in loads] == [None, None]
        assert {row.kind for row in loads} == {"FIXED IMPEDANCE"}


def test_the_unloaded_captures_say_so_in_one_line():
    for cid in ("0010", "0013", "0019", "0035", "0043", "0044"):
        assert extract(printout_text(cid)).loads == ()
        assert "THIS STRUCTURE IS NOT LOADED" in printout_text(cid)


def test_the_charge_block_answers_pq_and_matches_the_current_block():
    """``PQ 0`` is in all ten decks and all ten printouts carry the block, one
    charge row per current row, at the same segment centres."""
    for cid in GATED_IDS:
        data = extract(printout_text(cid))
        assert len(data.charges) == len(data.currents) > 0
        assert [row.center for row in data.charges] == [
            row.center for row in data.currents
        ]


def test_the_printout_is_written_with_lf_endings():
    rendered = render_printout(
        parse_nec5(deck_text("0019")), extract(printout_text("0019"))
    )
    assert "\r" not in rendered
    assert rendered.endswith(" RUN TIME =     0.000\n")


def test_a_deck_remembers_the_text_it_was_read_from():
    """The renderer echoes card images, so the parsed deck carries its own
    source; a deck built by hand without one echoes no cards rather than
    inventing them."""
    text = deck_text("0010")
    assert parse_nec5(text).source_text == text
