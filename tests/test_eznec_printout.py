"""The rest of the NEC-5 printout, byte-gated against the captures (#497 U3).

One gate carries this unit: **extract and re-render**.  For each of the
fifty-seven captured printouts under ``tests/fixtures/eznec/printouts/`` (ten
when the unit landed; #504 U1/U2/U3 quadrupled them and momwire#511/#516 took
them past fifty), :func:`extract` reads the file back into a
:class:`~momwire.eznec.RunData` — every number lifted from the engine's own
text, none of them recomputed — and
:func:`~momwire.eznec.render_printout` writes it out again.  The result has
to be the captured file, byte for byte.

That shape is deliberate.  U3 renders FORMAT ONLY: it is handed numbers and
lays them out, so the honest question to ask it is "given exactly the numbers
the engine had, do you print the file the engine printed?".  U4 will answer
the other half (are they the right numbers) by feeding a solve into the same
:class:`RunData`, which is why :func:`extract` builds one the way a solver
would rather than a parallel structure of its own.

The comparison is byte-exact after two of the manifest's normalizations:
CRLF-to-LF, and the ``SOMMPD.NEX`` cache blocks (:func:`printout_text`).  The
cache blocks are dropped HERE, at the reader, rather than in the byte-gate,
because they are not a rendering choice this engine gets to make: the seam
never reads and never writes that file, so it has no cache state to report and
no printout it writes can carry one.  Every consumer of a captured printout
therefore wants the same file, which makes one normalization at the reader
right and several at the gates wrong.

The manifest's other two — the ``FILL=``/``RUN TIME`` timing lines and signed
zero in the pattern TILT column — are not needed here: both are carried
through the round trip verbatim.  They exist for the serve gates, where the
numbers come from a solver and a machine's clock instead of from the file.
"""

from __future__ import annotations

import cmath
import json
import math
import re
from pathlib import Path

import pytest

from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _printout
from momwire.portal import _portal as nec_portal
from momwire.eznec._printout import (
    ChargeRow,
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
    render_printout,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "eznec"
MANIFEST = json.loads((FIXTURE_DIR / "manifest.json").read_text())

# Every capture that shipped a printout, and the subset this renderer has a
# form for.  Since momwire#516 they are the SAME set and the ``unrendered``
# manifest key has no tenant: 0022's ``NEAR ELECTRIC FIELDS`` block was the
# only thing this renderer had no layout for, and the near-field rung gives it
# one.  The key is deliberately not deleted from the manifest's vocabulary —
# a capture whose printout carries a block this package cannot lay out is a
# real thing to be able to say — but nothing wears it today.
#
# Rendering a block is not serving it.  Six of the seven near-field captures
# stand over a finite ground, whose numbers :mod:`momwire.eznec._serve`
# refuses by name; their printouts round-trip here all the same, because this
# unit is handed the engine's own numbers and asked only whether it prints the
# file the engine printed.
CAPTURED = tuple(entry for entry in MANIFEST["captures"] if entry.get("printout"))
CAPTURED_IDS = tuple(entry["id"] for entry in CAPTURED)
GATED = tuple(entry for entry in CAPTURED if not entry.get("unrendered"))
GATED_IDS = tuple(entry["id"] for entry in GATED)


def capture(cid: str) -> dict:
    for entry in MANIFEST["captures"]:
        if entry["id"] == cid:
            return entry
    raise AssertionError(f"no capture {cid} in the manifest")


def deck_text(cid: str) -> str:
    return (FIXTURE_DIR / capture(cid)["deck"]).read_bytes().decode("latin-1")


# The ``SOMMPD.NEX`` cache lines, by the sentence each one opens with.  Four
# block forms are observed and they are listed in the manifest's
# ``normalizations``; what they have in common is that every one of them
# reports the state of a FILE — read, stale, missing, written — and none of
# them reports anything about the antenna.
_SOMMPD_MARKERS = (
    "GMPINO:",
    "should be",
    "Will compute Sommerfeld-ground tables",
    "Sommerfeld integral tables",
    "Potentials:",
    "E, H:",
    "E and H:",
)


def _sommpd_line(line: str) -> bool:
    return any(line.strip().startswith(marker) for marker in _SOMMPD_MARKERS)


def drop_sommpd_blocks(text: str) -> str:
    """Every ``SOMMPD.NEX`` cache block, blank line and all.

    A block is ONE BLANK LINE followed by a run of cache lines, and the blank
    belongs to it — see :func:`~momwire.eznec._printout._environment`, where
    that reading is what leaves a finite-ground environment block the same
    shape as ``FREE SPACE``'s.  The rule is uniform over all four observed
    forms and over a two-frequency run's TWO consecutive blocks, which is what
    picked it: no other reading gives every block the same shape.
    """
    lines = text.split("\n")
    out: list[str] = []
    index = 0
    while index < len(lines):
        if (
            not lines[index].strip()
            and index + 1 < len(lines)
            and _sommpd_line(lines[index + 1])
        ):
            index += 1
            while index < len(lines) and _sommpd_line(lines[index]):
                index += 1
            continue
        out.append(lines[index])
        index += 1
    return "\n".join(out)


def printout_text(cid: str) -> str:
    """A captured printout, CRLF- and cache-normalized per the manifest."""
    raw = (FIXTURE_DIR / capture(cid)["printout"]).read_bytes().decode("latin-1")
    return drop_sommpd_blocks(raw.replace("\r\n", "\n"))


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


def _network_row(row: str) -> NetworkRow | LineRow:
    """One ``NETWORK DATA`` row, in whichever of the two forms it is.

    A ``TL`` row ends in a word (STRAIGHT / CROSSED) where an ``NT`` row ends
    in a number, which is the whole of the difference: the four addresses and
    the six cells before it are the same columns.  Read off the row itself
    rather than off which sub-table it came from, so a mixed table's two runs
    need no bookkeeping to tell apart.
    """
    fields = row.split()
    crossed = fields[-1] if fields[-1][-1].isalpha() else None
    values = [float(v) for v in fields[4 : 10 if crossed else None]]
    address = (int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]))
    if crossed:
        return LineRow(
            *address,
            z0=values[0],
            length_m=values[1],
            shunt_a=complex(values[2], values[3]),
            shunt_b=complex(values[4], values[5]),
            crossed=crossed == "CROSSED",
        )
    return NetworkRow(
        *address,
        y11=complex(values[0], values[1]),
        y12=complex(values[2], values[3]),
        y22=complex(values[4], values[5]),
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

    networks: tuple[NetworkRow | LineRow, ...] = ()
    if (at := _section(lines, "- - - NETWORK DATA - - -")) is not None:
        rows = []
        # One heading, then one sub-table per run of same-kind rows, each with
        # its own three column-header lines and separated by a single blank
        # (0000/0023/0025, the mixed decks).  Walked as "while the next thing
        # is a header block" so that a single-kind table is one turn of the
        # same loop rather than a case of its own.
        index = at + 2
        while index < len(lines) and "- FROM -" in lines[index]:
            block = _rows(lines, index + 3)
            rows += [_network_row(row) for row in block]
            index += 3 + len(block) + 1
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

    near_fields: list[NearFieldBlock] = []
    for heading, magnetic in (
        ("- - - NEAR ELECTRIC FIELDS - - -", False),
        ("- - - NEAR MAGNETIC FIELDS - - -", True),
    ):
        if (at := _section(lines, heading)) is None:
            continue
        near_fields.append(
            NearFieldBlock(
                rows=tuple(
                    NearFieldRow(
                        point=(v[0], v[1], v[2]),
                        magnitudes=(v[3], v[5], v[7]),
                        phases_deg=(v[4], v[6], v[8]),
                    )
                    for v in (_floats(row) for row in _rows(lines, at + 5))
                ),
                magnetic=magnetic,
            )
        )

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
    # A finite-ground block names its medium in the three lines under the
    # banner.  Read by splitting on the ``=`` each label ends with, so the
    # widths this reading recovers are independent of the ones the renderer
    # writes — the complex constant's two glued cells being the one place they
    # could not be, so they are sliced at the E12.5 boundary the capture's own
    # column shows.
    ground = None
    if lines[environment + 2].strip() == _printout.ENVIRONMENT_FINITE_GROUND:
        epsc = lines[environment + 5].partition("=")[2]
        ground = _printout.GroundMedium(
            eps_r=float(lines[environment + 3].partition("=")[2]),
            sigma=float(lines[environment + 4].partition("=")[2].split()[0]),
            eps_c=complex(float(epsc[:12]), float(epsc[12:24])),
        )

    return RunData(
        node_count=_count(lines, " Number of nodes :"),
        wire_element_count=_count(lines, " Number of wire elements :"),
        patch_element_count=_count(lines, " Number of patch elements:"),
        unknown_count=_count(lines, " Number unknowns:"),
        frequency_mhz=float(frequency.split()[1]),
        wavelength_m=float(wavelength.split()[1]),
        environment=lines[environment + 2].strip(),
        ground=ground,
        loads=loads,
        networks=networks,
        network_excitation=network_excitation,
        sources=sources,
        currents=currents,
        charges=charges,
        power=power,
        near_fields=tuple(near_fields),
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
    """The unit's whole contract, fifty-seven times over.

    Failure here is a formatting bug and nothing else: the numbers came out
    of the same file they are being compared against, so any difference is a
    column width, a float format, or a blank line this renderer has wrong.
    """
    captured = printout_text(cid)
    rendered = render_printout(parse_nec5(deck_text(cid)), extract(captured))
    assert rendered == captured


@pytest.mark.parametrize("cid", GATED_IDS)
def test_the_allocation_is_the_square_of_the_unknown_count(cid):
    """``ALLOCATE CM:`` is unknowns² on all fifty-seven captures — 100 = 10²,
    169 = 13², 784 = 28², 8100 = 90², and fourteen distinct counts between —
    which is why the renderer computes it from the unknown count instead of
    carrying it as a number of its own."""
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


def test_a_finite_ground_environment_prints_its_medium_under_the_banner():
    """0047's four-line block, verbatim, and the two constants it turns on.

    ``RELATIVE DIELECTRIC CONST.`` is an ``F7.3`` — 13.000 fills six of its
    seven columns and the oracle's ``100.000`` fills all seven — the
    conductivity an ``E10.3``, and the complex constant two glued ``E12.5``
    cells whose imaginary part carries no ``j`` and wears its sign against the
    mantissa.  ``FREE SPACE`` and ``PERFECT GROUND`` print their name centred
    instead, which is a different format and not a different indent.
    """
    lines = printout_text("0047").split("\n")
    at = _section(lines, "- - - ANTENNA ENVIRONMENT - - -")
    assert lines[at + 1] == ""
    assert lines[at + 2 : at + 6] == [
        " " * 40 + "FINITE GROUND.  SOMMERFELD SOLUTION",
        " " * 40 + "RELATIVE DIELECTRIC CONST.= 13.000",
        " " * 40 + "CONDUCTIVITY= 5.000E-03 MHOS/METER",
        " " * 40 + "COMPLEX DIELECTRIC CONSTANT= 1.30000E+01-1.28400E+01",
    ]
    assert lines[at + 6] == ""

    data = extract(printout_text("0047"))
    assert data.environment == _printout.ENVIRONMENT_FINITE_GROUND
    assert data.ground == _printout.GroundMedium(13.0, 0.005, complex(13.0, -12.84))
    # 0048 is the same medium a fifth of a percent up the band, and its own
    # cell is what pins the frequency dependence.
    assert extract(printout_text("0048")).ground.eps_c == complex(13.0, -12.8034)
    # The two bannerless forms keep the centred layout they had.
    assert extract(printout_text("0019")).ground is None
    assert extract(printout_text("0010")).ground is None


def test_a_cache_block_is_dropped_with_the_blank_line_that_opens_it():
    """All four observed ``SOMMPD.NEX`` forms, and the reading that unifies them.

    Each block is one blank line and the cache lines after it.  That reading —
    rather than "the lines and the blank after them" — is what makes the
    finite-ground environment block come out with ONE blank under its heading,
    the same shape ``FREE SPACE`` has, and it is the only reading that also
    survives a two-frequency run's two consecutive blocks (measured on the
    linux oracle 2026-08-20; the alternative needs the trailing blank to be
    optional, and optional exactly where a banner follows).
    """
    stale = "\n".join(
        (
            " GMPINO: EPSC from file = 1.30000E+01-4.90612E+01",
            "              should be  1.30000E+01-1.28400E+01",
            " Will compute Sommerfeld-ground tables",
        )
    )
    valid = "Sommerfeld integral tables read:\nPotentials: 11 12 13 14 15\nE, H:"
    absent = (
        " GMPINO: Unable to open file SOMMPD.NEX" + " " * 30
    ) + "\n Will compute Sommerfeld-ground tables"
    written = (
        "Sommerfeld integral tables written in previous run:\n"
        "Potentials: 11 12 13 14 15\nE and H: 11 12 13 14 15"
    )
    for form in (stale, valid, absent, written):
        assert drop_sommpd_blocks(f"HEAD\n\n{form}\nTAIL") == "HEAD\nTAIL"
        # Two blocks back to back — the two-frequency case — collapse too.
        assert drop_sommpd_blocks(f"HEAD\n\n{form}\n\n{absent}\nTAIL") == "HEAD\nTAIL"
    # A blank line with no cache under it is an ordinary blank and survives.
    assert drop_sommpd_blocks("HEAD\n\nTAIL") == "HEAD\n\nTAIL"


def test_the_captures_still_carry_their_cache_blocks_on_disk():
    """The normalization is a READER's, not an edit: the committed bytes are
    the engine's own, cache reports and all.

    Worth its own gate because the fixtures were converted CRLF-to-LF on the
    way in, and a conversion is exactly the moment someone might also tidy.
    0021 and 0047 are the same deck run four days apart with a differently
    stale cache; 0022 carries the corpus's only VALID block; and all four
    carry the postamble the engine writes after the ``EN`` echo.
    """
    for cid, marker in (
        ("0021", " GMPINO: EPSC from file = 2.00000E+01-3.89052E+01"),
        ("0047", " GMPINO: EPSC from file = 1.30000E+01-4.90612E+01"),
        ("0048", " GMPINO: EPSC from file = 1.30000E+01-1.28400E+01"),
        ("0022", "Sommerfeld integral tables read:"),
    ):
        raw = (FIXTURE_DIR / capture(cid)["printout"]).read_bytes().decode("latin-1")
        assert "\r" not in raw  # committed LF, per the manifest
        assert marker in raw
        assert "Sommerfeld integral tables written in previous run:" in raw
        assert marker not in printout_text(cid)


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
    for cid in ("0010", "0013", "0019", "0021", "0035", "0043", "0044", "0047", "0048"):
        assert extract(printout_text(cid)).loads == ()
        assert "THIS STRUCTURE IS NOT LOADED" in printout_text(cid)


def test_the_charge_block_answers_pq_and_matches_the_current_block():
    """``PQ 0`` is in all fifty-seven decks and all fifty-seven printouts carry
    the block, one charge row per current row, at the same segment centres."""
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


def test_the_near_field_block_sits_between_the_budget_and_the_patterns():
    """Section order, on the seven captures that carry one.

    Every near-field printout in the corpus prints its table directly under
    ``EFFICIENCY`` and none of them carries an ``RP`` as well, so this is the
    only placement the captures can settle — and it is settled on all seven.
    """
    for cid in ("0022", "0107", "0108", "0110", "0111", "0112", "0113", "0115"):
        text = printout_text(cid)
        heading = next(h for h in ("ELECTRIC", "MAGNETIC") if f"NEAR {h}" in text)
        assert text.index("- - - POWER BUDGET - - -") < text.index(f"NEAR {heading}")
        assert "- - - RADIATION PATTERNS - - -" not in text


def test_the_two_near_field_mnemonics_differ_in_three_words_and_nothing_else():
    """0111 against 0110 — the same deck, the same grid, ``NE`` respelled ``NH``.

    The block is byte-identical after ``MAGNETIC``/``ELECTRIC``, ``H``/``E`` on
    the component letters and ``AMPS/M``/``VOLTS/M`` in the units, which is
    what lets one renderer serve both.  Compared on the HEADER lines only: the
    rows below them are two different fields and are supposed to differ.
    """

    def header(cid: str) -> list[str]:
        lines = printout_text(cid).split("\n")
        at = next(i for i, line in enumerate(lines) if "- - - NEAR " in line)
        return lines[at : at + 5]

    electric = header("0110")
    magnetic = header("0111")
    folded = [
        line.replace("MAGNETIC", "ELECTRIC")
        .replace("HX", "EX")
        .replace("HY", "EY")
        .replace("HZ", "EZ")
        .replace(" AMPS/M", "VOLTS/M")
        for line in magnetic
    ]
    assert folded == electric


def test_the_near_field_points_are_metres_and_the_current_table_is_not():
    """The one geometry column in the printout that is NOT wavelength-normalized.

    0108's ``NE`` grid asks for ``1., 0., 2.`` at 7 MHz and the row prints
    ``1.0000E+00``; the current table's own centres on the same printout are
    normalized by the 42.83 m wavelength (element 1 at 1.20247E-02 for a
    0.515 m height).  Reading a near-field point as normalized would put the
    whole grid 43 times too close to the wire.
    """
    data = extract(printout_text("0108"))
    (block,) = data.near_fields
    assert block.rows[0].point == (1.0, 0.0, 2.0)
    assert block.rows[-1].point == (5.0, 0.0, 10.0)
    assert data.currents[0].center[2] == pytest.approx(0.515 / data.wavelength_m, 1e-3)


def test_a_deck_remembers_the_text_it_was_read_from():
    """The renderer echoes card images, so the parsed deck carries its own
    source; a deck built by hand without one echoes no cards rather than
    inventing them."""
    text = deck_text("0010")
    assert parse_nec5(text).source_text == text


# --- the pattern table is reproducible (momwire#578) --------------------------


def test_a_dust_e_field_prints_as_a_clean_zero_rather_than_its_own_angle():
    """momwire#578 class A, at the seam that decides it.

    An E-field component below this printout's dust floor has a magnitude that
    is noise and a phase that is the angle of noise; both moved with thread
    count and process history, which is nondeterminism in a printout two of
    which are asserted byte-identical. 0015 printed `2.45557E-14 / 72.65` and
    `2.44145E-14 / 73.74` for the same row on the same machine.
    """
    from momwire.eznec import _serve

    floor = math.sqrt(_serve._PRINTED_DUST_FLOOR2)
    assert _serve._PRINTED_DUST_FLOOR2 != nec_portal._PRINTED_DUST_FLOOR2, (
        "the seams derive this bar separately and in different units — see "
        "the note beside the constant"
    )
    # Above the bar a reading keeps its digits; below it, it is a clean zero.
    assert floor == pytest.approx(1e-7, rel=1e-9)


def test_the_dust_floor_clears_what_moves_and_spares_what_does_not():
    """The derivation, kept as an assertion rather than only as prose.

    Measured with the bar itself switched off, over all 80 capture decks at
    OMP_NUM_THREADS 1 vs 8: the largest E-field magnitude that MOVED was
    1.648e-9 V/m, and the weakest legitimate reading in the captured
    printouts is 2.5e-5 V/m — with nothing at all in between. The bar has to
    sit inside that void, and a future retune that walks it out of the void
    should fail here rather than silently start zeroing readings or stop
    zeroing dust.

    Re-derived at 80 decks (the corpus was 62 when the bar was set) and both
    ends held: the moving maximum landed on the same 1.648e-9, and the void
    is still empty end to end. With the bar ON, nothing in the pattern tables
    moves between the two thread counts at all.
    """
    from momwire.eznec import _serve

    floor = math.sqrt(_serve._PRINTED_DUST_FLOOR2)
    assert floor > 1.65e-9 * 10, "the bar must clear everything measured to move"
    assert floor < 1e-5 / 10, "the bar must stay well under the weakest reading"


@pytest.mark.parametrize(
    "delta_deg,expect",
    [(90.0, "LEFT"), (-90.0, "RIGHT"), (0.0, "LINEAR")],
)
def test_a_real_polarisation_sense_still_reads_off_the_phase(delta_deg, expect):
    """The control for class B: raising the LINEAR bar must not flatten a
    genuine circular row into LINEAR."""
    et = complex(1.0, 0.0)
    ep = cmath.exp(1j * math.radians(delta_deg))
    _axial, _tilt, sense = nec_portal._polarisation(et, ep, 1.0, linear_below=5e-6)
    assert sense == expect


def test_a_zero_axial_ratio_cannot_print_beside_a_handed_sense():
    """momwire#578 class B. The two columns are one statement: if the ratio
    prints 0.00000 the row IS linear, and a `RIGHT`/`LEFT` beside it is the
    sign bit of a number that rounded away. 46 rows flipped between runs,
    every one of them printing an axial ratio of zero.

    Asserted at a ratio the old `1e-8` bar let through and this one does not.
    """
    # delta small enough that minor/major lands between the two bars
    et = complex(1.0, 0.0)
    ep = complex(1.0, 1e-6)
    axial, _tilt, sense = nec_portal._polarisation(et, ep, 1.0, linear_below=5e-6)
    assert f"{axial:11.5f}".strip() == "0.00000"
    assert sense == "LINEAR", "a zero axial ratio may not carry a handedness"
