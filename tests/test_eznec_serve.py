"""Rung-1 and rung-2 physics, gated against the captures (#497 U4, #504 U1).

Three gates carry this unit, and the split between them is the point.

**The structure gate.**  A served printout, taken all the way through the
shell that EZNEC launches, has to BE the captured file everywhere the file is
not a solved number: every heading, every blank line, every card echo, every
count, every row of every table, the frequency block, the drive current cell,
and the ``-999.99`` null convention with its blank SENSE column.  Only the
cells a solver filled in are masked, on both sides, and everything else is
compared byte for byte.

**The envelope pins.**  momwire's B-spline formulation and NEC-5's are two
discretizations of one physics and they do not agree to printed precision —
the same Ω-class offset ``tests/test_contact_nec5_lane.py`` measures on a
base-fed monopole, and the reason the golden lanes in this tree gate deltas
rather than impedances.  So the physics is pinned the golden-lane way: a bar
per capture at the MEASURED difference plus 25 %, with the measurements in
the table below so the next person can see what moved rather than only that
something did.  A pin that tightens is a finding; a pin that loosens is a
regression.

**The refusals.**  Everything above rung 1 still refuses BY NAME, through the
real shell, at exit 0, with the comment stamp intact — because a refusal that
does not reach EZNEC's viewer reaches nobody.

The manifest's normalizations for a served run are applied on the way in and
nowhere else: CRLF to LF and the ``SOMMPD.NEX`` cache blocks at the reader
(``test_eznec_printout.printout_text``), the ``FILL=``/``RUN TIME`` timing
lines dropped here (a timing is a property of the machine), and signed zero
folded in the pattern TILT column.
"""

from __future__ import annotations

import functools
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from momwire import BSplineSolver
from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _serve
from momwire.eznec._shell import render
from test_eznec_printout import (
    FIXTURE_DIR,
    GATED_IDS,
    capture,
    deck_text,
    drop_sommpd_blocks,
    extract,
    printout_text,
)

# The BARE-STRUCTURE captures that shipped a printout — the six this unit is
# gated against.  The other seven gated captures carry ``TL``/``NT`` cards and
# belong to the network unit, which measures its own table in
# ``test_eznec_networks.py``; they still appear below, in the counting-rule
# gate, because the count is a property of the deck's geometry and not of what
# answers it.
SERVED_IDS = (
    "0010",
    "0013",
    "0019",
    "0021",
    "0035",
    "0043",
    "0044",
    "0047",
    "0048",
)

# The three of them that stand over a finite ``GN 0`` ground — the rung this
# unit adds.  One 10.3 m base-fed vertical over 13/0.005 earth, three times:
# 0047 with a 181-point elevation cut at 7.00 MHz, 0021 its 2026-08-16 twin
# (same deck, different launch stamp, a differently stale ``SOMMPD.NEX``), and
# 0048 the same antenna answered by ``XQ`` at 7.02 MHz.
FINITE_IDS = ("0021", "0047", "0048")

# The rung-1 captures with no printout: seven ``XQ``-only dipoles EZNEC wrote
# while stepping a frequency.  Nothing to byte-compare, but a deck that
# refuses is a deck that stopped being served, so they are run for their exit
# status.
SERVED_UNGATED_IDS = ("0036", "0037", "0038", "0039", "0040", "0041", "0042")

# The captures the GROUND rung brought in with no printout of their own, and
# they are the four most interesting decks this seam now answers: 0011/0030
# hang a coax ``TL`` off a dipole over ``GN 0`` (the first network over a
# finite ground), and 0033/0034 are elevated radial systems whose radials
# stand 1.8 cm — 1e-4 λ — above it.  Nothing here can be gated against a
# printout; what IS gated is that they answer at all, with finite numbers,
# under the right banner.  A capture errand for the next Windows session.
FINITE_UNGATED_IDS = ("0011", "0030", "0033", "0034")


# --------------------------------------------------------------------------
# the measured table the pins come from
# --------------------------------------------------------------------------
#
# Measured 2026-08-20 on this tree, with the served printout read back through
# `test_eznec_printout.extract` and compared cell for cell with the capture's:
#
#   id    Z (capture)          Z (served)          |dZ|    peak dB   d(peak)
#   0010  79.948 +29.919j      85.073 +45.369j    16.278   2.18/2.09   0.09
#   0013  55.621 + 8.2725j     58.876 +19.210j    11.412   1.90/1.80   0.10
#   0019  35.571 - 1.4223j     36.499 + 2.0789j    3.622   (XQ, none)   —
#   0021  47.789 - 0.78525j    48.867 + 2.5635j    3.518  -1.31/-1.33  0.02
#   0035  23.343 -24.594j      23.926 -20.079j     4.552   9.88/9.84   0.04
#   0043  35.571 - 1.4223j     36.499 + 2.0789j    3.622   (XQ, none)   —
#   0044  35.571 - 1.4223j     36.499 + 2.0789j    3.622   5.15/5.13   0.02
#   0047  47.789 - 0.78525j    48.867 + 2.5635j    3.518  -1.31/-1.33  0.02
#   0048  48.155 + 0.65170j    49.254 + 4.0055j    3.529   (XQ, none)   —
#
# The impedance bar is that |dZ| plus 25 %.  The reactance carries almost all
# of it — a node source and a segment gap store different amounts of energy in
# the feed region, which is exactly the formulation difference the scored
# matrix priced when it moved this unit's gate off byte equality.
#
# The finite-ground rows are the SMALLEST offsets in the table (3.5 Ω on a
# 48 Ω row, against 16 on the free-space dipole), and the reason is worth
# writing down: the Sommerfeld ground loads the feed region, so the same
# formulation difference sits on a bigger, lossier admittance and shows less.
# The offset is also within 3 % of the perfect-ground twins' 3.622 on the same
# 10.3 m wire (0019/0043/0044), which says the ground model added almost none
# of it — the gap is the feed's, as it is everywhere else in this table.
#
# The peak-gain bar is |d| plus 25 % OR 0.05 dB, whichever is larger: the
# printed cell is quantized at 0.01 dB, so a bar under a few hundredths would
# be pinning the rounding rather than the physics.
Z_BAR = {
    "0010": 20.35,
    "0013": 14.27,
    "0019": 4.53,
    "0021": 4.40,
    "0035": 5.69,
    "0043": 4.53,
    "0044": 4.53,
    "0047": 4.40,
    "0048": 4.42,
}

# (peak total gain dB, theta, phi) as the capture prints it, and the bar.
PEAK_BAR = {
    "0010": 0.12,
    "0013": 0.13,
    "0021": 0.05,
    "0035": 0.05,
    "0044": 0.05,
    "0047": 0.05,
}

# The finite-ground patterns are gated at EVERY printed angle and not only at
# the peak, which is the evidence that the Fresnel-weighted image is the right
# one rather than merely well aimed: over a real ground the whole elevation
# cut is the ground's doing, and a wrong medium would show up in the low-angle
# rows long before it moved the peak.  Measured worst |d(TOTAL dB)| over the
# 178 rows that are not the three -999.99 nulls: 0.06 dB, at theta = +-1 and
# +-2 degrees, where the cut is falling through 30 dB in two degrees and the
# printed cell is quantized at 0.01.  Plus 25 %.
PATTERN_BAR = 0.075

# NEC's own floor for a gain cell, as a number rather than as the text the
# mask compares.
_NULL_DB = -999.99

# Largest per-element difference in the WIRE CURRENTS table, relative to that
# table's own peak current, and in the CHARGE DENSITIES table relative to its
# peak magnitude.  Measured, plus 25 %.  The charge bar is the gate on the
# analytic charge readout: a q that were merely plausible would not sit
# inside 8 % of a different code's on five decks at once.
CURRENT_BAR = {
    "0010": 0.028,
    "0013": 0.025,
    "0019": 0.013,
    "0021": 0.0123,
    "0035": 0.228,
    "0043": 0.013,
    "0044": 0.013,
    "0047": 0.0123,
    "0048": 0.0124,
}
CHARGE_BAR = {
    "0010": 0.086,
    "0013": 0.078,
    "0019": 0.078,
    "0021": 0.076,
    "0035": 0.186,
    "0043": 0.078,
    "0044": 0.078,
    "0047": 0.076,
    "0048": 0.077,
}


@functools.lru_cache(maxsize=None)
def served(cid: str) -> str:
    """The printout this engine writes for a capture's deck."""
    return render(deck_text(cid))


def _z_of(text: str) -> complex:
    """The feedpoint impedance one deck's served printout reports."""
    return extract(render(text)).sources[0].impedance


# --------------------------------------------------------------------------
# the mask: every cell a solver filled in, and nothing else
# --------------------------------------------------------------------------
#
# Column edges are the captures', read off the printouts rather than off the
# renderer's own table — a mask built from the code it gates would agree with
# a wrong width.

_MASK = "#" * 4

# ANTENNA INPUT PARAMETERS: three integers, then nine E12.4 cells.  Cells 2
# and 3 are the CURRENT, which for an `EX 4` deck is the card's own drive and
# has to survive unmasked; the rest is solved.
_PORT_CELLS = tuple(k for k in range(9) if k not in (2, 3))
# The network table shares the format and not that exemption: the current at a
# connection point is the STRUCTURE's, solved, even on the row whose port the
# deck drives (0027 prints 1.4142E+00 under ANTENNA INPUT PARAMETERS and
# 3.3124E-09 for the same port here — all of the difference went down the two
# transmission lines).
_NETWORK_PORT_CELLS = tuple(range(9))

# Pattern row column edges, measured on 0013 (which prints every form: a
# blank SENSE, a `-999.99` null, and ordinary rows).
_PATTERN_COLUMNS = {
    "angles": (0, 17),
    "vert": (17, 28),
    "hor": (28, 36),
    "total": (36, 44),
    "axial": (44, 55),
    "tilt": (55, 64),
    "sense": (64, 72),
    "fields": (72, 120),
}

_NULL = "-999.99"
_NUMBER = re.compile(r"[-+]?\d+\.\d+(?:[Ee][-+]?\d+)?")

# How many lines each table puts between its heading and its first row.
#
# NETWORK DATA is deliberately absent: every cell in it is an ECHO of a card
# (0012's admittances, 0028's impedances and STRAIGHT/CROSSED flags) or a
# distance between two nodes (0028's four zero-length feeders), so none of it
# is a solved number and all of it is compared byte for byte.
_TABLES = {
    "- - - STRUCTURE EXCITATION DATA AT NETWORK CONNECTION POINTS - - -": (
        "network",
        3,
    ),
    "- - - ANTENNA INPUT PARAMETERS - - -": ("port", 3),
    "- - - Wire Currents - - -": ("wire", 5),
    "- - - Wire Charge Densities - - -": ("wire", 5),
    "- - - POWER BUDGET - - -": ("power", 1),
    "- - - RADIATION PATTERNS - - -": ("pattern", 4),
}


def _mask_port_row(line: str, solved: tuple[int, ...] = _PORT_CELLS) -> str:
    cells = [line[12 + 12 * k : 24 + 12 * k] for k in range(9)]
    for k in solved:
        cells[k] = _MASK.rjust(12)
    return line[:12] + "".join(cells)


def _mask_pattern_row(line: str) -> str:
    def cell(name: str) -> str:
        lo, hi = _PATTERN_COLUMNS[name]
        return line[lo:hi]

    def gain(name: str) -> str:
        text = cell(name)
        # The null itself is the convention, so it is never masked: a served
        # row that fell short of NEC's own DB10 floor has to fall short on the
        # same rows the capture did, and a row that did not has to print a
        # number on the same rows too.
        return text if text.strip() == _NULL else _MASK.rjust(len(text))

    sense = cell("sense")
    return (
        cell("angles")
        + gain("vert")
        + gain("hor")
        + gain("total")
        # AXIAL / TILT are the polarisation ellipse of a field whose CROSS
        # component is five to sixteen orders under its co-polar one; which
        # way that dust leans is a property of a formulation, not of an
        # antenna, and the two codes disagree about it on 1848 of 0013's 2701
        # rows.  Masked, and the blank-vs-filled SENSE below is what still
        # gates the convention they DO share.
        + _MASK.rjust(len(cell("axial")))
        + _MASK.rjust(len(cell("tilt")))
        + (sense if not sense.strip() else _MASK.rjust(len(sense)))
        + _MASK.rjust(len(cell("fields")))
    )


def mask(text: str) -> str:
    """A printout with its solved cells replaced, table by table.

    Also applies the manifest's two remaining served-run normalizations: the
    timing lines are dropped outright, and nothing else about them is looked
    at.
    """
    lines = text.split("\n")
    out: list[str] = []
    kind: str | None = None
    skip = 0
    for line in lines:
        if "FILL=" in line or line.startswith(" RUN TIME ="):
            continue
        if "AVERAGE POWER GAIN" in line or "POWER RADIATED" in line:
            # The 3-D form's trailer sits BELOW the blank line that ends the
            # pattern table, so it is recognised by what it says rather than
            # by which table it is inside.
            out.append(_NUMBER.sub(_MASK, line))
            continue
        heading = next((h for h in _TABLES if h in line), None)
        if heading is not None:
            kind, skip = _TABLES[heading]
            out.append(line)
            continue
        if skip:
            skip -= 1
            out.append(line)
            continue
        if kind is None or not line.strip():
            kind = None
            out.append(line)
            continue
        if kind == "port":
            out.append(_mask_port_row(line))
        elif kind == "network":
            out.append(_mask_port_row(line, _NETWORK_PORT_CELLS))
        elif kind == "wire":
            out.append(line[:12] + _MASK)
        elif kind == "power":
            label, sep, _value = line.partition("=")
            out.append(label + sep + _MASK)
        elif kind == "pattern":
            out.append(_mask_pattern_row(line))
    return "\n".join(out)


# --------------------------------------------------------------------------
# gate 1 — the structure
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cid", SERVED_IDS)
def test_a_served_printout_is_the_capture_wherever_it_is_not_a_solved_number(cid):
    """The unit's whole shape, six times over.

    What this proves, and it is most of the file: the row counts, the tag and
    element numbering, the geometry summary, the ``ALLOCATE CM`` allocation,
    the card echoes, the environment banner, the frequency and wavelength, the
    drive current cell, the section order and the blank lines between sections
    are all correct — and the ``-999.99`` nulls fall on exactly the directions
    the capture put them, with SENSE blank on exactly the same rows.
    """
    assert mask(served(cid)) == mask(printout_text(cid))


def test_the_shell_writes_the_served_printout_and_still_exits_zero(tmp_path):
    """Through the real process, the way EZNEC launches it: two positional
    paths, nothing on stdin, exit 0, and a full result file on disk."""
    deck = tmp_path / "EZN5.NEC"
    deck.write_bytes((FIXTURE_DIR / capture("0044")["deck"]).read_bytes())
    out = tmp_path / "NEC5.OUT"

    proc = subprocess.run(
        [sys.executable, "-m", "momwire.eznec", str(deck), str(out)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert proc.returncode == 0
    assert proc.stdout == ""
    written = out.read_bytes().decode("latin-1")
    assert "NEC ERROR" not in written
    assert mask(written) == mask(printout_text("0044"))


# --------------------------------------------------------------------------
# gate 1b — the counting rule, on all ten captures
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cid", GATED_IDS)
def test_the_counting_rule_reproduces_every_captured_count(cid):
    """``Σ max(degree − 1, 0)`` over the fused nodes, on every printout.

    The rule was DERIVED from ten of these numbers and has to answer all of them
    or it is a coincidence: a free wire end contributes nothing, an interior
    node one, a junction of m wires m − 1 (0013's five-wire apex is the only
    capture that separates that from "one per junction"), and a wire end
    standing in a declared ground plane counts its image as one more element
    end (0019, the only capture that separates that from "free end").

    The seven network captures are here too, and three of them were added
    after the rule was written: 0018, 0027 and 0028 answered it unchanged,
    which is the only kind of evidence a derived rule can get.  0012's 17
    nodes / 15 elements / 13 unknowns is the deck with several two-wire
    junctions AND a wire that touches nothing; 0027 pairs a ``GE 1`` ground
    plane with a virtual wire far above it.
    """
    structure = _serve.structure_of(parse_nec5(deck_text(cid)))
    data = extract(printout_text(cid))
    assert (
        structure.node_count,
        structure.wire_element_count,
        structure.patch_element_count,
        structure.unknown_count,
    ) == (
        data.node_count,
        data.wire_element_count,
        data.patch_element_count,
        data.unknown_count,
    )


# --------------------------------------------------------------------------
# gate 1c — the geometry columns are the deck's own
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cid", SERVED_IDS)
def test_the_current_table_s_geometry_columns_are_the_captured_ones(cid):
    """Segment centres and lengths, compared as NUMBERS rather than bytes.

    They are masked in the structure gate for one cell's sake: 0010's sixth
    element is centred at the dipole's midpoint, where the true coordinate is
    zero and both engines print their own rounding dust (3.46936E-17 against
    -1.38774E-17).  Neither engine's digits mean anything there, and no
    arithmetic reproduces the other's, so the column is gated at 1e-9 metres
    instead — three hundred million times tighter than the dust and eight
    orders tighter than the printed precision on every cell that is not dust.
    """
    want = extract(printout_text(cid))
    got = extract(served(cid))
    assert len(got.currents) == len(want.currents)
    for a, b in zip(want.currents, got.currents):
        assert (a.element, a.tag) == (b.element, b.tag)
        assert a.length == pytest.approx(b.length, abs=1e-9)
        for wanted, served_ in zip(a.center, b.center):
            assert wanted == pytest.approx(served_, abs=1e-9)


# --------------------------------------------------------------------------
# gate 2 — the envelope pins
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cid", SERVED_IDS)
def test_the_feedpoint_impedance_sits_inside_its_measured_envelope(cid):
    """|Z_served − Z_captured| against a per-capture bar (see the table above).

    An ENVELOPE and not an agreement claim: the difference is a formulation
    difference, it does not shrink with anything this unit controls, and the
    bar exists so that a change to the basis, the drive spelling or the port
    algebra is visible the day it lands.
    """
    want = extract(printout_text(cid)).sources[0].impedance
    got = extract(served(cid)).sources[0].impedance
    assert abs(got - want) <= Z_BAR[cid], f"{cid}: served {got}, captured {want}"


@pytest.mark.parametrize("cid", sorted(PEAK_BAR))
def test_the_pattern_peak_lands_where_the_capture_puts_it(cid):
    """The peak DIRECTION is exact on all six patterned captures; the peak
    LEVEL is inside a tenth of a dB.

    The direction is the part a formulation difference does not get to move —
    it is set by the geometry and the current's shape, not by the feed
    region — so it is gated exactly, and the level is gated by envelope.
    """
    (want,) = extract(printout_text(cid)).patterns
    (got,) = extract(served(cid)).patterns
    assert len(got.rows) == len(want.rows)
    best_want = max(want.rows, key=lambda row: row.total_db)
    best_got = max(got.rows, key=lambda row: row.total_db)
    assert (best_got.theta_deg, best_got.phi_deg) == (
        best_want.theta_deg,
        best_want.phi_deg,
    )
    assert abs(best_got.total_db - best_want.total_db) <= PEAK_BAR[cid]


@pytest.mark.parametrize("cid", ("0021", "0047"))
def test_the_finite_ground_elevation_cut_agrees_at_every_printed_angle(cid):
    """All 181 rows of the ``RP 0`` cut, not just its peak.

    Over a real ground the whole elevation pattern is the ground's doing — the
    direct wave and its Fresnel-weighted image interfere direction by
    direction — so agreeing at 181 angles is a statement about the MEDIUM in a
    way that agreeing at the peak is not.  Worst row is 0.06 dB, at
    theta = ±1° and ±2°, where the cut falls through 30 dB in two degrees.

    The three ``-999.99`` rows are skipped HERE and gated harder elsewhere:
    they are never masked, so the structure gate already compares them byte
    for byte and would fail if the served run put a null anywhere the capture
    did not.  Which it does not — the horizon nulls at θ = ±90° (a vertical
    over any ground has no grazing field) and the zenith null at θ = 0° land
    on exactly the captured rows.
    """
    (want,) = extract(printout_text(cid)).patterns
    (got,) = extract(served(cid)).patterns
    assert len(got.rows) == len(want.rows) == 181
    nulls = [row.theta_deg for row in want.rows if row.total_db == _NULL_DB]
    assert nulls == [90.0, 0.0, -90.0]
    worst = max(
        abs(a.total_db - b.total_db)
        for a, b in zip(want.rows, got.rows)
        if a.total_db != _NULL_DB
    )
    assert worst <= PATTERN_BAR, f"{cid}: worst row {worst:.4f} dB"


@pytest.mark.parametrize("cid", SERVED_IDS)
def test_the_current_and_charge_tables_sit_inside_their_measured_envelopes(cid):
    """Every element's current AND its charge density, against the capture.

    The charge block is the one readout in this unit with no second opinion
    anywhere in the tree, so this is the gate that says it is physics rather
    than plausible-looking arithmetic: ``q = −(1/jω)·dI/ds``, differentiated
    in the B-spline basis, lands within 8 % of NEC-5's own on eight of the
    nine captures and within 19 % on the Yagi, whose currents differ by as
    much.  The three finite-ground captures are the TIGHTEST in the table —
    6.1 % on the charge and 1.0 % on the current — which is the same story the
    impedance envelope tells from the other end.
    """
    want = extract(printout_text(cid))
    got = extract(served(cid))
    peak = max(row.magnitude for row in want.currents)
    assert (
        max(
            abs(complex(a.real, a.imag) - complex(b.real, b.imag))
            for a, b in zip(want.currents, got.currents)
        )
        / peak
        <= CURRENT_BAR[cid]
    )
    charge_peak = max(row.magnitude for row in want.charges)
    assert (
        max(abs(a.magnitude - b.magnitude) for a, b in zip(want.charges, got.charges))
        / charge_peak
        <= CHARGE_BAR[cid]
    )
    # The magnitude envelope cannot see the charge's SIGN — q and -q print
    # the same Mag column and differ by 180 in Phase.  The continuity sign
    # (q = -(1/jw) dI/ds) lands within ~10 deg of NEC-5's printed phase on
    # every element that carries real charge (measured on 0010; the worst
    # rows sit beside the charge null, where phase swings), so 45 deg is a
    # bar that survives formulation noise and kills a sign flip dead.
    for a, b in zip(want.charges, got.charges):
        if a.magnitude < 0.1 * charge_peak:
            continue
        wrap = (b.phase_deg - a.phase_deg + 180.0) % 360.0 - 180.0
        assert abs(wrap) <= 45.0


def test_a_current_source_is_a_readout_transform_and_not_a_second_solve():
    """0010's ``EX 4 … 1.414214,0.`` prints its drive back EXACTLY, and the
    rest of the row is that drive times the driving-point impedance.

    Which is the whole of what the scored matrix means by "readout
    transform": with one port, ``I = Y·V`` is linear, so the current source is
    the voltage-driven solve rescaled, and the printed row has to close on
    itself — ``V = Z·I``, ``Y = 1/Z``, ``P = ½·Re(V·I*) = ½·|I|²·R``, which is
    how the capture's own 7.9948E+01 watts is |1.414214|²/2 times its 79.948 Ω.
    """
    row = extract(served("0010")).sources[0]
    assert row.current == complex(1.4142, 0.0)  # printed E12.4, drive verbatim
    assert row.voltage == pytest.approx(row.impedance * row.current, rel=1e-4)
    assert row.admittance == pytest.approx(1.0 / row.impedance, rel=1e-4)
    assert row.power == pytest.approx(
        0.5 * abs(row.current) ** 2 * row.impedance.real, rel=1e-4
    )


@pytest.mark.parametrize("cid", SERVED_IDS)
def test_a_lossless_deck_radiates_everything_it_is_given(cid):
    """Every wire here is a perfect conductor and this dialect has no
    conductivity card, so INPUT = RADIATED, WIRE LOSS = 0 and EFFICIENCY reads
    100.00 — which is what all nine captures print.

    Including the three over LOSSY GROUND, which is the entry worth reading
    twice: 0047 dumps a good fraction of its input into the earth and still
    prints ``EFFICIENCY    = 100.00 PERCENT``.  NEC's POWER BUDGET counts what
    the STRUCTURE dissipates, and the ground is not part of the structure —
    so a reader who wants the ground loss has to take it out of the pattern's
    average gain, not out of this block.  Reproducing that means reproducing
    the convention, not correcting it."""
    power = extract(served(cid)).power
    assert power.wire_loss == 0.0
    assert power.input_power == power.radiated_power
    assert power.efficiency_percent == 100.0
    assert power.network_loss is None


# --------------------------------------------------------------------------
# the ground rung's own facts
# --------------------------------------------------------------------------
#
# Three things about `GN 0` that no capture states on its own and that the
# seam would otherwise be free to get wrong quietly: the second spelling of the
# card, the second spelling of its loss field, and the cache file it must not
# touch.


def _with_ground(cid: str, card: str) -> str:
    """0047's deck with its ground line replaced, and nothing else."""
    text = deck_text(cid)
    assert "GN 0,0,0,0,13.,.005,1.,0." in text
    return text.replace("GN 0,0,0,0,13.,.005,1.,0.", card)


def test_the_gn_two_spelling_is_the_same_ground_and_still_echoes_its_own_card():
    """``GN 2`` is the NEC-4-compatible spelling of ``GN 0`` and the dialect
    records which one arrived rather than normalizing it away.

    Both halves of that matter and they pull opposite ways: the ANSWER has to
    be identical to the last printed digit (probe family 1 measured the two
    grounds equal, so a seam that treated them differently would be inventing
    a distinction), while the CARD ECHO has to show the deck's own field,
    because the echo is a card image and 0047's shows ``GN   0``.  Comparing
    the whole printout catches both at once: exactly one line may differ.
    """
    zero = render(deck_text("0047"))
    two = render(_with_ground("0047", "GN 2,0,0,0,13.,.005,1.,0."))
    differ = [
        (a, b) for a, b in zip(zero.split("\n"), two.split("\n"), strict=True) if a != b
    ]
    assert len(differ) == 1
    assert differ[0][0].startswith(" ***** INPUT LINE  2  GN   0")
    assert differ[0][1].startswith(" ***** INPUT LINE  2  GN   2")


def test_a_negative_conductivity_field_is_the_imaginary_part_itself():
    """``GN 0,…,-12.84,…`` and ``GN 0,…,.005,…`` are ONE ground at 7 MHz.

    Measured on the linux oracle 2026-08-20: the negative spelling sets
    Im(εc) directly, the engine back-derives the conductivity from it, and
    both decks print ``CONDUCTIVITY= 5.000E-03``, ``1.30000E+01-1.28400E+01``
    and 4.7789E+01 − 7.8525E-01 Ω to every digit.  So this seam folds the
    spelling at the door and everything downstream sees one medium — which is
    what makes the two printouts below identical rather than merely close.

    The captures cannot say this: all four write a positive field.  It is here
    because the convention is real, the dialect's ``GD`` record already flags
    it, and a seam that read ``-12.84`` as a conductivity would hand momwire
    an ACTIVE ground and answer with a straight face.

    Gated at the printed digit rather than at the byte, and the gap between
    those two is the whole reason to say so: the equivalent σ is recovered by
    a DIVISION, so it is 0.005 to fifteen digits and not to sixteen, and the
    last ulps travel as far as the horizon rows' 2.4E-14 V of numerical dust
    (which prints ``-999.99`` on both sides either way).  Every cell that
    carries an antenna — the environment block, the impedance, all 543 gain
    cells, both tables — is identical.
    """
    plain = render(deck_text("0047"))
    folded = render(_with_ground("0047", "GN 0,0,0,0,13.,-12.84,1.,0."))
    assert "CONDUCTIVITY= 5.000E-03 MHOS/METER" in folded
    # The card echo is a card image and prints the deck's own field, which is
    # the same exemption `GN 2` gets above.
    lines = [
        (a, b)
        for a, b in zip(plain.split("\n"), folded.split("\n"), strict=True)
        if a != b and not a.startswith(" ***** INPUT LINE  2  GN")
    ]
    assert all(row[0][:72] == row[1][:72] for row in lines), lines

    want, got = extract(plain), extract(folded)
    assert (want.ground, want.environment) == (got.ground, got.environment)
    assert want.sources == got.sources
    assert want.currents == got.currents
    assert want.charges == got.charges
    (wp,), (gp,) = want.patterns, got.patterns
    for a, b in zip(wp.rows, gp.rows, strict=True):
        assert (a.vert_db, a.hor_db, a.total_db) == (b.vert_db, b.hor_db, b.total_db)


def test_the_engine_writes_no_sommerfeld_cache_and_reads_none(tmp_path):
    """``SOMMPD.NEX`` is inert at this seam: not read, not written, not made.

    The captured printouts are full of that file — stale, valid, missing,
    written — because the Windows engine caches its Sommerfeld tables there
    and reports what it found.  This engine has no such cache, so a run in a
    directory carrying one has to be BYTE-IDENTICAL to a run in a directory
    that does not, the file has to come back untouched, and no new one may
    appear.  Anything else is a seam that has started keeping state between
    runs in the user's model directory.
    """
    junk = b"not a Sommerfeld table\n\x00\xff"
    outputs = []
    for cache in (False, True):
        room = tmp_path / f"cache-{cache}"
        room.mkdir()
        deck = room / "EZN5.NEC"
        deck.write_bytes((FIXTURE_DIR / capture("0047")["deck"]).read_bytes())
        out = room / "NEC5.OUT"
        if cache:
            (room / "SOMMPD.NEX").write_bytes(junk)
        proc = subprocess.run(
            [sys.executable, "-m", "momwire.eznec", str(deck), str(out)],
            capture_output=True,
            text=True,
            cwd=room,
        )
        assert proc.returncode == 0
        outputs.append(out.read_bytes())
        assert sorted(p.name for p in room.iterdir()) == (
            ["EZN5.NEC", "NEC5.OUT", "SOMMPD.NEX"]
            if cache
            else ["EZN5.NEC", "NEC5.OUT"]
        )
    assert (tmp_path / "cache-True" / "SOMMPD.NEX").read_bytes() == junk
    assert mask(outputs[0].decode("latin-1")) == mask(outputs[1].decode("latin-1"))


def test_the_served_printout_carries_no_cache_preamble_of_its_own():
    """The other half of the same rule, read off the bytes.

    The captures' cache blocks are normalized out of the CAPTURED side
    (``test_eznec_printout.drop_sommpd_blocks``), and the structure gate would
    pass either way if the served side quietly grew one too — a normalization
    applied to both sides cannot see a line it deletes.  So the served side is
    checked directly: none of the four block forms may appear anywhere in it.
    """
    for cid in FINITE_IDS:
        text = served(cid)
        assert drop_sommpd_blocks(text) == text
        for marker in ("SOMMPD", "GMPINO", "Sommerfeld integral tables"):
            assert marker not in text, f"{cid}: {marker}"
    # And the banner that DOES belong to a finite ground is present, once.
    assert served("0047").count("FINITE GROUND.  SOMMERFELD SOLUTION") == 1


def test_the_epsilon_c_constant_is_the_engine_s_own_and_not_the_si_fold():
    """``εc = εr − j·σ·λ·59.96``, measured off the printed cell.

    0047 at 7.00 MHz prints ``1.30000E+01-1.28400E+01`` and 0048 at 7.02 MHz
    prints ``-1.28034E+01``; the SI fold ``σ/(ωε₀)`` gives 12.8393 and 12.8027
    and would print two different cells.  The difference is 5e-5 of the
    imaginary part — far below this seam's basis offset, and irrelevant to the
    solve — but it is the difference between a byte-gate that passes and one
    that does not, so the constant is pinned rather than derived.
    """
    assert _serve.EPSC_CONDUCTIVITY_FACTOR == 59.96
    for cid, wanted in (("0047", -12.8400), ("0048", -12.8034)):
        medium = extract(served(cid)).ground
        assert (medium.eps_r, medium.sigma) == (13.0, 0.005)
        # The cell is an E12.5, so "matches" means to the printed digit.
        assert medium.eps_c.imag == pytest.approx(wanted, abs=5e-5)
        assert medium == extract(printout_text(cid)).ground


# --------------------------------------------------------------------------
# gate 3 — the refusals
# --------------------------------------------------------------------------

# One capture per refusal, each the smallest deck in the corpus carrying the
# card.  `TL` and `NT` left this table with U5, which serves them; the deck
# that carries BOTH is refused for its table layout instead, and that refusal
# is gated in ``test_eznec_networks.py`` beside the rest of the network unit.
#
# 0031 is the corpus's own deck with its ground card swapped for `GN 1`, and it
# has to be: every captured multi-`EX` deck stands over a bare `GD`, so as
# written it is refused a rung earlier and its own card never gets a hearing.
# The card under test is the capture's verbatim (four phased `EX 4` rows); only
# the ground line moved.  0022 left this list with the ground rung — its
# `GN 0` is served now, so its `NE 0,1,1,1,…` is heard exactly as EZNEC wrote
# it, which is the point of the ordering test below.
_ON_PERFECT_GROUND = ("0031",)

REFUSALS = {
    "0045": "GD asks for the MININEC-type ground",
    "0031": "this deck carries 4 EX cards",
    "0022": "NE (near electric field) is not served at this seam yet",
}


def _refusal_deck(cid: str) -> bytes:
    text = (FIXTURE_DIR / capture(cid)["deck"]).read_bytes().decode("latin-1")
    if cid in _ON_PERFECT_GROUND:
        text = re.sub(r"^G[ND] 0,.*$", "GN 1", text, flags=re.MULTILINE)
    return text.encode("latin-1")


@pytest.mark.parametrize("cid,reason", sorted(REFUSALS.items()))
def test_an_out_of_scope_capture_refuses_by_name_through_the_shell(
    cid, reason, tmp_path
):
    """Named, after the comment echo, in a file that exists, at exit 0.

    All four obligations at once, because dropping any one of them is a
    refusal that never reaches the operator: EZNEC reads the printout and
    nothing else, and it discards one whose stamp echo is missing as belonging
    to an earlier run.
    """
    deck = tmp_path / "EZN5.NEC"
    deck.write_bytes(_refusal_deck(cid))
    out = tmp_path / "NEC5.OUT"

    proc = subprocess.run(
        [sys.executable, "-m", "momwire.eznec", str(deck), str(out)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )

    assert proc.returncode == 0
    written = out.read_bytes().decode("latin-1")
    error = " ***** NEC ERROR - "
    assert error in written
    stamp = "EZNEC Pro/2+ v. 7.0.4"
    assert written.index(stamp) < written.index(error)
    assert reason in written
    assert "ANTENNA INPUT PARAMETERS" not in written


def test_a_served_ground_hands_the_refusal_to_the_card_still_out_of_scope():
    """0022 carries BOTH a ``GN 0`` and an ``NE``, and which one speaks moved
    when the ground rung landed.

    It used to name the GROUND — a near field over a ground the seam could not
    solve is not a near-field problem, so naming ``NE`` there would have sent
    the reader after the wrong card.  Now the ground IS solved, so the request
    is the only thing left to fix and it is the request that answers.  The
    ordering in :func:`~momwire.eznec._serve.refusal` did not change; what
    changed is which of its rungs this deck falls through, which is the shape
    a ladder is supposed to have.
    """
    printout = render(deck_text("0022"))
    assert REFUSALS["0022"] in printout
    assert "GN 0" not in printout.split(" ***** NEC ERROR - ")[1]


def test_the_stub_refusal_no_longer_answers_anything_in_the_corpus():
    """U1's catch-all is now a backstop and not a behaviour.

    Every one of the 49 captured decks comes back either solved or refused by
    a sentence that names its card, so the stub reason — which said only that
    the dialect was unserved — must appear nowhere.
    """
    for path in sorted((FIXTURE_DIR / "decks").glob("*.nec")):
        text = path.read_bytes().decode("latin-1")
        assert "NEC-5 DIALECT NOT YET SERVED" not in render(text), path.name
        assert "INTERNAL ERROR IN MOMWIRE ENGINE" not in render(text), path.name


@pytest.mark.parametrize("cid", SERVED_UNGATED_IDS)
def test_the_ungated_rung_one_captures_still_serve(cid):
    """0036-0042 are the same dipole at seven frequencies with ``XQ`` and no
    ``RP``, so they carry no printout to compare against — but a deck that
    started refusing would be a regression, and the frequency block is one
    thing about them that IS checkable."""
    text = render(deck_text(cid))
    assert "NEC ERROR" not in text
    assert "- - - ANTENNA INPUT PARAMETERS - - -" in text
    assert "- - - RADIATION PATTERNS - - -" not in text
    # The FREQUENCY cell is an E11.4, so a deck asking for 299.793 MHz gets
    # 2.9979E+02 back and the comparison is at the printed precision.
    assert extract(text).frequency_mhz == pytest.approx(
        parse_nec5(deck_text(cid)).frequency_mhz, rel=1e-4
    )


@pytest.mark.parametrize("cid", FINITE_UNGATED_IDS)
def test_the_ungated_finite_ground_captures_answer_with_finite_numbers(cid):
    """The four decks the ground rung brought in with no printout to gate.

    A liveness gate and it says so: with nothing to compare against, the
    honest claims are that the deck is answered rather than refused, that it
    is answered under the FINITE GROUND banner, that its ``RP`` came back with
    all 181 rows, and that no cell in it is a NaN or an infinity — which is
    the failure mode a finite ground actually has, since ``sqrt(εc − sin²θ)``
    and a wire 1e-4 λ off the plane are both places an answer can stop being a
    number without stopping being printed.

    0011/0030 are the first NETWORK over a finite ground anywhere in this
    corpus (a coax ``TL`` from a dipole down to a feedpoint), and 0033/0034
    are elevated radial systems — the geometry the reflection-coefficient
    ground model would have been wrong about and the Sommerfeld one is not
    (``docs/refl-coef-ground-plan.md``, momwire#151).
    """
    text = render(deck_text(cid))
    assert "NEC ERROR" not in text
    assert "FINITE GROUND.  SOMMERFELD SOLUTION" in text
    for token in ("NAN", "nan", "INF", "Infinity", "*****E"):
        assert token not in text.replace(" ***** INPUT LINE", ""), token
    data = extract(text)
    (block,) = data.patterns
    assert len(block.rows) == 181
    assert all(math.isfinite(row.total_db) for row in block.rows)
    assert math.isfinite(abs(data.sources[0].impedance))
    assert data.ground is not None


def test_the_favored_wire_carries_physics_at_a_five_wire_junction():
    """0013's apex is one geometric point that five ``GW`` cards name, and
    WHICH card names it changes the answer.

    ``EX 4,5,-1`` — the capture's own card, the source in series with the
    vertical's first element — is served as 58.876 + 19.210 Ω.  Move the same
    card to ``EX 4,1,-1``, the identical point through a radial's tag, and the
    seam answers 43.684 − 7.5465 Ω: 35 % away, because the gap now separates a
    different arm from the node.  The four symmetric radials answer alike,
    which is the control — a difference between tag 1 and tag 2 would be a
    bug, and a difference between tag 1 and tag 5 is the physics.

    This is the W7EL oracle's principle one unit early.  A seam that
    canonicalized ``(tag, node)`` into a geometric point could not tell these
    three cards apart, and the gate unit's 70 % error is the same mistake
    further downstream.
    """
    vertical = _z_of(deck_text("0013"))
    radial_one = _z_of(deck_text("0013").replace("EX 4,5,-1,", "EX 4,1,-1,"))
    radial_two = _z_of(deck_text("0013").replace("EX 4,5,-1,", "EX 4,2,-1,"))
    assert radial_one == radial_two
    assert abs(vertical - radial_one) / abs(vertical) > 0.3


def test_a_source_on_a_free_wire_end_refuses_rather_than_guessing():
    """Nothing carries current past a lone conductor end, so there is no path
    for a series source to sit in — momwire refuses one at its constructor and
    this seam refuses it earlier, with the address in the sentence.

    Built from 0010 by moving its ``EX`` to node 0 of the same wire, which in
    free space is the dipole's tip.  No capture asks for this; the refusal
    exists so that one arriving is a sentence rather than a traceback.
    """
    text = deck_text("0010").replace("EX 4,1,6,", "EX 4,1,-1,")
    printout = render(text)
    assert "addresses a FREE end of wire 1" in printout
    assert "ANTENNA INPUT PARAMETERS" not in printout


# --------------------------------------------------------------------------
# LD 4 — in scope, and with no capture of its own
# --------------------------------------------------------------------------
#
# Rung 1 includes `LD 4`, but every captured deck that carries one also carries
# a network card (the loaded decks in the corpus are exactly the network ones),
# so the loaded path is byte-gated in `test_eznec_networks.py` and not here.
# These two tests are what this unit has instead: an identity that holds
# exactly, and the `1.E+10` pin idiom doing the one thing it exists to do.


def test_a_load_at_the_driven_node_adds_its_own_ohms_to_the_feedpoint():
    """An ``LD 4`` is an impedance in the port's own current path, so one
    stamped at the DRIVEN node is in series with the source and moves the
    printed impedance by exactly itself — an identity, not an approximation,
    and the sharpest available check that the port algebra put the load where
    NEC puts it."""
    bare = _z_of(deck_text("0010"))
    loaded = _z_of(deck_text("0010").replace("PQ 0\n", "LD 4,1,6,0,50.,25.\nPQ 0\n"))
    # The cells are E12.4, so "exactly" means to the printed digit: the
    # measured shift is 49.997 + 25.000j against a 135 Ω row.
    assert loaded - bare == pytest.approx(complex(50.0, 25.0), abs=0.02)


def test_the_pin_idiom_leaves_the_virtual_wire_carrying_nothing():
    """W7EL's own idiom, on a deck that has no network to need it yet.

    The ``Network Connection Test`` captures build a 100 λ-away three-segment
    wire and pin both of its interior nodes with ``LD 4,4,n,0,1.E+10,0.``, so
    the wire exists only to hang network cards off.  Bolted onto 0010 — same
    ``GW`` card, same two ``LD`` cards, verbatim — it has to be invisible:
    the dipole's feedpoint impedance must not move by so much as a printed
    digit, and the pinned wire's own current must be nothing at all.

    Measured: the virtual wire carries 4.7e-17 A against the dipole's 1.44 A,
    seventeen orders down, and the loading table renders the two rows exactly
    as 0012 printed them.
    """
    virtual = "GW 4,3,99.99998,99.99998,99.99998,100.003,100.003,100.003,1.0000E-4\n"
    pins = "LD 4,4,1,0,1.E+10,0.\nLD 4,4,2,0,1.E+10,0.\n"
    text = deck_text("0010").replace("GE 0,-1\n", virtual + "GE 0,-1\n")
    text = text.replace("PQ 0\n", pins + "PQ 0\n")

    printout = render(text)
    data = extract(printout)
    assert data.sources[0].impedance == extract(served("0010")).sources[0].impedance
    dipole = max(row.magnitude for row in data.currents if row.tag == 1)
    assert max(row.magnitude for row in data.currents if row.tag == 4) < 1e-12 * dipole
    # 0012's own two rows, byte for byte (the trailing blank in the TYPE field
    # is real).
    rows = [line for line in printout.split("\n") if "FIXED IMPEDANCE" in line]
    captured = [
        line for line in printout_text("0012").split("\n") if "FIXED IMPEDANCE" in line
    ]
    assert rows == captured


# --------------------------------------------------------------------------
# the charge readout's own gate
# --------------------------------------------------------------------------
#
# `BSplineSolver.current_slopes` is new with this unit and the charge block is
# its only consumer, so its gate lives here: the envelope pins above say it
# lands near NEC-5's charge densities, and these two say it is the derivative
# it claims to be rather than something that happens to be close.


@functools.lru_cache(maxsize=None)
def _dipole_solution():
    """0010's dipole, solved directly, with its element-centre arc positions."""
    solver = BSplineSolver(
        wires=[np.array([[0.0, -0.25, 0.0], [0.0, 0.25, 0.0]])],
        n_per_edge_per_wire=[[11]],
        feeds=[(0, 6 * 0.5 / 11, 1.0 + 0j)],
        wire_radius=0.0005,
        wavelength=299.8 / 299.7925,
    )
    coeffs = solver.compute_port_solution().coeffs[:, 0]
    step = 0.5 / 11
    return solver, coeffs, [(np.arange(11) + 0.5) * step]


def test_the_slope_readout_is_the_current_s_own_derivative():
    """``current_slopes`` against a central difference of ``currents_at_knots``.

    An independent check by construction — one evaluates the derivative
    spline, the other samples the value spline twice — and it has to agree to
    the difference's own truncation error, which at ``h = 1e-6`` on a segment
    of 0.045 m is parts in 1e9.  Taken at segment CENTRES because that is
    where the charge block reads it and because a degree-2 spline's slope,
    while continuous, is only piecewise smooth at a knot.
    """
    solver, coeffs, centres = _dipole_solution()
    exact = solver.current_slopes(coeffs, centres)
    h = 1e-6
    ahead = solver.currents_at_knots(coeffs, [c + h for c in centres])
    behind = solver.currents_at_knots(coeffs, [c - h for c in centres])
    for got, up, down in zip(exact, ahead, behind):
        difference = (up - down) / (2.0 * h)
        scale = max(abs(v) for v in difference)
        assert max(abs(got - difference)) < 1e-6 * scale


def test_the_slope_readout_refuses_the_enriched_basis_rather_than_dropping_it():
    """The singular enrichment's shape has a log-divergent slope at the
    junction knot, so a readout that ignored it would be wrong exactly where
    the charge density is most interesting.  It says so instead."""
    solver = BSplineSolver(
        wires=[np.array([[0.0, -0.25, 0.0], [0.0, 0.25, 0.0]])],
        n_per_edge_per_wire=[[11]],
        feeds=[(0, 0.25, 1.0 + 0j)],
        wire_radius=0.0005,
        wavelength=1.0,
        use_singular_enrichment=True,
    )
    with pytest.raises(NotImplementedError, match="singular"):
        solver.current_slopes(np.zeros(1), [np.zeros(1)])


# --------------------------------------------------------------------------
# the constants a reader would otherwise have to take on trust
# --------------------------------------------------------------------------


def test_the_wavelength_is_nec_s_own_metre_megahertz_product():
    """299.8, not the SI c.  0019 prints ``WAVELENGTH= 4.2829E+01 METERS`` at
    7 MHz: 299.8/7 = 42.8286 rounds to that cell and 299792458/7e6 = 42.8275
    does not.  The same constant normalizes the geometry columns of the
    current and charge tables, which is why gate 1c passes at 1e-9 rather
    than at 1e-5."""
    assert _serve.SPEED_OF_LIGHT_MHZ_M == 299.8
    for cid in SERVED_IDS:
        wanted = 299.8 / parse_nec5(deck_text(cid)).frequency_mhz
        # Both cells are E11.4, so the comparison is at printed precision —
        # which is exactly where 299.8 and the SI c part company (0019 prints
        # 4.2829E+01; the SI c would print 4.2827E+01).
        assert extract(served(cid)).wavelength_m == pytest.approx(wanted, rel=1e-4)
        assert extract(printout_text(cid)).wavelength_m == pytest.approx(
            wanted, rel=1e-4
        )


def test_the_basis_is_the_house_default():
    """``bspline`` — ``BSplineSolver`` at degree 2, which is what
    ``momwire.deck.build_solver``'s default entry constructs and what the
    portal answers every NEC-2 deck with.  Recorded as a gate because a
    silent change of basis would move every number in every printout this
    seam writes and break nothing else."""
    assert _serve.BASIS == "bspline"
    assert _serve._DEGREE == 2


def test_every_refusal_sentence_survives_the_printout_s_own_codec():
    """A printout is written latin-1 (U1's ``_CODEC``), so a refusal carrying
    a character that codec cannot spell is a refusal that never gets written.

    Measured the hard way: the ``TL`` and ``NT`` sentences (U4's, since
    retired) were first drafted with an em dash, and the shell answered 0012
    with an internal-error printout instead of the sentence.  This gate is why
    they are ASCII.
    """
    for name, value in vars(_serve).items():
        if name.startswith("_REFUSE") and isinstance(value, str):
            value.encode("latin-1")


def test_the_module_never_reaches_for_a_second_far_field_implementation():
    """The pattern readout is the portal's, by import: one owner for the
    moments, the gain floor and the polarisation ellipse (the ranked
    extraction backlog, momwire#429, is where that import is paid off)."""
    source = Path(_serve.__file__).read_text()
    assert "from ..portal._portal import" in source
    assert "def _far_moments" not in source
