"""Rung-1 to rung-3 physics, gated (#497 U4, #504 U1, #504 U2, #504 U3).

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

**The refusals.**  Everything above rung 3 still refuses BY NAME, through the
real shell, at exit 0, with the comment stamp intact — because a refusal that
does not reach EZNEC's viewer reaches nobody.  With the ground cards finished
there is nothing hand-edited left in the refusal tables: every deck in them is
the capture verbatim, reaching its own card at last.

**And one deck class that is gated by neither.**  #504 U3 landed a printout
for every served capture that had one, which is what turned the U1/U2
liveness gates into structure gates and envelope pins — except on 0033/0034,
where the numbers arrived and DISAGREED, far outside the family (see
:data:`DIVERGENT_IDS`).  Their layout is gated like everything else and their
physics is not pinned at all, because a bar drawn round a 275 Ω gap would be
recording the gap as normal rather than as the finding it is.

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

# The far-field readout the seam imports, imported here for the one gate that
# is ABOUT that import: `refl` and `sommerfeld` are one branch of it.
from momwire.portal._portal import Ground, _far_moments
from test_eznec_printout import (
    FIXTURE_DIR,
    GATED_IDS,
    MANIFEST,
    capture,
    deck_text,
    drop_sommpd_blocks,
    extract,
    printout_text,
)

# The BARE-STRUCTURE captures whose physics is PINNED — the thirteen this unit
# measures envelopes for.  The other gated captures carry ``TL``/``NT`` cards
# and belong to the network unit, which measures its own table in
# ``test_eznec_networks.py``; they still appear below, in the counting-rule
# gate, because the count is a property of the deck's geometry and not of what
# answers it.
SERVED_IDS = (
    "0010",
    "0013",
    "0015",
    "0019",
    "0020",
    "0021",
    "0031",
    "0032",
    "0035",
    "0043",
    "0044",
    "0045",
    "0046",
    "0047",
    "0048",
)

# The two PHASED captures — #504 U4's rung, and the corpus's only decks whose
# ``ANTENNA INPUT PARAMETERS`` table has more than one row.  0031 is a 40 m
# four-square: four 9.98 m verticals on a 10.49 m square over a bare ``GD``,
# driven ``1.414214∠0``, ``-j1.414214`` twice and ``-1.414214`` — the classic
# 0/-90/-90/-180 four-square phasing — with a 361-point azimuth cut at
# theta = 67.  0032 is a two-element cardioid, quarter-wave spacing and
# quarter-wave phasing at 299.7925 MHz over ``GN 1``, with a 361-point azimuth
# cut at the horizon.
#
# Both drive their ports with a set CURRENT and no network anywhere, which is
# not a coincidence: it is how EZNEC spells a phased array in this dialect, and
# it is the whole of what rung 4 serves.
PHASED_IDS = ("0031", "0032")

# The three of them that stand over a finite ``GN 0`` ground — the rung #504 U1
# added.  One 10.3 m base-fed vertical over 13/0.005 earth, three times:
# 0047 with a 181-point elevation cut at 7.00 MHz, 0021 its 2026-08-16 twin
# (same deck, different launch stamp, a differently stale ``SOMMPD.NEX``), and
# 0048 the same antenna answered by ``XQ`` at 7.02 MHz.
FINITE_IDS = ("0021", "0047", "0048")

# The four that stand over a bare ``GD`` — U2's rung, and the same 10.3 m
# vertical again.  0045/0015/0020 are ONE deck (``GD 0,…,13.,.005,1.,0.`` with
# a 181-point cut at 7.00 MHz) captured at three separate launches, and 0046 is
# its ``XQ`` twin at 7.01 MHz.  Paired against 0047/0021/0048 they are the
# corpus's controlled experiment: same wire, same medium, same frequency, one
# card changed.
MININEC_IDS = ("0015", "0020", "0045", "0046")

# The rung-1 captures with no printout: seven ``XQ``-only dipoles EZNEC wrote
# while stepping a frequency.  Nothing to byte-compare, but a deck that
# refuses is a deck that stopped being served, so they are run for their exit
# status.
SERVED_UNGATED_IDS = ("0036", "0037", "0038", "0039", "0040", "0041", "0042")

# The two elevated radial systems, and the only decks in the corpus whose
# printout landed and whose numbers then did not fit.  Both are the same
# antenna at 1.832 MHz — a 35.56 m vertical over four 39.624 m radials — with
# every radial and the feedpoint standing 1.778 cm above a ``GN 0`` earth,
# which is 1.09e-4 of a 163.6 m wavelength.  0033 meshes it in five segments a
# wire, 0034 in a 44-wire taper down to 1.8 cm elements.
#
# What is gated: the STRUCTURE, exactly as for every other capture — the
# counting rule, the geometry columns, every heading and row and null.  All of
# it passes, which is what says the divergence is not this seam's addressing.
#
# What is NOT gated: the physics.  Measured 2026-08-20 (#504 U3):
#
#   id    Z (capture)          Z (served)          |dZ|     peak cap/served
#   0033  38.791 -49.583j      45.325+225.59j     275.25    -0.76 / -1.26
#   0034  40.730 -43.452j      91.615+137.87j     188.33    -0.91 / -4.22
#
# The family this table sits in spans 0.05 to 19.5 Ω and 0.01 to 0.25 dB, so
# these are not wide envelopes, they are a different answer: the reactance
# changes SIGN.  Probing the ground card (the same deck at ``GN 1``, ``GD``,
# ``GN -1``, and lifted to 1 m and 5 m) puts it in momwire's finite-ground
# solve rather than at this seam — 0033 answers 25.550 − 83.466j over a
# perfect image, 25.659 − 128.65j at 1 m and 16.103 − 234.55j at 5 m, so the
# height dependence is steep and the 1.8 cm answer is the outlier of its own
# family.  That is momwire#151/#157 territory (a Sommerfeld grid asked about a
# horizontal wire 1e-4 λ up), it is not fixable inside this package, and it is
# reported rather than pinned: a bar at 344 Ω would record the gap as normal.
DIVERGENT_IDS = ("0033", "0034")

# Every capture id this seam answers after #504 U4, pinned.  The ladder number
# is 48 of 49 and the one that is left is named in
# :func:`test_the_corpus_ladder_stands_where_u4_left_it`.
SERVED_AFTER_U4 = (
    "0000",
    "0001",
    "0002",
    "0003",
    "0004",
    "0005",
    "0006",
    "0007",
    "0008",
    "0009",
    "0010",
    "0011",
    "0012",
    "0013",
    "0014",
    "0015",
    "0016",
    "0017",
    "0018",
    "0019",
    "0020",
    "0021",
    "0023",
    "0024",
    "0025",
    "0026",
    "0027",
    "0028",
    "0029",
    "0030",
    *PHASED_IDS,  # 0031-0032
    *DIVERGENT_IDS,  # 0033-0034
    "0035",
    *SERVED_UNGATED_IDS,  # 0036-0042
    "0043",
    "0044",
    "0045",
    "0046",
    "0047",
    "0048",
)


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
#   0015  35.571 - 1.4223j     36.499 + 2.0789j    3.622  -0.02/-0.05  0.03
#   0019  35.571 - 1.4223j     36.499 + 2.0789j    3.622   (XQ, none)   —
#   0020  35.571 - 1.4223j     36.499 + 2.0789j    3.622  -0.02/-0.05  0.03
#   0021  47.789 - 0.78525j    48.867 + 2.5635j    3.518  -1.31/-1.33  0.02
#   0031  -1.7790-24.192j      -1.7032-17.926j     6.267   5.38/5.32   0.06
#   0032  19.139 -24.569j      19.769 -18.570j     6.032   8.23/8.18   0.05
#   0035  23.343 -24.594j      23.926 -20.079j     4.552   9.88/9.84   0.04
#   0043  35.571 - 1.4223j     36.499 + 2.0789j    3.622   (XQ, none)   —
#   0044  35.571 - 1.4223j     36.499 + 2.0789j    3.622   5.15/5.13   0.02
#   0045  35.571 - 1.4223j     36.499 + 2.0789j    3.622  -0.02/-0.05  0.03
#   0046  35.735 - 0.68499j    36.675 + 2.8243j    3.633   (XQ, none)   —
#   0047  47.789 - 0.78525j    48.867 + 2.5635j    3.518  -1.31/-1.33  0.02
#   0048  48.155 + 0.65170j    49.254 + 4.0055j    3.529   (XQ, none)   —
#
# The impedance bar is that |dZ| plus 25 %.  The reactance carries almost all
# of it — a node source and a segment gap store different amounts of energy in
# the feed region, which is exactly the formulation difference the scored
# matrix priced when it moved this unit's gate off byte equality.
#
# The ``GN 0`` rows are the SMALLEST offsets in the table (3.5 Ω on a 48 Ω row,
# against 16 on the free-space dipole), and the reason is worth writing down:
# the Sommerfeld ground loads the feed region, so the same formulation
# difference sits on a bigger, lossier admittance and shows less.  The offset
# is also within 3 % of the perfect-ground twins' 3.622 on the same 10.3 m wire
# (0019/0043/0044), which says the ground model added almost none of it — the
# gap is the feed's, as it is everywhere else in this table.
#
# The four ``GD`` rows carry that argument to its end.  0015/0020/0045 do not
# merely sit NEAR 0019/0043/0044's row, they ARE it — 35.571 − 1.4223j captured
# and 36.499 + 2.0789j served, the same |dZ| = 3.622 to fifteen digits — because
# the MININEC-type ground solves over a perfect image and the two decks are one
# solve wearing two ground cards.  A ``GD`` row that drifted off the
# perfect-ground row would be the aliasing bug, visible right here in the table
# before any dedicated gate ran.
#
# The peak-gain bar is |d| plus 25 % OR 0.05 dB, whichever is larger: the
# printed cell is quantized at 0.01 dB, so a bar under a few hundredths would
# be pinning the rounding rather than the physics.
#
# The two PHASED rows are this table's FIRST rows rather than its only ones —
# 0031 prints four ``ANTENNA INPUT PARAMETERS`` rows and 0032 two, and every
# one of them is pinned in :data:`PHASED_Z_BAR` below.  The row quoted here is
# the first, so that the sweep this table drives keeps one shape.  0031's first
# row is also the corpus's only NEGATIVE resistance, which makes it the one
# entry in this table where the served number has to land inside the bar AND on
# the correct side of zero (:func:`test_the_four_square_s_absorbing_element_
# stays_negative`).
Z_BAR = {
    "0010": 20.35,
    "0013": 14.27,
    "0015": 4.53,
    "0019": 4.53,
    "0020": 4.53,
    "0021": 4.40,
    "0031": 7.84,
    "0032": 7.55,
    "0035": 5.69,
    "0043": 4.53,
    "0044": 4.53,
    "0045": 4.53,
    "0046": 4.55,
    "0047": 4.40,
    "0048": 4.42,
}

# Every row of the two phased tables, measured 2026-08-20 the same way and
# barred the same way (|dZ| plus 25 %).  Written per ROW because that is what a
# phased drive newly has and what a regression would newly break: the four
# ports of a four-square are four different impedances, coupled, and a seam
# that solved one of them wrong could still average out into a plausible first
# row.
#
#   id    tag  Z (capture)          Z (served)          |dZ|
#   0031   1   -1.7790-24.192j      -1.7032-17.926j     6.267
#   0031   2   36.783 -27.955j      37.749 -22.631j     5.411
#   0031   3   36.783 -27.955j      37.749 -22.631j     5.411
#   0031   4   56.525 +41.575j      59.333 +49.392j     8.306
#   0032   1   19.139 -24.569j      19.769 -18.570j     6.032
#   0032   2   49.318 +12.169j      51.224 +19.361j     7.440
#
# The family is the same one the single-source table sits in — the reactance
# carries nearly all of it, 5.3 to 7.8 Ω of X against 0.08 to 2.8 Ω of R, which
# is the feed-region formulation difference these captures' six-segment
# verticals show at exactly the size 0010's dipole shows it at 16.3.  Nothing
# about the PHASING is loose: the two symmetric rows of 0031 are identical on
# both sides (:func:`test_the_four_square_s_two_quadrature_elements_print_one_
# row_twice`), which is a statement about the solve that no envelope makes.
PHASED_Z_BAR = {
    "0031": (7.84, 6.77, 6.77, 10.39),
    "0032": (7.55, 9.31),
}

# (peak total gain dB, theta, phi) as the capture prints it, and the bar.
PEAK_BAR = {
    "0010": 0.12,
    "0013": 0.13,
    "0015": 0.05,
    "0020": 0.05,
    "0021": 0.05,
    "0031": 0.075,
    "0032": 0.07,
    "0035": 0.05,
    "0044": 0.05,
    "0045": 0.05,
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
#
# The same 0.06 on the ``GD`` cuts (0015/0020/0045), which is the number that
# says the MININEC mode's far field is the RIGHT Fresnel-weighted image and not
# just a plausible one: the currents underneath it are a different solve
# (PEC, not Sommerfeld-loaded), so a cut that agreed at 178 angles by accident
# would have had to do it twice over.
PATTERN_BAR = 0.075

# The AZIMUTH cuts get the same every-angle treatment and cannot get the same
# single bar, because what they sweep through is not a horizon — it is an
# array's own null, and a null is where two nearly-equal fields cancel and a
# small difference in either becomes a large difference in the remainder.
#
# So each phased cut is gated twice.  ``AZIMUTH_BAR`` is the whole 361 rows,
# nulls included, at the measured worst plus 25 %; ``AZIMUTH_LOBE_BAR`` is the
# rows within 10 dB of the peak, where the level is a level rather than a
# cancellation, at the same plus 25 %.  Measured 2026-08-20:
#
#   id    rows  worst (all)  worst (<=10 dB down)   depth of the deepest null
#   0031   361     0.58 dB          0.13 dB          -25.32 dB (30.7 down)
#   0032   361     0.76 dB          0.09 dB          -25.24 dB (33.5 down)
#
# The shape of that is the evidence, and it is worth more than either number:
# the error climbs MONOTONICALLY as the captured level falls — 0031 runs 0.07 /
# 0.09 / 0.13 / 0.19 / 0.29 / 0.36 / 0.58 dB as the floor drops 3 / 6 / 10 / 15
# / 20 / 25 / 30 dB below its peak — which is what a small current difference
# does to a cancellation and is not what a wrong phasing, a wrong drive
# convention or a mis-normalized pattern does to one.  Those show up at the
# PEAK, and the peak is inside 0.06 dB on both.
AZIMUTH_BAR = {"0031": 0.73, "0032": 0.95}
AZIMUTH_LOBE_BAR = {"0031": 0.17, "0032": 0.12}
# How far below the peak "the lobe" reaches, in dB.
_LOBE_FLOOR = 10.0

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
    "0015": 0.0129,
    "0019": 0.013,
    "0020": 0.0129,
    "0021": 0.0123,
    "0031": 0.0265,
    "0032": 0.0164,
    "0035": 0.228,
    "0043": 0.013,
    "0044": 0.013,
    "0045": 0.0129,
    "0046": 0.0129,
    "0047": 0.0123,
    "0048": 0.0124,
}
CHARGE_BAR = {
    "0010": 0.086,
    "0013": 0.078,
    "0015": 0.0777,
    "0019": 0.078,
    "0020": 0.0777,
    "0021": 0.076,
    "0031": 0.0842,
    "0032": 0.0587,
    "0035": 0.186,
    "0043": 0.078,
    "0044": 0.078,
    "0045": 0.0777,
    "0046": 0.0778,
    "0047": 0.076,
    "0048": 0.077,
}


@functools.lru_cache(maxsize=None)
def served(cid: str) -> str:
    """The printout this engine writes for a capture's deck."""
    return render(deck_text(cid))


@functools.lru_cache(maxsize=None)
def corpus() -> dict[str, str]:
    """Every one of the 49 captured decks, rendered once.

    Two gates read the whole corpus — the served-id ladder and the catch-all
    stub — and rendering it twice would double the slowest thing in this file
    for nothing.  Cached here so the sweep happens once per session.
    """
    return {
        entry["id"]: render(deck_text(entry["id"])) for entry in MANIFEST["captures"]
    }


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

# ANTENNA INPUT PARAMETERS: three integers, then nine E12.4 cells.  Which pair
# of them is the CARD's rather than the solver's follows the EX card's own
# kind, and #504 U3's captures are what forced that to be a lookup instead of a
# constant: an `EX 4` deck sets its CURRENT (cells 2 and 3) and reads back a
# voltage, an `EX 0` deck sets its VOLTAGE (cells 0 and 1) and reads back a
# current.  0033 and 0034 are the corpus's only two `EX 0` decks and they
# arrived with this unit; before them every gated capture was `EX 4` and the
# exemption could be — and was — written down as one pair.
_DRIVE_CELLS = {0: (0, 1), 4: (2, 3)}
# The network table shares the format and not that exemption: the current at a
# connection point is the STRUCTURE's, solved, even on the row whose port the
# deck drives (0027 prints 1.4142E+00 under ANTENNA INPUT PARAMETERS and
# 3.3124E-09 for the same port here — all of the difference went down the two
# transmission lines).
_NETWORK_PORT_CELLS = tuple(range(9))


def drive_cells(cid: str) -> tuple[int, ...]:
    """The ``ANTENNA INPUT PARAMETERS`` cells this capture's ``EX`` cards SET.

    Read off the deck, because that is where the answer is: a printout cannot
    say which of its own numbers was a boundary condition.

    ONE answer for the whole table, which is a claim and not a convenience: a
    deck whose cards disagreed about their kind refuses by name
    (``_REFUSE_MIXED_DRIVE_KINDS``), so every row a served table carries was
    fixed in the same two cells.  #504 U4's two multi-``EX`` captures write
    ``EX 4`` four times and twice, and the set is asserted rather than sampled.
    """
    kinds = {source.kind for source in parse_nec5(deck_text(cid)).sources}
    assert len(kinds) == 1, f"{cid}: mixed EX kinds {sorted(kinds)}"
    return _DRIVE_CELLS[kinds.pop()]


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


def _mask_port_row(line: str, solved: tuple[int, ...]) -> str:
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


def mask(text: str, drive: tuple[int, ...] = (2, 3)) -> str:
    """A printout with its solved cells replaced, table by table.

    ``drive`` names the two ``ANTENNA INPUT PARAMETERS`` cells the deck's own
    ``EX`` card fixed, which survive unmasked; :func:`drive_cells` reads it off
    the deck.  The default is the ``EX 4`` pair, which is what every capture in
    the corpus but 0033 and 0034 writes.

    Also applies the manifest's two remaining served-run normalizations: the
    timing lines are dropped outright, and nothing else about them is looked
    at.
    """
    port_cells = tuple(k for k in range(9) if k not in drive)
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
            out.append(_mask_port_row(line, port_cells))
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


@pytest.mark.parametrize("cid", SERVED_IDS + DIVERGENT_IDS)
def test_a_served_printout_is_the_capture_wherever_it_is_not_a_solved_number(cid):
    """The unit's whole shape, fifteen times over.

    What this proves, and it is most of the file: the row counts, the tag and
    element numbering, the geometry summary, the ``ALLOCATE CM`` allocation,
    the card echoes, the environment banner, the frequency and wavelength, the
    drive cell, the section order and the blank lines between sections are all
    correct — and the ``-999.99`` nulls fall on exactly the directions the
    capture put them, with SENSE blank on exactly the same rows.

    The two divergent captures are here and belong here.  A printout whose
    numbers are wrong can still have every heading, count and null right, and
    saying which half is which is the entire value of splitting a structure
    gate from an envelope: 0033 and 0034 pass this line and are pinned
    nowhere, so the finding stays a finding instead of hiding inside a wide
    bar.
    """
    assert mask(served(cid), drive_cells(cid)) == mask(
        printout_text(cid), drive_cells(cid)
    )


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
    written = out.read_bytes().decode("latin-1").replace("\r\n", "\n")
    assert "NEC ERROR" not in written
    assert mask(written) == mask(printout_text("0044"))


# --------------------------------------------------------------------------
# gate 1b — the counting rule, on every capture that shipped a printout
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

    Twenty-nine captures have been added since the rule was written and every
    one of them answered it unchanged, which is the only kind of evidence a
    derived rule can get.  0012's 17 nodes / 15 elements / 13 unknowns is the
    deck with several two-wire junctions AND a wire that touches nothing; 0027
    pairs a ``GE 1`` ground plane with a virtual wire far above it; and #504
    U3's 0034 is the hardest of them, 44 wires meeting five at a time at a
    tapered base, 81 nodes / 80 elements / 79 unknowns.
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


@pytest.mark.parametrize("cid", SERVED_IDS + DIVERGENT_IDS)
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


# The 181-row elevation cuts and the ``-999.99`` rows each puts where.  The
# five verticals null at the horizon AND at the zenith; the three coax-fed
# dipoles (#504 U3's, and the corpus's only networks over a finite ground with
# an elevation cut) null at the horizon only, because a horizontal dipole's
# field does not vanish overhead.
_ELEVATION_NULLS = {
    "0011": [90.0, -90.0],
    "0015": [90.0, 0.0, -90.0],
    "0020": [90.0, 0.0, -90.0],
    "0021": [90.0, 0.0, -90.0],
    "0029": [90.0, -90.0],
    "0030": [90.0, -90.0],
    "0045": [90.0, 0.0, -90.0],
    "0047": [90.0, 0.0, -90.0],
}


@pytest.mark.parametrize("cid", sorted(_ELEVATION_NULLS))
def test_the_finite_ground_elevation_cut_agrees_at_every_printed_angle(cid):
    """All 181 rows of the ``RP 0`` cut, not just its peak.

    Over a real ground the whole elevation pattern is the ground's doing — the
    direct wave and its Fresnel-weighted image interfere direction by
    direction — so agreeing at 181 angles is a statement about the MEDIUM in a
    way that agreeing at the peak is not.  Worst row is 0.06 dB, at
    theta = ±1° and ±2°, where the cut falls through 30 dB in two degrees.

    Both finite modes are here, over the same medium and the same wire: 0021
    and 0047 solve IN it, 0015/0020/0045 only reflect off it.  Same bar, same
    worst row, same three nulls — the medium is doing the same job in both, and
    what differs is the current it is doing it to.

    #504 U3 adds a third shape to that list and it is the one that generalizes
    the claim off the base-fed vertical: 0011/0029/0030 are a half-wave dipole
    9.144 m up with a coax ``TL`` running down to a feedpoint 3 mm off the
    ground, over 20/0.0303 earth, and their cuts agree at 0.04 dB across all
    179 non-null rows.  A different antenna, a different height, a different
    medium, a network in the middle of it — and both ground modes are
    represented (0011/0030 ``GN 0``, 0029 ``GD``), so the same three decks say
    the two modes' far fields are each right rather than jointly plausible.

    The ``-999.99`` rows are skipped HERE and gated harder elsewhere: they are
    never masked, so the structure gate already compares them byte for byte and
    would fail if the served run put a null anywhere the capture did not.
    Which it does not — the horizon nulls at θ = ±90° (nothing over any ground
    has a grazing field) and the vertical's zenith null at θ = 0° land on
    exactly the captured rows, and the dipoles' overhead rows carry numbers on
    both sides.
    """
    (want,) = extract(printout_text(cid)).patterns
    (got,) = extract(served(cid)).patterns
    assert len(got.rows) == len(want.rows) == 181
    nulls = [row.theta_deg for row in want.rows if row.total_db == _NULL_DB]
    assert nulls == _ELEVATION_NULLS[cid]
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


@pytest.mark.parametrize("cid", SERVED_IDS + DIVERGENT_IDS)
def test_a_lossless_deck_radiates_everything_it_is_given(cid):
    """Every wire here is a perfect conductor and this dialect has no
    conductivity card, so INPUT = RADIATED, WIRE LOSS = 0 and EFFICIENCY reads
    100.00 — which is what all fifteen captures print.

    Including the three over LOSSY GROUND, which is the entry worth reading
    twice: 0047 dumps a good fraction of its input into the earth and still
    prints ``EFFICIENCY    = 100.00 PERCENT``.  NEC's POWER BUDGET counts what
    the STRUCTURE dissipates, and the ground is not part of the structure —
    so a reader who wants the ground loss has to take it out of the pattern's
    average gain, not out of this block.  Reproducing that means reproducing
    the convention, not correcting it.

    The two divergent captures belong in a STRUCTURAL claim like this one:
    with no loads and no networks the budget is an identity, so it holds on a
    deck whose numbers are wrong for reasons of its own."""
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

    The ``GD`` captures are in the loop too, and their own cache lines are the
    reason the rule has to be about the FILE rather than about the mode: all
    four of them print a STALE block and ``Will compute Sommerfeld-ground
    tables`` over a run that computes none (module docstring, "One medium").
    The engine checks that file before it knows what it is about to do; a seam
    that emitted the block only for the modes that need it would still be
    emitting a block it has no cache for.
    """
    for cid in FINITE_IDS + MININEC_IDS:
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
# the MININEC rung's own facts
# --------------------------------------------------------------------------
#
# A bare `GD` and a `GN 0` carry the same two media fields, print the same four
# environment lines and mean two different grounds (module docstring of
# `momwire.eznec._serve`, "One medium, two things to do with it").  Each gate
# below is one way they differ or one way they deliberately do not, and each is
# written so that the ALIASING bug — routing `GD` down the `GN 0` branch, or
# down `GN 1`'s all the way to the pattern — fails it loudly.
#
# The two decks under test are the corpus's own, one per geometry class, which
# is what the scored matrix's probe family 1 measured the mode on: 0045 stands
# a wire END in the plane (the contact case) and 0029 hangs a dipole 9.144 m up
# with a coax feedline (the elevated case, nothing touching the ground at all).

_GD_CARDS = {
    "0045": "GD 0,0,0,0,13.,.005,1.,0.",
    "0029": "GD 0,0,0,0,20.,.0303,1.,0.",
}

# The measured |Z(GD) - Z(GN 0)| on each, this tree, 2026-08-20: 0045 answers
# 36.499 + 2.0789j against 48.867 + 2.5635j, and 0029 25.697 - 13.306j against
# 26.597 - 11.755j.  Gated as a FLOOR at half the measured value rather than as
# a pin, because the gate's job is "these are two grounds" and not "this
# difference never moves"; the aliasing bug puts it at exactly zero.
_ALIAS_GAP = {"0045": 12.3775, "0029": 1.7932}


def _swap_ground(cid: str, card: str) -> str:
    """A ``GD`` deck with its ground line replaced, and nothing else."""
    text = deck_text(cid)
    assert _GD_CARDS[cid] in text, cid
    return text.replace(_GD_CARDS[cid], card)


def _run(text: str):
    """One deck's :class:`RunData`, straight off the solve.

    Not through the printout: the identity below is exact, and reading it back
    out of an ``E12.4`` cell would only prove it to four digits.
    """
    return _serve.serve(parse_nec5(text))


@pytest.mark.parametrize("cid", sorted(_GD_CARDS))
def test_a_bare_gd_is_the_perfect_ground_solve_wearing_another_card(cid):
    """The rung's headline identity, on both geometry classes, at ``==``.

    NEC-5's MININEC-type ground solves its currents over a PERFECT image and
    spends its medium entirely on the far field, so ``Z`` under ``GD`` IS ``Z``
    under ``GN 1``: the engine answers 35.571 − j1.4223 both ways on the
    contact vertical and 89.933 + j52.053 both ways on an elevated half-wave
    (scored matrix, probe family 1).

    Gated at EXACT equality rather than inside an envelope, and the difference
    matters.  This seam does not APPROXIMATE the identity, it inherits it:
    ``_solver_for`` hands ``GD`` the same lone ``ground_z=0.0`` kwarg it hands
    ``GN 1``, so the two runs are the same constructor call on the same mesh
    and every solved float in them is the same float.  ``==`` on the whole
    dataclass is therefore the honest bar, and it is also the sharpest — a
    ``GD`` that had picked up so much as a rounding difference would have
    picked up a code path.

    What must NOT be equal is here too, because an identity with nothing on the
    other side of it would be satisfied by a seam that ignored the ground card
    entirely: the environment block is the finite one and not ``PERFECT
    GROUND``, and the pattern is the medium's rather than the image's.
    """
    gd = _run(deck_text(cid))
    pec = _run(_swap_ground(cid, "GN 1"))

    assert gd.sources == pec.sources
    assert gd.currents == pec.currents
    assert gd.charges == pec.charges
    assert gd.power == pec.power
    assert gd.network_excitation == pec.network_excitation

    assert gd.environment == "FINITE GROUND.  SOMMERFELD SOLUTION"
    assert pec.environment == "PERFECT GROUND"
    assert gd.ground is not None and pec.ground is None
    (gd_pattern,), (pec_pattern,) = gd.patterns, pec.patterns
    worst = max(
        abs(a.total_db - b.total_db)
        for a, b in zip(gd_pattern.rows, pec_pattern.rows)
        if a.total_db > -900 and b.total_db > -900
    )
    # 23.3 dB apart on 0045, 7.7 on 0029 — the medium is doing all of it.
    assert worst > 5.0, f"{cid}: patterns only {worst:.2f} dB apart"


@pytest.mark.parametrize("cid", sorted(_GD_CARDS))
def test_a_bare_gd_is_not_the_sommerfeld_solve_of_the_same_medium(cid):
    """The other half of the identity, and the trap it exists to kill.

    ``GD`` and ``GN 0`` differ only in their mnemonic — same ε, same σ, same
    printed cells, same banner — so a seam that read the medium and forgot to
    ask which card carried it would route ``GD`` into the Sommerfeld solve and
    answer with a straight face.  Measured on the engine, that is 34 % wrong in
    R on the contact vertical (35.571 against 47.789); measured on this tree it
    is the two gaps in ``_ALIAS_GAP``.

    Both geometry classes are gated because the trap is not equally visible in
    them: the elevated dipole's two answers sit 1.8 Ω apart where the contact
    vertical's sit 12.4, and a gate written only on the loud case would let a
    half-fixed routing through.
    """
    gd = _run(deck_text(cid)).sources[0].impedance
    somm = _run(_swap_ground(cid, "GN" + _GD_CARDS[cid][2:])).sources[0].impedance
    gap = abs(gd - somm)
    assert gap > 0.5 * _ALIAS_GAP[cid], f"{cid}: GD and GN 0 only {gap:.4f} Ohm apart"


def _environment_block(text: str) -> list[str]:
    """The ``ANTENNA ENVIRONMENT`` section's lines, heading included.

    Ends at the next section heading; trailing blanks are dropped so that the
    comparison is about the block and not about how much air follows it.
    """
    lines = text.split("\n")
    start = next(i for i, line in enumerate(lines) if "ANTENNA ENVIRONMENT" in line)
    end = next(i for i in range(start + 1, len(lines)) if "- - -" in lines[i])
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def test_the_two_finite_grounds_print_one_environment_block_byte_for_byte():
    """The banner lies, and the renderer lies with it.

    0045 and 0047 are one wire over one medium at one frequency under two
    different ground modes, and their ``ANTENNA ENVIRONMENT`` sections are the
    same four lines to the byte — ``FINITE GROUND.  SOMMERFELD SOLUTION``
    included, over a run that computes no Sommerfeld anything.  The captures
    say so and the seam reproduces it, which is the whole obligation: the block
    is a card image plus a medium, not a report of what the engine did.

    The rule that follows is the one worth guarding.  Nothing anywhere in this
    tree may read that banner back as a mode signal, and this gate is the
    reason it cannot: the two modes are byte-identical exactly here.  What
    tells them apart in a captured printout is the MATRIX TIMING cell (0045's
    ``FILL= 0.000 SEC.`` against 0047's ``0.094``), which this seam drops as a
    property of the machine, and 0047's ``Sommerfeld integral tables written in
    previous run`` postamble, which is a cache report and is normalized away.
    """
    assert _environment_block(printout_text("0045")) == _environment_block(
        printout_text("0047")
    )
    served_block = _environment_block(served("0045"))
    assert served_block == _environment_block(served("0047"))
    assert served_block == _environment_block(printout_text("0045"))
    assert served_block[2].strip() == "FINITE GROUND.  SOMMERFELD SOLUTION"


def test_the_two_finite_grounds_answer_patterns_that_differ_by_the_captured_amount():
    """One byte of deck, 178 rows of consequence — and the seam reproduces it.

    0045 and 0047 print elevation cuts that are 1.28-1.29 dB apart at every one
    of the 178 rows that are not the three ``-999.99`` nulls, the MININEC mode
    hotter throughout.  The SHAPE is common (both are the same medium's
    Fresnel-weighted image, and the difference varies by less than the printed
    cell's own 0.01 dB quantization across a 90° sweep); what moved is the
    LEVEL, because a PEC structure keeps the current a lossy ground would have
    loaded and the pattern is normalized by an input power that moved with it.

    Gating the DIFFERENCE rather than the two cuts is what makes this the
    anti-aliasing gate on the far-field side: the two cuts are each already
    pinned against their own capture, but a seam that answered ``GN 0`` for
    ``GD`` would print two identical cuts and a difference of 0.00 dB.  Worst
    row here is 0.02 dB against the captured difference.
    """
    cut = {
        name: {row.theta_deg: row.total_db for row in extract(text).patterns[0].rows}
        for name, text in (
            ("cap_gd", printout_text("0045")),
            ("cap_gn", printout_text("0047")),
            ("gd", served("0045")),
            ("gn", served("0047")),
        )
    }
    angles = [
        theta
        for theta in sorted(cut["cap_gd"], reverse=True)
        if cut["cap_gd"][theta] > -900 and cut["cap_gn"][theta] > -900
    ]
    assert len(angles) == 178

    captured = [cut["cap_gd"][t] - cut["cap_gn"][t] for t in angles]
    got = [cut["gd"][t] - cut["gn"][t] for t in angles]
    assert min(captured) >= 1.27 and max(captured) <= 1.30
    # The floor is what an aliased GD fails: it would put every row at 0.00.
    assert min(got) > 1.0
    worst = max(abs(a - b) for a, b in zip(captured, got))
    assert worst <= 0.025, f"worst row {worst:.4f} dB"


def test_the_reflection_and_the_sommerfeld_spelling_are_one_far_field():
    """``GD`` names its far ground ``refl`` and ``GN 0`` names it
    ``sommerfeld``, and the two names reach the same arithmetic.

    Both fall through to ``_image_coeffs`` in the portal's ``_far_moments`` —
    one branch, bit-identical output — so the choice buys nothing but a name,
    which is why the seam spends it on the truth: under ``GD`` no Sommerfeld
    integral answered anything, the currents came off a PEC image, and the
    medium's only job was to weight the reflection.  Calling that a Sommerfeld
    far field would be the banner's lie repeated where nobody forced it.

    Gated so that the naming decision stays a naming decision.  If the two
    kinds ever stop being one path, this fails here rather than silently
    changing what every ``GD`` deck in the corpus answers.
    """
    mid = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 3.0], [0.5, 0.25, 2.0]])
    moment = np.array(
        [[0.0, 0.0, 1.0 + 0j], [0.0, 0.0, 0.5 - 0.25j], [0.1 + 0.2j, 0.0, 0.0]]
    )
    theta = np.radians(np.array([90.0, 63.0, 30.0, 1.0, -45.0]))
    phi = np.radians(np.array([0.0, 37.0]))
    answers = [
        _far_moments(mid, moment, 2.0 * math.pi / 42.8286, theta, phi, ground, 0.0, 7e6)
        for ground in (Ground("refl", 13.0, 0.005), Ground("sommerfeld", 13.0, 0.005))
    ]
    for a, b in zip(*answers):
        assert np.array_equal(a, b)

    gd, gn = parse_nec5(deck_text("0045")), parse_nec5(deck_text("0047"))
    medium = _serve._medium(gd.ground, 42.8286)
    assert _serve._far_ground(gd, medium).kind == "refl"
    assert (
        _serve._far_ground(gn, _serve._medium(gn.ground, 42.8286)).kind == "sommerfeld"
    )


def test_the_negative_conductivity_convention_is_read_off_the_gd_flag():
    """``GD 0,…,-38.9052,…`` and ``GD 0,…,.0303,…`` are ONE ground at 14 MHz.

    The same fold ``GN 0`` gets, reached the other way round.  ``GN 0`` carries
    no flag, so ``_medium`` reads the SIGN of its sixth field; the ``GD``
    record carries ``sigma_sets_im_epsc``, which the parser sets from exactly
    that test, so the flag is what is read here.  Two spellings of one
    convention, and this gate is what says the seam did not implement it twice.

    0029's own medium supplies the numbers: 20/0.0303 at 14 MHz folds to
    Im εc = −38.905217, and the negative spelling of it has to come back as the
    same medium, the same impedance and the same cut.

    The tolerances are what the CARD can carry and no tighter, which is the
    honest reading of a convention that travels through a decimal field.  The
    ``E12.5`` cell a printout shows is six significant figures, so a deck
    written from one — ``-38.9052`` — recovers σ to 4.5e-7 relative and no
    better; pinning that at machine precision would be pinning the
    transcription rather than the fold.  Written with the digits the arithmetic
    actually has, the same test closes at 8e-9, two orders tighter, which is
    what says the residual is the card's and not the code's.

    The impedance is EXACTLY equal on both spellings, and that is not slack in
    the gate — it is this rung's own identity showing through from the other
    side.  Under ``GD`` the medium never reaches the solve at all, so no
    spelling of σ can move a single current; the only place the last digits can
    land is the pattern, and they land 2e-7 dB into it.
    """
    positive = _run(deck_text("0029"))
    printed = _run(_swap_ground("0029", "GD 0,0,0,0,20.,-38.9052,1.,0."))
    exact = _run(_swap_ground("0029", "GD 0,0,0,0,20.,-38.905217,1.,0."))
    assert positive.ground.eps_c.imag == pytest.approx(-38.905217, abs=5e-7)
    for negative, tolerance in ((printed, 1e-6), (exact, 1e-7)):
        assert negative.ground.eps_r == positive.ground.eps_r
        assert negative.ground.sigma == pytest.approx(
            positive.ground.sigma, rel=tolerance
        )
        assert negative.sources[0].impedance == positive.sources[0].impedance
        (want,), (got,) = positive.patterns, negative.patterns
        for a, b in zip(want.rows, got.rows, strict=True):
            assert a.total_db == pytest.approx(b.total_db, abs=1e-4)


# --------------------------------------------------------------------------
# the phased rung's own facts
# --------------------------------------------------------------------------
#
# Four things a phased drive has that a single source does not, and each is
# something the seam would otherwise be free to get wrong quietly: several rows
# instead of one, a set current per row instead of a global scale, a budget
# that is a SUM rather than a product, and a row that comes out negative.


@pytest.mark.parametrize("cid", PHASED_IDS)
def test_every_phased_row_sits_inside_its_own_measured_envelope(cid):
    """All four of 0031's rows and both of 0032's, row by row.

    The single-source envelope gate above reads ``sources[0]`` and would pass a
    seam that solved the four-square's first port correctly and the other three
    by accident, so the phased captures get their whole table pinned instead.
    Same bar shape as everywhere else — the measured ``|dZ|`` plus 25 %, listed
    in :data:`PHASED_Z_BAR` — and the same reading: an envelope, not an
    agreement claim.

    The TAG and SEGMENT of each row are gated exactly, because they are the
    deck's own and not a solved number: 0031 prints tags 1/2/3/4 at global
    segments 1/7/13/19, in DECK order, which is the order its four ``EX`` cards
    are written and not the order anything else in this module sorts things in.
    """
    want = extract(printout_text(cid)).sources
    got = extract(served(cid)).sources
    bars = PHASED_Z_BAR[cid]
    assert len(got) == len(want) == len(bars)
    for a, b, bar in zip(want, got, bars):
        assert (b.tag, b.segment, b.end_index) == (a.tag, a.segment, a.end_index)
        assert abs(b.impedance - a.impedance) <= bar, (
            f"{cid} tag {a.tag}: served {b.impedance}, captured {a.impedance}"
        )


@pytest.mark.parametrize("cid", PHASED_IDS)
def test_every_phased_row_prints_the_current_its_own_card_set(cid):
    """The drive constraint, to the byte, once per row.

    This is the one thing about a phased solve that is NOT an envelope.  Each
    ``EX 4`` fixes its port's current exactly, so every served row's CURRENT
    cell has to be the card's own complex pair and the captured cell at the
    same time — 0031's four are ``1.4142+0j``, ``-1.4142j``, ``-1.4142j`` and
    ``-1.4142+0j``, read back out of an ``E12.4`` cell on both sides.

    The exact zeros are the part that would break first.  A drive read back out
    of the solve rather than restored comes back with 1e-16 of the other
    component on it and prints ``1.1102E-16`` where the capture prints
    ``0.0000E+00`` — U1's lesson, applied here once per row rather than once
    per deck — and comparing the CELLS rather than the complex numbers is what
    makes that visible: the two agree as floats either way.
    """
    (fixed,) = {source.kind for source in parse_nec5(deck_text(cid)).sources}
    assert fixed == 4
    drives = [source.drive for source in parse_nec5(deck_text(cid)).sources]
    want = extract(printout_text(cid)).sources
    got = extract(served(cid)).sources
    for row, captured, drive in zip(got, want, drives, strict=True):
        assert row.current == captured.current
        # The deck writes 1.414214 and the cell holds four digits, so "is the
        # card's current" means to the printed digit.
        assert row.current == pytest.approx(drive, abs=5e-5)
        assert (row.current.real == 0.0) == (drive.real == 0.0)
        assert (row.current.imag == 0.0) == (drive.imag == 0.0)


def test_the_four_square_s_two_quadrature_elements_print_one_row_twice():
    """0031's tags 2 and 3 are the same row, and tags 1 and 4 are not.

    The phase RELATIONSHIPS are what a phased drive is for, and an envelope
    cannot see them: four rows could each sit inside their bar with the
    coupling between them wrong.  0031's geometry settles it.  Its four
    verticals sit on a square, tags 2 and 3 are the two side elements, they
    carry the SAME drive (``-j1.414214``) and stand in mirror-image positions
    about the array's own diagonal — so the engine prints them with identical
    voltages and identical impedances, 36.783 − 27.955j both, and this seam has
    to as well, to the last bit rather than to the printed digit.

    Tags 1 and 4 are the ends of the same diagonal with opposite drives and
    they must NOT match: -1.7790 − 24.192j against 56.525 + 41.575j is the
    array doing its job, and a seam that had lost the drives' phases would
    flatten that difference.  So the symmetry is gated to the PRINTED BYTE and
    the asymmetry at a floor, which together say the four drives arrived as
    four different complex numbers and reached the right four ports.

    "To the printed byte" and not to the last bit, deliberately.  The 4x4 drive
    solve is an LU with pivoting and it does not have to be symmetric in its
    round-off — the two rows come out 2e-13 apart in X — so ``==`` on the
    floats would be pinning the pivot order.  What the capture actually asserts
    is that the two ROWS are the same row, and comparing the rendered table
    lines is that claim exactly, at the precision the file has.
    """
    printout = served("0031")
    rows = extract(printout).sources
    assert [row.tag for row in rows] == [1, 2, 3, 4]
    assert rows[1].voltage == rows[2].voltage
    assert rows[1].impedance == rows[2].impedance
    assert rows[1].power == rows[2].power
    # The same claim on the bytes, with the two address columns taken off:
    # everything a solver filled in is character for character identical.
    table = printout.split("- - - ANTENNA INPUT PARAMETERS - - -")[1].split("\n")
    printed = [line[12:] for line in table[4:8]]
    assert printed[1] == printed[2]
    assert printed[0] != printed[3]
    # The captured pair is identical too, which is what says the symmetry is
    # the antenna's rather than this seam's arithmetic being reused.
    captured = extract(printout_text("0031")).sources
    assert captured[1].impedance == captured[2].impedance
    # 58 Ω apart served, 71 Ω captured — the diagonal is not a mirror.
    assert abs(rows[0].impedance - rows[3].impedance) > 25.0


def test_the_four_square_s_absorbing_element_stays_negative():
    """0031's tag-1 row prints NEGATIVE watts and negative ohms, and so does
    this seam's.

    The capture's own row is ``-1.7790E+00`` in the IMPEDANCE REAL cell and
    ``-1.7790E+00`` in POWER: at this phasing the element takes more power out
    of the array through the mutual coupling than its generator puts in, so its
    driving-point resistance is below zero and its generator is being driven.
    NEC-5 prints that as-is and the printed ``INPUT POWER = 1.2831E+02`` is the
    other three rows with it SUBTRACTED.

    Gated as a SIGN rather than as a value because that is the part a bar
    cannot hold: ``|dZ|`` of 6.27 on a row whose R is −1.78 would be satisfied
    by +4.5, which is a different physical claim about the same antenna.  A
    seam that took a magnitude anywhere, or that clamped a budget to positive
    watts, lands on the wrong side of this line and inside every envelope in
    the file.
    """
    captured = extract(printout_text("0031")).sources[0]
    row = extract(served("0031")).sources[0]
    assert captured.impedance.real < 0 and captured.power < 0
    assert row.impedance.real < 0, f"served R = {row.impedance.real}"
    assert row.power < 0, f"served P = {row.power}"
    # P = 1/2 Re(V I*) with the same current every other row uses: the negative
    # is arithmetic on an ordinary row, not a special case.
    assert row.power == pytest.approx(
        0.5 * (row.voltage * row.current.conjugate()).real, rel=1e-4
    )


@pytest.mark.parametrize("cid", PHASED_IDS)
def test_the_phased_budget_is_the_sum_of_its_printed_rows(cid):
    """``INPUT POWER`` = Σ rows, to the printed digit, on both captures.

    Read off the CAPTURES first, because that is where the rule comes from:
    0031's four rows are −1.7790 + 36.783 + 36.783 + 56.525 = 128.312 against a
    printed ``1.2831E+02``, and 0032's two are 19.139 + 49.318 = 68.457 against
    ``6.8457E+01``.  Then required of the served run, whose own rows are
    different numbers and have to close the same way.

    Which is the gate on the one arithmetic decision this rung makes.  The sum
    is signed — 0031's negative row comes OFF the total, and a seam that summed
    magnitudes would print 131.87 and still radiate 100 % of it.
    """
    for data in (extract(printout_text(cid)), extract(served(cid))):
        total = sum(row.power for row in data.sources)
        # The budget cell is an E11.4 and the row cells are E12.4, so the sum
        # closes at the coarser of the two.
        assert data.power.input_power == pytest.approx(total, rel=2e-4)
        assert data.power.radiated_power == pytest.approx(total, rel=2e-4)


@pytest.mark.parametrize("cid", PHASED_IDS)
def test_the_phased_azimuth_cut_agrees_at_every_printed_angle(cid):
    """All 361 rows of the ``RP 0`` azimuth cut, and the lobe rows twice.

    The elevation gate above says the MEDIUM is right by agreeing at every
    angle of a cut the ground shapes.  This is its azimuth sibling and it says
    something else: an array's azimuth cut is shaped by the PHASING, direction
    by direction, so agreeing across a full 360° turn is the statement that the
    four (two) drives reached the right four (two) ports with the right four
    (two) phases.  Nothing about the feed region's Ω-class offset can aim a
    pattern.

    Two bars, and the second is where the claim lives.  The whole cut is barred
    at :data:`AZIMUTH_BAR`, which the deep back nulls set — 0031 is 30.7 dB down
    at φ = 235° and 0032 is 33.5 dB down behind the cardioid, and a null is a
    cancellation whose remainder amplifies any difference in what cancelled.
    The rows within 10 dB of the peak are barred at :data:`AZIMUTH_LOBE_BAR`,
    0.17 and 0.12 dB, which is the same family the elevation cuts sit in.

    Neither cut has a single ``-999.99`` row: over a ground at a fixed theta
    there is no direction where the field vanishes to NEC's own floor, so all
    361 rows carry numbers and all 361 are compared.  That is checked here
    rather than assumed, because a null appearing on one side and not the other
    would otherwise be silently skipped.
    """
    (want,) = extract(printout_text(cid)).patterns
    (got,) = extract(served(cid)).patterns
    assert len(got.rows) == len(want.rows) == 361
    assert not [row for row in want.rows + got.rows if row.total_db == _NULL_DB]

    peak = max(row.total_db for row in want.rows)
    worst = max(abs(a.total_db - b.total_db) for a, b in zip(want.rows, got.rows))
    assert worst <= AZIMUTH_BAR[cid], f"{cid}: worst row {worst:.4f} dB"
    lobe = max(
        abs(a.total_db - b.total_db)
        for a, b in zip(want.rows, got.rows)
        if peak - a.total_db <= _LOBE_FLOOR
    )
    assert lobe <= AZIMUTH_LOBE_BAR[cid], f"{cid}: worst lobe row {lobe:.4f} dB"


def test_a_phased_drive_is_a_solve_and_not_a_scale():
    """The rung's own mechanism, stated as the thing it is NOT.

    A single ``EX 4`` is a readout transform: one port, ``I = Y·V``, so the
    answer is the unit-probe solve divided through and the impedance does not
    depend on the drive at all.  Four cards are not that.  The four ports are
    coupled, the drives are a simultaneous boundary condition, and each port's
    printed impedance is a function of ALL FOUR drives — which is exactly why
    0031's first element can present a negative resistance while its own card
    is unchanged.

    Measured here by moving one card and watching a DIFFERENT port answer
    differently: reverse tag 4's drive and tag 1's impedance moves from
    −1.703 − 17.93j to 7.231 − 56.29j, 39 Ω, on a port whose own ``EX`` card
    was not touched and whose printed current is unchanged.  A seam that had
    implemented the phased drive as four independent scales would print the
    same tag-1 row both times.

    The array stops being one, too, and in the way the geometry says it should:
    with tags 1 and 4 driven alike on one diagonal and 2 and 3 alike on the
    other, the four-square becomes symmetric, its two diagonals print identical
    rows, and its azimuth cut flattens from a 30.6 dB spread to 0.94.
    """
    text = deck_text("0031")
    assert "EX 4,4,-1,0,-1.414214,0." in text
    flipped = text.replace("EX 4,4,-1,0,-1.414214,0.", "EX 4,4,-1,0,1.414214,0.")

    base = _serve.serve(parse_nec5(text)).sources
    moved = _serve.serve(parse_nec5(flipped)).sources
    # Tag 4's own card changed, so its current changed and nothing is claimed
    # about it; tag 1's card did not, and its current is identical.
    assert moved[0].current == base[0].current
    assert abs(moved[0].impedance - base[0].impedance) > 20.0

    # And the array stopped being one: the deep-null cut flattens out.
    def span(data) -> float:
        rows = data.patterns[0].rows
        return max(r.total_db for r in rows) - min(r.total_db for r in rows)

    assert span(_serve.serve(parse_nec5(text))) > 25.0
    assert span(_serve.serve(parse_nec5(flipped))) < 3.0
    # Symmetric drives, symmetric rows: the two diagonals answer alike, to the
    # drive solve's own round-off (the same 2e-13 the captured deck's symmetric
    # pair comes out at above, and far under the printed cell either way).
    assert moved[0].impedance == pytest.approx(moved[3].impedance, rel=1e-12)


# --------------------------------------------------------------------------
# gate 3 — the refusals
# --------------------------------------------------------------------------

# One capture per refusal, and after #504 U4 there is exactly ONE.  `TL` and
# `NT` left this table with U5, which serves them; the deck that carries BOTH
# is refused for its table layout instead, and that refusal is gated in
# ``test_eznec_networks.py`` beside the rest of the network unit.  0031 left it
# with this unit, which serves its four phased `EX 4` cards.
#
# 0022 is the last tenant and it is the capture VERBATIM: its `GN 0` was served
# by the ground rung, so its `NE 0,1,1,1,…` is heard exactly as EZNEC wrote it.
# Every OTHER refusal this seam can reach is now a shape no capture writes, and
# each is probed below by editing a capture deck rather than by finding one.
REFUSALS = {
    "0022": "NE (near electric field) is not served at this seam yet",
}


def _refusal_deck(cid: str) -> bytes:
    return (FIXTURE_DIR / capture(cid)["deck"]).read_bytes()


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
    written = out.read_bytes().decode("latin-1").replace("\r\n", "\n")
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


def test_nh_parses_now_and_refuses_beside_ne_by_its_own_name():
    """momwire#513, the other half: NH is in the vocabulary (capture 0111
    falsified the never-emitted premise), so a deck carrying one must reach
    :func:`refusal` and be told the CARD is unserved — not that the card does
    not exist.  0022's deck with its NE respelled NH is the probe: same ten
    fields, one mnemonic, and the refusal switches sentences with it."""
    nh_deck = deck_text("0022").replace("NE 0,1,1,1,", "NH 0,1,1,1,")
    printout = render(nh_deck)
    reason = printout.split(" ***** NEC ERROR - ")[1]
    assert reason.startswith("NH (near magnetic field) is not served")
    assert "not a card" not in reason
    assert "NE (near electric field)" not in reason


def test_the_stub_refusal_no_longer_answers_anything_in_the_corpus():
    """U1's catch-all is now a backstop and not a behaviour.

    Every one of the 49 captured decks comes back either solved or refused by
    a sentence that names its card, so the stub reason — which said only that
    the dialect was unserved — must appear nowhere.
    """
    assert len(corpus()) == 49
    assert sorted(entry["deck"] for entry in MANIFEST["captures"]) == sorted(
        f"decks/{path.name}" for path in (FIXTURE_DIR / "decks").glob("*.nec")
    )
    for cid, text in corpus().items():
        assert "NEC-5 DIALECT NOT YET SERVED" not in text, cid
        assert "INTERNAL ERROR IN MOMWIRE ENGINE" not in text, cid


def test_the_corpus_ladder_stands_where_u4_left_it():
    """The served-id SET, pinned — 48 of the 49 captured decks.

    A rung that lands moves decks across this line and a regression moves them
    back, and neither should be able to happen quietly: the set is written out
    id by id so that changing it is an edit somebody made on purpose, with the
    new number in the diff beside the old one.  #504 U1 took it from 20 to 27,
    U2 from 27 to 43, U3 from 43 to 46, and U4 from 46 to 48 — the last two are
    0031 and 0032, the corpus's only phased arrays.

    ONE deck is left, and it names a card no ground and no drive can reach:
    0022's ``NE``, the near electric field, whose printed block has a layout of
    its own that no rung of this arc renders.  That is the whole refusal
    roster now — this seam answers every ground card, every network table and
    every drive the corpus writes, and says so about the one request it does
    not.

    What this number does NOT claim is that all 48 are right.  0033 and 0034
    are served, are in this set, and disagree with their captures by 275 and
    188 Ω (:data:`DIVERGENT_IDS`); the ladder counts decks that get an ANSWER
    rather than decks that get a pinned one, and that distinction is why the
    envelope tables are written out separately.
    """
    served_ids = tuple(
        cid for cid, text in sorted(corpus().items()) if "NEC ERROR" not in text
    )
    assert served_ids == SERVED_AFTER_U4
    assert len(served_ids) == 48
    refused = sorted(set(corpus()) - set(served_ids))
    assert refused == ["0022"]


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


@pytest.mark.parametrize("cid", DIVERGENT_IDS)
def test_the_elevated_radial_systems_disagree_with_their_captures_on_purpose(cid):
    """The finding, written down as a gate so it cannot be pinned by accident.

    0033 and 0034 are the two decks whose printouts landed with #504 U3 and
    whose numbers did not fit anything: 275 Ω and 188 Ω of impedance against a
    family that spans 0.05 to 19.5, with the REACTANCE changing sign, and 0.50
    and 3.31 dB on the pattern peak against a family that spans 0.01 to 0.25.
    Both are the same antenna — a 35.56 m vertical over four 39.624 m radials
    at 1.832 MHz — with everything but the vertical standing 1.778 cm above a
    ``GN 0`` earth, 1.09e-4 of the 163.6 m wavelength.

    Three things are asserted and each one is load-bearing.

    The gap is REAL and large, so nobody quietly widens an envelope until it
    covers this; if a fix lands, this test fails and the fixer has to come back
    and pin the new number properly, which is exactly the conversation the
    failure should start.

    The two decks agree with EACH OTHER about the antenna, roughly.  0033
    meshes it in five segments a wire and 0034 in a 44-wire taper down to 1.8 cm
    elements, and they are the same physical model; the captures answer 38.791
    − 49.583j and 40.730 − 43.452j, 7 Ω apart, and this seam answers 45.325 +
    225.59j and 91.615 + 137.87j, which is 100 Ω apart.  The engine is mesh-
    insensitive here and this seam is not, and a mesh sensitivity that appears
    only 1e-4 λ off a Sommerfeld plane is a statement about the ground model
    rather than about the basis.

    And the STRUCTURE is right, which is what puts the finding outside this
    package: the counting rule, the geometry columns and the whole printout
    layout pass their own gates on these two decks.  What is left is momwire's
    finite-ground solve at grazing height (``docs/refl-coef-ground-plan.md``,
    momwire#151/#157), reached through kwargs this seam only passes on.
    """
    want = extract(printout_text(cid)).sources[0].impedance
    got = extract(served(cid)).sources[0].impedance
    assert abs(got - want) > 100.0, (
        f"{cid}: served {got} against captured {want} — if this closed, "
        "measure the new offset and give the deck a real envelope pin"
    )
    # The captures agree with each other about the antenna; the seam does not.
    captured = [extract(printout_text(k)).sources[0].impedance for k in DIVERGENT_IDS]
    served_z = [extract(served(k)).sources[0].impedance for k in DIVERGENT_IDS]
    assert abs(captured[0] - captured[1]) < 10.0
    assert abs(served_z[0] - served_z[1]) > 50.0


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


# --------------------------------------------------------------------------
# the four multi-source shapes nothing has printed
# --------------------------------------------------------------------------
#
# #504 U4 serves ONE multi-`EX` shape — every card an `EX 4`, no network on the
# deck, one card per port — because that is the only shape the corpus prints.
# Four ways out of it are refused BY NAME, and none of them has a capture, so
# each is probed by editing one that does.  0032 is the base for all four: the
# smallest phased deck in the corpus, two six-segment verticals over `GN 1`.
#
# Why four sentences rather than one "unsupported drive" line: each would be
# fixed by a different capture, and a refusal's whole job is to tell the person
# reading it in EZNEC's viewer which card to go and change.

_PROBE_BASE = "0032"


def _refused(text: str) -> str:
    """The sentence a deck comes back refused with, through the real render."""
    printout = render(text)
    assert " ***** NEC ERROR - " in printout, "this deck was SERVED"
    assert "ANTENNA INPUT PARAMETERS" not in printout
    return printout.rsplit(" ***** NEC ERROR - ", 1)[1].strip()


def test_a_drive_that_mixes_the_two_ex_kinds_refuses_by_name():
    """One ``EX 0`` and one ``EX 4`` on the same deck.

    A voltage source and a current source in one matrix is a mixed boundary
    condition, the engine has a rule for it and no captured printout shows the
    rule, so this seam does not invent one.  Zero of the forty-nine captures
    write it — 47 are all-``EX 4`` and 2 are single ``EX 0`` — and the refusal
    counts both kinds in its sentence so the reader can see which card is the
    odd one.
    """
    text = deck_text("0032").replace("EX 4,2,-1", "EX 0,2,-1")
    reason = _refused(text)
    assert reason.startswith("this deck carries 2 EX cards writing BOTH kinds")
    assert "1 EX 0 (a set voltage) and 1 EX 4 (a set current)" in reason


def test_a_multi_voltage_drive_refuses_by_name():
    """Two ``EX 0`` cards.

    The other half of the same hole and a different sentence, because it is a
    different missing capture: a multi-voltage drive is coherent physics —
    voltages pinned, currents solved — and this seam simply has no printed row
    anywhere to say what NEC-5 does with the pair.  The corpus's two ``EX 0``
    decks (0033, 0034) each write exactly one.
    """
    text = deck_text("0032").replace("EX 4,1,-1", "EX 0,1,-1")
    text = text.replace("EX 4,2,-1", "EX 0,2,-1")
    reason = _refused(text)
    assert reason.startswith("this deck carries 2 EX 0 cards")
    assert "multi-VOLTAGE drive is not served" in reason


def test_a_phased_drive_through_a_network_refuses_by_name():
    """A ``TL`` card on a two-``EX`` deck.

    This is the refusal that costs something to keep, and it is the one worth
    keeping most.  The machinery is all present — the reducer serves networks,
    the sub-block solve serves phased drives — and composing them is a short
    edit with an obvious-looking answer.  What is missing is any evidence that
    the obvious answer is NEC-5's: not one of the forty-nine captures drives
    more than one port on a deck carrying ``TL`` or ``NT``, so a served answer
    here would be a guess wearing a printout.

    Both card kinds reach it, because a deck can carry either.
    """
    for card in ("TL 1,3,2,3,50.,1.,0.,0.,0.,0.", "NT 1,3,2,3,0.,.01,0.,0.,0.,.01"):
        reason = _refused(deck_text("0032").replace("PQ 0\n", f"{card}\nPQ 0\n"))
        assert reason.startswith("this deck carries 2 EX cards and a TL/NT network")
        assert "constrained drive composed with the network reducer" in reason


def test_two_ex_cards_at_one_address_refuse_by_name():
    """The same ``(tag, node)`` twice.

    One node is one port, so the second card is a second generator in series
    across the same gap and the two set currents are one constraint written
    twice.  Caught at the CARDS, before any mesh is built, because the deck can
    already say it.
    """
    text = deck_text("0032").replace("PQ 0\n", "EX 4,1,-1,0,1.,0.\nPQ 0\n")
    reason = _refused(text)
    assert reason.startswith("two EX cards address 1,-1")
    assert "second generator in series across the same gap" in reason


def test_two_ex_cards_on_the_two_sides_of_one_cut_refuse_by_name():
    """``1,6`` and ``2,-1`` at a node where wires 1 and 2 meet.

    The same collision reached the other way, and the only one the CARDS cannot
    see: two different addresses, two sites, one cut — the rank-1 two-port
    ``_assign_columns`` shares a column for, which is config C of the W7EL
    triple and which the network unit relies on.  One ``EX`` there is a source
    in the gap; two are one boundary condition written twice, with nothing in
    any capture to say which of the two currents the engine honours.

    So this refusal lives in the translation rather than in :func:`refusal`,
    and it names both addresses.  The probe bends 0032's second vertical into
    an inverted-L hanging off the first one's top, which is the shortest way to
    put a two-wire node in a deck that has none.
    """
    text = deck_text("0032").replace(
        "GW 2,6,.25,0.,0.,.25,0.,.242,.000121",
        "GW 2,6,0.,0.,.242,.25,0.,.242,.000121",
    )
    text = text.replace("EX 4,1,-1,0,1.414214,0.", "EX 4,1,6,0,1.414214,0.")
    reason = _refused(text)
    assert reason.startswith("1,6 and 2,-1 are the two sides of ONE cut")
    assert "one boundary condition written twice" in reason


def test_the_served_phased_shape_is_the_one_the_probes_all_left():
    """The control on the five refusals above: 0032 itself is SERVED.

    Each probe is one edit away from a capture that answers, so a refusal
    firing on all six decks would look exactly like a working gate.  This is
    the line that says the edits are what did it.
    """
    for cid in PHASED_IDS:
        assert "NEC ERROR" not in served(cid)
    assert "ANTENNA INPUT PARAMETERS" in served(_PROBE_BASE)


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
