"""Node-addressed networks, gated against the captures (#497 U5, #504 U3).

The gate unit.  U4's three gates carry over unchanged — a structure gate that
byte-compares everything that is not a solved number, envelope pins on the
physics, refusals by name — and this file adds the one thing the arc was
scoped around: **the W7EL oracle triple**, the smallest model in the world
that can tell a correct favored-wire seam from a plausible-but-wrong one.

Configs A (``NT 3,-1`` twice) and B (``NT 2,3`` twice) name the SAME
geometric node through two different tags and must answer identically;
config C (one of each) puts the two admittances on OPPOSITE sides of that
node, where they combine in SERIES rather than in parallel and the answer
moves 70 %.  A seam that canonicalizes ``2,3`` and ``3,-1`` into one node
returns A's number for C, so the gate is not "C is close to 195" alone — it
is also that C is FAR from A, by the gap the capture prints.

Beside them: 0018, where EZNEC snapped a 50 % request to a genuine node one
segment away from the junction; 0027, the Cardioid whose whole feed system is
two transmission lines to a virtual wire 100 λ away; and 0028, the
log-periodic whose four crossed feeders are the corpus's only ``CROSSED``
lines and whose phase reversal is worth 100 Ω of feedpoint resistance.

What is gated EXACTLY and what by envelope, as U4's re-spec has it: an
identity holds exactly (a connection point terminated by an admittance and
nothing else reports ``-1/Y``, whatever the antenna does; two ports in series
carry one current), a card echo is compared byte for byte, and a solved
number is pinned at its measured difference plus 25 %.

**#504 U3** takes this file from seven captures to twenty-four, and the new
seventeen are the corpus's whole feed-system population: ten 4-squares run
through six ``TL`` cards off a virtual anchor 100 λ out (0001-0009, 0024),
three coax-fed dipoles over real earth (0011, 0029, 0030), two cardioid feed
systems (0000, 0026), and the two L-network 4-squares (0023, 0025).  They were
served from #504 U2 and gated only for liveness, because the fixture manifest
had no printout for any of them; the printouts existed all along, in the
capture tree, and landing them turned nineteen liveness assertions into real
gates.

Three of the new ones are the reason this unit had a refusal at all.  0000,
0023 and 0025 carry ``TL`` AND ``NT`` cards, and their ``NETWORK DATA`` table
was refused for four units as an unobserved layout.  Their own printouts show
it: one heading, the ``TL`` sub-table under its own column header block, one
blank, and the ``NT`` sub-table under its.  The refusal is gone and the layout
is gated byte for byte, headers included.

**momwire#511** adds four more and a seventh gate: the corpus's only decks that
are BOTH phased and networked, which is the shape the last drive refusal named.
0116/0117 are 0031's four-square with ONE ``TL`` bolted across a diagonal;
0120/0121 are 0000's L-network cardioid with ONE second ``EX 4`` added.  Each
pair therefore has a captured control on the other side of exactly one card,
which is what lets gate 7 ask what the added card was WORTH rather than only
whether the composed answer is plausible — and the answers are large: the line
breaks a symmetry the four-square's own capture prints twice, and the second
source moves the first source's port by 31 Ω.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest

from momwire.deck._nec5 import parse_nec5
from momwire.deck._networks import card_branches
from momwire.eznec import _printout, _serve, serve
from momwire.eznec._printout import LineRow, NetworkRow
from momwire.eznec._shell import render
from momwire.networks import TL
from test_eznec_printout import deck_text, extract, printout_text
from test_eznec_serve import _NETWORK_LOSS_DUST, drive_cells, mask, served

# Every capture carrying a ``TL`` or an ``NT`` card AND a printout to gate it
# against — the seven #497 U5 brought into scope, and the seventeen whose
# printouts landed with #504 U3.
#
# momwire#511 takes it to twenty-eight and the four it adds are the corpus's
# only decks that are BOTH phased and networked — the shape the drive refusal
# named until they arrived.
NETWORK_IDS = (
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
    "0011",
    "0012",
    "0014",
    "0016",
    "0017",
    "0018",
    "0023",
    "0024",
    "0025",
    "0026",
    "0027",
    "0028",
    "0029",
    "0030",
    # momwire#511's four: the corpus's only decks that are BOTH phased and
    # networked.  Gate 7 is theirs.
    "0116",
    "0117",
    "0120",
    "0121",
)

# The FIVE whose table carries both card kinds.  All five stand over a bare
# ``GD``, all five write every ``TL`` before every ``NT``, and all five print
# the two sub-tables in that order — which is the measurement that dissolved
# ``_REFUSE_MIXED_NETWORKS``.
#
# It was three until momwire#511, and the two that joined are 0000's own cards
# with a second ``EX`` on them: the mixed-table layout is a property of the
# NETWORK DATA block and not of the drive, which is worth having on the record
# now that a deck can be mixed and phased at once.
MIXED_IDS = ("0000", "0023", "0025", "0120", "0121")

# The oracle triple, by configuration.  0014 is A again with ``XQ`` in place
# of ``RP`` — the same cards, so it is the control on the request card rather
# than on the addressing.
CONFIG_A = ("0012", "0014")
CONFIG_B = "0016"
CONFIG_C = "0017"

# W7EL's published answers, as the captured printouts give them.
Z_A = complex(114.47, 21.096)
Z_C = complex(195.34, -57.458)
Z_SNAP = complex(225.39, -60.593)


# --------------------------------------------------------------------------
# the measured table the pins come from
# --------------------------------------------------------------------------
#
# Measured 2026-08-20 on this tree, each served printout read back through
# `test_eznec_printout.extract` and compared cell for cell with its capture's:
#
#   id    Z (capture)        Z (served)        |dZ|    peak cap/served  d
#   0012  114.47 +21.096j    120.07 +33.244j   13.377    0.50 / 0.44   0.06
#   0014  114.47 +21.096j    120.07 +33.244j   13.377    (XQ, none)     —
#   0016  114.47 +21.096j    120.07 +33.244j   13.377    0.50 / 0.44   0.06
#   0017  195.34 -57.458j    198.33 -51.329j    6.819    (XQ, none)     —
#   0018  225.39 -60.593j    228.81 -55.339j    6.269    (XQ, none)     —
#   0027   23.422+12.770j     23.203+12.869j    0.240    7.89 / 8.14   0.25
#   0028  129.50 -73.453j    119.24 -56.832j   19.533    4.71 / 4.67   0.04
#
# and #504 U3's seventeen, measured the same way on the same day:
#
#   id    Z (capture)        Z (served)        |dZ|    peak cap/served  d
#   0000   31.592+25.945j     31.564+26.048j    0.107   -1.34 /-1.28   0.06
#   0001   13.949 +5.6027j    13.853 +5.1309j   0.482    5.12 / 5.25   0.13
#   0002   13.949 +5.6027j    13.853 +5.1309j   0.482    (XQ, none)     —
#   0003   13.977 +5.2920j    13.807 +5.1872j   0.200    (XQ, none)     —
#   0004   13.938 +5.1811j    13.849 +5.4369j   0.271    (XQ, none)     —
#   0005   13.920 +5.2737j    14.035 +5.8309j   0.569    (XQ, none)     —
#   0006   13.995 +5.5407j    14.396 +6.3146j   0.872    (XQ, none)     —
#   0007   14.210 +5.9359j    14.944 +6.8373j   1.162    (XQ, none)     —
#   0008   14.591 +6.4102j    15.681 +7.3555j   1.443    (XQ, none)     —
#   0009   15.148 +6.9187j    16.603 +7.8314j   1.718    (XQ, none)     —
#   0011   24.442-11.533j     26.597-11.755j    2.166    7.05 / 7.03   0.02
#   0023   10.453 +3.7826j    10.449 +3.7179j   0.065    3.51 / 3.52   0.01
#   0024   11.294 +0.36884j   11.252 +0.40380j  0.055    3.50 / 3.52   0.02
#   0025   42.902 -0.093298j  42.888 -0.35190j  0.259    3.40 / 3.41   0.01
#   0026   34.162+13.096j     34.167+13.214j    0.118   -1.31 /-1.28   0.03
#   0029   23.760-13.033j     25.697-13.306j    1.956    6.77 / 6.76   0.01
#   0030   24.442-11.533j     26.597-11.755j    2.166    7.05 / 7.03   0.02
#
# Three things in those tables are worth reading twice.
#
# The three W7EL configs sit 13.4 Ω from their captures and 0017 sits 6.8 Ω
# from ITS capture, which is the same formulation offset U4 measured on the
# bare decks and not something the networks added — the network arithmetic is
# exact, and what differs is the antenna underneath it.  And 0028, the
# log-periodic, is the widest at 19.5 Ω on a 148 Ω row: five coupled elements
# whose currents are set by a feeder network is the most basis-sensitive model
# in the corpus.
#
# U3's seventeen are the TIGHTEST rows anywhere in this arc — 0.055 to 2.2 Ω,
# where the bare captures span 3.5 to 16.3 and the older network ones 0.24 to
# 19.5 — and the reason is the same one that makes 0027's 0.240 the tightest
# of the old seven: every one of them is DRIVEN THROUGH A NETWORK.  The source
# stands on a virtual node 100 λ from the antenna and the printed impedance is
# what a transmission line presents at its own end, so the antenna's feed
# region — the place the two formulations disagree — reaches the printed cell
# through a line that flattens it.  0011/0029/0030's 2 Ω is the widest of the
# seventeen and they are the three with the SHORTEST line (a 13.85 m coax
# rather than a 100 λ anchor).
#
# 0002-0009 are one deck stepped 7.15 to 7.5 MHz and their offsets climb
# monotonically with frequency, 0.20 Ω to 1.72 Ω: a 4-square is a phased array
# and the further it is driven from the frequency its feed system was cut for,
# the more the element currents depend on exactly what the elements' mutual
# impedances are.  That is a formulation difference being AMPLIFIED by the
# array, and it is the clearest picture of the offset's shape anywhere in the
# corpus.
#
# And momwire#511's four, measured 2026-08-21.  The row quoted is the FIRST
# ``ANTENNA INPUT PARAMETERS`` row, so that the sweep this table drives keeps
# one shape; every other row of every one of them is pinned in
# :data:`PHASED_NETWORK_Z_BAR` below.
#
#   id    Z (capture)        Z (served)        |dZ|    peak cap/served  d
#   0116  -45.158-52.766j    -45.494-46.064j    6.710   (XQ, none)     —
#   0117  -45.158-52.766j    -45.494-46.064j    6.710    5.45 / 5.39   0.06
#   0120   0.22391+48.345j   -0.064422+48.581j  0.373   (XQ, none)     —
#   0121   0.22391+48.345j   -0.064422+48.581j  0.373   -0.94 /-0.97   0.03
#
# Two families, and each lands in the one its DRIVE puts it in rather than the
# one its cards do.  0116/0117 are 0031's four-square with one ``TL`` bolted
# across it and their four rows sit at 5.13 to 7.92 Ω, which is 0031's own
# 6.27-to-8.31 family to the tenth: the sources still stand on the antenna's
# four bases, the network hangs between two UNDRIVEN interior nodes, and the
# printed impedance is a feedpoint impedance with all of the feed region's
# formulation offset in it.  0120/0121 are the cardioid whose two sources stand
# ON connection points, and their 0.37 and 0.56 Ω sit with the seventeen
# driven-through-a-network rows above rather than with the four-square — the
# same argument those rows make, arriving now on a deck with TWO generators.
Z_BAR = {
    "0000": 0.14,
    "0001": 0.61,
    "0002": 0.61,
    "0003": 0.25,
    "0004": 0.34,
    "0005": 0.72,
    "0006": 1.09,
    "0007": 1.46,
    "0008": 1.81,
    "0009": 2.15,
    "0011": 2.71,
    "0012": 16.72,
    "0014": 16.72,
    "0016": 16.72,
    "0017": 8.52,
    "0018": 7.84,
    "0023": 0.082,
    "0024": 0.069,
    "0025": 0.33,
    "0026": 0.15,
    "0027": 0.30,
    "0028": 24.42,
    "0029": 2.45,
    "0030": 2.71,
    "0116": 8.39,
    "0117": 8.39,
    "0120": 0.47,
    "0121": 0.47,
}

# |(Z_C - Z_A) served - (Z_C - Z_A) captured|: the SEPARATION between the
# series and parallel answers, measured at 6.56 Ω against a captured gap of
# 112.74 Ω, and pinned at that plus 25 %.  This is the number the unit exists
# for: a seam that folded the two addresses into one node would answer a gap
# of zero here, and 8.2 is a very long way from 112.7.
GAP_BAR = 8.2

# Peak total gain: |d| plus 25 % or 0.05 dB, whichever is larger, the printed
# cell being quantized at 0.01 dB.
PEAK_BAR = {
    "0000": 0.075,
    "0001": 0.163,
    "0011": 0.05,
    "0012": 0.08,
    "0016": 0.08,
    "0023": 0.05,
    "0024": 0.05,
    "0025": 0.05,
    "0026": 0.05,
    "0027": 0.32,
    "0028": 0.05,
    "0029": 0.05,
    "0030": 0.05,
    "0117": 0.075,
    "0121": 0.05,
}

# Largest per-row difference in TOTAL gain among the rows the capture puts
# within 10 dB of its own peak, measured and plus 25 % — or 0.05 dB, whichever
# is larger, for the peak bar's reason.  0027 is the outlier by an order of
# magnitude and it is the physics: a cardioid is a NULL pattern, its null depth
# is set by how exactly two feed currents cancel, and two formulations that
# disagree by 0.2 Ω at the feedpoint disagree by whole dB down in the notch
# (8.3 dB at 20 dB below the peak, 1.7 within 10).
#
# The same effect is why the ten new AZIMUTH cuts are gated within 10 dB and
# not at every angle.  0011/0029/0030's 181-row ELEVATION cuts agree at 0.04 dB
# across every non-null row and are gated that way in
# ``test_eznec_serve.py::test_the_finite_ground_elevation_cut_agrees_at_every_printed_angle``;
# the 361-row azimuth cuts of the arrays are 1.96 dB (0000), 6.59 (0001), 2.32
# (0023/0025), 3.06 (0024) and 2.45 (0026) at their worst row, and every one of
# those worst rows sits in the array's deep rear null, 20 to 40 dB down, where
# a null's depth is a difference of two nearly equal numbers and a bar drawn
# round it would be pinning a cancellation rather than a pattern.  The
# within-10-dB numbers on the same cuts are 0.29, 0.30, 0.04, 0.09 and 0.18.
SHAPE_BAR = {
    "0000": 0.363,
    "0001": 0.375,
    "0011": 0.05,
    "0012": 0.15,
    "0016": 0.15,
    "0023": 0.05,
    "0024": 0.113,
    "0025": 0.05,
    "0026": 0.225,
    "0027": 2.10,
    "0028": 0.08,
    "0029": 0.05,
    "0030": 0.05,
    "0117": 0.25,
    "0121": 0.34,
}

# Wire-current and charge-density tables, worst element relative to that
# table's own peak, measured and plus 25 % — same shape as U4's.
#
# 0001-0009 are the widest current rows in the whole arc at 15 to 22 % and the
# reason is the same one that made 0028 wide: a 4-square's element currents are
# what its feed system SETS, so a formulation difference at four feedpoints
# arrives in the current table four times over.  Their impedance rows are the
# tightest in the arc at the same time, which is not a contradiction — the
# printed impedance is measured 100 λ away through a line, and the currents are
# measured on the elements.
CURRENT_BAR = {
    "0000": 0.033,
    "0001": 0.269,
    "0002": 0.269,
    "0003": 0.270,
    "0004": 0.265,
    "0005": 0.255,
    "0006": 0.241,
    "0007": 0.225,
    "0008": 0.209,
    "0009": 0.193,
    "0011": 0.077,
    "0012": 0.023,
    "0014": 0.023,
    "0016": 0.023,
    "0017": 0.022,
    "0018": 0.030,
    "0023": 0.046,
    "0024": 0.047,
    "0025": 0.046,
    "0026": 0.026,
    "0027": 0.177,
    "0028": 0.113,
    "0029": 0.073,
    "0030": 0.077,
    "0116": 0.026,
    "0117": 0.026,
    "0120": 0.029,
    "0121": 0.029,
}
CHARGE_BAR = {
    "0000": 0.095,
    "0001": 0.155,
    "0002": 0.155,
    "0003": 0.128,
    "0004": 0.104,
    "0005": 0.087,
    "0006": 0.078,
    "0007": 0.090,
    "0008": 0.114,
    "0009": 0.136,
    "0011": 0.096,
    "0012": 0.079,
    "0014": 0.079,
    "0016": 0.079,
    "0017": 0.071,
    "0018": 0.072,
    "0023": 0.091,
    "0024": 0.111,
    "0025": 0.091,
    "0026": 0.099,
    "0027": 0.129,
    "0028": 0.114,
    "0029": 0.092,
    "0030": 0.096,
    "0116": 0.083,
    "0117": 0.083,
    "0120": 0.097,
    "0121": 0.097,
}

# Connection points that are ANTENNA terminals — the pinned virtual nodes,
# which report 1e10 and 2e5 Ω against currents of 1e-9 and 1e-16 A, are dust
# on both sides and are left out by the cutoff.  Worst row measured, plus
# 25 %: 0027's two line ends at 6.8 and 8.3 Ω, 0028's five at up to 13.0.
#
# #504 U3's seventeen sit either side of that.  The 4-squares' six line ends
# land at 15.2 to 18.4 Ω on 60 Ω-class points and the cardioids' at 1.4 to 3.2,
# which is the same story the impedance table tells.  The three coax dipoles
# are the entry to read the scale off rather than the number: their far
# connection point is a near-open at 6.7 to 7.4 kΩ, and 1104 to 1165 Ω on that
# is 15 % — the SAME relative offset as the 6.3 Ω their antenna-side point
# carries on 85 Ω.  One formulation difference, two points, two very different
# absolute numbers, because a half-wave of coax transforms it.
CONNECTION_BAR = {
    "0000": 2.18,
    "0001": 19.03,
    "0002": 19.03,
    "0003": 19.43,
    "0004": 19.89,
    "0005": 20.40,
    "0006": 20.98,
    "0007": 21.61,
    "0008": 22.30,
    "0009": 23.04,
    "0011": 1379.7,
    "0023": 3.96,
    "0024": 3.76,
    "0025": 3.96,
    "0026": 1.75,
    "0027": 10.4,
    "0028": 16.3,
    "0029": 1456.2,
    "0030": 1379.7,
    "0116": 4.03,
    "0117": 4.03,
    "0120": 2.51,
    "0121": 2.51,
}
CONNECTION_CUTOFF = 1e4

# EFFICIENCY, in percentage points: 0.29 on the three W7EL configs that share
# an answer, 0.66 on config C, 0.58 on the snapped one, and 0.00 on the two
# that read 200 % — where the number is structural rather than solved.
# Measured worst, plus 25 %.  #504 U3's seventeen add nothing to it: 0.20 on
# 0023/0025, 0.13 on 0000, 0.11 on 0024 and exactly 0.00 on the other thirteen.
# Nor do momwire#511's four: 0.00 on 0116/0117, whose budget is structural for
# the opposite reason (a lossless line between two undriven points delivers
# nothing, so RADIATED is INPUT and the efficiency is 100.00 on both sides),
# and 0.18 on 0120/0121, the corpus's first budget strictly BETWEEN 100 % and
# 200 % (:func:`test_the_cardioid_s_two_driven_connection_points_read_161_
# percent`).
EFFICIENCY_BAR = 0.83


@functools.lru_cache(maxsize=None)
def _impedance(cid: str) -> complex:
    """A capture's served feedpoint impedance at SOLVER precision.

    Off the :class:`~momwire.eznec.RunData` rather than off the rendered
    printout, because the alias gate below asks whether two decks give the
    same number and an ``E12.4`` cell would answer that question with five
    digits of its own.
    """
    return serve(parse_nec5(deck_text(cid))).sources[0].impedance


# --------------------------------------------------------------------------
# gate 1 — the structure
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("cid", NETWORK_IDS)
def test_a_served_network_printout_is_the_capture_wherever_it_is_not_solved(cid):
    """U4's structure gate, with two more tables under it.

    ``NETWORK DATA`` is compared BYTE FOR BYTE and nothing in it is masked:
    every cell there is either a card echo (the admittances, the impedances,
    the STRAIGHT/CROSSED flag) or a distance between two nodes, so a wrong
    signed segment, a wrong resolved length or a dropped polarity flag fails
    here rather than quietly moving a number somewhere else.  The
    ``STRUCTURE EXCITATION DATA`` block is masked cell by cell — all nine,
    the drive-current exemption not applying to a structure current — leaving
    its row COUNT, its row ORDER, its tags, its segments and its sign-tracking
    end index compared exactly.
    """
    assert mask(served(cid), drive_cells(cid)) == mask(
        printout_text(cid), drive_cells(cid)
    )


@pytest.mark.integration
@pytest.mark.parametrize("cid", NETWORK_IDS)
def test_the_network_table_echoes_the_cards_exactly(cid):
    """The same bytes again, read back as numbers, so a failure says WHICH.

    A byte gate on the whole file tells you a line moved; this one tells you
    the resolved length of 0028's third feeder is wrong.  Worth the
    duplication precisely there: four of 0028's five cards write a zero
    length and the printed metres are the node-to-node distances NEC resolved
    them to, which is the only place in the printout where a geometric rule of
    this seam's own is visible.
    """
    want = extract(printout_text(cid)).networks
    got = extract(served(cid)).networks
    assert got == want


# --------------------------------------------------------------------------
# gate 2 — the oracle triple
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_the_two_spellings_of_one_node_answer_identically():
    """Config A against config B: ``3,-1`` and ``2,3`` are a true alias.

    The two decks differ in which WIRE'S end the series gap is declared on,
    which is a different solver column and a different fill, so this is
    equality at solver precision rather than bit equality — measured at 2.3e-15
    relative, pinned at 1e-12, and eleven orders of magnitude away from the
    0.05 relative difference a seam that got the alias wrong would show.

    0014 is A's own cards with ``XQ`` for ``RP``, so it must be A to the BIT:
    same geometry, same addresses, same solve, and only the request differs.
    """
    a, also_a = (_impedance(cid) for cid in CONFIG_A)
    b = _impedance(CONFIG_B)
    assert also_a == a
    assert abs(b - a) <= 1e-12 * abs(a), f"A {a}, B {b}"


@pytest.mark.integration
def test_config_c_puts_the_two_admittances_in_series_and_not_in_parallel():
    """The gate the whole unit was scoped around.

    Three assertions and they fail differently.  The first says C landed near
    the number NEC-5 printed.  The second says the DISTANCE from A to C is the
    one the capture prints — 112.74 Ω, reproduced to 6.6 — and it is the one
    that kills the plausible-but-wrong seam outright: fold ``2,3`` and
    ``3,-1`` into one geometric node and C answers A, a gap of exactly zero.
    The third is the same statement said as W7EL says it: a parallel
    combination of .01 and .005 mhos is 66.67 Ω and a series one is 300, so C
    must sit far outside A's own envelope rather than just outside it.
    """
    a, c = _impedance(CONFIG_A[0]), _impedance(CONFIG_C)
    assert abs(c - Z_C) <= Z_BAR[CONFIG_C], f"served {c}, captured {Z_C}"
    assert abs((c - a) - (Z_C - Z_A)) <= GAP_BAR, f"served gap {c - a}"
    assert abs(c - a) > 0.5 * abs(Z_C - Z_A)


@pytest.mark.integration
def test_the_snapped_config_moves_the_load_a_whole_segment_off_the_junction():
    """0018: ``NT 2,2`` and ``NT 2,3``, a genuine element apart.

    EZNEC snapped a 50 % request up to node 2 and told the user so; the deck
    that reaches the engine carries no fraction at all.  Both admittances are
    still in series, but now with a length of wire between them, and the
    answer moves again — 225.39 − j60.593, a third distinct number from one
    unchanged geometry.
    """
    assert abs(_impedance("0018") - Z_SNAP) <= Z_BAR["0018"]


@pytest.mark.integration
@pytest.mark.parametrize("cid", NETWORK_IDS)
def test_the_feedpoint_impedance_sits_inside_its_measured_envelope(cid):
    """|Z_served − Z_captured| against the per-capture bar in the table above.

    An envelope and not an agreement claim, for U4's reason: two
    discretizations of one physics sit Ω apart at these mesh densities.  What
    it does gate is that the NETWORKS did not add to that — the three W7EL
    configs land the same 13.4 Ω from their captures that U4's bare decks land
    from theirs.
    """
    want = extract(printout_text(cid)).sources[0].impedance
    got = extract(served(cid)).sources[0].impedance
    assert abs(got - want) <= Z_BAR[cid], f"{cid}: served {got}, captured {want}"


# --------------------------------------------------------------------------
# gate 3 — the identities, which hold exactly
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "cid,expected",
    [
        ("0012", [-66.666666666666]),
        ("0014", [-66.666666666666]),
        ("0016", [-66.666666666666]),
        ("0017", [-100.0, -200.0]),
        ("0018", [-100.0, -200.0]),
    ],
)
def test_a_terminated_connection_point_reports_minus_one_over_its_admittance(
    cid, expected
):
    """``Z = -1/Y`` at a connection point, whatever the antenna is doing.

    An identity, so it is gated as one.  A point that carries no source and
    whose only external branch is an admittance settles where the network says
    it must — 0012's two parallel cards give -66.667 Ω, 0017's two single ones
    -100 and -200 — and the antenna's own admittance cancels out of the ratio
    entirely.  The captures print exactly those three numbers, which is how a
    reader knows the sign convention in this seam is NEC's own: the ratio is
    NEGATIVE because the printed current flows INTO the structure while the
    printed voltage is across the network that drove it.
    """
    rows = [
        row
        for row in serve(parse_nec5(deck_text(cid))).network_excitation
        if abs(row.impedance) < CONNECTION_CUTOFF
    ]
    assert [row.impedance.real for row in rows] == pytest.approx(expected, rel=1e-9)
    assert [row.impedance.imag for row in rows] == pytest.approx(
        [0.0] * len(expected), abs=1e-9
    )


@pytest.mark.integration
def test_the_two_sides_of_one_cut_carry_one_current():
    """0017's two connection points are the same cut, so their current is one
    number — and the capture agrees, printing 7.7427E-01 − 4.9839E-01j on both
    rows.

    This is the series statement in its sharpest form, and it is exact rather
    than approximate: whatever flows out of wire 2 at that node flows into
    wire 3, so the two ports the deck named there see one current and add
    their voltages.  The voltages come out in the 2:1 ratio of the two
    admittances, which the capture prints too (-154.85 + 99.678j against
    -77.427 + 49.839j).
    """
    rows = serve(parse_nec5(deck_text("0017"))).network_excitation
    wire_three, wire_two = rows[0], rows[2]
    assert (wire_three.tag, wire_two.tag) == (3, 2)
    assert wire_three.current == wire_two.current
    assert wire_two.voltage == pytest.approx(2.0 * wire_three.voltage, rel=1e-9)


@pytest.mark.integration
@pytest.mark.parametrize("cid", NETWORK_IDS)
def test_the_sign_tracking_end_index_follows_the_deck_and_nothing_else(cid):
    """The trailing index after ``TAG``/``SEG.``: 2 for a ``-1`` node, 1 for
    a positive one, on every captured row of every network block.

    The capture study measured 9 of 9 rows; these twenty-four printouts carry
    123, and every one of them is compared against the deck's own spelling AND
    against the captured row.  It is not an end-one/end-two flag and not a
    port index — 0017 prints 2 and 1 for two points on ONE geometric node —
    which is why it is carried from the address rather than derived.

    Comparing the LISTS rather than the sets is what makes this the
    connection-point ORDER gate too, and #504 U3 is where that stopped being
    free: the mixed decks are the first that could have had their two card
    kinds discovered in either order, and 0025's eight rows off seven cards
    (four ``TL``, three ``NT``, one node named twice) are the longest such
    block in the corpus.
    """
    deck = parse_nec5(deck_text(cid))
    want = extract(printout_text(cid)).network_excitation
    got = extract(served(cid)).network_excitation
    assert [(r.tag, r.segment, r.end_index) for r in got] == [
        (r.tag, r.segment, r.end_index) for r in want
    ]
    written = {
        (end.tag, _serve._segment_of(_serve.structure_of(deck), end)): end.written
        for end in _serve._network_ends(deck)
    }
    for row in got:
        assert row.end_index == (2 if written[(row.tag, row.segment)] == -1 else 1)


# --------------------------------------------------------------------------
# gate 4 — the transmission lines
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_the_crossed_flag_reaches_the_network_solve_as_a_polarity():
    """0028's four feeders write ``-490.0875`` and the fifth ``+490.0875``.

    Checked at the MODEL level, where it can be seen rather than inferred: the
    branches this seam hands the reducer are four
    :class:`~momwire.networks.TL` with ``transposed=True`` and one without,
    each carrying the MAGNITUDE as its characteristic impedance.  A negated Z0
    would be a different circuit — it flips the chain matrix's self terms too —
    and the printout says so in its own column, ``CROSSED`` on four rows and
    ``STRAIGHT`` on the fifth.
    """
    deck = parse_nec5(deck_text("0028"))
    structure = _serve.structure_of(deck)
    mesh = _serve.build_mesh(deck, structure)
    cards = _serve._cards(deck, structure, mesh)
    lines = [
        branch
        for entry in cards
        for branch in card_branches(entry.card, entry.site_a, entry.site_b, ())
        if isinstance(branch, TL)
    ]
    assert [line.transposed for line in lines] == [True, True, True, True, False]
    assert {line.z0 for line in lines} == {490.0875}
    assert [row.line_type for row in extract(served("0028")).networks] == [
        "CROSSED",
        "CROSSED",
        "CROSSED",
        "CROSSED",
        "STRAIGHT",
    ]


@pytest.mark.integration
def test_the_phase_reversal_is_worth_the_whole_log_periodic():
    """Straighten 0028's four crossed feeders and the antenna changes species.

    119.24 − j56.832 becomes 10.861 − j297.50: an LPDA without its
    element-to-element phase reversal is not a mismatched LPDA, it is a
    different array.  The gate is here because ``transposed`` is one boolean
    deep inside a network solve, and a boolean that never changes an answer is
    a boolean nobody notices going wrong.
    """
    crossed = _impedance("0028")
    straight = (
        serve(parse_nec5(deck_text("0028").replace("-490.0875", "490.0875")))
        .sources[0]
        .impedance
    )
    assert abs(straight - crossed) > 0.5 * abs(crossed)


@pytest.mark.integration
def test_a_zero_length_line_is_resolved_node_to_node():
    """0028's four crossed feeders all write ``0.`` for their length and the
    printout gives back 9.9764E-01, 7.9826E-01, 6.3831E-01 and 5.1090E-01.

    Those are the distances between the two addressed NODES.  Segment centres —
    the rule the NEC-2 dialect's own reader applies, because a NEC-2 network
    lands on a segment — would give four different numbers, so the captured
    column is what picks the reading, and it is worth its own test because
    every other cell in that table is an echo and this one is arithmetic.
    """
    rows = [row for row in extract(served("0028")).networks if row.crossed]
    assert [round(row.length_m, 5) for row in rows] == [
        0.99764,
        0.79826,
        0.63831,
        0.51090,
    ]


@pytest.mark.integration
@pytest.mark.parametrize("cid", sorted(CONNECTION_BAR))
def test_the_line_ends_see_the_impedances_the_capture_reports(cid):
    """The two ``TL`` decks' antenna-side connection points, pinned.

    The feedpoint envelope only says the source saw the right impedance; this
    says the network did too, at every point where it touches the antenna —
    0027's two verticals through their quarter-wave and half-wave lines,
    0028's five elements down the feeder.  The pinned virtual nodes are left
    out: 1e10 Ω against 1e-9 A is the same dust on both sides.
    """
    want = extract(printout_text(cid)).network_excitation
    got = extract(served(cid)).network_excitation
    worst = max(
        abs(a.impedance - b.impedance)
        for a, b in zip(want, got)
        if abs(a.impedance) < CONNECTION_CUTOFF
    )
    assert worst <= CONNECTION_BAR[cid]


# --------------------------------------------------------------------------
# gate 5 — the tables and the budget
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("cid", sorted(PEAK_BAR))
def test_the_pattern_peak_and_shape_land_where_the_capture_puts_them(cid):
    """Level and shape, both by envelope — and the DIRECTION deliberately not.

    U4 gates the peak direction exactly and is right to on a dipole or a Yagi.
    Not here: every one of these four peaks is a PLATEAU at the printed
    quantization.  0012's capture prints 0.50 dB on two rows and 0.49 on
    twenty more; 0027's cardioid maximum reads 7.89 dB continuously from 21
    to 337 degrees.  An argmax over that is a tie-break, not a measurement, so
    what is gated is the level and the whole shape down to 10 dB below it.
    """
    (want,) = extract(printout_text(cid)).patterns
    (got,) = extract(served(cid)).patterns
    assert len(got.rows) == len(want.rows)
    peak = max(row.total_db for row in want.rows)
    assert abs(max(row.total_db for row in got.rows) - peak) <= PEAK_BAR[cid]
    assert (
        max(
            abs(a.total_db - b.total_db)
            for a, b in zip(want.rows, got.rows)
            if a.total_db > peak - 10.0
        )
        <= SHAPE_BAR[cid]
    )


@pytest.mark.integration
@pytest.mark.parametrize("cid", NETWORK_IDS)
def test_the_current_and_charge_tables_sit_inside_their_measured_envelopes(cid):
    """Every element's current and charge density, against the capture.

    Including the virtual wire's, which is the point of running it as ordinary
    geometry: 0012's wire 4 sits 100 λ away carrying 1e-33 A and its three
    elements are in this comparison like any others.
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


@pytest.mark.integration
@pytest.mark.parametrize("cid", NETWORK_IDS)
def test_the_budget_closes_the_way_the_engine_closes_it(cid):
    """``RADIATED = INPUT − NETWORK LOSS − WIRE LOSS``, and the line's own
    presence rule.

    Both halves are the engine's and neither is what a reader would guess.
    NETWORK LOSS is the negative of the connection points' power, so a deck
    whose SOURCE stands on a connection point counts the source's own watts a
    second time through it and the budget comes out at 200 % — which is
    exactly what 0027 and 0028 print, and the line is then omitted rather
    than printed negative.  Reproducing that is not endorsing it; it is what
    a drop-in replacement is for.

    The presence claim carries one exemption and momwire#511 is what found it.
    0116/0117 hang ONE lossless ``TL`` between two undriven points, so the sum
    is exactly zero and the engine's own crumb decided the sign: it printed
    ``2.1316E-13 WATTS`` against a 1.1538E+02 input, while this seam's crumb
    moved with machine and process history (momwire#677: −3.6e-14 here,
    +2.1316E-14 cold on GitHub runners, non-positive on the same runners out
    of a resident process).  The sign of a zero is not a number, so since #677
    the seam floors its own print decision on
    :data:`momwire.eznec._serve._NETWORK_LOSS_DUST` and deterministically
    omits the dust line the engine's crumb happened to earn — asserted below.
    What is still gated on those two is everything else in the budget — the
    sum, the arithmetic and the efficiency — and the presence claim itself
    everywhere the number means anything, which is the other twenty-six.
    """
    data = extract(served(cid))
    power, want = data.power, extract(printout_text(cid)).power
    captured_loss = want.network_loss
    dust = captured_loss is not None and abs(captured_loss) <= (
        _NETWORK_LOSS_DUST * power.input_power
    )
    if not dust:
        # Including every capture that prints NO line: an ABSENT one is never
        # dust, so 0027 and 0028 keep the claim in full.
        assert (power.network_loss is None) == (captured_loss is None)
    else:
        # The #677 pin: at dust the seam's floor omits the line, whatever
        # side of zero this machine's (or this process history's) crumb
        # landed on.
        assert power.network_loss is None
    loss = power.network_loss or -sum(row.power for row in data.network_excitation)
    assert loss == pytest.approx(
        -sum(row.power for row in data.network_excitation), rel=1e-4
    )
    assert power.radiated_power == pytest.approx(
        power.input_power - loss - power.wire_loss, rel=1e-4
    )
    # Both sides are read back off the printout, at DIFFERENT precisions, so
    # this comparison has a floor that has nothing to do with the physics.
    # EFFICIENCY prints `:7.2f` (_printout.py, quantum 0.005); the two powers
    # print with a 4-decimal mantissa, worth <= 1e-4 relative on their ratio,
    # which is what `rel` covers. pytest.approx takes the LARGER of rel and
    # abs rather than their sum, so `abs` has to carry both: 0.005 + 0.0031 at
    # this magnitude. The gate was always narrower than its own inputs -
    # before momwire#743 moved the numbers it read 31.21 against 31.2062 and
    # passed only because the value happened to sit near the 2dp grid.
    assert power.efficiency_percent == pytest.approx(
        100.0 * power.radiated_power / power.input_power, rel=1e-4, abs=0.0081
    )
    # And the whole budget lands where the capture's did, by envelope: 0.29,
    # 0.66 and 0.58 percentage points on the three loaded configs, measured,
    # and exactly zero on the two that read 200 %.
    assert abs(power.efficiency_percent - want.efficiency_percent) <= EFFICIENCY_BAR


@pytest.mark.integration
def test_the_two_decks_whose_source_stands_on_a_connection_point_read_200_percent():
    """0027 and 0028 against the other five, in one place, because a 200 %
    efficiency looks like a bug until you find the deck that explains it."""
    for cid in ("0027", "0028"):
        power = extract(served(cid)).power
        assert power.network_loss is None
        assert power.efficiency_percent == 200.0
        assert power.radiated_power == pytest.approx(2.0 * power.input_power, rel=1e-3)
    for cid in ("0012", "0017", "0018"):
        power = extract(served(cid)).power
        assert power.network_loss > 0.0
        assert power.efficiency_percent < 100.0


# --------------------------------------------------------------------------
# gate 6 — what still refuses, and where in the ladder
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_the_deck_the_refusal_order_existed_for_now_falls_all_the_way_through():
    """0001 — the 4-square with six ``TL`` cards over a bare ``GD`` — used to
    be the deck the refusal ORDER existed for, and is now the deck that says
    the ladder was climbed.

    Its networks were served by U5 and its ground by #504 U2, so nothing about
    it refuses any more: six ``TL`` rows, a driven virtual anchor 100 λ out,
    three ``1.E+10`` pins and a 361-point azimuth cut over 13/0.005 earth.  Ten
    of the corpus's forty-nine decks are this shape and all ten now answer.

    #504 U3 landed its printout, so every claim below is now also made against
    the capture in the gates above and the counts here are the captured ones
    rather than plausible ones.  Kept anyway, and deliberately: it reads as the
    one-line description of the deck the whole refusal ORDER was built around,
    which a parametrized byte-gate cannot do.

    The order itself did not change and did not need to; what changed is that
    there is no ground rung left to fall through, and since this unit no mixed
    table either.  The tests below are where the ordering is still visible.
    """
    printout = render(deck_text("0001"))
    assert "NEC ERROR" not in printout
    assert "- - - ANTENNA INPUT PARAMETERS - - -" in printout
    assert "FINITE GROUND.  SOMMERFELD SOLUTION" in printout
    data = extract(printout)
    assert len(data.networks) == 6
    assert len(data.loads) == 3
    (block,) = data.patterns
    assert len(block.rows) == 361


@pytest.mark.integration
@pytest.mark.parametrize("cid", MIXED_IDS)
def test_a_deck_carrying_both_card_types_prints_two_sub_tables_under_one_heading(cid):
    """``TL`` and ``NT`` together: the layout, measured at last.

    This was a REFUSAL for four units — the ``NETWORK DATA`` table has one
    column header block per card kind and no committed printout showed a table
    carrying both, so the seam said so rather than guess a heading.  The
    printouts existed; they were in the capture tree, unlanded, because the
    U1/U2 capture errand was scoped off the fixture manifest instead of off the
    corpus.  #504 U3 landed them and there is nothing left to guess.

    What they show, and every clause of it is gated here: ONE
    ``- - - NETWORK DATA - - -`` heading; one blank; the ``TL`` sub-table under
    the ``TL`` column header block; ONE blank; the ``NT`` sub-table under the
    ``NT`` block; then the section's ordinary three.  The two header blocks are
    compared BYTE for byte against the capture rather than masked, which is the
    point of doing it here as well as in the structure gate — they are the
    bytes that were unobserved, they differ from each other in ways a
    transcription could easily smooth (the ``TL`` block spells ``SEG.`` one
    space narrower on its first row), and a renderer that emitted one heading
    for both kinds would still pass a gate that only counted rows.
    """
    captured = printout_text(cid).split("\n")
    got = served(cid).split("\n")
    at = next(
        i for i, line in enumerate(captured) if "- - - NETWORK DATA - - -" in line
    )
    assert got.index(captured[at]) == at
    # The whole table, heading to the blank run that ends it, byte for byte.
    end = next(
        i
        for i in range(at + 1, len(captured))
        if not captured[i].strip() and not captured[i + 1].strip()
    )
    block = captured[at:end]
    assert got[at:end] == block

    # And the shape of it, said out loud so a failure names the clause.
    assert block[1] == ""
    headers = [i for i, line in enumerate(block) if "- FROM -" in line]
    assert len(headers) == 2
    assert block[headers[0] : headers[0] + 3] == list(_printout._LINE_COLUMNS)
    assert block[headers[1] : headers[1] + 3] == list(_printout._NETWORK_COLUMNS)
    # Exactly one blank between the last TL row and the NT header block.
    assert block[headers[1] - 1] == ""
    assert block[headers[1] - 2].endswith("STRAIGHT")

    rows = extract(served(cid)).networks
    lines = [row for row in rows if isinstance(row, LineRow)]
    nets = [row for row in rows if isinstance(row, NetworkRow)]
    assert rows == tuple(lines + nets), "the sub-tables are not TL-then-NT"
    assert lines and nets


@pytest.mark.integration
def test_a_deck_that_interleaves_the_two_card_kinds_refuses_by_name():
    """The one thing the mixed captures still cannot say.

    All three of them write every ``TL`` before every ``NT``, so two different
    rules fit every observed table — "the engine groups the sub-tables by card
    kind" and "the engine prints cards in order and re-heads whenever the kind
    changes" — and they part company only on a deck that interleaves.  So do
    the ``STRUCTURE EXCITATION DATA`` connection points, which print in card
    discovery order and would therefore start somewhere else.

    No capture writes one and EZNEC does not emit one; the probe below is
    synthetic, 0000's own deck with its ``NT`` card moved above its two ``TL``
    cards and nothing else touched.  It refuses, and it refuses for the ORDER
    rather than for the mixture — which is the distinction the sentence has to
    carry, because a reader whose deck was refused for carrying both kinds
    would go and remove one.
    """
    text = deck_text("0000")
    (nt,) = [line for line in text.split("\n") if line.startswith("NT ")]
    moved = text.replace(nt + "\n", "").replace("TL 3,1,", nt + "\nTL 3,1,", 1)
    assert moved.index("NT ") < moved.index("TL ")

    printout = render(moved)
    assert "this deck writes an NT card before a TL card" in printout
    assert "ANTENNA INPUT PARAMETERS" not in printout
    # The cards themselves are fine and the seam says so by not naming them:
    # the same deck in the captured order serves.
    assert "NEC ERROR" not in render(text)


@pytest.mark.integration
def test_two_addresses_at_a_five_wire_junction_refuse_rather_than_guess():
    """The two sides of ONE cut are served; two different cuts are not.

    At a node where exactly two wires meet, ``2,3`` and ``3,-1`` are the two
    ends of a single break and share the solver's port — that is config C, and
    it is the whole gate.  At 0013's five-wire apex two addresses would
    separate two DIFFERENT arms from the node, which is two independent series
    gaps at one junction; momwire declines to build them and no captured deck
    asks for it, so the seam says which addresses collided and how many wires
    met there.
    """
    text = deck_text("0013").replace(
        "PQ 0\n", "NT 1,-1,2,-1,.01,0.,0.,0.,0.,0.\nPQ 0\n"
    )
    printout = render(text)
    assert "address a node where 5 wires meet" in printout
    assert "1,-1 and 2,-1" in printout
    assert "ANTENNA INPUT PARAMETERS" not in printout


# --------------------------------------------------------------------------
# gate 7 — the phased drive THROUGH a network (momwire#511)
# --------------------------------------------------------------------------
#
# The composition, and the four captures that made it checkable.  Until
# 2026-08-20 this was a refusal by name: the reducer served networks, the
# sub-block solve served phased drives, composing them was a short edit, and
# nothing in forty-nine captures said what the engine did with the pair.  Four
# printouts arrived and said it.
#
# What the composition IS, in one sentence, because every gate below is a way
# of not taking it on trust: an `EX 4` fixes a port's SOURCE current — antenna
# plus network, the reducer's termination branch — so the seam measures the map
# from the driven sites' applied voltages to their source currents one unit
# probe at a time through the same reduced system the final solve uses, inverts
# it against the set currents, and solves once more at the answer.
#
# The two pairs are a CONTROLLED EXPERIMENT and that is why these four and not
# any four.  0117 is 0031 with one `TL` card added and nothing else changed;
# 0121 is 0000 with one `EX` card added and nothing else changed.  So each half
# of "phased AND networked" has a captured control on the other side of it, and
# the gates below can ask what the added card was WORTH rather than only
# whether the answer is plausible.
PHASED_NETWORK_IDS = ("0116", "0117", "0120", "0121")

# The two with an `RP` cut; the other two answer `XQ`.
PHASED_NETWORK_CUTS = ("0117", "0121")

# Every row of the four tables, measured 2026-08-21 and barred |dZ| plus 25 %,
# the same shape ``test_eznec_serve.py``'s :data:`~test_eznec_serve.
# PHASED_Z_BAR` uses on the network-free pair.  Per ROW, for that data's
# reason: four coupled ports are four different impedances and a seam that
# solved one wrong could still average into a plausible first row.
#
#   id    tag  Z (capture)        Z (served)         |dZ|
#   0116   1   -45.158-52.766j    -45.494-46.064j    6.710
#   0116   2    32.820-27.022j     33.414-21.526j    5.528
#   0116   3    72.032-65.530j     72.811-60.461j    5.129
#   0116   4    55.690+37.801j     58.338+45.265j    7.920
#   0120   3    0.22391+48.345j   -0.064422+48.581j  0.373
#   0120   2    49.710-4.6729j     50.224-4.4527j    0.559
#
# 0117 and 0121 are the same six rows again (only the request card differs), so
# they carry the same bars rather than measured twins of them — which is itself
# a claim, and :func:`test_the_request_card_does_not_move_the_drive` is where
# it is checked.
#
# 0120's first row is the one entry here where the SIGN of R differs between
# the two codes, +0.224 Ω captured against −0.064 Ω served, and it is not the
# 0031 case: that port is a virtual node behind an `LD 4 … 1.E+10` pin and an
# L-network, its resistance is 0.5 % of its own |Z|, and the whole row is
# 0.22 W of a 49.93 W budget.  0031's −1.779 Ω and 0116's −45.158 Ω are
# absorbing ELEMENTS, tens of ohms deep and reproduced with the sign intact
# (:func:`test_the_networked_four_square_s_absorbing_element_stays_negative`);
# this is a number passing through zero.
PHASED_NETWORK_Z_BAR = {
    "0116": (8.39, 6.91, 6.41, 9.90),
    "0117": (8.39, 6.91, 6.41, 9.90),
    "0120": (0.47, 0.70),
    "0121": (0.47, 0.70),
}

# The 361-row azimuth cuts, gated at EVERY printed angle and again over the
# rows within 10 dB of the peak — U4's two-bar treatment
# (``test_eznec_serve.py::test_the_phased_azimuth_cut_agrees_at_every_printed_
# angle``), for U4's reason: an array's azimuth cut is shaped by the PHASING,
# direction by direction, so agreeing across a full turn says the drives
# reached the right ports with the right phases.  Here it says one thing more,
# because the phases arrive through a network: a composition that solved the
# map but mis-ordered its columns would aim the pattern somewhere else.
#
#   id    rows  worst (all)  worst (<=10 dB down)  peak cap/served
#   0117   361     0.76 dB         0.20 dB           5.45 / 5.39
#   0121   361     0.50 dB         0.27 dB          -0.94 / -0.97
#
# Measured plus 25 %.  0117's whole-cut worst is 0031's own 0.58 family and
# lands, as 0031's does, in the deep rear null.
PHASED_NETWORK_AZIMUTH_BAR = {"0117": 0.95, "0121": 0.63}
PHASED_NETWORK_LOBE_BAR = {"0117": 0.25, "0121": 0.34}
_LOBE_FLOOR = 10.0


@pytest.mark.integration
@pytest.mark.parametrize("cid", PHASED_NETWORK_IDS)
def test_every_phased_network_row_sits_inside_its_own_measured_envelope(cid):
    """All four of 0116's rows and both of 0120's, row by row.

    The feedpoint envelope above reads ``sources[0]`` and would pass a seam
    that composed the first port correctly and the rest by accident, so these
    four get their whole table pinned instead.  The TAG and SEGMENT of each row
    are gated exactly, because they are the deck's own: 0116 prints tags
    1/2/3/4 at global segments 1/7/13/19 in DECK order, and 0120 prints tag 3
    BEFORE tag 2 because that is the order its two ``EX`` cards are written.
    """
    want = extract(printout_text(cid)).sources
    got = extract(served(cid)).sources
    bars = PHASED_NETWORK_Z_BAR[cid]
    assert len(got) == len(want) == len(bars)
    for a, b, bar in zip(want, got, bars):
        assert (b.tag, b.segment, b.end_index) == (a.tag, a.segment, a.end_index)
        assert abs(b.impedance - a.impedance) <= bar, (
            f"{cid} tag {a.tag}: served {b.impedance}, captured {a.impedance}"
        )


@pytest.mark.integration
@pytest.mark.parametrize("cid", PHASED_NETWORK_IDS)
def test_every_phased_network_row_prints_the_current_its_own_card_set(cid):
    """The drive constraint, to the byte, once per row — and here it is a
    statement about the COMPOSITION rather than about a restore.

    On a network-free deck the set current is the sub-block solve's own answer
    and printing it back is bookkeeping.  Here the port's source current and
    its structure current are two different numbers — 0120's driven virtual
    node reports 1.4142 A of source against 3.1645E-11 A of structure, all of
    the rest having gone down an ``NT`` and two ``TL``s — so a seam that had
    composed the map on the wrong one of them would still print the card's
    current in this cell and would be wrong everywhere else.  Which is why this
    test is cheap and the envelope above is the expensive one; both are needed.

    The exact zeros are the part that would break first: a drive read back out
    of the solve rather than restored prints ``1.1102E-16`` where the capture
    prints ``0.0000E+00``.
    """
    (fixed,) = {source.kind for source in parse_nec5(deck_text(cid)).sources}
    assert fixed == 4
    drives = [source.drive for source in parse_nec5(deck_text(cid)).sources]
    want = extract(printout_text(cid)).sources
    got = extract(served(cid)).sources
    for row, captured, drive in zip(got, want, drives, strict=True):
        assert row.current == captured.current
        assert row.current == pytest.approx(drive, abs=5e-5)
        assert (row.current.real == 0.0) == (drive.real == 0.0)
        assert (row.current.imag == 0.0) == (drive.imag == 0.0)


@pytest.mark.integration
def test_one_added_tl_card_is_worth_the_whole_four_square():
    """0117 IS 0031 with one ``TL`` card added, so the network's worth is a
    number both captures print.

    Nothing else about the two decks differs — same four wires, same ``GD``,
    same four ``EX 4`` cards, same 361-point cut at theta = 67 — and the single
    ``TL 1,3,3,3,50.,3.048`` between two UNDRIVEN interior nodes moves tag 1
    from −1.7790 − 24.192j to −45.158 − 52.766j.

    The sharpest part is a SYMMETRY, and it is the reason this pair is worth
    more than a synthetic edit.  0031's tags 2 and 3 are the array's two side
    elements, carry the same drive and print one row twice (36.783 − 27.955j on
    both).  0117's line runs from wire 1 to wire 3, which is a diagonal, so the
    mirror is gone and the capture prints 32.820 − 27.022j against
    72.032 − 65.530j — 45 Ω apart on two ports whose ``EX`` cards are identical.
    A seam that hung the network anywhere but where the deck says would keep the
    two rows equal, and it would pass every envelope in this file doing it.
    """
    bare = extract(printout_text("0031")).sources
    lined = extract(printout_text("0117")).sources
    assert bare[1].impedance == bare[2].impedance
    assert abs(lined[1].impedance - lined[2].impedance) > 25.0

    served_bare = extract(served("0031")).sources
    served_lined = extract(served("0117")).sources
    assert served_bare[1].impedance == served_bare[2].impedance
    assert abs(served_lined[1].impedance - served_lined[2].impedance) > 25.0
    # And the move the card is worth, on a port it does not touch, on both
    # sides: 43 Ω captured, 44 Ω served.
    assert abs(lined[0].impedance - bare[0].impedance) > 25.0
    assert abs(served_lined[0].impedance - served_bare[0].impedance) > 25.0
    # The two decks really are one edit apart.
    assert (
        deck_text("0117")
        .replace("TL 1,3,3,3,50.,3.048,0.,0.,0.,0.\n", "")
        .split("CE\n")[1]
        == deck_text("0031").split("CE\n")[1]
    )


@pytest.mark.integration
def test_one_added_ex_card_is_worth_the_whole_cardioid():
    """0121 IS 0000 with one ``EX`` card added, which is the same experiment
    from the other side.

    0000 drives the L-network's virtual node alone and prints 31.592 + 25.945j
    there.  0121 adds ``EX 4,2,-1,0,0.,-1.414214`` — a second generator, on the
    far vertical's base, itself a ``TL`` endpoint — and the SAME port's row
    becomes 0.22391 + 48.345j.  The second source is not a perturbation of the
    first deck's answer; it is a different boundary condition on the same
    circuit, and the port that did not change reads 31 Ω differently because of
    it.

    Which is the claim a single-source seam cannot make and a wrongly composed
    one gets wrong quietly: if the composition had solved each source
    independently and added, the printed row would still be near 0000's.
    """
    alone = extract(printout_text("0000")).sources[0]
    paired = extract(printout_text("0121")).sources[0]
    assert (alone.tag, alone.segment) == (paired.tag, paired.segment)
    assert abs(paired.impedance - alone.impedance) > 25.0

    served_alone = extract(served("0000")).sources[0]
    served_paired = extract(served("0121")).sources[0]
    assert abs(served_paired.impedance - served_alone.impedance) > 25.0
    assert (
        deck_text("0121").replace("EX 4,2,-1,0,0.,-1.414214\n", "").split("CE\n")[1]
        == deck_text("0000").split("CE\n")[1]
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "cid,card,flipped",
    [
        ("0117", "EX 4,4,-1,0,-1.414214,0.", "EX 4,4,-1,0,1.414214,0."),
        ("0121", "EX 4,2,-1,0,0.,-1.414214", "EX 4,2,-1,0,0.,1.414214"),
    ],
)
def test_reversing_one_drive_moves_the_ports_it_did_not_touch(cid, card, flipped):
    """The composition's own mechanism, stated as the thing it is NOT.

    Four (two) cards are a simultaneous boundary condition on coupled ports, so
    every port's answer is a function of ALL the drives — and here the coupling
    runs through the STRUCTURE and through the NETWORK at once.  Reverse one
    card and a port whose own card was not touched has to move, with its printed
    current unchanged: 0117's tag 1 moves 33 Ω, 0121's first row 78 Ω.  A seam
    that had composed the drive as independent scales would print the same row
    both times.

    And the array stops being one: 0117's 361-row cut flattens from a 24.6 dB
    spread to 1.2 dB, which is the four-square with two elements now in phase.
    """
    text = deck_text(cid)
    assert card in text
    base = serve(parse_nec5(text)).sources
    moved = serve(parse_nec5(text.replace(card, flipped))).sources
    assert moved[0].current == base[0].current
    assert abs(moved[0].impedance - base[0].impedance) > 20.0


@pytest.mark.integration
def test_the_networked_four_square_s_absorbing_element_stays_negative():
    """0116's tag-1 row prints −45.158 Ω and −45.158 W, and so does this seam's.

    0031's own absorbing element is −1.7790 Ω; add the ``TL`` and it becomes
    twenty-five times deeper, because the line now carries power into that
    element on top of what the mutual coupling was already delivering.  The
    generator on tag 1 is being driven hard, and the printed ``INPUT POWER =
    1.1538E+02`` is the other three rows with 45 W taken OFF.

    Gated as a sign and an arithmetic identity rather than as a value, for the
    reason U4's twin gives: a bar of 6.7 Ω on a row whose R is −45 is satisfied
    by −38, and by nothing on the wrong side of zero.  A composition that took a
    magnitude anywhere — or that clamped the budget to positive watts — lands
    there and inside every envelope in this file.
    """
    captured = extract(printout_text("0116")).sources[0]
    row = extract(served("0116")).sources[0]
    assert captured.impedance.real < 0 and captured.power < 0
    assert row.impedance.real < 0, f"served R = {row.impedance.real}"
    assert row.power < 0, f"served P = {row.power}"
    assert row.power == pytest.approx(
        0.5 * (row.voltage * row.current.conjugate()).real, rel=1e-4
    )
    # The budget is the SIGNED sum: summing magnitudes would print 2.0570E+02.
    data = extract(served("0116"))
    assert data.power.input_power == pytest.approx(
        sum(r.power for r in data.sources), rel=2e-4
    )


@pytest.mark.integration
def test_the_cardioid_s_two_driven_connection_points_read_161_percent():
    """0120/0121 answer the question the brief for this unit asked: what
    happens when a driven site is ALSO a network connection point, twice.

    Both are.  ``EX 4,3,1`` stands on the ``NT``'s end A and on a ``TL``'s end
    A; ``EX 4,2,-1`` stands on the other ``TL``'s end B.  So 0027's
    double-count precedent governs and the capture confirms it to the digit:
    the four connection-point powers sum to 4.9934E+01, which IS the input
    power (a lossless feed system delivers everything it is given), and
    ``RADIATED = INPUT + Σ − WIRE LOSS`` prints 8.0582E+01 against an input of
    4.9934E+01 — 161.38 %, the corpus's first budget strictly between 100 and
    200.  What keeps it off 200 is the 19.285 W in the two 18 Ω base loads.

    0116/0117 are the control and they are the OTHER answer to the same
    question: their ``TL`` hangs between two nodes neither ``EX`` touches, the
    two connection-point powers are ±4.0323E+01 and cancel, and the budget
    reads 100.00 % with ``RADIATED = INPUT``.  Nothing in the seam tests for
    which case it is in — the sum is the sum — and that these two decks land on
    two different budgets from one rule is the evidence the rule is the
    engine's.
    """
    for cid in ("0120", "0121"):
        power = extract(served(cid)).power
        data = extract(served(cid))
        assert power.network_loss is None
        assert power.efficiency_percent == pytest.approx(161.38, abs=EFFICIENCY_BAR)
        assert sum(row.power for row in data.network_excitation) == pytest.approx(
            power.input_power, rel=1e-3
        )
    for cid in ("0116", "0117"):
        data = extract(served(cid))
        assert data.power.efficiency_percent == pytest.approx(100.0, abs=1e-2)
        assert data.power.radiated_power == pytest.approx(
            data.power.input_power, rel=1e-6
        )
        assert sum(row.power for row in data.network_excitation) == pytest.approx(
            0.0, abs=1e-9 * data.power.input_power
        )


@pytest.mark.integration
@pytest.mark.parametrize("cid", PHASED_NETWORK_CUTS)
def test_the_phased_network_azimuth_cut_agrees_at_every_printed_angle(cid):
    """All 361 rows, and the lobe rows twice — U4's two-bar treatment, on the
    two composed decks that ask for a cut.

    Neither has a single ``-999.99`` row, which is checked rather than assumed:
    a null appearing on one side and not the other would otherwise be silently
    skipped.
    """
    (want,) = extract(printout_text(cid)).patterns
    (got,) = extract(served(cid)).patterns
    assert len(got.rows) == len(want.rows) == 361
    assert not [row for row in want.rows + got.rows if row.total_db == -999.99]

    peak = max(row.total_db for row in want.rows)
    worst = max(abs(a.total_db - b.total_db) for a, b in zip(want.rows, got.rows))
    assert worst <= PHASED_NETWORK_AZIMUTH_BAR[cid], f"{cid}: worst row {worst:.4f} dB"
    lobe = max(
        abs(a.total_db - b.total_db)
        for a, b in zip(want.rows, got.rows)
        if peak - a.total_db <= _LOBE_FLOOR
    )
    assert lobe <= PHASED_NETWORK_LOBE_BAR[cid], f"{cid}: worst lobe row {lobe:.4f} dB"


@pytest.mark.integration
@pytest.mark.parametrize("pair", [("0116", "0117"), ("0120", "0121")])
def test_the_request_card_does_not_move_the_drive(pair):
    """0116/0117 and 0120/0121 are two decks each, differing only in ``XQ``
    against ``RP``, so the composed drive must come out BIT identical.

    The pins above lean on that — the two members of each pair share one row of
    the measured table rather than carrying twins of it — and it is the control
    on the whole pipeline besides: a solve that depended in any way on which
    request card followed it would be reading the deck in the wrong order.
    """
    xq, rp = (serve(parse_nec5(deck_text(cid))).sources for cid in pair)
    assert [row.impedance for row in xq] == [row.impedance for row in rp]
    assert [row.voltage for row in xq] == [row.voltage for row in rp]


@pytest.mark.integration
def test_the_composition_solves_the_drive_rather_than_asserting_it():
    """The sharpest statement gate 7 can make, and the only one that reaches
    past the printed cell.

    Every ``ANTENNA INPUT PARAMETERS`` row prints the current its card set
    because :func:`~momwire.eznec._serve._source_row` restores it — that is the
    U1 rule and it would print the same cell over a composition that had solved
    nothing at all.  So this test goes back to the reduced system with the
    applied voltages the seam actually printed and reads the TERMINATION-branch
    currents out of it: antenna plus network, at every driven site, on the same
    circuit the answer came from.  They have to BE the set currents, and to the
    solve's own round-off rather than to the cell's four digits.

    Measured at 1e-12 relative on all four, which is a linear solve's residual
    and not an agreement.
    """
    for cid in PHASED_NETWORK_IDS:
        deck = parse_nec5(deck_text(cid))
        structure = _serve.structure_of(deck)
        mesh = _serve.build_mesh(deck, structure)
        by_address = {site.at: site for site in mesh.sites}
        for load in deck.loads:
            by_address[load.at].load += load.impedance
        for source in deck.sources:
            by_address[source.at].driven = True
        wavelength = _serve.SPEED_OF_LIGHT_MHZ_M / float(deck.frequency_mhz)
        medium = _serve._medium(deck.ground, wavelength)
        cards = _serve._cards(deck, structure, mesh)
        solver = _serve._solver_for(deck, mesh, wavelength, medium)
        state = _serve._port_state(
            deck, mesh, cards, solver.compute_port_solution().y, wavelength
        )

        n = len(mesh.sites)
        t = _serve._transform(mesh)
        y = t.T @ solver.compute_port_solution().y @ t
        z_load = np.array([site.load for site in mesh.sites], dtype=np.complex128)
        loaded = np.eye(n, dtype=np.complex128) + z_load[:, None] * y
        y_eff = np.linalg.solve(loaded.T, y.T).T if np.any(z_load) else y
        driven = tuple(site.index for site in mesh.sites if site.driven)
        _v, _i, i_source = _serve._reduced_state(
            cards, n, state.v_applied, driven, y_eff, wavelength
        )

        for source in deck.sources:
            index = by_address[source.at].index
            assert i_source[index] == pytest.approx(
                source.drive, rel=1e-12, abs=1e-12
            ), f"{cid} at {source.at.tag},{source.at.written}"


# --------------------------------------------------------------------------
# the layering
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_the_card_semantics_are_the_engine_s_and_not_a_second_copy():
    """What six real fields on a ``TL`` or an ``NT`` MEAN is written down once,
    in ``momwire.deck._networks``, and read from there here.

    The design doc's rule is *parse per-reader, semantics once*
    (``networks-move-into-the-engine.md``), and this seam is the second
    reader.  What it owns is the ADDRESSING — nodes, favored wires, and the
    zero-length rule that measures between them — and it owns none of the
    circuit math: no ``transposed``, no ``Y21 = Y12``, no shunt branch is
    constructed anywhere in this module.
    """
    source = Path(_serve.__file__).read_text()
    assert "from ..deck._networks import card_branches" in source
    assert "transposed" not in source
    assert "def card_branches" not in source
    # And the rows it hands the renderer are the two the table has, no more —
    # separately on twenty-one of the captures and both at once on the three
    # mixed ones.
    for cid in NETWORK_IDS:
        kinds = {type(row) for row in serve(parse_nec5(deck_text(cid))).networks}
        assert kinds in ({NetworkRow}, {LineRow}, {NetworkRow, LineRow})
        assert (len(kinds) == 2) == (cid in MIXED_IDS)
