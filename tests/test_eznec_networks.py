"""Node-addressed networks, gated against the captures (#497, U5).

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
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from momwire.deck._nec5 import parse_nec5
from momwire.deck._networks import card_branches
from momwire.eznec import _serve, serve
from momwire.eznec._printout import LineRow, NetworkRow
from momwire.eznec._shell import render
from momwire.networks import TL
from test_eznec_printout import deck_text, extract, printout_text
from test_eznec_serve import mask, served

# The seven captures this unit brings into scope, taking the served corpus
# from 13 of 49 to 20.
NETWORK_IDS = ("0012", "0014", "0016", "0017", "0018", "0027", "0028")

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
# Two things in that table are worth reading twice.  The three W7EL configs
# sit 13.4 Ω from their captures and 0017 sits 6.8 Ω from ITS capture, which
# is the same formulation offset U4 measured on the bare decks and not
# something the networks added — the network arithmetic is exact, and what
# differs is the antenna underneath it.  And 0028, the log-periodic, is the
# widest at 19.5 Ω on a 148 Ω row: five coupled elements whose currents are
# set by a feeder network is the most basis-sensitive model in the corpus.
Z_BAR = {
    "0012": 16.72,
    "0014": 16.72,
    "0016": 16.72,
    "0017": 8.52,
    "0018": 7.84,
    "0027": 0.30,
    "0028": 24.42,
}

# |(Z_C - Z_A) served - (Z_C - Z_A) captured|: the SEPARATION between the
# series and parallel answers, measured at 6.56 Ω against a captured gap of
# 112.74 Ω, and pinned at that plus 25 %.  This is the number the unit exists
# for: a seam that folded the two addresses into one node would answer a gap
# of zero here, and 8.2 is a very long way from 112.7.
GAP_BAR = 8.2

# Peak total gain: |d| plus 25 % or 0.05 dB, whichever is larger, the printed
# cell being quantized at 0.01 dB.
PEAK_BAR = {"0012": 0.08, "0016": 0.08, "0027": 0.32, "0028": 0.05}

# Largest per-row difference in TOTAL gain among the rows the capture puts
# within 10 dB of its own peak, measured and plus 25 %.  0027 is the outlier
# by an order of magnitude and it is the physics: a cardioid is a NULL
# pattern, its null depth is set by how exactly two feed currents cancel, and
# two formulations that disagree by 0.2 Ω at the feedpoint disagree by whole
# dB down in the notch (8.3 dB at 20 dB below the peak, 1.7 within 10).
SHAPE_BAR = {"0012": 0.15, "0016": 0.15, "0027": 2.10, "0028": 0.08}

# Wire-current and charge-density tables, worst element relative to that
# table's own peak, measured and plus 25 % — same shape as U4's.
CURRENT_BAR = {
    "0012": 0.023,
    "0014": 0.023,
    "0016": 0.023,
    "0017": 0.022,
    "0018": 0.030,
    "0027": 0.177,
    "0028": 0.113,
}
CHARGE_BAR = {
    "0012": 0.079,
    "0014": 0.079,
    "0016": 0.079,
    "0017": 0.071,
    "0018": 0.072,
    "0027": 0.129,
    "0028": 0.114,
}

# Connection points that are ANTENNA terminals — the pinned virtual nodes,
# which report 1e10 and 2e5 Ω against currents of 1e-9 and 1e-16 A, are dust
# on both sides and are left out by the cutoff.  Worst row measured, plus
# 25 %: 0027's two line ends at 6.8 and 8.3 Ω, 0028's five at up to 13.0.
CONNECTION_BAR = {"0027": 10.4, "0028": 16.3}
CONNECTION_CUTOFF = 1e4

# EFFICIENCY, in percentage points: 0.29 on the three W7EL configs that share
# an answer, 0.66 on config C, 0.58 on the snapped one, and 0.00 on the two
# that read 200 % — where the number is structural rather than solved.
# Measured worst, plus 25 %.
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
    assert mask(served(cid)) == mask(printout_text(cid))


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


def test_the_snapped_config_moves_the_load_a_whole_segment_off_the_junction():
    """0018: ``NT 2,2`` and ``NT 2,3``, a genuine element apart.

    EZNEC snapped a 50 % request up to node 2 and told the user so; the deck
    that reaches the engine carries no fraction at all.  Both admittances are
    still in series, but now with a length of wire between them, and the
    answer moves again — 225.39 − j60.593, a third distinct number from one
    unchanged geometry.
    """
    assert abs(_impedance("0018") - Z_SNAP) <= Z_BAR["0018"]


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


@pytest.mark.parametrize("cid", NETWORK_IDS)
def test_the_sign_tracking_end_index_follows_the_deck_and_nothing_else(cid):
    """The trailing index after ``TAG``/``SEG.``: 2 for a ``-1`` node, 1 for
    a positive one, on every captured row of every network block.

    The capture study measured 9 of 9 rows; these seven printouts carry 26,
    and every one of them is compared against the deck's own spelling AND
    against the captured row.  It is not an end-one/end-two flag and not a
    port index — 0017 prints 2 and 1 for two points on ONE geometric node —
    which is why it is carried from the address rather than derived.
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
    """
    data = extract(served(cid))
    power, want = data.power, extract(printout_text(cid)).power
    assert (power.network_loss is None) == (want.network_loss is None)
    loss = power.network_loss or -sum(row.power for row in data.network_excitation)
    assert loss == pytest.approx(
        -sum(row.power for row in data.network_excitation), rel=1e-4
    )
    assert power.radiated_power == pytest.approx(
        power.input_power - loss - power.wire_loss, rel=1e-4
    )
    assert power.efficiency_percent == pytest.approx(
        100.0 * power.radiated_power / power.input_power, rel=1e-4
    )
    # And the whole budget lands where the capture's did, by envelope: 0.29,
    # 0.66 and 0.58 percentage points on the three loaded configs, measured,
    # and exactly zero on the two that read 200 %.
    assert abs(power.efficiency_percent - want.efficiency_percent) <= EFFICIENCY_BAR


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


def test_the_deck_the_refusal_order_existed_for_now_falls_all_the_way_through():
    """0001 — the 4-square with six ``TL`` cards over a bare ``GD`` — used to
    be the deck the refusal ORDER existed for, and is now the deck that says
    the ladder was climbed.

    Its networks were served by U5 and its ground by #504 U2, so nothing about
    it refuses any more: six ``TL`` rows, a driven virtual anchor 100 λ out,
    three ``1.E+10`` pins and a 361-point azimuth cut over 13/0.005 earth.  Ten
    of the corpus's forty-nine decks are this shape and all ten now answer.

    The order itself did not change and did not need to; what changed is that
    there is no ground rung left to fall through.  The test below is where the
    ordering is still visible.
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


def test_a_deck_carrying_both_card_types_refuses_for_its_table_and_not_its_physics():
    """``TL`` and ``NT`` together: the physics is served, the LAYOUT is not.

    The ``NETWORK DATA`` table has one column heading per card type and no
    captured printout shows a table carrying both, so the honest answer is a
    sentence rather than a guessed heading.

    0000 is the capture VERBATIM here, which it was not before.  The corpus's
    three mixed decks (0000, 0023, 0025) all stand over a bare ``GD``, so until
    #504 U2 they were told their GROUND was unserved and this refusal had to be
    given a hearing by swapping the card.  Now the ground is served and the
    mixed table is the first rung they fall through — which is the refusal
    order doing exactly its job, one rung further down than it used to.
    """
    printout = render(deck_text("0000"))
    assert "this deck carries TL and NT cards together" in printout
    assert "ANTENNA INPUT PARAMETERS" not in printout
    # And the reason names the table rather than the ground it stands over.
    assert "GD" not in printout.split(" ***** NEC ERROR - ")[1]


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
# the layering
# --------------------------------------------------------------------------


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
    # And the rows it hands the renderer are the two the table has, no more.
    for cid in NETWORK_IDS:
        kinds = {type(row) for row in serve(parse_nec5(deck_text(cid))).networks}
        assert kinds in ({NetworkRow}, {LineRow})
