"""The ``nec5`` deck dialect: the fourteen mnemonics EZNEC emits, and no more.

Three gates live here, in rising order of consequence.

**The corpus.** All 49 captured EZNEC launches under
``tests/fixtures/eznec/decks/`` parse.  That is the vocabulary gate: a card
form this front-end has not been taught refuses loudly, and the corpus is
what says the teaching is complete.

**The addressing.** W7EL's ``Network Connection Test`` triple — captures
``0012`` (A), ``0016`` (B), ``0017`` (C) — declares one geometric point two
different ways.  A and B name it through different favored wires and the
oracle answers 114.47 + j21.096 for both; C mixes them and answers
195.34 - j57.458.  So the parser must keep A's and B's addresses UNEQUAL,
and that inequality is the test.  A front-end that canonicalizes ``2,3`` and
``3,-1`` to one node returns A's impedance for C, 70 % wrong.

**The ground grammar.** Every semantic asserted below was measured against
the licensed oracle on 2026-08-20 and is recorded in the antennaknobs status
doc ``docs/status/2026-08-20-eznec-nec5-scored-matrix.md``, "Probe family 1".
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from momwire.deck import DeckError
from momwire.deck._nec5 import (
    Nec5ExecuteRequest,
    Nec5FarFieldRequest,
    Nec5FreeSpace,
    Nec5MininecGround,
    Nec5NearFieldRequest,
    Nec5Node,
    Nec5PerfectGround,
    Nec5SommerfeldGround,
    parse_nec5,
)

DECKS = Path(__file__).parent / "fixtures" / "eznec" / "decks"
CORPUS = sorted(DECKS.glob("*.nec"))


def deck(name: str):
    """One captured deck by its four-digit capture number."""
    (path,) = DECKS.glob(f"{name}_*.nec")
    return parse_nec5(path.read_text())


# A minimal well-formed deck, in the shape EZNEC writes: the bare-wires
# control from the capture study ("The bare-wires control", capture 0010).
DIPOLE = """CM a dipole
CE
GW 1,11,0.,-.25,0.,0.,.25,0.,.0005
GE 0,-1
FR 0,1,0,0,299.7925
GN -1
EX 4,1,6,0,1.414214,0.
PQ 0
RP 0,1,361,1000,90.,0.,0.,1.,0.
EN
"""


# -- gate 1: the corpus -----------------------------------------------------


def test_the_corpus_is_all_80_captures():
    """The gate is only a gate if it is the whole corpus.

    49 until momwire#511 landed the four decks that are BOTH phased and
    networked (0116/0117/0120/0121), 53 until momwire#516 landed the nine of
    the near-field errand (0107-0115), and 62 until the seven models below
    were PROMOTED out of the antennaknobs capture area.

    Those seven were captured in the 2026-08-20 sitting and then sat in
    `scratch/eznec-capture/` for five days because promotion is a separate
    step from capture and nothing failed while it was skipped. The handoff
    doc for the NEXT sitting duly listed them as uncaptured — reading this
    corpus rather than the capture index — and they were nearly re-captured
    on a second trip to the machine. Hence this sentence: the capture area
    is the source of record for what EXISTS, and this corpus is the source of
    record for what is GATED, and the gap between them is invisible from
    either side alone.
    """
    assert len(CORPUS) == 80


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.stem)
def test_every_captured_deck_parses(path: Path):
    parsed = parse_nec5(path.read_text())
    # Every capture is a real launch, so each carries geometry, a frequency
    # and a drive; a deck that parsed into nothing would pass a bare
    # "no exception" assertion.
    assert parsed.wires
    assert parsed.frequency_mhz is not None
    assert parsed.sources
    assert parsed.requests
    # PQ 0 precedes the request card in every deck (capture study, "Request
    # cards follow the display").
    assert parsed.pq == 0
    # The CM block is what the printout must echo for EZNEC to believe the
    # run is this one's (capture study, the fault-injection table).
    assert parsed.comments


def test_the_corpus_census_reproduces_the_published_weights():
    """Parsing all 80 decks recovers the scored matrix's own counts.

    "Parses without error" is a weak gate on its own: a front-end that read
    every ``GD`` as a ``GN 0`` would pass it while being 34 % wrong in R on
    any ground-mounted vertical.  The weights in
    ``docs/status/2026-08-20-eznec-nec5-scored-matrix.md`` ("The scored
    matrix") were counted independently of this code, so agreeing with them
    says the decks were read as the study read them.

    momwire#511's four decks were the first addition since the matrix was
    scored, and every column moved by exactly what they are: four more bare
    ``GD`` grounds (20 to 24), four more ``GE 1,-1`` (33 to 37), and two more
    of each request card, because the four are two decks captured twice, once
    under ``XQ`` and once under ``RP``.

    momwire#516's nine are the second, and they move the census the way a
    capture errand aimed at ONE card should: the ``NE``/``NH`` count goes from
    1 to 9 while ``XQ`` does not move at all.  The other columns move because
    the errand needed the same grid under four different grounds — two more
    ``GD``, four more ``GN 0``, one more ``GN 1``, two more ``GN -1`` — and one
    of the nine (0114) is an ``RP``, the free-space control 0115's ``NE`` is
    scored against.

    The seven PROMOTED models are the third, and they move it the way a
    broad-coverage batch should rather than the way an errand does: eighteen
    decks, no near field at all, and every ground column growing at once
    (``GN 0`` +10, ``GN -1`` +4, ``GD`` +4).  These are bundled EZNEC models
    — a quad, two Yagis, a W8JK, a Field Day dipole, an N4PC loop and K5RP —
    so they are the first corpus addition motivated by MODEL VARIETY instead
    of by a card, and the ``GN 0``/``GD`` pairs among them are the same
    antenna captured under both finite-ground spellings.
    """
    grounds = Counter()
    ge = Counter()
    requests = Counter()
    magnetic = 0
    for path in CORPUS:
        parsed = parse_nec5(path.read_text())
        ground = parsed.ground
        magnetic += bool(getattr(parsed.requests[0], "magnetic", False))
        if isinstance(ground, Nec5SommerfeldGround):
            grounds[f"GN {ground.spelling}"] += 1
        else:
            grounds[
                {Nec5FreeSpace: "GN -1", Nec5PerfectGround: "GN 1"}.get(
                    type(ground), "GD"
                )
            ] += 1
        ge[(parsed.ge_flag, parsed.ge_second)] += 1
        requests[type(parsed.requests[0]).__name__] += 1

    assert grounds == {"GD": 30, "GN -1": 22, "GN 0": 22, "GN 1": 6}
    assert ge == {(1, -1): 58, (0, -1): 22}
    assert requests == {
        Nec5FarFieldRequest.__name__: 38,
        Nec5ExecuteRequest.__name__: 33,
        Nec5NearFieldRequest.__name__: 9,
    }
    # Eight ``NE`` and ONE ``NH``: the fifteenth mnemonic costs one radio
    # button in the Near Field Analysis dialog and the errand spent it once,
    # on 0111, which is 0110 with nothing else changed.
    assert magnetic == 1


# -- gate 2: the favored wire ----------------------------------------------


def test_config_a_addresses_one_tag_twice():
    """``0012``: both loads declared on wire 3 at 0 % from end 1."""
    a = deck("0012")
    assert [n.end_a for n in a.networks] == [Nec5Node(3, 0), Nec5Node(3, 0)]
    # The node-0 spelling round-trips: the deck wrote -1 on both.
    assert [n.end_a.written for n in a.networks] == [-1, -1]


def test_config_b_addresses_the_same_point_through_another_tag():
    """``0016``: both loads declared on wire 2 at 100 % from end 1.

    Wire 2 runs (0,0,0) to (0,.125,0) in 3 segments and wire 3 begins at
    (0,.125,0), so wire 2's node 3 IS wire 3's node 0.
    """
    b = deck("0016")
    assert [n.end_a for n in b.networks] == [Nec5Node(2, 3), Nec5Node(2, 3)]
    assert [n.end_a.written for n in b.networks] == [3, 3]
    # And the geometry agrees that it is one point.
    wire2, wire3 = b.wire(2), b.wire(3)
    assert wire2 is not None and wire3 is not None
    assert wire2.end2 == wire3.end1


def test_a_and_b_name_one_point_and_stay_unequal():
    """**The gate.**  Same point, different favored wire, different model.

    The oracle answers A and B identically (114.47 + j21.096) — so the
    encoding is a true alias — and answers C 70 % away on a point that has
    not moved.  Only a model that keeps the tag can tell C from A, so the
    parser must NOT collapse these two addresses.
    """
    a, b = deck("0012"), deck("0016")
    assert a.networks[0].end_a != b.networks[0].end_a
    assert a.networks != b.networks
    # ...and it is the favored wire that differs, not the far end.
    assert a.networks[0].end_b == b.networks[0].end_b


def test_config_c_mixes_the_two_declarations():
    """``0017``: one load on each wire — the configuration that answers 195.34."""
    c = deck("0017")
    assert [n.end_a for n in c.networks] == [Nec5Node(3, 0), Nec5Node(2, 3)]
    a = deck("0012")
    assert c.networks != a.networks
    # C's first card is A's first card; only the second moved.
    assert c.networks[0] == a.networks[0]
    assert c.networks[1] != a.networks[1]


def test_addressing_reaches_every_connection_card():
    """``EX``, ``LD``, ``TL`` and ``NT`` all take the node address."""
    a = deck("0012")
    assert a.sources[0].at == Nec5Node(2, 0)  # EX 4,2,-1
    assert [ld.at for ld in a.loads] == [Nec5Node(4, 1), Nec5Node(4, 2)]
    coax = deck("0011")  # dipole with coax feedline: TL 2,-1,4,1
    assert coax.transmission_lines[0].end_a == Nec5Node(2, 0)
    assert coax.transmission_lines[0].end_b == Nec5Node(4, 1)


def test_minus_one_is_the_only_negative_in_the_whole_corpus():
    """The node-0 encoding's prediction, checked against all 49 decks."""
    for path in CORPUS:
        parsed = parse_nec5(path.read_text())
        addresses = [s.at for s in parsed.sources]
        addresses += [ld.at for ld in parsed.loads]
        for line in parsed.transmission_lines:
            addresses += [line.end_a, line.end_b]
        for network in parsed.networks:
            addresses += [network.end_a, network.end_b]
        assert addresses, path.name
        for address in addresses:
            assert address.written == -1 or address.written >= 1
            wire = parsed.wire(address.tag)
            assert wire is not None
            assert 0 <= address.node <= wire.segment_count


# -- gate 3: the ground grammar --------------------------------------------


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        pytest.param(("GN -1",), Nec5FreeSpace(), id="gn-free-space"),
        pytest.param(("GN 1",), Nec5PerfectGround(), id="gn-perfect-bare"),
        pytest.param(
            ("GN 0,0,0,0,13.,.005,1.,0.",),
            Nec5SommerfeldGround(13.0, 0.005, 1 + 0j, spelling=0),
            id="gn-0-sommerfeld",
        ),
        pytest.param(
            ("GN 2,0,0,0,13.,.005,1.,0.",),
            # Measured identical to GN 0 to every printed digit; the spelling
            # is recorded because the printout echoes the card as written.
            Nec5SommerfeldGround(13.0, 0.005, 1 + 0j, spelling=2),
            id="gn-2-is-gn-0",
        ),
        pytest.param(
            ("GD 0,0,0,0,13.,.005,1.,0.",),
            Nec5MininecGround(0, 13.0, 0.005, 1 + 0j),
            id="gd-mininec",
        ),
        pytest.param(
            # The integer fields 2-4 are ignored: GD 0,7,8,9 measured
            # equivalent to GD 0,0,0,0.
            ("GD 0,7,8,9,13.,.005,1.,0.",),
            Nec5MininecGround(0, 13.0, 0.005, 1 + 0j),
            id="gd-ignores-fields-2-to-4",
        ),
        pytest.param(
            ("GD -1,0,0,0,13.,.005,1.,0.",),
            Nec5FreeSpace(),
            id="gd-minus-one-cancels",
        ),
        pytest.param(
            # A negative sigma sets Im(eps_c) directly rather than being a
            # conductivity; the flag records which reading the deck asked for.
            ("GD 0,0,0,0,13.,-12.48,1.,0.",),
            Nec5MininecGround(0, 13.0, -12.48, 1 + 0j, sigma_sets_im_epsc=True),
            id="gd-negative-sigma",
        ),
        pytest.param(
            ("GD 0,0,0,0,13.,.005",),
            Nec5MininecGround(0, 13.0, 0.005, 1 + 0j),
            id="gd-omitted-mu-is-one",
        ),
        pytest.param(
            ("GD 0,0,0,0,13.,.005,0.,0.",),
            Nec5MininecGround(0, 13.0, 0.005, 1 + 0j),
            id="gd-zero-mu-is-one",
        ),
        # Last ground card wins — no NEC-2-style merge of GD into GN.
        pytest.param(
            ("GN 1", "GD 0,0,0,0,13.,.005,1.,0."),
            Nec5MininecGround(0, 13.0, 0.005, 1 + 0j),
            id="last-wins-gn1-then-gd",
        ),
        pytest.param(
            ("GD 0,0,0,0,13.,.005,1.,0.", "GN 1"),
            Nec5PerfectGround(),
            id="last-wins-gd-then-gn1",
        ),
        pytest.param(
            ("GD 0,0,0,0,13.,.005,1.,0.", "GN 0,0,0,0,20.,.0303,1.,0."),
            Nec5SommerfeldGround(20.0, 0.0303, 1 + 0j, spelling=0),
            id="last-wins-gd-then-gn0",
        ),
        pytest.param(
            ("GN 0,0,0,0,20.,.0303,1.,0.", "GD 0,0,0,0,13.,.005,1.,0."),
            Nec5MininecGround(0, 13.0, 0.005, 1 + 0j),
            id="last-wins-gn0-then-gd",
        ),
        pytest.param(
            ("GD 0,0,0,0,13.,.005,1.,0.", "GD 0,0,0,0,20.,.0303,1.,0."),
            Nec5MininecGround(0, 20.0, 0.0303, 1 + 0j),
            id="last-wins-gd-then-gd",
        ),
    ],
)
def test_ground_grammar(cards, expected):
    text = "\n".join(line for line in DIPOLE.splitlines() if not line.startswith("GN"))
    lines = text.splitlines()
    at = lines.index("EN")
    body = "\n".join([*lines[:at], *cards, "EN"]) + "\n"
    assert parse_nec5(body).ground == expected


def test_the_two_ground_mnemonics_do_not_alias():
    """``GD`` and ``GN 0`` carry the same media and are not the same ground.

    Both print ``FINITE GROUND.  SOMMERFELD SOLUTION``; ``Vert1`` answers
    35.571 - j1.4223 under bare ``GD`` and 47.789 - j0.78525 under ``GN 0``,
    a 34 % difference in R.  Captures ``0020`` and ``0021`` differ in nothing
    but the mnemonic, so the parser must too.
    """
    gd, gn = deck("0020"), deck("0021")
    assert isinstance(gd.ground, Nec5MininecGround)
    assert isinstance(gn.ground, Nec5SommerfeldGround)
    assert gd.ground != gn.ground
    assert (gd.ground.eps_r, gd.ground.sigma) == (gn.ground.eps_r, gn.ground.sigma)


def test_free_space_and_perfect_ground_ride_the_ge_flag():
    free, perfect = deck("0010"), deck("0043")
    assert free.ground == Nec5FreeSpace()
    assert (free.ge_flag, free.ge_second) == (0, -1)
    assert perfect.ground == Nec5PerfectGround()
    assert (perfect.ge_flag, perfect.ge_second) == (1, -1)


# -- gate 4: refusals name their card --------------------------------------


REFUSALS = [
    pytest.param(("LD 0,1,1,1,50.,0.",), "LD", id="ld-0"),
    pytest.param(("LD 1,1,1,1,50.,0.",), "LD", id="ld-1"),
    pytest.param(("LD 5,1,1,1,5.8E7,0.",), "LD", id="ld-5"),
    pytest.param(("LD 4,1,1,2,50.,0.",), "LD", id="ld-4-nonzero-fourth-field"),
    pytest.param(("EX 1,1,6,0,1.,0.",), "EX", id="ex-1"),
    pytest.param(("EX 5,1,6,0,1.,0.",), "EX", id="ex-5"),
    pytest.param(("RP 1,1,361,1000,90.,0.,0.,1.,0.",), "RP", id="rp-mode-1"),
    pytest.param(("RP 4,1,361,1000,90.,0.,0.,1.,0.",), "RP", id="rp-mode-4"),
    pytest.param(("RP 0,1,361,1002,90.,0.,0.,1.,0.",), "RP", id="rp-xnda-1002"),
    pytest.param(("QQ 0",), "QQ", id="unknown-card"),
    pytest.param(("NH 1,1,1,1,0.,0.,0.,0.,0.,0.",), "NH", id="nh-spherical"),
    pytest.param(("GX 0,101",), "GX", id="symmetry"),
    pytest.param(("FR 0,3,0,0,14.,0.05",), "FR", id="fr-multipoint"),
    pytest.param(("EX 4,1,-2,0,1.,0.",), "EX", id="node-minus-two"),
    pytest.param(("EX 4,1,0,0,1.,0.",), "EX", id="node-literal-zero"),
    pytest.param(("EX 4,1,12,0,1.,0.",), "EX", id="node-past-the-wire-end"),
    pytest.param(("EX 4,9,1,0,1.,0.",), "EX", id="node-on-an-undeclared-tag"),
    pytest.param(("GD 0,0,0,0,13.,.005,2.,0.",), "GD", id="gd-mu-2"),
    pytest.param(("GD 0,0,0,0,13.,.005,1.,0.5",), "GD", id="gd-mu-complex"),
    pytest.param(("GN 0,0,0,0,13.,.005,2.,0.",), "GN", id="gn-0-mu-2"),
    pytest.param(("GN 3,0,0,0,13.,.005,1.,0.",), "GN", id="gn-type-3"),
    pytest.param(("GN 0,0,0,0",), "GN", id="gn-0-short"),
    pytest.param(("GE 1",), "GE", id="ge-one-field"),
    pytest.param(("NE 1,1,1,1,0.,0.,0.,0.,0.,0.",), "NE", id="ne-spherical"),
    pytest.param(("TL 1,1,1,6,0.,1.,0.,0.,0.,0.",), "TL", id="tl-zero-z0"),
    pytest.param(("NT 1,1,1,6,.01,0.,0.,0.",), "NT", id="nt-short"),
    pytest.param(("GW 1,5,0.,0.,0.,0.,1.,0.,.0005",), "GW", id="gw-duplicate-tag"),
]


@pytest.mark.parametrize(("cards", "mnemonic"), REFUSALS)
def test_refusals_name_the_offending_card(cards, mnemonic):
    lines = DIPOLE.splitlines()
    at = lines.index("EN")
    body = "\n".join([*lines[:at], *cards, "EN"]) + "\n"
    with pytest.raises(DeckError) as excinfo:
        parse_nec5(body)
    assert mnemonic in str(excinfo.value)


def test_a_deck_without_en_refuses_by_name():
    """A divergence from the nec2 front-end, and a deliberate one.

    ``parse_nec2`` closes a deck at EOF whether or not a terminator arrived,
    because NEC's own reader does.  This seam serves EZNEC, which writes
    ``EN`` on all 49 captures, so a deck that stops early is a truncated file
    — the failure mode the fault-injection table showed EZNEC cannot
    diagnose for itself.
    """
    body = "\n".join(line for line in DIPOLE.splitlines() if line != "EN") + "\n"
    with pytest.raises(DeckError, match="EN"):
        parse_nec5(body)


def test_text_after_en_is_not_this_deck():
    trailing = DIPOLE + "GW 9,1,0.,0.,0.,1.,0.,0.,.001\nQQ 0\n"
    parsed = parse_nec5(trailing)
    assert [w.tag for w in parsed.wires] == [1]


def test_a_deck_without_ge_refuses_by_name():
    body = "\n".join(line for line in DIPOLE.splitlines() if not line.startswith("GE"))
    with pytest.raises(DeckError, match="GE"):
        parse_nec5(body + "\n")


def test_a_data_card_before_ce_refuses():
    with pytest.raises(DeckError, match="CE"):
        parse_nec5("CM a dipole\nGW 1,11,0.,-.25,0.,0.,.25,0.,.0005\nEN\n")


def test_a_comment_after_ce_refuses():
    lines = DIPOLE.splitlines()
    at = lines.index("EN")
    body = "\n".join([*lines[:at], "CM late", "EN"]) + "\n"
    with pytest.raises(DeckError, match="CM"):
        parse_nec5(body)


# -- the remaining cards ----------------------------------------------------


def test_comments_and_the_ce_terminator_are_kept_for_the_echo():
    parsed = deck("0012")
    assert parsed.comments[0] == " Network Connection Test"
    assert any("EZNEC Pro/2+ v. 7.0.4" in c for c in parsed.comments)
    assert parsed.ce_text == ""


def test_wires_keep_their_tags_and_segment_counts():
    parsed = deck("0012")
    assert [(w.tag, w.segment_count) for w in parsed.wires] == [
        (1, 6),
        (2, 3),
        (3, 3),
        (4, 3),
    ]
    assert parsed.wires[0].end1 == (0.0, -0.25, 0.0)
    assert parsed.wires[0].radius == 0.0005


def test_both_source_kinds_and_their_complex_drive():
    current = deck("0012").sources[0]
    assert (current.kind, current.drive) == (4, complex(1.414214, 0.0))
    voltage = deck("0033").sources[0]
    assert voltage.kind == 0


def test_a_phased_array_is_several_ex_cards():
    """The 40 m four-square: 0 / -90 / -90 / -180, no network at all."""
    parsed = deck("0031")
    assert [s.at.tag for s in parsed.sources] == [1, 2, 3, 4]
    assert [s.drive for s in parsed.sources] == [
        complex(1.414214, 0.0),
        complex(0.0, -1.414214),
        complex(0.0, -1.414214),
        complex(-1.414214, 0.0),
    ]


def test_the_pin_idiom_is_an_ordinary_load():
    pins = [ld for ld in deck("0012").loads if ld.impedance == complex(1e10, 0.0)]
    assert len(pins) == 2
    assert all(ld.node_to == 0 for ld in pins)


def test_a_crossed_line_keeps_the_sign_of_z0():
    """The log-periodic's phase-reversal feeder."""
    lines = deck("0028").transmission_lines
    crossed = [line for line in lines if line.crossed]
    assert crossed
    assert crossed[0].z0 == -490.0875
    assert all(line.z0 != 0.0 for line in lines)


def test_a_shorted_stub_is_a_shunt_admittance_on_end_two():
    (stub,) = [
        line
        for line in deck("0028").transmission_lines
        if line.shunt[2:] == (1e10, 1e10)
    ]
    assert stub.z0 == 490.0875
    assert stub.length_m == 0.1524


def test_an_nt_carries_three_complex_admittances():
    (junction, _) = deck("0012").networks
    assert (junction.y11, junction.y12, junction.y22) == (0.01 + 0j, 0j, 0j)
    l_network = deck("0025").networks[0]
    assert l_network.y11 != l_network.y22


def test_the_request_cards_follow_the_display():
    pattern = deck("0010").requests
    assert pattern == (
        Nec5FarFieldRequest(
            n_theta=1,
            n_phi=361,
            xnda=1000,
            theta0_deg=90.0,
            phi0_deg=0.0,
            d_theta_deg=0.0,
            d_phi_deg=1.0,
            range_m=0.0,
        ),
    )
    assert deck("0035").requests[0].xnda == 1001  # the 3-D form
    assert deck("0014").requests == (Nec5ExecuteRequest(flag=0),)
    assert deck("0022").requests == (
        Nec5NearFieldRequest(
            coordinates=0,
            counts=(1, 1, 1),
            origin=(0.0, 0.0, 0.0),
            step=(0.0, 0.0, 0.0),
        ),
    )


def test_nh_parses_as_ne_s_twin_with_the_magnetic_flag_up():
    """momwire#513: capture sitting 4 falsified "EZNEC never emits NH" —
    capture 0111 wrote one (the Near Field Analysis dialog's E/H radio
    button picks the mnemonic).  Same ten fields, same handler, one flag;
    the spherical form refuses exactly as NE's does (the REFUSALS table)."""

    def dipole_with(card: str) -> str:
        lines = DIPOLE.splitlines()
        at = lines.index("RP 0,1,361,1000,90.,0.,0.,1.,0.")
        return "\n".join([*lines[:at], card, *lines[at + 1 :]]) + "\n"

    (request,) = parse_nec5(dipole_with("NH 0,5,3,1,-1.,-2.,10.,0.5,1.,0.")).requests
    assert request == Nec5NearFieldRequest(
        coordinates=0,
        counts=(5, 3, 1),
        origin=(-1.0, -2.0, 10.0),
        step=(0.5, 1.0, 0.0),
        magnetic=True,
    )
    # And NE's own record keeps the flag DOWN - 0022 above is the gate that
    # the default did not drift.
    (electric,) = parse_nec5(dipole_with("NE 0,1,1,1,0.,0.,0.,0.,0.,0.")).requests
    assert electric.magnetic is False


def test_the_frequency_is_one_point_in_mhz():
    assert deck("0010").frequency_mhz == 299.7925
    assert deck("0002").frequency_mhz == 7.15
    assert deck("0009").frequency_mhz == 7.5
