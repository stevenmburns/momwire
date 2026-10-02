"""momwire#1281: one GW through z = 0 over a finite ground, split there.

The EZNEC seam used to refuse a card with ends strictly on both sides of a
``GN 0`` / ``GN 2`` plane and ask for two cards meeting at z = 0.  It now cuts
the card where its line meets the plane, which for a straight card is the
deck's own statement.  Every deck node keeps its point; z = 0 inside segment
``s`` makes ``s`` two one-element halves, one per medium, junctioned to the
runs either side at ``s``'s old nodes.

The decisive gate is EQUIVALENCE, not accuracy: the auto-split deck solves as
the four-card deck a user would write by hand, in the same piece order, to
the last bit, on every basis that serves the hand-written spelling.  The
hand-written decks here are built from the through deck's own node points
(``repr`` floats), because a deck typed to six digits is a different mesh —
``hs_n21.nec`` writes the cut segment's top node as 0.221435 where it is
0.2214347619..., and the two answers differ in the eighth digit for that
reason alone.

Fixtures (``tests/fixtures/eznec_split_1281/README.txt``) are Dan AC6LA's
buried-radial vertical with the vertical and its 1-inch connector as ONE
card, plus the licensed engine's printouts, quoted as numbers only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from momwire.deck._nec5 import parse_nec5
from momwire.eznec import _serve
from momwire.eznec._shell import render

FIX = Path(__file__).parent / "fixtures" / "eznec_split_1281"
RAD = "1.02616E-3"
RADIALS = "".join(
    f"GW {t},30,0.,0.,-.0254,{x},{y},-.0254,{RAD}\n"
    for t, x, y in (
        (2, "5.1816", "0."),
        (3, "0.", "5.1816"),
        (4, "-5.1816", "0."),
        (5, "0.", "-5.1816"),
    )
)
TAIL = (
    "GE -1,-1\n{ld}FR 0,1,0,0,14.\nGN 0,0,0,0,13.,.005,1.,0.\n{ex}PQ 0\n"
    "RP 0,1,1,1000,90.,0.,0.,0.,0.\nEN\n"
)
LD5_ALL = "LD 5,0,0,0,5.7471E+7,1.\n"

# The bases that serve the hand-written crossing junction on this deck.
# `razor-nec5` is the shipped alias of `razor-2p`; `sinusoidal` refuses every
# knot feed (`test_sinusoidal_refuses_both_spellings_alike`).
SERVED = ("bspline", "razor-2p", "razor-nec5", "bspline-d1", "sinusoidal-galerkin")
# The two the PR lane carries; the rest are the integration lane's.
FAST = ("bspline", "razor-2p")


def fixture(name: str) -> str:
    """A fixture deck as momwire can read it.  The captured decks write
    ``LD 5,0,1,141`` — EZNEC's absolute range for the 141-segment original —
    which on thr_n204 (324 segments) coats only part of wire 1, and momwire
    refuses a material range that splits a wire.  So the copper is put on
    every wire, which is what the original deck meant."""
    return (FIX / f"{name}.nec").read_text().replace("LD 5,0,1,141,", "LD 5,0,0,0,")


def through(n: int, *, ex: str, ld: str = LD5_ALL, top: str = "5.158130") -> str:
    return (
        f"CE\nGW 1,{n},0.,0.,-.0254,0.,0.,{top},{RAD}\n"
        + RADIALS
        + TAIL.format(ld=ld, ex=ex)
    )


def _z(value: float) -> str:
    return repr(float(value))


class HandSplit:
    """The four-card spelling of ``through``'s wire 1, in the auto split's
    piece order: below run (tag 11), below half (12), above half (13), above
    run (14).  A run that would be empty is not written."""

    def __init__(self, n: int, top: float = 5.158130, bottom: float = -0.0254):
        wire = _serve.Nec5Wire(1, n, (0.0, 0.0, bottom), (0.0, 0.0, top), 1.02616e-3)
        z = [p[2] for p in _serve.node_points(wire)]
        self.n = n
        self.j = next(k for k in range(n) if z[k] < 0.0 < z[k + 1])
        j = self.j
        cards = []
        if j > 0:
            cards.append(f"GW 11,{j},0.,0.,{_z(bottom)},0.,0.,{_z(z[j])},{RAD}\n")
        cards.append(f"GW 12,1,0.,0.,{_z(z[j])},0.,0.,0.,{RAD}\n")
        cards.append(f"GW 13,1,0.,0.,0.,0.,0.,{_z(z[j + 1])},{RAD}\n")
        if j + 1 < n:
            cards.append(
                f"GW 14,{n - j - 1},0.,0.,{_z(z[j + 1])},0.,0.,{_z(top)},{RAD}\n"
            )
        self.cards = "".join(cards)

    def address(self, node: int) -> str:
        """Deck node ``node`` of wire 1 as the hand-split deck names it: the
        START of the later piece wherever two meet, which is the end the
        seam's own `piece_of_node` records."""
        j, n = self.j, self.n
        if node < j:
            return f"11,{node if node else -1}"
        if node == j:
            return "12,-1"
        if node == j + 1:
            return "14,-1" if j + 1 < n else "13,1"
        return f"14,{node - j - 1}"

    def deck(self, *, ex: str, ld: str = LD5_ALL) -> str:
        return "CE\n" + self.cards + RADIALS + TAIL.format(ld=ld, ex=ex)


def serve_z(text: str, basis: str) -> complex:
    return _serve.serve(parse_nec5(text), basis=basis).sources[0].impedance


def ex(address: str) -> str:
    return f"EX 0,{address},0,1.,0.\n"


# --------------------------------------------------------------------------
# where the split lands
# --------------------------------------------------------------------------


def _mesh(text: str):
    """The mesh `serve` builds, crossing flag included: a deck with a crossing
    junction takes the delta-gap spelling and is never cut at an address."""
    deck = parse_nec5(text)
    return deck, _serve.build_mesh(
        deck,
        _serve.structure_of(deck),
        crossing=bool(_serve._crossing_nodes(deck)),
    )


def test_inside_a_segment_the_segment_becomes_two_and_every_node_stays():
    deck, mesh = _mesh(fixture("thr_n21"))
    (crossing,) = _serve._plane_crossings(deck).values()
    assert crossing.node is None and crossing.segment == 0
    assert crossing.point == (0.0, 0.0, 0.0)
    wire1 = [p for p in mesh.pieces if p.tag == 1]
    # 21 deck segments solve as 22: two one-element halves, then 20.
    assert [p.n_elements for p in wire1] == [1, 1, 20]
    assert sum(p.n_elements for p in wire1) == 21 + 1
    nodes = _serve.node_points(deck.wire(1))
    assert tuple(wire1[0].points[0]) == nodes[0]
    assert tuple(wire1[0].points[1]) == (0.0, 0.0, 0.0)
    assert tuple(wire1[1].points[0]) == (0.0, 0.0, 0.0)
    assert tuple(wire1[1].points[1]) == nodes[1]
    assert tuple(wire1[2].points[0]) == nodes[1]
    assert (wire1[0].cut_end, wire1[1].cut_start) == (True, True)
    # The printout keeps deck numbering: one row per DECK element.
    assert len(mesh.element_of) == deck.wire(1).segment_count + 4 * 30
    # Segment 1 spans -0.0254 .. 0.2214, so its centre (0.098) is ABOVE.
    split = mesh.split_elements[0]
    assert (split.lower, split.upper, split.centre_piece) == (0, 1, 1)
    centre = 0.5 * (nodes[0][2] + nodes[1][2])
    assert split.centre_s == pytest.approx(centre)
    assert mesh.element_of[0] == (1, 0)
    # The crossing is a junction of the two halves, and so is the old node 1.
    assert [(0, "end"), (1, "start")] in mesh.junctions
    assert [(1, "end"), (2, "start")] in mesh.junctions


def test_a_centre_below_the_plane_is_read_on_the_below_half():
    # 10 one-metre segments from z = -1.95: segment 2 spans -0.95 .. 0.05,
    # centre -0.45, so the charge is read on the BELOW half.
    text = (
        "CE\nGW 1,10,0.,0.,-1.95,0.,0.,8.05,.001\nGE -1,-1\nFR 0,1,0,0,14.\n"
        "GN 0,0,0,0,13.,.005\nEX 0,1,5,0,1.,0.\nPQ 0\nXQ 0\nEN\n"
    )
    deck, mesh = _mesh(text)
    assert _serve._plane_crossings(deck)[1].segment == 1
    split = mesh.split_elements[1]
    assert split.centre_piece == split.lower
    assert split.centre_s == pytest.approx(0.5)
    assert mesh.element_of[1] == (split.lower, 0)


def test_a_node_on_the_plane_becomes_the_crossing_node_and_nothing_else_moves():
    # Ten 1 m segments from z = -2: node 2 is at z = 0 exactly.
    text = (
        "CE\nGW 1,10,0.,0.,-2.,0.,0.,8.,.001\nGE -1,-1\nFR 0,1,0,0,14.\n"
        "GN 0,0,0,0,13.,.005\nEX 0,1,5,0,1.,0.\nPQ 0\nXQ 0\nEN\n"
    )
    deck, mesh = _mesh(text)
    crossing = _serve._plane_crossings(deck)[1]
    assert (crossing.node, crossing.segment) == (2, None)
    assert [p.n_elements for p in mesh.pieces] == [2, 8]
    assert not mesh.split_elements
    assert mesh.junctions == [[(0, "end"), (1, "start")]]


@pytest.mark.parametrize("ground", ["GN -1", None])
def test_no_interface_no_split(ground):
    """Free space (and no ground card at all) has no plane to split at."""
    card = f"{ground}\n" if ground else ""
    text = (
        "CE\nGW 1,10,0.,0.,-2.,0.,0.,8.,.001\nGE 0,-1\nFR 0,1,0,0,14.\n"
        f"{card}EX 0,1,5,0,1.,0.\nPQ 0\nXQ 0\nEN\n"
    )
    deck, mesh = _mesh(text)
    assert _serve._plane_crossings(deck) == {}
    assert not mesh.split_elements
    # Cut only at the addressed node (the node-gap spelling), never at z = 0.
    assert [(p.first_node, p.last_node) for p in mesh.pieces] == [(0, 5), (5, 10)]
    assert not any(p.cut_start or p.cut_end for p in mesh.pieces)
    assert _serve.refusal(deck) is None


def test_a_wire_that_only_touches_the_plane_is_not_split():
    hand = HandSplit(21)
    deck, mesh = _mesh(hand.deck(ex=ex(hand.address(1))))
    assert _serve._plane_crossings(deck) == {}
    assert not mesh.split_elements
    assert [p.tag for p in mesh.pieces][:3] == [12, 13, 14]


# --------------------------------------------------------------------------
# the decisive gate: the split IS the hand-written junction
# --------------------------------------------------------------------------


def _equivalence(basis: str, n: int, feed: int, ld: str = LD5_ALL) -> None:
    hand = HandSplit(n)
    auto = serve_z(through(n, ex=ex(f"1,{feed}"), ld=ld), basis)
    written = serve_z(hand.deck(ex=ex(hand.address(feed)), ld=ld), basis)
    assert auto == written, f"{basis} n={n}: {auto!r} vs {written!r}"


@pytest.mark.slow
@pytest.mark.parametrize("basis", FAST)
@pytest.mark.parametrize(
    "n,feed",
    [(21, 1), (204, 1), (21, 2)],
    ids=["thr_n21", "thr_n204", "dan_through"],
)
def test_the_split_deck_is_the_hand_split_deck_bit_for_bit(basis, n, feed):
    """thr_n21 is the cut-segment-becomes-two mesh; dan_through is the same
    card fed one node higher, inside the above run; thr_n204's z = 0 falls
    9.46 um below node 1 (204 x 25.41 mm is not the 5.18353 m card), so its
    above half is a 9.46 um sliver — the deck's geometry, not the split's."""
    _equivalence(basis, n, feed)


@pytest.mark.integration
@pytest.mark.parametrize("basis", [b for b in SERVED if b not in FAST])
@pytest.mark.parametrize(
    "n,feed",
    [(21, 1), (204, 1), (21, 2)],
    ids=["thr_n21", "thr_n204", "dan_through"],
)
def test_the_other_served_bases_see_the_same_equivalence(basis, n, feed):
    _equivalence(basis, n, feed)


@pytest.mark.parametrize("name", ["dan_through", "thr_n21", "thr_n204"])
def test_the_fixture_decks_are_in_scope(name):
    assert _serve.refusal(parse_nec5(fixture(name))) is None


@pytest.mark.slow
@pytest.mark.parametrize("basis", FAST)
@pytest.mark.parametrize("name,n,feed", [("thr_n21", 21, 1), ("dan_through", 21, 2)])
def test_the_fixture_decks_serve_as_their_hand_split(basis, name, n, feed):
    hand = HandSplit(n)
    assert serve_z(fixture(name), basis) == serve_z(
        hand.deck(ex=ex(hand.address(feed))), basis
    )


# --------------------------------------------------------------------------
# addresses on both sides of the cut, and loads across it
# --------------------------------------------------------------------------

# 10 one-metre segments from z = -1.95: the plane cuts segment 2 (nodes 1 and
# 2 at z = -0.95 and 0.05), so node 1 is a junction BELOW the plane and node 2
# one ABOVE it, both K = 2 and both real deck nodes.
N10 = dict(n=10, top=8.05, bottom=-1.95)


def _ten(ex_lines: str, ld: str = "") -> str:
    return (
        "CE\nGW 1,10,0.,0.,-1.95,0.,0.,8.05,.001\nGE -1,-1\n"
        + ld
        + "FR 0,1,0,0,14.\nGN 0,0,0,0,13.,.005\n"
        + ex_lines
        + "PQ 0\nXQ 0\nEN\n"
    )


def _ten_hand(ex_lines: str, ld: str = "") -> str:
    hand = HandSplit(**N10)
    cards = hand.cards.replace(RAD, ".001")
    return (
        "CE\n"
        + cards
        + "GE -1,-1\n"
        + ld
        + "FR 0,1,0,0,14.\nGN 0,0,0,0,13.,.005\n"
        + ex_lines
        + "PQ 0\nXQ 0\nEN\n"
    )


def _addr(node: int) -> str:
    return HandSplit(**N10).address(node)


@pytest.mark.parametrize("basis", FAST)
@pytest.mark.parametrize(
    "node", [1, 2, 6], ids=["below-the-cut", "above-the-cut", "inside-a-run"]
)
def test_a_source_by_node_on_either_side_is_the_hand_split_source(basis, node):
    auto = serve_z(_ten(ex(f"1,{node}")), basis)
    written = serve_z(_ten_hand(ex(_addr(node))), basis)
    assert auto == written


@pytest.mark.parametrize("basis", FAST)
def test_loads_either_side_and_a_wire_wide_ld5_are_the_hand_split_loads(basis):
    """LD 4 at the node below the cut, LD 0 at the node above it, and a
    WIRE-WIDE LD 5 on the split card, which must coat all four pieces exactly
    as one card per piece does."""
    auto_ld = "LD 4,1,1,0,10.,5.\nLD 0,1,2,0,5.,1.E-7,0.\nLD 5,1,0,0,5.7471E+7,1.\n"
    tags = (11, 12, 13, 14)
    hand_ld = f"LD 4,{_addr(1)},0,10.,5.\nLD 0,{_addr(2)},0,5.,1.E-7,0.\n" + "".join(
        f"LD 5,{t},0,0,5.7471E+7,1.\n" for t in tags
    )
    feed = ex("1,6")
    auto = _serve.serve(parse_nec5(_ten(feed, auto_ld)), basis=basis)
    written = _serve.serve(parse_nec5(_ten_hand(ex(_addr(6)), hand_ld)), basis=basis)
    assert auto.sources[0].impedance == written.sources[0].impedance
    assert auto.power.wire_loss == written.power.wire_loss
    assert auto.power.wire_loss > 0.0


@pytest.mark.parametrize("basis", FAST)
def test_the_tables_keep_deck_numbering(basis):
    """Every element away from the cut prints, bit for bit, what the
    hand-split deck prints for the same element.  The cut element prints one
    row at its old centre: the mean of its two END-node currents.  The
    hand-split deck prints two rows there, each the mean of one end node and
    the crossing — whose two sides need not carry the same number, since the
    crossing junction's continuity emerges from the fill rather than being
    imposed — so the cut element's two end-node currents are recovered from
    the rows AWAY from it, walking in from each free end, where the current
    is zero."""
    auto = _serve.serve(parse_nec5(_ten(ex("1,6"))), basis=basis)
    written = _serve.serve(parse_nec5(_ten_hand(ex(_addr(6)))), basis=basis)
    a = [row.real + 1j * row.imag for row in auto.currents]
    w = [row.real + 1j * row.imag for row in written.currents]
    assert len(a) == 10 and len(w) == 11
    assert a[0] == w[0]
    assert a[2:] == w[3:]
    i_node1 = 2.0 * w[0]  # the bottom end is free: I(node 0) = 0
    i_node2 = 0j  # and so is the top one
    for row in reversed(w[3:]):
        i_node2 = 2.0 * row - i_node2
    assert a[1] == pytest.approx(0.5 * (i_node1 + i_node2), rel=1e-9, abs=1e-15)
    qa = [row.magnitude for row in auto.charges]
    qw = [row.magnitude for row in written.charges]
    assert qa[0] == qw[0]
    assert qa[2:] == qw[3:]
    wavelength = auto.wavelength_m
    assert auto.currents[1].center[2] * wavelength == pytest.approx(-0.45)
    assert auto.currents[1].length * wavelength == pytest.approx(1.0)


@pytest.mark.parametrize("basis", FAST)
def test_a_split_on_a_node_is_the_two_card_deck_bit_for_bit(basis):
    """z = 0 ON node 2 of ten 1 m segments: two pieces meeting there, and
    the two-card deck a user would write solves to the same bits."""
    tail = "GE -1,-1\nFR 0,1,0,0,14.\nGN 0,0,0,0,13.,.005\n{ex}PQ 0\nXQ 0\nEN\n"
    auto = "CE\nGW 1,10,0.,0.,-2.,0.,0.,8.,.001\n" + tail.format(ex=ex("1,5"))
    written = (
        "CE\nGW 11,2,0.,0.,-2.,0.,0.,0.,.001\nGW 12,8,0.,0.,0.,0.,0.,8.,.001\n"
        + tail.format(ex=ex("12,3"))
    )
    assert serve_z(auto, basis) == serve_z(written, basis)


def test_a_port_at_the_crossing_node_refuses_naming_1282():
    text = (
        "CE\nGW 1,10,0.,0.,-2.,0.,0.,8.,.001\nGE -1,-1\nFR 0,1,0,0,14.\n"
        "GN 0,0,0,0,13.,.005\nEX 0,1,2,0,1.,0.\nPQ 0\nXQ 0\nEN\n"
    )
    with pytest.raises(_serve.ServeRefusal, match=r"momwire#1282"):
        _serve.serve(parse_nec5(text))


# --------------------------------------------------------------------------
# what did not change
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ground,card", [("GN 1", "GN 1"), ("GD 0,0,0,0,13.,.005", "GD")]
)
def test_over_a_ground_with_no_lower_medium_the_buried_part_still_refuses(ground, card):
    text = (
        "CE\nGW 1,10,0.,0.,-1.95,0.,0.,8.05,.001\nGE 1,-1\nFR 0,1,0,0,14.\n"
        f"{ground}\nEX 0,1,5,0,1.,0.\nPQ 0\nXQ 0\nEN\n"
    )
    reason = _serve.refusal(parse_nec5(text))
    assert reason is not None
    assert reason.startswith("wire 1 runs below the ground plane (min z = -1.95 m)")
    assert f"under a {card} card" in reason


@pytest.mark.parametrize("split", [True, False], ids=["one-card", "two-card"])
def test_a_crossing_decks_near_field_refuses_as_a_buried_decks(split):
    """The crossing junction's below member is a buried element, so the
    buried near-field refusal (momwire#524 phase 3) covers both spellings of
    a crossing deck, not only decks with a wholly-buried wire."""
    cards = (
        "GW 1,10,0.,0.,-2.,0.,0.,8.,.001\n"
        if split
        else "GW 1,2,0.,0.,-2.,0.,0.,0.,.001\nGW 2,8,0.,0.,0.,0.,0.,8.,.001\n"
    )
    text = (
        f"CE\n{cards}GE -1,-1\nFR 0,1,0,0,14.\nGN 0,0,0,0,13.,.005\n"
        "EX 0,1,1,0,1.,0.\nNE 0,1,1,1,2.,0.,1.,0.,0.,0.\nXQ 0\nEN\n"
    )
    reason = _serve.refusal(parse_nec5(text))
    assert reason is not None
    assert "buried deck's near field is not served" in reason


def test_sinusoidal_refuses_both_spellings_alike():
    """Unrelated to the split: SinusoidalSolver places no gap at a knot, so
    every node-addressed source on this dialect refuses, split or not."""
    hand = HandSplit(21)
    for text in (fixture("thr_n21"), hand.deck(ex=ex(hand.address(1)))):
        with pytest.raises(_serve.ServeRefusal, match="knot feeds are not served"):
            _serve.serve(parse_nec5(text), basis="sinusoidal")


# --------------------------------------------------------------------------
# the printout
# --------------------------------------------------------------------------


@pytest.mark.integration
def test_the_printout_says_the_wire_was_split_in_deck_numbering():
    out = render(fixture("thr_n21"))
    assert "NEC ERROR" not in out
    assert (
        "   WIRE 1 CROSSES Z = 0 INSIDE SEGMENT 1: SPLIT THERE INTO SEGMENT 1 BELOW "
        "AND SEGMENTS 1-21 ABOVE, SEGMENT 1 SOLVED AS TWO AND PRINTED AT ITS "
        "CENTRE, ABOVE (momwire#1281)."
    ) in out.splitlines()
    # The counts are the deck's, which is what the engine prints.
    assert " Number of wire elements :   141" in out


@pytest.mark.integration
def test_a_deck_with_nothing_to_split_prints_no_note():
    hand = HandSplit(21)
    assert "CROSSES Z = 0" not in render(hand.deck(ex=ex(hand.address(1))))


# --------------------------------------------------------------------------
# the oracle — recorded, never gated (house rule: never gate cross-formulation
# agreement).  NEC-5 numbers are the fixture printouts' ANTENNA INPUT rows.
# --------------------------------------------------------------------------

NEC5 = {
    "thr_n204": 71.111 + 1.2312j,
    "thr_n21": 72.373 + 4.3355j,
    "hs_n21": 71.296 + 0.044322j,
    "dan_through": 73.715 + 4.7422j,
}


@pytest.mark.integration
@pytest.mark.parametrize("basis", SERVED)
@pytest.mark.parametrize("name", ["thr_n204", "thr_n21", "dan_through"])
def test_record_the_split_decks_next_to_the_licensed_engine(
    name, basis, record_property
):
    z = serve_z(fixture(name), basis)
    record_property(f"z_{name}_{basis}", f"{z:.4f}")
    record_property(f"nec5_{name}", f"{NEC5[name]:.4f}")
    assert z.real > 0.0
