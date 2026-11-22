"""momwire#2093 / #2094: the NEC-5 seam connects wires at an interior node.

A node that two wires share is a junction whether it is an end of both, an end
of one landing inside the other (a T) or inside both (a crossing). The
licensed NEC-5 (nec5cl, black box, 2026-11-21) connects all three, and gives a
crossing or a T the answer of the same metal written as split wires. The
decks are invented and live in `tests/fixtures/nec5_junction_cuts_2093/`,
which the C++ printout sweeps (`test_razor_core_standalone.py`) also walk, so
the executable route is held to these printouts byte for byte there.

Measured on the licensed engine, at an interior node ``k`` of wire ``t`` that
another wire joins: the signed address ``t,k``, a TL or NT end there, and the
explicit ``(segment k, end 2)`` each put the card in the branch of segment
``k``; ``(segment k + 1, end 1)`` puts it in the branch of segment ``k + 1``.
Each equals the split spelling's address on that piece to every printed digit.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from momwire.dropin.nec5 import _serve
from momwire.dropin.nec5._shell import render
from momwire.deck._nec5 import parse_nec5

DECKS = pathlib.Path(__file__).resolve().parent / "fixtures" / "nec5_junction_cuts_2093"

_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d)((?:\s+[-+]?\d\.\d+E[-+]\d+){9})\s*$")

# Licensed NEC-5's ANTENNA INPUT PARAMETERS row per deck: (tag, seg, end, Z).
ORACLE = {
    "cross_two_wires": (1, 5, 1, 69.718 - 106.22j),
    "cross_four_wires": (1, 5, 1, 69.718 - 106.22j),
    "tee_two_wires": (2, 24, 1, 19.191 - 392.18j),
    "tee_three_wires": (3, 24, 1, 19.191 - 392.18j),
    "tee_stem_end_source": (2, 21, 2, 9.9781 - 211.86j),
    "tee_stem_end_load": (1, 5, 1, 96.463 - 112.73j),
    "branch_signed_node": (1, 10, 1, 314.97 - 600.06j),
    "branch_end_code_one": (1, 11, 2, 121.63 + 244.80j),
    "branch_load_end_code_one": (1, 4, 1, 138.38 - 655.99j),
    "branch_line": (1, 4, 1, 131.44 - 791.90j),
    "branch_network": (1, 4, 1, 138.27 - 784.56j),
}
# NEC-5 prints its own row to five figures; razor2p is its twin, and the
# survey's twin agreement on junction decks is ~1e-5.
TWIN_REL = 2e-4

_HEAD = "CM split\nCE\n"
_FS = "GE 0,-1\nFR 0,1,0,0,14.\nGN -1\n"
_TAIL = "PQ 0\nXQ 0\nEN\n"
_SPLIT_T = (
    "GW 1,10,0.,-5.,10.,0.,0.,10.,.001\n"
    "GW 2,8,0.,0.,10.,0.,0.,6.,.001\n"
    "GW 3,14,0.,0.,10.,0.,7.,10.,.001\n"
)
_SPLIT_T_LINE = _SPLIT_T + "GW 9,11,4.,-5.,10.,4.,5.,10.,.001\n"
_SPLIT_TEE = (
    "GW 1,10,0.,-5.,10.,0.,0.,10.,.001\n"
    "GW 2,10,0.,0.,10.,0.,5.,10.,.001\n"
    "GW 3,8,0.,0.,10.,0.,0.,6.,.001\n"
)

# (deck, its split spelling): the cut deck is the same metal and the same
# address, so every basis must give the split spelling's answer.
SPLIT = {
    "cross_two_wires": (DECKS / "cross_four_wires.nec").read_text(),
    "tee_two_wires": (DECKS / "tee_three_wires.nec").read_text(),
    "tee_stem_end_source": _HEAD + _SPLIT_TEE + _FS + "EX 0,3,-1,0,1.,0.\n" + _TAIL,
    "tee_stem_end_load": _HEAD
    + _SPLIT_TEE
    + _FS
    + "LD 4,3,-1,0,50.,25.\nEX 0,1,5,0,1.,0.\n"
    + _TAIL,
    # segment 10's branch is the first piece's end ...
    "branch_signed_node": _HEAD + _SPLIT_T + _FS + "EX 0,1,10,0,1.,0.\n" + _TAIL,
    # ... and segment 11's the second piece's start
    "branch_end_code_one": _HEAD + _SPLIT_T + _FS + "EX 0,3,-1,0,1.,0.\n" + _TAIL,
    "branch_load_end_code_one": _HEAD
    + _SPLIT_T
    + _FS
    + "LD 4,3,-1,0,50.,25.\nEX 0,1,4,0,1.,0.\n"
    + _TAIL,
    "branch_line": _HEAD
    + _SPLIT_T_LINE
    + _FS
    + "EX 0,1,4,0,1.,0.\nTL 1,10,9,5,50.,4.,0.,0.,0.,0.\n"
    + _TAIL,
    "branch_network": _HEAD
    + _SPLIT_T_LINE
    + _FS
    + "EX 0,1,4,0,1.,0.\nNT 1,10,9,5,.01,0.,-.005,0.,.01,0.\n"
    + _TAIL,
}

BASES = ("razor2p", "bs2")


def _rows(out: str, table: str = "ANTENNA INPUT PARAMETERS"):
    """``(tag, seg, end, I, Z)`` per row of one port table."""
    assert "NEC ERROR" not in out, out[-600:]
    body = out.split(table, 1)[1]
    rows = []
    for line in body.splitlines()[1:]:
        m = _ROW.match(line)
        if m:
            f = [float(x) for x in m.group(4).split()]
            rows.append(
                (
                    int(m.group(1)),
                    int(m.group(2)),
                    int(m.group(3)),
                    complex(f[2], f[3]),
                    complex(f[4], f[5]),
                )
            )
        elif rows:
            break
    return rows


def _deck(name: str) -> str:
    return (DECKS / f"{name}.nec").read_text()


@pytest.mark.parametrize("name", sorted(ORACLE))
def test_razor2p_answers_the_licensed_engine(name):
    tag, seg, end, z = ORACLE[name]
    (row,) = _rows(render(_deck(name), basis="razor2p"))
    assert row[:3] == (tag, seg, end)
    assert abs(row[4] - z) / abs(z) < TWIN_REL, (row[4], z)


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("name", sorted(SPLIT))
def test_the_joined_deck_is_its_split_spelling(name, basis):
    (joined,) = _rows(render(_deck(name), basis=basis))
    (split,) = _rows(render(SPLIT[name], basis=basis))
    # the same mesh with its pieces in another order: rounding only
    assert joined[3] == pytest.approx(split[3], rel=1e-9, abs=1e-15)
    assert joined[4] == pytest.approx(split[4], rel=1e-9)


@pytest.mark.parametrize("basis", BASES)
def test_the_network_rows_keep_the_decks_own_addresses(basis):
    """The connection-point rows name the deck's tag and segment, not a
    piece's, and carry the split spelling's currents."""
    joined = _rows(render(_deck("branch_line"), basis=basis), "NETWORK CONNECTION")
    split = _rows(render(SPLIT["branch_line"], basis=basis), "NETWORK CONNECTION")
    # NEC-5 prints `1 10 1` and `9 37 1` for the joined deck
    assert [r[:3] for r in joined] == [(1, 10, 1), (9, 37, 1)]
    for a, b in zip(joined, split, strict=True):
        assert a[3] == pytest.approx(b[3], rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("basis", BASES)
def test_an_address_past_the_cut_lands_on_its_own_segment(basis):
    """A source at node 15 of the 20-segment wire, beyond the T at node 10, is
    the split spelling's source at node 5 of the second half, and its row
    still prints the deck's segment 15."""
    joined = render(
        _HEAD
        + "GW 1,20,0.,-5.,10.,0.,5.,10.,.001\nGW 2,8,0.,0.,10.,0.,0.,6.,.001\n"
        + _FS
        + "EX 0,1,15,0,1.,0.\n"
        + _TAIL,
        basis=basis,
    )
    split = render(_HEAD + _SPLIT_TEE + _FS + "EX 0,2,5,0,1.,0.\n" + _TAIL, basis=basis)
    ((tag, seg, end, i, z),) = _rows(joined)
    assert (tag, seg, end) == (1, 15, 1)
    assert z == pytest.approx(_rows(split)[0][4], rel=1e-9)


@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("name", ["tee_stem_end_source", "tee_stem_end_load"])
def test_a_card_at_the_stem_end_is_served_2094(name, basis):
    """momwire#2094: a card at a wire end landing mid-wire used to escape as
    an INTERNAL ERROR (`node_gaps[0]: the start of wire 1 is not a member`)."""
    out = render(_deck(name), basis=basis)
    assert "INTERNAL ERROR" not in out
    assert _rows(out)[0][:3] == ORACLE[name][:3]


@pytest.mark.parametrize("basis", BASES)
def test_two_branches_of_one_address_refuse_by_name(basis):
    """NEC-5 serves cards on both branches of one joined node (231.49 -
    j441.32 here); this seam keys a site by (tag, node), so it refuses."""
    out = render(_deck("branch_both_sides"), basis=basis)
    assert "through both of its segments (10 and 11)" in out
    assert "INTERNAL ERROR" not in out


def test_a_deck_with_no_shared_interior_node_is_meshed_as_before():
    deck = parse_nec5(
        _HEAD
        + "GW 1,11,0.,-2.5,5.,0.,2.5,5.,.001\nGW 2,11,1.,-2.5,5.,1.,2.5,5.,.001\n"
        + _FS
        + "EX 0,1,6,0,1.,0.\n"
        + _TAIL
    )
    structure = _serve.structure_of(deck)
    assert _serve._junction_cuts(structure, frozenset()) == {}
    from momwire.razor import RazorSolver

    mesh = _serve.build_mesh(deck, structure, solver_class=RazorSolver)
    assert [(p.tag, p.first_node, p.last_node) for p in mesh.pieces] == [
        (1, 0, 11),
        (2, 0, 11),
    ]


def test_the_cut_keeps_the_decks_node_numbering():
    deck = parse_nec5(_deck("cross_two_wires"))
    structure = _serve.structure_of(deck)
    assert _serve._junction_cuts(structure, frozenset()) == {0: (10,), 1: (10,)}
    from momwire.razor import RazorSolver

    mesh = _serve.build_mesh(deck, structure, solver_class=RazorSolver)
    assert [(p.tag, p.first_node, p.last_node) for p in mesh.pieces] == [
        (1, 0, 10),
        (1, 10, 20),
        (2, 0, 10),
        (2, 10, 20),
    ]
    assert [len(j) for j in mesh.junctions] == [4]
    # one element entry per deck element, in deck order
    assert mesh.element_of[9] == (0, 9) and mesh.element_of[10] == (1, 0)
    # NEC-5 prints 41 nodes, 40 elements and 39 unknowns for this deck
    assert (structure.node_count, structure.unknown_count) == (41, 39)
