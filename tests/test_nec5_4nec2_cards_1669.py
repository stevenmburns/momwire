"""GA, GH, GM, GX, GS and PT on the nec5 dialect, as 4nec2 writes them — momwire#1669.

4nec2's NEC-5 package writes these six cards and EZNEC does not, so the
dialect refused them until now. Every probe deck under
``tests/fixtures/nec5_4nec2_cards_1669`` (README there) was run once on our
licensed NEC-5, nec5cl-simd-d0c1383 (sha256 e8d2c60a...), as a black box,
and these tests hold momwire to its printouts:

* the STRUCTURE SPECIFICATION block — every geometry card's echo, the four
  counts and the Reflection symmetry block — byte for byte;
* every segment centre and length in the Wire Currents table, to the
  printout's own precision;
* the feed impedance, razor-2p (NEC-5's own basis) within a tolerance
  measured on these decks;
* PT -1 dropping the Wire Currents table and nothing else;
* and the forms this dialect refuses, each by name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from momwire.deck import DeckError
from momwire.deck._nec5 import _joined, parse_nec5
from momwire.eznec import _serve
from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures" / "nec5_4nec2_cards_1669"
PROBES = sorted(p.stem for p in FIXTURES.glob("*.nec"))


def deck(name: str) -> str:
    return (FIXTURES / f"{name}.nec").read_text()


def oracle(name: str) -> str:
    return (FIXTURES / f"{name}.out").read_text()


def _specification(text: str) -> list[str]:
    """The STRUCTURE SPECIFICATION block from the model commands to the
    last line before the first card echo, trailing blanks dropped."""
    lines = text.splitlines()
    start = next(
        i for i, line in enumerate(lines) if "Model definition commands" in line
    )
    end = next(i for i, line in enumerate(lines) if "***** INPUT LINE" in line)
    block = lines[start:end]
    while block and not block[-1].strip():
        block.pop()
    return block


def _currents(text: str) -> list[list[float]]:
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if "Wire Currents" in line) + 6
    except StopIteration:
        return []
    rows = []
    for line in lines[start:]:
        cells = line.split()
        if len(cells) < 9:
            break
        rows.append([float(c) for c in cells[2:6]])
    return rows


def _impedances(text: str) -> list[complex]:
    lines = text.splitlines()
    start = next(
        i for i, line in enumerate(lines) if "ANTENNA INPUT PARAMETERS" in line
    )
    out, seen = [], False
    for line in lines[start + 1 :]:
        values = re.findall(r"-?\d\.\d+E[+-]\d+", line)
        if len(values) < 6:
            if seen:
                break
            continue
        seen = True
        out.append(complex(float(values[4]), float(values[5])))
    return out


def test_the_probes_are_there():
    assert len(PROBES) == 29, PROBES


@pytest.mark.parametrize("name", PROBES)
def test_the_structure_specification_is_the_engines(name):
    """Every geometry card echoed in the engine's layout, the four counts,
    and the GX decks' Reflection symmetry block, byte for byte."""
    assert _specification(render(deck(name))) == _specification(oracle(name))


@pytest.mark.parametrize("name", PROBES)
def test_every_segment_centre_and_length_is_the_engines(name):
    """The printout carries six significant figures, so a row agrees to
    half a unit in its last place -- plus, where a wire end was joined to an
    earlier node (a5), the join's own reach: the engine joins the two nodes
    and keeps each element's coordinates, momwire moves the later end onto
    the node, by under 1e-4 of the earlier element's length."""
    ours = _currents(render(deck(name)))
    theirs = _currents(oracle(name))
    assert len(ours) == len(theirs)
    for mine, engine in zip(ours, theirs, strict=True):
        reach = 1e-4 * engine[3]
        for a, b in zip(mine, engine, strict=True):
            assert abs(a - b) <= 5e-6 * max(abs(b), 1e-2) + reach, (name, mine, engine)


# razor-2p is NEC-5's own basis. Measured against the licensed engine on these
# 29 decks (2026-10-24), its worst row is 8.7e-5 of |Z| (y10); bs2 on the same
# coarse meshes is 1.6 % to 31 % off, its own basis rather than a misread card.
RAZOR_RTOL = 2e-4


@pytest.mark.parametrize("name", PROBES)
def test_the_feed_impedance_agrees_with_the_engine_on_razor_2p(name):
    ours = _impedances(render(deck(name), basis="razor-2p"))
    theirs = _impedances(oracle(name))
    assert len(ours) == len(theirs)
    for z, ref in zip(ours, theirs, strict=True):
        assert abs(z - ref) <= RAZOR_RTOL * abs(ref), (name, z, ref)


def test_pt_minus_one_drops_the_wire_currents_table_and_nothing_else():
    with_pt = render(deck("pt1"))
    without = render(deck("p0"))
    assert "Wire Currents" not in with_pt
    assert "Wire Currents" in without
    assert "Wire Currents" not in oracle("pt1")
    assert _impedances(with_pt) == _impedances(without)


def test_the_last_pt_wins():
    assert "Wire Currents" in render(deck("p5"))
    assert parse_nec5(deck("p5")).pt == 0


def test_the_pt_printout_has_the_engines_sections():
    def sections(text: str) -> list[str]:
        return re.findall(r"- - - ([A-Za-z ]+?) - - -", text)

    assert sections(render(deck("pt1"))) == sections(oracle("pt1"))


def test_a_gm_move_in_place_renumbers_by_itgi():
    """q6: after ``GM 5 0 ..`` tag 1 answers as tag 6 (the engine stops on
    an EX naming tag 1 there)."""
    assert [w.tag for w in parse_nec5(deck("q6")).wires] == [6]


def test_gs_reads_no_tag_range():
    assert parse_nec5(deck("s1")).wires == parse_nec5(deck("s2")).wires


def test_an_arc_end_and_a_typed_gw_end_join():
    """a5: the GW's ``1.732`` stands 5e-5 m off the arc's 2 cos 30, and the
    engine reads one node there (14 nodes, 12 unknowns)."""
    parsed = parse_nec5(deck("a5"))
    arc, gw = parsed.wires
    assert gw.end1 == arc.end1


@pytest.mark.parametrize(
    "gap, joined", [(9e-5, True), (1e-4, False)], ids=["inside", "outside"]
)
def test_ends_join_inside_a_ten_thousandth_of_the_earlier_element(gap, joined):
    text = (
        "CM j\nCE\nGW 1 10 0 0 -10 0 0 0 1e-3\n"
        f"GW 2 10 {gap} 0 0 {gap} 0 10 1e-3\nGE 0\nEX 0 1 9 0 1 0\n"
        "FR 0 1 0 0 14 0\nXQ\nEN\n"
    )
    structure = _serve.structure_of(parse_nec5(text))
    assert structure.unknown_count == (19 if joined else 18)


def test_the_earlier_elements_length_sets_the_join():
    def unknowns(first: float, second: float, gap: float) -> int:
        text = (
            f"CM j\nCE\nGW 1 10 0 0 {-10 * first} 0 0 0 1e-3\n"
            f"GW 2 10 {gap} 0 0 {gap} 0 {10 * second} 1e-3\nGE 0\n"
            "EX 0 1 9 0 1 0\nFR 0 1 0 0 14 0\nXQ\nEN\n"
        )
        return _serve.structure_of(parse_nec5(text)).unknown_count

    assert unknowns(1.0, 0.1, 5e-5) == 19
    assert unknowns(0.1, 1.0, 2e-5) == 18


def test_no_eznec_deck_moves_at_the_join():
    """EZNEC writes a shared end as the same string twice, so the join is a
    no-op on every captured deck."""
    root = Path(__file__).parent / "fixtures"
    checked = 0
    for path in sorted(root.glob("eznec*/**/*.nec")):
        try:
            parsed = parse_nec5(path.read_text(encoding="latin-1"))
        except DeckError:
            continue
        assert _joined(list(parsed.wires)) == parsed.wires, path
        checked += 1
    assert checked > 50


# --------------------------------------------------------------------------
# what refuses, by name
# --------------------------------------------------------------------------

HEAD = "CM r\nCE\n"
WIRE = "GW 1 11 0 0.5 -5 0 0.8 5 1e-3\n"
TAIL = "GE 0\nEX 0 1 3 0 1 0\nFR 0 1 0 0 14 0\nXQ\nEN\n"


@pytest.mark.parametrize(
    "body, match",
    [
        ("GH 1 20 2 3 0.5 0.3 1e-3 2e-3 0\n", "two wire radii"),
        ("GH 1 21 0 3 0.5 0.5 1e-3 1e-3 0\n", "zero turns"),
        (WIRE + "GX 0 010\n", "tag 1, which another wire"),
        ("GW 1 11 0 0 -5 0 0 5 1e-3\nGX 1 100\n", "lies in or crosses"),
        (WIRE + "GM 1 1 0 0 0 0 1 0 7\n", "starts its block at tag 7"),
        (WIRE + "GM 1 1 0 0 0 0 1 0 0 3\n", "nine fields"),
        (WIRE + "GS 0 0 -1\n", "positive scale factor"),
        (
            "GW 1 11 0.5 0.5 1 0.6 0.8 6 1e-3\nGX 1 001\nGE 1\nGN 1\n",
            "Z = 0 plane over a ground",
        ),
    ],
    ids=[
        "gh-taper",
        "gh-no-turns",
        "duplicate-tag",
        "gx-in-plane",
        "gm-no-its",
        "gm-extra-field",
        "gs-negative",
        "gx-z-over-ground",
    ],
)
def test_a_form_this_dialect_does_not_serve_refuses_by_name(body, match):
    tail = TAIL if "GE" not in body else TAIL[5:]
    with pytest.raises(DeckError, match=match):
        parse_nec5(HEAD + body + tail)


@pytest.mark.parametrize("pt", ["PT 2 1 3 5", "PT 1 0 0 0", "PT 0 1 3 5"])
def test_a_pt_other_than_minus_one_or_zero_refuses(pt):
    with pytest.raises(DeckError, match="PT -1"):
        parse_nec5(
            HEAD + WIRE + "GE 0\nEX 0 1 3 0 1 0\n" + pt + "\nFR 0 1 0 0 14 0\nXQ\nEN\n"
        )


def test_a_geometry_card_after_ge_refuses():
    with pytest.raises(DeckError, match="arrives after GE"):
        parse_nec5(
            HEAD + WIRE + "GE 0\nGS 0 0 2\nEX 0 1 3 0 1 0\nFR 0 1 0 0 14 0\nXQ\nEN\n"
        )


def test_an_arc_below_the_ground_plane_refuses_by_name():
    text = (
        HEAD
        + "GA 1 8 2 -30 200 1e-3\nGE 1\nGN 2 0 0 0 13 0.005\nEX 0 1 3 0 1 0\nFR 0 1 0 0 14 0\nXQ\nEN\n"
    )
    assert "GA arc or GH helix that reaches below" in render(text)
