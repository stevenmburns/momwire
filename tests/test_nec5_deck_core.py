"""The C++ NEC-5 deck reader is the Python one, field for field.

`src/momwire/_nec5_deck_core.cpp` copies momwire's NEC-5 reader
(`deck/_cards.py`, `deck/_nec5.py`) and the EZNEC seam's text-level front
(`_serve.split_runs`, `_shell._answer` up to the solve) so that a Python-free
executable reads a deck exactly as `momwire-nec5-razor-2p` does. Python stays
the reference; these tests hold the copy to it:

* every NEC deck in `tests/fixtures`, through both views (`parse_nec5`, and
  the seam's front: NX structures, XQ runs, which refusal prints);
* hand-written decks, one or more per card and per refusal -- each case
  first checks that the PYTHON reader lands where the case says (parsed, or
  the named refusal), so a case cannot silently stop exercising its branch,
  then that the C++ dumps are byte-equal;
* the latin-1 `str` rules and number spellings the reader is built on, pinned
  one code point / token at a time against Python's own.

The comparison format is `scripts/core_split/deck_compare.py`'s: one line per
field, floats as their bits, ints in decimal, strings and messages as UTF-8
hex, an exception as its type and message.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import random
import re
import struct

import pytest

from momwire.deck._cards import _to_float, parse_card

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


def _compare_module():
    spec = importlib.util.spec_from_file_location(
        "core_split_deck_compare", ROOT / "scripts" / "core_split" / "deck_compare.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dc = _compare_module()


@pytest.fixture(scope="module")
def acc():
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "nec5_deck_dump"):
        pytest.skip("no compiled accelerator with the NEC-5 deck reader")
    return _accel.acc


def _same(text: str, acc) -> None:
    for view, py, cxx in (
        ("parse", dc.py_parse, dc.cxx_parse),
        ("stream", dc.py_stream, dc.cxx_stream),
    ):
        diff = dc.first_difference(py(text), cxx(text, acc))
        assert diff is None, f"{view} view:\n{diff}"


# --- the fixture decks ------------------------------------------------------

FIXTURE_DECKS = [
    p
    for p in sorted(FIXTURES.rglob("*"))
    if p.is_file() and p.suffix in (".nec", ".deck")
]


def test_the_fixture_decks_are_there():
    # 176 .nec and 84 .deck files when this was written; a glob that matches
    # nothing would pass every parametrised case below vacuously.
    assert len(FIXTURE_DECKS) >= 250


@pytest.mark.parametrize(
    "path", FIXTURE_DECKS, ids=[str(p.relative_to(FIXTURES)) for p in FIXTURE_DECKS]
)
def test_every_fixture_deck_reads_the_same(path, acc):
    _same(path.read_bytes().decode("latin-1"), acc)


def test_the_fixtures_include_both_outcomes(acc):
    """The fixture comparison covers decks the reader accepts AND refuses,
    multi-run decks and NX chains -- not only one kind."""
    seen = set()
    for path in FIXTURE_DECKS:
        text = path.read_bytes().decode("latin-1")
        seen.add(dc.classify(text, acc)[0])
        stream = dc.py_stream(text)
        if "= multi" in stream:
            seen.add("multi")
        if "stream.structures : list" in stream:
            seen.add("chain")
    assert {"parsed", "refused", "multi", "chain"} <= seen, seen


# --- hand-written decks -----------------------------------------------------

GW1 = "GW 1,11,0.,-.25,0.,0.,.25,0.,.0005\n"
GW2 = "GW 2,5,0.,-.25,.5,0.,.25,.5,.0005\n"
REST = (
    "GE 0,-1\nFR 0,1,0,0,299.7925\nGN -1\nEX 4,1,6,0,1.414214,0.\nPQ 0\n"
    "RP 0,1,361,1000,90.,0.,0.,1.,0.\nEN\n"
)


def deck(*cards: str, wires: str = GW1 + GW2, rest: str = REST) -> str:
    """A well-formed two-wire deck with ``cards`` inserted after GE."""
    head, tail = rest.split("\n", 1)
    return (
        "CM edge case\nCE\n"
        + wires
        + head
        + "\n"
        + "".join(c + "\n" for c in cards)
        + tail
    )


DIPOLE = deck()


def runs(*cards: str) -> str:
    """The dipole with ``cards`` inserted before its EN: XQ blocks after the
    deck's own first block."""
    return DIPOLE.replace("EN\n", "".join(c + "\n" for c in cards) + "EN\n")


BIG = "1" + "0" * 400  # an integer literal that parses to inf

CASES = {
    # -- framing and the tokenizer --
    "dipole": (DIPOLE, "parsed"),
    "crlf": (DIPOLE.replace("\n", "\r\n"), "parsed"),
    "cr only": (DIPOLE.replace("\n", "\r"), "parsed"),
    "vt ff fs nel line breaks": (
        DIPOLE.replace("CE\n", "CE\x0b").replace("GE 0,-1\n", "GE 0,-1\x0c\x1c\x85"),
        "parsed",
    ),
    "unit separator is whitespace not a break": (
        DIPOLE.replace("GE 0,-1", "GE\x1f0,-1"),
        "parsed",
    ),
    "blank and indented lines, tabs, lowercase": (
        "cm x\n\n   \t\nce\n  gw 1\t11 0 -.25 0 0 .25 0 .0005\nge 0 -1\n"
        "fr 0 1 0 0 300\ngn -1\nex 4 1 6 0 1 0\nen\n",
        "parsed",
    ),
    "fused first fields": (
        "CM\nCE\nGW1,11,0.,-.25,0.,0.,.25,0.,.0005\nGE0,-1\nFR0,1,0,0,300\nGN-1\n"
        "EX4,1,6,0,1.,0.\nGN.5,0,0,0,13,.005\nGN+1\nEN\n",
        "parsed",
    ),
    "comma runs and trailing commas": (
        DIPOLE.replace("GE 0,-1", "GE 0,,-1,,, "),
        "parsed",
    ),
    "nbsp splits fields": (DIPOLE.replace("GE 0,-1", "GE\xa00\xa0-1"), "parsed"),
    "cm text kept verbatim": (
        '  CM it\'s a "dipole" \xe9\xff\\ \t x\nCM\tcm\nCE  closing text \xb5\n'
        + GW1
        + REST,
        "parsed",
    ),
    "mnemonic three letters": (deck("GWX 1"), "CARD'S MNEMONIC CODE TOO SHORT"),
    "mnemonic one letter": (deck("G 1"), "CARD'S MNEMONIC CODE TOO SHORT"),
    "mnemonic digits": (deck("12 3"), "CARD'S MNEMONIC CODE TOO SHORT"),
    "mnemonic sharp s upper-cases to three": (
        deck("\xdfE 1"),
        "CARD'S MNEMONIC CODE TOO SHORT",
    ),
    "mnemonic micro upper-cases out of latin-1": (
        deck("\xb5X 1"),
        "is not a card in this engine's nec5 dialect",
    ),
    "mnemonic y-diaeresis": (deck("\xff\xff"), "is not a card"),
    "mnemonic ordinal indicator": (deck("\xaa\xba"), "is not a card"),
    "mnemonic with quote in the line": (deck("1' x\"y"), "CARD'S MNEMONIC"),
    "only commas": (deck(", ,"), "IndexError"),
    "non-numeric field": (deck("GW 3,5,0,0,0,1,abc,0,.001"), "NON-NUMERICAL"),
    "non-numeric quote": (deck("GW 3,5,0,0,0,1,it's,0,.001"), "NON-NUMERICAL"),
    "non-numeric both quotes": (deck("GW 3,5,0,0,'\"',1,1,0,.001"), "NON-NUMERICAL"),
    "non-numeric control and soft hyphen": (
        deck("GW 3,5,0,\x01\xad\x7f\x9f,0,1,1,0,.001"),
        "NON-NUMERICAL",
    ),
    "non-numeric superscript digit": (
        deck("GW 3,\xb2,0,0,0,1,1,0,.001"),
        "NON-NUMERICAL",
    ),
    "double underscore": (deck("GW 3,1__0,0,0,0,1,1,0,.001"), "NON-NUMERICAL"),
    "hex is not a float": (deck("GW 3,0x10,0,0,0,1,1,0,.001"), "NON-NUMERICAL"),
    "python float spellings": (
        deck(
            "GW 3,1_0,1.5D-3,2d2,-0.,.5,5.,+.5e-3,1_0e1_0",
            "GW 4,1,inf,-Infinity,nan,+NaN,1e400,-1e-400,4.9e-324",
            "GW 5,2,2.4703282292062327e-324,2.4703282292062328e-324,1e-310,"
            "1.7976931348623157e308,1.7976931348623159e308,0.1,1e23",
        ),
        "parsed",
    ),
    "round half to even": (
        deck(wires=GW1 + "GW 2.5,4.5,0,0,1,0,0,2,.001\nGW -0.5,1.5,0,0,1,0,0,2,.001\n"),
        "parsed",
    ),
    "cm after ce": (DIPOLE.replace("CE\n", "CE\nCM late\n"), "CM appears after the CE"),
    "ce twice": (DIPOLE.replace("CE\n", "CE\nCE\n"), "CE appears twice"),
    "card before ce": (DIPOLE.replace("CE\n", "", 1), "arrives before any CE card"),
    "no ce at all, first card named": ("GW 1,1,0,0,0,1,0,0,.1\nEN\n", "arrives before"),
    "no en": (DIPOLE.replace("EN\n", ""), "deck has no EN card"),
    "empty text": ("", "deck has no EN card"),
    "only en": ("EN\n", "deck has no GE card"),
    "no ge": (DIPOLE.replace("GE 0,-1\n", ""), "deck has no GE card"),
    "en with a bad field": (DIPOLE.replace("EN", "EN x"), "NON-NUMERICAL"),
    "garbage after en": (DIPOLE + "QQ 1\nnot a card\n,,\n", "parsed"),
    "lowercase en": (DIPOLE.replace("EN", "en"), "parsed"),
    # -- GW --
    "gw duplicate tag": (deck(wires=GW1 + GW1), "GW declares tag 1 a second time"),
    "gw zero segments": (
        deck(wires=GW1 + "GW 2,0,0,0,0,1,0,0,.1\n"),
        "asks for 0 segments",
    ),
    "gw negative segments": (
        deck(wires=GW1 + "GW 2,-3,0,0,0,1,0,0,.1\n"),
        "asks for -3",
    ),
    "gw short": (deck(wires=GW1 + "GW 2,3,0,0,0,1,0,0\n"), "GW carries 8 fields"),
    "gw one field": (deck(wires=GW1 + "GW 2\n"), "GW carries 1 field and"),
    "gw huge tag named later": (
        deck("EX 4,1e300,1,0,1,0", wires=GW1 + "GW 1e300,3,0,0,0,1,0,0,.1\n"),
        "parsed",
    ),
    "gw huge tag in a refusal": (
        deck("EX 4,1e300,9,0,1,0", wires=GW1 + "GW 1e300,3,0,0,0,1,0,0,.1\n"),
        "addresses node 9 of tag 1000000000000000052504760255204420248704468581",
    ),
    "gw infinite tag": (deck(wires="GW inf,3,0,0,0,1,0,0,.1\n"), "OverflowError"),
    "gw nan segments": (deck(wires="GW 1,nan,0,0,0,1,0,0,.1\n"), "ValueError"),
    "gw integer literal too big for a double": (
        deck(wires=f"GW {BIG},3,0,0,0,1,0,0,.1\n"),
        "OverflowError",
    ),
    # -- GE --
    "ge bare": (DIPOLE.replace("GE 0,-1", "GE"), "GE carries 0 fields"),
    "ge one field": (DIPOLE.replace("GE 0,-1", "GE 1"), "parsed"),
    "ge three fields": (DIPOLE.replace("GE 0,-1", "GE -1,5,7"), "parsed"),
    "ge infinite second": (DIPOLE.replace("GE 0,-1", "GE 0,-inf"), "OverflowError"),
    # GE's third field (momwire#1488): any nonzero integer asks for the .GMO
    # file; it is read as an integer at the GE card, so a non-finite one
    # raises there.
    "ge third field one": (DIPOLE.replace("GE 0,-1", "GE 0,-1,1"), "parsed"),
    "ge third field two": (DIPOLE.replace("GE 0,-1", "GE 0,-1,2"), "parsed"),
    "ge third field rounds to zero": (
        DIPOLE.replace("GE 0,-1", "GE 0,-1,0.4"),
        "parsed",
    ),
    "ge third field negative": (DIPOLE.replace("GE 0,-1", "GE 0,-1,-1"), "parsed"),
    "ge infinite third": (DIPOLE.replace("GE 0,-1", "GE 0,-1,inf"), "OverflowError"),
    "ge nan third": (DIPOLE.replace("GE 0,-1", "GE 0,-1,nan"), "ValueError"),
    # -- GN --
    "gn perfect": (deck("GN 1"), "parsed"),
    "gn sommerfeld": (deck("GN 0,0,0,0,13.,.005,1.,0."), "parsed"),
    "gn 2 spelling": (deck("GN 2,0,0,0,13.,.005"), "parsed"),
    "gn nofile": (deck("GN 0,0,0,0,13.,.005,1.,0.,NOFILE"), "parsed"),
    "gn other file": (deck("GN 0,0,0,0,13.,.005,1.,0.,soil.tab"), "names a Sommerfeld"),
    "gn numeric last token is a field": (
        deck("GN 0,0,0,0,13.,.005,1.,0.,1e5"),
        "parsed",
    ),
    "gn file on perfect": (deck("GN 1 NOFILE"), "GN 1 carries a trailing token"),
    "gn file on free space": (deck("GN -1 it's"), "GN -1 carries a trailing token"),
    "gn name mid card": (deck("GN 0,0,0,0,X,.005,1.,0."), "NON-NUMERICAL"),
    "gn finite short": (deck("GN 0,0,0,0,13."), "parsed"),
    "gn finite no permittivity": (
        deck("GN 0,0,0,0"),
        "needs its relative permittivity",
    ),
    "gn bare": (deck("GN"), "GN 0 carries 0 fields"),
    "gn bare file": (deck("GN NOFILE"), "GN 0 carries 0 fields"),
    "gn type 3": (deck("GN 3"), "GN type 3 is not part"),
    "gn magnetic": (deck("GN 0,0,0,0,13.,.005,2.,0."), "relative permeability 2+0j"),
    "gn magnetic imaginary": (deck("GN 0,0,0,0,13.,.005,1.,-.5"), "1-0.5j"),
    "gn mu zero is one": (deck("GN 0,0,0,0,13.,.005,-0.,0."), "parsed"),
    "gn mu signed zero imaginary": (deck("GN 0,0,0,0,13.,.005,1.,-0."), "parsed"),
    "gn mu nan": (deck("GN 0,0,0,0,13.,.005,nan,-nan"), "nan+nanj"),
    "gn mu inf": (deck("GN 2,0,0,0,13.,.005,-inf,inf"), "-inf+infj"),
    "gn mu tiny": (
        deck("GN 0,0,0,0,13.,.005,1.0000001,1e-300"),
        "relative permeability 1+1e-300j",
    ),
    "gn infinite code": (deck("GN inf"), "OverflowError"),
    # -- GD and the last ground card --
    "gd": (deck("GD 0,0,0,0,13.,.005,1.,0."), "parsed"),
    "gd odd flag": (deck("GD 7,8,9,9,13.,.005"), "parsed"),
    "gd cancel skips mu": (deck("GD -1,0,0,0,13.,.005,9.,9."), "parsed"),
    "gd negative sigma": (deck("GD 0,0,0,0,13.,-.005"), "parsed"),
    "gd magnetic": (deck("GD 0,0,0,0,13.,.005,1.,1."), "GD asks for a ground"),
    "gd short": (deck("GD 0,0,0,0,13."), "parsed"),
    "last ground wins": (
        deck("GN 1", "GD 0,0,0,0,13.,.005", "GN 0,0,0,0,5.,.001"),
        "parsed",
    ),
    # -- EX --
    "ex voltage": (deck("EX 0,2,3,0,1.,0."), "parsed"),
    "ex zero volts is one": (deck("EX 0,2,3,0,0.,-0."), "parsed"),
    "ex zero current kept": (deck("EX 4,2,3,0,0.,0."), "parsed"),
    "ex five fields": (deck("EX 0 2 3 0 1.000000e+00"), "parsed"),
    "ex short": (deck("EX 0 2 3 0"), "parsed"),
    "ex type 1": (deck("EX 1,2,3,0,1.,0."), "EX type 1 is not part"),
    "ex type 5 reads end code never": (deck("EX 5,9,99,7,1.,0."), "EX type 5"),
    "ex node -1": (deck("EX 4,2,-1,0,1.,0."), "parsed"),
    "ex node literal 0": (
        deck("EX 4,2,0,0,1.,0."),
        "addresses node 0 of tag 2 as a literal",
    ),
    "ex node -2": (deck("EX 4,2,-2,0,1.,0."), "addresses node -2 of tag 2"),
    "ex node past the end": (
        deck("EX 4,2,6,0,1.,0."),
        "addresses node 6 of tag 2, whose 5",
    ),
    "ex last node": (deck("EX 4,2,5,0,1.,0."), "parsed"),
    "ex unknown tag": (deck("EX 4,9,1,0,1.,0."), "EX names tag 9, which no GW"),
    "ex end code 1": (deck("EX 4,2,1,1,1.,0."), "parsed"),
    "ex end code 2": (deck("EX 4,2,5,2,1.,0."), "parsed"),
    "ex end code 3": (
        deck("EX 4,2,1,3,1.,0."),
        "carries 3 in its EX's own end-code field",
    ),
    "ex end code segment 0": (
        deck("EX 4,2,0,1,1.,0."),
        "names segment 0 with an explicit",
    ),
    "ex end code negative segment": (
        deck("EX 4,2,-1,2,1.,0."),
        "names segment -1 with",
    ),
    "ex end code segment past": (
        deck("EX 4,2,6,1,1.,0."),
        "addresses segment 6 of tag 2",
    ),
    "ex end code unknown tag": (deck("EX 4,7,1,1,1.,0."), "EX names tag 7"),
    "ex end code rounds": (deck("EX 4,2,2.5,1.5,1.,0."), "parsed"),
    "ex end code infinite": (deck("EX 4,2,1,inf,1.,0."), "OverflowError"),
    "ex end code on a huge wire": (
        deck("EX 4,3,1e300,1,1.,0.", wires=GW1 + "GW 3,1e300,0,0,0,1,0,0,.1\n"),
        "parsed",
    ),
    "ex multiple": (deck("EX 4,2,1,1,1.,0.", "EX 0,2,3,0,.5,.25"), "parsed"),
    # -- LD 0 / 1 / 4 --
    "ld 4": (deck("LD 4,2,3,0,50.,-25."), "parsed"),
    "ld 0 series": (deck("LD 0,2,3,0,50.,1e-6,1e-12"), "parsed"),
    "ld 1 parallel end code": (deck("LD 1,2,3,2,50.,1e-6,1e-12"), "parsed"),
    "ld 4 no-op": (deck("LD 4,2,3,0,0.,-0."), "parsed"),
    "ld 0 no-op still addressed": (
        deck("LD 0,2,9,0,0.,0.,0."),
        "addresses node 9 of tag 2",
    ),
    "ld 4 short": (deck("LD 4,2,3,0,50."), "parsed"),
    "ld 0 short": (deck("LD 0,2,3,0,50.,1e-6"), "parsed"),
    "ld 6": (deck("LD 6,2,3,0,50.,1."), "LD type 6 is not part"),
    "ld ldtagt 7": (deck("LD 4,2,3,7,50.,1."), "in its LDTAGT field"),
    "ld ldtagt 1 segment 0": (deck("LD 4,2,0,1,50.,1."), "once LDTAGT selects the end"),
    "ld bare": (deck("LD"), "as a literal 0"),
    # -- LD 5 --
    "ld 5 whole structure wildcard": (deck("LD 5,0,0,0,5.8e7"), "parsed"),
    "ld 5 whole structure explicit": (deck("LD 5,0,1,16,5.8e7,0."), "parsed"),
    "ld 5 run of whole wires": (deck("LD 5,0,1,11,5.8e7,1."), "parsed"),
    "ld 5 second wire alone": (deck("LD 5,0,12,16,5.8e7"), "parsed"),
    "ld 5 split wire": (deck("LD 5,0,2,16,5.8e7"), "does not start and end on wire"),
    "ld 5 backwards": (deck("LD 5,0,12,11,5.8e7"), "segments 12 to 11"),
    "ld 5 tag wildcard": (deck("LD 5,2,0,0,5.8e7"), "parsed"),
    "ld 5 tag explicit": (deck("LD 5,2,1,5,5.8e7"), "parsed"),
    "ld 5 tag partial": (deck("LD 5,2,1,4,5.8e7"), "covers whole wires only"),
    "ld 5 unknown tag": (deck("LD 5,8,0,0,5.8e7"), "LD 5 names tag 8"),
    "ld 5 zero sigma": (deck("LD 5,0,0,0,0."), "conductivity of 0 S/m"),
    "ld 5 negative sigma": (deck("LD 5,0,0,0,-1.5e-7"), "conductivity of -1.5e-07"),
    "ld 5 nan sigma": (deck("LD 5,0,0,0,nan"), "parsed"),
    "ld 5 short": (deck("LD 5,0,0,0"), "must be positive"),
    "ld 5 mu 2": (deck("LD 5,0,0,0,5.8e7,2."), "relative permeability of 2;"),
    "ld 5 mu nan": (deck("LD 5,0,0,0,5.8e7,nan"), "relative permeability of nan"),
    "ld 5 last wins and fans out": (
        deck("LD 5,0,0,0,5.8e7", "LD 5,2,0,0,3.5e7", "LD 5,0,1,11,1e6"),
        "parsed",
    ),
    "ld 5 on huge wires": (
        deck("LD 5,0,1,2,1e6", wires=GW1 + "GW 3,1e20,0,0,0,1,0,0,.1\n"),
        "this deck's wires span 1-11, 12-100000000000000000011",
    ),
    "ld 5 tag 0 over huge wires": (
        deck(
            "LD 5,0,1,1e20,1e6",
            "LD 2,0,1,2e20,0.,1e-7,0.",
            wires="GW 1,1e20,0,0,0,1,0,0,.1\nGW 3,1e20,0,0,0,1,0,0,.1\n",
        ),
        "parsed",
    ),
    # -- LD 2 / 3 --
    "ld 2": (deck("LD 2,0,0,0,0.,1e-7,0."), "parsed"),
    "ld 3": (deck("LD 3,2,1,5,.5,1e-7,0."), "parsed"),
    "ld 2 capacitance": (deck("LD 2,0,0,0,0.,1e-7,1e-12"), "capacitance of 1e-12"),
    "ld 3 negative r": (
        deck("LD 3,0,0,0,-1.,1e-7,0."),
        "per-unit-length resistance (-1, 1e-07)",
    ),
    "ld 2 negative l": (deck("LD 2,0,0,0,1.,-1e-7,0."), "inductance (1, -1e-07)"),
    "ld 2 no-op still addressed": (deck("LD 2,9,0,0,0.,0.,0."), "LD 2 names tag 9"),
    "ld 2 no-op": (deck("LD 2,2,0,0,0.,0.,0."), "parsed"),
    "ld 2 short": (deck("LD 2,0,0,0,0.,1e-7"), "parsed"),
    "ld 2 partial": (deck("LD 2,1,2,3,0.,1e-7,0."), "per-unit-length RLC covers whole"),
    "ld mixed order": (
        deck(
            "LD 4,2,3,0,50.,-25.",
            "LD 5,0,0,0,5.8e7",
            "LD 2,1,0,0,0.,1e-7,0.",
            "LD 0,1,1,1,0.,1e-6,0.",
            "LD 3,0,0,0,1.,0.,0.",
        ),
        "parsed",
    ),
    # -- TL / NT --
    "tl": (deck("TL 1,6,2,-1,50.,.1,0.,0.,0.,0."), "parsed"),
    "tl crossed": (deck("TL 1,6,2,3,-490.0875,1.5,1e10,0.,0.,1e10"), "parsed"),
    "tl zero z0": (
        deck("TL 1,6,2,3,0.,1.,0.,0.,0.,0."),
        "TL with a zero characteristic",
    ),
    "tl short": (deck("TL 1,6,2,3,50.,1."), "parsed"),
    "tl bad second address": (
        deck("TL 1,6,2,0,50.,1.,0.,0.,0.,0."),
        "TL addresses node 0",
    ),
    "nt": (deck("NT 2,-1,1,6,.01,0.,0.,-.005,.01,0."), "parsed"),
    "nt short": (deck("NT 2,-1,1,6,.01"), "parsed"),
    "nt unknown tag": (deck("NT 3,-1,1,6,0,0,0,0,0,0"), "NT names tag 3"),
    # -- FR / RP / NE / NH / PQ / XQ --
    "fr zero points": (deck("FR 0,0,0,0,7.1"), "parsed"),
    # a stepping FR (momwire#1508): the list, accumulated; NEC's two types only
    "fr two points": (deck("FR 0,2,0,0,7.1,.1"), "parsed"),
    "fr multiplicative": (deck("FR 1,4,0,0,7.1,1.1"), "parsed"),
    "fr stepping type 2": (deck("FR 2,3,0,0,7.1,.1"), "FR stepping type 2"),
    "fr short": (deck("FR 0,1,0,0"), "parsed"),
    "rp 3-d": (deck("RP 0,37,73,1001,0.,0.,5.,5.,1000."), "parsed"),
    "rp counts floor at one": (deck("RP 0,0,-4,1000,0.,0.,5.,5."), "parsed"),
    "rp mode 1": (deck("RP 1,1,361,1000,90.,0.,0.,1."), "RP mode 1 is not part"),
    # momwire#1503: 1000-1003 and 1010-1013 served, the rest refused by digit
    "rp xnda 1003": (deck("RP 0,1,361,1003,90.,0.,0.,1."), "parsed"),
    "rp xnda 1012": (deck("RP 0,1,361,1012,90.,0.,0.,1."), "parsed"),
    "rp xnda": (deck("RP 0,1,361,1100,90.,0.,0.,1."), "RP 0 with XNDA 1100 asks for"),
    "rp xnda x0": (
        deck("RP 0,1,361,1,90.,0.,0.,1."),
        "RP 0 with XNDA 1 asks for major",
    ),
    "rp xnda d2 a5": (deck("RP 0,1,361,3025,90.,0.,0.,1."), "output format X = 3"),
    "rp xnda a4": (deck("RP 0,1,361,1004,90.,0.,0.,1."), "averaging A = 4"),
    "rp xnda past": (deck("RP 0,1,361,12000,90.,0.,0.,1."), "is not an XNDA"),
    "rp xnda negative": (deck("RP 0,1,361,-1,90.,0.,0.,1."), "is not an XNDA"),
    "rp short": (deck("RP 0,1,361,1000,90.,0.,0."), "parsed"),
    # Dan AC6LA's seven-field RP (0.79.2): the NEC-5 engine reads the rest as 0
    "rp seven fields": (deck("RP 0 91 1 1000 -90 0 2"), "parsed"),
    "rp infinite count": (deck("RP 0,inf,nan,1000,90.,0.,0.,1."), "OverflowError"),
    "ne": (deck("NE 0,1,1,1,0.,0.,0.,0.,0.,0."), "parsed"),
    "ne spherical counts floor": (deck("NE 1,0,5,-2,10.,30.,60.,0.,5.,5."), "parsed"),
    "nh": (deck("NH 0,5,5,1,-1.,-1.,1.,.5,.5,0."), "parsed"),
    "ne coordinates 2": (
        deck("NE 2,1,1,1,0.,0.,0.,0.,0.,0."),
        "NE coordinate system 2",
    ),
    "nh coordinates -1": (
        deck("NH -1,1,1,1,0.,0.,0.,0.,0.,0."),
        "NH coordinate system -1",
    ),
    "ne short": (deck("NE 0,1,1,1,0.,0.,0.,0.,0."), "parsed"),
    "pq bare": (deck("PQ"), "parsed"),
    "pq -1": (deck("PQ -1"), "parsed"),
    "xq bare": (deck("XQ"), "parsed"),
    "xq 1": (deck("XQ 1"), "parsed"),
    # -- SP, refused by name, outside the vocabulary --
    "sp sphere": (deck("SP 0 1 1 0 0 0 0 1 0 90 0 360"), "SP (sphere)"),
    "sp manual patch": (deck("SP 0 0 .1 .05 .05 0. 0."), "surface-patch form"),
    "sp integral reals": (
        deck("SP 0 1 1.0 0 0 0 0 1 0 90 0 360"),
        "surface-patch form",
    ),
    "sp zero radius": (deck("SP 0 1 1 0 0 0 0 0 0 90 0 360"), "surface-patch form"),
    "sp zero count": (deck("SP 0 0 1 0 0 0 0 1"), "surface-patch form"),
    "sp signs": (deck("SP +-0 +1 -+1 -0 0 0 0 1"), "NON-NUMERICAL"),
    "sp signed literals": (deck("SP -0 +1 1 +0 0 0 0 1"), "SP (sphere)"),
    "sp underscore literal": (deck("SP 0 1_0 1 0 0 0 0 1"), "surface-patch form"),
    "sp nan radius": (deck("SP 0 1 1 0 0 0 0 nan"), "SP (sphere)"),
    "sp fused": (deck("SP0 1 1 0 0 0 0 1"), "SP (sphere)"),
    "sp huge count": (deck(f"SP 0 {BIG} 1 0 0 0 0 1"), "OverflowError"),
    "cw": (deck("CW 1 2 3"), "CW is not a card"),
    "qq lowercase": (deck("qq"), "QQ is not a card"),
    "nx in one structure": (deck("NX"), "NX (next structure) ends one structure"),
    **{
        f"refused by name {m}": (deck(f"{m} 1"), f"{m} ")
        for m in ("GM GX GR GS GA GH GC GF SM SC CP PL WG ZO KH EK IS PT MP SY".split())
    },
    "refused by name before vocabulary minimum": (deck("PT"), "PT (element-current"),
    # -- multi-run decks (XQ blocks) --
    "two runs": (
        runs("XQ 0", "EX 4,2,3,0,1.,0.", "RP 0,1,361,1000,90.,0.,0.,1.", "XQ 0"),
        "parsed",
    ),
    "three runs lowercase": (
        runs(
            "xq",
            "ex 4,2,3,0,1.,0.",
            "pq 0",
            "xq",
            "ne 0,1,1,1,0,0,0,0,0,0",
            "nh 0,1,1,1,0,0,0,0,0,0",
        ),
        "parsed",
    ),
    "a later run changes the frequency": (
        runs("XQ 0", "FR 0,1,0,0,7.", "XQ 0"),
        "parsed",  # the parse accepts it; the seam's split refuses (stream view)
    ),
    "a later run changes a card the upper-case of which leaves latin-1": (
        "CM\nCE\n" + GW1 + "GE 0\nXQ\n\xb5x 1\nXQ\nEN\n",
        "is not a card",
    ),
    "a later run with a sharp s": (
        "CM\nCE\n" + GW1 + "GE 0\nXQ\n\xdfa\nXQ\nEN\n",
        "CARD'S",
    ),
    "a run's own card refuses": (
        runs("XQ 0", "EX 4,9,3,0,1.,0.", "XQ 0"),
        "EX names tag 9",
    ),
    "a run's own card overflows": (
        DIPOLE.replace("EN\n", "XQ\nEX 4,2,3,0,1.,0.\nXQ\nEX 4,inf,3,0,1,0\nXQ\nEN\n"),
        "OverflowError",
    ),
    "runs after a refused deck": (
        runs("XQ 0", "FR 0,1,0,0,7.", "XQ 0", "GN 3"),
        "GN type 3",
    ),
    "runs with no ge line": (
        "CM\nCE\n" + GW1 + "XQ\nXQ\nEN\n",
        "deck has no GE card",
    ),
    "runs without a closing xq": (runs("XQ 0", "EX 4,2,3,0,1.,0."), "parsed"),
    "the shared prefix refuses": (
        "CM\nCE\nGW 1,0,0,0,0,1,0,0,.1\nGE 0\nXQ\nEX 4,1,1,0,1,0\nXQ\nEN\n",
        "asks for 0 segments",
    ),
    "the shared prefix overflows": (
        "CM\nCE\nGW 1,inf,0,0,0,1,0,0,.1\nGE 0\nXQ\nEX 4,1,1,0,1,0\nXQ\nEN\n",
        "OverflowError",
    ),
    # -- NX chains --
    "two structures": (DIPOLE.replace("EN\n", "NX\n") + DIPOLE, "NX (next structure)"),
    "three structures, lowercase, indented, fields": (
        DIPOLE.replace("EN\n", "  nx 1 2\n") + DIPOLE.replace("EN\n", "NX,\n") + DIPOLE,
        "NX (next structure)",
    ),
    "truncated after nx": (DIPOLE.replace("EN\n", "NX\n"), "NX (next structure)"),
    "truncated chain body": (
        DIPOLE.replace("EN\n", "NX\n") + "CM\nCE\n" + GW1,
        "NX (next structure)",
    ),
    "second structure refuses": (
        DIPOLE.replace("EN\n", "NX\n") + deck("GN 3") + DIPOLE,
        "NX (next structure)",
    ),
    "second structure overflows": (
        DIPOLE.replace("EN\n", "NX\n") + deck("PQ inf"),
        "NX (next structure)",
    ),
    "nx sniffed from a bad mnemonic": (
        DIPOLE.replace("EN\n", "NXQ 1\n") + DIPOLE,
        "CARD'S MNEMONIC",
    ),
    "en sniffed": (DIPOLE.replace("EN\n", "ENd\n") + "garbage\n", "CARD'S MNEMONIC"),
    "chained multi-run": (
        runs("XQ", "EX 4,2,3,0,1,0", "XQ").replace("EN\n", "NX\n") + DIPOLE,
        "NX (next structure)",
    ),
    "a chain whose first structure is empty": (
        "NX\n" + DIPOLE,
        "NX arrives before any CE",
    ),
}


def _python_outcome(text: str) -> str:
    """Python's own parse, as the expectation strings name it."""
    head = dc.py_parse(text).split("\n", 1)[0]
    if not head.startswith("parse !"):
        return "parsed"
    _, _, kind, message = head.split(" ", 3)
    return f"{kind} {bytes.fromhex(message).decode('utf-8')}"


@pytest.mark.parametrize("name", list(CASES))
def test_a_hand_written_deck_reads_the_same(name, acc):
    text, expect = CASES[name]
    got = _python_outcome(text)
    if expect == "parsed":
        assert got == "parsed", got
    else:
        assert got != "parsed" and expect in got, got
    _same(text, acc)


def test_the_stream_cases_reach_every_answer_kind():
    """The hand-written decks drive the seam's front to each of its outcomes:
    a single deck, a multi-run deck, a parse refusal, a split refusal
    (ServeRefusal's message), a raised exception, and NX chains that stop at a
    refusal and at an exception."""
    answer = re.compile(r"^stream\.answers\[\d+\] ([=!]) (\S+)(?: (\S+))?")
    kinds = set()
    for text, _ in CASES.values():
        stream = dc.py_stream(text)
        chained = "stream.structures : list" in stream
        for line in stream.splitlines():
            m = answer.match(line)
            if not m:
                continue
            sign, word, rest = m.groups()
            kind = "raised" if sign == "!" else word
            kinds.add(kind)
            if kind == "refused" and "XQ block 2 carries FR" in bytes.fromhex(
                rest
            ).decode("utf-8"):
                kinds.add("split refusal")
            if chained and kind in ("refused", "raised"):
                kinds.add(f"chain {kind}")
    assert {
        "single",
        "multi",
        "refused",
        "raised",
        "split refusal",
        "chain refused",
        "chain raised",
    } <= kinds, kinds


# --- the str rules and number spellings, one at a time ----------------------


def _f(x):
    return struct.pack(">d", x).hex()


def _s(x):
    return x.encode("utf-8").hex()


def py_probe(op: str, s: str) -> str:
    out = []
    if op == "strip":
        out.append(f"probe = str {_s(s.strip())}")
    elif op in ("split", "splitlines"):
        parts = s.split() if op == "split" else s.splitlines()
        out.append(f"probe : list {len(parts)}")
        out += [f"probe[{k}] = str {_s(p)}" for k, p in enumerate(parts)]
    elif op == "upper":
        out.append(f"probe = str {_s(s.upper())}")
        out.append(f"probe.len = int {len(s.upper())}")
    elif op in ("isalpha", "isdigit", "isspace"):
        out.append(f"probe = bool {getattr(s, op)()}")
    elif op in ("float", "round", "g", "+g"):
        v = _to_float(s)
        if v is None:
            out.append("probe = None")
        elif op == "float":
            out.append(f"probe = float {_f(v)}")
        elif op == "round":
            if math.isfinite(v):
                out.append(f"probe = int {int(round(v))}")
            else:
                out.append("probe = nonfinite")
        else:
            out.append(f"probe = str {_s(format(v, op))}")
    elif op == "card":
        try:
            card = parse_card(s)
        except Exception as exc:  # noqa: BLE001 — the outcome IS the exception
            out.append(f"probe ! {type(exc).__name__} {_s(str(exc))}")
        else:
            if card is None:
                out.append("probe = None")
            else:
                out.append(f"probe.mnemonic = str {_s(card.mnemonic)}")
                out.append(f"probe.values : tuple {len(card.values)}")
                out += [
                    f"probe.values[{k}] = float {_f(v)}"
                    for k, v in enumerate(card.values)
                ]
                out.append(f"probe.raw = str {_s(card.raw)}")
                out.append(
                    "probe.trailer = "
                    + ("None" if card.trailer is None else f"str {_s(card.trailer)}")
                )
    return "".join(line + "\n" for line in out)


def _check(acc, op, strings):
    bad = []
    for s in strings:
        want = py_probe(op, s)
        got = acc.nec5_text_probe(op, s.encode("latin-1"))
        if want != got:
            bad.append((s, want, got))
    assert not bad, f"{len(bad)} of {len(strings)} differ; first: {bad[0]!r}"


LATIN1 = [chr(c) for c in range(256)]


@pytest.mark.parametrize(
    "op",
    ["strip", "split", "splitlines", "upper", "isalpha", "isdigit", "isspace"],
)
def test_every_latin1_code_point_follows_python(acc, op):
    singles = LATIN1
    pairs = [a + b for a in LATIN1 for b in ("a", "\n", " ", "'", '"', "\r")]
    rng = random.Random(1234)
    noisy = [
        "".join(rng.choice(LATIN1) for _ in range(rng.randrange(0, 12)))
        for _ in range(3000)
    ]
    _check(acc, op, singles + pairs + noisy + ["", "\r\n", "\r\r\n\n", "'\"", "a\x1fb"])


NUMBER_TOKENS = [
    "0",
    "-0",
    "+0",
    "0.",
    ".0",
    "-.0",
    "1",
    "1.",
    ".5",
    "5.5",
    "007",
    "1e5",
    "1E5",
    "1e+5",
    "1e-5",
    "1D5",
    "1d-5",
    "1.5D+03",
    "-2.5d2",
    "1_000",
    "1_0.0_1e1_0",
    "1__0",
    "_1",
    "1_",
    "1_.5",
    "1._5",
    "1e_5",
    "1e5_",
    "inf",
    "-inf",
    "+inf",
    "INF",
    "Infinity",
    "-infinity",
    "infinit",
    "nan",
    "-nan",
    "+NAN",
    "nAn",
    "nan(1)",
    "e5",
    "1e",
    ".",
    "-",
    "+",
    "",
    "--1",
    "+-1",
    "0x10",
    "1e400",
    "-1e400",
    "1e-400",
    "-1e-400",
    "4.9e-324",
    "5e-324",
    "2.4703282292062327e-324",
    "2.4703282292062328e-324",
    "2.2250738585072011e-308",
    "2.2250738585072014e-308",
    "1.7976931348623157e308",
    "1.7976931348623158e308",
    "1.7976931348623159e308",
    "9007199254740993",
    "9007199254740992.5",
    "0.1",
    "0.3",
    "1e23",
    "8.5",
    "9.5",
    "-0.5",
    "0.5",
    "1.5",
    "2.5",
    "-2.5",
    "4503599627370495.5",
    "1e300",
    "-1e300",
    "123456789012345678901234567890",
    "1" + "0" * 400,
    "0." + "0" * 400 + "1",
    "1" * 800 + "e-800",
    "0.000001",
    "1e-05",
    "123456",
    "1234567",
    "99999.95",
    "999999.5",
    "100000",
    "1e16",
    "1.5e-07",
    "\xb2",
    "1\xb2",
    "1,5",
    "1 5",
]


@pytest.mark.parametrize("op", ["float", "round", "g", "+g"])
def test_every_number_spelling_follows_python(acc, op):
    rng = random.Random(99)
    tokens = list(NUMBER_TOKENS)
    for _ in range(4000):
        bits = rng.getrandbits(64)
        x = struct.unpack(">d", bits.to_bytes(8, "big"))[0]
        tokens.append(repr(x))
        tokens.append(f"{x:.17g}".replace("e", rng.choice("eEdD")))
        tokens.append(f"{rng.uniform(-1e6, 1e6):.{rng.randrange(0, 12)}f}")
        tokens.append(
            str(rng.randrange(-(10**25), 10**25)) + rng.choice(["", ".5", ".", ".50"])
        )
    alphabet = "0123456789._eEdD+-ni"
    tokens += [
        "".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 9)))
        for _ in range(4000)
    ]
    _check(acc, op, tokens)


def test_card_lines_follow_python(acc):
    rng = random.Random(7)
    pieces = [
        "GW",
        "gn",
        "EX",
        "CM",
        "ce",
        "Gw1",
        "SP",
        "\xdfE",
        "\xb5x",
        "1",
        ",",
        " ",
        "\t",
        "1.5",
        "-3",
        "D",
        "inf",
        "NOFILE",
        "x'y",
        "\xa0",
        "\x85",
        "GE0",
        "1_0",
    ]
    lines = [
        "".join(rng.choice(pieces) for _ in range(rng.randrange(0, 8)))
        for _ in range(5000)
    ]
    _check(
        acc,
        "card",
        lines + ["", " ", ",", "  ,  ,", "CM", "CMx", "GN", "GN x y", "GN 1 x"],
    )
