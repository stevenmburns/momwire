"""momwire#1929: the geometry refusal asks each wire once per card that built
or moved it, not every wire after every card.

`wire_geometry_refusal` used to run over the whole wire list after every
geometry card, which is O(cards x wires): 7,432,440 calls on the ELT whip's
3,856 GW cards, about 2.5 s of every parse. A card that only APPENDS wires
(`_nec2_geometry._APPENDING_CARDS`, and a replicating GM) cannot have changed
the ones before it, which passed after the card that last touched them; only
GS and an in-place GM rewrite wires already there, and they still re-check
the whole list.

The gates:

* the call count on a many-wire deck is the wire count, in each Python reader;
* every builder named as appending leaves the wires before it untouched, so
  the classification is measured rather than asserted;
* a NaN or a coordinate past 1e299 m that a LATER in-place GM or GS leaves in
  an EARLY wire still refuses, naming that GM or GS, in the Python readers
  and in both C++ readers (the copies carry the same per-card rule).
"""

from __future__ import annotations

import copy
import importlib.util
import pathlib

import pytest

from momwire.deck import DeckError, parse
from momwire.deck import _nec2_geometry as G
from momwire.deck import _nec5
from momwire.deck._cards import parse_card
from momwire.deck._nec4 import _NEC4_GEOMETRY_BUILDERS
from momwire.deck._nec5 import parse_nec5

ROOT = pathlib.Path(__file__).resolve().parent.parent

N_WIRES = 300


def _module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gws(n: int, sep: str = " ") -> str:
    """``n`` parallel half-wave dipoles 1 m apart, tags 1..n, so no two meet
    and the geometry is n wires whatever the dialect does with junctions."""
    return "".join(
        sep.join(
            ["GW", str(t), "3", f"{t}.", "0.", "-.25", f"{t}.", "0.", ".25", ".001"]
        )
        + "\n"
        for t in range(1, n + 1)
    )


def nec2_deck(geometry: str) -> str:
    return geometry + "GE 0\nFR 0 1 0 0 299.7925\nEX 0 1 2 0 1 0\nXQ\nEN\n"


def nec5_deck(geometry: str) -> str:
    return (
        "CM many wires\nCE\n"
        + geometry
        + "GE 0,-1\nFR 0,1,0,0,299.7925\nGN -1\nEX 4,1,2,0,1.414214,0.\nPQ 0\n"
        "RP 0,1,361,1000,90.,0.,0.,1.,0.\nEN\n"
    )


class _Counter:
    def __init__(self, real):
        self.real = real
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.real(*args, **kwargs)


@pytest.mark.parametrize("dialect", ["nec2", "nec4"])
def test_the_refusal_asks_each_wire_once_on_an_appending_deck(dialect, monkeypatch):
    counter = _Counter(G.wire_geometry_refusal)
    monkeypatch.setattr(G, "wire_geometry_refusal", counter)
    parse(nec2_deck(_gws(N_WIRES)), dialect=dialect)
    # One build per parse (the portal reuses it, `parse_with_structure`), one
    # ask per wire: N, where every wire after every card was N (N + 1) / 2.
    assert counter.calls == N_WIRES


def test_the_nec5_reader_asks_each_wire_once(monkeypatch):
    counter = _Counter(_nec5.wire_geometry_refusal)
    monkeypatch.setattr(_nec5, "wire_geometry_refusal", counter)
    parse_nec5(nec5_deck(_gws(N_WIRES, ",")))
    assert counter.calls == N_WIRES


@pytest.mark.parametrize("dialect", ["nec2", "nec4"])
def test_a_transform_rechecks_the_whole_structure(dialect, monkeypatch):
    """An in-place GM and a GS ask every wire again; a replicating GM asks
    only its copies. The count is the rule's, read back."""
    counter = _Counter(G.wire_geometry_refusal)
    monkeypatch.setattr(G, "wire_geometry_refusal", counter)
    n = 20
    geometry = (
        _gws(n)
        + "GS 0 0 1.5\n"  # in place: n
        + "GM 0 0 0 0 10 0 0 0 0\n"  # in place, NRPT 0: n
        + f"GM {n} 1 0 0 0 0 0 5 0\n"  # replicating: its n copies
    )
    parse(nec2_deck(geometry), dialect=dialect)
    assert counter.calls == n + n + n + n


# --- the classification is measured -----------------------------------------

_PRIOR = "GW 1 2 0.3 0.2 0.1 0.3 0.2 1.1 .001\nGW 2 2 0.5 0.6 0.1 0.9 0.6 0.4 .002\n"
_APPENDING = {
    "GW": "GW 7 3 1 1 1 2 2 2 .001",
    "GA": "GA 7 4 1. 0. 90. .001",
    "GH": "GH 7 8 .5 2. .3 .3 .3 .3 .001",
    "CW": "CW 7 6 0 0 1 4 0 1 .001 1 2 0 .5",
    "GX": "GX 10 110",
    "GR": "GR 10 4",
    "GM": "GM 10 2 0 0 30 0 0 .5 0",
}


@pytest.mark.parametrize(
    "builders,mnemonic",
    [
        (builders, m)
        for builders in ("nec2", "nec4")
        for m in sorted(G._APPENDING_CARDS | {"GM"})
        if m
        in (G._GEOMETRY_BUILDERS if builders == "nec2" else _NEC4_GEOMETRY_BUILDERS)
    ],
)
def test_an_appending_builder_leaves_the_wires_before_it_alone(builders, mnemonic):
    table = G._GEOMETRY_BUILDERS if builders == "nec2" else _NEC4_GEOMETRY_BUILDERS
    wires: list[G.FlatWire] = []
    for line in _PRIOR.splitlines():
        table["GW"](parse_card(line), wires)
    before = copy.deepcopy(wires)
    ids = [id(w) for w in wires]
    card = parse_card(_APPENDING[mnemonic])
    table[mnemonic](card, wires)
    assert G._appends_only(card)
    assert len(wires) > len(before)
    assert [id(w) for w in wires[: len(before)]] == ids
    assert wires[: len(before)] == before


def test_every_builder_is_classified_or_rechecks():
    """A builder named in no class is re-checked whole (the conservative
    default); the two that rewrite wires in place must stay out of the set."""
    for table in (G._GEOMETRY_BUILDERS, _NEC4_GEOMETRY_BUILDERS):
        assert set(table) <= G._APPENDING_CARDS | {"GM", "GS"}
    assert "GS" not in G._APPENDING_CARDS
    assert "GM" not in G._APPENDING_CARDS
    assert not G._appends_only(parse_card("GM 0 0 0 0 0 1 0 0 0"))
    assert G._appends_only(parse_card("GM 1 1 0 0 0 1 0 0 0"))


# --- a later in-place card still names itself --------------------------------

NONFINITE = (
    "with a coordinate or radius that is not a finite number; a wire's ends and "
    "radius must stay finite"
)
FAR = (
    "more than 1e+299 m from the origin; this engine meshes wires within 1e+299 m of it"
)

# (the card after N_LATE wires, the sentence). Each moves or scales an EARLY
# wire -- tag 1, or the first wire of ITS's block -- past meshing, so the
# wire refused is one an appending card's check would never have asked again.
N_LATE = 40
LATE = {
    "gs_past_the_doubles": ("GS 0 0 1e308", f"GS puts tag 1 {FAR}"),
    "gs_nan": ("GS 0 0 nan", f"GS leaves tag 1 {NONFINITE}"),
    "gm_in_place_nan": ("GM 0 0 0 0 0 nan 0 0 0", f"GM leaves tag 1 {NONFINITE}"),
    "gm_in_place_far": ("GM 0 0 0 0 0 2e299 0 0 0", f"GM puts tag 1 {FAR}"),
    "gm_in_place_from_its": (
        "GM 0 0 0 0 0 0 0 2e299 25",
        f"GM puts tag 25 {FAR}",
    ),
}


def _late(card: str, sep: str = " ") -> str:
    return _gws(N_LATE, sep) + card.replace(" ", sep) + "\n"


@pytest.mark.parametrize("name", sorted(LATE))
@pytest.mark.parametrize("dialect", ["nec2", "nec4"])
def test_a_later_transform_names_itself_in_python(name, dialect):
    card, sentence = LATE[name]
    with pytest.raises(DeckError) as err:
        parse(nec2_deck(_late(card)), dialect=dialect)
    assert str(err.value) == sentence


@pytest.mark.parametrize("name", sorted(LATE))
def test_a_later_transform_names_itself_in_the_nec5_reader(name):
    card, sentence = LATE[name]
    with pytest.raises(DeckError) as err:
        parse_nec5(nec5_deck(_late(card, ",")))
    assert str(err.value) == sentence


@pytest.fixture(scope="module")
def acc():
    from momwire import _accel

    if _accel.acc is None or not hasattr(_accel.acc, "nec4_deck_dump"):
        pytest.skip("no compiled accelerator with the C++ deck readers")
    return _accel.acc


@pytest.mark.parametrize("name", sorted(LATE))
@pytest.mark.parametrize("dialect", ["nec2", "nec4"])
def test_a_later_transform_names_itself_in_the_cxx_nec4_reader(name, dialect, acc):
    nc = _module(
        "core_split_nec4_deck_compare_1929",
        ROOT / "scripts" / "core_split" / "nec4_deck_compare.py",
    )
    card, sentence = LATE[name]
    text = nec2_deck(_late(card))
    first = acc.nec4_deck_dump(text.encode("latin-1"), dialect).split("\n")[0]
    assert first.startswith("parse ! DeckError "), first
    assert bytes.fromhex(first.split(" ")[3]).decode("utf-8") == sentence
    assert (
        nc.dc.first_difference(
            nc.py_parse(text, dialect), nc.cxx_parse(text, dialect, acc)
        )
        is None
    )


@pytest.mark.parametrize("name", sorted(LATE))
def test_a_later_transform_names_itself_in_the_cxx_nec5_reader(name, acc):
    if not hasattr(acc, "nec5_deck_dump"):
        pytest.skip("no NEC-5 deck reader in this accelerator")
    dc = _module(
        "core_split_deck_compare_1929",
        ROOT / "scripts" / "core_split" / "deck_compare.py",
    )
    card, sentence = LATE[name]
    text = nec5_deck(_late(card, ","))
    ref = dc.py_parse(text)
    head = ref.split("\n", 1)[0]
    assert head.startswith("parse ! DeckError "), head
    assert bytes.fromhex(head.split(" ", 3)[3]).decode("utf-8") == sentence
    assert dc.first_difference(ref, dc.cxx_parse(text, acc)) is None


@pytest.mark.parametrize("dialect", ["nec2", "nec4"])
def test_the_cxx_nec4_reader_reads_a_many_wire_deck_as_python(dialect, acc):
    nc = _module(
        "core_split_nec4_deck_compare_1929",
        ROOT / "scripts" / "core_split" / "nec4_deck_compare.py",
    )
    geometry = _gws(N_WIRES) + "GS 0 0 1.5\nGX 1000 100\nGM 0 0 0 0 0 0 0 1 0\n"
    text = nec2_deck(geometry)
    assert nc.classify(nc.cxx_parse(text, dialect, acc)) == "parsed"
    assert (
        nc.dc.first_difference(
            nc.py_parse(text, dialect), nc.cxx_parse(text, dialect, acc)
        )
        is None
    )
