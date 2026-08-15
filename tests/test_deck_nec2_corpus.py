"""Geometry equivalence: ``momwire.deck``'s nec2 front-end vs antennaknobs.

The nec2 dialect spec (``site/src/content/docs/reference/deck-grammar-nec2.md``,
appendix ``#appendix-differences-from-antennaknobs-importer``) says the two
readers "agree on everything load-bearing for geometry: ``GW``, ``GM``,
``GS``, free-format tolerance, fused mnemonics, ``(tag, segment)`` addressing,
and the connection tolerance and snapping — that code is shared, not merely
equivalent."  This module is the measurement behind that sentence, run over
the 44-deck reference corpus.

It compares, per deck and BITWISE where the arithmetic is the same expression:

  * the flat NEC wire list after every ``GM``/``GS`` transform and both
    connection passes — tag, segment count, both endpoints, radius;
  * the junction cut set, against antennaknobs' own ``_junction_cuts``;
  * the shattered pieces, against the spans antennaknobs' ``wire_tuples()``
    emits for the same wires;
  * ``(tag, segment)`` resolution, against ``nec_import._locate_segment``; and
  * the resolved ``(wire, arclength)`` of every ``EX``/``LD``/``PT`` address
    in the deck, as a POINT IN SPACE — checked against both the flat wire's
    own segment centre and the arclength antennaknobs' polyline translation
    lands the same feed on.  The last comparison is a tolerance rather than a
    bitwise one: the two paths sum a different partition of the same
    polyline, so their arclengths differ by float summation order alone.

antennaknobs is not a momwire dependency and will never be one, so the whole
module skips when it is not importable — which is every CI run.  To run it,
build one venv with momwire editable FIRST and antennaknobs editable
``--no-deps`` second (antennaknobs pins an exact momwire, which must not
displace the checkout under test).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from momwire.deck._cards import parse_card
from momwire.deck._nec2 import _GEOMETRY_CARDS
from momwire.deck._nec2_geometry import build_geometry

nec_import = pytest.importorskip(
    "antennaknobs.nec_import", reason="the equivalence reference is not installed"
)
ak_geometry = pytest.importorskip("antennaknobs.geometry")
np = pytest.importorskip("numpy")


def _corpus() -> list[Path]:
    """The 44 reference decks, found next to the installed antennaknobs."""
    import antennaknobs

    root = Path(antennaknobs.__file__).resolve().parents[2]
    return sorted((root / "tests" / "fixtures" / "nec_portal").glob("*.deck"))


CORPUS = _corpus()

pytestmark = pytest.mark.skipif(
    not CORPUS, reason="the nec_portal deck corpus is not on disk"
)


class _Where:
    """``nec_import``'s card protocol, reduced to what ``_locate_segment``
    uses: a way to raise with context."""

    def error(self, msg):  # pragma: no cover - only on a mismatch
        return ValueError(msg)


def _cards(text: str):
    """Every card of a deck body, stopping at its terminator."""
    out = []
    for line in text.splitlines():
        card = parse_card(line)
        if card is None:
            continue
        if card.mnemonic in ("NX", "EN"):
            break
        out.append(card)
    return out


def _addresses(cards) -> list[tuple[int, int]]:
    """Every ``(tag, segment)`` the deck addresses, in discovery order.

    ``EX`` names one segment, ``LD``/``IS`` a range, ``PT 0`` a range when it
    carries one.  Duplicates are dropped: antennaknobs refuses a segment
    driven twice, and the union set is what the portal builds anyway.
    """
    seen: list[tuple[int, int]] = []

    def add(tag, seg):
        if (tag, seg) not in seen:
            seen.append((tag, seg))

    for card in cards:
        if card.mnemonic == "EX" and card.i(0) == 0:
            add(card.i(1), max(card.i(2), 1))
        elif card.mnemonic in ("LD", "IS"):
            if card.mnemonic == "LD" and card.i(0) in (-1, 5):
                continue
            first, last = card.i(2), card.i(3)
            if first <= 0 and last <= 0:
                continue
            for seg in range(max(first, 1), max(last, first) + 1):
                add(card.i(1), seg)
        elif card.mnemonic == "PT" and card.i(0) == 0 and (card.i(2) or card.i(3)):
            for seg in range(max(card.i(2), 1), max(card.i(3), card.i(2)) + 1):
                add(card.i(1), seg)
    return seen


def _reference(cards, addresses):
    """antennaknobs' reading of the same geometry.

    A synthesized deck of the geometry cards plus one ``EX`` per address —
    the portal's own ``_synthesize_union_deck`` trick, which reuses that
    repo's only card parser and gives one geometry translation for the whole
    deck.  Each ``EX`` carries its address's index as its voltage, so the
    feeds that come back out of the polyline translation can be matched to
    the address that asked for them.
    """
    lines = [c.raw for c in cards if c.mnemonic in _GEOMETRY_CARDS]
    lines += [f"EX 0 {tag} {seg} 0 {k + 1}." for k, (tag, seg) in enumerate(addresses)]
    return nec_import.parse_nec("\n".join(lines) + "\n", name="equivalence deck")


def _point_on_polyline(polyline, arclength: float):
    """The point ``arclength`` metres along an (M, 3) polyline."""
    remaining = arclength
    for a, b in zip(polyline[:-1], polyline[1:]):
        seg = float(np.linalg.norm(b - a))
        if remaining <= seg:
            return a + (b - a) * (remaining / seg)
        remaining -= seg
    return polyline[-1]


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.stem)
def test_geometry_matches_antennaknobs(path: Path):
    text = path.read_text()
    cards = _cards(text)
    geometry_cards = [c for c in cards if c.mnemonic in _GEOMETRY_CARDS]
    addresses = _addresses(cards)
    assert addresses, "every corpus deck addresses at least one segment"

    ours = build_geometry(geometry_cards)
    theirs = _reference(cards, addresses)

    # 1. the flat wire list, bitwise.
    assert len(ours.wires) == len(theirs.wires)
    for mine, ref in zip(ours.wires, theirs.wires):
        assert mine.tag == ref.tag
        assert mine.n_seg == ref.n_seg
        assert tuple(mine.p1) == tuple(ref.p1)
        assert tuple(mine.p2) == tuple(ref.p2)
        assert mine.radius == ref.radius

    # 2. the junction cut set, against antennaknobs' own.
    assert ours.cuts == dict(theirs._junction_cuts)

    # 3. the shattered pieces, against the spans wire_tuples() emits. Their
    #    tuples carry the extra splits an EX segment forces (the engine there
    #    feeds a wire at its middle segment); ours carry only the junction
    #    cuts, so each of our pieces must be an exact consecutive RUN of
    #    theirs — same endpoints, same segment total.
    ref_tuples = theirs.wire_tuples()
    cursor = 0
    for wire_index, ref_wire in enumerate(theirs.wires):
        run: list = []
        total = 0
        while total < ref_wire.n_seg:
            run.append(ref_tuples[cursor])
            total += ref_tuples[cursor][2]
            cursor += 1
        assert total == ref_wire.n_seg
        pos = 0
        for piece_index in ours.pieces_of_wire[wire_index]:
            piece = ours.pieces[piece_index]
            covered = 0
            start = pos
            while covered < piece.n_seg:
                covered += run[pos][2]
                pos += 1
            assert covered == piece.n_seg
            assert tuple(run[start][0]) == piece.p1
            assert tuple(run[pos - 1][1]) == piece.p2
        assert pos == len(run)
    assert cursor == len(ref_tuples)

    # 4/5. addressing, and the point every address lands on.
    flat = [[w.tag, w.n_seg, list(w.p1), list(w.p2), w.radius] for w in theirs.wires]
    translated = ak_geometry.flat_wires_to_polylines(ref_tuples)
    # feed voltage k+1 was written by address k.
    ref_feeds = {
        int(round(voltage.real)) - 1: (poly, arclength)
        for poly, arclength, voltage in translated["feeds"]
    }
    assert len(ref_feeds) == len(addresses)

    for k, (tag, seg) in enumerate(addresses):
        ref_wire, ref_local = nec_import._locate_segment(flat, tag, seg, _Where())
        assert ours.locate(tag, seg) == (ref_wire, ref_local)

        wire_index, arclength = ours.resolve(tag, seg)
        model_wire = ours.pieces[wire_index]
        ours_point = tuple(
            a + (b - a) * (arclength / model_wire.length)
            for a, b in zip(model_wire.p1, model_wire.p2)
        )

        # (a) the flat wire's own segment centre — NEC's definition.
        w = theirs.wires[ref_wire]
        centre = tuple(
            a + (b - a) * ((ref_local - 0.5) / w.n_seg) for a, b in zip(w.p1, w.p2)
        )
        assert math.dist(ours_point, centre) <= 1e-12 * (1.0 + w.radius)

        # (b) where antennaknobs' polyline translation puts the same feed.
        poly, ref_arclength = ref_feeds[k]
        ref_point = _point_on_polyline(translated["polylines"][poly], ref_arclength)
        assert math.dist(ours_point, tuple(float(c) for c in ref_point)) <= 1e-9


def test_corpus_is_the_whole_reference_set():
    """The gate is the WHOLE corpus, not whatever happened to be on disk."""
    assert len(CORPUS) == 44


# The corpus is 44 clean exported decks: measured, not one of them has an
# endpoint the connection passes move, or a wire another wire's end lands on
# mid-span. So the two most delicate pieces of shared arithmetic — the
# snapping tolerance and the junction shatter — would go unmeasured on the
# corpus alone. These decks measure them, against the same reference.
_SNAPPING_DECKS = {
    # A corner whose two wires sit 1 nm apart: inside the tolerance, so both
    # readers must fuse it into one node.
    "near_miss_corner": """
GW 1 10 0. 0. 0. 1. 0. 0. 1.E-3
GW 2 10 1.000000001 0. 0. 1. 1. 0. 1.E-3
GE 0
EX 0 1 5 0 1.
""",
    # The same corner 1.5 tolerances apart: outside, so both readers must
    # leave it open. Deliberately gapped geometry is unaffected.
    "wide_gap_corner": """
GW 1 10 0. 0. 0. 1. 0. 0. 1.E-3
GW 2 10 1.00015 0. 0. 1. 1. 0. 1.E-3
GE 0
EX 0 1 5 0 1.
""",
    # A tee: wire 2's end lands on wire 1's interior segment boundary, which
    # becomes a cut so the crossing carries current.
    "mid_wire_tee": """
GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3
GW 2 4 0.5 0. 1.E-7 0.5 1. 0. 1.E-3
GE 0
EX 0 1 3 0 1.
EX 0 2 2 0 1.
""",
    # A tee whose stub is finely meshed: the tolerance is the MINIMUM of the
    # pair's segment lengths, so this one stays open in both readers.
    "mid_wire_tee_fine_stub": """
GW 1 4 0. 0. 0. 1. 0. 0. 1.E-3
GW 2 400 0.5 0. 1.E-4 0.5 1. 0. 1.E-3
GE 0
EX 0 1 3 0 1.
""",
    # A GM-rotated copy: six-significant-digit card coordinates land the
    # replica's corner micrometres off the original's.
    "gm_rotated_polygon": """
GW 1 8 0. 0. 0. 1.000000 0. 0. 1.E-3
GM 1 3 0. 0. 90.000000 1.000000 0. 0. 0
GE 0
EX 0 1 4 0 1.
""",
}


@pytest.mark.parametrize("name", sorted(_SNAPPING_DECKS))
def test_connection_passes_match_antennaknobs(name):
    """The connection tolerance and the junction shatter, measured."""
    text = _SNAPPING_DECKS[name]
    cards = _cards(text)
    geometry_cards = [c for c in cards if c.mnemonic in _GEOMETRY_CARDS]
    addresses = _addresses(cards)
    ours = build_geometry(geometry_cards)
    theirs = _reference(cards, addresses)

    for mine, ref in zip(ours.wires, theirs.wires):
        assert tuple(mine.p1) == tuple(ref.p1)
        assert tuple(mine.p2) == tuple(ref.p2)
    assert ours.cuts == dict(theirs._junction_cuts)

    flat = [[w.tag, w.n_seg, list(w.p1), list(w.p2), w.radius] for w in theirs.wires]
    for tag, seg in addresses:
        assert ours.locate(tag, seg) == nec_import._locate_segment(
            flat, tag, seg, _Where()
        )


def test_the_synthetic_decks_actually_exercise_both_passes():
    """A guard on the guard: these decks must move an endpoint and cut a
    wire, or they are measuring nothing the corpus did not."""

    def geometry(name):
        return build_geometry(
            [c for c in _cards(_SNAPPING_DECKS[name]) if c.mnemonic in _GEOMETRY_CARDS]
        )

    tee = geometry("mid_wire_tee")
    assert tee.cuts == {0: frozenset({2})}
    assert len(tee.pieces) == 3
    assert geometry("mid_wire_tee_fine_stub").cuts == {}
    corner = geometry("near_miss_corner")
    assert tuple(corner.wires[1].p1) == tuple(corner.wires[0].p2)
    wide = geometry("wide_gap_corner")
    assert tuple(wide.wires[1].p1) != tuple(wide.wires[0].p2)
