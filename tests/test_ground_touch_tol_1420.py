"""momwire#1420 step 1: the ground-touch tolerance and the EZNEC seam's
geometry rules, asked of arrays instead of one tiny array at a time.

Both are speed changes that must move no bit, so every gate here is EXACT
equality against the spelling it replaced:

* ``ground_touch_tol`` against the old ``np.sum(np.linalg.norm(np.diff(...)))``
  spelling, and ``segment_touch_tols`` against ``ground_touch_tol`` row by
  row, on random and adversarial polylines;
* the five geometry rules (``_plane_crossings``, ``_spans``,
  ``_crossing_nodes``, ``_has_buried_wire``, ``_geometry_refusal``) against
  verbatim copies of the per-wire walks they replaced (``_ref_*`` below), on
  every ``.nec`` fixture in this tree and on randomly generated decks whose
  ends sit on, just off, above and below the plane under every ground card;
* and through the production seam (``_shell.render``), that the table path
  is the one that ran and that no rule fell back to asking the tolerance one
  card at a time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from momwire import _ground_spec
from momwire.deck._nec5 import DeckError, parse_nec5
from momwire.eznec import _serve as S
from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures"


def _old_tol(polyline):
    """``ground_touch_tol`` as it was spelled before momwire#1420."""
    pl = np.asarray(polyline, dtype=np.float64)
    length = float(np.sum(np.linalg.norm(np.diff(pl, axis=0), axis=1)))
    return 1e-6 * max(length, 1e-30)


# ---- verbatim copies of the per-wire walks (docstrings dropped) -----------
def _ref_plane_crossings(deck):
    crossings: dict = {}
    if not S._has_interface(deck):
        return crossings
    for wire in deck.wires:
        a = np.asarray(wire.end1, dtype=float)
        b = np.asarray(wire.end2, dtype=float)
        tol = _old_tol(np.array([a, b]))
        za, zb = (float(a[2]), float(b[2]))
        if not (za < -tol and zb > tol or (za > tol and zb < -tol)):
            continue
        z = [p[2] for p in S.node_points(wire)]
        node = next((k for k, zk in enumerate(z) if abs(zk) <= tol), None)
        if node is not None:
            x, y = S.node_points(wire)[node][:2]
            segment = None
        else:
            t = za / (za - zb)
            x, y = (a + (b - a) * t)[:2]
            segment = next(
                (
                    k
                    for k in range(wire.segment_count)
                    if (z[k] < 0.0) != (z[k + 1] < 0.0)
                )
            )
        crossings[wire.tag] = S._PlaneCrossing(
            tag=wire.tag, point=(float(x), float(y), 0.0), node=node, segment=segment
        )
    return crossings


def _ref_spans(deck):
    crossings = _ref_plane_crossings(deck)
    spans = []
    for wire in deck.wires:
        pl = np.array([wire.end1, wire.end2], dtype=float)
        crossing = crossings.get(wire.tag)
        if crossing is None:
            spans.append((wire.tag, pl))
            continue
        x = np.asarray(crossing.point, dtype=float)
        spans.append((wire.tag, np.array([pl[0], x])))
        spans.append((wire.tag, np.array([x, pl[1]])))
    return spans


def _ref_crossing_nodes(deck):
    nodes: set = set()
    if not S._has_interface(deck):
        return nodes
    for _tag, pl in _ref_spans(deck):
        tol = _old_tol(pl)
        if float(pl[:, 2].min()) < -tol:
            for end in pl:
                if abs(float(end[2])) <= tol:
                    nodes.add(S._node_key(tuple(end)))
    return nodes


def _ref_has_buried_wire(deck):
    if not S._has_interface(deck):
        return False
    for _tag, pl in _ref_spans(deck):
        tol = _old_tol(pl)
        if float(pl[:, 2].min()) < -tol:
            return True
    return False


def _ref_geometry_refusal(deck):
    if not S._has_interface(deck):
        return None
    sommerfeld = isinstance(deck.ground, S.Nec5SommerfeldGround)
    card = "GD" if isinstance(deck.ground, S.Nec5MininecGround) else "GN 1"
    spans = _ref_spans(deck)
    for tag, pl in spans:
        tol = _old_tol(pl)
        zmin = float(pl[:, 2].min())
        if zmin < -tol:
            if not sommerfeld:
                return S._REFUSE_BURIED_NO_MEDIUM.format(
                    tag=tag, zmin=zmin, card=card, why=S._WHY_NO_MEDIUM[card]
                )
            continue
        if abs(pl[0, 2]) <= tol and abs(pl[1, 2]) <= tol:
            return S._REFUSE_IN_PLANE_WIRE.format(tag=tag)
    buried = [tag for tag, pl in spans if float(pl[:, 2].max()) < -_old_tol(pl)]
    if buried:
        crossing_nodes = _ref_crossing_nodes(deck)
        contacts = []
        for tag, pl in spans:
            if float(pl[:, 2].max()) <= _old_tol(pl):
                continue
            for end_index in _ground_spec.contact_ends([pl], 0.0):
                end = pl[0] if end_index[1] == "start" else pl[1]
                if S._node_key(tuple(end)) not in crossing_nodes:
                    contacts.append(tag)
                    break
        if contacts:
            return S._REFUSE_BURIED_WITH_CONTACT.format(cw=contacts[0], bw=buried[0])
    return None


# ---- the tolerance -------------------------------------------------------
def _bits(x) -> bytes:
    return np.asarray(x, dtype=np.float64).tobytes()


def _random_points(rng, n):
    scale = 10.0 ** rng.integers(-9, 5, size=(n, 1))
    pts = rng.standard_normal((n, 3)) * scale
    # sprinkle exact zeros, repeats and signed zeros
    pts[rng.random((n, 3)) < 0.1] = 0.0
    pts[rng.random((n, 3)) < 0.03] = -0.0
    return pts


@pytest.mark.parametrize("k", [1, 2, 3, 5, 9, 17])
def test_ground_touch_tol_is_the_old_spelling_bit_for_bit(k):
    rng = np.random.default_rng(1420 + k)
    for _ in range(400):
        pl = _random_points(rng, k)
        assert _bits(_ground_spec.ground_touch_tol(pl)) == _bits(_old_tol(pl))
        # list-of-tuples input, as the deck layer hands it
        as_list = [tuple(p) for p in pl.tolist()]
        assert _bits(_ground_spec.ground_touch_tol(as_list)) == _bits(_old_tol(pl))


def test_ground_touch_tol_degenerate_inputs_match():
    for pl in ([[0.0, 0.0, 0.0]], [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]):
        assert _bits(_ground_spec.ground_touch_tol(pl)) == _bits(_old_tol(pl))
    nan = [[0.0, 0.0, np.nan], [0.0, 0.0, 1.0]]
    assert np.isnan(_ground_spec.ground_touch_tol(nan)) and np.isnan(_old_tol(nan))


@pytest.mark.parametrize("n", [0, 1, 3, 1000, 20000])
def test_segment_touch_tols_is_ground_touch_tol_per_row(n):
    rng = np.random.default_rng(n)
    a, b = _random_points(rng, n), _random_points(rng, n)
    b[: n // 10] = a[: n // 10]  # zero-length rows hit the 1e-30 floor
    got = _ground_spec.segment_touch_tols(a, b)
    assert got.shape == (n,)
    want = [_old_tol(np.array([a[i], b[i]])) for i in range(n)]
    assert _bits(got) == _bits(want)


# ---- the geometry rules --------------------------------------------------
def _fixture_decks():
    decks = []
    for path in sorted(FIXTURES.rglob("*.nec")):
        try:
            decks.append(
                (
                    str(path.relative_to(FIXTURES)),
                    parse_nec5(path.read_text(encoding="latin-1")),
                )
            )
        except (DeckError, ValueError, UnicodeError):
            continue
    return decks


GROUNDS = {
    "free": "GN -1\n",
    "perfect": "GN 1\n",
    "refl": "GN 0,0,0,0,13.,.005\n",
    "sommerfeld": "GN 2,0,0,0,13.,.005\n",
    "mininec": "GD 0,0,0,0,13.,.005\n",
}
# z values chosen to land on, just inside and just outside every tolerance
ZS = (-2.0, -0.5, -3e-7, -1e-9, -0.0, 0.0, 1e-9, 3e-7, 0.5, 2.0)


def _random_deck_text(rng, ground):
    lines = ["CM momwire#1420 random geometry", "CE"]
    n = int(rng.integers(1, 9))
    hub = (0.0, 0.0, float(rng.choice(ZS)))
    for tag in range(1, n + 1):

        def pt():
            if rng.random() < 0.3:
                return hub
            return (
                float(rng.uniform(-3, 3)),
                float(rng.uniform(-3, 3)),
                float(rng.choice(ZS)),
            )

        a, b = pt(), pt()
        if a == b:
            b = (a[0] + 1.0, a[1], a[2])
        segs = int(rng.integers(1, 12))
        coords = " ".join(repr(c) for c in (*a, *b))
        lines.append(f"GW {tag} {segs} {coords} 1e-3")
    lines += [
        "GE 1",
        GROUNDS[ground].rstrip("\n"),
        "FR 0 1 0 0 14",
        "EX 0 1 1 0 1 0",
        "XQ",
        "EN",
    ]
    return "\n".join(lines) + "\n"


def _random_decks():
    rng = np.random.default_rng(14201)
    out = []
    for ground in GROUNDS:
        for k in range(150):
            text = _random_deck_text(rng, ground)
            try:
                out.append((f"random-{ground}-{k}", parse_nec5(text)))
            except (DeckError, ValueError):
                continue
    return out


def _same_spans(a, b):
    assert [t for t, _ in a] == [t for t, _ in b]
    for (_, x), (_, y) in zip(a, b, strict=True):
        assert x.shape == y.shape and x.tobytes() == y.tobytes()


def _check(name, deck):
    assert S._plane_crossings(deck) == _ref_plane_crossings(deck), name
    _same_spans(S._spans(deck), _ref_spans(deck))
    assert S._crossing_nodes(deck) == _ref_crossing_nodes(deck), name
    assert S._has_buried_wire(deck) == _ref_has_buried_wire(deck), name
    assert S._geometry_refusal(deck) == _ref_geometry_refusal(deck), name


def test_rules_match_the_walks_on_every_fixture_deck():
    decks = _fixture_decks()
    assert len(decks) > 100
    for name, deck in decks:
        _check(name, deck)


def test_rules_match_the_walks_on_random_geometry():
    decks = _random_decks()
    assert len(decks) > 600
    seen = {"crossing": 0, "buried": 0, "refused": 0, "contact": 0}
    for name, deck in decks:
        _check(name, deck)
        seen["crossing"] += bool(_ref_plane_crossings(deck))
        seen["buried"] += _ref_has_buried_wire(deck)
        seen["refused"] += _ref_geometry_refusal(deck) is not None
        seen["contact"] += bool(_ref_crossing_nodes(deck))
    # the generator must actually reach every branch it claims to test
    assert all(v >= 20 for v in seen.values()), seen


def test_the_seam_runs_the_table_path(monkeypatch):
    """Through ``render`` (the production seam), the rules read the span
    table, and none of them asks ``ground_touch_tol`` card by card."""
    counts = {"table": 0, "vector": 0, "per_card_in_rules": 0}
    table, vector, tol = (
        S._span_table,
        _ground_spec.segment_touch_tols,
        _ground_spec.ground_touch_tol,
    )
    rules = {
        "_plane_crossings",
        "_spans",
        "_span_table",
        "_crossing_nodes",
        "_has_buried_wire",
        "_geometry_refusal",
        "_card_ends",
        "crossings_add",
    }

    def spy_table(deck):
        counts["table"] += 1
        return table(deck)

    def spy_vector(a, b):
        counts["vector"] += 1
        return vector(a, b)

    def spy_tol(pl):
        if sys._getframe(1).f_code.co_name in rules:
            counts["per_card_in_rules"] += 1
        return tol(pl)

    monkeypatch.setattr(S, "_span_table", spy_table)
    monkeypatch.setattr(_ground_spec, "segment_touch_tols", spy_vector)
    monkeypatch.setattr(_ground_spec, "ground_touch_tol", spy_tol)
    for name in ("dan_through.nec", "thr_n21.nec"):
        text = (FIXTURES / "eznec_split_1281" / name).read_text(encoding="latin-1")
        out = render(text, basis="razor-2p")
        assert "NEC ERROR" not in out or "crossing" in out
    assert counts["table"] > 0 and counts["vector"] > 0
    assert counts["per_card_in_rules"] == 0


def test_a_seeded_one_bit_change_fails_the_tolerance_gate(monkeypatch):
    """Red control: the gate above sees a single-ulp change."""
    good = _ground_spec.segment_touch_tols

    def off_by_one_ulp(a, b):
        t = good(a, b)
        t[0:1] = np.nextafter(t[0:1], np.inf)
        return t

    monkeypatch.setattr(_ground_spec, "segment_touch_tols", off_by_one_ulp)
    with pytest.raises(AssertionError):
        test_segment_touch_tols_is_ground_touch_tol_per_row(3)
