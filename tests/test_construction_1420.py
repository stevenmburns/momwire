"""momwire#1420 step 5: per-call construction checks without O(wires^2) or
per-wire numpy round trips.

* ``_serve._phantom_tags`` stops at the first test that settles a wire (too
  long; or an endpoint nearer than the larger clearance floor) instead of
  measuring every endpoint pair; it is held to the full scan, kept as
  ``_phantom_tags_full``, on every fixture deck and on random decks with
  parked phantom wires, and decks with a non-finite coordinate take the full
  scan.
* ``_wire_spec.find_duplicated_wires`` and ``short_segment_wires`` read every
  straight wire from one array; each is held to its per-wire walk (forced by
  making the vector helper decline) on random geometry with duplicates,
  reversed duplicates, near misses and short segments.
* through the production seam (``_shell.render``) the vector paths are the
  ones that ran.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from momwire import _wire_spec
from momwire.deck._nec5 import DeckError, parse_nec5
from momwire.eznec import _serve
from momwire.eznec._shell import render

FIXTURES = Path(__file__).parent / "fixtures"


def _decks():
    out = []
    for path in sorted(FIXTURES.rglob("*.nec")):
        try:
            out.append(parse_nec5(path.read_text(encoding="latin-1")))
        except (DeckError, ValueError):
            continue
    return out


def _full(deck, wavelength):
    if len(deck.wires) < 2:  # `_phantom_tags`' own guard, ahead of either scan
        return frozenset()
    ends = [(tuple(w.end1), tuple(w.end2)) for w in deck.wires]
    return _serve._phantom_tags_full(deck, ends, wavelength)


def _phantom_deck(rng):
    """A few real wires near the origin and, often, short wires parked far
    away — at, inside and outside the clearance floors."""
    lam = float(rng.choice([2.0, 20.0, 80.0]))
    lines = ["CM phantom", "CE"]
    n = int(rng.integers(2, 7))
    for tag in range(1, n + 1):
        if rng.random() < 0.4:
            far = float(rng.choice([9.0, 10.0, 10.5, 50.0, 173.0])) * lam
            ext = float(rng.choice([0.001, 0.049, 0.05, 0.2])) * lam
            a = (far, 0.0, 1.0)
            b = (far + ext, 0.0, 1.0)
        else:
            a = tuple(float(x) for x in rng.uniform(-3, 3, 3))
            b = tuple(float(x) for x in rng.uniform(-3, 3, 3))
        lines.append(f"GW {tag} 3 {' '.join(repr(c) for c in (*a, *b))} 1e-3")
    lines += ["GE 0", "FR 0 1 0 0 14", "EX 0 1 1 0 1 0", "XQ", "EN"]
    return parse_nec5("\n".join(lines) + "\n"), lam


def test_phantom_tags_match_the_full_scan():
    decks = _decks()
    assert len(decks) > 100
    found = 0
    for deck in decks:
        for lam in (0.5, 21.3, 80.0, 1e4):
            got = _serve._phantom_tags(deck, lam)
            assert got == _full(deck, lam)
            found += bool(got)
    rng = np.random.default_rng(1420)
    for _ in range(600):
        deck, lam = _phantom_deck(rng)
        got = _serve._phantom_tags(deck, lam)
        assert got == _full(deck, lam)
        found += bool(got)
    assert found >= 50  # the phantom branch itself is reached


def test_phantom_tags_take_the_full_scan_on_non_finite(monkeypatch):
    deck, lam = _phantom_deck(np.random.default_rng(0))
    calls = []
    real = _serve._phantom_tags_full
    monkeypatch.setattr(
        _serve, "_phantom_tags_full", lambda *a: calls.append(1) or real(*a)
    )
    _serve._phantom_tags(deck, math.nan)
    _serve._phantom_tags(deck, math.inf)
    assert len(calls) == 2


def _random_wires(rng, n):
    pls, npe, radius = [], [], []
    for _ in range(n):
        r = rng.random()
        if pls and r < 0.25:
            twin = pls[int(rng.integers(0, len(pls)))]
            pl = twin[::-1].copy() if rng.random() < 0.5 else twin.copy()
            if rng.random() < 0.3:
                pl = pl + float(
                    rng.choice([1e-13, 1e-10, 4e-10, 6e-10])
                ) * rng.standard_normal(pl.shape)
        else:
            pl = rng.uniform(-5, 5, (2, 3)) * float(rng.choice([1e-3, 1.0, 100.0]))
        pls.append(pl)
        npe.append([int(rng.integers(1, 40))])
        radius.append(float(rng.choice([0.0, 1e-4, 1e-3, 0.05, 0.5])))
    return pls, npe, radius


def test_duplicate_and_short_segment_checks_match_their_walks():
    rng = np.random.default_rng(14205)
    hits = {"dup": 0, "short": 0}
    for _ in range(800):
        pls, npe, radius = _random_wires(rng, int(rng.integers(1, 12)))
        fast_dup = _wire_spec.find_duplicated_wires(pls)
        fast_short = _wire_spec.short_segment_wires(pls, npe, radius)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(_wire_spec, "_vertex_keys", lambda pls: None)
            m.setattr(_wire_spec, "_straight_segment_lengths", lambda *a: None)
            assert fast_dup == _wire_spec.find_duplicated_wires(pls)
            walk_short = _wire_spec.short_segment_wires(pls, npe, radius)
        assert fast_short == walk_short
        hits["dup"] += bool(fast_dup)
        hits["short"] += bool(fast_short)
    assert hits["dup"] > 50 and hits["short"] > 50, hits


def test_the_seam_takes_the_vector_paths(monkeypatch):
    calls = {"keys": 0, "lengths": 0, "full": 0}
    keys, lengths, full = (
        _wire_spec._vertex_keys,
        _wire_spec._straight_segment_lengths,
        _serve._phantom_tags_full,
    )

    def spy_keys(pls):
        got = keys(pls)
        calls["keys"] += got is not None
        return got

    def spy_lengths(*a):
        got = lengths(*a)
        calls["lengths"] += got is not None
        return got

    def spy_full(*a):
        calls["full"] += 1
        return full(*a)

    monkeypatch.setattr(_wire_spec, "_vertex_keys", spy_keys)
    monkeypatch.setattr(_wire_spec, "_straight_segment_lengths", spy_lengths)
    monkeypatch.setattr(_serve, "_phantom_tags_full", spy_full)
    out = render(
        "CM monopole\nCE\nGW 1 10 0 0 0 0 0 5 1e-3\nGW 2 4 0 0 5 1 0 5 1e-3\n"
        "GE 1\nGN 1\nFR 0 1 0 0 14\nEX 0 1 1 0 1 0\nXQ\nEN\n",
        basis="razor-2p",
    )
    assert "NEC ERROR" not in out
    assert calls["keys"] >= 1 and calls["lengths"] >= 1 and calls["full"] == 0


def test_a_seeded_floor_fails_the_phantom_gate(monkeypatch):
    """Red control: a clearance floor one ulp low is caught."""
    real_max = max

    def low_max(*a, **k):
        return float(np.nextafter(real_max(*a, **k), -np.inf))

    monkeypatch.setattr(_serve, "max", low_max, raising=False)
    deck = parse_nec5(
        "CM\nCE\nGW 1 3 0 0 1 1 0 1 1e-3\n"
        f"GW 2 3 {201.0!r} 0 1 {201.01!r} 0 1 1e-3\n"
        "GE 0\nFR 0 1 0 0 14\nEX 0 1 1 0 1 0\nXQ\nEN\n"
    )
    # the parked wire's nearest endpoint sits EXACTLY on the 10-lambda floor
    lam = 20.0
    assert _full(deck, lam) == frozenset()
    with pytest.raises(AssertionError):
        assert _serve._phantom_tags(deck, lam) == _full(deck, lam)
