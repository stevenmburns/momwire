"""momwire#1420 step 4: a multi-run deck's cards parsed once, not once per run.

``_shell._answer`` used to parse a SimNEC multi-run deck four times — the
deck itself, then each of the three single-run decks ``split_runs`` makes of
it — and every one of those re-read the same comment block and geometry.
``parse_nec5_shared`` reads the lines all the texts share once and continues
each text from a copy of the parser. The gates:

* ``parse_nec5_shared(texts)`` is ``[parse_nec5(t) for t in texts]`` outcome
  for outcome — equal decks, or the same exception type and message — on
  every fixture deck and Dan's decks with their runs, and on mutated decks
  that fail at every stage (in the shared lines, in one run, at the split);
* ``render`` with the shared parse is ``render`` with a verbatim copy of the
  old ``_answer`` on those same mutated decks, refusal text included;
* through the production seam, the geometry cards are parsed once;
* a red control: a clone that SHARES the parser's lists fails the gate.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from momwire.deck import _cards
from momwire.deck import _nec5
from momwire.deck._cards import DeckError
from momwire.deck._nec5 import parse_nec5, parse_nec5_shared
from momwire.eznec import _printout, _serve, _shell

FIXTURES = Path(__file__).parent / "fixtures"


def _outcome(text):
    try:
        return parse_nec5(text)
    except Exception as exc:  # noqa: BLE001 — compared, not swallowed
        return exc


def _same(a, b):
    if isinstance(a, Exception) or isinstance(b, Exception):
        assert type(a) is type(b) and str(a) == str(b), (a, b)
    else:
        assert a == b


def _families():
    """Each fixture deck with its runs (when it has several)."""
    out = []
    for path in sorted(FIXTURES.rglob("*.nec")):
        text = path.read_text(encoding="latin-1")
        try:
            runs = _serve.split_runs(text) or []
        except _serve.ServeRefusal:
            runs = []
        out.append((path.name, [text, *runs]))
    return out


def _mutants(text, rng, n):
    lines = text.splitlines()
    out = []
    for _ in range(n):
        ls = list(lines)
        k = int(rng.integers(0, len(ls)))
        op = rng.integers(0, 6)
        if op == 0:
            del ls[k]
        elif op == 1:
            ls.insert(k, "ZZ 1 2 3")
        elif op == 2:
            ls.insert(k, ls[k])
        elif op == 3:
            ls = [ln for ln in ls if not ln.startswith("EN")]
        elif op == 4:
            ls.insert(k, "EX 0 1 1 0 1 0")
        else:
            ls[k] = ls[k][:2] + " x" + ls[k][2:]
        out.append("\n".join(ls) + "\n")
    return out


MULTI = sorted((FIXTURES / "eznec_multi_run_1237").glob("*.nec"))


def test_shared_parse_is_the_one_at_a_time_parse_on_every_fixture():
    families = _families()
    assert sum(len(f) > 1 for _, f in families) >= 3  # multi-run decks present
    for _name, texts in families:
        got = parse_nec5_shared(texts)
        assert len(got) == len(texts)
        for g, t in zip(got, texts, strict=True):
            _same(g, _outcome(t))


def test_shared_parse_agrees_on_decks_broken_everywhere():
    rng = np.random.default_rng(1420)
    kinds = set()
    for path in MULTI:
        for text in _mutants(path.read_text(encoding="latin-1"), rng, 120):
            try:
                runs = _serve.split_runs(text) or []
            except _serve.ServeRefusal:
                runs = []
            texts = [text, *runs]
            got = parse_nec5_shared(texts)
            for g, t in zip(got, texts, strict=True):
                want = _outcome(t)
                _same(g, want)
                kinds.add(type(want).__name__)
    assert {"Nec5Deck", "DeckError"} <= kinds, kinds


def _old_answer(text, *, basis, echo_text=None):
    """``_shell._answer`` as spelled before momwire#1420."""
    echo = text if echo_text is None else echo_text
    try:
        deck = parse_nec5(text)
    except DeckError as exc:
        return _printout.render_refusal(echo, str(exc), basis=basis)
    if echo_text is not None:
        deck = replace(deck, source_text=echo_text)
    try:
        runs = _serve.split_runs(text)
    except _serve.ServeRefusal as exc:
        return _printout.render_refusal(echo, str(exc), basis=basis)
    if runs is not None:
        try:
            answers = _serve.serve_runs([parse_nec5(t) for t in runs], basis=basis)
        except (DeckError, _serve.ServeRefusal) as exc:
            return _printout.render_refusal(echo, str(exc), basis=basis)
        return deck, answers, True
    try:
        data = _serve.serve(deck, basis=basis)
    except _serve.ServeRefusal as exc:
        return _printout.render_refusal(echo, str(exc), basis=basis)
    return deck, [data], False


def _strip(printout):
    return "\n".join(ln for ln in printout.splitlines() if "RUN TIME" not in ln)


def test_render_refuses_exactly_as_before(monkeypatch):
    """Mutated multi-run decks refuse (or solve) with the same printout
    whichever ``_answer`` reads them. Only decks that REFUSE are rendered
    both ways, so the gate stays fast; the solving ones are the identity
    gate's business."""
    rng = np.random.default_rng(7)
    new = _shell._answer
    refusals = 0
    for path in MULTI:
        for text in _mutants(path.read_text(encoding="latin-1"), rng, 60):
            got = new(text, basis="razor-2p")
            if not isinstance(got, str):
                continue
            refusals += 1
            assert got == _old_answer(text, basis="razor-2p")
    assert refusals >= 50


def test_the_seam_parses_the_geometry_once(monkeypatch):
    path = next(iter(sorted(FIXTURES.glob("eznec_multi_run_1237/*.nec"))))
    text = path.read_text(encoding="latin-1")
    runs = _serve.split_runs(text)
    assert runs and len(runs) >= 2
    gw_lines = sum(1 for ln in text.splitlines() if ln[:2].upper() == "GW")
    seen = {"gw": 0, "clone": 0}
    parse_card, clone = _cards.parse_card, _nec5._Nec5Parser.clone

    def spy_card(line):
        if line[:2].upper() == "GW":
            seen["gw"] += 1
        return parse_card(line)

    def spy_clone(self):
        seen["clone"] += 1
        return clone(self)

    monkeypatch.setattr(_nec5, "parse_card", spy_card)
    monkeypatch.setattr(_nec5._Nec5Parser, "clone", spy_clone)
    _shell.render(text, basis="razor-2p")
    assert seen["gw"] == gw_lines  # once, where it was 1 + len(runs) times
    assert seen["clone"] == 1 + len(runs)


def test_a_seeded_shared_list_fails_the_gate(monkeypatch):
    """Red control: a clone that shares its lists lets one run's cards leak
    into the next, and the fixture gate sees it."""

    def leaky(self):
        other = _nec5._Nec5Parser.__new__(_nec5._Nec5Parser)
        other.__dict__.update(self.__dict__)
        return other

    monkeypatch.setattr(_nec5._Nec5Parser, "clone", leaky)
    with pytest.raises(AssertionError):
        test_shared_parse_is_the_one_at_a_time_parse_on_every_fixture()


def test_the_multi_run_fixtures_are_there():
    """A list, not a generator: every gate above reads it."""
    assert len(MULTI) >= 3
