"""``momwire-nec2c --save-decks DIR`` (momwire#1080).

A SimNEC circuit scripted in its portal language carries no cards in its
``.ssn``: SimNEC evaluates the script and sends the deck to the engine, so the
deck the engine receives is the only place that model exists as cards.  The
option writes every received deck to a user-chosen directory, verbatim and
terminated ``EN`` so the file is one standalone deck, and is off unless asked
for.
"""

from __future__ import annotations

import io
import os

import pytest

import momwire_nec2c_client as client
from momwire.deck import parse
from momwire.portal import _portal as nec_portal
from momwire.portal import main

_DECK = (
    "CM version SimNEC test\n"
    "CE\n"
    "GW 1 9 0. 0. -2.5 0. 0. 2.5 0.001\n"
    "GE 0\n"
    "EX 0 1 5 0 1.\n"
    "FR 0 1 0 0 30. 0\n"
    "XQ\n"
)


def _run(argv, stdin_text):
    out = io.StringIO()
    with nec_portal.engine_scope():
        rc = main(argv, stdin=io.StringIO(stdin_text), stdout=out, stderr=io.StringIO())
    return rc, out.getvalue()


def test_every_deck_is_saved_verbatim_as_a_standalone_deck(tmp_path):
    rc, _out = _run(["--save-decks", str(tmp_path)], _DECK + "NX\n" + _DECK + "NX\n")
    assert rc == 0
    saved = sorted(tmp_path.iterdir())
    assert len(saved) == 2
    for path in saved:
        assert path.suffix == ".nec"
        text = path.read_text()
        assert text == _DECK + "EN\n", "the deck body must come back card for card"
        # And it is a deck: the dialect reads it as one structure.
        assert parse(text).feeds


def test_the_environment_variable_selects_it_and_the_flag_wins(tmp_path, monkeypatch):
    by_env = tmp_path / "env"
    by_flag = tmp_path / "flag"
    by_env.mkdir()
    by_flag.mkdir()
    monkeypatch.setenv("MOMWIRE_NEC2C_SAVE_DECKS", str(by_env))
    _run([], _DECK + "NX\n")
    assert len(list(by_env.iterdir())) == 1
    _run([f"--save-decks={by_flag}"], _DECK + "NX\n")
    assert len(list(by_flag.iterdir())) == 1
    assert len(list(by_env.iterdir())) == 1


def test_off_by_default_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("MOMWIRE_NEC2C_SAVE_DECKS", raising=False)
    monkeypatch.chdir(tmp_path)
    rc, _out = _run([], _DECK + "NX\n")
    assert rc == 0
    assert nec_portal._save_decks_dir is None
    assert list(tmp_path.iterdir()) == []


def test_a_missing_directory_fails_at_the_probe(tmp_path):
    missing = tmp_path / "nope"
    rc, out = _run(["--save-decks", str(missing), "-version"], "")
    assert rc == 3
    assert "not a directory" in out
    assert not missing.exists(), (
        "the engine must not create a directory it was not given"
    )


def test_saving_never_changes_the_answer(tmp_path):
    _rc, plain = _run([], _DECK + "NX\n")
    _rc, saving = _run(["--save-decks", str(tmp_path)], _DECK + "NX\n")

    def untimed(text):
        # The MATRIX TIMING and RUN TIME lines are wall clock.
        return [line for line in text.splitlines() if "msec" not in line]

    assert untimed(saving) == untimed(plain)


def test_a_crlf_deck_is_saved_with_crlf(tmp_path):
    crlf = _DECK.replace("\n", "\r\n")
    _run(["--save-decks", str(tmp_path)], crlf + "NX\r\n")
    (path,) = tmp_path.iterdir()
    assert path.read_bytes() == (crlf + "EN\r\n").encode()


@pytest.mark.skipif(os.name != "posix", reason="the shared client is POSIX only")
def test_the_shared_client_spells_the_directory_into_the_engine(tmp_path, monkeypatch):
    """The shared server saves only if its command line says so, so a
    directory chosen by environment becomes the flag, absolute, and lands in
    the socket identity: two clients saving to two places are two servers."""
    monkeypatch.delenv("MOMWIRE_NEC2C_BASIS", raising=False)
    monkeypatch.setenv("MOMWIRE_NEC2C_SAVE_DECKS", str(tmp_path))
    engine, error = client.resolve_engine([], prog="momwire-nec2c-shared")
    assert error is None
    assert engine == ["--save-decks", str(tmp_path)]
    monkeypatch.delenv("MOMWIRE_NEC2C_SAVE_DECKS")
    plain, _ = client.resolve_engine([], prog="momwire-nec2c-shared")
    assert client.config_key(engine, 900.0) != client.config_key(plain, 900.0)
