"""The ``momwire-nec5[-<basis>]`` command family — momwire#1239.

SimNEC picks its engine class, and so the deck SYNTAX it writes, from a
substring of the engine's path: ``nec2c`` gives NEC-2 decks and ``nec5``
NEC-5 decks (``-version`` is only recorded). So a command meant for SimNEC's
NEC-5 slot has to carry ``nec5`` in its name and must not carry ``nec2c``.
The basis rides on the name past ``nec5-``, as it rides past ``eznec-``.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import momwire_eznec_client as eznec_client
import momwire_serve_client as mech
from momwire.deck._solver import BASES

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _family():
    scripts = tomllib.loads(PYPROJECT.read_text())["project"]["scripts"]
    return {k: v for k, v in scripts.items() if k.startswith("momwire-nec5")}


def test_the_family_ships_on_the_eznec_client():
    family = _family()
    assert "momwire-nec5" in family and "momwire-nec5-razor-2p" in family
    assert set(family.values()) == {"momwire_eznec_client:main"}


def test_every_name_reads_as_a_nec5_engine_to_simnec():
    """SimNEC checks `nec2c` first, then `nec5`, on the lowercased path."""
    for name in _family():
        assert "nec5" in name and "nec2c" not in name


def test_no_name_offers_a_basis_that_refuses_nec5_decks():
    """NEC-5 drives at KNOTS; the centre-fed bases refuse every NEC-5 deck,
    so they get no nec5 name."""
    bases = {mech.filename_basis(n, "nec5-") for n in _family()} - {None}
    assert bases <= set(BASES)
    assert not bases & {"sinusoidal", "pulse"}


@pytest.mark.parametrize(
    ("prog", "basis"),
    [
        ("momwire-nec5", None),
        ("momwire-nec5-razor-2p", "razor-2p"),
        ("/usr/local/bin/momwire-nec5-bspline-d1", "bspline-d1"),
        ("C:\\tools\\Momwire-Nec5-Razor-2p.exe", "razor-2p"),
        # The EZNEC spellings keep their meaning, the old alias included.
        ("momwire-eznec-client-razor-2p", "razor-2p"),
        ("momwire-eznec-client-razor-nec5", "razor-nec5"),
        ("momwire-eznec-client", None),
    ],
)
def test_the_basis_rides_on_the_name(prog, basis, monkeypatch, tmp_path):
    seen = {}

    def served(deck_bytes, b, idle, dialect):
        seen["basis"] = b
        seen["dialect"] = dialect
        return b"ok"

    monkeypatch.setattr(eznec_client, "_served_bytes", served)
    deck, out = tmp_path / "in.nec", tmp_path / "out.txt"
    deck.write_text("CE\nEN\n")
    monkeypatch.setattr("sys.argv", [prog, str(deck), str(out)])
    assert eznec_client.main(None) == 0
    assert seen["basis"] == basis
    assert seen["dialect"] == "nec5"


def test_the_version_probe_answers_under_a_nec5_name(capsys, monkeypatch):
    monkeypatch.setattr("sys.argv", ["momwire-nec5-razor-2p", "-version"])
    assert eznec_client.main(None) == 0
    assert capsys.readouterr().out.startswith("NEC5momwire.")
