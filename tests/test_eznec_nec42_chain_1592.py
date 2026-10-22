"""SimNEC's batch and stream modes in EZNEC's NEC-4.2 slot (momwire#1592).

SimNEC queues several runs into one engine call with ``NX`` between them, in
a file (batch) or on stdin as ``- -`` (stream). Before momwire#1592 the
NEC-4.2 slot handed the whole chain to its one-deck path: the deck reader
stops at the first ``NX`` while the echo counted every execute card up to
``EN``, so a three-run chain refused as ``3 execute cards against 1 execute
runs: this engine cannot place the card echo`` (Dan AC6LA, 0.80.0), and
``- -`` refused outright on stderr.

The chain is now the NEC-5 slot's (``tests/test_eznec_stream_1512.py``):
each structure solved as the standalone deck its body makes, printed in this
slot's layout, ending on its ``NX`` echo, the last one alone carrying
``RUN TIME``. No NEC-4.2 printout of a chain has been captured, so what is
pinned here is that rule, not licensed bytes.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

import momwire_eznec_client
from momwire.eznec import _shell

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "eznec_nec42"
BASIS = "sinusoidal"
_RUN_TIME = re.compile(r"^ RUN TIME =.*$", re.MULTILINE)
_FILL = re.compile(r"FILL= +[\d.]+ SEC\.")


def _deck(capture: str) -> str:
    (path,) = FIXTURES.glob(f"{capture}_*.nec")
    return path.read_bytes().decode("latin-1")


def _body(deck: str) -> str:
    return deck[: deck.rindex("EN")]


# Three different structures: SimNEC's chains change the design between runs.
DECKS = [_deck("0233"), _deck("0223"), _deck("0229")]


def _chain(decks: list[str], *, last: str = "EN") -> str:
    return "".join(_body(d) + "NX\n" for d in decks[:-1]) + _body(decks[-1]) + last


def _render(text: str) -> str:
    return _shell.render(text, basis=BASIS, dialect="nec4")


def _as_chained(printout: str, *, closing: bool) -> str:
    """A standalone printout as it stands inside a chain: its ``EN`` echo an
    ``NX`` echo unless it closes the chain, and ``RUN TIME`` only if it
    does."""
    lines = printout.splitlines()
    assert _RUN_TIME.match(lines[-1]) and lines[-2] == "", lines[-2:]
    if closing:
        return "\n".join(lines) + "\n"
    echo = lines[-3]
    assert re.match(r" \*\*\*\*\* INPUT LINE +\d+  EN ", echo), echo
    lines[-3] = echo.replace("  EN ", "  NX ", 1)
    return "\n".join(lines[:-2]) + "\n"


def _masked(text: str) -> str:
    """The printout without its two measured times."""
    return _FILL.sub("FILL=<masked>", _RUN_TIME.sub(" RUN TIME = <masked>", text))


def test_a_chain_no_longer_refuses_at_the_card_echo():
    printout = _render(_chain(DECKS))
    assert "NEC ERROR" not in printout
    assert printout.count("NUMERICAL ELECTROMAGNETICS CODE (NEC-4.2)") == 3
    assert len(_RUN_TIME.findall(printout)) == 1


def test_each_structure_prints_as_its_own_deck_would():
    expected = "".join(
        _as_chained(_render(deck), closing=k == len(DECKS) - 1)
        for k, deck in enumerate(DECKS)
    )
    assert _masked(_render(_chain(DECKS))) == _masked(expected)


def test_a_chain_ending_on_nx_prints_no_run_time():
    """A file that ends on its last ``NX`` is a truncated chain: the empty
    structure after it refuses as truncated, after the ones that solved."""
    printout = _render(_chain(DECKS, last="NX\n"))
    assert printout.count("NUMERICAL ELECTROMAGNETICS CODE (NEC-4.2)") == 4
    assert "NEC ERROR" in printout.splitlines()[-1]


def test_a_refusing_structure_ends_the_chain_after_the_solved_ones():
    # A card the nec4 dialect refuses by name (GA was, before momwire#1592
    # served it).
    refused = _deck("0223").replace("PQ", "GC 0 0 1 .001 .002\nPQ", 1)
    decks = [DECKS[0], refused, DECKS[2]]
    printout = _render(_chain(decks))
    assert printout.count("NUMERICAL ELECTROMAGNETICS CODE (NEC-4.2)") == 2
    first = _as_chained(_render(DECKS[0]), closing=False)
    assert _masked(printout).startswith(_masked(first))
    assert "NEC ERROR" in printout.splitlines()[-1]
    assert _RUN_TIME.search(printout) is None


def _streamed(chunks: list[bytes]) -> list[bytes]:
    feed = list(chunks)
    writes: list[bytes] = []

    def read() -> bytes:
        return feed.pop(0) if feed else b""

    assert _shell.stream(read, writes.append, basis=BASIS, dialect="nec4") == 0
    return writes


def test_the_stream_answers_each_structure_as_it_arrives():
    """SimNEC's ``- -``: each structure written ending on ``NX``, stdin closed
    after the last with no ``EN``. Each printout is the chained one."""
    chunks = [(_body(d) + "NX\n").encode("latin-1") for d in DECKS]
    writes = _streamed(chunks)
    assert len(writes) == len(DECKS)
    for deck, wrote in zip(DECKS, writes, strict=True):
        expected = _as_chained(_render(deck), closing=False)
        assert _masked(wrote.decode("latin-1")) == _masked(
            _shell.printout_bytes(expected).decode("latin-1")
        )


def test_a_stream_closed_by_en_ends_with_run_time():
    writes = _streamed([_chain(DECKS[:2]).encode("latin-1")])
    assert len(writes) == 2
    assert _RUN_TIME.search(writes[-1].decode("latin-1"))
    assert b"RUN TIME" not in writes[0]


def test_the_shell_streams_in_the_nec4_slot(monkeypatch):
    seen = {}

    def fake_stream(**kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(_shell, "stream", fake_stream)
    assert _shell.main(["-", "-"], basis=BASIS, dialect="nec4") == 0
    assert seen == {"basis": BASIS, "dialect": "nec4"}


def test_the_python_launcher_streams_in_the_nec4_slot(monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        momwire_eznec_client.sys, "argv", ["momwire-nec42-pulse", "-", "-"]
    )
    assert momwire_eznec_client.main() == 0
    code = seen["command"][2]
    assert "main(['-', '-'], basis='pulse', dialect='nec4')" in code


@pytest.mark.parametrize("prog", ["momwire-nec5", "momwire-nec5-razor-2p"])
def test_the_python_launchers_nec5_stream_is_unchanged(monkeypatch, prog):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(momwire_eznec_client.sys, "argv", [prog, "-", "-"])
    assert momwire_eznec_client.main() == 0
    assert "dialect" not in seen["command"][2]
