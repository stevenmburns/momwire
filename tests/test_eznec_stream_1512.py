"""SimNEC's streaming call, ``- -`` (momwire#1512): decks on stdin, printouts
on stdout, one structure at a time.

SimNEC starts the engine once, writes one structure ending on ``NX``, reads
that structure's printout up to its ``NX`` echo, writes the next, and closes
stdin after the last ``NX`` with no ``EN`` (the A4 captures,
``scripts/core_split/nx_decks/simnec_stream_*``). `_shell.stream` answers each
structure as soon as its ``NX`` line is complete and flushes its printout
before reading on; the printouts are the ones a file holding the whole
stream gets, without the truncated empty structure a file ending on ``NX``
is refused for. The no-Python executable is held to these bytes through real
pipes by ``tests/test_standalone_nopython.py`` and
``scripts/core_split/stream_compare.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from momwire.deck._nec5 import next_structure, split_structures
from momwire.eznec import _shell

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "razor_printout_3a"
DIPOLE = (FIXTURES / "node_gap_two_sources.nec").read_text(encoding="latin-1")
CHAIN_THEN_REFUSAL = (FIXTURES / "chain_then_refusal.nec").read_text(encoding="latin-1")
RAISES = (FIXTURES / "echo_raises_inf_field.nec").read_text(encoding="latin-1")
BASIS = "razor-2p"


def _body(deck: str) -> str:
    """A deck's cards before its EN."""
    return deck[: deck.rindex("EN")]


def _streamed(chunks: list[bytes]) -> tuple[list[bytes], list[str]]:
    """Run `stream` on ``chunks`` (one read each): the bytes of each write,
    and the order of reads and writes."""
    feed = list(chunks)
    writes: list[bytes] = []
    events: list[str] = []

    def read() -> bytes:
        events.append("read")
        return feed.pop(0) if feed else b""

    def write(data: bytes) -> None:
        events.append("write")
        writes.append(data)

    assert _shell.stream(read, write, basis=BASIS) == 0
    return writes, events


def _file_form(text: str) -> bytes:
    return _shell.printout_bytes(_shell.render(text, basis=BASIS))


def test_next_structure_cuts_after_a_complete_nx_or_en_line():
    assert next_structure("GW 1\nNX\nGW 2", final=False) == (8, "NX")
    assert next_structure("GW 1\nNX", final=False) is None  # NX not complete yet
    assert next_structure("GW 1\nNX", final=True) == (7, "NX")
    assert next_structure("GW 1\r\nen\r\n", final=False) == (10, "EN")
    assert next_structure("GW 1\nXQ\n", final=True) is None
    # a bare CR completes the line, as str.splitlines reads it
    assert next_structure("NX\rGW", final=False) == (3, "NX")


def test_an_open_end_keeps_no_empty_structure():
    text = "GW 1\nNX\n"
    closed = split_structures(text)
    assert [s.terminator for s in closed] == ["NX", None]
    (only,) = split_structures(text, open_end=True)
    assert only.terminator == "NX"
    assert split_structures("GW 1\nEN\n", open_end=True) is None


def test_each_structure_is_written_before_the_next_is_read():
    one = _body(DIPOLE) + "NX\n"
    writes, events = _streamed([one.encode(), one.encode()])
    assert events == ["read", "write", "read", "write", "read"]
    whole = _file_form(one + one)
    # the file form refuses the empty structure after the last NX; the
    # stream just ends there
    assert whole.startswith(b"".join(writes))
    assert b"deck has no EN card" in whole[len(b"".join(writes)) :]
    assert all(w.rstrip().split(b"\r\n")[-1].split()[4] == b"NX" for w in writes)
    assert b"RUN TIME" not in b"".join(writes)


def test_a_stream_in_one_read_prints_the_same_bytes():
    one = _body(DIPOLE) + "NX\n"
    writes, _ = _streamed([(one * 3).encode()])
    separate, _ = _streamed([one.encode()] * 3)
    assert writes == separate


def test_en_closes_the_stream_with_run_time_and_nothing_after_it_is_read():
    text = _body(DIPOLE) + "NX\n" + DIPOLE + _body(DIPOLE) + "NX\n"
    writes, events = _streamed([text.encode()])
    assert len(writes) == 2
    assert events == ["read", "write", "write"]
    assert b"".join(writes) == _file_form(text)
    assert b"RUN TIME" in writes[1]


def test_a_refusal_ends_the_stream_as_it_ends_a_chain():
    text = CHAIN_THEN_REFUSAL.replace("\nEN", "\nNX") + _body(DIPOLE) + "NX\n"
    writes, _ = _streamed([text.encode()])
    assert len(writes) == 3
    assert b"NEC ERROR" in writes[2]
    assert b"".join(writes) == _file_form(CHAIN_THEN_REFUSAL)


def test_an_exception_is_the_internal_error_printout_and_ends_the_stream():
    piece = _body(RAISES) + "NX\n"
    text = _body(DIPOLE) + "NX\n" + piece + _body(DIPOLE) + "NX\n"
    writes, _ = _streamed([text.encode()])
    assert len(writes) == 2
    assert b"INTERNAL ERROR IN MOMWIRE ENGINE - OverflowError" in writes[1]
    assert b"unit 3a: PQ's fourth field" in writes[1]


def test_cards_left_when_stdin_closes_are_a_truncated_deck():
    half = _body(DIPOLE)[:120]
    writes, _ = _streamed([(_body(DIPOLE) + "NX\n" + half).encode()])
    assert len(writes) == 2
    assert b"deck has no EN card" in writes[1]
    assert writes[1] == _file_form(half)


@pytest.mark.parametrize("tail", [b"", b"\n", b"  \r\n\t\n"])
def test_stdin_closing_after_nx_ends_quietly(tail):
    writes, _ = _streamed([(_body(DIPOLE) + "NX\n").encode() + tail])
    assert len(writes) == 1


def test_a_crlf_split_across_reads_is_one_line_break():
    one = (_body(DIPOLE) + "NX\n").replace("\n", "\r\n").encode()
    writes, _ = _streamed([one[:-1], b"\n" + one, b""])
    together, _ = _streamed([one + one])
    assert writes == together


def test_nothing_on_stdin_prints_nothing():
    assert _streamed([]) == ([], ["read"])


def _launch(args: list[str], data: bytes) -> subprocess.CompletedProcess:
    code = (
        "import sys, momwire_eznec_client as c; "
        "sys.argv[0] = 'momwire-nec5-razor-2p'; sys.exit(c.main())"
    )
    return subprocess.run(
        [sys.executable, "-c", code, *args],
        input=data,
        capture_output=True,
        check=False,
        timeout=300,
    )


@pytest.mark.integration
def test_the_launcher_streams_through_its_own_stdin_and_stdout():
    one = (_body(DIPOLE) + "NX\n").encode()
    r = _launch(["-", "-"], one + one)
    assert r.returncode == 0
    want, _ = _streamed([one + one])
    assert r.stdout == b"".join(want)
