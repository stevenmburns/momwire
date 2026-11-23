"""The NEC-2 launchers end a run with ``RUN TIME =`` -- momwire#2133.

4nec2 rejects a NEC-2 engine's printout that has no ``RUN TIME =`` line
("Check output"), so before this every run of ``momwire-nec2c*`` in 4nec2's
NEC-2 slot read as failed, right answer and all. The line is the one 4nec2's
bundled nec2dxs writes (its captures, 2026-11-23): the ``EN`` echo, one blank
line, then `` RUN TIME =`` and the run's wall-clock seconds in ``%10.3f``.

Once per program run, as nec2dxs and the NEC-5 slot have it: an ``NX``-closed
deck carries none and the ``EN`` that ends the run carries the one. So
SimNEC's resident stream, ``NX``-framed and never sent an ``EN``, prints
exactly what it printed before. A refusal ends without one, as NEC-2's
engines and the NEC-5 slot's refusal printout do.
"""

from __future__ import annotations

import io
import itertools
import re
import types

import pytest

from momwire.dropin.nec2c import _portal, engine_scope, main

_DECK = (
    "CE run time\n"
    "GW 1 11 0 -2.5 0 0 2.5 0 .001\n"
    "GE 0\n"
    "EX 0 1 6 0 1\n"
    "FR 0 1 0 0 30\n"
    "XQ\n"
)
_REFUSED = (  # RP mode 1, the surface wave: refused by name
    "CE refused\nGW 1 9 0 0 -2.5 0 0 2.5 .001\nGE 0\nEX 0 1 5 0 1 0\n"
    "FR 0 1 0 0 30\nRP 1 1 1 1000 0 0 0 0 1000\n"
)
_RUN_TIME = re.compile(r"^ RUN TIME =[ 0-9]{5}\d\.\d{3}$")


def _stdout(text: str, argv=()) -> str:
    out = io.StringIO()
    with engine_scope():
        main(list(argv), stdin=io.StringIO(text), stdout=out, prog="momwire-nec2c")
    return out.getvalue()


def _run_times(printout: str) -> list[int]:
    lines = printout.splitlines()
    return [i for i, ln in enumerate(lines) if "RUN TIME" in ln]


def _assert_closes_on_en(printout: str) -> None:
    lines = printout.splitlines()
    assert printout.endswith("\n")
    assert _RUN_TIME.match(lines[-1]), lines[-1]
    assert lines[-2] == ""
    assert lines[-3].startswith("  DATA CARD No:")
    assert lines[-3].split()[4] == "EN"


def test_a_deck_ended_by_en_closes_with_one_run_time():
    printout = _stdout(_DECK + "EN\n")
    assert len(_run_times(printout)) == 1
    _assert_closes_on_en(printout)


def test_a_deck_left_open_at_end_of_input_closes_as_en_does():
    printout = _stdout(_DECK)
    assert len(_run_times(printout)) == 1
    _assert_closes_on_en(printout)


def test_an_nx_chain_carries_one_run_time_after_its_en():
    printout = _stdout(_DECK + "NX\n" + _DECK + "NX\n" + _DECK + "EN\n")
    lines = printout.splitlines()
    (at,) = _run_times(printout)
    assert at == len(lines) - 1
    nx = [
        i
        for i, ln in enumerate(lines)
        if ln.startswith("  DATA CARD No:") and ln.split()[4] == "NX"
    ]
    assert len(nx) == 2 and all(i < at for i in nx)
    _assert_closes_on_en(printout)


def test_simnecs_resident_stream_never_sees_one():
    """NX-framed decks and then the end of stdin, as NEC2Daemon sends them:
    no EN arrives, so the printout is what it always was."""
    printout = _stdout(_DECK + "NX\n" + _DECK + "NX\n")
    assert _run_times(printout) == []
    assert printout.rstrip("\n").splitlines()[-1].startswith("VERSION:")


def test_the_file_form_writes_it_crlf(tmp_path):
    deck = tmp_path / "d.nec"
    deck.write_text(_DECK + "EN\n")
    out = tmp_path / "d.out"
    _stdout("", argv=[str(deck), str(out)])
    raw = out.read_bytes()
    assert raw.endswith(b"\r\n\r\n RUN TIME =" + raw[-12:-2] + b"\r\n")
    _assert_closes_on_en(raw.decode("latin-1").replace("\r\n", "\n"))


def test_a_refused_deck_ends_without_one():
    printout = _stdout(_REFUSED + "EN\n")
    assert "ERROR: " in printout
    assert _run_times(printout) == []


def test_the_seconds_are_the_runs_wall_clock(monkeypatch):
    """Counted from the start of the run, :func:`_portal.resident_loop`."""
    clock = itertools.chain([100.0], itertools.repeat(102.5))
    monkeypatch.setattr(
        _portal, "time", types.SimpleNamespace(perf_counter=lambda: next(clock))
    )
    printout = _stdout(_DECK + "EN\n")
    assert printout.splitlines()[-1] == " RUN TIME =     2.500"


@pytest.mark.parametrize("terminator", ["NX", "EN"])
def test_deck_frame_alone(terminator):
    """A frame with no run around it (:func:`_portal.deck_frame` called
    directly) counts its own seconds, and only EN closes with the line."""
    out, _err = _portal.deck_frame(_DECK, terminator=terminator)
    has = any(ln.startswith(" RUN TIME =") for ln in out)
    assert has == (terminator == "EN")
