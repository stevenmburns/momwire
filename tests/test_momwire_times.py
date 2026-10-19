"""MOMWIRE_TIMES=1: per-stage timing on the Python NEC-5 path.

Off (the default, and every other test's: conftest pops it) the printout's
MATRIX TIMING and RUN TIME lines print zeros and nothing reaches stderr; the
byte gates hold the rest. On, those lines carry measured seconds, every other
byte of the printout is the same, and each answer writes one ``times:`` line
whose cells are non-negative and add up to its total
(`tests/times_check.py`). The standalone executables' half is in
`test_standalone_nopython.py` / `test_standalone_exe.py`.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pytest

import momwire_serve_client as _mech
from momwire import _times
from momwire.eznec import _shell

import times_check as tc

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DIPOLE = FIXTURES / "razor_printout_3a" / "node_gap_two_sources.nec"
BURIED = FIXTURES / "momwire_times" / "buried_radials.nec"
MULTI = FIXTURES / "eznec_multi_run_1237" / "two-port.nec"


def _stepped(n: int) -> str:
    """The dipole deck stepping its FR over ``n`` frequencies."""
    return DIPOLE.read_text().replace("FR 0,1,0,0,14.2", f"FR 0,{n},0,0,14.0,0.2")


def _one_shot(deck: Path, tmp_path: Path, capsys, *, basis: str = "razor-2p"):
    out = tmp_path / f"{deck.stem}.out"
    assert _shell.main([str(deck), str(out)], basis=basis) == 0
    return out.read_text(encoding="latin-1"), capsys.readouterr().err


def test_off_prints_zeros_and_writes_no_line(tmp_path, capsys):
    printout, err = _one_shot(DIPOLE, tmp_path, capsys)
    fills, runs = tc.printout_timing(printout)
    assert fills == [(0.0, 0.0)] and runs == [0.0]
    assert "times:" not in err


def test_a_stage_is_free_when_off():
    # one shared null context: no clock read, nothing allocated
    assert _times.stage("fill") is _times.stage("factor")
    assert _times.current() is None


@pytest.mark.parametrize("basis", ["razor-2p", "bspline"])
@pytest.mark.parametrize("deck", [DIPOLE, BURIED], ids=["dipole", "buried"])
def test_on_fills_the_timing_lines_and_moves_nothing_else(
    deck, basis, tmp_path, capsys, monkeypatch
):
    off, _ = _one_shot(deck, tmp_path, capsys, basis=basis)
    monkeypatch.setenv("MOMWIRE_TIMES", "1")
    # as this process's first answer: its line carries the momwire import
    monkeypatch.setattr(_shell, "_TIMES_PRE_TAKEN", False)
    on, err = _one_shot(deck, tmp_path, capsys, basis=basis)
    assert tc.zero_timing(on) == off
    (line,) = tc.times_lines(err)
    assert line["route"] == "python" and line["runs"] == 1
    assert line["fill_ms"] > 0.0 and line["load_ms"] > 0.0
    ((fill, factor),), (run_time,) = tc.printout_timing(on)
    # the printout's cells are the stage's own milliseconds, rounded
    assert abs(fill * 1e3 - line["fill_ms"]) <= 0.5 + 1e-9
    assert factor * 1e3 <= line["factor_ms"] + 0.5
    assert fill + factor <= run_time + 1e-3
    assert run_time * 1e3 <= line["total_ms"] + 1.0


def test_a_stepping_fr_times_each_frequency(tmp_path, capsys, monkeypatch):
    deck = tmp_path / "stepped.nec"
    deck.write_text(_stepped(3))
    off, _ = _one_shot(deck, tmp_path, capsys)
    monkeypatch.setenv("MOMWIRE_TIMES", "1")
    on, err = _one_shot(deck, tmp_path, capsys)
    assert tc.zero_timing(on) == off
    (line,) = tc.times_lines(err)
    fills, runs = tc.printout_timing(on)
    assert len(fills) == line["runs"] == 3 and len(runs) == 1
    assert sum(f for f, _ in fills) * 1e3 == pytest.approx(line["fill_ms"], abs=1.5)


def test_a_multi_run_deck_times_its_one_solve(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("MOMWIRE_TIMES", "1")
    on, err = _one_shot(MULTI, tmp_path, capsys)
    (line,) = tc.times_lines(err)
    assert line["runs"] == 2
    ((fill, _factor),), _ = tc.printout_timing(on)
    assert fill > 0.0


def test_the_resident_answer_frames_its_line_ahead(monkeypatch):
    body = DIPOLE.read_text()
    answer = _shell.seam(basis="razor-2p").answer
    plain, _ = answer(body, "")
    timed, _ = answer(_mech.TIMES_REQUEST.decode("latin-1") + body, "")
    # the wire's text wrapper writes \n as CRLF: read it as the client does
    raw = timed.replace("\n", "\r\n").encode("latin-1")
    line, rest = _mech.split_times(raw)
    assert line is not None
    (cells,) = tc.times_lines(line)
    assert cells["route"] == "resident" and cells["runs"] == 1
    assert tc.zero_timing(rest.decode("latin-1").replace("\r\n", "\n")) == plain
    # an untimed request is answered exactly as before: no frame
    assert _mech.split_times(plain.encode("latin-1"))[0] is None


def test_the_launcher_reports_the_engines_line_with_its_own_wall(capsys):
    from momwire_eznec_client import _report_times

    served = _mech.times_line(
        "resident",
        {"seam": 1.0, "fill": 20.0, "factor": 2.0},
        total_ms=25.0,
        threads=4,
        runs=1,
        counts={"built": 1},
    )
    now = time.perf_counter()
    _report_times(served.encode("latin-1"), now - 0.05, now)
    (cells,) = tc.times_lines(capsys.readouterr().err)
    assert cells["fill_ms"] == 20.0 and cells["threads"] == 4
    assert cells["ground_built"] == 1 and cells["total_ms"] >= 50.0


def test_a_stream_writes_a_line_per_structure(monkeypatch, capsys):
    monkeypatch.setenv("MOMWIRE_TIMES", "1")
    data = DIPOLE.read_bytes()
    one = data[: data.rindex(b"EN")] + b"NX\n"
    chunks = [one + one, b""]
    out = io.BytesIO()
    _shell.stream(lambda: chunks.pop(0), out.write, basis="razor-2p")
    lines = tc.times_lines(capsys.readouterr().err)
    assert [c["route"] for c in lines] == ["stream", "stream"]
    fills, runs = tc.printout_timing(out.getvalue())
    assert len(fills) == 2 and runs == []  # an NX-closed structure has no RUN TIME


@pytest.mark.parametrize("value", ["", "0", "off", "false", "no", "OFF"])
def test_the_switch_reads_off(value):
    assert not _mech.times_requested({"MOMWIRE_TIMES": value})


@pytest.mark.parametrize("value", ["1", "yes", "on"])
def test_the_switch_reads_on(value):
    assert _mech.times_requested({"MOMWIRE_TIMES": value})
