"""The time-budget hook REPORTS and never fails by default (tests/time_budget.py).

Policy, 2026-10-03: CI wall time never gates — runners spread ~3x on the same
code, so the old 20 s hard ceiling failed on noise. These run the plugin in a
throwaway pytest session (its own rootdir, so neither this repo's conftest nor
its addopts apply) with the ceilings scaled down to fractions of a second, and
check the three things that matter: an over-HARD test leaves the exit code 0;
the local opt-in still fails; and the duration log is written once per test,
including under xdist, where every report reaches both worker and controller.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

_SLOW_TESTS = """
import time
import pytest

def test_over_hard():
    time.sleep(0.4)

@pytest.mark.slow
def test_marked_slow():
    time.sleep(0.4)

def test_fast():
    pass
"""


_FIXTURE_TESTS = """
import time
import pytest

@pytest.fixture
def unavailable():
    time.sleep(0.3)
    pytest.skip("not here")

def test_skips_in_setup(unavailable):
    pass

@pytest.fixture(scope="module")
def built():
    time.sleep(0.4)
    yield
    time.sleep(0.2)

def test_pays_the_module_fixture(built):
    pass

def test_rides_on_it(built):
    pass
"""


def _run(
    tmp_path: Path, *args: str, source: str = _SLOW_TESTS, **env: str
) -> subprocess.CompletedProcess:
    (tmp_path / "test_sample.py").write_text(source)
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    slow: s\n    memgate: m\n"
        "    crossgate: c\n    integration: i\n"
    )
    e = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("MOMWIRE_TIME_BUDGET", "MOMWIRE_ENFORCE", "MOMWIRE_DUR"))
        and k not in ("PYTEST_ADDOPTS", "GITHUB_JOB", "RUNNER_OS")
    }
    e.update(
        PYTHONPATH=os.pathsep.join([str(HERE), e.get("PYTHONPATH", "")]),
        MOMWIRE_TIME_BUDGET_CEILING_S="0.1",
        MOMWIRE_TIME_BUDGET_HARD_CEILING_S="0.2",
        **env,
    )
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "time_budget", "-p", "no:cacheprovider"]
        + list(args)
        + [str(tmp_path)],
        cwd=tmp_path,
        env=e,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_over_the_hard_ceiling_is_reported_not_failed(tmp_path):
    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "test time-budget report" in r.stdout
    assert "test_over_hard <-- OVER HARD CEILING" in r.stdout
    # A marked test is in its own lane and is never named.
    assert "test_marked_slow" not in r.stdout.split("test time-budget report")[1]


@pytest.mark.parametrize("mode", ["hard", "1"])
def test_the_local_opt_in_still_fails(tmp_path, mode):
    r = _run(tmp_path, MOMWIRE_ENFORCE_TIME_BUDGET=mode)
    assert r.returncode == 1, r.stdout + r.stderr
    assert f"MOMWIRE_ENFORCE_TIME_BUDGET={'hard' if mode == 'hard' else 'soft'}" in (
        r.stdout
    )


def test_enforce_zero_means_off(tmp_path):
    assert _run(tmp_path, MOMWIRE_ENFORCE_TIME_BUDGET="0").returncode == 0


@pytest.mark.parametrize("xdist", [False, True], ids=["serial", "xdist"])
def test_durations_are_logged_once_per_test(tmp_path, xdist):
    if xdist:
        pytest.importorskip("xdist")
    out = tmp_path / "out" / "d.jsonl"
    args = ("-n", "2") if xdist else ()
    r = _run(tmp_path, *args, MOMWIRE_DURATIONS_OUT=str(out), GITHUB_JOB="test")
    assert r.returncode == 0, r.stdout + r.stderr
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    by = {rec["nodeid"].split("::")[-1]: rec for rec in recs}
    assert len(recs) == 3 and set(by) == {
        "test_over_hard",
        "test_marked_slow",
        "test_fast",
    }
    assert by["test_marked_slow"]["markers"] == ["slow"]
    assert by["test_over_hard"]["markers"] == []
    assert by["test_over_hard"]["duration"] >= 0.3
    assert by["test_fast"]["job"] == "test"
    assert by["test_fast"]["python"] and by["test_fast"]["platform"] == sys.platform
    # Appends: a second session (one per interpreter in a wheel build) adds.
    _run(tmp_path, MOMWIRE_DURATIONS_OUT=str(out))
    assert len(out.read_text().splitlines()) == 6


@pytest.mark.parametrize("xdist", [False, True], ids=["serial", "xdist"])
def test_setup_and_teardown_are_logged_beside_the_call(tmp_path, xdist):
    """A module fixture's cost lands in the SETUP of the first test that
    asks for it and the TEARDOWN of the last, never in a call; a call-only
    log hid ten minutes of per-worker C++ builds that way (2026-10-28).
    `duration` stays the call phase, so the median script reads it as before."""
    if xdist:
        pytest.importorskip("xdist")
    out = tmp_path / "out" / "d.jsonl"
    # one worker under xdist: which test pays a module fixture is otherwise
    # the scheduler's choice
    args = ("-n", "1") if xdist else ()
    r = _run(tmp_path, *args, source=_FIXTURE_TESTS, MOMWIRE_DURATIONS_OUT=str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    by = {rec["nodeid"].split("::")[-1]: rec for rec in recs}
    assert len(recs) == 3
    first, second = by["test_pays_the_module_fixture"], by["test_rides_on_it"]
    for rec in (first, second):
        assert rec["outcome"] == "passed"
        assert rec["duration"] < 0.2  # the call is empty
        assert isinstance(rec["setup"], float) and isinstance(rec["teardown"], float)
    assert first["setup"] >= 0.35 and second["setup"] < 0.2
    assert second["teardown"] >= 0.15 and first["teardown"] < 0.15
    # Skipped in setup: no call phase, so no call duration, and the setup
    # it spent is still on the record.
    skipped = by["test_skips_in_setup"]
    assert skipped["outcome"] == "skipped" and skipped["duration"] is None
    assert skipped["setup"] >= 0.25
