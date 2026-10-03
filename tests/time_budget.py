"""Test time-budget REPORT and per-test duration log (a pytest plugin).

Registered by `tests/conftest.py`; kept in its own module, with no momwire
import, so `tests/test_time_budget_hook.py` can load it into a throwaway
pytest session (`-p time_budget`) and check what it does.

Policy (Steve, 2026-10-03): CI wall time NEVER gates
=====================================================
Ported from antennaknobs' #393 hook, the guardrail exists because the PR lane
decides how many edit->signal turns an hour this repo supports, and nothing
else kept new slow tests out of it (#692's marker pass got the lane to 2:31 by
moving 403 integration tests out; the count held only because authors
remembered). It used to have two thresholds: a 5 s ceiling that REPORTED and a
20 s HARD ceiling that FAILED the run.

The hard ceiling is now a report too, by default. The instrument is noisier
than the effect it was gating:

* GitHub's runners spread ~3x on the SAME code. On windows-2025 the v0.70.0
  tag's own run measured two enrichment tests at 5.1-6.7 s; a re-run of that
  tag on 2026-10-03 measured 7.5-12.5 s; main's full-matrix run reached 19.3 s.
  (The 20 s figure was originally sized from a 1.55x spread on Linux — the
  spread was bigger than that, and bigger still on Windows.)
* A gate whose threshold sits inside the instrument's spread fails on noise,
  and a release tag's full matrix runs the default lane once per interpreter,
  so it rolls the dice six times per OS. A gate that reddens when nothing
  changed is a gate people learn to ignore — and a red release tag is worse.

So the two concerns are now handled by instruments fit for each:

(A) Test-cost hygiene — keeping the default lane fast. A long test may land.
    Every run REPORTS unmarked tests over the ceilings (below), and every CI
    pytest step writes each test's call duration to a JSONL file that is
    uploaded as an artifact (see `MOMWIRE_DURATIONS_OUT`). At release time
    `scripts/test_durations.py` takes the MEDIAN over the last ~25 main runs,
    per OS and lane, and a default-lane test whose median exceeds the budget
    moves to the slow lane in its own PR before the tag. A median over many
    runs is robust to the runner spread that a single run's wall time is not.

(B) Speed regressions — judged ONLY by paired, interleaved, same-machine
    base-vs-change runs and the pre-tag sweep on a quiet box (Skylake; the
    Windows box for Windows). Never by CI wall time.

PR #1307 (two bspline enrichment tests marked `slow` for Windows variance
against the old 20 s gate) was opened before this rule. Its marks stand —
those tests are legitimately slow on Windows — unless the median data says
otherwise; revisit them with `scripts/test_durations.py`, not by hand.

The thresholds, both REPORT-only by default
-------------------------------------------
  CEILING      (5 s)   an unmarked test over it is named in the summary.
  HARD_CEILING (20 s)  the same test is additionally tagged
                       "<-- OVER HARD CEILING": worth a look, never fatal.

Environment:
  MOMWIRE_TIME_BUDGET_CEILING_S       soft ceiling, seconds (default 5)
  MOMWIRE_TIME_BUDGET_HARD_CEILING_S  hard ceiling, seconds (default 20)
  MOMWIRE_ENFORCE_TIME_BUDGET         unset/""/"0": report only (DEFAULT).
                                      "hard": fail when any unmarked test is
                                      over the HARD ceiling. Any other value
                                      ("1"): fail over the SOFT ceiling.
                                      For local use on a quiet box — CI must
                                      never set it.
  MOMWIRE_DURATIONS_OUT               path; when set, every test's CALL-phase
                                      duration is APPENDED to it as JSONL
                                      (one session may share a file with
                                      another, e.g. one per interpreter in a
                                      cibuildwheel run).

The four exempt markers are the four lanes that are not the edit loop.
`integration` is exempt for a reason worth stating: it is the one marker here
that is NOT about duration — 306 of #692's 403 integration tests run under
3 s. A test can be exempt because it is slow OR because it is not an edit-loop
guard, and those are different claims.
"""

from __future__ import annotations

import json
import os
import platform
import sys

import pytest

TIME_BUDGET_CEILING_S = float(os.environ.get("MOMWIRE_TIME_BUDGET_CEILING_S", "5.0"))
TIME_BUDGET_HARD_CEILING_S = float(
    os.environ.get("MOMWIRE_TIME_BUDGET_HARD_CEILING_S", "20.0")
)
EXEMPT_MARKERS = ("slow", "memgate", "crossgate", "integration")

_offenders: list[tuple[str, float]] = []
_durations: list[dict] = []
_state = {"worker": False}


def _enforce_mode() -> str:
    """'' (report only), 'hard' or 'soft'."""
    v = os.environ.get("MOMWIRE_ENFORCE_TIME_BUDGET", "").strip().lower()
    if v in ("", "0", "false", "no", "off"):
        return ""
    return "hard" if v == "hard" else "soft"


def pytest_configure(config):
    # Only xdist workers carry `workerinput`. Under xdist every report fires
    # on its worker AND on the controller (which is handed all of them), so
    # recording on the controller alone sees the whole run exactly once.
    _state["worker"] = hasattr(config, "workerinput")


def pytest_runtest_logreport(report):
    if report.when != "call" or _state["worker"]:
        return
    lanes = [m for m in EXEMPT_MARKERS if m in report.keywords]
    if os.environ.get("MOMWIRE_DURATIONS_OUT"):
        _durations.append(
            {
                "nodeid": report.nodeid,
                "duration": round(report.duration, 4),
                "outcome": report.outcome,
                "markers": lanes,
            }
        )
    # Collect above the LOWER of the two ceilings: keyed to the soft one
    # alone, a tuned HARD < SOFT would never see its offenders.
    if lanes or report.duration <= min(
        TIME_BUDGET_CEILING_S, TIME_BUDGET_HARD_CEILING_S
    ):
        return
    _offenders.append((report.nodeid, report.duration))


def _over_hard() -> list[tuple[str, float]]:
    return [x for x in _offenders if x[1] > TIME_BUDGET_HARD_CEILING_S]


def pytest_terminal_summary(terminalreporter):
    if not _offenders:
        return
    tr = terminalreporter
    tr.section("test time-budget report", sep="!", yellow=True, bold=True)
    tr.line(
        f"{len(_offenders)} unmarked test(s) over the "
        f"{TIME_BUDGET_CEILING_S:g}s ceiling on THIS run:"
    )
    for nodeid, dur in sorted(_offenders, key=lambda x: -x[1]):
        over_hard = " <-- OVER HARD CEILING" if dur > TIME_BUDGET_HARD_CEILING_S else ""
        tr.line(f"  {dur:6.2f}s  {nodeid}{over_hard}")
    mode = _enforce_mode()
    if mode:
        tr.line(f"MOMWIRE_ENFORCE_TIME_BUDGET={mode}: this run FAILS on the above.")
    else:
        tr.line(
            "Report only: one run's wall time is not evidence (CI runners spread "
            "~3x on the same code). Lane moves are decided at release by the "
            "MEDIAN over many main runs — scripts/test_durations.py."
        )
    tr.line(
        "If it is genuinely slow: make it faster, or mark it — `slow` (push "
        "lane), `integration` (crosses a process/socket/CLI/printout seam), "
        "`crossgate`/`memgate` (certification). See the pyproject markers."
    )


def _write_durations() -> None:
    path = os.environ.get("MOMWIRE_DURATIONS_OUT")
    if not path or not _durations:
        return
    run = {
        "python": platform.python_version(),
        "platform": sys.platform,
        # Set by GitHub Actions (and inherited by cibuildwheel's host-side
        # test runs); absent locally, which the reader tolerates.
        "job": os.environ.get("GITHUB_JOB", ""),
        "runner_os": os.environ.get("RUNNER_OS", ""),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "sha": os.environ.get("GITHUB_SHA", ""),
    }
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in _durations:
            f.write(json.dumps({**run, **rec}) + "\n")


def pytest_sessionfinish(session, exitstatus):
    # Controller (or serial) only: a worker's exitstatus does not reach the
    # caller, and the controller is where the whole run's reports are.
    if _state["worker"]:
        return
    _write_durations()
    mode = _enforce_mode()
    failing = _over_hard() if mode == "hard" else _offenders if mode else []
    if failing and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
