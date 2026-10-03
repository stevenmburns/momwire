"""scripts/test_durations.py's aggregation, offline, on fixture JSONL.

The fixture (tests/fixtures/test_durations/) is laid out the way
`gh run download -D <run_id>` leaves it: one directory per run, one per
artifact inside it. It encodes the cases the release step relies on:

* test_big   — median 6.5 s over 5 Linux runs, one 30 s outlier: FLAGGED.
* test_noisy — one 19 s spike, median 2.5 s: NOT flagged. A single run's wall
               time is exactly what this script exists not to trust.
* test_slow_marked — 40 s but already in the slow lane: never flagged.
* test_flaky — a failed outcome at 99 s: not timed at all.
* Windows build_wheels — test_big at 9/11 s, but only 2 runs: below the
               default --min-runs 3, so flagged only when that is lowered.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "test_durations.py"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "test_durations"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists(), reason="run from a source checkout (scripts/ absent)"
)


@pytest.fixture(scope="module")
def td():
    spec = importlib.util.spec_from_file_location("durations_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Registered first: a dataclass resolves its module's string annotations
    # through sys.modules, and an unregistered module is not there.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def groups(td):
    return td.aggregate(td.iter_records([FIXTURE]))


def test_percentile_matches_numpys_linear_default(td):
    xs = [2.0, 3.0, 19.0, 2.5, 2.2]
    assert td.percentile(xs, 50) == 2.5
    assert td.percentile(xs, 90) == pytest.approx(12.6)
    assert td.percentile([4.0], 90) == 4.0


def test_groups_are_job_by_os_and_failures_are_not_timed(groups):
    assert set(groups) == {("test", "Linux"), ("build_wheels", "Windows")}
    linux = groups[("test", "Linux")]
    assert "tests/test_a.py::test_flaky" not in linux
    big = linux["tests/test_a.py::test_big"]
    assert len(big.samples) == 5 and len(big.runs) == 5


def test_the_median_flags_and_a_spike_does_not(td, groups):
    c = td.candidates(groups, budget=5.0, min_runs=3)
    assert [(job, os_, nodeid) for _, job, os_, nodeid, _ in c] == [
        ("test", "Linux", "tests/test_a.py::test_big")
    ]
    assert c[0][0] == 6.5


def test_min_runs_and_budget_are_knobs(td, groups):
    c = td.candidates(groups, budget=5.0, min_runs=2)
    assert [(job, med) for med, job, *_ in c] == [
        ("build_wheels", 10.0),
        ("test", 6.5),
    ]
    assert td.candidates(groups, budget=7.0, min_runs=3) == []


def test_the_report_names_the_candidate(td, groups):
    out = td.report(groups, budget=5.0, top=10, min_runs=3)
    assert "tests/test_a.py::test_big  <-- slow-lane candidate" in out
    assert "test_slow_marked  [slow]" in out
    assert "Slow-lane candidates (default lane, median > 5s, >= 3 run(s)): 1" in out


def test_main_reads_a_local_directory(td, capsys, tmp_path):
    assert td.main(["--dir", str(FIXTURE)]) == 0
    assert "slow-lane candidate" in capsys.readouterr().out
    assert td.main(["--dir", str(tmp_path)]) == 1
