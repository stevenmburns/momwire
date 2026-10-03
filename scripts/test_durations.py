#!/usr/bin/env python3
"""Per-test duration medians over many CI runs: the default lane's hygiene check.

CI wall time never gates (tests/time_budget.py has the policy and the
measurements behind it): GitHub's runners spread ~3x on the same code, so one
run's 12 s is not evidence a test is slow. A MEDIAN over ~25 runs is. Every CI
pytest step uploads its per-test call durations as a `durations-*` artifact
(JSONL, written by tests/time_budget.py when MOMWIRE_DURATIONS_OUT is set);
this script pulls the last N successful main-branch runs' artifacts and
reports, per (job, OS) group, each test's median, p90 and sample count.

A test in the DEFAULT lane (none of the slow/memgate/crossgate/integration
markers) whose median exceeds the budget is flagged as a slow-lane candidate.
At release prep, move flagged tests to the slow lane in their own PR before
tagging (.claude/skills/release/SKILL.md).

    python scripts/test_durations.py                 # fetch via gh, last 25 runs
    python scripts/test_durations.py --runs 40 --budget 4
    python scripts/test_durations.py --dir some/dir  # offline: read *.jsonl under it

This measures cost hygiene only. A SPEED REGRESSION is judged by paired,
interleaved, same-machine base-vs-change runs and the pre-tag sweep on a quiet
box — never from these numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = "stevenmburns/momwire"
WORKFLOWS = ("ci.yml", "wheels.yml")
EXEMPT_MARKERS = ("slow", "memgate", "crossgate", "integration")
DEFAULT_CACHE = Path.home() / ".cache" / "momwire" / "test-durations"


@dataclass
class Stats:
    samples: list[float] = field(default_factory=list)
    runs: set[str] = field(default_factory=set)
    markers: set[str] = field(default_factory=set)


def percentile(xs: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100] (numpy's default)."""
    s = sorted(xs)
    if not s:
        raise ValueError("no samples")
    pos = (len(s) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def group_key(rec: dict) -> tuple[str, str]:
    os_name = rec.get("runner_os") or {
        "linux": "Linux",
        "darwin": "macOS",
        "win32": "Windows",
    }.get(rec.get("platform", ""), rec.get("platform", "?"))
    return (rec.get("job") or "local", os_name)


def aggregate(records) -> dict[tuple[str, str], dict[str, Stats]]:
    """(job, os) -> nodeid -> stats. Only passed tests are timed."""
    out: dict[tuple[str, str], dict[str, Stats]] = defaultdict(
        lambda: defaultdict(Stats)
    )
    for rec in records:
        if rec.get("outcome", "passed") != "passed":
            continue
        st = out[group_key(rec)][rec["nodeid"]]
        st.samples.append(float(rec["duration"]))
        st.runs.add(str(rec.get("run_id", "")))
        st.markers.update(rec.get("markers") or ())
    return out


def iter_records(roots):
    for p in sorted(f for r in roots for f in Path(r).rglob("*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    print(f"warning: {p}:{n}: not JSON, skipped", file=sys.stderr)


def candidates(groups, budget: float, min_runs: int):
    """Default-lane tests whose MEDIAN exceeds the budget, in any group."""
    out = []
    for (job, os_name), tests in groups.items():
        for nodeid, st in tests.items():
            if st.markers & set(EXEMPT_MARKERS) or len(st.runs) < min_runs:
                continue
            med = percentile(st.samples, 50)
            if med > budget:
                out.append((med, job, os_name, nodeid, st))
    return sorted(out, key=lambda x: (-x[0], x[3]))


def report(groups, budget: float, top: int, min_runs: int) -> str:
    lines = []
    for job, os_name in sorted(groups):
        tests = groups[(job, os_name)]
        n_runs = len(set().union(*(st.runs for st in tests.values())))
        rows = sorted(
            tests.items(), key=lambda kv: (-percentile(kv[1].samples, 50), kv[0])
        )
        lines.append(f"== {job} / {os_name}: {len(tests)} tests, {n_runs} run(s)")
        lines.append(f"  {'median':>7} {'p90':>7} {'n':>4} {'runs':>4}  test")
        for nodeid, st in rows[:top]:
            med = percentile(st.samples, 50)
            flag = ""
            if not (st.markers & set(EXEMPT_MARKERS)):
                if med > budget and len(st.runs) >= min_runs:
                    flag = "  <-- slow-lane candidate"
            else:
                flag = f"  [{','.join(sorted(st.markers))}]"
            lines.append(
                f"  {med:7.2f} {percentile(st.samples, 90):7.2f} "
                f"{len(st.samples):4d} {len(st.runs):4d}  {nodeid}{flag}"
            )
        if len(rows) > top:
            lines.append(f"  ... {len(rows) - top} more")
        lines.append("")
    cands = candidates(groups, budget, min_runs)
    lines.append(
        f"Slow-lane candidates (default lane, median > {budget:g}s, "
        f">= {min_runs} run(s)): {len(cands)}"
    )
    for med, job, os_name, nodeid, st in cands:
        lines.append(
            f"  {med:7.2f}s median ({len(st.runs)} runs)  {job}/{os_name}  {nodeid}"
        )
    return "\n".join(lines)


def _gh(*args: str) -> str:
    return subprocess.run(
        ["gh", *args], check=True, capture_output=True, text=True
    ).stdout


def fetch(repo: str, n_runs: int, cache: Path, workflows) -> list[Path]:
    """Download the durations artifacts of the last n successful main runs.

    Returns one directory per run. Only THESE runs are read, not everything
    the cache has accumulated, so the window really is the last n.
    """
    cache.mkdir(parents=True, exist_ok=True)
    dirs: list[Path] = []
    for wf in workflows:
        runs = json.loads(
            _gh(
                "run",
                "list",
                "-R",
                repo,
                "--workflow",
                wf,
                "--branch",
                "main",
                "--status",
                "success",
                "--limit",
                str(n_runs),
                "--json",
                "databaseId",
            )
        )
        for run in runs:
            dest = cache / str(run["databaseId"])
            if dest.exists():
                dirs.append(dest)
                continue
            try:
                _gh(
                    "run",
                    "download",
                    str(run["databaseId"]),
                    "-R",
                    repo,
                    "-p",
                    "durations-*",
                    "-D",
                    str(dest),
                )
            except subprocess.CalledProcessError as e:
                # Runs from before the artifacts existed have none to match;
                # not cached, so a transient failure is retried next time.
                print(
                    f"note: run {run['databaseId']} ({wf}): no durations "
                    f"artifact ({e.stderr.strip()[:80]})",
                    file=sys.stderr,
                )
                continue
            dirs.append(dest)
    return dirs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--dir", type=Path, help="read *.jsonl under this directory instead of fetching"
    )
    ap.add_argument(
        "--runs",
        type=int,
        default=25,
        help="successful main runs per workflow to fetch (default 25)",
    )
    ap.add_argument(
        "--budget",
        type=float,
        default=5.0,
        help="median budget in seconds for the default lane (default 5)",
    )
    ap.add_argument(
        "--min-runs",
        type=int,
        default=3,
        help="runs a test needs before it can be flagged (default 3)",
    )
    ap.add_argument(
        "--top", type=int, default=25, help="rows shown per group (default 25)"
    )
    ap.add_argument("--repo", default=REPO)
    ap.add_argument(
        "--workflow",
        action="append",
        help=f"workflow file(s) to fetch (default {', '.join(WORKFLOWS)})",
    )
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    a = ap.parse_args(argv)

    roots = (
        [a.dir] if a.dir else fetch(a.repo, a.runs, a.cache, a.workflow or WORKFLOWS)
    )
    groups = aggregate(iter_records(roots))
    if not groups:
        print(f"no duration records under {[str(r) for r in roots]}", file=sys.stderr)
        return 1
    print(report(groups, a.budget, a.top, a.min_runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
