#!/usr/bin/env python3
"""Workflow-efficiency snapshot for momwire's CI. Re-run at each release.

Baseline for the 2026-10-03 change to CI timing methodology (CI wall time no
longer gates; see CLAUDE.md, "Test duration and performance"). Run it at
release prep over the window since the previous release and compare with that
release's numbers:

    python scripts/ci_metrics.py [--days 30] [--repo stevenmburns/momwire]

Reports, over the window:
  - merged PRs, median/p90 open->merge lead time;
  - CI + wheels workflow runs per merged PR (by head branch: push + dispatch);
  - failed runs whose failed-job log shows the 20 s HARD ceiling FAILING the
    run (by construction zero once the ceiling only reports — kept so the
    before/after comparison reads off one script);
  - median wall time of the default-lane `test` job on main (hygiene).

Baseline, 2026-10-03, 30 days: 257 merged PRs; lead time median 0.2 h, p90
2.7 h; runs/PR median 1, p90 3; 4 HARD-ceiling failures of 31 failed runs;
default-lane `test` job median 11.8 min (p90 12.9, n=40).

The p90 here is nearest-rank (unlike scripts/test_durations.py's linear
interpolation), kept as it was so later runs compare with the baseline.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
from datetime import datetime, timedelta, timezone

# What a run that the hard ceiling FAILED prints. Deliberately not the bare
# "HARD CEILING": since 2026-10-03 every run with an offender tags it
# "<-- OVER HARD CEILING" without failing, so a run that failed for another
# reason would otherwise be miscounted. The first pattern is the pre-change
# hook's verdict line; the second is the local-only opt-in's (CI never sets
# MOMWIRE_ENFORCE_TIME_BUDGET).
_CEILING_FAILED = re.compile(
    r"FAILING: \d+ test\(s\) over the .*HARD ceiling"
    r"|MOMWIRE_ENFORCE_TIME_BUDGET=\w+: this run FAILS"
)


def gh(*args: str) -> str:
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return out.stdout


def ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, default=30, help="window (default 30)")
    ap.add_argument("--repo", default="stevenmburns/momwire")
    a = ap.parse_args(argv)
    since = datetime.now(timezone.utc) - timedelta(days=a.days)

    prs = json.loads(
        gh(
            "pr", "list", "-R", a.repo, "--state", "merged", "--limit", "300",
            "--json", "number,headRefName,createdAt,mergedAt",
        )
    )  # fmt: skip
    prs = [p for p in prs if p["mergedAt"] and ts(p["mergedAt"]) >= since]
    lead_h = [
        (ts(p["mergedAt"]) - ts(p["createdAt"])).total_seconds() / 3600 for p in prs
    ]

    raw = gh(
        "api", "--paginate",
        f"repos/{a.repo}/actions/runs?per_page=100"
        f"&created=>={since.date().isoformat()}",
        "--jq",
        ".workflow_runs[] | {id, name, head_branch, event, conclusion, created_at}",
    )  # fmt: skip
    runs = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ci = [r for r in runs if r["name"] in ("CI", "wheels")]
    by_branch: dict[str, list[dict]] = {}
    for r in ci:
        by_branch.setdefault(r["head_branch"], []).append(r)
    per_pr = [len(by_branch.get(p["headRefName"], [])) for p in prs]

    ceiling = []
    for r in ci:
        if r["conclusion"] != "failure":
            continue
        try:
            log = gh("run", "view", str(r["id"]), "-R", a.repo, "--log-failed")
        except subprocess.CalledProcessError:
            continue
        if _CEILING_FAILED.search(log):
            ceiling.append(r)

    main_ci = [
        r
        for r in ci
        if r["name"] == "CI"
        and r["head_branch"] == "main"
        and r["conclusion"] == "success"
    ]
    test_job = []
    for r in main_ci[:40]:
        jobs = json.loads(
            gh(
                "api", f"repos/{a.repo}/actions/runs/{r['id']}/jobs",
                "--jq", "[.jobs[] | {name, started_at, completed_at}]",
            )
        )  # fmt: skip
        for j in jobs:
            if j["name"] == "test" and j["started_at"] and j["completed_at"]:
                dt = ts(j["completed_at"]) - ts(j["started_at"])
                test_job.append(dt.total_seconds() / 60)

    failed = [r for r in ci if r["conclusion"] == "failure"]
    print(f"window: last {a.days} days (since {since.date()}), repo {a.repo}")
    print(f"merged PRs: {len(prs)}")
    if lead_h:
        print(
            f"lead time open->merge (h): median {statistics.median(lead_h):.1f}, "
            f"p90 {pct(lead_h, 0.9):.1f}"
        )
    else:
        print("lead time: n/a")
    if per_pr:
        print(
            f"CI+wheels runs per merged PR: median {statistics.median(per_pr):.1f}, "
            f"p90 {pct(per_pr, 0.9):.0f}, max {max(per_pr)}"
        )
    else:
        print("runs/PR: n/a")
    print(
        f"CI+wheels runs total: {len(ci)}, failed: {len(failed)}, "
        f"failed on the 20 s HARD ceiling: {len(ceiling)}"
    )
    for r in ceiling:
        print(
            f"  ceiling failure: run {r['id']} {r['name']} {r['head_branch']} "
            f"{r['created_at'][:10]}"
        )
    if test_job:
        print(
            f"default-lane `test` job on main (min): median "
            f"{statistics.median(test_job):.1f}, p90 {pct(test_job, 0.9):.1f}, "
            f"n={len(test_job)}"
        )
    else:
        print("test job: n/a")


if __name__ == "__main__":
    main()
