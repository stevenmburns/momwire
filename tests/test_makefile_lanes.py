"""The Makefile's lanes and ci.yml's commands are one contract (momwire#715).

The Makefile exists so each CI lane is a command you can type instead of an
incantation you retype from memory — momwire runs SIX distinct marker
expressions, and any command selecting `memgate` needs `-n0` or it measures
peak RSS while sharing a box with xdist workers.

That only helps if "make test passed" and "the PR gate will pass" really mean
the same thing. This file CHECKS it, lane-to-job (not lane-to-anywhere: a
lane accidentally given a DIFFERENT job's command is drift too), in both
directions:

* every Makefile lane's pytest command appears verbatim in the ci.yml job it
  claims to mirror — so a lane cannot quietly drift into testing a different
  set, nor swap sets with another lane;
* every ci.yml job that runs pytest is mapped to a Makefile lane — so a new
  CI lane cannot be added without a local way to run it.

The #716 review's fail-open lesson shaped the parsers. ci.yml is YAML-parsed
(jobs -> steps -> run scalars, block scalars included), not line-grepped:
a pytest step rewritten as `run: |` or under `- name:` must not silently
leave the guarantee. (test_release_notes.py greps the same file by design —
its claim is about one literal line; this file's claim is structural, which
is the difference that justifies the PyYAML dependency, declared in the
[test] extra.) The Makefile side accepts any `$(PY)`/`$(PYTHON) -m` recipe
spelling, keeps EVERY pytest line of a multi-command recipe, and — the
fail-closed backstop — asserts the expected lane names all parsed: a lane
the parser stops seeing is itself a failure, never a silent skip.

Why pin rather than have ci.yml call `make`: the pytest line is the only
part that would move. The `if:` guards, the ccache CC/CXX env, and the
per-job install steps all stay in the workflow, so the DRY win is partial —
while `- run: pytest tests/ -m "..."` in the CI log is self-documenting when
a lane fails, and `make test` is not. Same idiom as antennaknobs'
`test_backend_roster.py`.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = ROOT / "Makefile"

# Guard on the SOURCE-CHECKOUT marker only: in a wheel neither file exists
# and skipping is right. ci.yml going missing from a checkout that has the
# Makefile is a broken contract and must FAIL below, not skip (the #716
# review: a relocation would have silently disabled this whole module).
pytestmark = pytest.mark.skipif(
    not MAKEFILE.exists(),
    reason="run from a source checkout (Makefile absent in a wheel)",
)

# The contract's spine: which ci.yml job each lane mirrors. A NEW ci.yml
# job that runs pytest must be added here (and given a lane) — the reverse
# test fails until it is.
LANE_TO_JOB = {
    "test": "test",
    "pynec": "test-pynec",
    "macos-set": "test-macos",
    "integration": "test-integration",
    "slow": "test-slow",
    "crossgate": "test-crossgate",
    "memgate": "test-memgate",
}

_PYTEST_LINE = re.compile(r"(?:^|\s)(pytest\s+.*?)\s*$")


def _ci_pytest_commands() -> dict[str, list[str]]:
    """job name -> every pytest command any of its run scalars execute."""
    doc = yaml.safe_load(CI.read_text())
    out: dict[str, list[str]] = {}
    for job_name, job in (doc.get("jobs") or {}).items():
        for step in job.get("steps") or []:
            run = step.get("run")
            if not run:
                continue
            for line in str(run).splitlines():
                line = line.strip()
                if line.startswith("#"):
                    continue
                # `python -m pytest ...` counts as a pytest invocation too.
                line = re.sub(r"^\S*python[0-9.]*\s+-m\s+", "", line)
                m = _PYTEST_LINE.search(line)
                if m:
                    out.setdefault(job_name, []).append(m.group(1))
    return out


def _makefile_lanes() -> dict[str, list[str]]:
    """target -> every pytest command its recipe runs.

    Accepts `$(PY) pytest ...` and `$(PYTHON) -m pytest ...`, an optional
    `@` echo suppressor, and env-var prefixes — the equivalent spellings the
    #716 review showed escaping the original single-spelling anchor.
    """
    lanes: dict[str, list[str]] = {}
    target = None
    recipe = re.compile(
        r"^\t@?(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)*"
        r"\$\((?:PY|PYTHON)\)(?:\s+-m)?\s+(pytest\s+.*?)\s*$"
    )
    for line in MAKEFILE.read_text().splitlines():
        if re.match(r"^[A-Za-z][\w.-]*:", line):
            target = line.split(":", 1)[0]
        m = recipe.match(line)
        if m and target:
            lanes.setdefault(target, []).append(m.group(1))
    return lanes


def test_the_expected_lanes_all_parse():
    """Fail-closed backstop: a lane the parser stops seeing is a failure.

    Without this, rewriting a recipe in a spelling the parser misses would
    silently exempt that lane from every check below."""
    lanes = _makefile_lanes()
    missing = set(LANE_TO_JOB) - set(lanes)
    assert not missing, (
        f"Makefile lanes not seen by the parser: {sorted(missing)} — either "
        "the lane was removed (update LANE_TO_JOB deliberately) or its "
        "recipe uses a spelling _makefile_lanes() must learn."
    )


def test_every_makefile_lane_matches_its_own_ci_job():
    ci, lanes = _ci_pytest_commands(), _makefile_lanes()
    assert ci, "parsed no pytest commands out of ci.yml — parser broken?"
    problems = []
    for lane, job in LANE_TO_JOB.items():
        for cmd in lanes.get(lane, []):
            if cmd not in ci.get(job, []):
                problems.append(
                    f"  make {lane}: {cmd!r}\n"
                    f"    ci.yml job {job!r} runs: {ci.get(job, [])!r}"
                )
    assert not problems, (
        "Makefile lane(s) no longer match THEIR OWN ci.yml job:\n"
        + "\n".join(problems)
        + "\n\nA lane that drifts tests a different set than the gate it "
        "claims to mirror. Fix the Makefile, or update both deliberately."
    )


def test_every_ci_pytest_job_is_mapped_and_covered():
    ci, lanes = _ci_pytest_commands(), _makefile_lanes()
    unmapped = set(ci) - set(LANE_TO_JOB.values())
    assert not unmapped, (
        f"ci.yml job(s) run pytest but map to no Makefile lane: "
        f"{sorted(unmapped)} — add a lane and a LANE_TO_JOB entry so the "
        "gate can be run locally before it gates a PR."
    )
    covered = {c for cmds in lanes.values() for c in cmds}
    missing = {c for cmds in ci.values() for c in cmds} - covered
    assert not missing, "ci.yml pytest command(s) with no Makefile lane:\n" + "\n".join(
        f"  {c}" for c in sorted(missing)
    )


def test_anything_selecting_memgate_runs_serial():
    """`-n0` is a property of the MARKER, not of one make target: peak-RSS
    certification cannot share a box with xdist workers, whichever lane or
    job selects it (the #716 review: a name-keyed guard misses a second
    memgate-selecting command)."""
    everything = [
        ("make " + lane, cmd)
        for lane, cmds in _makefile_lanes().items()
        for cmd in cmds
    ] + [
        ("ci.yml " + job, cmd)
        for job, cmds in _ci_pytest_commands().items()
        for cmd in cmds
    ]

    def selects_memgate(cmd: str) -> bool:
        return (
            re.search(r'-m\s+"?[^"]*\bmemgate\b', cmd) is not None
            and "not memgate" not in cmd
        )

    selecting = [(w, c) for w, c in everything if selects_memgate(c)]
    # Sanity that the detector is alive: the memgate lane and its ci.yml job
    # must both be seen selecting the marker, or this test guards nothing.
    assert len(selecting) >= 2, (
        f"memgate-marker detector matched {selecting!r} — parser or regex "
        "broke; this guard would pass vacuously."
    )
    bad = [f"  {w}: {c}" for w, c in selecting if "-n0" not in c]
    assert not bad, (
        "command(s) select the memgate marker without -n0 — they would "
        "inherit addopts' `-n auto` and measure RSS under xdist workers:\n"
        + "\n".join(bad)
    )
