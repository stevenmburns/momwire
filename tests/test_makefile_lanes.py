"""The Makefile's lanes and ci.yml's commands are one contract (momwire#715).

The Makefile exists so each CI lane is a command you can type instead of an
incantation you retype from memory — momwire runs SIX distinct marker
expressions, and `memgate` additionally needs `-n0` or it measures peak RSS
while sharing a box with xdist workers.

That only helps if "make test passed" and "the PR gate will pass" really mean
the same thing. Line-number comments in the Makefile assert that; this file
CHECKS it, in both directions:

* every Makefile lane runs a pytest command that appears verbatim in
  ci.yml — so a lane cannot quietly drift into testing a different set;
* every pytest command in ci.yml has a Makefile lane — so a new CI lane
  cannot be added without a local way to run it.

Why this rather than having ci.yml call `make`: the pytest line is the only
part that would move. The `if: needs.changes.outputs.code` guards, the
ccache CC/CXX env, and the per-job install steps all stay in the workflow, so
the DRY win is partial — while `- run: pytest tests/ -m "..."` in the CI log
is self-documenting when a lane fails, and `make test` is not. This is the
same idiom the repo already uses for cross-file contracts (antennaknobs'
`test_backend_roster.py` greps the frontend TS for `BACKEND_ORDER` rather
than making one generate the other).
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / ".github" / "workflows" / "ci.yml"
MAKEFILE = ROOT / "Makefile"

pytestmark = pytest.mark.skipif(
    not (CI.exists() and MAKEFILE.exists()),
    reason="run from a source checkout (ci.yml / Makefile absent in a wheel)",
)


def _ci_pytest_commands() -> set[str]:
    out = set()
    for line in CI.read_text().splitlines():
        m = re.search(r"- run:\s+(pytest\s+.*?)\s*$", line.strip())
        if m:
            out.add(m.group(1))
    return out


def _makefile_lanes() -> dict[str, str]:
    """target -> the pytest command its recipe runs."""
    lanes, target = {}, None
    for line in MAKEFILE.read_text().splitlines():
        if re.match(r"^[a-z][a-z0-9-]*:", line):
            target = line.split(":", 1)[0]
        m = re.match(r"^\t\$\(PY\)\s+(pytest\s+.*?)\s*$", line)
        if m and target:
            lanes[target] = m.group(1)
    return lanes


def test_every_makefile_lane_matches_a_ci_command():
    ci, lanes = _ci_pytest_commands(), _makefile_lanes()
    assert lanes, "parsed no pytest lanes out of the Makefile — parser broken?"
    drifted = {t: c for t, c in lanes.items() if c not in ci}
    assert not drifted, (
        "Makefile lane(s) no longer match any ci.yml command:\n"
        + "\n".join(f"  make {t}: {c}" for t, c in sorted(drifted.items()))
        + "\n\nci.yml runs:\n"
        + "\n".join(f"  {c}" for c in sorted(ci))
        + "\n\nA lane that drifts tests a different set than the gate it claims "
        "to mirror. Fix the Makefile, or update it deliberately with ci.yml."
    )


def test_every_ci_pytest_command_has_a_lane():
    ci, lanes = _ci_pytest_commands(), _makefile_lanes()
    assert ci, "parsed no pytest commands out of ci.yml — parser broken?"
    missing = ci - set(lanes.values())
    assert not missing, (
        "ci.yml runs pytest command(s) with no Makefile lane:\n"
        + "\n".join(f"  {c}" for c in sorted(missing))
        + "\n\nAdd a target so the lane can be run locally before it gates a PR."
    )


def test_the_memgate_lane_keeps_its_serial_flag():
    """`-n0` is load-bearing, not incidental: memgate certifies peak RSS and
    cannot share a box with concurrent xdist workers (ci.yml says so at the
    lane). A lane that lost it would still pass while measuring nothing."""
    lanes = _makefile_lanes()
    assert "memgate" in lanes, "no memgate lane"
    assert "-n0" in lanes["memgate"], (
        f"memgate lane lost its -n0: {lanes['memgate']!r} — it would then "
        "inherit addopts' `-n auto` and measure RSS under xdist workers."
    )
