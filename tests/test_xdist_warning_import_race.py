"""The xdist controller must not race itself importing momwire.

The controller decodes each worker's warning in that worker's receiver
thread, importing the warning's module; two first warnings from different
momwire modules then import momwire in two threads at once and CPython's
import locks invert (tests/conftest.py's `_serialize_warning_imports` has
the mechanism). It surfaced as an intermittent INTERNALERROR on the first
suite run after `make build`: a circular import of `DistributedRLC` from a
partially initialised `momwire._wire_loading`, then `KeyError: 'momwire'`.

`xdist_race/warn_probe.py` makes it certain rather than intermittent: eight
tests, each raising one advisory from a different momwire module, on eight
workers. Without the conftest's lock it hits on every run (10 of 10 on
Haswell, warm or cold pycache); with it, never. Run in a fresh interpreter,
because the race is in the controller's FIRST import of momwire.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROBE = Path(__file__).parent / "xdist_race" / "warn_probe.py"

pytestmark = pytest.mark.integration


def test_the_probe_is_not_collected_by_the_suite():
    assert PROBE.exists()
    assert not PROBE.name.startswith("test_")


@pytest.mark.parametrize("attempt", range(2))
def test_the_controller_decodes_concurrent_warnings(attempt, tmp_path):
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(PROBE),
            "-n",
            "8",
            "-p",
            "no:cacheprovider",
            "-q",
            "-W",
            "default",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    out = run.stdout + run.stderr
    assert "INTERNALERROR" not in out, out[-4000:]
    assert run.returncode == 0, out[-4000:]
    assert "8 passed" in out, out[-2000:]
