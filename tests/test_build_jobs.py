"""How many compilers a build runs at once (`momwire._build_jobs`).

The rule setup.py and the standalone-core tests share: MOMWIRE_BUILD_JOBS if
set, else the physical cores capped by the memory available at JOB_MB each.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from momwire import _build_jobs as bj

ROOT = Path(__file__).resolve().parent.parent


def test_the_variable_overrides_the_default(monkeypatch):
    monkeypatch.setenv(bj.ENV, "3")
    assert bj.build_jobs() == 3


def test_zero_or_unset_takes_the_default(monkeypatch):
    monkeypatch.setattr(bj, "physical_cores", lambda: 6)
    monkeypatch.setattr(bj, "available_mb", lambda: 64 * 1024)
    monkeypatch.delenv(bj.ENV, raising=False)
    assert bj.build_jobs() == 6
    monkeypatch.setenv(bj.ENV, "0")
    assert bj.build_jobs() == 6


def test_a_word_is_refused_by_name(monkeypatch):
    monkeypatch.setenv(bj.ENV, "lots")
    with pytest.raises(SystemExit, match=bj.ENV):
        bj.build_jobs()


def test_the_default_is_physical_cores_not_threads(monkeypatch):
    monkeypatch.delenv(bj.ENV, raising=False)
    monkeypatch.setattr(bj, "physical_cores", lambda: 4)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(bj, "available_mb", lambda: None)
    assert bj.build_jobs() == 4


def test_memory_caps_the_default(monkeypatch):
    """6 GB free at 1 GiB a compiler is 5 jobs, whatever the core count."""
    monkeypatch.delenv(bj.ENV, raising=False)
    monkeypatch.setattr(bj, "physical_cores", lambda: 16)
    monkeypatch.setattr(bj, "available_mb", lambda: 6000)
    assert bj.build_jobs() == 5
    monkeypatch.setattr(bj, "available_mb", lambda: 300)
    assert bj.build_jobs() == 1  # never zero


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_on_linux_the_core_count_is_what_lscpu_counts():
    try:
        out = subprocess.run(
            ["lscpu", "-p=CORE,SOCKET"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("no lscpu")
    cores = {line for line in out.splitlines() if line and not line.startswith("#")}
    assert bj.physical_cores() == len(cores)


def test_setup_py_uses_the_rule_and_the_old_name_is_gone():
    text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "_build_jobs.py" in text
    assert "NPY_NUM_BUILD_JOBS" not in text
