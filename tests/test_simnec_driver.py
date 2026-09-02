"""SimNECDriver (scripts/simnec_driver) against the installed entry point.

The driver is Java, so this needs a `java` on PATH and the `momwire-nec2c-*`
entry points next to the interpreter, which an installed momwire has and a
bare source tree does not. Without either it skips rather than fails: the
verdict that matters is the one `simnec-driver.yml` buys on the platforms
SimNEC runs on, and this is the local guard that the driver itself still
works.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DRIVER = ROOT / "scripts" / "simnec_driver" / "SimNECDriver.java"
DECK = ROOT / "tests" / "fixtures" / "nec_portal" / "dipole_free_space.deck"
JAVA = shutil.which("java")
_bindir = Path(sys.executable).parent
ENGINE = _bindir / (
    "momwire-nec2c-bspline.exe" if os.name == "nt" else "momwire-nec2c-bspline"
)

pytestmark = [
    pytest.mark.skipif(JAVA is None, reason="no java on PATH"),
    pytest.mark.skipif(not ENGINE.exists(), reason=f"no entry point at {ENGINE}"),
]


def _drive(*args: str, timeout: float = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [JAVA, str(DRIVER), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_installed_entry_point_passes_simnecs_launch_contract(tmp_path):
    done = _drive(str(ENGINE), str(DECK), "--repeat", "2", "--home", str(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "PASS" in done.stdout
    assert "ended on NX data-card echo" in done.stdout
    assert done.stdout.count("  ok    deck") == 2


def test_driver_reports_the_exit_code_simnec_hides(tmp_path):
    # An engine that dies at the probe: SimNEC shows only "NEC Failure Code
    # 7"; the driver must show that AND the stderr behind it.
    if os.name == "nt":
        pytest.skip("the broken-wrapper shape is a sh script")
    wrapper = tmp_path / "nec2c-broken"
    wrapper.write_text("#!/bin/sh\necho the real reason >&2\nexit 7\n")
    wrapper.chmod(0o755)
    done = _drive(str(wrapper), "--probe-only")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "NEC Failure Code 7" in done.stdout
    assert "the real reason" in done.stdout


def test_driver_reports_a_name_that_selects_no_basis(tmp_path):
    # momwire#528: the file name is the engine selector, and an unknown one
    # exits 3 at the probe with the reason on stdout, which is where the
    # driver has to look because SimNEC would not.
    if os.name == "nt":
        pytest.skip("symlink shape")
    link = tmp_path / "momwire-nec2c-bogus"
    link.symlink_to(ENGINE)
    done = _drive(str(link), "--probe-only")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "NEC Failure Code 3" in done.stdout
    assert "unknown basis 'bogus'" in done.stdout
