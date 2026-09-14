"""`pip install -e ".[test]"` must leave a venv in which `make build` runs (momwire#1051).

`make build` calls `setup.py build_ext --inplace` in the venv itself, so it needs
the `[build-system] requires` (setuptools, wheel, pybind11) installed there. The
editable install satisfies them only inside pip's isolated build environment,
and it leaves built extensions behind that import fine. So a fresh dev venv
passed an import check and still could not rebuild after a C++ change or a
pull, which is the stale-`.so` failure `make build`'s `MOMWIRE_REQUIRE_ACCEL=1`
exists to prevent. CI never saw it: every build job installs the three by hand
first.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _dist_name(requirement: str) -> str:
    name = re.split(r"[<>=!~;\[\s]", requirement, maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def test_the_test_extra_carries_every_build_system_requirement():
    data = tomllib.loads(PYPROJECT.read_text())
    build = {_dist_name(r) for r in data["build-system"]["requires"]}
    test = {_dist_name(r) for r in data["project"]["optional-dependencies"]["test"]}

    missing = sorted(build - test)
    assert not missing, (
        f"`make build` needs {missing} in the venv, but the `test` extra does not "
        'install them, so `pip install -e ".[test]"` leaves a venv that cannot '
        "rebuild the accelerator (momwire#1051)"
    )
