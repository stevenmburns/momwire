"""The executables' OpenBLAS is the launcher's (momwire#1751).

`standalone/build_openblas.sh` builds the Linux executables two static
OpenBLAS archives, one at the commit numpy's wheel bundles and one at
SciPy's, because the launcher they replay runs on both and the two releases
round threaded products differently. The Windows executables download
numpy's release (standalone.yml reads NUMPY_OPENBLAS_VERSION from the same
script). That script is the one place the releases are recorded.

These tests hold the installed numpy and SciPy to it: when a numpy or SciPy
bump brings a different OpenBLAS, they fail here, loudly, rather than the
executables' threaded digits drifting from the launcher's unseen. Moving
the pin is then a deliberate edit to build_openblas.sh: the release, the
version string it reports and the OpenBLAS commit its wheel was built from
(MacPython/openblas-libs, `openblas_commit.txt` at the wheel's tag).

A numpy or SciPy that does not bundle scipy-openblas (macOS arm64, which
links Accelerate; a source build) has nothing to compare and skips.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "standalone" / "build_openblas.sh"


def recorded(name: str) -> str:
    m = re.search(rf"^{name}=(\S+)$", SCRIPT.read_text(), re.MULTILINE)
    assert m, f"{SCRIPT.name} records no {name}"
    return m.group(1)


def bundled(module) -> str:
    blas = module.__config__.CONFIG["Build Dependencies"]["blas"]
    if blas.get("name") != "scipy-openblas":
        pytest.skip(f"{module.__name__} links {blas.get('name')!r}, not scipy-openblas")
    return str(blas["version"])


@pytest.mark.parametrize(
    ("module_name", "prefix"),
    [("numpy", "NUMPY"), ("scipy", "SCIPY")],
)
def test_bundled_openblas_is_the_one_the_executables_link(module_name, prefix):
    module = pytest.importorskip(module_name)
    want = recorded(f"{prefix}_OPENBLAS_REPORTS")
    got = bundled(module)
    assert got == want, (
        f"{module_name} {module.__version__} bundles OpenBLAS {got}, but "
        f"standalone/build_openblas.sh builds the executables {want} "
        f"({prefix}_OPENBLAS_VERSION {recorded(f'{prefix}_OPENBLAS_VERSION')}). "
        "Move the pin there (version, reported string and commit) with the bump."
    )


def test_the_script_records_a_full_commit_for_each_library():
    for prefix in ("NUMPY", "SCIPY"):
        assert re.fullmatch(r"[0-9a-f]{40}", recorded(f"{prefix}_OPENBLAS_COMMIT"))
        assert re.fullmatch(r"\d+(\.\d+){3,4}", recorded(f"{prefix}_OPENBLAS_VERSION"))
