"""The single-file Windows preview's release plumbing (standalone.yml's
`windows-single`, standalone/WINDOWS-SINGLE-FILE.md).

No executable is built here: the zip gate's rules are held on synthetic member
lists, and the workflow, the zip's README and its notices on their text.
"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "standalone.yml"
RELEASE = ROOT / "standalone" / "release"
ASSET = "momwire-nec5-windows-x64-single.zip"

_spec = importlib.util.spec_from_file_location(
    "check_single_zip", ROOT / "scripts" / "windows_static" / "check_single_zip.py"
)
pefile = pytest.importorskip("pefile")
CZ = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("check_single_zip", CZ)
_spec.loader.exec_module(CZ)

GOOD = [f"{CZ.TOP}/{n}" for n in sorted(CZ.EXPECTED)]


def _jobs():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]


def test_the_expected_tree_passes():
    assert CZ.tree_problems(GOOD) == []


@pytest.mark.parametrize(
    "extra",
    [
        f"{CZ.TOP}/libopenblas.dll",
        f"{CZ.TOP}/win7/momwire-nec5-razor-2p.exe",
        "stray.txt",
    ],
)
def test_a_stray_file_fails(extra):
    assert CZ.tree_problems([*GOOD, extra]) == [f"unexpected {extra}"]


@pytest.mark.parametrize("missing", sorted(CZ.EXPECTED))
def test_a_missing_file_fails(missing):
    names = [n for n in GOOD if not n.endswith(f"/{missing}")]
    assert CZ.tree_problems(names) == [f"missing {CZ.TOP}/{missing}"]


def test_backslash_members_and_folders_are_read_as_files(tmp_path):
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{CZ.TOP}/", "")
        z.writestr(f"{CZ.TOP}\\README.txt", "x")
    with zipfile.ZipFile(path) as z:
        assert CZ.zip_names(z) == [f"{CZ.TOP}/README.txt"]


def test_only_system_dlls_and_the_ucrt_are_allowed():
    ok = ["kernel32.dll", "api-ms-win-crt-math-l1-1-0.dll", "ntdll.dll"]
    assert CZ.disallowed(ok) == []
    for dll in (
        "libgomp-1.dll",
        "libwinpthread-1.dll",
        "libstdc++-6.dll",
        "libopenblas.dll",
        "vcruntime140.dll",
        "msvcrt.dll",  # a UCRT64 build must not mix in the other C runtime
    ):
        assert CZ.disallowed([*ok, dll]) == [dll]


def test_the_job_never_runs_on_a_pull_request():
    job = _jobs()["windows-single"]
    assert job["if"] == "${{ github.event_name != 'pull_request' }}"


def test_the_shim_is_one_switch():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert _jobs()["windows-single"]["env"]["MW_EXE_VMATH_SHIM"] in ("ON", "OFF")
    assert text.count('MW_EXE_VMATH_SHIM: "') == 1
    # every build that can take the shim reads the switch, not a literal
    steps = _jobs()["windows-single"]["steps"]
    build = next(s for s in steps if s.get("name", "").startswith("configure + build"))
    assert build["run"].count("-DMW_EXE_VMATH_SHIM=ON") == 1
    assert 'cfg "build-$b-sse2" $basis sse2\n' in build["run"]


def test_the_release_attach_is_tag_only_and_gated():
    job = _jobs()["release-windows-single"]
    assert job["needs"] == "windows-single"
    assert job["if"] == _jobs()["release-windows"]["if"]
    attach = job["steps"][-1]
    assert attach["with"]["files"] == ASSET


def test_the_msvc_zip_does_not_wait_for_the_preview():
    assert _jobs()["release-windows"]["needs"] == "windows-nopython"


def test_the_readme_states_the_requirement_and_that_it_is_a_preview():
    text = (RELEASE / "README-windows-single.txt").read_text(encoding="utf-8")
    assert "PREVIEW" in text
    assert "momwire-nec5-windows-x64.zip" in text  # the default, named
    assert "Universal C Runtime" in text
    assert "KB2999226" in text
    assert "Visual C++ 2015" in text
    for program in CZ.PROGRAMS:
        assert program in text


def _squash(text):
    return " ".join(text.split())


@pytest.mark.parametrize(
    "needle",
    [
        "The OpenBLAS Project",
        "The University of Tennessee",
        "mingw-w64 project",
        "Lockless Inc.",
        "MinGW-w64 runtime licensing",
        "GCC Runtime Library Exception",
    ],
)
def test_the_notices_carry_each_linked_runtime(needle):
    text = (RELEASE / "THIRD-PARTY-NOTICES-windows-single.txt").read_text(
        encoding="utf-8"
    )
    assert needle in text


@pytest.mark.parametrize(
    "source", ["LICENSE", "extern/xsf/LICENSE", "extern/xsf/LICENSES_bundled.txt"]
)
def test_the_notices_carry_the_in_tree_licences_whole(source):
    # momwire's own and the vendored xsf's, compiled into both programs
    text = (RELEASE / "THIRD-PARTY-NOTICES-windows-single.txt").read_text(
        encoding="utf-8"
    )
    assert _squash((ROOT / source).read_text(encoding="utf-8")) in _squash(text)
