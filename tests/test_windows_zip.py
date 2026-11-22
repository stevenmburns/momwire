"""The MSVC Windows zip's tree (momwire#2090, scripts/windows_fat/check_zip.py).

No executable is built here: the gate's rules are held on synthetic trees.
The Windows runner runs the same gate on the staged folder and on the zip
(standalone.yml `windows-nopython`).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "standalone.yml"

_spec = importlib.util.spec_from_file_location(
    "check_zip", ROOT / "scripts" / "windows_fat" / "check_zip.py"
)
CZ = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("check_zip", CZ)
_spec.loader.exec_module(CZ)

T = CZ.TOP
UCRT = ["ucrtbase.dll", "api-ms-win-crt-math-l1-1-0.dll", "api-ms-win-core-file-l1-1-0.dll"]
GOOD = sorted(
    set(CZ.expected_programs())
    | {f"{T}/{d}" for d in CZ.DLLS}
    | {f"{T}/win7/{d}" for d in CZ.DLLS}
    | {f"{T}/win7/{u}" for u in UCRT}
    | {f"{T}/README.txt"}
)


def test_the_programs_are_the_fat_pair_and_the_old_names():
    progs = CZ.expected_programs()
    assert sorted(p for p in progs if p == progs[p]) == [
        f"{T}/momwire-nec5-bs2.exe",
        f"{T}/momwire-nec5-razor2p.exe",
    ]
    # the 0.96.0 zip's six programs, each now a copy of its basis's fat one
    for b in ("razor2p", "bs2"):
        for name in (
            f"{T}/momwire-nec5-{b}-avx2.exe",
            f"{T}/momwire-nec5-{b}-sse2.exe",
            f"{T}/win7/momwire-nec5-{b}-sse2.exe",
            f"{T}/win7/momwire-nec5-{b}.exe",
        ):
            assert progs[name] == f"{T}/momwire-nec5-{b}.exe"
    assert len(progs) == 10


def test_the_expected_tree_passes():
    assert CZ.tree_problems(GOOD) == []


@pytest.mark.parametrize(
    "extra",
    [
        f"{T}/win7/momwire-nec5-razor2p-avx2.exe",  # never shipped there
        f"{T}/momwire-nec5-razor-2p-avx2.exe",
        f"{T}/api-ms-win-crt-math-l1-1-0.dll",  # the UCRT is win7's alone
        f"{T}/ucrtbase.dll",
        "stray.txt",
    ],
)
def test_a_stray_file_fails(extra):
    assert CZ.tree_problems([*GOOD, extra]) == [f"unexpected {extra}"]


@pytest.mark.parametrize(
    "missing",
    [
        f"{T}/momwire-nec5-bs2.exe",
        f"{T}/momwire-nec5-razor2p-sse2.exe",
        f"{T}/win7/momwire-nec5-bs2-sse2.exe",
        f"{T}/win7/libscipy_openblas.dll",
        f"{T}/win7/ucrtbase.dll",
        f"{T}/README.txt",
    ],
)
def test_a_missing_file_fails(missing):
    assert CZ.tree_problems([n for n in GOOD if n != missing]) == [f"missing {missing}"]


def _stage(root: Path) -> None:
    for name in GOOD:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode())
    for name, fat in CZ.expected_programs().items():
        (root / name).write_bytes(f"fat {fat}".encode())


def test_copies_of_the_fat_programs_pass(tmp_path):
    _stage(tmp_path)
    assert CZ.main([str(tmp_path / T)]) == 0


def test_a_copy_that_is_not_its_programs_bytes_fails(tmp_path, capsys):
    _stage(tmp_path)
    # the old one-variant sse2 build under its old name: not the fat program
    (tmp_path / T / "momwire-nec5-razor2p-sse2.exe").write_bytes(b"one-variant sse2")
    assert CZ.main([str(tmp_path / T)]) == 1
    assert "momwire-nec5-razor2p-sse2.exe is not a copy of" in capsys.readouterr().out


def test_the_workflow_gates_the_staged_tree_and_the_zip():
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["windows-nopython"]["steps"]
    runs = "\n".join(s.get("run", "") for s in steps)
    assert "scripts/windows_fat/check_zip.py dist\\momwire-nec5-windows-x64" in runs
    assert 'scripts/windows_fat/check_zip.py "dist\\$base.zip"' in runs
