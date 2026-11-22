"""Gate: what the MSVC Windows zip holds (standalone.yml `windows-nopython`).

    python scripts/windows_fat/check_zip.py ZIP_OR_DIR

ZIP_OR_DIR is momwire-nec5-windows-x64.zip (any of its run names), or the
staged folder it is made from (named momwire-nec5-windows-x64). Since
momwire#2090 the zip ships ONE fat program per engine, which picks avx2 or
sse2 by CPU:

    momwire-nec5-razor2p.exe    momwire-nec5-bs2.exe

and, for the deprecation window, the 0.96.0 names as byte-for-byte copies of
those programs (a copy runs the variant its name names, standalone/exe_name.h):

    momwire-nec5-<basis>-avx2.exe    momwire-nec5-<basis>-sse2.exe
    win7/momwire-nec5-<basis>-sse2.exe

win7/ holds the same fat pair and its old sse2 names, beside Microsoft's
app-local Universal C Runtime. Every folder carries the DLLs the programs
load; nothing else may be there. Exits 1 on any departure, a copy that is not
its program's bytes included.
"""

from __future__ import annotations

import hashlib
import re
import sys
import tempfile
import zipfile
from pathlib import Path

TOP = "momwire-nec5-windows-x64"
BASES = ("razor2p", "bs2")
DLLS = {
    "libscipy_openblas.dll",
    "msvcp140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "libomp140.x86_64.dll",
}
# Microsoft's app-local UCRT, win7/ only: its file set is the SDK's
UCRT = re.compile(r"(api-ms-win-[a-z0-9-]+\.dll|ucrtbase\.dll)")


def program(basis: str) -> str:
    return f"momwire-nec5-{basis}.exe"


def deprecated(basis: str, variant: str) -> str:
    return f"momwire-nec5-{basis}-{variant}.exe"


def expected_programs() -> dict[str, str]:
    """Each program's path in the zip, mapped to the fat program it copies."""
    want = {}
    for b in BASES:
        fat = f"{TOP}/{program(b)}"
        want[fat] = fat
        want[f"{TOP}/win7/{program(b)}"] = fat
        for v in ("avx2", "sse2"):
            want[f"{TOP}/{deprecated(b, v)}"] = fat
        want[f"{TOP}/win7/{deprecated(b, 'sse2')}"] = fat
    return want


def tree_problems(names: list[str]) -> list[str]:
    """What is wrong with the member list (POSIX paths, files only)."""
    want = set(expected_programs())
    want |= {f"{TOP}/{d}" for d in DLLS} | {f"{TOP}/win7/{d}" for d in DLLS}
    want.add(f"{TOP}/README.txt")
    have = set(names)
    ucrt = {
        n
        for n in have - want
        if n.startswith(f"{TOP}/win7/") and UCRT.fullmatch(n.rsplit("/", 1)[1])
    }
    problems = [f"missing {n}" for n in sorted(want - have)]
    problems += [f"unexpected {n}" for n in sorted(have - want - ucrt)]
    if f"{TOP}/win7/ucrtbase.dll" not in have:
        problems.append(f"missing {TOP}/win7/ucrtbase.dll")
    return problems


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(names: list[str], root: Path) -> int:
    """Report on the member list `names`, whose files sit under `root`."""
    bad = tree_problems(names)
    for problem in bad:
        print(f"::error::{problem}")
    for name in sorted(names):
        print(f"{name} {(root / name).stat().st_size}")
    programs = expected_programs()
    for name, fat in sorted(programs.items()):
        if not (root / name).is_file() or not (root / fat).is_file():
            continue
        if sha256(root / name) != sha256(root / fat):
            print(f"::error::{name} is not a copy of {fat}")
            bad.append(name)
    print(f"{len(programs)} programs, copies of {len(BASES)} fat executables")
    return 1 if bad else 0


def zip_names(archive: zipfile.ZipFile) -> list[str]:
    """The archive's files. Windows PowerShell writes backslashes."""
    names = [n.replace("\\", "/") for n in archive.namelist()]
    return [n for n in names if not n.endswith("/")]


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    target = Path(argv[0])
    if target.is_dir():
        if target.name != TOP:
            print(f"::error::{target} is not a folder named {TOP}")
            return 2
        names = [
            f"{TOP}/{p.relative_to(target).as_posix()}"
            for p in target.rglob("*")
            if p.is_file()
        ]
        return check(names, target.parent)
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(target) as archive:
        archive.extractall(tmp)
        return check(zip_names(archive), Path(tmp))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
