"""Gate: what the single-file Windows zip holds (standalone.yml `windows-single`).

    python scripts/windows_static/check_single_zip.py ZIP_OR_DIR

ZIP_OR_DIR is the release zip, or the staged folder it is made from (the\nfolder named momwire-nec5-windows-x64-single). It must
hold exactly the two programs, README.txt and THIRD-PARTY-NOTICES.txt in one
top-level folder, and nothing else: a DLL beside the programs would mean the
single file is not one. Each program's import table (the delay-import table
too) may name only the DLLs every supported Windows has, and the Universal C
Runtime's api-ms-win-crt-* sets, which the zip's README states as the
requirement. Exits 1 on any departure.
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path

import pefile

TOP = "momwire-nec5-windows-x64-single"
PROGRAMS = {"momwire-nec5-razor-2p.exe", "momwire-nec5-bspline.exe"}
EXPECTED = PROGRAMS | {"README.txt", "THIRD-PARTY-NOTICES.txt"}
SYSTEM = {"kernel32.dll", "advapi32.dll", "ntdll.dll"}
UCRT_PREFIX = "api-ms-win-crt-"


def imported_dlls(exe: Path) -> list[str]:
    """Every DLL `exe` names in its import and delay-import tables, lower case."""
    pe = pefile.PE(str(exe), fast_load=True)
    pe.parse_data_directories(
        directories=[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"],
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"],
        ]
    )
    entries = list(getattr(pe, "DIRECTORY_ENTRY_IMPORT", []))
    entries += list(getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", []))
    return sorted({e.dll.decode().lower() for e in entries})


def disallowed(dlls: list[str]) -> list[str]:
    return [d for d in dlls if d not in SYSTEM and not d.startswith(UCRT_PREFIX)]


def tree_problems(names: list[str]) -> list[str]:
    """What is wrong with the member list (POSIX paths, files only)."""
    want = {f"{TOP}/{n}" for n in EXPECTED}
    have = set(names)
    return [f"missing {n}" for n in sorted(want - have)] + [
        f"unexpected {n}" for n in sorted(have - want)
    ]


def check(names: list[str], root: Path) -> int:
    """Report on the member list `names`, whose files sit under `root`."""
    bad = tree_problems(names)
    for problem in bad:
        print(f"::error::{problem}")
    for name in sorted(names):
        print(name)
    for program in sorted(PROGRAMS):
        exe = root / TOP / program
        if not exe.is_file():
            continue
        dlls = imported_dlls(exe)
        print(f"{program}: {exe.stat().st_size} bytes, imports {' '.join(dlls)}")
        if not dlls:
            print(f"::error::{program}: no import table read")
            bad.append(program)
        for dll in disallowed(dlls):
            print(f"::error::{program} imports {dll}")
            bad.append(dll)
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
