"""Gate: each executable's import table names system DLLs only (windows-static.yml).

    python scripts/windows_static/pe_imports.py DIR [DIR ...]

Lists every `*.exe` under each DIR with the DLLs it imports and fails on any
outside the allow-list: KERNEL32, msvcrt, ADVAPI32 and ntdll -- plus, for a
directory named `dist-ucrt` (the MSYS2 UCRT64 builds), the UCRT's
`api-ms-win-crt-*` API sets, system DLLs on Windows 10 and later and on
Windows 7 with KB2999226. Anything else (an OpenBLAS, libgomp, winpthread,
libstdc++ or libgcc DLL) means something did not link in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pefile

SYSTEM = {"kernel32.dll", "msvcrt.dll", "advapi32.dll", "ntdll.dll"}


def imports(exe: Path) -> list[str]:
    pe = pefile.PE(str(exe), fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
    )
    return sorted(
        i.dll.decode().lower() for i in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
    )


def main(argv: list[str]) -> int:
    bad = 0
    for d in map(Path, argv):
        ucrt = d.name == "dist-ucrt"
        for exe in sorted(d.glob("*.exe")):
            dlls = imports(exe)
            print(f"{exe} {exe.stat().st_size} bytes: {' '.join(dlls)}")
            for x in dlls:
                if x in SYSTEM or (ucrt and x.startswith("api-ms-win-crt-")):
                    continue
                print(f"::error::{exe} imports {x}")
                bad = 1
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
