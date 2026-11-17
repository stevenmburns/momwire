"""The fat MSVC build's avx2 copies as two static libraries whose functions
outside the renamed namespaces are their own (MW_EXE_VARIANT=fat,
CMakeLists.txt).

    python fat_private_comdats.py --objcopy LLVM_OBJCOPY --nm LLVM_NM --lib LIB_EXE
        --out DIR NAME=OBJECT_DIR [NAME=OBJECT_DIR ...]

WHY. The `-D` renames give each copy its own namespaces (`mwa`, `mwas_xsf`,
...), but a template or inline function outside them -- `std::`, a header's
global-namespace helper -- has one name in every copy. The linker keeps one
body per name, and the sse2 objects, linked first, supply it, so the avx2
code would call an sse2-compiled body: correct, but not the one-variant avx2
executable's bits (run 44003593261: three decks' avx2 printouts differed).
Every external function an avx2 object defines outside the avx2 namespaces
is renamed `mwa$<name>`, definition and every reference, in both avx2
copies (one prefix: the serial copy shares the threaded copy's bodies, as
`mws` shares `mw`'s in a one-variant build). Data symbols keep their names
(one value, one identity). Names only: no code changes. The objects are
rewritten by llvm-objcopy and archived by MSVC's lib.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PREFIX = "mwa$"
# a renamed namespace of an avx2 copy, as MSVC mangles it (`@mwa@`, `@mwas_xsf@`)
AVX2_NS = re.compile(r"@mwas?(?:_[A-Za-z0-9_]+)?@")


def defined_functions(nm: str, obj: Path) -> set[str]:
    out = subprocess.run(
        [nm, "--defined-only", "--extern-only", str(obj)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    names = set()
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 3 and f[1] == "T":
            names.add(f[2])
    return names


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objcopy", required=True)
    ap.add_argument("--nm", required=True)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("copies", nargs="+")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    copies = []
    for spec in a.copies:
        name, d = spec.split("=", 1)
        objs = sorted(Path(d).rglob("*.obj"))
        if not objs:
            raise SystemExit(f"no objects under {d}")
        copies.append((name, objs))
    rename: set[str] = set()
    for _, objs in copies:
        for obj in objs:
            rename |= {
                n
                for n in defined_functions(a.nm, obj)
                if not AVX2_NS.search(n) and not n.startswith(PREFIX)
            }
    table = out / "renames.txt"
    table.write_text("".join(f"{n} {PREFIX}{n}\n" for n in sorted(rename)))
    for name, objs in copies:
        d = out / name
        d.mkdir(exist_ok=True)
        rewritten = []
        for obj in objs:
            dst = d / obj.name
            if dst in rewritten:
                raise SystemExit(f"two objects named {obj.name} in {name}")
            subprocess.run(
                [a.objcopy, f"--redefine-syms={table}", str(obj), str(dst)],
                check=True,
            )
            rewritten.append(dst)
        subprocess.run(
            [a.lib, "/nologo", f"/OUT:{out / (name + '.lib')}", *map(str, rewritten)],
            check=True,
        )
        print(f"{name}.lib: {len(rewritten)} objects")
    print(f"{len(rename)} functions outside the avx2 namespaces renamed {PREFIX}<name>")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
