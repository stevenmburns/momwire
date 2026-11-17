"""The fat MSVC executable's link, read back from its image and map file.

    python scripts/windows_fat/check_link.py EXE MAP BUILD_DIR

The fat build (standalone/CMakeLists.txt, MW_EXE_VARIANT=fat) carries the
cores twice: sse2 as objects, avx2 as two static libraries, `mw_core_mwa` and
`mw_core_mwas`, compiled /arch:AVX2. On a CPU without AVX2 no instruction of
the avx2 copies may run, before main() included. Three checks, each against
the linked bytes:

1. START-UP. Every function pointer the C runtime runs on its own -- the
   `.CRT$X*` tables (C and C++ initializers, pre-terminators, terminators,
   TLS callbacks' table) and the TLS directory's callbacks -- points into an
   sse2 object or the runtime, never into an avx2 library member. Every
   pointer in `.mwav$m` (fat_avx2_init.h: the avx2 copies' initializers,
   which main() runs only once it has picked avx2) points into an avx2
   member, and `MW_EXE_TIMES=1` reports that many run on avx2 and none on
   sse2.

2. COMDATS. A symbol an avx2 member supplies whose name is not the avx2
   copies' own -- in none of their namespaces (`mwa`, `mwas`, `mwa_xsf`,
   ...) and not renamed `mwa$` by fat_private_comdats.py: data, say -- must
   be defined by no sse2 object: had an sse2 object defined it too, the sse2
   path could reach it, and the linker must have kept the sse2 copy. Objects
   are read with `dumpbin /symbols`.

3. The counts: how much of the image each copy supplies.

Exits 1 on any violation.
"""

from __future__ import annotations

import bisect
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import pefile

AVX2_LIBS = ("mw_core_mwa:", "mw_core_mwas:")
# an avx2 copy's own name: in a renamed namespace, as MSVC mangles it
# (`@mwa@`, `@mwas_xsf@`), or a function fat_private_comdats.py renamed
AVX2_NS = re.compile(r"^mwa\$|@mwas?(?:_[A-Za-z0-9_]+)?@")
# constants named by their own bytes: two copies with one name are one value
CONTENT_NAMED = ("__real@", "__xmm@", "__ymm@", "__zmm@", "??_C@", "__mask@")
STARTUP_GROUPS = re.compile(r"^\.CRT\$X")


def parse_map(path: Path):
    base = None
    groups = []  # (sec, off, length, name)
    syms = []  # (va, name, obj, is_public)
    mode = None
    for line in path.read_text(errors="replace").splitlines():
        if "Preferred load address is" in line:
            base = int(line.split()[-1], 16)
            continue
        s = line.strip()
        if s.startswith("Start") and "Length" in s and "Class" in s:
            mode = "groups"
            continue
        if "Publics by Value" in s:
            mode = "publics"
            continue
        if s.startswith("Static symbols"):
            mode = "statics"
            continue
        if s.startswith("entry point at"):
            mode = None
            continue
        t = s.split()
        if not t or ":" not in t[0]:
            continue
        try:
            sec, off = (int(x, 16) for x in t[0].split(":"))
        except ValueError:
            continue
        if mode == "groups" and len(t) >= 3 and t[1].endswith("H"):
            groups.append((sec, off, int(t[1][:-1], 16), t[2]))
        elif mode in ("publics", "statics") and len(t) >= 4:
            try:
                va = int(t[2], 16)
            except ValueError:
                continue
            syms.append((va, t[1], t[-1], mode == "publics"))
    if base is None:
        raise SystemExit(f"{path}: no preferred load address")
    return base, groups, syms


def obj_symbols(dumpbin: str, obj: Path) -> set[str]:
    out = subprocess.run(
        [dumpbin, "/nologo", "/symbols", str(obj)],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    ).stdout
    names = set()
    for line in out.splitlines():
        if "| " not in line:
            continue
        left, right = line.split("| ", 1)
        f = left.split()
        if len(f) >= 3 and f[2].startswith("SECT") and "External" in f:
            names.add(right.split()[0])
    return names


def main(argv: list[str]) -> int:
    exe, mapfile, build = Path(argv[0]), Path(argv[1]), Path(argv[2])
    dumpbin = os.environ.get("DUMPBIN", "dumpbin")
    base, groups, syms = parse_map(mapfile)
    pe = pefile.PE(str(exe), fast_load=False)
    sec_rva = {i + 1: s.VirtualAddress for i, s in enumerate(pe.sections)}
    syms.sort()
    addrs = [s[0] for s in syms]

    def owner(va: int) -> tuple[str, str]:
        i = bisect.bisect_right(addrs, va) - 1
        if i < 0:
            return "?", "?"
        return syms[i][2], syms[i][1]

    def is_avx2(obj: str) -> bool:
        return obj.startswith(AVX2_LIBS)

    def pointers(sec: int, off: int, length: int) -> list[int]:
        rva = sec_rva[sec] + off
        data = pe.get_data(rva, length)
        return [v for (v,) in struct.iter_unpack("<Q", data[: len(data) // 8 * 8]) if v]

    bad = 0
    # 1. start-up
    n_startup = 0
    for sec, off, length, name in groups:
        if not STARTUP_GROUPS.match(name):
            continue
        for v in pointers(sec, off, length):
            n_startup += 1
            obj, sym = owner(v)
            if is_avx2(obj):
                print(f"START-UP {name} -> {sym} ({obj})")
                bad = 1
    tls = getattr(pe, "DIRECTORY_ENTRY_TLS", None)
    n_tls = 0
    if tls is not None and tls.struct.AddressOfCallBacks:
        rva = tls.struct.AddressOfCallBacks - base
        while True:
            (v,) = struct.unpack("<Q", pe.get_data(rva, 8))
            if not v:
                break
            n_tls += 1
            obj, sym = owner(v)
            if is_avx2(obj):
                print(f"START-UP TLS callback -> {sym} ({obj})")
                bad = 1
            rva += 8
    deferred = 0
    for sec, off, length, name in groups:
        if name != ".mwav$m":
            continue
        for v in pointers(sec, off, length):
            deferred += 1
            obj, sym = owner(v)
            if not is_avx2(obj):
                print(f"DEFERRED .mwav$m -> {sym} ({obj}): not an avx2 member")
                bad = 1
    print(
        f"start-up: {n_startup} C runtime table entries and {n_tls} TLS callbacks,"
        f" none in an avx2 copy: {'no' if bad else 'yes'}"
    )
    print(f"deferred avx2 initializers in the image (.mwav$m): {deferred}")

    # the run-time count matches the image's (this runner has AVX2)
    for variant, want in (("avx2", deferred), ("sse2", None)):
        env = dict(os.environ, MW_EXE_TIMES="1", MOMWIRE_FORCE_VARIANT=variant)
        r = subprocess.run(
            [str(exe), "-version"], env=env, capture_output=True, text=True
        )
        m = re.search(r"avx2 initializers (\d+)", r.stderr)
        got = int(m.group(1)) if m else None
        print(f"  {variant}: -version rc={r.returncode}, initializers run: {got}")
        if r.returncode != 0 or got != want:
            print(
                f"::error::{exe.name} {variant}: ran {got} avx2 initializers, want {want}"
            )
            bad = 1

    # 2. COMDATs
    avx2_pub = [s for s in syms if s[3] and is_avx2(s[2])]
    own = [s for s in avx2_pub if AVX2_NS.search(s[1])]
    renamed = [s for s in avx2_pub if s[1].startswith("mwa$")]
    shared = {
        s[1]
        for s in avx2_pub
        if not AVX2_NS.search(s[1]) and not s[1].startswith(CONTENT_NAMED)
    }
    objs = [
        p
        for p in build.rglob("*.obj")
        if not any(part in ("mw_core_mwa.dir", "mw_core_mwas.dir") for part in p.parts)
        and p.relative_to(build).parts[0] != "fat"  # the renamed avx2 objects
    ]
    defined: set[str] = set()
    for p in objs:
        defined |= obj_symbols(dumpbin, p)
    clash = sorted(shared & defined)
    for name in clash[:40]:
        print(f"COMDAT {name}: an sse2 object defines it, the image has the avx2 copy")
    if clash:
        bad = 1
    print(
        f"comdats: {len(avx2_pub)} public symbols from the avx2 copies, {len(own)} in their"
        f" own names ({len(renamed)} of them renamed mwa$), {len(shared)} outside them (data and the like);"
        f" {len(objs)} sse2/main objects define {len(defined)} externals;"
        f" {len(clash)} resolved to the avx2 copy"
    )

    # 3. counts
    by = {"avx2": 0, "other": 0}
    for s in syms:
        by["avx2" if is_avx2(s[2]) else "other"] += 1
    print(f"map symbols: {by['avx2']} from the avx2 libraries, {by['other']} elsewhere")
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
