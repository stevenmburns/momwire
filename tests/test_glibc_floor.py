"""The Linux wheels' glibc floor, read off the built extensions' symbol table.

The Linux wheels build on manylinux_2_28, so every symbol an extension takes
from glibc must exist in glibc 2.28. For the vectorised math this is easy to
break without noticing: libmvec grew most of its functions in 2.35 (expm1,
log1p, tan, atan2, ...), a build on any recent distro links them happily, and
the extension then fails to IMPORT from the wheel -- never locally, and
silently in effect, since a failed import drops every solver to numpy. Design
D4 (`_accel_razor_cplx.cpp`) exists in its shape because `_ZGVdN4v_expm1` is
GLIBC_2.35; this is the check that keeps the next kernel from reaching for it.

Two rules, because a symbol's version on a LOCAL build is not always the one
the wheel will carry:

  * **libmvec symbols (`_ZGV*`), everywhere.** libmvec never re-versions a
    function, so the version a local build records is the version the wheel
    needs. It must be <= GLIBC_2.28. An UNVERSIONED `_ZGV*` is a failure too:
    that is what a vector function the build's glibc does not have looks like
    after a link that allows undefined symbols.
  * **Every glibc symbol, where the host IS the floor.** A build on a newer
    glibc binds the newest default version of re-versioned functions
    (exp@GLIBC_2.29, hypot@GLIBC_2.35) that also exist at an older version,
    so the strict rule is only meaningful inside the manylinux_2_28 image --
    which is where cibuildwheel's test-command runs it (pyproject.toml).

Read with `objdump -T` rather than `nm -D`: both print the dynamic symbol
table, but binutils' `nm` only learned to print versions for it
(`--with-symbol-versions`, then by default) in releases newer than the
manylinux_2_28 image's, while `objdump -T` has printed the version column for
as long as either has existed.
"""

from __future__ import annotations

import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys

import pytest

import momwire

FLOOR = (2, 28)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="glibc symbol versions are Linux-only"
)

# cibuildwheel sets this in the build AND the test environment. There, an
# absent tool or an absent extension is a broken wheel check, not a skip.
_IN_WHEEL_BUILD = os.environ.get("CIBUILDWHEEL") == "1"


def _extensions() -> list[pathlib.Path]:
    pkg = pathlib.Path(momwire.__file__).parent
    return sorted(
        p
        for pat in ("_accelerators*.so", "_near_interface_accel*.so")
        for p in pkg.glob(pat)
    )


def _undefined_versions(so: pathlib.Path) -> dict[str, str]:
    """{symbol: version} for every UNDEFINED dynamic symbol ('' if unversioned)."""
    out = subprocess.run(
        ["objdump", "-T", str(so)], capture_output=True, text=True, check=True
    ).stdout
    syms: dict[str, str] = {}
    # e.g. "0000000000000000      DF *UND*  0000000000000000 (GLIBC_2.22) _ZGVdN4v_exp"
    # (newer binutils parenthesise the version; older print it bare, and a
    # hidden version is flagged with a leading '(' too).
    row = re.compile(r"\*UND\*\s+[0-9a-f]+\s+(?:\(?([A-Za-z_][\w.]*)\)?\s+)?(\S+)\s*$")
    for line in out.splitlines():
        m = row.search(line)
        if m:
            syms[m.group(2)] = m.group(1) or ""
    return syms


def _glibc(version: str) -> tuple[int, ...] | None:
    m = re.fullmatch(r"GLIBC_(\d+(?:\.\d+)*)", version)
    return tuple(int(x) for x in m.group(1).split(".")) if m else None


def _require_tools_and_extensions() -> list[pathlib.Path]:
    if shutil.which("objdump") is None:
        if _IN_WHEEL_BUILD:
            pytest.fail(
                "objdump is missing in the wheel build; the glibc check cannot run"
            )
        pytest.skip("objdump (binutils) not installed")
    exts = _extensions()
    if not exts:
        if _IN_WHEEL_BUILD:
            pytest.fail("no compiled momwire extension found in the installed wheel")
        pytest.skip("momwire installed without its compiled extensions")
    return exts


def test_libmvec_symbols_exist_in_glibc_2_28():
    exts = _require_tools_and_extensions()
    bad = []
    seen = 0
    for so in exts:
        for sym, ver in _undefined_versions(so).items():
            if not sym.startswith("_ZGV"):
                continue
            seen += 1
            v = _glibc(ver)
            if v is None or v > FLOOR:
                bad.append(f"{so.name}: {sym}@{ver or '<unversioned>'}")
    assert not bad, (
        "vector math newer than the wheel's glibc 2.28 floor:\n" + "\n".join(bad)
    )
    # Not vacuous: on x86-64 the AVX2 build binds libmvec (sin/cos since
    # #1032, exp since D4). If nothing was seen, the scan read nothing.
    avx2 = [p for p in exts if "_accelerators_avx2" in p.name]
    if avx2 and platform.machine().lower() in ("x86_64", "amd64"):
        assert seen, (
            "no libmvec symbol found in the AVX2 build; the scan is not reading it"
        )
        assert "_ZGVdN4v_exp" in _undefined_versions(avx2[0]), (
            "the AVX2 accelerator no longer binds libmvec exp (Design D4's bracket)"
        )


def test_every_glibc_symbol_exists_in_glibc_2_28_on_a_2_28_host():
    libc = os.confstr("CS_GNU_LIBC_VERSION") if hasattr(os, "confstr") else None
    m = re.fullmatch(r"glibc (\d+)\.(\d+)", libc or "")
    if not m:
        pytest.skip(f"not a glibc host ({libc!r})")
    host = (int(m.group(1)), int(m.group(2)))
    if host > FLOOR:
        if _IN_WHEEL_BUILD:
            pytest.fail(
                f"the Linux wheel is meant to build on glibc 2.28; this host is {host}"
            )
        pytest.skip(
            f"host glibc {host[0]}.{host[1]} binds newer default versions of "
            "re-versioned functions; the strict rule runs in the manylinux_2_28 wheel build"
        )
    exts = _require_tools_and_extensions()
    bad = [
        f"{so.name}: {sym}@{ver}"
        for so in exts
        for sym, ver in _undefined_versions(so).items()
        if (v := _glibc(ver)) is not None and v > FLOOR
    ]
    assert not bad, "symbols newer than glibc 2.28:\n" + "\n".join(bad)
