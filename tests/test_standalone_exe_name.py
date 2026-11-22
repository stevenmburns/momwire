"""The fat Windows executable's own name (momwire#2090, standalone/exe_name.h).

The Windows zip ships one fat program per engine and, for the deprecation
window, copies of it under the 0.96.0 names (`momwire-nec5-razor2p-avx2.exe`,
...). A copy reads the variant its name asks for; this compiles the header on
its own and holds the rule. What the copies COMPUTE (an old -sse2 name on an
avx2 runner answering the sse2 build's bytes, line 2's note) is gated on the
Windows runner, in standalone.yml's windows-nopython job (fat_compare.py
--by-name).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    # path, the variant it asks for
    (r"C:\momwire\momwire-nec5-razor2p-avx2.exe", "avx2"),
    (r"C:\momwire\momwire-nec5-razor2p-sse2.exe", "sse2"),
    (r"C:\momwire\win7\momwire-nec5-bs2-sse2.exe", "sse2"),
    (r"C:\MOMWIRE\MOMWIRE-NEC5-BS2-AVX2.EXE", "avx2"),
    ("/x/momwire-nec5-bs2-sse2", "sse2"),
    ("momwire-nec5-bs2-avx2.exe", "avx2"),
    # the new names, and anything else, ask for nothing
    (r"C:\momwire\momwire-nec5-razor2p.exe", ""),
    (r"C:\momwire\momwire-nec5-bs2.exe", ""),
    (r"C:\avx2-sse2\momwire-nec5-bs2.exe", ""),
    (r"C:\momwire\momwire-nec5-bs2-avx2.exe.bak", ""),
    (r"C:\momwire\momwire-nec5-bs2-avx2x.exe", ""),
    (r"C:\momwire\momwire-nec5-bs2-legacy.exe", ""),
    ("-avx2.exe", ""),
    ("", ""),
]

DRIVER = r"""
#include "exe_name.h"
#include <cstdio>
#include <iostream>
#include <string>
int main() {
    std::string line;
    while (std::getline(std::cin, line)) {
        const std::string v = mw_exe_name::variant_from_path(line);
        std::cout << "[" << v << "] [" << mw_exe_name::deprecation_note("momwire-nec5-bs2", v) << "]\n";
    }
}
"""


@pytest.fixture(scope="module")
def driver(tmp_path_factory) -> Path:
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.skip("no C++ compiler")
    d = tmp_path_factory.mktemp("exe_name")
    src = d / "driver.cpp"
    src.write_text(DRIVER)
    exe = d / "driver"
    subprocess.run(
        [
            cxx,
            "-std=c++17",
            "-Wall",
            "-Werror",
            f"-I{ROOT / 'standalone'}",
            str(src),
            "-o",
            str(exe),
        ],
        check=True,
    )
    return exe


def test_the_name_asks_for_its_variant(driver: Path) -> None:
    out = subprocess.run(
        [str(driver)],
        input="".join(f"{p}\n" for p, _ in CASES),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert len(out) == len(CASES)
    for (path, want), got in zip(CASES, out, strict=True):
        note = f"(renamed from momwire-nec5-bs2-{want})" if want else ""
        assert got == f"[{want}] [{note}]", path
