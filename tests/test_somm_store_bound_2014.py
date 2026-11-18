"""The executables' Sommerfeld store keeps ONE bound across builds (momwire#2014).

Each build of an executable has its own directory, `exe/<tag>`, and a replaced
build's is never read again. `MOMWIRE_SOMM_CACHE_MAX_MB` therefore bounds every
build's files together, least recently used first across them; see
docs/design/exe-somm-store-bound.md.

The driver (`scripts/core_split/somm_store_driver.cpp`) is the store's own C++
with no solver in front of it, so several builds stand side by side in seconds.
What the shipped executable does with the same code is held by
tests/test_standalone_nopython.py, on Linux, Windows and macOS.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "momwire"
SOURCES = ("_somm_kernel_core.cpp", "_somm_grid_core.cpp", "_cpu_count_core.cpp")

MB = 1 << 20
DAY = 86400.0

# Compiling the driver takes ~8 s, over the default lane's budget.
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        sys.platform != "linux", reason="the driver build is the Linux one"
    ),
    pytest.mark.skipif(
        shutil.which(os.environ.get("CXX", "g++")) is None, reason="no C++ compiler"
    ),
]


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    exe = tmp_path_factory.mktemp("somm-store-driver") / "somm_store_driver"
    cmd = [
        os.environ.get("CXX", "g++"),
        "-std=c++17",
        "-O1",
        "-fopenmp",
        "-I",
        str(PKG),
        *[str(PKG / s) for s in SOURCES],
        str(ROOT / "scripts" / "core_split" / "somm_store_driver.cpp"),
        "-o",
        str(exe),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return exe


def _env(store: Path, max_mb: float) -> dict[str, str]:
    return dict(
        os.environ,
        MOMWIRE_SOMM_CACHE="1",
        MOMWIRE_SOMM_CACHE_DIR=str(store),
        MOMWIRE_SOMM_CACHE_MAX_MB=str(max_mb),
    )


def _start(driver, store, max_mb, tag, rounds, tables, kb, seed=1):
    return subprocess.Popen(
        [str(driver), tag, str(rounds), str(tables), str(kb), str(seed)],
        env=_env(store, max_mb),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _finish(proc) -> dict[str, int]:
    out, err = proc.communicate(timeout=240)
    assert proc.returncode == 0, err
    return json.loads(out)


def _run(driver, store, max_mb, tag, rounds, tables, kb, seed=1) -> dict[str, int]:
    return _finish(_start(driver, store, max_mb, tag, rounds, tables, kb, seed))


def _total(store: Path) -> int:
    return sum(
        p.stat().st_size for p in (store / "exe").glob("*/*") if p.suffix != ".tmp"
    )


def _builds(store: Path) -> list[str]:
    return sorted(p.name for p in (store / "exe").iterdir())


def _age(path: Path, days: float) -> None:
    then = time.time() - days * DAY
    os.utime(path, (then, then))


def test_rebuilds_stay_under_one_bound(driver, tmp_path):
    """Twelve successive builds, each filling 2 MB, under a 5 MB bound: the
    total never passes it, the replaced builds' directories go, and the newest
    build stays whole, so its next run reads everything back."""
    store = tmp_path / "store"
    evicted = 0
    for n in range(12):
        got = _run(driver, store, 5, f"0.94.0-build{n:02d}", 64, 8, 256)
        evicted += got["evicted"]
        assert got["corrupt"] == 0
        assert _total(store) <= 5 * MB, n
    assert evicted > 0, "the bound was never reached: this measured nothing"
    assert len(_builds(store)) <= 3, _builds(store)
    assert "0.94.0-build11" in _builds(store)
    warm = _run(driver, store, 5, "0.94.0-build11", 64, 8, 256)
    assert warm["misses"] == 0 and warm["hits"] == 64, warm


def test_builds_that_fit_together_leave_each_other_warm(driver, tmp_path):
    """Two builds used in turn, their tables under the bound together: neither
    costs the other a single hit, as with one directory each."""
    store = tmp_path / "store"
    for tag in ("msvc", "mingw"):
        _run(driver, store, 8, tag, 64, 8, 256)
    for tag in ("msvc", "mingw", "msvc", "mingw"):
        warm = _run(driver, store, 8, tag, 64, 8, 256, seed=7)
        assert warm["misses"] == 0 and warm["evicted"] == 0, (tag, warm)


def test_concurrent_builds_only_ever_miss(driver, tmp_path):
    """Four builds, three processes each, fighting over a bound far below what
    they ask for: every process exits cleanly, no read ever returns a wrong
    byte, and the store ends under the bound."""
    store = tmp_path / "store"
    procs = [
        _start(driver, store, 3, f"build-{b}", 400, 12, 128, seed=10 * b + p)
        for b in range(4)
        for p in range(3)
    ]
    results = [_finish(p) for p in procs]
    assert sum(r["corrupt"] for r in results) == 0, results
    assert sum(r["evicted"] for r in results) > 0, results
    assert sum(r["misses"] for r in results) > 0, results
    assert sum(r["hits"] for r in results) > 0, results
    assert _total(store) <= 3 * MB
    assert not list((store / "exe").glob("*/*.tmp"))


def test_a_seen_list_outlives_its_tables_for_thirty_days(driver, tmp_path):
    """A seen list steers the below table's sweep anchors, so it is not evicted
    with the tables; a build with no table left keeps it for 30 days and then
    goes altogether."""
    store = tmp_path / "store"
    recent = store / "exe" / "recent"
    stale = store / "exe" / "stale"
    for d in (recent, stale):
        d.mkdir(parents=True)
        (d / "seen-0123456789abcdef.bin").write_bytes(b"s" * 64)
        (d / "master-0.grid").write_bytes(b"g" * MB)
        _age(d / "master-0.grid", 40)
    _age(stale / "seen-0123456789abcdef.bin", 31)
    got = _run(driver, store, 0.5, "current", 4, 1, 64)
    assert got["evicted"] == 2 and got["retired"] == 1, got
    assert not stale.exists()
    assert sorted(p.name for p in recent.iterdir()) == ["seen-0123456789abcdef.bin"]


def test_only_the_stores_own_files_are_touched(driver, tmp_path):
    """Whatever else sits under exe/ stays: a stranger's file keeps its
    directory, and a link is never followed out of the store."""
    store = tmp_path / "store"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "master-0.grid").write_bytes(b"o" * MB)
    _age(outside / "master-0.grid", 40)
    other = store / "exe" / "other"
    other.mkdir(parents=True)
    (other / "notes.txt").write_text("keep me")
    (other / "master-0.grid").write_bytes(b"g" * MB)
    _age(other / "master-0.grid", 40)
    (store / "exe" / "linked").symlink_to(outside, target_is_directory=True)
    got = _run(driver, store, 0.5, "current", 4, 1, 64)
    assert got["evicted"] == 1, got
    assert (outside / "master-0.grid").stat().st_size == MB
    assert (other / "notes.txt").read_text() == "keep me"
    assert not (other / "master-0.grid").exists()
