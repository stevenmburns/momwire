"""How many C++ compilers a build runs at once.

``setup.py`` loads this file by path (it runs before momwire is importable),
and the standalone-core tests import it, so the two builds follow one rule.
Standard library only, for that reason.

The default is the smaller of:

- the machine's PHYSICAL cores, not its hardware threads: a compile is
  CPU-bound, so two on one core's two threads each run at about half speed,
  and the second only adds memory;
- the memory available now, at `JOB_MB` per compiler: an -O3 core
  translation unit peaks at 330-530 MB, and all 61 standalone TUs at once
  peaked near 20 GB and OOM-killed a 15 GB machine (momwire#1703).

``MOMWIRE_BUILD_JOBS=<n>`` overrides both; ``1`` compiles serially.
"""

from __future__ import annotations

import os
import subprocess
import sys

ENV = "MOMWIRE_BUILD_JOBS"
JOB_MB = 1024  # memory budgeted per concurrent compiler


def physical_cores() -> int | None:
    """Physical cores, or None where this cannot tell."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            return None
        cores, phys, core = set(), None, None
        for line in text.splitlines() + [""]:
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "physical id":
                phys = value.strip()
            elif key == "core id":
                core = value.strip()
            elif not key:  # the blank line that ends one processor's block
                if core is not None:
                    cores.add((phys, core))
                phys = core = None
        return len(cores) or None
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                capture_output=True, text=True, timeout=5, check=True,
            )  # fmt: skip
            return int(out.stdout.strip()) or None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return None


def available_mb() -> int | None:
    """Memory available to a new process, in MB, or None where unknown."""
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            return None
        return None
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5, check=True,
            )  # fmt: skip
            # macOS has no MemAvailable; half the RAM stands in for it.
            return int(out.stdout.strip()) // (2 * 1024 * 1024)
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return None


def build_jobs() -> int:
    """`ENV` if set to a positive integer, else min(physical cores, available
    memory / `JOB_MB`), never below 1."""
    raw = os.environ.get(ENV, "").strip()
    if raw:
        try:
            n = int(raw)
        except ValueError:
            raise SystemExit(
                f"{ENV}={raw!r}: expected a whole number of jobs"
            ) from None
        if n > 0:
            return n
    cores = physical_cores() or os.cpu_count() or 1
    mem = available_mb()
    by_memory = max(1, mem // JOB_MB) if mem is not None else cores
    return max(1, min(cores, by_memory))


if __name__ == "__main__":
    print(build_jobs())
