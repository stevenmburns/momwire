"""Minimal pyitt + VTune smoke test.

Runs three artificial workloads under @pyitt.task decorators. If VTune
picks up the ITT API, the report should show "fast_work", "slow_work",
and "mixed_work" as task entries with distinguishable CPU time.

Run with:
    vtune -collect hotspots -knob enable-stack-collection=true \
        -- .venv/bin/python -m scripts.pyitt_smoke
"""

from __future__ import annotations

import math
import sys
import time

import pyitt


@pyitt.task
def fast_work() -> float:
    # ~10 ms of pure CPU work
    x = 0.0
    for i in range(200_000):
        x += math.sin(i * 1e-5)
    return x


@pyitt.task
def slow_work() -> float:
    # ~50 ms of pure CPU work
    x = 0.0
    for i in range(1_000_000):
        x += math.sin(i * 1e-5)
    return x


@pyitt.task
def mixed_work() -> float:
    # ~30 ms with a nested sub-task
    x = fast_work()
    for i in range(500_000):
        x += math.cos(i * 1e-5)
    return x


def main() -> int:
    total = 0.0
    for rep in range(20):
        total += fast_work()
        total += slow_work()
        total += mixed_work()
    print(f"total = {total:.4f}")
    return 0


if __name__ == "__main__":
    t0 = time.perf_counter()
    rc = main()
    print(f"wall = {time.perf_counter() - t0:.2f}s")
    sys.exit(rc)
