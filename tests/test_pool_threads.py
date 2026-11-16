"""The in-library pools' thread count (`_cpu_count.h`, momwire#1952).

The crossing cores' team cap and the Sommerfeld grid's node pool both read
`mw::cpu::pool_threads()`: MOMWIRE_THREADS clamped to the CPUs the affinity
mask allows, else one per physical core in the mask. Before it, the crossing
caps were hardware_concurrency() / 2 whatever the mask or MOMWIRE_THREADS
said, and the grid pool counted every CPU on the machine.

Each case runs in a fresh interpreter: the count is read from the process's
own environment and mask, and the mask is set before it starts.
"""

import json
import os
import subprocess
import sys

import pytest

from momwire import _accel

pytestmark = pytest.mark.skipif(
    _accel.acc is None, reason="no compiled accelerator loaded"
)

_PROBE = "import json; from momwire import _accel; print(json.dumps(_accel.acc._pool_threads()))"
_HAS_MASK = hasattr(os, "sched_setaffinity")


def _pools(threads=None, cpus=None):
    env = {k: v for k, v in os.environ.items() if k != "MOMWIRE_THREADS"}
    if threads is not None:
        env["MOMWIRE_THREADS"] = str(threads)
    preexec = (lambda: os.sched_setaffinity(0, cpus)) if cpus is not None else None
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        preexec_fn=preexec,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def _two_cpus():
    if not _HAS_MASK:
        pytest.skip("no affinity masks on this platform")
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 3:
        pytest.skip("needs a mask narrower than the machine")
    return set(allowed[:2])


def test_momwire_threads_sets_the_pool():
    # The machine's CPUs bound it too, so the expectation is the smaller.
    allowed = len(os.sched_getaffinity(0)) if _HAS_MASK else os.cpu_count()
    assert _pools(threads=2)["pool"] == min(2, allowed)


def test_a_two_cpu_mask_caps_the_pool():
    got = _pools(cpus=_two_cpus())
    assert got["allowed_threads"] == 2
    assert 1 <= got["allowed_cores"] <= 2
    assert got["pool"] == got["allowed_cores"]


def test_momwire_threads_is_clamped_to_the_mask():
    # Eight asked for on two allowed CPUs: the extra six would only queue.
    assert _pools(threads=8, cpus=_two_cpus())["pool"] == 2


def test_unset_is_one_per_physical_core():
    got = _pools()
    assert got["pool"] == got["allowed_cores"]
    assert 1 <= got["allowed_cores"] <= got["allowed_threads"]
