"""Suite-wide wiring for the OPT-IN parallel run (momwire#400).

The default `pytest tests` is serial and needs nothing from this file. The
fast spelling is

    pytest tests -n auto --dist loadgroup        (~1m52s vs ~4m41s serial)

and this conftest is what makes it correct as well as fast:

* **Per-worker thread pinning.** The C++ accelerator is built `-fopenmp`,
  so without pinning each xdist worker spawns a full-width OpenMP pool and
  the run measures 486 s — 74% SLOWER than serial (thread contention, not
  parallelism). `OMP_NUM_THREADS` is read at the first parallel region,
  which is safely after `pytest_configure`, so the pin below lands in time
  on every worker. `setdefault`, so an explicit environment wins.

* **Group markers.** Under `--dist loadgroup` everything in one group runs
  sequentially on one worker:
  - the portal tests share server sockets and warm caches across modules —
    measured colliding (4 failures) when a fast parallel run interleaves
    them, and passing serially under the same pinning;
  - the memgate residency gates each hold a deliberately large working set,
    so running them CONCURRENTLY would multiply peaks the budgets were
    never certified against (the 8 GB machine rule). One group ⇒ one at a
    time, whatever `-n` says.

Groups are inert outside xdist: a serial run ignores them entirely.
"""

import os


def pytest_configure(config):
    # xdist worker detection: only workers carry `workerinput`. The
    # controller and any serial run are left untouched, so the serial
    # lanes keep the threading they were certified with.
    if hasattr(config, "workerinput"):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# Modules whose tests share portal server state (sockets, cache warmth).
_PORTAL_GROUP_FILES = (
    "test_portal.py",
    "test_portal_shared.py",
    "test_portal_differential.py",
    "test_portal_fixtures.py",
)


def pytest_collection_modifyitems(config, items):
    import pytest

    for item in items:
        if item.get_closest_marker("memgate") is not None:
            item.add_marker(pytest.mark.xdist_group("memgate"))
        elif item.path.name in _PORTAL_GROUP_FILES:
            item.add_marker(pytest.mark.xdist_group("portal"))
