"""momwire#1108: an exported OMP_NUM_THREADS must not reach xdist workers.

`OMP_NUM_THREADS=8 make test` measured 910 s and 14 tests over the per-test
ceiling against 236 s with nothing exported, because the worker pin was a
`setdefault`. The helper is exercised on a fake environ; the real effect is
measured in the PR (wall time with the variable exported, before/after).
"""

import conftest
import pytest


def test_exported_width_is_overridden_to_one():
    env = {"OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "4"}
    conftest._pin_worker_threads(env, modules={})
    assert env == {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def test_nothing_exported_is_pinned_to_one():
    env: dict[str, str] = {}
    conftest._pin_worker_threads(env, modules={})
    assert env == {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}


def test_width_one_with_momwire_already_imported_is_fine():
    env = {"OMP_NUM_THREADS": "1"}
    conftest._pin_worker_threads(env, modules={"momwire": object()})
    assert env["OMP_NUM_THREADS"] == "1"


def test_wide_export_after_momwire_import_fails_loudly():
    env = {"OMP_NUM_THREADS": "8"}
    with pytest.raises(pytest.UsageError, match="momwire#1108"):
        conftest._pin_worker_threads(env, modules={"momwire.bspline": object()})
